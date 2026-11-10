"""Cache-only primary PermWorld repetition analyses.

This module never constructs models or optimizers. The frozen metric code and
the original joint-answer selectors are reused. A completed unit is an entire
data/initialization/training repetition, not one of its dependent 32 cells.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict, defaultdict
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
METRIC_PATH = ROOT / "diagnostics/permworld_scope_inventory_20261009/metrics.py"
METRIC_SHA = "c2a384a8212e661d1917c084a74699297bbef354f06dac42ac75ea90ce3d0b5c"
PHASES = ("initial", "trained")
ROLES = ("correct", "wrong", "identity")
CONTROLS = ("raw_within_length", "answer_strata")
METRICS = ("linear_cka", "debiased_cka", "knn_overlap", "procrustes")
COMPONENTS = tuple(p + "_" + r for p in PHASES for r in ROLES)
EFFECTS = ("B0", "Bt", "dc", "dw", "di", "Delta")
A_KEYS = ("architecture", "encoding", "relation", "control", "branch", "metric")
B_KEYS = ("architecture", "encoding", "relation", "control", "right_task")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def _safe(value):
    if isinstance(value, dict):
        return {str(k): _safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    return None if isinstance(value, float) and not math.isfinite(value) else value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(_safe(value), indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def read_csv(path):
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for r in records for k in r))
    if not fields:
        fields = ["status"]
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(_safe(records))
    temp.replace(path)


def number(value):
    if value is None or value in ("", "None", "nan", "NaN"):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def fixed_mean(values, expected_count):
    """Undefined components propagate; lengths are never silently reweighted."""
    if len(values) != expected_count:
        raise ValueError("Incomplete fixed length grid")
    vals = [number(v) for v in values]
    return math.fsum(vals) / expected_count if all(v is not None for v in vals) else None


def difference(a, b):
    a, b = number(a), number(b)
    return a - b if a is not None and b is not None else None


def components(values):
    out = {p + "_" + r: number(values[p][r]) for p in PHASES for r in ROLES}
    out["B0"] = difference(out["initial_correct"], out["initial_wrong"])
    out["Bt"] = difference(out["trained_correct"], out["trained_wrong"])
    for role, name in zip(ROLES, ("dc", "dw", "di")):
        out[name] = difference(out["trained_" + role], out["initial_" + role])
    out["Delta"] = difference(out["Bt"], out["B0"])
    if out["Delta"] is not None:
        if abs(out["Delta"] - (out["dc"] - out["dw"])) > 1e-10:
            raise AssertionError("Additive decomposition failed")
    return out


def frozen_metrics():
    if sha(METRIC_PATH) != METRIC_SHA:
        raise AssertionError("Frozen original metric source changed")
    spec = importlib.util.spec_from_file_location("repetition_original_metrics", METRIC_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def selection(data, tasks, relation, right_task, split, n, control):
    """Original ALL-three-right-actions joint keys, also for fixed-T candidates."""
    ns, labels = data[split + "_lengths"], data[split + "_labels"]
    yl = labels[:, 0, tasks.index(relation["left"])].astype(np.int64)
    yr = labels[:, :, tasks.index(right_task)].astype(np.int64)
    keys = np.column_stack((ns, yl, yr - int(relation.get("right_label_offset", 0))))
    use = ns == n
    if control == "answer_strata":
        _, groups, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
        use &= counts[groups] >= 3
    elif control != "raw_within_length":
        raise ValueError("Unknown control: " + control)
    ids = np.flatnonzero(use)
    selected_keys = keys[ids] if control == "answer_strata" else None
    group_count = len(np.unique(selected_keys, axis=0)) if selected_keys is not None else int(bool(len(ids)))
    audit = dict(anchor_count=int((ns == n).sum()), selected_count=len(ids), groups=group_count,
                 residual_df=len(ids) - group_count, excluded_count=int((ns == n).sum()) - len(ids))
    return ids, selected_keys, audit


class Cache:
    """Bounded read-only activation cache; no checkpoint construction."""
    def __init__(self, root, max_arrays=48):
        self.root = Path(root)
        self.cache = OrderedDict()
        self.max_arrays = max_arrays
        self.files = {}

    def path(self, architecture, encoding, task, phase, split):
        return self.root / "features" / f"{architecture}_{encoding}_{task}_{phase}_{split}.npz"

    def get(self, architecture, encoding, task, phase, split, field):
        path = self.path(architecture, encoding, task, phase, split)
        key = (str(path), field)
        if key not in self.cache:
            with np.load(path, allow_pickle=False) as archive:
                value = archive[field].copy()
            if not np.isfinite(value).all():
                raise ValueError("Nonfinite activations: " + str(path))
            value.setflags(write=False)
            self.cache[key] = value
            if str(path) not in self.files:
                self.files[str(path)] = dict(sha256=sha(path), size_bytes=path.stat().st_size)
            while len(self.cache) > self.max_arrays:
                self.cache.popitem(last=False)
        self.cache.move_to_end(key)
        return self.cache[key]


def _metric_reason(value, x, y, metric):
    if value is not None:
        return ""
    if len(x) < (11 if metric == "knn_overlap" else 4 if metric == "debiased_cka" else 2):
        return "insufficient_selected_samples"
    if metric == "procrustes":
        return "fixed_rank8_fit_degenerate_or_test_energy_zero"
    return "degenerate_centered_energy_or_eligible_neighbors"


def covariance_cka(x, y):
    """Independent feature-covariance identity for ordinary Gram CKA."""
    if len(x) < 2:
        return None
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    x, y = x - x.mean(0), y - y.mean(0)
    denominator = np.linalg.norm(x.T @ x) * np.linalg.norm(y.T @ y)
    return float(np.square(x.T @ y).sum() / denominator) if denominator > 1e-20 else None


def source_learning(config, data, cache):
    """Same fixed length averages, trained-label modes, and identity accuracy.

    Output-probability caches contain only numeric-class probabilities, so CE
    is named numeric-normalized CE. It is not silently substituted for the
    training full-vocabulary answer CE.
    """
    tasks = list(config.get("tasks", config.get("label_task_names", [])))
    rows = []
    for arch in config["architectures"]:
        for enc in config["encodings"]:
            for task in tasks:
                column = tasks.index(task)
                for phase in PHASES:
                    for split in ("validation", "test"):
                        prediction = cache.get(arch, enc, task, phase, split, "prediction")
                        probabilities = cache.get(arch, enc, task, phase, split, "output_prob")
                        labels, ns = data[split + "_labels"][:, :, column], data[split + "_lengths"]
                        for n in config["lengths"]:
                            use = ns == n
                            y = labels[use, 0].astype(np.int64)
                            pred = prediction[use, 0]
                            p = np.asarray(probabilities[use, 0], np.float64)
                            mode = None
                            if "train_labels" in data and "train_lengths" in data:
                                train = data["train_labels"][data["train_lengths"] == n, 0, column]
                                if len(train):
                                    values, counts = np.unique(train, return_counts=True)
                                    mode = int(values[np.argmax(counts)])  # original lower-label tie break
                            rows.append(dict(architecture=arch, encoding=enc, task=task, phase=phase, split=split,
                                length=n, samples=len(y), answer_accuracy=float((pred == y).mean()) if len(y) else None,
                                numeric_normalized_answer_ce=float(-np.log(np.maximum(p[np.arange(len(y)), y], np.finfo(np.float64).tiny)).mean()) if len(y) else None,
                                train_length_mode=mode, length_mode_accuracy=float((y == mode).mean()) if mode is not None and len(y) else None))
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in ("architecture", "encoding", "task", "phase", "split"))].append(row)
    summary = []
    for key, group in groups.items():
        out = dict(zip(("architecture", "encoding", "task", "phase", "split"), key))
        out.update(length_count=len(group), weights=f"1/{len(config['lengths'])} fixed")
        for field in ("answer_accuracy", "numeric_normalized_answer_ce", "length_mode_accuracy"):
            out[field] = fixed_mean([r[field] for r in group], len(config["lengths"]))
        out["accuracy_minus_length_mode"] = difference(out["answer_accuracy"], out["length_mode_accuracy"])
        summary.append(out)
    return rows, summary


def aggregate_a(rows, lengths):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in A_KEYS)].append(row)
    result = []
    for key, group in groups.items():
        group.sort(key=lambda r: int(r["length"]))
        if [int(r["length"]) for r in group] != lengths:
            raise AssertionError("Missing/duplicate Comparison A lengths")
        row = dict(zip(A_KEYS, key))
        row.update(layer="final_norm" if row["branch"] == "hidden" else "output",
                   contrast_type="condition_summary", length_count=len(lengths),
                   weights=f"1/{len(lengths)} fixed; undefined lengths propagate")
        for field in COMPONENTS + EFFECTS:
            row[field] = fixed_mean([r[field] for r in group], len(lengths))
        for field in ("Bt", "Delta"):
            row[field + "_positive_lengths"] = sum(number(r[field]) is not None and number(r[field]) > 0 for r in group)
            row[field + "_length_signs"] = "".join("?" if number(r[field]) is None else "+" if number(r[field]) > 0 else "-" if number(r[field]) < 0 else "0" for r in group)
        row["undefined_lengths"] = sum(number(r["Delta"]) is None for r in group)
        result.append(row)
    return result


def aggregate_b(rows, lengths):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in B_KEYS)].append(row)
    result = []
    for key, group in groups.items():
        group.sort(key=lambda r: int(r["length"]))
        if [int(r["length"]) for r in group] != lengths:
            raise AssertionError("Missing/duplicate Comparison B lengths")
        row = dict(zip(B_KEYS, key))
        row.update(category=group[0]["category"], designated_right_task=group[0]["designated_right_task"],
                   transform=group[0]["transform"], length_count=len(lengths),
                   weights=f"1/{len(lengths)} fixed; undefined lengths propagate")
        for phase in PHASES:
            field = "fixed_transform_score_" + phase
            row[field] = fixed_mean([r[field] for r in group], len(lengths))
            row[phase + "_defined_lengths"] = sum(number(r[field]) is not None for r in group)
        row["training_score_change"] = difference(row["fixed_transform_score_trained"], row["fixed_transform_score_initial"])
        result.append(row)
    return result


def candidate_contrasts(rows):
    """Exact ranking/ties and full 14-candidate mean match manuscript code.

    A finite-subset contrast is also retained, explicitly distinguished from the
    original full-candidate statistic. Missing candidates never count as losses.
    """
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in B_KEYS[:-1])].append(row)
    out = []
    for key, group in groups.items():
        des = [r for r in group if r["category"] == "designated"]
        if len(des) != 1:
            raise AssertionError("One designated candidate is required")
        des = des[0]
        others = sorted((r for r in group if r["category"] == "not_designated"), key=lambda r: r["right_task"])
        row = dict(zip(B_KEYS[:-1], key))
        row.update(candidate_count=len(group), other_count=len(others), designated_right_task=des["right_task"])
        for phase in PHASES:
            field = "fixed_transform_score_" + phase
            score = number(des[field])
            vals = [number(r[field]) for r in others]
            finite = [v for v in vals if v is not None]
            full_mean = math.fsum(finite) / len(others) if others and len(finite) == len(others) else None
            subset_mean = math.fsum(finite) / len(finite) if finite else None
            row[phase + "_designated_score"] = score
            row[phase + "_other_mean"] = full_mean
            row[phase + "_finite_other_count"] = len(finite)
            row[phase + "_finite_other_mean"] = subset_mean
            row[phase + "_designated_minus_other_mean"] = difference(score, full_mean)
            row[phase + "_designated_minus_finite_other_mean"] = difference(score, subset_mean)
            complete = score is not None and len(finite) == len(others)
            row[phase + "_rank"] = 1 + sum(v > score for v in finite) if complete else None
            row[phase + "_ties_with_designated"] = sum(v == score for v in finite) if complete else None
            row[phase + "_unique_first"] = (row[phase + "_rank"] == 1 and row[phase + "_ties_with_designated"] == 0) if complete else None
        common = [r for r in others if all(number(r["fixed_transform_score_" + p]) is not None for p in PHASES)]
        row["common_finite_other_count"] = len(common)
        for phase in PHASES:
            mean = math.fsum(number(r["fixed_transform_score_" + phase]) for r in common) / len(common) if common else None
            row["common_" + phase + "_contrast"] = difference(des["fixed_transform_score_" + phase], mean)
        row["Gcat"] = difference(row["common_trained_contrast"], row["common_initial_contrast"])
        row["Gcat_definition"] = "same finite candidates at both phases; fixed-T contrast change; not Comparison-A Delta"
        out.append(row)
    return out


def unit_counts(a_rows, b_contrasts, expected):
    result = []
    for control in CONTROLS:
        for metric in METRICS:
            selected = [r for r in a_rows if r["control"] == control and r["branch"] == "hidden" and r["metric"] == metric]
            if not selected:
                continue
            row = dict(comparison="A", control=control, metric=metric, expected_conditions=expected, observed_conditions=len(selected))
            for field in ("Bt", "Delta"):
                vals = [number(r[field]) for r in selected]
                row[field + "_finite"] = sum(v is not None for v in vals)
                row[field + "_positive"] = sum(v is not None and v > 0 for v in vals)
                row[field + "_negative"] = sum(v is not None and v < 0 for v in vals)
                row[field + "_zero"] = sum(v == 0 for v in vals if v is not None)
            result.append(row)
        selected = [r for r in b_contrasts if r["control"] == control]
        finite = [r for r in selected if number(r["trained_designated_minus_other_mean"]) is not None]
        result.append(dict(comparison="B", control=control, metric="linear_cka", expected_conditions=expected,
                           observed_conditions=len(selected), finite_conditions=len(finite),
                           above_other_mean=sum(number(r["trained_designated_minus_other_mean"]) > 0 for r in finite),
                           unique_first=sum(r["trained_unique_first"] is True for r in finite),
                           ranking_rule="strict greater; exact equality is a tie; all original candidates required"))
    return result


def analyze_unit(config, unit_root, output_dir):
    """Public CPU-limited wrapper around the resumable cache-only analysis."""
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:
        # A resource optimization, never a scientific dependency. The runner
        # uses the project environment, which includes this historical helper.
        return _analyze_unit(config, unit_root, output_dir)
    with threadpool_limits(limits=2):
        return _analyze_unit(config, unit_root, output_dir)


def _analyze_unit(config, unit_root, output_dir):
    """Evaluate one complete unit's existing NPZ activation caches.

    Each relation/configuration is checkpointed as a shard, bound to data,
    configuration, metric source, and analysis source. Changed inputs invalidate
    completion rather than overwriting an incompatible analysis.
    """
    config = read_json(config) if isinstance(config, (str, Path)) else config
    unit_root, output_dir = Path(unit_root), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    tasks = list(config.get("tasks", config.get("label_task_names", [])))
    lengths = list(config["lengths"])
    weights = config.get("length_weights", [1 / len(lengths)] * len(lengths))
    weights = list(weights.values()) if isinstance(weights, dict) else weights
    if len(weights) != len(lengths) or not np.allclose(weights, 1 / len(lengths), rtol=0, atol=1e-15):
        raise ValueError("Original protocol uses fixed equal length weights")
    actions = config.get("actions", ["identity", "complement", "inverse"])
    if actions != ["identity", "complement", "inverse"]:
        raise ValueError("Original action ordering changed")
    metric_names = tuple(config.get("analysis_metrics", METRICS))
    if not metric_names or set(metric_names) - set(METRICS):
        raise ValueError("Unrecognized metric scope")
    data_path = unit_root / "data.npz"
    binding = dict(data_sha256=sha(data_path), config_sha256=hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
                   metric_sha256=METRIC_SHA, analysis_sha256=sha(__file__))
    bind_path = output_dir / "analysis_binding.json"
    if bind_path.exists() and read_json(bind_path) != binding:
        raise ValueError("Incompatible existing analysis binding; use a new output directory")
    write_json(bind_path, binding)
    with np.load(data_path, allow_pickle=False) as archive:
        data = {k: archive[k].copy() for k in archive.files if k.endswith(("_permutations", "_lengths", "_labels"))}
    for split in ("validation", "test"):
        ns, labs = data[split + "_lengths"], data[split + "_labels"]
        if labs.shape != (len(ns), 3, len(tasks)):
            raise ValueError("Data label schema differs")
    cache, metric = Cache(unit_root, config.get("analysis_cache_arrays", 48)), frozen_metrics()
    a_rows, b_rows, checks = [], [], []
    for arch in config["architectures"]:
        for enc in config["encodings"]:
            for relation in config["relations"]:
                relation_id = relation["pair_id"]
                shard = output_dir / "shards" / f"{arch}_{enc}_{relation_id}"
                shard.mkdir(parents=True, exist_ok=True)
                paths = {name: shard / (name + ".csv") for name in ("a", "b", "checks")}
                receipt = shard / "completion.json"
                if receipt.exists():
                    saved = read_json(receipt)
                    if saved["binding"] != binding or any(sha(paths[k]) != saved["table_sha256"][k] for k in paths):
                        raise ValueError("Invalid resumed analysis shard")
                    for p, sig in saved.get("activation_files", {}).items():
                        if sha(p) != sig["sha256"]:
                            raise ValueError("Resumed shard activation changed: " + p)
                    a_rows.extend(read_csv(paths["a"])); b_rows.extend(read_csv(paths["b"])); checks.extend(read_csv(paths["checks"]))
                    continue
                aa, bb, cc = [], [], []
                ai = actions.index(relation["input_transform"])
                roles = dict(correct=ai, wrong=1 if ai == 2 else 2, identity=0)
                for n in lengths:
                    for control in CONTROLS:
                        masks = {s: selection(data, tasks, relation, relation["right"], s, n, control) for s in ("validation", "test")}
                        for branch, field in (("hidden", "prefix_final"), ("output_prob", "output_prob")):
                            measured = {name: {p: {} for p in PHASES} for name in metric_names}
                            reasons = {name: {} for name in metric_names}
                            for phase in PHASES:
                                for role, action in roles.items():
                                    arrays = []
                                    for split in ("validation", "test"):
                                        ids, keys, _ = masks[split]
                                        x = cache.get(arch, enc, relation["left"], phase, split, field)[ids, 0]
                                        y = cache.get(arch, enc, relation["right"], phase, split, field)[ids, action]
                                        arrays.extend((metric.center(x, keys), metric.center(y, keys)))
                                    xf, yf, xe, ye = arrays
                                    ids = masks["test"][0]
                                    duplicates = data["test_permutations"][ids] if "test_permutations" in data else None
                                    if set(metric_names) == set(METRICS):
                                        scores, _ = metric.scores(xf, yf, xe, ye, ids=ids, duplicate_keys=duplicates)
                                    else:
                                        scores = {}
                                        for name in metric_names:
                                            if name in ("linear_cka", "debiased_cka"):
                                                value = metric.cka(xe, ye, name == "debiased_cka")
                                            elif name == "knn_overlap":
                                                value = metric.knn_overlap(xe, ye, ids=ids, duplicate_keys=duplicates)
                                            else:
                                                error, _ = metric.procrustes(xf, yf, xe, ye)
                                                value = -error if error is not None else None
                                            scores[name] = (value, None)
                                    for name in metric_names:
                                        val = scores[name][0]
                                        measured[name][phase][role] = val
                                        reason = _metric_reason(val, xe, ye, name)
                                        if reason:
                                            reasons[name][phase + "_" + role] = reason
                                    if n == lengths[0] and branch == "hidden" and role == "correct" and "linear_cka" in metric_names:
                                        production = measured["linear_cka"][phase][role]
                                        independent = covariance_cka(xe, ye)
                                        passed = (production is None and independent is None) or (production is not None and independent is not None and abs(production - independent) <= 1e-8)
                                        cc.append(dict(architecture=arch, encoding=enc, relation=relation_id, control=control,
                                            length=n, phase=phase, check="fixed_first_length_independent_covariance_cka",
                                            passed=passed, absolute_difference=abs(production - independent) if production is not None and independent is not None else None))
                                        if not passed:
                                            raise AssertionError("Independent covariance CKA failed")
                            for name in metric_names:
                                aa.append(dict(architecture=arch, encoding=enc, relation=relation_id,
                                    control=control, branch=branch, layer="final_norm" if branch == "hidden" else "output",
                                    metric=name, length=n, fit_samples=len(masks["validation"][0]),
                                    test_samples=len(masks["test"][0]), **components(measured[name]),
                                    undefined_reason=json.dumps(reasons[name], sort_keys=True)))
                        for right in tasks:
                            if right == relation["left"]:
                                continue
                            ids, keys, audit = selection(data, tasks, relation, right, "test", n, control)
                            row = dict(architecture=arch, encoding=enc, relation=relation_id, control=control,
                                       right_task=right, designated_right_task=relation["right"], transform=relation["input_transform"],
                                       category="designated" if right == relation["right"] else "not_designated", length=n, **audit)
                            for phase in PHASES:
                                x = cache.get(arch, enc, relation["left"], phase, "test", "prefix_final")[ids, 0]
                                y = cache.get(arch, enc, right, phase, "test", "prefix_final")[ids, ai]
                                val = metric.cka(metric.center(x, keys), metric.center(y, keys))
                                row["fixed_transform_score_" + phase] = val
                                row[phase + "_undefined_reason"] = _metric_reason(val, x, y, "linear_cka")
                                if right == relation["right"] and "linear_cka" in metric_names:
                                    same = next(r for r in aa if r["length"] == n and r["control"] == control and r["metric"] == "linear_cka" and r["branch"] == "hidden")[phase + "_correct"]
                                    passed = (same is None and val is None) or (same is not None and val is not None and abs(same - val) <= 1e-8)
                                    cc.append(dict(architecture=arch, encoding=enc, relation=relation_id, control=control,
                                                   length=n, phase=phase, check="A_correct_equals_B_designated", passed=passed,
                                                   absolute_difference=abs(same - val) if same is not None and val is not None else None))
                                    if not passed:
                                        raise AssertionError("Comparison A/B designated inconsistency")
                            bb.append(row)
                for key, value in (("a", aa), ("b", bb), ("checks", cc)):
                    write_csv(paths[key], value)
                write_json(receipt, dict(status="completed", binding=binding, table_sha256={k: sha(v) for k, v in paths.items()},
                                         activation_files=cache.files))
                a_rows.extend(aa); b_rows.extend(bb); checks.extend(cc)
                print(json.dumps(dict(event="analysis_shard_completed", architecture=arch, encoding=enc, relation=relation_id,
                                      elapsed_seconds=time.monotonic() - started)), flush=True)
    # Normalize resumed CSV fields before arithmetic.
    for row in a_rows:
        for field in COMPONENTS + EFFECTS:
            row[field] = number(row[field])
    for row in b_rows:
        for phase in PHASES:
            field = "fixed_transform_score_" + phase
            row[field] = number(row[field])
    a_summary = aggregate_a(a_rows, lengths)
    b_summary = aggregate_b(b_rows, lengths)
    contrasts = candidate_contrasts(b_summary)
    raw_initial = [number(r["initial_designated_minus_other_mean"]) for r in contrasts if r["control"] == "raw_within_length"]
    zero_ok = all(v is None or abs(v) <= 1e-8 for v in raw_initial)
    if not zero_ok:
        raise AssertionError("Raw Comparison B initial zero reference failed")
    expected = len(config["architectures"]) * len(config["encodings"]) * len(config["relations"])
    counts = unit_counts(a_summary, contrasts, expected)
    learning_length, learning_summary = source_learning(config, data, cache)
    for name, records in (("comparison_a_per_length.csv", a_rows), ("comparison_a_summary.csv", a_summary),
                          ("comparison_b_per_length.csv", b_rows), ("comparison_b_summary.csv", b_summary),
                          ("comparison_b_contrasts.csv", contrasts), ("unit_counts.csv", counts), ("verification.csv", checks),
                          ("source_learning_per_length.csv", learning_length), ("source_learning_summary.csv", learning_summary)):
        write_csv(output_dir / name, records)
    completion = dict(status="completed", binding=binding, analysis_seconds=time.monotonic() - started,
                      expected_conditions=expected, metrics=list(metric_names), counts=counts,
                      raw_initial_zero_reference_verified=zero_ok,
                      raw_initial_zero_reference_defined_conditions=sum(v is not None for v in raw_initial),
                      answer_control_initial_reference="Not assumed zero: candidate-specific joint-answer groups and support",
                      retained_scope="final_norm hidden and output probabilities; no repeated all-layer extraction",
                      rows=dict(a_per_length=len(a_rows), a_summary=len(a_summary), b_per_length=len(b_rows), b_summary=len(b_summary)),
                      independent_repetition_unit="complete data, initialization, and training unit; cells remain dependent")
    write_json(output_dir / "completion.json", completion)
    return completion


def import_historical(repo_root, output_dir, unit="unit_0_historical"):
    """Recompute historical primary aggregates/ranks from untouched saved rows."""
    repo_root, output_dir = Path(repo_root), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    source_a = repo_root / "diagnostics/permworld_eight_relations_20261010/per_length_results.csv"
    source_b = repo_root / "diagnostics/representation_pairing_review_20261011/pair_per_length.csv"
    original_a = read_csv(source_a)
    aa = [r for r in original_a if r["layer"] in ("final_norm", "output") and r["metric"] in METRICS]
    for row in aa:
        for field in COMPONENTS + EFFECTS:
            row[field] = number(row[field])
    bb = [r for r in read_csv(source_b) if r["domain"] == "permworld"]
    for row in bb:
        for phase in PHASES:
            field = "fixed_transform_score_" + phase
            row[field] = number(row[field])
    lengths = list(range(10, 31))
    a_summary, b_summary = aggregate_a(aa, lengths), aggregate_b(bb, lengths)
    contrasts = candidate_contrasts(b_summary)
    counts = unit_counts(a_summary, contrasts, 32)
    main_a = next(r for r in counts if r["comparison"] == "A" and r["control"] == "answer_strata" and r["metric"] == "linear_cka")
    main_b = next(r for r in counts if r["comparison"] == "B" and r["control"] == "answer_strata")
    if (main_a["Bt_positive"], main_a["Delta_positive"], main_b["above_other_mean"], main_b["unique_first"]) != (26, 18, 29, 19):
        raise AssertionError("Historical manuscript counts not reproduced")
    for name, records in (("comparison_a_per_length.csv", aa), ("comparison_a_summary.csv", a_summary),
                          ("comparison_b_per_length.csv", bb), ("comparison_b_summary.csv", b_summary),
                          ("comparison_b_contrasts.csv", contrasts), ("unit_counts.csv", counts)):
        write_csv(output_dir / name, records)
    write_json(output_dir / "completion.json", dict(status="completed", unit=unit, historical_saved_table_recomputation=True,
               source_sha256={str(source_a): sha(source_a), str(source_b): sha(source_b)}, counts=counts,
               limitations="No new independent forward pass; historical data and training RNG structure retained"))
    return dict(unit=unit, analysis_dir=str(output_dir), status="completed")


def descriptives(values):
    vals = [number(x) for x in values]
    vals = [x for x in vals if x is not None]
    return dict(finite_units=len(vals), mean=statistics.mean(vals) if vals else None,
                sample_sd=statistics.stdev(vals) if len(vals) >= 2 else None,
                minimum=min(vals) if vals else None, maximum=max(vals) if vals else None,
                positive_units=sum(v > 0 for v in vals), negative_units=sum(v < 0 for v in vals),
                zero_units=sum(v == 0 for v in vals))


def summarize_units(records, outdir):
    """Descriptive across-unit summaries; incomplete units stay explicitly listed."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    count_rows, effects, statuses = [], [], []
    for record in records:
        unit, path = record["unit"], Path(record["analysis_dir"])
        completed = record.get("status") == "completed" and (path / "completion.json").is_file()
        completed = completed and read_json(path / "completion.json").get("status") == "completed"
        statuses.append(dict(unit=unit, requested_status=record.get("status", "unknown"),
                             analysis_completed=completed, analysis_dir=str(path)))
        if not completed:
            continue
        for row in read_csv(path / "unit_counts.csv"):
            count_rows.append(dict(unit=unit, **row))
        for row in read_csv(path / "comparison_a_summary.csv"):
            for field in COMPONENTS + EFFECTS:
                effects.append(dict(unit=unit, comparison="A", **{k: row[k] for k in A_KEYS},
                                    effect=field, value=number(row[field])))
        for row in read_csv(path / "comparison_b_contrasts.csv"):
            for field in ("initial_designated_minus_other_mean", "trained_designated_minus_other_mean", "Gcat"):
                effects.append(dict(unit=unit, comparison="B", architecture=row["architecture"], encoding=row["encoding"],
                                    relation=row["relation"], control=row["control"], branch="hidden", metric="linear_cka",
                                    effect=field, value=number(row[field])))
    grouped = defaultdict(list)
    keys = ("comparison",) + A_KEYS + ("effect",)
    for row in effects:
        grouped[tuple(row[k] for k in keys)].append(row)
    summary = []
    for key, group in grouped.items():
        row = dict(zip(keys, key))
        row.update(descriptives([r["value"] for r in group]), observed_unit_ids=";".join(r["unit"] for r in group))
        # All-unit sign agreement is explicitly descriptive, without binomial p values.
        row["all_finite_units_same_nonzero_direction"] = row["finite_units"] >= 2 and (row["positive_units"] == row["finite_units"] or row["negative_units"] == row["finite_units"])
        summary.append(row)
    count_stats = []
    count_groups = defaultdict(list)
    for row in count_rows:
        count_groups[row["comparison"], row["control"], row["metric"]].append(row)
    for key, group in count_groups.items():
        fields = ("Bt_positive", "Delta_positive") if key[0] == "A" else ("above_other_mean", "unique_first")
        for field in fields:
            count_stats.append(dict(comparison=key[0], control=key[1], metric=key[2], statistic=field,
                                    denominator="32 dependent conditions per complete original-size unit", **descriptives([r.get(field) for r in group])))
    write_csv(outdir / "unit_status.csv", statuses)
    write_csv(outdir / "unit_counts.csv", count_rows)
    write_csv(outdir / "condition_effects_by_unit.csv", effects)
    write_csv(outdir / "condition_effects_summary.csv", summary)
    write_csv(outdir / "unit_count_summary.csv", count_stats)
    done = sum(r["analysis_completed"] for r in statuses)
    text = ["# Independent-unit repetition results", "", f"Completed analyses: {done}/{len(records)} listed units.", "",
            "A unit jointly resamples the original data mechanism, shared initialization within an architecture, and training randomness. Its 32 conditions, shared tasks, candidate pairs, lengths, and metrics are dependent measurements.", "",
            "Comparison A reports trained pairing selectivity Bt separately from B0, correct/wrong changes, and Delta. Comparison B fixes the input transform and changes the right specialist; its raw shared-initialization reference is zero, whereas candidate-specific answer strata need not give a zero reference.", "",
            "The table below describes unit-level counts. The denominator remains 32 conditions within each unit; counts are not pooled into a binomial trial. Sample standard deviation is undefined with fewer than two completed units.", "",
            "| Comparison | Control | Metric | Count | Finite units | Mean | Sample SD | Min–max |", "|---|---|---|---|---:|---:|---:|---:|"]
    fmt = lambda x: "undefined" if x is None else f"{x:.6g}"
    for row in count_stats:
        text.append("| " + " | ".join(str(row[k]) for k in ("comparison", "control", "metric", "statistic", "finite_units")) + f" | {fmt(row['mean'])} | {fmt(row['sample_sd'])} | {fmt(row['minimum'])}–{fmt(row['maximum'])} |")
    text += ["", "Continuous effects and all per-unit values are in the companion CSV files. Exact signs use the original strict zero boundary; no new near-zero threshold is introduced. Interpret any sign reversal alongside the unrounded magnitude, range, and within-unit decomposition. Shared-initialization repetitions do not test specialists initialized independently within a unit.", ""]
    (outdir / "results.md").write_text("\n".join(text))
    _plot_effects(effects, outdir)
    result = dict(listed_units=len(records), completed_units=done, pending_units=len(records) - done,
                  statistical_unit="complete repetition unit", no_condition_level_independence=True)
    write_json(outdir / "summary_completion.json", result)
    return result


def _plot_effects(effects, outdir):
    selected = [r for r in effects if r["control"] == "answer_strata" and r["branch"] == "hidden" and r["metric"] == "linear_cka" and
                ((r["comparison"] == "A" and r["effect"] in ("Bt", "Delta")) or (r["comparison"] == "B" and r["effect"] == "trained_designated_minus_other_mean"))]
    if not selected:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    conditions = sorted({(r["architecture"], r["encoding"], r["relation"]) for r in selected})
    units = sorted({r["unit"] for r in selected})
    fig, axes = plt.subplots(1, 3, figsize=(13, max(5, len(conditions) * .24)), sharey=True, constrained_layout=True)
    for ax, effect, title in zip(axes, ("Bt", "Delta", "trained_designated_minus_other_mean"), ("A: trained selectivity", "A: change from initialization", "B: designated minus other mean")):
        for ui, unit in enumerate(units):
            rows = [r for r in selected if r["unit"] == unit and r["effect"] == effect and r["value"] is not None]
            ax.scatter([r["value"] for r in rows], [conditions.index((r["architecture"], r["encoding"], r["relation"])) + (ui - (len(units) - 1) / 2) * .1 for r in rows], s=14, label=unit)
        ax.axvline(0, color=".5", linewidth=.7)
        ax.set_title(title, fontsize=9)
        ax.set_yticks(range(len(conditions)), [" / ".join(c) for c in conditions], fontsize=6)
        ax.grid(axis="x", alpha=.2)
    axes[-1].legend(fontsize=6, loc="best")
    fig.savefig(outdir / "condition_effects_by_unit.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    one = sub.add_parser("analyze")
    one.add_argument("--config", required=True); one.add_argument("--unit-root", required=True); one.add_argument("--output-dir", required=True)
    old = sub.add_parser("import-historical")
    old.add_argument("--repo-root", default=str(ROOT)); old.add_argument("--output-dir", required=True)
    summary = sub.add_parser("summarize")
    summary.add_argument("--records", required=True); summary.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    if args.command == "analyze":
        value = analyze_unit(args.config, args.unit_root, args.output_dir)
    elif args.command == "import-historical":
        value = import_historical(args.repo_root, args.output_dir)
    else:
        value = summarize_units(read_json(args.records), args.output_dir)
    print(json.dumps(_safe(value), indent=2))


if __name__ == "__main__":
    main()
