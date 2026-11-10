"""Independent saved-array/CSV audit of the step-0 learning diagnostics.

CPU-only: no Torch import, model constructor, forward, optimizer or new CKA.
Run once the combined learning_summary.csv is available; missing input is not
treated as a completed diagnostic. This does not modify the frozen pipeline.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics
import sys

import numpy as np
from scipy.stats import spearmanr
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments.permworld_answer_controls_20261109.common import Sources, load_config, sha


def csv_rows(path):
    with Path(path).open(newline="") as f:
        return list(csv.DictReader(f))


def numeric(value):
    return None if value in (None, "", "None", "nan", "NaN") else float(value)


def safe(value):
    if isinstance(value, dict):
        return {str(k): safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None if math.isnan(value) else "Infinity" if value > 0 else "-Infinity"
    return value


def identity(array):
    a = np.asarray(array)
    return a[:, 0] if a.ndim == 2 else a


def expectation_rho(x, y):
    if len(x) < 2:
        return None, "fewer_than_two_samples"
    if len(set(map(float, y))) == 1:
        return None, "constant_true_answer"
    if len(set(map(float, x))) == 1:
        return None, "constant_expected_prediction"
    return float(spearmanr(x, y).statistic), ""


def run(diag):
    diag = Path(diag)
    config = load_config()
    root = Path(config["output"])
    combined_path = diag / "learning_summary.csv"
    if not combined_path.is_file():
        raise FileNotFoundError("Learning stage is not ready: " + str(combined_path))
    combined = csv_rows(combined_path)
    expected_models = {(u["unit"], a, e, t) for u in config["units"]
                       for a in config["architectures"] for e in config["encodings"] for t in config["tasks"]}
    actual = [(r["unit"], r["architecture"], r["encoding"], r["task"]) for r in combined]
    assert len(combined) == 320 and len(set(actual)) == 320 and set(actual) == expected_models
    assert all(r["phase"] == "trained" and r["split"] == "test" for r in combined)
    source = Sources(config)
    failures, checks, rho_reasons = [], 0, Counter()
    primary_missing, conditional_infinite, primary_nonfinite = [], [], []
    peaks_summary, peaks_lengths, model_summary = [], [], []
    maximum_gap_arithmetic_error = maximum_ce_mean_roundoff = maximum_prior_error = 0.
    maximum_ce_log_mass_identity_error = 0.
    log_mass_identity_samples = log_mass_zero_or_underflow = 0
    per_unit_counts = Counter()

    def check(ok, kind, key, **detail):
        nonlocal checks
        checks += 1
        if not ok:
            failures.append(dict(kind=kind, key=key, **detail))

    def close(x, y, tolerance=1e-10):
        x, y = numeric(x), numeric(y)
        if x is None or y is None:
            return x is None and y is None
        if math.isinf(x) or math.isinf(y):
            return x == y
        return abs(x-y) <= tolerance

    for unit in config["units"]:
        name = unit["unit"]
        data = source.data(name)
        unit_dir = root / "learning" / name
        length_rows = csv_rows(unit_dir / "learning_per_length.csv")
        lookup = {(r["architecture"], r["encoding"], r["task"], int(r["length"])): r for r in length_rows}
        check(len(lookup) == len(length_rows) == 64*21, "length_grid", name)
        for arch in config["architectures"]:
            for enc in config["encodings"]:
                for column, task in enumerate(config["tasks"]):
                    key = (name, arch, enc, task)
                    summary = next(r for r in combined if (r["unit"], r["architecture"], r["encoding"], r["task"]) == key)
                    check(summary["full_nll_available_all_lengths"] == "True", "full_nll_availability", key)
                    if summary["full_nll_available_all_lengths"] != "True":
                        primary_missing.append(key)
                    pred = identity(source.get(name, arch, enc, task, "trained", "test", "prediction"))
                    probabilities = np.asarray(source.get(name, arch, enc, task, "trained", "test", "output_prob")[:, 0], np.float64)
                    probabilities /= probabilities.sum(1, keepdims=True)
                    full_ce = identity(source.get(name, arch, enc, task, "trained", "test", "answer_ce"))
                    try:
                        mass = identity(source.get(name, arch, enc, task, "trained", "test", "numeric_mass"))
                    except (KeyError, FileNotFoundError):
                        mass = None
                    y = data["test_labels"][:, 0, column].astype(np.int64)
                    n_test = data["test_lengths"]
                    target_probability = probabilities[np.arange(len(y)), y]
                    with np.errstate(divide="ignore"):
                        conditional_ce = -np.log(target_probability)
                    if mass is not None:
                        valid = (target_probability > 0) & (mass > 0)
                        log_mass_zero_or_underflow += int((~valid).sum())
                        error = np.asarray(full_ce[valid], np.float64) - conditional_ce[valid] + np.log(np.asarray(mass[valid], np.float64))
                        maximum_ce_log_mass_identity_error = max(maximum_ce_log_mass_identity_error, float(np.max(np.abs(error))) if len(error) else 0.)
                        log_mass_identity_samples += len(error)
                        # Fixed numerical sanity tolerance for serialized
                        # float32 probabilities, not an inferential threshold.
                        check(bool(np.all(np.abs(error) <= 1e-4)), "full_conditional_mass_identity", key,
                              max_abs_error=float(np.max(np.abs(error))) if len(error) else None)
                    conditional_underflow = int(np.sum(target_probability == 0))
                    if conditional_underflow:
                        conditional_infinite.append(dict(unit=name, architecture=arch, encoding=enc, task=task, zero_targets=conditional_underflow))
                    if not np.isfinite(full_ce).all():
                        primary_nonfinite.append(key)
                    local = []
                    for n in config["lengths"]:
                        row = lookup[arch, enc, task, n]
                        use = n_test == n
                        yy = y[use]
                        train = data["train_labels"][data["train_lengths"] == n, 0, column]
                        count = Counter(map(int, train))
                        mode = min(v for v in count if count[v] == max(count.values()))
                        prior = np.asarray([(count.get(j, 0)+.5)/(len(train)+.5*31) for j in range(31)])
                        expected_prior_nll = float(np.mean(-np.log(prior[yy])))
                        expected_full_mean = math.fsum(map(float, full_ce[use])) / int(use.sum())
                        actual_full_mean = numeric(row["model_full_vocabulary_nll"])
                        ce_error = abs(actual_full_mean-expected_full_mean) if actual_full_mean is not None else math.inf
                        maximum_ce_mean_roundoff = max(maximum_ce_mean_roundoff, ce_error)
                        check(ce_error <= 8*np.finfo(np.float32).eps*max(1., abs(expected_full_mean)), "full_ce_mean_precision", key+(n,), error=ce_error)
                        prior_error = abs(numeric(row["length_prior_nll"])-expected_prior_nll)
                        maximum_prior_error = max(maximum_prior_error, prior_error)
                        check(prior_error <= 1e-12, "training_only_prior_nll", key+(n,), error=prior_error)
                        check(int(row["train_length_mode"]) == mode, "train_mode", key+(n,))
                        check(close(row["exact_match_accuracy"], float(np.mean(pred[use] == yy))), "full_argmax_accuracy", key+(n,))
                        check(close(row["train_length_mode_accuracy"], float(np.mean(yy == mode))), "mode_accuracy", key+(n,))
                        gap = numeric(row["model_full_vocabulary_nll"])-numeric(row["length_prior_nll"])
                        gap_error = abs(numeric(row["full_nll_minus_length_prior_nll"])-gap)
                        maximum_gap_arithmetic_error = max(maximum_gap_arithmetic_error, gap_error)
                        check(gap_error <= 1e-12, "full_nll_gap_identity", key+(n,), error=gap_error)
                        check(close(row["length_prior_nll_minus_full_nll"], -gap), "full_nll_gain_sign", key+(n,))
                        check(close(row["model_conditional_numeric_nll"], float(conditional_ce[use].mean())), "conditional_nll", key+(n,))
                        xx = probabilities[use] @ np.arange(31, dtype=np.float64)
                        rho, reason = expectation_rho(xx, yy)
                        rho_reasons[reason or "defined"] += 1
                        check(close(row["expected_answer_spearman"], rho), "spearman", key+(n,))
                        check(row["spearman_undefined_reason"] == reason, "spearman_reason", key+(n,))
                        check(close(row["full_vocabulary_prediction_outside_answer_support_fraction"], float(np.mean((pred[use] < 0) | (pred[use] >= 31)))), "outside_answer_support", key+(n,))
                        local.append(row)
                        if arch == "gru" and enc == "canonical_cycle" and task in ("peaks", "valleys"):
                            peaks_lengths.append(row)
                    for field in ("accuracy_minus_length_mode", "full_nll_minus_length_prior_nll", "conditional_numeric_nll_minus_length_prior_nll", "expected_answer_spearman"):
                        values = [numeric(row[field]) for row in local]
                        expected = None if any(x is None for x in values) else float(np.mean(values))
                        check(close(summary[field], expected), "equal_length_summary", key+(field,))
                    per_unit_counts[name] += 1
                    out = {k:summary[k] for k in ("unit", "architecture", "encoding", "task", "accuracy_minus_length_mode", "full_nll_minus_length_prior_nll", "conditional_numeric_nll_minus_length_prior_nll", "expected_answer_spearman", "spearman_undefined_lengths")}
                    model_summary.append(out)
                    if arch == "gru" and enc == "canonical_cycle" and task in ("peaks", "valleys"):
                        peaks_summary.append(out)
    end_stat_changes=[]
    for path, sig in source.checked.items():
        stat=Path(path).stat()
        if (stat.st_size, stat.st_mtime_ns) != (sig["size_bytes"], sig["mtime_ns"]):end_stat_changes.append(path)
    check(not end_stat_changes, "initial_SHA_and_end_stat_guard", "all_saved_inputs", changes=end_stat_changes)
    result=dict(status="passed" if not failures else "failed", captured_utc=datetime.now(timezone.utc).isoformat(),
        model_summaries=len(combined),expected_model_summaries=320,per_length_rows=320*21,models_by_unit=dict(per_unit_counts),
        checks=checks,failures=failures,missing_primary_full_nll_models=primary_missing,nonfinite_primary_full_nll_models=primary_nonfinite,
        conditional_probability_underflows=conditional_infinite,Spearman_reason_counts=dict(rho_reasons),
        maximum_prior_nll_error=maximum_prior_error,maximum_full_nll_gap_arithmetic_error=maximum_gap_arithmetic_error,
        maximum_float32_CE_mean_roundoff=maximum_ce_mean_roundoff,maximum_full_conditional_log_mass_identity_error=maximum_ce_log_mass_identity_error,
        log_mass_identity_finite_samples=log_mass_identity_samples,log_mass_zero_or_underflow_samples=log_mass_zero_or_underflow,
        full_NLL_gap_direction="Model minus training-length prior; negative is lower logloss than the prior",
        rho_interpretation="Conditional numeric expectation association; a positive rank correlation does not ensure calibrated probabilities or task mastery",
        comparison_policy="No equivalence or near-baseline bound; constant labels/predictions remain undefined rather than zero",
        unit_model_summary=model_summary,gru_cycle_peaks_valleys_summary=peaks_summary,gru_cycle_peaks_valleys_per_length=peaks_lengths,
        protection=dict(files_SHA_checked_on_first_access=len(source.checked),end_size_mtime_changes=end_stat_changes,
                        scope="First-access full-file SHA and final size/mtime guards for read inputs; not proof of global or continuous immutability"),
        no_model_construction=True,no_forward=True,no_training=True,no_probe=True,no_new_CKA=True)
    (diag/"learning_review.json").write_text(json.dumps(safe(result),indent=2,allow_nan=False)+"\n")
    lines=["# Independent learning-diagnostic review", "", f"Status: **{result['status']}**; {len(combined)} specialist summaries, {320*21} prescribed model/length rows, {checks} checks, {len(failures)} failures.", "",
        "This reads saved outputs, full answer-position cross-entropies, training labels and diagnostic CSVs only. It does not load or fit model weights, run a forward pass, generate inputs, or compute new CKA.", "",
        "All full-vocabulary NLLs are checked separately from the conditional numeric NLL. Negative `full_nll_minus_length_prior_nll` means better observed logloss than the train-only smoothed prior. Spearman uses the conditional numeric expectation. Neither a small NLL gap nor a nonsignificant association establishes practical equivalence. Undefined correlations are retained.", "",
        "## GRU + canonical-cycle peaks and valleys", "", "| Unit | Task | Accuracy minus mode | Full NLL minus prior | Conditional NLL minus prior | Expectation Spearman | Undefined rho lengths |", "|---|---|---:|---:|---:|---:|---:|"]
    for row in peaks_summary:
        lines.append("| "+" | ".join(str(row[k]) or "undefined" for k in ("unit","task","accuracy_minus_length_mode","full_nll_minus_length_prior_nll","conditional_numeric_nll_minus_length_prior_nll","expected_answer_spearman","spearman_undefined_lengths"))+" |")
    lines += ["", "These observations concern identity inputs only. They do not assert that the same learning statistics were measured on transformed partner inputs. The geometry association join records this identity fallback explicitly.", "",
        "Positive expectation correlation can coexist with worse-than-prior logloss: ordering, calibration and probability mass outside the answer support are different properties. This audit therefore does not turn exact-match ties, near-zero gaps, or undefined correlations into a declaration that a model is exactly at the length baseline.", "",
        "The JSON retains every specialist's summary, all 210 peaks/valleys length rows, undefined reasons, numerical identity checks and source guards. Counted specialists/lengths are an inventory, not independent repetitions for a statistical test.", ""]
    (diag/"learning_review.md").write_text("\n".join(lines))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostic",type=Path,default=ROOT/"diagnostics/permworld_answer_controls_20261109")
    args=parser.parse_args()
    with threadpool_limits(limits=2):
        result=run(args.diagnostic)
    print(json.dumps({k:result[k] for k in ("status","model_summaries","per_length_rows","checks","missing_primary_full_nll_models","maximum_full_conditional_log_mass_identity_error")},indent=2))
    if result["status"]!="passed":raise SystemExit(1)


if __name__=="__main__":main()
