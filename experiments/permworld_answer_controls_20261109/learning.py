"""Read-only specialist learning diagnostics against training-length priors.

This module does not construct a model, import Torch, fit a probe, or generate
inputs. ``get_features`` must return the already frozen evaluation arrays.
Conditional numeric probabilities are never substituted for unavailable
full-vocabulary NLL or full-vocabulary argmax predictions.
"""
from __future__ import annotations

from collections import defaultdict
import csv
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import rankdata


NUMERIC_CLASSES = 31
PRIOR_ALPHA = 0.5


def smoothed_length_prior(train_answers, classes=NUMERIC_CLASSES, alpha=PRIOR_ALPHA):
    """Jeffreys-smoothed numeric class frequencies; deterministic lower-mode tie.

    The declared numeric support is fixed across lengths, including classes
    absent or impossible at a particular length. No test label sets the prior.
    """
    y = np.asarray(train_answers, dtype=np.int64)
    if y.ndim != 1 or not len(y):
        raise ValueError("A nonempty one-dimensional training answer set is required")
    if classes < 2 or alpha <= 0 or np.any(y < 0) or np.any(y >= classes):
        raise ValueError("Invalid numeric support, smoothing, or training answers")
    count = np.bincount(y, minlength=classes).astype(np.int64)
    prior = (count.astype(np.float64) + alpha) / (len(y) + alpha * classes)
    return prior, int(np.argmax(count)), count


def categorical_nll(probabilities, answers):
    """Return mean NLL and the count of stored zero target probabilities.

    Stored zeros produce +infinity rather than an invented finite logit. The
    separate full-vocabulary CE is preferred when cached probabilities have
    underflowed. JSON output records infinity as an explicit string.
    """
    p = np.asarray(probabilities, dtype=np.float64)
    y = np.asarray(answers, dtype=np.int64)
    if p.ndim == 1:
        p = np.broadcast_to(p, (len(y), len(p)))
    if p.ndim != 2 or p.shape[0] != len(y) or np.any(y < 0) or np.any(y >= p.shape[1]):
        raise ValueError("Categorical probabilities and labels do not align")
    if not len(y):
        return None, 0
    target = p[np.arange(len(y)), y]
    if not np.isfinite(p).all() or np.any(p < 0):
        raise ValueError("Invalid class probabilities")
    with np.errstate(divide="ignore"):
        value = float((-np.log(target)).mean())
    return value, int(np.count_nonzero(target == 0))


def spearman_expectation(expected_answers, true_answers):
    """Average-rank Spearman; constants remain undefined, never zero-filled."""
    x, y = np.asarray(expected_answers, np.float64), np.asarray(true_answers, np.float64)
    if x.ndim != 1 or y.shape != x.shape or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Finite aligned one-dimensional Spearman inputs required")
    if len(x) < 2:
        return None, "fewer_than_two_samples"
    if np.ptp(y) == 0:
        return None, "constant_true_answer"
    if np.ptp(x) == 0:
        return None, "constant_expected_prediction"
    a, b = rankdata(x, method="average"), rankdata(y, method="average")
    a, b = a - a.mean(), b - b.mean()
    denominator = math.sqrt(float(a @ a) * float(b @ b))
    return float(a @ b / denominator), ""


def _number(value):
    if value is None:
        return None
    value = float(value)
    return None if math.isnan(value) else value


def _difference(left, right):
    left, right = _number(left), _number(right)
    if left is None or right is None:
        return None
    result = left - right
    return None if math.isnan(result) else result


def _fixed_mean(values, expected):
    if len(values) != expected:
        raise ValueError("Missing a prescribed length")
    values = [_number(x) for x in values]
    if any(x is None for x in values):
        return None
    # All diagnostic NLLs are nonnegative; signed differences can contain
    # infinities. An undefined +infinity - infinity is not disguised as zero.
    with np.errstate(invalid="ignore"):
        result = float(np.asarray(values, np.float64).mean())
    return None if math.isnan(result) else result


def _identity(array, n, field, classes):
    if array is None:
        return None
    a = np.asarray(array)
    if field == "output_prob":
        if a.ndim == 3:
            a = a[:, 0, :]
        if a.shape != (n, classes):
            raise ValueError(f"Invalid {field} shape: {a.shape}, expected {(n, classes)}")
    else:
        if a.ndim == 2:
            a = a[:, 0]
        if a.shape != (n,):
            raise ValueError(f"Invalid {field} shape: {a.shape}, expected {(n,)}")
    if not np.isfinite(a).all():
        raise ValueError(f"Nonfinite saved {field}")
    return a


def _safe(value):
    if isinstance(value, dict):
        return {str(k): _safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_safe(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None if math.isnan(value) else "Infinity" if value > 0 else "-Infinity"
    return value


def _write_csv(path, rows):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for row in rows for k in row)) or ["status"]
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    temporary.replace(path)


def _write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(_safe(value), indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


SUMMARY_FIELDS = (
    "exact_match_accuracy", "train_length_mode_accuracy", "accuracy_minus_length_mode",
    "numeric_argmax_accuracy", "model_full_vocabulary_nll", "length_prior_nll",
    "full_nll_minus_length_prior_nll", "length_prior_nll_minus_full_nll",
    "model_conditional_numeric_nll", "conditional_numeric_nll_minus_length_prior_nll",
    "expected_answer_spearman", "full_vocabulary_prediction_outside_answer_support_fraction",
    "mean_numeric_probability_mass",
)


def evaluate_learning(config, data, get_features, outdir):
    """Evaluate one unit's saved specialist predictions without new fitting.

    Args:
      config: tasks, architectures, encodings, lengths; optional unit,
        learning_phases (default trained), learning_splits (default test).
      data: original split_lengths and split_labels [N, actions, tasks],
        including the training split for the mode/prior.
      get_features: callable(architecture, encoding, task, phase, split, field).
        Fields are prediction, output_prob, optional answer_ce and numeric_mass.
        Except output_prob, arrays may be [N] or [N, actions]; action0 is used.
        Missing fields return None or raise KeyError/FileNotFoundError.
      outdir: a new diagnostic directory; no original cache is modified.

    All n values have the original equal weights. Undefined length statistics
    propagate to the summary; finite-length counts/reasons remain explicit.
    Full-vocabulary NLL is comparable to the numeric training prior because
    every observed true answer lies on that prior's declared numeric support.
    """
    tasks = list(config.get("tasks", config.get("label_task_names", [])))
    lengths = list(config["lengths"])
    if not tasks or not lengths or len(set(lengths)) != len(lengths):
        raise ValueError("Distinct prescribed lengths and task names required")
    classes = int(config.get("numeric_classes", NUMERIC_CLASSES))
    alpha = float(config.get("learning_prior_alpha", PRIOR_ALPHA))
    if alpha != PRIOR_ALPHA:
        raise ValueError("This fixed diagnostic uses alpha=0.5, not a searched smoothing value")
    phases = list(config.get("learning_phases", ["trained"]))
    splits = list(config.get("learning_splits", ["test"]))
    unit = str(config.get("unit", config.get("unit_id", "unspecified")))
    train_n = np.asarray(data["train_lengths"])
    train_y = np.asarray(data["train_labels"])
    if train_y.ndim != 3 or train_y.shape[0] != len(train_n) or train_y.shape[2] != len(tasks):
        raise ValueError("Original training-label schema differs")
    priors = {}
    prior_rows = []
    for column, task in enumerate(tasks):
        for n in lengths:
            selected = train_y[train_n == n, 0, column]
            if not len(selected):
                priors[task, n] = None
                continue
            prior, mode, count = smoothed_length_prior(selected, classes, alpha)
            priors[task, n] = prior, mode
            for label, (number, probability) in enumerate(zip(count, prior)):
                prior_rows.append(dict(unit=unit, task=task, length=n, numeric_label=label,
                    training_count=int(number), training_samples=len(selected), smoothing_alpha=alpha,
                    smoothed_probability=float(probability), train_length_mode=mode))
    rows, missing = [], []
    for arch in config["architectures"]:
        for enc in config["encodings"]:
            for column, task in enumerate(tasks):
                for phase in phases:
                    for split in splits:
                        ns = np.asarray(data[split + "_lengths"])
                        labs = np.asarray(data[split + "_labels"])
                        if labs.shape != (len(ns), 3, len(tasks)):
                            raise ValueError("Original evaluation-label schema differs")
                        fields = {}
                        missing_names = []
                        for field in ("prediction", "output_prob", "answer_ce", "numeric_mass"):
                            try:
                                value = get_features(arch, enc, task, phase, split, field)
                            except (KeyError, FileNotFoundError):
                                value = None
                            fields[field] = _identity(value, len(ns), field, classes)
                            if value is None:
                                missing_names.append(field)
                                missing.append(dict(unit=unit, architecture=arch, encoding=enc, task=task,
                                    phase=phase, split=split, field=field, reason="not_available_in_saved_cache"))
                        p = fields["output_prob"]
                        sum_error = None
                        if p is not None:
                            p = np.asarray(p, np.float64)
                            totals = p.sum(axis=1)
                            sum_error = float(np.max(np.abs(totals - 1))) if len(p) else 0.
                            if np.any(p < 0) or np.any(totals <= 0) or sum_error > 1e-5:
                                raise ValueError("Saved numeric probabilities are not a normalized categorical distribution")
                            # Correct float32 serialization rounding only, using
                            # a fixed numerical tolerance, not a learned scale.
                            p = p / totals[:, None]
                        if fields["answer_ce"] is not None and np.any(fields["answer_ce"] < -1e-6):
                            raise ValueError("Negative full-vocabulary answer CE")
                        if fields["numeric_mass"] is not None and (np.any(fields["numeric_mass"] < 0) or np.any(fields["numeric_mass"] > 1 + 1e-6)):
                            raise ValueError("Numeric probability mass lies outside [0,1]")
                        for n in lengths:
                            use = ns == n
                            y = labs[use, 0, column].astype(np.int64)
                            if np.any(y < 0) or np.any(y >= classes):
                                raise ValueError("True answers lie outside the fixed numeric support")
                            base = priors[task, n]
                            prior, mode = base if base is not None else (None, None)
                            baseline_nll = categorical_nll(prior, y)[0] if prior is not None else None
                            exact = float((fields["prediction"][use] == y).mean()) if fields["prediction"] is not None and len(y) else None
                            mode_accuracy = float((y == mode).mean()) if mode is not None and len(y) else None
                            full_nll = float(fields["answer_ce"][use].mean()) if fields["answer_ce"] is not None and len(y) else None
                            conditional_nll, zero_count = categorical_nll(p[use], y) if p is not None else (None, 0)
                            expectation = p[use] @ np.arange(classes, dtype=np.float64) if p is not None else None
                            rho, rho_reason = spearman_expectation(expectation, y) if expectation is not None else (None, "numeric_distribution_unavailable")
                            non_numeric = None
                            if fields["prediction"] is not None and len(y):
                                prediction = fields["prediction"][use]
                                non_numeric = float(((prediction < 0) | (prediction >= classes)).mean())
                            row = dict(unit=unit, architecture=arch, encoding=enc, task=task, phase=phase,
                                split=split, length=n, samples=len(y), training_samples=int((train_n == n).sum()),
                                train_length_mode=mode, smoothing_alpha=alpha, numeric_classes=classes,
                                exact_match_accuracy=exact, train_length_mode_accuracy=mode_accuracy,
                                accuracy_minus_length_mode=_difference(exact, mode_accuracy),
                                numeric_argmax_accuracy=float((p[use].argmax(1) == y).mean()) if p is not None and len(y) else None,
                                model_full_vocabulary_nll=full_nll, length_prior_nll=baseline_nll,
                                full_nll_minus_length_prior_nll=_difference(full_nll, baseline_nll),
                                length_prior_nll_minus_full_nll=_difference(baseline_nll, full_nll),
                                model_conditional_numeric_nll=conditional_nll,
                                conditional_numeric_nll_minus_length_prior_nll=_difference(conditional_nll, baseline_nll),
                                conditional_zero_target_probability_count=zero_count,
                                expected_answer_spearman=rho, spearman_undefined_reason=rho_reason,
                                expected_answer_definition=f"Expectation conditional on answer labels 0..{classes-1}; not expectation over all 188 vocabulary tokens",
                                full_vocabulary_prediction_outside_answer_support_fraction=non_numeric,
                                mean_numeric_probability_mass=float(fields["numeric_mass"][use].mean()) if fields["numeric_mass"] is not None and len(y) else None,
                                numeric_probability_row_sum_max_error=sum_error,
                                full_nll_source="saved_full_vocabulary_answer_ce" if fields["answer_ce"] is not None else "unavailable; conditional NLL is separate",
                                missing_features=";".join(missing_names),
                                baseline_status="training_prior_available" if base is not None else "missing_training_length",
                                learning_classification="not_thresholded; inspect full-NLL difference and Spearman with their uncertainty")
                            rows.append(row)
    grouped = defaultdict(list)
    keys = ("unit", "architecture", "encoding", "task", "phase", "split")
    for row in rows:
        grouped[tuple(row[k] for k in keys)].append(row)
    summary = []
    for key, group in grouped.items():
        group.sort(key=lambda x: lengths.index(x["length"]))
        if [x["length"] for x in group] != lengths:
            raise ValueError("Missing or duplicate prescribed learning lengths")
        out = dict(zip(keys, key)); out.update(length_count=len(lengths), weights=f"1/{len(lengths)} fixed; undefined lengths propagate")
        for field in SUMMARY_FIELDS:
            out[field] = _fixed_mean([r[field] for r in group], len(lengths))
            out[field + "_defined_lengths"] = sum(r[field] is not None for r in group)
            finite = [r[field] for r in group if r[field] is not None and math.isfinite(r[field])]
            out[field + "_finite_lengths"] = len(finite)
        out["conditional_zero_target_probability_count"] = sum(r["conditional_zero_target_probability_count"] for r in group)
        out["spearman_undefined_lengths"] = sum(r["expected_answer_spearman"] is None for r in group)
        out["full_nll_available_all_lengths"] = all(r["model_full_vocabulary_nll"] is not None for r in group)
        out["missing_features"] = ";".join(sorted({f for row in group for f in row["missing_features"].split(";") if f}))
        summary.append(out)
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    _write_csv(outdir / "learning_per_length.csv", rows)
    _write_csv(outdir / "learning_summary.csv", summary)
    _write_csv(outdir / "learning_training_priors.csv", prior_rows)
    _write_csv(outdir / "learning_missing_features.csv", missing)
    policy = dict(unit=unit, smoothing_alpha=alpha, numeric_classes=classes,
        prior_source="Identity labels in the original training set, independently for each length/task; no validation/test frequency fitting",
        exact_match_definition="Saved full-vocabulary argmax equals the correct numeric answer ID",
        outside_answer_support="Argmax outside labels 0..30; includes numeric tokens 31..99 as well as EOS/other tokens",
        primary_nll="Full-vocabulary answer_ce minus train-length numeric-prior NLL, when full CE is available",
        conditional_nll="Separate NLL of the numeric-renormalized cached 31-class distribution; never substituted for full CE",
        expectation="Conditional numeric expectation, not an expectation over all vocabulary tokens",
        Spearman="Average ranks; constant true answers or expected predictions remain undefined",
        probability_rounding="Renormalize cached numeric rows once in float64 if initial row sums are within 1e-5 of one",
        zero_probability="Stored zeros give +infinity conditional NLL and remain counted; JSON uses an explicit Infinity string",
        aggregation="Each prescribed length has the same fixed weight; any undefined component makes that summary undefined, with availability counts reported",
        no_convergence_or_near_baseline_threshold=True, rows=len(rows), summaries=len(summary),
        missing_feature_records=len(missing), no_model_construction=True, no_training=True, no_probe=True)
    _write_json(outdir / "learning_policy.json", policy)
    return dict(per_length=rows, summary=summary, policy=policy, missing_features=missing)
