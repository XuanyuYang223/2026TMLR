"""Exact position-preserving input geometry without large activation caches.

For a permutation p of length n, the input feature is
    concat(E[p[0]], ..., E[p[n-1]]).
The numeric token IDs in the frozen Passage vocabulary equal their values.
The corresponding value-token positions are 4, 6, ..., 2*n+2. Adding the
position embedding at these fixed positions adds the same vector to every
example of a given length. Both global and answer-stratum centering eliminate
that vector exactly. Consequently, omitting it leaves all frozen metrics
unchanged, including independently centered fit/test cross Gram matrices.

The Gram implementation below is algebraically equivalent to the frozen
feature-based CKA, Euclidean KNN and rank-8 held-out Procrustes. It is not a new
input kernel or a metric selected using neural outcomes. PCA is evaluated via
the sample Gram eigendecomposition instead of materializing n*d features.

The embedding of ONE_END, and mean embedding of all n value tokens, are
structurally constant at fixed n. They must be reported as undefined, even
when floating-point reduction order creates tiny apparent variations.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np


HERE = Path(__file__).resolve().parent
FROZEN_METRICS = HERE.parent / "permworld_scope_inventory_20261009" / "metrics.py"
STRUCTURAL_CONSTANT_BRANCHES = {
    "embedding_end": "ONE_END token and its position are fixed within length.",
    "embedding_value_mean": "Every value 1..n occurs once; the mean discards order.",
}


def _permutations(permutations, length=None):
    p = np.asarray(permutations)
    if not np.issubdtype(p.dtype, np.integer) or p.ndim < 2:
        raise ValueError("Permutations must be an integer array with sample and value axes.")
    n = p.shape[-1] if length is None else int(length)
    if n < 1 or n > p.shape[-1]:
        raise ValueError("Invalid permutation length.")
    p = p[..., :n]
    if np.any(np.sort(p, axis=-1) != np.arange(1, n + 1)):
        raise ValueError("Each selected row must be a permutation of 1..n.")
    return p, n


def position_features(embedding_weight, permutations, length=None):
    """Temporary explicit reference features, preserving arbitrary leading axes."""
    p, n = _permutations(permutations, length)
    weight = np.asarray(embedding_weight, dtype=np.float64)
    if weight.ndim != 2 or len(weight) <= n:
        raise ValueError("Embedding table does not include every numeric token.")
    return weight[p].reshape(*p.shape[:-1], n * weight.shape[1])


def onehot_position_features(permutations, length=None, categories=31):
    """Static numeric-token onehot baseline, with permutation positions retained."""
    p, n = _permutations(permutations, length)
    if int(categories) <= n:
        raise ValueError("Onehot categories must include token value n.")
    return np.eye(int(categories), dtype=np.float64)[p].reshape(
        *p.shape[:-1], n * int(categories)
    )


def embedding_similarity_table(embedding_weight, length):
    """Small numeric-token dot-product table; no positional pooling."""
    weight = np.asarray(embedding_weight, dtype=np.float64)
    if weight.ndim != 2 or len(weight) <= int(length):
        raise ValueError("Invalid embedding table.")
    numeric = weight[: int(length) + 1]
    return numeric @ numeric.T


def position_gram_from_table(table, left, right=None, length=None):
    """Raw Gram = sum_j <E[p_j], E[q_j]>, computed with token lookups."""
    left, n = _permutations(left, length)
    right, rn = _permutations(left if right is None else right, n)
    table = np.asarray(table, dtype=np.float64)
    if left.ndim != 2 or right.ndim != 2 or rn != n:
        raise ValueError("Gram inputs must be two-dimensional and length matched.")
    if table.ndim != 2 or min(table.shape) <= n:
        raise ValueError("Token table does not include every value.")
    out = np.zeros((len(left), len(right)), dtype=np.float64)
    for j in range(n):
        out += table[left[:, j, None], right[None, :, j]]
    return out


def position_gram(embedding_weight, left, right=None, length=None):
    left, n = _permutations(left, length)
    return position_gram_from_table(
        embedding_similarity_table(embedding_weight, n), left, right, n
    )


def onehot_position_gram(left, right=None, length=None):
    """Onehot Gram has the exact number of equal values at corresponding positions."""
    left, n = _permutations(left, length)
    right, rn = _permutations(left if right is None else right, n)
    if left.ndim != 2 or right.ndim != 2 or rn != n:
        raise ValueError("Gram inputs must be two-dimensional and length matched.")
    out = np.zeros((len(left), len(right)), dtype=np.float64)
    for j in range(n):
        out += left[:, j, None] == right[None, :, j]
    return out


def centering_matrix(count, keys=None):
    """Center globally or within exactly the supplied frozen answer strata."""
    count = int(count)
    h = np.eye(count, dtype=np.float64)
    if count == 0:
        return h
    if keys is None:
        return h - np.ones((count, count), dtype=np.float64) / count
    keys = np.asarray(keys)
    if len(keys) != count:
        raise ValueError("Centering keys do not match sample count.")
    if keys.ndim == 1:
        keys = keys[:, None]
    _, groups, sizes = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    for group, size in enumerate(sizes):
        selected = np.flatnonzero(groups == group)
        h[np.ix_(selected, selected)] -= 1.0 / size
    return h


def center_gram(gram, row_keys=None, column_keys=None):
    """Apply the corresponding feature centering on each side of a raw Gram.

    For square self Gram matrices and identical strata, pass the same keys as
    row_keys and column_keys. For test/fit cross Grams, pass test then fit keys.
    """
    gram = np.asarray(gram, dtype=np.float64)
    if gram.ndim != 2:
        raise ValueError("Gram must be a matrix.")
    return (
        centering_matrix(gram.shape[0], row_keys)
        @ gram
        @ centering_matrix(gram.shape[1], column_keys)
    )


def split_position_grams(embedding_weight, permutations, fit_ids, test_ids,
                         length=None, keys=None, onehot=False):
    """Return (fit Gram, test Gram, test-fit Gram) under the frozen centering."""
    p, n = _permutations(permutations, length)
    if p.ndim != 2:
        raise ValueError("Split input must contain one permutation per row.")
    fi, ti = np.asarray(fit_ids, dtype=int), np.asarray(test_ids, dtype=int)
    fk = None if keys is None else np.asarray(keys)[fi]
    tk = None if keys is None else np.asarray(keys)[ti]
    table = None if onehot else embedding_similarity_table(embedding_weight, n)
    def kernel(a, b):
        return (onehot_position_gram(a, b, n) if onehot
                else position_gram_from_table(table, a, b, n))
    return (
        center_gram(kernel(p[fi], p[fi]), fk, fk),
        center_gram(kernel(p[ti], p[ti]), tk, tk),
        center_gram(kernel(p[ti], p[fi]), tk, fk),
    )


def _align(k, l):
    den = np.linalg.norm(k) * np.linalg.norm(l)
    return float(np.sum(k * l) / den) if den > 1e-20 else None


def cka_from_grams(k, l, debiased=False):
    """Exact centered linear CKA using a pair of self Gram matrices."""
    k, l = np.asarray(k, dtype=float), np.asarray(l, dtype=float)
    if k.shape != l.shape or k.ndim != 2 or k.shape[0] != k.shape[1]:
        raise ValueError("CKA requires aligned square Gram matrices.")
    n = len(k)
    if n < 2 or (debiased and n < 4):
        return None
    k, l = center_gram(k), center_gram(l)
    if debiased:
        for a in (k, l):
            np.fill_diagonal(a, 0)
            sums = a.sum(1)
            a -= sums[:, None] / (n - 2) + sums[None, :] / (n - 2)
            a += sums.sum() / ((n - 1) * (n - 2))
            np.fill_diagonal(a, 0)
    return _align(k, l)


def _neighbors_from_gram(gram, k=10, ids=None, duplicate_keys=None):
    gram = np.asarray(gram, dtype=np.float64)
    n = len(gram)
    if gram.shape != (n, n):
        raise ValueError("Neighbor Gram must be square.")
    if n <= k or float(np.trace(center_gram(gram))) <= 1e-20:
        return None
    ids = np.arange(n) if ids is None else np.asarray(ids)
    if ids.shape != (n,) or len(np.unique(ids)) != n:
        raise ValueError("Neighbor sample IDs must be unique and aligned.")
    order = np.argsort(ids, kind="stable")
    gram = gram[np.ix_(order, order)]
    diag = np.diag(gram)
    distance = np.maximum(diag[:, None] + diag[None, :] - 2 * gram, 0)
    np.fill_diagonal(distance, np.inf)
    if duplicate_keys is not None:
        keys = np.asarray(duplicate_keys)[order].reshape(n, -1)
        distance[np.all(keys[:, None, :] == keys[None, :, :], axis=-1)] = np.inf
    if np.any(np.isfinite(distance).sum(1) < k):
        return None
    chosen = np.argsort(distance, axis=1, kind="stable")[:, :k]
    sets = [set(ids[order][row].tolist()) for row in chosen]
    return [sets[i] for i in np.argsort(order)]


def _pca_from_grams(fit, testfit, rank):
    fit = np.asarray(fit, dtype=np.float64)
    testfit = np.asarray(testfit, dtype=np.float64)
    if len(fit) <= rank or float(np.trace(fit)) <= 1e-20:
        return None
    eigenvalues, vectors = np.linalg.eigh((fit + fit.T) * 0.5)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues, vectors = eigenvalues[order], vectors[:, order]
    if eigenvalues[0] <= 0:
        return None
    # Frozen feature implementation: s > s[0]*1e-6. Eigenvalues are s^2.
    if np.sum(eigenvalues > eigenvalues[0] * 1e-12) < rank:
        return None
    roots = np.sqrt(eigenvalues[:rank])
    vectors = vectors[:, :rank]
    fit_projected = vectors * roots
    scale = np.sqrt(np.square(fit_projected).sum() / len(fit_projected))
    if scale <= 1e-12:
        return None
    return fit_projected / scale, (testfit @ vectors / roots) / scale


def scores_from_grams(Kfit, Lfit, Ktest, Ltest, Ktestfit, Ltestfit,
                      ids=None, rank=8, duplicate_keys=None):
    """Frozen metric values from exact Grams; returns (metric dict, PCA rank).

    All six Grams must already reflect the independently centered feature
    splits (global or answer strata). Ktestfit/Ltestfit have test rows and fit
    columns. rank is the predeclared feature rank, normally eight. Undefined
    criteria and sample-ID tie handling match the frozen feature metrics.
    """
    rank = int(rank)
    if rank < 1:
        raise ValueError("PCA rank must be positive.")
    x = _pca_from_grams(Kfit, Ktestfit, rank)
    y = _pca_from_grams(Lfit, Ltestfit, rank)
    error = None
    if x is not None and y is not None and len(Ktest) >= 2:
        xf, xt = x
        yf, yt = y
        u, _, vt = np.linalg.svd(xf.T @ yf, full_matrices=False)
        den = np.square(yt).sum()
        if den > 1e-20:
            error = float(np.sqrt(np.square(xt @ (u @ vt) - yt).sum() / den))
    a = _neighbors_from_gram(Ktest, ids=ids, duplicate_keys=duplicate_keys)
    b = _neighbors_from_gram(Ltest, ids=ids, duplicate_keys=duplicate_keys)
    neighbors = None if a is None or b is None else float(
        np.mean([len(u & v) / 10 for u, v in zip(a, b)])
    )
    return {
        "linear_cka": (cka_from_grams(Ktest, Ltest), None),
        "debiased_cka": (cka_from_grams(Ktest, Ltest, True), None),
        "knn_overlap": (neighbors, None),
        "procrustes": (-error if error is not None else None, error),
    }, rank


def structural_constant_metric_values():
    return {key: (None, None) for key in
            ("linear_cka", "debiased_cka", "knn_overlap", "procrustes")}


def _decomposed(values):
    """Pure equivalent of the inherited evaluator's decomposition, no writers."""
    trained, initial = values["trained"], values.get("initial", {})
    result = {}
    def difference(a, b):
        return float(a-b) if a is not None and b is not None else None
    for condition, collection in (("trained", trained), ("initial", initial)):
        for role in ("correct", "wrong", "identity"):
            value, error = collection.get(role, (None, None))
            result[f"{condition}_{role}"] = value
            result[f"{condition}_{role}_raw_error"] = error
        result[f"{condition}_preference"] = difference(
            result[f"{condition}_correct"], result[f"{condition}_wrong"])
    result["B0"] = result["initial_preference"]
    result["Bt"] = result["trained_preference"]
    result["delta_correct"] = difference(result["trained_correct"], result["initial_correct"])
    result["delta_wrong"] = difference(result["trained_wrong"], result["initial_wrong"])
    result["delta"] = difference(result["Bt"], result["B0"])
    if result["delta"] is not None:
        assert abs(result["delta"]-(result["delta_correct"]-result["delta_wrong"])) < 1e-10
    return result


def relation_input_rows(metadata, fit, test, permutations, actions, embeddings, labels):
    """Input-baseline rows compatible with the inherited comparison_rows schema.

    permutations is N x 4 x max_n; actions indexes identity/correct/wrong.
    embeddings maps initial/trained to (left numeric embedding, right embedding).
    labels has left N-vector and right N x 3 identity/correct/wrong labels.
    The onehot representation is static: it has no initialization or increment.
    """
    p = np.asarray(permutations)
    fi_all, ti_all = np.asarray(fit, dtype=int), np.asarray(test, dtype=int)
    actions = np.asarray(actions, dtype=int)
    if p.ndim != 3 or actions.shape != (3,) or actions[0] != 0:
        raise ValueError("Input array or identity/correct/wrong action indices invalid.")
    n = int(metadata["length"])
    p = p[..., :n]
    joint = np.column_stack((np.asarray(labels["left"]), np.asarray(labels["right"])))
    if joint.shape != (len(p), 4):
        raise ValueError("Joint answer control must contain all four frozen labels.")
    tables = {condition: (embedding_similarity_table(left, n),
                          embedding_similarity_table(right, n))
              for condition, (left, right) in embeddings.items()}
    rows = []
    for control in ("within_length", "answer_strata"):
        selected, keys = {}, {}
        for split, indices in (("fit", fi_all), ("test", ti_all)):
            _, groups, counts = np.unique(joint[indices], axis=0,
                                           return_inverse=True, return_counts=True)
            use = indices[counts[groups] >= 3] if control == "answer_strata" else indices
            selected[split] = use
            keys[split] = joint[use] if control == "answer_strata" else None
        fi, ti = selected["fit"], selected["test"]
        hfit, htest = centering_matrix(len(fi), keys["fit"]), centering_matrix(len(ti), keys["test"])
        def grams(table, action, onehot=False):
            selected_p = p[:, action]
            def raw(left, right):
                return (onehot_position_gram(left, right, n) if onehot
                        else position_gram_from_table(table, left, right, n))
            raw_fit, raw_test = raw(selected_p[fi], selected_p[fi]), raw(selected_p[ti], selected_p[ti])
            return (hfit @ raw_fit @ hfit, htest @ raw_test @ htest,
                    htest @ raw(selected_p[ti], selected_p[fi]) @ hfit, raw_test)
        for branch in ("position_numeric_embedding", "position_numeric_onehot"):
            collections = {metric: {} for metric in
                           ("linear_cka", "debiased_cka", "knn_overlap", "procrustes")}
            conditions = embeddings if branch == "position_numeric_embedding" else {"trained": None}
            for condition in conditions:
                onehot = branch == "position_numeric_onehot"
                left_table, right_table = (None, None) if onehot else tables[condition]
                left = grams(left_table, 0, onehot)
                for role, action in zip(("identity", "correct", "wrong"), actions):
                    right = grams(right_table, int(action), onehot)
                    values, _ = scores_from_grams(left[0], right[0], left[1], right[1],
                                                  left[2], right[2], ids=ti, rank=8)
                    if onehot and control == "within_length":
                        # Common centering changes no Euclidean distances.
                        # Raw integer Gram prevents rounding from spuriously
                        # splitting the many exact onehot distance ties.
                        a = _neighbors_from_gram(left[3], ids=ti)
                        b = _neighbors_from_gram(right[3], ids=ti)
                        overlap = None if a is None or b is None else float(
                            np.mean([len(u & v)/10 for u, v in zip(a, b)]))
                        values["knn_overlap"] = (overlap, None)
                    for metric, value in values.items():
                        collections[metric].setdefault(condition, {})[role] = value
            for metric, values in collections.items():
                rows.append(dict(**metadata, branch=branch, control=control, metric=metric,
                                 fit_anchors=len(fi), test_anchors=len(ti),
                                 planned_fit_anchors=len(fi_all), planned_test_anchors=len(ti_all),
                                 procrustes_rank=8 if metric == "procrustes" else "",
                                 input_numerical_rule=(
                                     "Raw integer onehot Gram for Euclidean KNN; common centering preserves distance; exact ties by sample ID."
                                     if branch == "position_numeric_onehot" and control == "within_length" and metric == "knn_overlap"
                                     else "Exact position Gram; split/answer-stratum centering; frozen metric definitions."),
                                 **_decomposed(values)))
    return rows


def _frozen_metrics():
    spec = importlib.util.spec_from_file_location("_frozen_permworld_metrics", FROZEN_METRICS)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate() -> dict[str, Any]:
    """Independent CPU checks, without source training or old-output writers."""
    frozen = _frozen_metrics()
    rng = np.random.default_rng(2026100961)
    n = 7
    p = np.stack([rng.permutation(n) + 1 for _ in range(80)])
    q = np.argsort(p, axis=1) + 1
    fi, ti = np.arange(40), np.arange(40, 80)
    weight = rng.normal(size=(31, 13))
    weight2 = rng.normal(size=(31, 17))
    positional = rng.normal(size=(n, 13))
    keys = np.repeat(np.arange(8), 10)[:, None]
    max_errors = {"position_centering": 0.0, "position_metric_invariance": 0.0,
                  "token_lookup_gram": 0.0, "gram_feature_metrics": 0.0,
                  "independent_covariance_cka": 0.0, "onehot_theoretical_cka": 0.0}
    x, y = position_features(weight, p), position_features(weight2, q)
    with_positions = (weight[p] + positional[None]).reshape(len(p), -1)
    for control_keys in (None, keys):
        fk = None if control_keys is None else control_keys[fi]
        tk = None if control_keys is None else control_keys[ti]
        xf, yf = frozen.center(x[fi], fk), frozen.center(y[fi], fk)
        xt, yt = frozen.center(x[ti], tk), frozen.center(y[ti], tk)
        xpfit = frozen.center(with_positions[fi], fk)
        xptest = frozen.center(with_positions[ti], tk)
        max_errors["position_centering"] = max(
            max_errors["position_centering"], float(np.max(np.abs(xf - xpfit))),
            float(np.max(np.abs(xt - xptest))))
        ks = split_position_grams(weight, p, fi, ti, keys=control_keys)
        ls = split_position_grams(weight2, q, fi, ti, keys=control_keys)
        for actual, expected in zip(ks, (xf @ xf.T, xt @ xt.T, xt @ xf.T)):
            max_errors["token_lookup_gram"] = max(
                max_errors["token_lookup_gram"], float(np.max(np.abs(actual - expected))))
        reference, r = frozen.scores(xf, yf, xt, yt, ids=ti)
        positions, _ = frozen.scores(xpfit, yf, xptest, yt, ids=ti)
        gram_values, gr = scores_from_grams(ks[0], ls[0], ks[1], ls[1], ks[2], ls[2], ids=ti)
        assert r == gr == 8
        for metric in reference:
            for reference_value, position_value, gram_value in zip(
                    reference[metric], positions[metric], gram_values[metric]):
                if reference_value is None:
                    assert position_value is None and gram_value is None
                    continue
                max_errors["position_metric_invariance"] = max(
                    max_errors["position_metric_invariance"], abs(reference_value-position_value))
                max_errors["gram_feature_metrics"] = max(
                    max_errors["gram_feature_metrics"], abs(reference_value-gram_value))
        independent = np.square(xt.T @ yt).sum() / np.sqrt(
            np.square(xt.T @ xt).sum() * np.square(yt.T @ yt).sum())
        max_errors["independent_covariance_cka"] = max(
            max_errors["independent_covariance_cka"],
            abs(independent-gram_values["linear_cka"][0]))
    onehot_x, onehot_y = onehot_position_features(p), onehot_position_features(q)
    theoretical_x, theoretical_y = onehot_position_gram(p), onehot_position_gram(q)
    assert np.array_equal(theoretical_x, onehot_x @ onehot_x.T)
    assert np.array_equal(theoretical_y, onehot_y @ onehot_y.T)
    max_errors["onehot_theoretical_cka"] = abs(
        frozen.cka(onehot_x, onehot_y)-cka_from_grams(theoretical_x, theoretical_y))
    onehot_feature_knn = frozen.knn_overlap(frozen.center(onehot_x[ti]),
                                            frozen.center(onehot_y[ti]), ids=ti)
    exact_a = _neighbors_from_gram(theoretical_x[np.ix_(ti, ti)], ids=ti)
    exact_b = _neighbors_from_gram(theoretical_y[np.ix_(ti, ti)], ids=ti)
    onehot_exact_knn = float(np.mean([len(a & b)/10 for a, b in zip(exact_a, exact_b)]))
    assert np.array_equal(theoretical_x, theoretical_y) and onehot_exact_knn == 1.0
    duplicate_p = p.copy(); duplicate_p[41] = duplicate_p[40]
    duplicate_q = q.copy(); duplicate_q[41] = duplicate_q[40]
    dx, dy = position_features(weight, duplicate_p), position_features(weight2, duplicate_q)
    dks = split_position_grams(weight, duplicate_p, fi, ti)
    dls = split_position_grams(weight2, duplicate_q, fi, ti)
    dref, _ = frozen.scores(frozen.center(dx[fi]), frozen.center(dy[fi]),
                           frozen.center(dx[ti]), frozen.center(dy[ti]),
                           ids=ti[::-1], duplicate_keys=duplicate_p[ti])
    dgram, _ = scores_from_grams(dks[0], dls[0], dks[1], dls[1], dks[2], dls[2],
                                ids=ti[::-1], duplicate_keys=duplicate_p[ti])
    assert dref["knn_overlap"] == dgram["knn_overlap"]
    # Frozen behavior in the insufficient-rank, constant and too-small cases.
    zero = np.zeros((12, 12))
    zvalues, _ = scores_from_grams(zero, zero, zero, zero, zero, zero)
    assert all(value == (None, None) for value in zvalues.values())
    assert _neighbors_from_gram(np.eye(10)) is None
    assert cka_from_grams(np.eye(3), np.eye(3), True) is None
    tied = np.array([0., 1., -1.])[:, None]
    assert _neighbors_from_gram(tied @ tied.T, k=1, ids=[5, 2, 9])[0] == {2}
    assert _pca_from_grams(np.ones((12, 12)), np.ones((12, 12)), 8) is None
    sample_perms = np.stack((p, p[:, ::-1], q, p[:, ::-1]), axis=1)
    sample_labels = {"left": np.zeros(len(p), dtype=int),
                     "right": np.zeros((len(p), 3), dtype=int)}
    sample_rows = relation_input_rows({"length": n}, fi, ti, sample_perms, [0, 2, 1],
                                     {"initial": (weight, weight2),
                                      "trained": (weight, weight2)}, sample_labels)
    assert len(sample_rows) == 16
    for row in sample_rows:
        if row["branch"] == "position_numeric_onehot":
            assert row["B0"] is None and row["delta"] is None
        else:
            assert row["delta"] is None or abs(row["delta"]) < 1e-12
        assert row["fit_anchors"] == row["test_anchors"] == 40
    for key, error in max_errors.items():
        assert error < 2e-10, (key, error)
    return {
        "status": "passed", "new_source_training": 0, "device": "CPU",
        "rng_seed": 2026100961, "maximum_absolute_errors": max_errors,
        "controls_checked": ["within_length", "answer_strata"],
        "duplicate_exclusion_and_sample_id_ties": True,
        "static_onehot_has_no_initialization_or_increment": True,
        "relation_rows_schema_and_same_initial_trained_zero_delta": True,
        "onehot_tie_numerical_boundary": {
            "equal_gram_under_inverse": True,
            "frozen_float_feature_knn": onehot_feature_knn,
            "exact_integer_distance_knn": onehot_exact_knn,
            "interpretation": "Floating centering can spuriously split exact onehot distance ties. Input within-length KNN uses raw integer Gram, mathematically identical Euclidean distances, with the predeclared sample-ID tie rule. Hidden feature metrics are unchanged.",
        },
        "structural_constant_branches": STRUCTURAL_CONSTANT_BRANCHES,
        "frozen_metric_code": str(FROZEN_METRICS),
        "note": "Exact algebraic input metric implementation; source weights and source outputs unchanged.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--output", type=Path, default=HERE / "input_geometry_validation.json")
    arguments = parser.parse_args()
    if not arguments.validate:
        parser.error("Specify --validate; this module does not launch training.")
    record = validate()
    arguments.output.write_text(json.dumps(record, indent=2, ensure_ascii=False)+"\n")
    print(json.dumps(record, ensure_ascii=False))
