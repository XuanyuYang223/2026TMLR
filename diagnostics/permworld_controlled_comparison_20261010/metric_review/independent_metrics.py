"""Fixed synthetic independent checks; CPU only, no real feature/test loading.

The production definitions remain the original frozen metrics.py. These
separate formulations support small, predetermined numerical audits without
changing k, rank, centering, sample masks or any scientific endpoint.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from pathlib import Path

import numpy as np


WORKSPACE = Path(__file__).resolve().parents[3]
ORIGINAL = WORKSPACE/'diagnostics/permworld_scope_inventory_20261009/metrics.py'
EXPECTED_ORIGINAL_SHA256 = 'c2a384a8212e661d1917c084a74699297bbef354f06dac42ac75ea90ce3d0b5c'
OUTPUT = Path(__file__).resolve().parent
TOLERANCE = 1e-10


def independent_center(values, keys=None):
    x = np.asarray(values, np.float64)
    if len(x) == 0:
        return x.copy()
    if keys is None:
        return x-np.array([math.fsum(x[:, j])/len(x) for j in range(x.shape[1])])
    _, groups = np.unique(keys, axis=0, return_inverse=True)
    design = np.eye(int(groups.max())+1)[groups]
    return x-design@((design.T@x)/design.sum(axis=0)[:, None])


def covariance_cka(left, right):
    if len(left) < 2:
        return None
    x, y = independent_center(left), independent_center(right)
    denominator = np.linalg.norm(x.T@x)*np.linalg.norm(y.T@y)
    return float(np.square(x.T@y).sum()/denominator) if denominator > 1e-20 else None


def independent_u_cka(left, right):
    n = len(left)
    if n < 4:
        return None
    def u_gram(values):
        z = independent_center(values)
        gram = z@z.T
        off = gram.copy()
        np.fill_diagonal(off, 0)
        rows = np.array([math.fsum(off[i]) for i in range(n)])
        total = math.fsum(rows)
        out = np.zeros_like(gram)
        for i in range(n):
            for j in range(n):
                if i != j:
                    out[i, j] = gram[i, j]-(rows[i]+rows[j])/(n-2)+total/((n-1)*(n-2))
        return out
    a, b = u_gram(left), u_gram(right)
    denominator = math.sqrt(math.fsum(a.ravel()**2)*math.fsum(b.ravel()**2))
    return float(math.fsum((a*b).ravel())/denominator) if denominator > 1e-20 else None


def brute_neighbors(values, k=10, ids=None, duplicate_keys=None):
    x = np.asarray(values, np.float64)
    n = len(x)
    if n <= k or np.square(independent_center(x)).sum() <= 1e-20:
        return None
    ids = np.arange(n) if ids is None else np.asarray(ids)
    result = []
    for i in range(n):
        candidates = []
        for j in range(n):
            if i == j or (duplicate_keys is not None and np.array_equal(duplicate_keys[i], duplicate_keys[j])):
                continue
            distance = math.fsum(float(v)**2 for v in x[i]-x[j])
            candidates.append((distance, int(ids[j])))
        if len(candidates) < k:
            return None
        result.append({identifier for _, identifier in sorted(candidates)[:k]})
    return result


def brute_overlap(left, right, k=10, ids=None, duplicate_keys=None):
    a = brute_neighbors(left, k, ids, duplicate_keys)
    b = brute_neighbors(right, k, ids, duplicate_keys)
    if a is None or b is None:
        return None
    return math.fsum(len(u&v)/k for u, v in zip(a, b))/len(a)


def covariance_pca(values, rank):
    x = np.asarray(values, np.float64)
    if len(x) <= rank or np.square(x).sum() <= 1e-20:
        return None
    eigenvalues, eigenvectors = np.linalg.eigh(x.T@x)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0)
    if np.count_nonzero(eigenvalues > eigenvalues[0]*1e-12) < rank:
        return None
    basis = eigenvectors[:, order[:rank]]
    projected = x@basis
    scale = math.sqrt(math.fsum(projected.ravel()**2)/len(x))
    return (basis, scale) if scale > 1e-12 else None


def independent_procrustes(xfit, yfit, xeval, yeval, rank_cap=8):
    rank = min(rank_cap, xfit.shape[1], yfit.shape[1])
    left, right = covariance_pca(xfit, rank), covariance_pca(yfit, rank)
    if left is None or right is None or len(xeval) < 2:
        return None, rank
    basis_x, scale_x = left
    basis_y, scale_y = right
    xf, yf = xfit@basis_x/scale_x, yfit@basis_y/scale_y
    u, _, vt = np.linalg.svd(xf.T@yf, full_matrices=False)
    q = u@vt
    x, y = xeval@basis_x/scale_x, yeval@basis_y/scale_y
    energy = math.fsum(y.ravel()**2)
    if energy <= 1e-20:
        return None, rank
    return math.sqrt(math.fsum((x@q-y).ravel()**2)/energy), rank


def load_original():
    digest = hashlib.sha256(ORIGINAL.read_bytes()).hexdigest()
    assert digest == EXPECTED_ORIGINAL_SHA256, 'Frozen original metric source differs'
    spec = importlib.util.spec_from_file_location('permworld_controlled_metric_original', ORIGINAL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_fixture():
    original = load_original()
    rng = np.random.default_rng(202610103041)
    records = []
    def check(label, measured, expected, tolerance=TOLERANCE):
        if measured is None or expected is None:
            assert measured is None and expected is None, label
            error = None
        else:
            error = abs(float(measured)-float(expected))
            assert error <= tolerance, (label, measured, expected, error)
        records.append(dict(check=label, original=measured, independent=expected,
                            absolute_error=error, tolerance=tolerance, passed=True))
    xfit, yfit = rng.normal(size=(47, 11)), rng.normal(size=(47, 13))
    xeval, yeval = rng.normal(size=(31, 11)), rng.normal(size=(31, 13))
    check('ordinary CKA: feature covariance versus original Gram', original.cka(xeval, yeval), covariance_cka(xeval, yeval))
    check('U-centered CKA: explicit off-diagonal row loops', original.cka(xeval, yeval, True), independent_u_cka(xeval, yeval))
    ids = rng.permutation(np.arange(31)*7+101)
    duplicate_keys = np.column_stack((np.arange(31)//2, np.arange(31)//2))
    check('k10 overlap: direct Euclidean distances/stable IDs/duplicate exclusion',
          original.knn_overlap(xeval, yeval, ids=ids, duplicate_keys=duplicate_keys),
          brute_overlap(xeval, yeval, ids=ids, duplicate_keys=duplicate_keys))
    assert original.neighbors(xeval, ids=ids, duplicate_keys=duplicate_keys) == brute_neighbors(xeval, ids=ids, duplicate_keys=duplicate_keys)
    records.append(dict(check='all k10 neighbor ID sets equal', samples=31, passed=True))
    keys_fit = np.column_stack((np.repeat(np.arange(15), 3).tolist()+[15, 15], np.zeros(47)))
    check('joint-stratum centering: loop versus indicator design matrix',
          float(np.max(np.abs(original.center(xfit, keys_fit)-independent_center(xfit, keys_fit)))), 0.)
    centered = tuple(independent_center(x) for x in (xfit, yfit, xeval, yeval))
    actual_error, actual_rank = original.procrustes(*centered)
    independent_error, independent_rank = independent_procrustes(*centered)
    assert actual_rank == independent_rank == 8
    check('held-out rank8 PCA/fit RMS/orthogonal map/error: covariance-eigen PCA', actual_error, independent_error)
    scores, rank = original.scores(*centered, ids=ids, duplicate_keys=duplicate_keys)
    assert rank == 8
    check('Procrustes score is negative raw NRMSE', scores['procrustes'][0], -scores['procrustes'][1], 0.)
    ties = np.array([0., 1., -1., 2., -2., 3., -3., 4., -4., 5., -5., 6.])[:, None]
    tie_ids = np.array([51, 28, 91, 40, 3, 76, 18, 5, 62, 10, 38, 100])
    assert original.neighbors(ties, k=10, ids=tie_ids) == brute_neighbors(ties, k=10, ids=tie_ids)
    records.append(dict(check='exact distance ties use ascending stable ID with k10', passed=True))
    constant = np.ones((31, 11))
    check('constant CKA undefined', original.cka(constant, constant), None)
    check('constant k10 undefined', original.knn_overlap(constant, constant), None)
    check('insufficient U-centering sample count remains undefined', original.cka(xeval[:3], yeval[:3], True), None)
    check('rank8 never adaptively reduces to rank7',
          original.procrustes(np.column_stack((xfit[:, :7], np.zeros((47, 4)))), yfit, xeval, yeval)[0], None)
    all_duplicates = np.ones((31, 2), dtype=np.int32)
    check('fewer than10 eligible neighbors undefined', original.knn_overlap(xeval, yeval, duplicate_keys=all_duplicates), None)
    # Two missing length cells stay missing under the original complete21 rule.
    incomplete = [float(i)/21 for i in range(21)]
    incomplete[4] = None
    complete = math.fsum(incomplete)/21 if all(x is not None for x in incomplete) else None
    check('fixed21 complete-length rule does not renormalize missing cells', complete, None)
    return dict(status='passed', fixture_only=True, synthetic_seed=202610103041,
        original_metrics_path=str(ORIGINAL), original_metrics_sha256=EXPECTED_ORIGINAL_SHA256,
        checks=records, checks_count=len(records), fixed_procrustes_rank=8, fixed_knn_k=10,
        scientific_definition_changed=False, trained=False, real_feature_arrays_loaded=False,
        new_test_read=False, original_verify=original.verify())


if __name__ == '__main__':
    result = run_fixture()
    (OUTPUT/'fixed_fixture.json').write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ('status', 'fixture_only', 'checks_count', 'scientific_definition_changed')}))
