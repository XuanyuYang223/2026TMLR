"""Read-only localization of the two frozen output-probability kNN failures.

This diagnostic consumes only the exact original 64 anchors and saved arrays.
It does not replace a production distance, score, tolerance, or failed audit.
Crossed arithmetic paths and high-precision distances are localization aids,
not additional scientific metrics or successful independent verification.
"""
from __future__ import annotations

from collections import Counter
from decimal import Decimal, localcontext
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import time

import numpy as np
from threadpoolctl import threadpool_limits, threadpool_info

DIAG = Path(__file__).resolve().parent
WORKSPACE = DIAG.parents[1]
ROOT = Path('/mnt/yangx-projects-f/20261011/matrixgroupworld')
METRICS = WORKSPACE / 'diagnostics/permworld_scope_inventory_20261009/metrics.py'
INDEPENDENT = WORKSPACE / 'diagnostics/permworld_controlled_comparison_20261010/metric_review/independent_metrics.py'
EXPECTED_METRICS = 'c2a384a8212e661d1917c084a74699297bbef354f06dac42ac75ea90ce3d0b5c'
EXPECTED_INDEPENDENT = '6dbb804d22b7c4559bc8e3b41140efc50df5008c8a24698652525d57ebf0b8a3'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def array_sha(value):
    a = np.ascontiguousarray(value)
    h = hashlib.sha256(str(a.dtype).encode() + repr(a.shape).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def gram_distances(values, ids, orbits):
    # Exact production arithmetic and stable-ID sorting; no repaired distance.
    order = np.argsort(ids, kind='stable')
    x = np.asarray(values, np.float64)[order]
    raw = np.square(x).sum(1)[:, None] + np.square(x).sum(1)[None, :] - 2*x@x.T
    distances = np.maximum(raw, 0)
    keys = orbits[order]
    excluded = np.all(keys[:, None, :] == keys[None, :, :], axis=-1)
    np.fill_diagonal(excluded, True)
    distances[excluded] = np.inf
    # Return in original row order, including candidate columns.
    inverse = np.argsort(order)
    return distances[inverse][:, inverse], raw[inverse][:, inverse]


def direct_distances(values, ids, orbits):
    x = np.asarray(values, np.float64)
    d = np.full((len(x), len(x)), np.inf, np.float64)
    for i in range(len(x)):
        for j in range(len(x)):
            if i == j or np.array_equal(orbits[i], orbits[j]):
                continue
            d[i, j] = math.fsum(float(v)**2 for v in x[i]-x[j])
    return d


def sorted_candidates(distance_row, ids):
    return sorted((float(distance_row[j]), int(ids[j]), j)
                  for j in range(len(ids)) if math.isfinite(distance_row[j]))


def sets_from_distances(distances, ids):
    return [{identifier for _, identifier, _ in sorted_candidates(row, ids)[:10]}
            for row in distances]


def exact_raw_distance(values, i, j):
    # FP32 cached numbers are exactly representable as binary64. Decimal.from_float
    # records that exact value; 100 digits is ample for these squared distances.
    with localcontext() as context:
        context.prec = 100
        return str(sum((Decimal.from_float(float(a))-Decimal.from_float(float(b)))**2
                       for a, b in zip(values[i], values[j])))


def overlap(left, right):
    return math.fsum(len(a & b)/10 for a, b in zip(left, right))/len(left)


def main():
    started = time.monotonic()
    assert sha(METRICS) == EXPECTED_METRICS
    assert sha(INDEPENDENT) == EXPECTED_INDEPENDENT
    production = load(METRICS, 'matrix_knn_original_failure_path')
    independent = load(INDEPENDENT, 'matrix_knn_independent_failure_path')
    audit_path = DIAG/'actual_numerical_verification.json'
    audit = read(audit_path)
    failures = [r for r in audit['records'] if r['status'] == 'failed']
    expected = {('unit001', 'R1', 6, 'raw_within_n', 'output_prob', 'trained', role, 'knn_overlap')
                for role in ('correct', 'wrong')}
    assert {tuple(r[k] for k in ('unit', 'relation', 'n', 'control', 'branch', 'phase', 'role', 'metric'))
            for r in failures} == expected
    assert audit['status'] == 'failed' and audit['failure_count'] == 2
    config = read(DIAG/'protocol.json')
    cache_index = read(ROOT/'evaluation_cache/index.json')
    assert cache_index['protocol_sha256'] == audit['protocol_sha256'] == sha(DIAG/'protocol.json')
    assert sha(ROOT/'evaluation_cache/index.json') == audit['cache_index_sha256']
    world = next(u for u in config['data_units'] if u['unit'] == 'unit001')
    assert sha(world['data_path']) == world['data_sha256']
    with np.load(world['data_path'], allow_pickle=False) as dataset:
        lengths = dataset['analysis_lengths']
        anchor_names = dataset['analysis_ids']
        all_orbits = dataset['analysis_orbit_ids']
    ids = np.flatnonzero(lengths == 6)[:64]
    assert len(ids) == 64
    assert all(array_sha(ids) == r['analysis_ids_sha256'] for r in failures)
    orbits = all_orbits[ids, None]
    saved = {}
    cache_files = []
    for task in ('F1', 'F2'):
        record = next(r for r in cache_index['records'] if
                      (r['unit'], r['task'], r['condition'], r['split']) ==
                      ('unit001', task, 'trained', 'analysis'))
        assert sha(record['path']) == record['metadata']['sha256']
        cache_files.append(Path(record['path']))
        with np.load(record['path'], allow_pickle=False) as arrays:
            saved[task] = arrays['output_prob'][ids]
    roles = {'correct': cache_index['actions'].index('transpose'),
             'wrong': cache_index['actions'].index('P_transpose')}
    raw = {'left_identity': saved['F1'][:, 0],
           **{'right_'+role: saved['F2'][:, action] for role, action in roles.items()}}
    protected = [audit_path, DIAG/'protocol.json', DIAG/'freeze_manifest.json',
                 ROOT/'evaluation_cache/index.json', Path(world['data_path']),
                 METRICS, INDEPENDENT, *cache_files,
                 *sorted((WORKSPACE/'experiments/matrixgroupworld_20261011').glob('*.py')),
                 *sorted(DIAG.glob('geometry_*.csv')), *sorted(DIAG.glob('pair_*.csv'))]
    before = {str(path): sha(path) for path in protected}
    rows, feature_reports, scores, paths = [], {}, [], {}
    with threadpool_limits(limits=2):
        pools = threadpool_info()
        for name, values in raw.items():
            pc = production.center(values)
            ic = independent.independent_center(values)
            psets = production.neighbors(pc, 10, ids=ids, duplicate_keys=orbits)
            isets = independent.brute_neighbors(ic, 10, ids=ids, duplicate_keys=orbits)
            gd, untrimmed = gram_distances(pc, ids, orbits)
            dd = direct_distances(ic, ids, orbits)
            assert sets_from_distances(gd, ids) == psets
            assert sets_from_distances(dd, ids) == isets
            # Arithmetic localization only: keep the frozen scientific methods.
            direct_pc = direct_distances(pc, ids, orbits)
            gram_ic, _ = gram_distances(ic, ids, orbits)
            paths[name] = dict(production=psets, independent=isets,
                               direct_production_center=sets_from_distances(direct_pc, ids),
                               gram_independent_center=sets_from_distances(gram_ic, ids))
            differing = [i for i, (a, b) in enumerate(zip(psets, isets)) if a != b]
            finite = np.isfinite(gd)
            feature_reports[name] = dict(
                raw_shape=list(values.shape), raw_dtype=str(values.dtype), raw_array_sha256=array_sha(values),
                max_centering_absolute_difference=float(np.max(np.abs(pc-ic))),
                exact_duplicate_raw_rows=64-len(np.unique(values, axis=0)),
                maximal_probability=dict(min=float(values.max(1).min()), median=float(np.median(values.max(1))),
                                         max=float(values.max(1).max()), exactly_one_count=int((values.max(1)==1).sum())),
                median_nonmax_probability=float(np.median(np.sort(values, axis=1)[:, :-1])),
                max_gram_vs_direct_distance_absolute_difference=float(np.max(np.abs(gd[finite]-dd[finite]))),
                negative_unclipped_gram_distance_count=int((untrimmed[finite]<0).sum()),
                changed_neighbor_sample_count=len(differing), changed_sample_ids=[int(ids[i]) for i in differing],
                centering_only_direct_set_changes=sum(a != b for a, b in zip(paths[name]['direct_production_center'], isets)),
                centering_only_gram_set_changes=sum(a != b for a, b in zip(psets, paths[name]['gram_independent_center'])),
                same_production_center_gram_vs_direct_set_changes=sum(a != b for a, b in zip(psets, paths[name]['direct_production_center'])))
            for i in differing:
                gorder = sorted_candidates(gd[i], ids)
                dorder = sorted_candidates(dd[i], ids)
                chosen = sorted({item[2] for item in gorder[:13]+dorder[:13]})
                gpos = {identifier: p+1 for p, (_, identifier, _) in enumerate(gorder)}
                dpos = {identifier: p+1 for p, (_, identifier, _) in enumerate(dorder)}
                feature_reports[name].setdefault('changed_rows', []).append(dict(
                    sample_id=int(ids[i]), anchor_id=str(anchor_names[ids[i]]), orbit_id=str(all_orbits[ids[i]]),
                    production_neighbors=sorted(psets[i]), independent_neighbors=sorted(isets[i]),
                    production_only=sorted(psets[i]-isets[i]), independent_only=sorted(isets[i]-psets[i]),
                    production_boundary_gap=float(gorder[10][0]-gorder[9][0]),
                    independent_boundary_gap=float(dorder[10][0]-dorder[9][0]),
                    production_boundary_tie=(gorder[10][0] == gorder[9][0]),
                    independent_boundary_tie=(dorder[10][0] == dorder[9][0])))
                for j in chosen:
                    rows.append(dict(feature=name, sample_id=int(ids[i]), candidate_id=int(ids[j]),
                        sample_anchor_id=str(anchor_names[ids[i]]), candidate_anchor_id=str(anchor_names[ids[j]]),
                        production_rank=gpos[int(ids[j])], independent_rank=dpos[int(ids[j])],
                        production_chosen=int(ids[j]) in psets[i], independent_chosen=int(ids[j]) in isets[i],
                        production_gram_squared_distance=float(gd[i, j]), independent_direct_squared_distance=float(dd[i, j]),
                        production_distance_hex=float(gd[i, j]).hex(), independent_distance_hex=float(dd[i, j]).hex(),
                        unclipped_gram_squared_distance=float(untrimmed[i, j]),
                        direct_distance_with_production_center=float(direct_pc[i, j]),
                        gram_distance_with_independent_center=float(gram_ic[i, j]),
                        high_precision_raw_squared_distance=exact_raw_distance(values, i, j),
                        high_precision_scope='Localization only; no repaired score or replacement method'))
        for failure in failures:
            right = paths['right_'+failure['role']]
            left = paths['left_identity']
            actual = production.knn_overlap(production.center(raw['left_identity']),
                production.center(raw['right_'+failure['role']]), ids=ids, duplicate_keys=orbits)
            expected_score = independent.brute_overlap(independent.independent_center(raw['left_identity']),
                independent.independent_center(raw['right_'+failure['role']]), ids=ids, duplicate_keys=orbits)
            assert abs(actual-failure['production']) <= 1e-14
            assert abs(expected_score-failure['independent']) <= 1e-14
            changed_rows = []
            for i in range(64):
                p = len(left['production'][i] & right['production'][i])
                q = len(left['independent'][i] & right['independent'][i])
                if p != q:
                    changed_rows.append(dict(sample_id=int(ids[i]), production_shared_neighbors=p,
                        independent_shared_neighbors=q, count_change=q-p,
                        left_set_changed=left['production'][i] != left['independent'][i],
                        right_set_changed=right['production'][i] != right['independent'][i]))
            scores.append(dict(role=failure['role'], original_failure_status='failed',
                frozen_production=failure['production'], frozen_independent=failure['independent'],
                production_path_reproduced=actual, independent_path_reproduced=expected_score,
                score_difference=expected_score-actual,
                shared_neighbor_count_difference=sum(r['count_change'] for r in changed_rows),
                changed_overlap_rows=changed_rows,
                crossed_arithmetic_localization_only={
                    'direct_distance_production_center': overlap(left['direct_production_center'], right['direct_production_center']),
                    'gram_distance_independent_center': overlap(left['gram_independent_center'], right['gram_independent_center'])}))
    after = {path: sha(path) for path in before}
    assert after == before, 'Protected source/result/cache changed during read-only diagnostic'
    failure_unchanged = read(audit_path)['status'] == 'failed' and sha(audit_path) == before[str(audit_path)]
    assert failure_unchanged
    result = dict(status='cause_localized_original_verification_remains_failed',
        scope=dict(unit='unit001', relation='R1', n=6, control='raw_within_n', phase='trained',
                   branch='output_prob', sample_count=64, neighbors_k=10, original_sample_ids_sha256=array_sha(ids),
                   original_orbit_exclusion=True, roles=['correct', 'wrong']),
        source_script_sha256=sha(__file__), metrics_source_sha256=sha(METRICS), independent_source_sha256=sha(INDEPENDENT),
        protocol_sha256=sha(DIAG/'protocol.json'), original_verification_sha256=sha(audit_path),
        original_verification_status='failed', original_status_counts=audit['status_counts'],
        retained_verified_scope='64 passed and 6 undefined-consistent fixed checks; all288 fixed selector checks passed. The2 output-probability/raw/trained kNN checks remain failed.',
        cpu_blas_threads=2, threadpools=pools, features=feature_reports, failed_score_reproduction=scores,
        protected_file_hashes=before, protected_file_count=len(before), protected_hashes_unchanged=True,
        models_constructed=0, new_forward=0, training_updates=0, new_production_metrics=0,
        frozen_parameters_changed=False, failed_status_replaced=False, tolerance_relaxed=False,
        wall_seconds=time.monotonic()-started)
    (DIAG/'knn_failure_diagnostic.json').write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    with (DIAG/'knn_failure_boundary_distances.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [])
        if rows:
            writer.writeheader(); writer.writerows(rows)
    print(json.dumps(dict(status=result['status'], features={name: {
        key: value for key, value in report.items() if key != 'changed_rows'} for name, report in feature_reports.items()},
        scores=scores, protected_file_count=len(before), wall_seconds=result['wall_seconds']), indent=2))


if __name__ == '__main__':
    main()
