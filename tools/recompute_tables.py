#!/usr/bin/env python3
"""Portable saved-table arithmetic and identity checks, Python standard library only.

This does not calculate CKA, neighbors or Procrustes from activations, generate
inputs, import original experiment modules, construct models or run forwards.
Absolute historical paths in frozen provenance are never used as filesystem
fallbacks. The original MatrixGroup numerical failure remains a failure.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics
import sys
import time

csv.field_size_limit(16 * 1024 * 1024)
DEFAULT_LAYOUT = {
    'pair_protocol': 'diagnostics/representation_pairing_review_20261011/protocol.json',
    'pair_candidates': 'diagnostics/representation_pairing_review_20261011/candidate_pairs.csv',
    'pair_lengths': 'diagnostics/representation_pairing_review_20261011/pair_per_length.csv',
    'pair_summary': 'diagnostics/representation_pairing_review_20261011/pair_summary.csv',
    'pair_categories': 'diagnostics/representation_pairing_review_20261011/category_summary.csv',
    'pair_designated_length': 'diagnostics/representation_pairing_review_20261011/designated_endpoint_verification.csv',
    'pair_designated_summary': 'diagnostics/representation_pairing_review_20261011/designated_aggregate_verification.csv',
    'perm_original': 'diagnostics/permworld_eight_relations_20261010/per_length_results.csv',
    'graph_original': 'diagnostics/graphworld_pilot_20261010/geometry_per_n.csv',
    'matrix_protocol': 'diagnostics/matrixgroupworld_20261011/protocol.json',
    'matrix_lengths': 'diagnostics/matrixgroupworld_20261011/geometry_per_n.csv',
    'matrix_summary': 'diagnostics/matrixgroupworld_20261011/geometry_summary.csv',
    'matrix_numerical': 'diagnostics/matrixgroupworld_20261011/actual_numerical_verification.json',
    'matrix_completion': 'diagnostics/matrixgroupworld_20261011/completion.json',
    'matrix_report_audit': 'diagnostics/matrixgroupworld_20261011/report_audit.json',
    'matrix_math_audit': 'diagnostics/matrixgroupworld_20261011/math_audit.json',
    'matrix_arithmetic_audit': 'diagnostics/matrixgroupworld_20261011/table_arithmetic_audit.json',
}
PHASES = ('initial', 'trained')
CATEGORIES = ('designated', 'other_known_exact', 'not_designated')
COMPONENTS = ('B0', 'Bt', 'dc', 'du', 'Delta')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def number(value):
    if value is None or str(value).strip() in ('', 'None', 'null', 'NaN', 'nan'):
        return None
    value = float(value)
    if not math.isfinite(value):
        raise ValueError('Unexpected infinity; undefined cells must be explicit')
    return value


def subtract(a, b):
    return a - b if a is not None and b is not None else None


def fixed_mean(values, expected_lengths):
    if len(values) != len(expected_lengths):
        raise ValueError('Incomplete fixed-length grid')
    return math.fsum(values) / len(expected_lengths) if all(v is not None for v in values) else None


class Review:
    def __init__(self, root, layout):
        self.root = Path(root).resolve(); self.layout = layout
        self.input_sha256 = {}; self.check_counts = Counter(); self.errors = []
        self.maximum_absolute_differences = defaultdict(float)

    def path(self, key):
        relative = Path(self.layout[key])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Layout paths must be safe repository-relative paths: ' + key)
        path = (self.root / relative).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError('External symlink/host fallback is forbidden: ' + key)
        if not path.is_file():
            raise FileNotFoundError(f'Required clone-relative file is missing: {relative}; no host fallback is used')
        self.input_sha256[str(relative)] = sha(path)
        return path

    def rows(self, key):
        with self.path(key).open(newline='') as f:
            return list(csv.DictReader(f))

    def obj(self, key):
        return json.loads(self.path(key).read_text())

    def check(self, kind, condition, identity=''):
        self.check_counts[kind] += 1
        if not condition:
            self.errors.append({'check': kind, 'identity': str(identity)})

    def equal(self, kind, actual, expected, tolerance, identity=''):
        actual, expected = number(actual), number(expected)
        difference = None if actual is None or expected is None else abs(actual - expected)
        if difference is not None:
            self.maximum_absolute_differences[kind] = max(self.maximum_absolute_differences[kind], difference)
        self.check(kind, (actual is None) == (expected is None) and (actual is None or difference <= tolerance), identity)


def verify_checksums(review, checksum_path):
    values = json.loads(Path(checksum_path).read_text())
    if values['scope'] not in {
        'Frozen published scientific inputs; table/math/state checks only',
        'English working-tree edition; preserved numerical tables and translated metadata',
    }:
        raise ValueError('Unexpected checksum manifest scope')
    for key, record in values['inputs'].items():
        path = review.path(key)
        review.check('published_input_SHA256', sha(path) == record['sha256'], key)
        review.check('published_input_bytes', path.stat().st_size == record['bytes'], key)
    return {'manifest_sha256': sha(checksum_path), 'input_files': len(values['inputs'])}


def check_fixed_transform(review):
    protocol = review.obj('pair_protocol'); tolerance = protocol['existing_tolerance']
    review.check('frozen_endpoint_tolerance', tolerance == 1e-8)
    candidates = review.rows('pair_candidates'); lengths = review.rows('pair_lengths')
    summaries = review.rows('pair_summary'); categories = review.rows('pair_categories')
    manifest = {r['pair_id']: r for r in candidates}
    review.check('candidate_unique_ids', len(manifest) == len(candidates) == 492)
    review.check('candidate_domains_480_12', Counter(r['domain'] for r in candidates) == Counter(permworld=480, graphworld=12))
    review.check('candidate_manifest_binding', sha(review.path('pair_candidates')) == protocol['pair_manifest_sha256'])
    expected = {domain: protocol['systems'][domain]['lengths'] for domain in ('permworld', 'graphworld')}
    controls = {domain: protocol['systems'][domain]['controls'] for domain in expected}
    review.check('original_fixed_lengths', expected == {'permworld': list(range(10, 31)), 'graphworld': [6, 7, 8]})
    catalogue_groups = defaultdict(list)
    for row in candidates:
        review.check('frozen_candidate_category', row['category'] in CATEGORIES, row['pair_id'])
        review.check('designated_identity', (row['category'] == 'designated') == (row['right_task'] == row['designated_right_task']), row['pair_id'])
        review.check('different_candidate_task', row['right_task'] != row['left_task'], row['pair_id'])
        catalogue_groups[tuple(row[k] for k in ('domain', 'relation', 'architecture', 'encoding'))].append(row)
    for key, rows in catalogue_groups.items():
        review.check('complete_candidate_task_grid', len(rows) == (15 if key[0] == 'permworld' else 3) and
                     len({r['right_task'] for r in rows}) == len(rows) and sum(r['category'] == 'designated' for r in rows) == 1, key)
    groups = defaultdict(list); original = {}
    for domain, file_key in (('permworld', 'perm_original'), ('graphworld', 'graph_original')):
        review.check('original_comparison_table_binding', sha(review.path(file_key)) == protocol['systems'][domain]['original_results_sha256'], domain)
        for row in review.rows(file_key):
            valid = row['metric'] == 'linear_cka' and row['branch'] == ('hidden' if domain == 'permworld' else 'prefix_final')
            if domain == 'permworld': valid = valid and row['layer'] == 'final_norm'
            if valid:
                key = (domain, row['relation'], row.get('architecture', 'transformer'), row.get('encoding', 'row_major_offdiagonal_binary'),
                       row['control'], int(row['length'] if domain == 'permworld' else row['n']))
                review.check('unique_original_CKA_cell', key not in original, key); original[key] = row
    designated_checks = 0
    for row in lengths:
        identifier = (row['pair_id'], row['control'], int(row['length']))
        item = manifest[row['pair_id']]
        review.check('pair_row_manifest_identity', all(row[k] == item[k] for k in ('domain', 'relation', 'architecture', 'encoding', 'left_task', 'right_task', 'transform', 'category')), identifier)
        review.check('frozen_pair_control', row['control'] in controls[row['domain']], identifier)
        groups[row['pair_id'], row['control']].append(row)
        a, b = number(row['fixed_transform_score_initial']), number(row['fixed_transform_score_trained'])
        review.equal('pair_training_score_change', row['training_score_change'], subtract(b, a), tolerance, identifier)
        if row['category'] == 'designated':
            key = (row['domain'], row['relation'], row['architecture'], row['encoding'], row['control'], int(row['length']))
            source = original[key]
            for phase in PHASES:
                review.equal('designated_original_per_length', row['fixed_transform_score_' + phase], source[phase + '_correct'], tolerance, key + (phase,))
                designated_checks += 1
    recomputed = {}; designated_means = []; expected_count = 0
    for pair in candidates: expected_count += len(controls[pair['domain']])
    review.check('pair_summary_complete_grid', len(groups) == len(summaries) == expected_count == 996)
    review.check('pair_per_length_complete_grid', len(lengths) == 20268)
    saved = {(r['pair_id'], r['control']): r for r in summaries}
    review.check('pair_summary_unique_keys', len(saved) == len(summaries))
    for key, rr in groups.items():
        rr.sort(key=lambda r: int(r['length'])); domain = rr[0]['domain']
        grid = expected[domain]
        review.check('pair_fixed_lengths_complete_once', [int(r['length']) for r in rr] == grid, key)
        if [int(r['length']) for r in rr] != grid:
            continue
        s = saved[key]; out = {'pair_id': key[0], 'control': key[1]}
        for phase in PHASES:
            field = 'fixed_transform_score_' + phase; values = [number(r[field]) for r in rr]
            mean = fixed_mean(values, grid); out[field] = mean
            review.equal('fixed_equal_length_aggregate', s[field], mean, tolerance, key + (phase,))
            review.check('undefined_lengths_retained', s[phase + '_undefined_lengths'] == ','.join(str(n) for n, v in zip(grid, values) if v is None), key + (phase,))
            review.check('finite_length_count', int(s[phase + '_defined_lengths']) == sum(v is not None for v in values), key + (phase,))
            if rr[0]['category'] == 'designated':
                source_values = [number(original[(domain, r['relation'], r['architecture'], r['encoding'], r['control'], int(r['length']))][phase + '_correct']) for r in rr]
                review.equal('designated_original_aggregate', mean, fixed_mean(source_values, grid), tolerance, key + (phase,))
                designated_means.append({'pair_id': key[0], 'control': key[1], 'phase': phase, 'recomputed': mean})
        out['training_score_change'] = subtract(out['fixed_transform_score_trained'], out['fixed_transform_score_initial'])
        review.equal('summary_training_change', s['training_score_change'], out['training_score_change'], tolerance, key)
        recomputed[key] = out
    # Independently compare the full finite candidate distributions; empty
    # known-exact categories stay explicit rather than being dropped or filled.
    category_map = defaultdict(list)
    for row in summaries:
        for phase in PHASES:
            key = tuple(row[k] for k in ('domain', 'relation', 'architecture', 'encoding', 'control')) + (phase, row['category'])
            category_map[key].append(number(row['fixed_transform_score_' + phase]))
    review.check('category_row_grid', len(categories) == 456)
    for row in categories:
        key = tuple(row[k] for k in ('domain', 'relation', 'architecture', 'encoding', 'control', 'phase', 'category'))
        values = category_map[key]; finite = [v for v in values if v is not None]
        review.check('category_total_and_finite_counts', int(row['total_count']) == len(values) and int(row['defined_count']) == len(finite), key)
        review.equal('category_finite_mean', row['finite_mean'], math.fsum(finite) / len(finite) if finite else None, tolerance, key)
        review.equal('category_finite_median', row['finite_median'], statistics.median(finite) if finite else None, tolerance, key)
    review.check('designated_per_length_phase_scope', designated_checks == 2760)
    for key, field in (('pair_designated_length', 'tolerance'), ('pair_designated_summary', 'tolerance')):
        records = review.rows(key)
        review.check('original_designated_verification_record_count', len(records) == (2760 if key == 'pair_designated_length' else 152), key)
        for r in records:
            review.check('original_endpoint_record_tolerance', float(r[field]) == tolerance and r['passed'] == 'True', key)
    return {'candidate_pairs': {'permworld': 480, 'graphworld': 12}, 'category_counts': dict(Counter(r['category'] for r in candidates)),
            'per_length_control_rows': len(lengths), 'aggregate_control_rows': len(summaries),
            'designated_per_length_phase_checks': designated_checks, 'designated_aggregate_phase_checks': len(designated_means),
            'category_rows': len(categories), 'tolerance': tolerance,
            'interpretation': 'Fixed-T candidate scores and training_score_change are not original correct-minus-wrong Bt/Delta. Pair-specific answer/density controls and empty categories are preserved.'}


def decompose(row):
    scores = {phase + '_' + role: number(row[phase + '_' + role]) for phase in PHASES for role in ('correct', 'wrong', 'identity')}
    out = {'B0': subtract(scores['initial_correct'], scores['initial_wrong']),
           'Bt': subtract(scores['trained_correct'], scores['trained_wrong']),
           'dc': subtract(scores['trained_correct'], scores['initial_correct']),
           'du': subtract(scores['trained_wrong'], scores['initial_wrong'])}
    out['Delta'] = subtract(out['Bt'], out['B0'])
    return out


def check_matrix(review):
    p = review.obj('matrix_protocol'); rows = review.rows('matrix_lengths'); summaries = review.rows('matrix_summary')
    state = review.obj('matrix_completion'); reader = review.obj('matrix_report_audit'); verification = review.obj('matrix_numerical')
    arithmetic = review.obj('matrix_arithmetic_audit'); tolerance = arithmetic['arithmetic_tolerance']
    review.check('matrix_frozen_table_tolerance', tolerance == 1e-12)
    review.check('matrix_fixed_scope', p['lengths'] == [6, 8, 10] and p['tasks'] == ['F1', 'F2', 'F3', 'F4'] and len(p['units']) == 2 and len(p['training_manifest']) == 8 and p['training']['updates'] == 20000)
    relation_map = {r['relation']: (r['left'], r['right'], r['correct'], r['wrong'], r['primary']) for r in p['relations']}
    review.check('matrix_exact_relation_catalogue', relation_map == {'R1': ('F1', 'F2', 'transpose', 'P_transpose', True),
        'R2': ('F3', 'F4', 'transpose', 'Qt_transpose', False), 'R3': ('F1', 'F3', 'right_Q', 'right_Qt', True),
        'R4': ('F2', 'F4', 'left_Qt', 'left_Q', False)})
    review.check('matrix_primary_relations', p['evaluation']['primary_relations'] == ['R1', 'R3'])
    review.check('matrix_protocol_state_binding', state['protocol_sha256'] == sha(review.path('matrix_protocol')) == verification['protocol_sha256'])
    review.check('matrix_full_tables_scope', len(rows) == 432 and len(summaries) == 144)
    selected = [r for r in rows if r['relation'] in ('R1', 'R3') and r['branch'] == 'prefix_final' and r['metric'] == 'linear_cka']
    group = defaultdict(list)
    for row in selected:
        key = row['unit'], row['relation'], row['control']; group[key].append(row)
        decomposed = decompose(row)
        for field in COMPONENTS: review.equal('matrix_primary_per_n_' + field, row[field], decomposed[field], tolerance, key + (row['n'],))
        review.equal('matrix_primary_Delta_equals_dc_minus_du', row['Delta'], subtract(decomposed['dc'], decomposed['du']), tolerance, key + (row['n'],))
    saved = {(r['unit'], r['relation'], r['control']): r for r in summaries if r['relation'] in ('R1', 'R3') and r['branch'] == 'prefix_final' and r['metric'] == 'linear_cka'}
    review.check('matrix_primary_complete_grid', len(selected) == 24 and len(group) == len(saved) == 8)
    recomputed = []
    for key, rr in sorted(group.items()):
        rr.sort(key=lambda r: int(r['n'])); grid = [6, 8, 10]
        review.check('matrix_primary_fixed_three_n', [int(r['n']) for r in rr] == grid, key)
        if [int(r['n']) for r in rr] != grid: continue
        row = {'unit': key[0], 'relation': key[1], 'control': key[2]}
        for field in COMPONENTS:
            mean = fixed_mean([decompose(r)[field] for r in rr], grid)
            review.equal('matrix_primary_aggregate_' + field, saved[key][field], mean, tolerance, key)
            row[field] = mean
        recomputed.append(row)
    review.check('matrix_preserved_run_caveat', state['status'] == reader['run_mode'] == 'completed_with_verification_caveat' and state['formal_models'] == reader['completed20k_models'] == 8 and state['tables_complete'] is True and reader['tables_complete'] is True and state['scientific_analysis_complete'] is False and reader['scientific_analysis_complete'] is False)
    counts = Counter(r['status'] for r in verification['records'])
    review.check('matrix_preserved_64_2_6', verification['status'] == state['numerical_verification_status'] == 'failed' and counts == Counter(passed=64, failed=2, undefined_consistent=6) and counts == Counter(verification['status_counts']) and verification['failure_count'] == 2)
    review.check('matrix_selector_scope', verification['selection_check_count'] == len(verification['selection_checks']) == 288 and all(r['passed'] is True for r in verification['selection_checks']))
    failed = [r for r in verification['records'] if r['status'] == 'failed']
    review.check('matrix_failed_numerical_scope', len(failed) == 2 and {r['role'] for r in failed} == {'correct', 'wrong'} and
                 all((r['unit'], r['relation'], r['n'], r['control'], r['branch'], r['phase'], r['metric'], r['fit_count'], r['analysis_count'], r['tolerance']) ==
                     ('unit001', 'R1', 6, 'raw_within_n', 'output_prob', 'trained', 'knn_overlap', 64, 64, 1e-8) for r in failed))
    for r in failed:
        review.equal('preserved_failed_score_difference', r['absolute_error'], abs(r['production'] - r['independent']), 0., r['role'])
        review.check('preserved_failed_exceeds_tolerance', r['absolute_error'] > r['tolerance'], r['role'])
    review.check('table_arithmetic_does_not_override_numerical_failure', arithmetic['status'] == 'passed_table_arithmetic_only' and arithmetic['existing_independent_numerical_verification_status'] == 'failed' and arithmetic['existing_independent_numerical_verification_failure_count'] == 2)
    math_audit = review.obj('matrix_math_audit')
    review.check('preserved_math_audit_scope', math_audit['status'] == 'passed' and math_audit['exhaustive_GL_sizes'] == {'2': 6, '3': 168})
    # Small exact frozen-formula check only: re-use the n2 mathematical cases,
    # not scientific inputs, model predictions or new table measurements.
    def rank2(a): return ((a[0] & 1) * ((a[1] >> 1) & 1)) ^ (((a[0] >> 1) & 1) * (a[1] & 1))
    def transpose(a): return tuple(sum(((a[i] >> j) & 1) << i for i in range(2)) for j in range(2))
    def multiply(a, b): return tuple((b[0] if row & 1 else 0) ^ (b[1] if row & 2 else 0) for row in a)
    def labels(a): return (sum(row & 1 for row in a), a[0].bit_count(), sum((row & 1) ^ ((row >> 1) & 1) for row in a), (a[0] ^ a[1]).bit_count())
    matrices = [a for a in itertools.product(range(4), repeat=2) if rank2(a)]
    q, qt, perm = (1, 3), (3, 2), (2, 1)
    for a in matrices:
        y = labels(a); at = transpose(a)
        correct = (labels(at)[1], labels(at)[3], labels(multiply(a, q))[2], labels(multiply(qt, a))[3])
        wrong = (labels(multiply(perm, at))[1], labels(multiply(qt, at))[3], labels(multiply(a, qt))[2], labels(multiply(q, a))[3])
        wrong_expected = (sum((row >> 1) & 1 for row in a), y[0], sum((row >> 1) & 1 for row in a), a[1].bit_count())
        review.check('frozen_GF2_n2_correct_formulas', correct == (y[0], y[2], y[0], y[1]), a)
        review.check('frozen_GF2_n2_wrong_formulas', wrong == wrong_expected, a)
    return {'primary_relations': ['R1', 'R3'], 'primary_per_n_rows': 24, 'primary_aggregate_rows': 8,
            'primary_components_recomputed': recomputed, 'arithmetic_tolerance': tolerance,
            'state': 'completed_with_verification_caveat', 'formal20k_models': 8, 'independent_units': 2,
            'scientific_analysis_complete': False, 'original_numerical_verification': dict(status='failed', counts=dict(counts), failures=failed),
            'selector_checks_retained': 288, 'no_numerical_verification_replacement': True,
            'formula_check_scope': 'Six original n2 exact mathematical cases plus protocol and preserved math audit identities; no formal analysis input or new geometric statistic.'}


def self_test():
    assert fixed_mean([1., None, 3.], [6, 8, 10]) is None
    assert fixed_mean([1., 2., 3.], [6, 8, 10]) == 2.
    try: fixed_mean([1., 2.], [6, 8, 10])
    except ValueError: pass
    else: raise AssertionError('Missing scale was reweighted')
    assert number('NaN') is None and number('0') == 0.
    try: number('inf')
    except ValueError: pass
    else: raise AssertionError('Unexpected infinity accepted')
    a = dict(initial_correct=.9, initial_wrong=.95, trained_correct=.1, trained_wrong=.12,
             initial_identity=1., trained_identity=.3)
    d = decompose(a)
    assert d['Bt'] < 0 and d['Delta'] > 0 and d['dc'] < 0
    return {'status': 'passed_synthetic_only', 'cases': 7, 'no_model_or_metric_calls': True}


def run(root, layout_path=None, checksums_path=None):
    started = time.monotonic(); layout = dict(DEFAULT_LAYOUT)
    if layout_path: layout.update(json.loads(Path(layout_path).read_text()))
    review = Review(root, layout)
    identity = verify_checksums(review, checksums_path or Path(__file__).with_name('table_checksums.json'))
    pairing = check_fixed_transform(review); matrix = check_matrix(review)
    result = {'status': 'passed_saved_table_arithmetic_with_preserved_verification_caveat' if not review.errors else 'failed_saved_table_checks',
              'checks_total': sum(review.check_counts.values()), 'failure_count': len(review.errors), 'failures': review.errors,
              'check_counts': dict(review.check_counts), 'maximum_absolute_differences': dict(review.maximum_absolute_differences),
              'input_sha256': review.input_sha256, 'checksum_manifest': identity, 'pairing': pairing, 'matrix': matrix,
              'source_sha256': sha(__file__), 'elapsed_seconds': time.monotonic() - started,
              'repository_root': str(review.root), 'host_absolute_path_fallbacks': 0, 'external_dependencies': [],
              'models_constructed': 0, 'forward_calls': 0, 'training_updates': 0, 'activation_metric_recomputations': 0,
              'scope': 'Saved table equal-length aggregation, designated-original comparisons, mathematical/protocol identity and preserved numerical state only; not a fresh scientific experiment or activation-level/model reproduction.'}
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--layout', type=Path); parser.add_argument('--checksums', type=Path)
    parser.add_argument('--output', type=Path); parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    try:
        result = self_test() if args.self_test else run(args.repo_root, args.layout, args.checksums)
    except Exception as exc:
        result = {'status': 'failed_saved_table_checks', 'error': type(exc).__name__ + ': ' + str(exc),
                  'models_constructed': 0, 'forward_calls': 0, 'training_updates': 0}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: result[k] for k in ('status', 'checks_total', 'failure_count', 'error', 'elapsed_seconds') if k in result}, ensure_ascii=False))
    raise SystemExit(not str(result['status']).startswith('passed'))
