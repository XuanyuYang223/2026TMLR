"""Derived reader for frozen explicit-edge scores, without any new fitting.

Uses existing per-length AUROCs and validates their original fixed-length
aggregates. Does not modify the separate prefix-final cross-pair analysis.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import tempfile

import numpy as np

HERE = Path(__file__).resolve().parent
UNITS = ('archdev001', 'unit02', 'unit03', 'unit04', 'unit05')
NEW_UNITS = UNITS[1:]
ENCODINGS = ('canonical_cycle', 'one_line')
SUPPORTS = ('eligible', 'matched')
POINTS = ('before', 'after')
LENGTHS = tuple(range(10, 31))
SHUFFLES = tuple(range(1026, 1034))
KINDS = ('one_line_value', 'cycle_internal', 'cycle_closure', 'cycle_singleton')
STRATA = ('all', *(f'{s}_{k}' for s in ('source', 'target') for k in KINDS), 'both_internal')
CONTROLS = ('means_only', 'fit_target_identity', 'fit_target_complement',
            *(f'fit_shuffle_{s}' for s in SHUFFLES))
PRIORS = ('source_n_i_prior', 'target_n_j_prior', 'oracle_count_position_prior',
          'source_token_kind_prior', 'target_token_kind_prior')
METHODS = ('source_own', 'target_own', 'direct', *CONTROLS,
           'fit_target_inverse', 'fit_initial_inverse', 'fit_fixed_inverse', *PRIORS)
METRICS = ('per_anchor_auc', 'pooled_edge_auc')
TOLERANCE = 1e-12


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read_csv(path):
    with Path(path).open(newline='') as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for r in rows for k in r)) or ['status']
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def read_json(path):
    return json.loads(Path(path).read_text())


def finite(value):
    if value is None or str(value).strip().lower() in ('', 'nan', 'none', 'null'):
        return None
    answer = float(value)
    if not math.isfinite(answer):
        return None
    if not -TOLERANCE <= answer <= 1 + TOLERANCE:
        raise ValueError(('AUROC outside range', answer))
    return answer


def mean(values):
    return math.fsum(values) / len(values)


def subtract(a, b):
    return a - b if a is not None and b is not None else None


def index_summaries(rows):
    keys = ('unit', 'encoding', 'support', 'stratum', 'method', 'point')
    index = {}
    for row in rows:
        key = tuple(row[k] for k in keys)
        if key in index:
            raise ValueError(('Duplicate original token summary', key))
        index[key] = row
    return index


def validate_and_aggregate(detail, original):
    groups = defaultdict(dict)
    for raw in detail:
        r = dict(raw)
        key = tuple(r[k] for k in ('unit', 'encoding', 'support', 'stratum', 'method', 'point'))
        u, e, s, st, m, p = key
        n = int(r['n'])
        if (u not in UNITS or e not in ENCODINGS or s not in SUPPORTS or
                st not in STRATA or m not in METHODS or p not in POINTS or n not in LENGTHS):
            raise ValueError(('Unregistered token score', key, n))
        if n in groups[key]:
            raise ValueError(('Duplicate token length', key, n))
        for metric in METRICS:
            r[metric] = finite(r[metric])
        for field in ('anchors', 'finite_anchors', 'edges', 'undefined_anchors'):
            r[field] = int(r[field])
        if (r['anchors'] < 0 or r['edges'] < 0 or r['finite_anchors'] < 0 or
                r['finite_anchors'] + r['undefined_anchors'] != r['anchors'] or
                (r['finite_anchors'] == 0) != (r['per_anchor_auc'] is None)):
            raise ValueError(('Invalid token coverage', key, n, r))
        groups[key][n] = r
    expected = {(u, e, s, st, m, p) for u in UNITS for e in ENCODINGS for s in SUPPORTS
                for st in STRATA for m in METHODS for p in POINTS}
    if set(groups) != expected:
        raise ValueError(('Token inventory differs', len(expected - set(groups)), len(set(groups) - expected)))
    prior = index_summaries(original)
    if set(prior) != expected:
        raise ValueError('Original summary inventory differs')
    rows, coverage, errors = [], [], []
    for key in sorted(expected):
        g = groups[key]
        if set(g) != set(LENGTHS):
            raise ValueError(('Token lengths missing', key, sorted(set(LENGTHS) - set(g))))
        row = dict(zip(('unit', 'encoding', 'support', 'stratum', 'method', 'point'), key))
        row['historical'] = key[0] == UNITS[0]
        for metric in METRICS:
            values = [g[n][metric] for n in LENGTHS]
            defined = [v for v in values if v is not None]
            value = mean(defined) if len(defined) == len(LENGTHS) else None
            old = finite(prior[key][metric])
            if ((value is None) != (old is None) or
                    (value is not None and abs(value - old) > TOLERANCE) or
                    int(prior[key][metric + '_finite_lengths']) != len(defined)):
                raise ValueError(('Original token mean mismatch', key, metric, value, old))
            row[metric] = value
            row[metric + '_finite_lengths'] = len(defined)
            row[metric + '_undefined_lengths'] = ';'.join(str(n) for n in LENGTHS if g[n][metric] is None)
            errors.append(dict(**{k: v for k, v in row.items() if k in ('unit','encoding','support','stratum','method','point')},
                               metric=metric, absolute_error=abs(value - old) if value is not None else None,
                               undefined_consistent=value is None, finite_lengths=len(defined)))
        row['anchors_sum'] = sum(r['anchors'] for r in g.values())
        row['finite_anchors_sum'] = sum(r['finite_anchors'] for r in g.values())
        row['edges_sum'] = sum(r['edges'] for r in g.values())
        rows.append(row)
        coverage.extend(dict(r) for r in g.values())
    return rows, coverage, errors


def wide_table(rows):
    ix = {tuple(r[k] for k in ('unit', 'encoding', 'support', 'stratum', 'point', 'method')): r for r in rows}
    output = []
    for u in UNITS:
        for e in ENCODINGS:
            for s in SUPPORTS:
                for st in STRATA:
                    for p in POINTS:
                        row = dict(unit=u, historical=u == UNITS[0], encoding=e, support=s, stratum=st, point=p)
                        selected = {m: ix[u,e,s,st,p,m] for m in METHODS}
                        for metric in METRICS:
                            suffix = '' if metric == 'per_anchor_auc' else '_pooled'
                            get = lambda m: selected[m][metric]
                            row.update({m + suffix: get(m) for m in METHODS})
                            control_values = {m: get(m) for m in CONTROLS}
                            complete = all(v is not None for v in control_values.values())
                            best = max(control_values.values()) if complete else None
                            row['best_common_control' + suffix] = best
                            row['best_common_control_names' + suffix] = ';'.join(m for m,v in control_values.items() if v == best) if complete else ''
                            row['correct_minus_best_common' + suffix] = subtract(get('fit_target_inverse'), best)
                            row['correct_minus_direct' + suffix] = subtract(get('fit_target_inverse'), get('direct'))
                            for prior in PRIORS:
                                row['correct_minus_' + prior + suffix] = subtract(get('fit_target_inverse'), get(prior))
                        reference = selected['fit_target_inverse']
                        row['correct_finite_lengths'] = reference['per_anchor_auc_finite_lengths']
                        row['correct_undefined_lengths'] = reference['per_anchor_auc_undefined_lengths']
                        row['correct_finite_anchor_length_pairs'] = reference['finite_anchors_sum']
                        row['correct_edges_sum'] = reference['edges_sum']
                        row['control_rule'] = '11 scalar controls individually averaged fixed1/21 before maximum; priors, direct, own, initial-target and fixed-target excluded'
                        output.append(row)
    return output


def before_after(rows):
    ix = {tuple(r[k] for k in ('unit','encoding','support','stratum','point','method')): r for r in rows}
    output = []
    for u in UNITS:
        for e in ENCODINGS:
            for s in SUPPORTS:
                for st in STRATA:
                    for m in METHODS:
                        a, b = ix[u,e,s,st,'after',m], ix[u,e,s,st,'before',m]
                        output.append(dict(unit=u, historical=u == UNITS[0], encoding=e, support=s, stratum=st, method=m,
                                           before=b['per_anchor_auc'], after=a['per_anchor_auc'],
                                           after_minus_before=subtract(a['per_anchor_auc'], b['per_anchor_auc']),
                                           caveat='Same scalar AFTER-fitted probe, scaler and AFTER means applied BEFORE; distribution-shift descriptive comparison'))
    return output


def cohort(rows, group_keys, metrics):
    groups = defaultdict(dict)
    for r in rows:
        groups[tuple(r[k] for k in group_keys)][r['unit']] = r
    output = []
    for key, group in sorted(groups.items()):
        for metric in metrics:
            chosen = [(u, group.get(u, {}).get(metric)) for u in NEW_UNITS]
            values = [v for _,v in chosen if v is not None]
            complete = len(values) == len(NEW_UNITS)
            output.append(dict(zip(group_keys, key), metric=metric, registered_units=4, finite_units=len(values),
                               mean=mean(values) if complete else None,
                               sample_sd=statistics.stdev(values) if complete else None,
                               minimum=min(values) if complete else None, maximum=max(values) if complete else None,
                               positive_units=sum(v > 0 for v in values), negative_units=sum(v < 0 for v in values),
                               per_unit_values=';'.join(u + ':' + str(v) for u,v in chosen),
                               undefined_reason='' if complete else 'One or more units undefined; no unit or length reweighting'))
    return output


def fmt(v):
    return 'undefined' if v is None else f'{v:.4f}'


def render_report(wide, stats, out):
    lines = ['# Explicit-edge token diagnostic: readout and control comparison', '',
        'This secondary diagnostic is separate from the completed prefix-final, vector-valued alignment experiment. It uses one shared scalar position classifier per length, fitted on source AFTER states and source labels only. All base models remain frozen. The historical unit is an engineering reference; statistics below use four new units.', '',
        'The primary score is per-anchor position AUROC, averaged within length and then across all 21 registered lengths with weight 1/21 each. Full eligible support is primary; matched support is secondary. Pooled-edge AUROC is supplementary. Every map scores the same paired inverse-target states; initial-target and fixed-point-target controls use their own states. Controls and priors are reported individually.', '',
        'The same AFTER-fitted probe, scaler and centering are applied to BEFORE states. Before/after differences describe a distribution shift and local serialization stage; they do not identify the earliest point where the mathematical relation was logically inferable.', '',
        '## Full-support per-unit comparison, all edges', '',
        'Each common-control maximum includes means-only, identity, complement and each of eight shuffled-pair maps, after all-length aggregation. Priors, own readouts, direct transfer, initial-target and fixed-target controls remain separate.', '',
        '|Unit|Encoding|Point|Source own|Correct map|Best common|Margin|Source n/i prior|Target n/j prior|Source kind prior|Target kind prior|Initial target|Fixed target|Target own|',
        '|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    fields = ['source_own','fit_target_inverse','best_common_control','correct_minus_best_common',
              'source_n_i_prior','target_n_j_prior','source_token_kind_prior','target_token_kind_prior',
              'fit_initial_inverse','fit_fixed_inverse','target_own']
    for r in wide:
        if r['support'] == 'eligible' and r['stratum'] == 'all':
            lines.append('|' + '|'.join([r['unit'],r['encoding'],r['point'],*(fmt(r[k]) for k in fields)]) + '|')
    lines += ['', '## Four-new-unit means and variability', '',
              '|Encoding|Point|Metric|Mean|Sample SD|Min|Max|', '|---|---|---|---:|---:|---:|---:|']
    for r in stats:
        if r['support'] == 'eligible' and r['stratum'] == 'all' and r['metric'] in fields:
            lines.append('|' + '|'.join([r['encoding'],r['point'],r['metric'],*(fmt(r[k]) for k in ('mean','sample_sd','minimum','maximum'))]) + '|')
    lines += ['', '## Coverage and interpretation', '',
        'All source/target internal, closure, singleton and both-internal strata are retained without selecting convenient lengths or dropping undefined cells. One-line and cycle-only kinds naturally have no edges in the other encoding. In canonical cycle notation, source closures are all negative, inverse-target closures are all paired positives, and singletons are negative. Their within-stratum AUROC can therefore be structurally undefined. Such entries are not zero accuracy or evidence of absent information.', '',
        'Positional and token-kind priors can rank labels without recovering an input-specific neural computation. A high correct-map AUROC is therefore read together with these controls, own-model readouts and initial/fixed-target controls. Differences between scores are comparisons under this frozen diagnostic, not an algebraic separation of independent information sources or proof that the counting network naturally uses this computation.', '',
        'Raw per-edge scores preserve anchor IDs, source i, paired target j, before/after token positions, edge kinds, labels and support masks in the external stage-two score archives listed by the per-unit completion receipts. This reader uses saved AUROCs only; it does not fit, forward, select methods, select lengths, or change the main cross-pair result.', '',
        'Tables: `token_four_unit_table.csv` (new-unit all-edge primary/sensitivity rows), `token_historical_reference.csv`, `token_per_unit_all_strata.csv`, `token_four_unit_stats.csv`, `token_before_after_by_unit.csv`, `token_before_after_stats.csv`, `token_coverage_by_length.csv`, `token_aggregation_checks.csv`.', '']
    (HERE / 'token_report.md').write_text('\n'.join(lines))
    (Path(out) / 'report.md').write_text('\n'.join(lines))


def derive(detail, original, out, *, write_reader_report=True):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rows, coverage, checks = validate_and_aggregate(detail, original)
    wide = wide_table(rows)
    numeric = [m for m in METHODS] + ['best_common_control','correct_minus_best_common','correct_minus_direct',
                                      *(f'correct_minus_{p}' for p in PRIORS)]
    stats = cohort(wide, ('encoding','support','stratum','point'), numeric)
    contrasts = before_after(rows)
    contrast_stats = cohort(contrasts, ('encoding','support','stratum','method'), ('before','after','after_minus_before'))
    tables = [('token_four_unit_table', [r for r in wide if not r['historical'] and r['stratum'] == 'all']),
              ('token_historical_reference', [r for r in wide if r['historical'] and r['stratum'] == 'all']),
              ('token_per_unit_all_strata', wide), ('token_four_unit_stats', stats),
              ('token_before_after_by_unit', contrasts), ('token_before_after_stats', contrast_stats),
              ('token_coverage_by_length', coverage), ('token_aggregation_checks', checks)]
    for name, values in tables:
        write_csv(out / (name + '.csv'), values)
    if write_reader_report:
        render_report(wide, stats, out)
        # User-facing requested top-level tables are derived copies, not edits
        # to any stage-two frozen or original result table.
        write_csv(HERE / 'token_four_unit_table.csv', tables[0][1])
        write_csv(HERE / 'token_four_unit_stats.csv', stats)
    finite_errors = [r['absolute_error'] for r in checks if r['absolute_error'] is not None]
    return dict(status='completed_saved_token_reader', units=list(UNITS), new_units=list(NEW_UNITS),
                length_rows=len(detail), summary_rows=len(rows), wide_rows=len(wide), checks=len(checks),
                maximum_aggregation_error=max(finite_errors, default=None), no_new_fits=True, no_neural_forward=True,
                common_max_after_equal_length_averaging=True, undefined_cells_not_reweighted=True,
                main_prefix_final_result_unchanged=True)


def synthetic_checks():
    detail, original = [], []
    for u in UNITS:
        for e in ENCODINGS:
            for s in SUPPORTS:
                for st in STRATA:
                    for m in METHODS:
                        for p in POINTS:
                            values = []
                            for i,n in enumerate(LENGTHS):
                                value = .75 if m == 'fit_target_inverse' else .55
                                if m == 'fit_target_identity':
                                    value = .9 if i < 10 else .4
                                if m == 'fit_target_complement':
                                    value = .4 if i < 10 else .9
                                if p == 'before':
                                    value -= .1
                                # A structurally single-class stratum persists
                                # as undefined instead of being filled with zero.
                                if st == 'source_cycle_closure':
                                    value = None
                                values.append(value)
                                detail.append(dict(unit=u,historical=u == UNITS[0],encoding=e,support=s,stratum=st,
                                                   method=m,point=p,n=n,anchors=20,finite_anchors=0 if value is None else 20,
                                                   undefined_anchors=20 if value is None else 0,edges=20*n,
                                                   per_anchor_auc=value,pooled_edge_auc=value))
                            v = mean(values) if all(x is not None for x in values) else None
                            count = sum(x is not None for x in values)
                            original.append(dict(unit=u,historical=u == UNITS[0],encoding=e,support=s,stratum=st,
                                                 method=m,point=p,per_anchor_auc=v,pooled_edge_auc=v,
                                                 per_anchor_auc_finite_lengths=count,pooled_edge_auc_finite_lengths=count))
    with tempfile.TemporaryDirectory(prefix='token_reader_synthetic_') as temporary:
        result = derive(detail, original, temporary, write_reader_report=False)
        wide = read_csv(Path(temporary) / 'token_per_unit_all_strata.csv')
        r = next(r for r in wide if r['stratum'] == 'all' and r['point'] == 'after')
        expected = (10*.4 + 11*.9)/21
        assert abs(float(r['best_common_control']) - expected) < TOLERANCE
        assert float(r['best_common_control']) < .9
        missing = next(r for r in wide if r['stratum'] == 'source_cycle_closure')
        assert missing['fit_target_inverse'] == '' and missing['correct_minus_best_common'] == ''
        deltas = read_csv(Path(temporary) / 'token_before_after_by_unit.csv')
        assert all(abs(float(r['after_minus_before']) - .1) < TOLERANCE
                   for r in deltas if r['stratum'] != 'source_cycle_closure')
        stats = read_csv(Path(temporary) / 'token_four_unit_stats.csv')
        assert all(int(r['registered_units']) == 4 for r in stats)
        assert all(r['mean'] == '' for r in stats if r['stratum'] == 'source_cycle_closure')
    return dict(status='passed',synthetic_only=True,length_rows=len(detail),summary_rows=len(original),
                checks=['complete registered inventory','fixed21 mean','max AFTER length mean',
                        'structural undefined retained','no undefined-length reweighting',
                        'four-new-unit statistics','paired before/after arithmetic'],
                source_sha256=sha(__file__))


def freeze_reader():
    path = HERE / 'token_reader_freeze.json'
    value = dict(source_sha256=sha(__file__), protocol_sha256=sha(HERE / 'protocol.json'),
                 token_freeze_sha256=sha(HERE / 'token_results/freeze.json'),
                 created_utc=datetime.now(timezone.utc).isoformat(),
                 derived_only=True, source_score_values_not_read_during_development=True,
                 synthetic_verification=read_json(HERE / 'token_reader_synthetic_verification.json'))
    if path.exists():
        old = read_json(path)
        assert all(old[k] == value[k] for k in ('source_sha256','protocol_sha256','token_freeze_sha256'))
        return old
    write_json(path,value)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--synthetic', action='store_true')
    parser.add_argument('--freeze', action='store_true')
    parser.add_argument('--out', type=Path, default=HERE / 'token_reader')
    args = parser.parse_args()
    if args.synthetic:
        receipt = synthetic_checks()
        write_json(HERE / 'token_reader_synthetic_verification.json',receipt)
        print(json.dumps(receipt,indent=2))
        return
    if args.freeze:
        print(json.dumps(freeze_reader(),indent=2))
        return
    frozen = read_json(HERE / 'token_reader_freeze.json')
    assert sha(__file__) == frozen['source_sha256']
    assert sha(HERE/'protocol.json') == frozen['protocol_sha256']
    assert sha(HERE/'token_results/freeze.json') == frozen['token_freeze_sha256']
    detail, original, sources = [], [], []
    for unit in UNITS:
        folder = HERE / 'token_results' / unit
        receipt = read_json(folder / 'completion.json')
        assert receipt['status'] == 'complete'
        assert receipt['token_freeze_sha256'] == frozen['token_freeze_sha256']
        for name in ('by_length.csv','summary.csv'):
            path = folder / name
            digest = sha(path)
            assert receipt['result_sha256'][str(path)] == digest
            sources.append(dict(path=str(path),sha256=digest))
            (detail if name == 'by_length.csv' else original).extend(read_csv(path))
    result = derive(detail, original, args.out)
    result.update(completed_utc=datetime.now(timezone.utc).isoformat(),reader_source_sha256=sha(__file__),
                  reader_freeze_sha256=sha(HERE/'token_reader_freeze.json'),source_tables=sources)
    write_json(args.out / 'completion.json',result)
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
