"""Read-only CSV arithmetic audit; no activation, model or metric evaluation."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import time

DIAG = Path(__file__).resolve().parent
FIELDS = tuple(f'{phase}_{role}' for phase in ('initial', 'trained')
               for role in ('correct', 'wrong', 'identity')) + ('B0', 'Bt', 'dc', 'du', 'di', 'Delta')
ERRORS = tuple(f'{phase}_{role}_error' for phase in ('initial', 'trained')
               for role in ('correct', 'wrong', 'identity'))
KEY = ('unit', 'relation', 'control', 'branch', 'metric')
METRICS = ('linear_cka', 'knn_overlap', 'procrustes')
CATEGORIES = ('designated', 'other_known_exact', 'not_designated')
ARITHMETIC_TOLERANCE = 1e-12
EXISTING_SCORE_TOLERANCE = 1e-8


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_rows(path):
    with Path(path).open(newline='') as stream:
        return list(csv.DictReader(stream))


def number(value):
    if value is None or value == '':
        return None
    value = float(value)
    assert math.isfinite(value), 'Non-finite printed numbers are not silently treated as missing'
    return value


def fixed_mean(values):
    assert len(values) == 3
    return math.fsum(values) / 3 if all(v is not None for v in values) else None


def sub(a, b):
    return a-b if a is not None and b is not None else None


def key(row, names=KEY):
    return tuple(row[n] for n in names)


def sign(value):
    return 'undefined' if value is None else 'positive' if value > 0 else 'negative' if value < 0 else 'exact_zero'


def audit(diag=DIAG):
    begun = time.monotonic()
    names = ['geometry_per_n.csv', 'geometry_summary.csv', 'pair_per_n.csv', 'pair_summary.csv',
             'candidate_manifest.csv', 'category_summary.csv', 'designated_category_comparisons.csv',
             'common_support_training_comparisons.csv', 'designated_rank.csv', 'selection_masks.csv',
             'W_actual_per_n.csv', 'W_actual_summary.csv', 'W_actual_per_anchor.csv', 'source_learning.csv',
             'protocol.json', 'freeze_manifest.json', 'analysis_completion.json',
             'actual_numerical_verification.json', 'candidate_classification_audit.json',
             'protection_start.json', 'protection_end.json']
    before = {str(diag / n): sha(diag / n) for n in names}
    data = {n: read_rows(diag / n) for n in names if n.endswith('.csv')}
    checks = Counter()
    maximum = defaultdict(float)
    failures = []

    def check(kind, condition, context=None):
        checks[kind] += 1
        if not condition:
            failures.append(dict(check=kind, context=context))

    def equal(kind, actual, expected, context=None, tolerance=ARITHMETIC_TOLERANCE):
        if actual is None or expected is None:
            check(kind, actual is None and expected is None, context)
            return
        diff = abs(actual-expected)
        maximum[kind] = max(maximum[kind], diff)
        check(kind, diff <= tolerance, dict(context=context, actual=actual, expected=expected, difference=diff))

    expected_counts = {'geometry_per_n.csv':432, 'geometry_summary.csv':144, 'pair_per_n.csv':144,
                       'pair_summary.csv':48, 'candidate_manifest.csv':24, 'category_summary.csv':96,
                       'designated_category_comparisons.csv':64, 'common_support_training_comparisons.csv':48,
                       'designated_rank.csv':32, 'selection_masks.csv':96, 'W_actual_per_n.csv':48,
                       'W_actual_summary.csv':16, 'W_actual_per_anchor.csv':12288, 'source_learning.csv':816}
    for filename, count in expected_counts.items():
        check('coverage_rows', len(data[filename]) == count, filename)
    completion = json.loads((diag/'analysis_completion.json').read_text())
    for path, digest in completion['output_sha256'].items():
        check('completed_analysis_table_identity', before[str(Path(path))] == digest, path)

    # Independently derive all six-score decompositions without zero filling.
    for filename in ('geometry_per_n.csv', 'geometry_summary.csv'):
        for row in data[filename]:
            v = {f:number(row[f]) for f in FIELDS}
            context = dict(zip(KEY, key(row)), n=row.get('n', 'equal_n'))
            expect = dict(B0=sub(v['initial_correct'], v['initial_wrong']),
                          Bt=sub(v['trained_correct'], v['trained_wrong']),
                          dc=sub(v['trained_correct'], v['initial_correct']),
                          du=sub(v['trained_wrong'], v['initial_wrong']),
                          di=sub(v['trained_identity'], v['initial_identity']))
            expect['Delta'] = sub(expect['Bt'], expect['B0'])
            for f, expected in expect.items():
                equal('six_score_decomposition', v[f], expected, dict(context, field=f))
            equal('Delta_equals_delta_correct_minus_delta_wrong', v['Delta'], sub(v['dc'], v['du']), context)
            if row['branch'] == 'gold_onehot':
                for f in ('initial_correct', 'initial_wrong', 'initial_identity', 'B0', 'dc', 'du', 'di', 'Delta'):
                    check('static_gold_initial_and_training_increments_undefined', v[f] is None, dict(context, field=f))
            if row['metric'] == 'procrustes':
                for score, error in zip(FIELDS[:6], ERRORS):
                    e = number(row[error])
                    equal('procrustes_error_sign', v[score], -e if e is not None else None, dict(context, field=score))

    groups = defaultdict(list)
    for row in data['geometry_per_n.csv']:
        groups[key(row)].append(row)
    for summary in data['geometry_summary.csv']:
        rr = groups[key(summary)]
        check('geometry_complete_fixed_n_grid', sorted(int(r['n']) for r in rr) == [6,8,10] and summary['weight_each']=='1/3', key(summary))
        for field in FIELDS + (ERRORS if summary['metric']=='procrustes' else ()):
            values = [number(r[field]) for r in rr]
            equal('geometry_equal_n_aggregate', number(summary[field]), fixed_mean(values), dict(key=key(summary), field=field))
            if field in FIELDS:
                missing = ','.join(r['n'] for r in rr if number(r[field]) is None)
                check('geometry_component_missing_n_record', summary[field+'_undefined_n']==missing, dict(key=key(summary), field=field))

    manifest = data['candidate_manifest.csv']
    class_counts = Counter(r['category'] for r in manifest)
    check('candidate_classification_8_0_16', {c:class_counts[c] for c in CATEGORIES} == dict(designated=8, other_known_exact=0, not_designated=16))
    check('candidate_ids_unique', len({r['pair_id'] for r in manifest})==24)
    for row in manifest:
        check('candidate_designated_identity', (row['category']=='designated') == (row['right_task']==row['designated_right_task']), row['pair_id'])
    pairgroups = defaultdict(list)
    for row in data['pair_per_n.csv']:
        pairgroups[row['pair_id'], row['control']].append(row)
        equal('pair_score_change', number(row['training_score_change']), sub(number(row['fixed_transform_score_trained']), number(row['fixed_transform_score_initial'])), row['pair_id'])
    for summary in data['pair_summary.csv']:
        rr=pairgroups[summary['pair_id'], summary['control']]
        check('pair_complete_fixed_n_grid', sorted(int(r['n']) for r in rr)==[6,8,10], summary['pair_id'])
        for phase in ('initial','trained'):
            field='fixed_transform_score_'+phase; vv=[number(r[field]) for r in rr]
            equal('pair_equal_n_aggregate', number(summary[field]), fixed_mean(vv), dict(pair=summary['pair_id'], control=summary['control'], phase=phase))
            check('pair_finite_n_counts', int(summary[phase+'_finite_n'])==sum(v is not None for v in vv), summary['pair_id'])
        equal('pair_summary_score_change', number(summary['training_score_change']), sub(number(summary['fixed_transform_score_trained']), number(summary['fixed_transform_score_initial'])), summary['pair_id'])

    geometry = {key(r)+((r['n'],) if 'n' in r else ()):r for r in data['geometry_per_n.csv']+data['geometry_summary.csv']}
    designated_matches=[]
    for filename in ('pair_per_n.csv','pair_summary.csv'):
        for row in data[filename]:
            if row['category']!='designated':continue
            target_key=(row['unit'],row['relation'],row['control'],'prefix_final','linear_cka')+((row['n'],) if 'n' in row else ())
            original=geometry[target_key]
            for phase in ('initial','trained'):
                a,b=number(row['fixed_transform_score_'+phase]), number(original[phase+'_correct'])
                equal('designated_fixed_T_equals_original_correct_CKA',a,b,dict(pair=row['pair_id'],control=row['control'],n=row.get('n','equal_n'),phase=phase),EXISTING_SCORE_TOLERANCE)
                designated_matches.append(dict(unit=row['unit'], relation=row['relation'],control=row['control'], n=row.get('n','equal_n'),phase=phase,difference=None if a is None or b is None else abs(a-b),both_undefined=a is None and b is None))

    # Exact initial-state raw equality is checked only within a fixed f/T,
    # unit and scale. Candidate-specific answer masks are deliberately excluded.
    initial_raw=[]
    for filename in ('pair_per_n.csv','pair_summary.csv'):
        gg=defaultdict(list)
        for row in data[filename]:
            if row['control']=='raw_within_n':gg[row['unit'],row['relation'],row.get('n','equal_n')].append(row)
        for k,rr in gg.items():
            scores=[number(r['fixed_transform_score_initial']) for r in rr]
            exact = len(rr)==3 and scores[0] is not None and all(v==scores[0] for v in scores)
            check('raw_initial_three_right_tasks_exact_equal',exact,k)
            initial_raw.append(dict(unit=k[0],relation=k[1],n=k[2],right_tasks=[r['right_task'] for r in rr],scores=scores,exact_equal=exact))

    ps=defaultdict(list)
    for row in data['pair_summary.csv']:ps[row['unit'],row['relation'],row['control']].append(row)
    for row in data['category_summary.csv']:
        rr=[r for r in ps[row['unit'],row['relation'],row['control']] if r['category']==row['category']]
        field='fixed_transform_score_'+row['phase']; finite=[number(r[field]) for r in rr if number(r[field]) is not None]
        check('category_total_finite_counts',int(row['total_count'])==len(rr) and int(row['finite_count'])==len(finite) and int(row['undefined_count'])==len(rr)-len(finite),key(row,('unit','relation','control','phase','category')))
        equal('category_finite_mean',number(row['mean']),math.fsum(finite)/len(finite) if finite else None)
        equal('category_finite_median',number(row['median']),statistics.median(finite) if finite else None)
        if row['category']=='other_known_exact':
            check('empty_other_known_exact_retained_undefined',row['status']=='empty_fixed_category' and number(row['mean']) is None and number(row['median']) is None)

    directions=[]
    summarygroups=defaultdict(dict)
    for row in data['geometry_summary.csv']:
        summarygroups[key(row,('unit','relation','control','branch'))][row['metric']]=row
    for k,rr in sorted(summarygroups.items()):
        check('all_three_fixed_metrics_present',set(rr)==set(METRICS),k)
        for field in ('Bt','Delta','dc','du'):
            values={m:number(rr[m][field]) for m in METRICS}
            signs={m:sign(v) for m,v in values.items()}
            defined=[s for s in signs.values() if s!='undefined']
            direction_agrees=len(defined)==3 and len(set(defined))==1
            directions.append(dict(unit=k[0],relation=k[1],control=k[2],branch=k[3],component=field,
                values=values,exact_stored_value_signs=signs,all_three_defined=len(defined)==3,
                all_three_direction_agree=direction_agrees,
                inconsistent_among_defined=len(set(defined))>1,
                near_zero_at_existing_1e8_tolerance={m:v is not None and abs(v)<=EXISTING_SCORE_TOLERANCE for m,v in values.items()}))
    after={str(diag/n):sha(diag/n) for n in names}
    check('all_input_bytes_before_after_unchanged',before==after)
    numerical=json.loads((diag/'actual_numerical_verification.json').read_text())
    protected=json.loads((diag/'protection_end.json').read_text())
    result=dict(status='passed_table_arithmetic_only' if not failures else 'failed_table_arithmetic',
        checks_total=sum(checks.values()),check_counts=dict(checks),failure_count=len(failures),failures=failures,
        maximum_absolute_differences=dict(maximum),coverage_rows={n:len(data[n]) for n in expected_counts},
        candidate_categories={c:class_counts[c] for c in CATEGORIES},designated_original_score_tolerance=EXISTING_SCORE_TOLERANCE,
        arithmetic_tolerance=ARITHMETIC_TOLERANCE,designated_score_checks=designated_matches,
        raw_initial_equal_right_task_checks=initial_raw,metric_direction_cells=directions,
        metric_direction_inconsistencies=[r for r in directions if r['inconsistent_among_defined']],
        main_hidden_answer_control=[r for r in data['geometry_summary.csv'] if r['branch']=='prefix_final' and r['metric']=='linear_cka' and r['control']=='answer_strata'],
        input_sha256_before=before,input_sha256_after=after,all_input_bytes_unchanged=before==after,
        protected_artifact_closing_audit_status=protected['status'],
        existing_independent_numerical_verification_status=numerical['status'],
        existing_independent_numerical_verification_failure_count=numerical.get('failure_count'),
        existing_numerical_verification_preserved=True,scope='Existing CSV arithmetic, exact classification/coverage and SHA review only. No activation-level metric recomputation.',
        interpretations=['Direction cells are coupled descriptions, not independent repeats or significance tests.',
          'The du column is delta_wrong; dc is delta_correct. Delta=dc-du is distinct from fixed-T training_score_change.',
          'Static gold has no initial values or training increments. Undefined fixed-n components are never filled or reweighted.',
          'Exact raw initial equalities cannot be required for candidate-specific answer-strata masks.',
          'A passed table-arithmetic audit does not override the preserved failed independent numerical verification.'],
        model_or_forward_calls=0,training_updates=0,Torch_imported=False,elapsed_seconds=time.monotonic()-begun,
        audit_source_sha256=sha(__file__))
    destination=diag/'table_arithmetic_audit.json'
    destination.write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',action='store_true',required=True)
    args=parser.parse_args()
    r=audit()
    print(json.dumps({k:r[k] for k in ('status','checks_total','failure_count','maximum_absolute_differences','candidate_categories','existing_independent_numerical_verification_status')},ensure_ascii=False))
    raise SystemExit(bool(r['failure_count']))
