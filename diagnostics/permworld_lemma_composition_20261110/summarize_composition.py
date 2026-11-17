"""Saved-answer scoring and descriptive unit-level composition summaries.

This helper never imports a model, hidden-state reader, or fitted probe.
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
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from experiments.permworld_lemma_composition_20261110.metrics import (
    TASKS, METRICS, COHORT_METRICS, finite, fixed_mean, score_unit_arrays,
    sample_metrics, one_length_row,
)

HISTORICAL = 'archdev001'
TASK_LABELS = {'longest': 'Longest monotone interval',
               'fixed_longest': 'Longest interval containing a fixed point',
               'turns_before_fixed': 'Turns strictly before the first fixed point'}


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):return json.loads(Path(path).read_text())


def safe(value):
    if isinstance(value, dict):return {k: safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):return [safe(v) for v in value]
    if isinstance(value, np.generic):value = value.item()
    return None if isinstance(value, float) and not math.isfinite(value) else value


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(safe(value), indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def write_csv(path, rows):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for row in rows for k in row)) or ['status']
    temp = path.with_suffix(path.suffix + '.tmp')
    with temp.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(safe(rows))
    temp.replace(path)


def units(config):
    return [u['unit'] if isinstance(u, dict) else str(u) for u in config['units']]


def applicable(task, metric):
    if metric in ('count_mae', 'normalized_count_error'):
        return task == 'turns_before_fixed'
    if metric in ('start_mae', 'end_mae', 'endpoint_mae', 'length_mae',
                  'normalized_endpoint_error', 'normalized_length_error'):
        return task != 'turns_before_fixed'
    return True


def variant_family(name):
    if name == 'original':return 'original'
    if name.startswith('turn_shuffle_'):return 'turn_shuffle_mean'
    if name.startswith('fixed_shuffle_'):return 'fixed_shuffle_mean'
    raise ValueError('Unregistered variant: ' + name)


def condition_summaries(unit_rows, shuffle_repetitions=8):
    keys = ('unit', 'route', 'task', 'support', 'stratum')
    groups = defaultdict(list)
    for row in unit_rows:
        groups[tuple(row[k] for k in keys) + (variant_family(row['variant']),)].append(row)
    output = []
    for key, group in groups.items():
        family = key[-1]
        expected = 1 if family == 'original' else shuffle_repetitions
        if len(group) != expected or len({r['variant'] for r in group}) != expected:
            raise ValueError('Missing or duplicate registered mask-shuffle realization')
        if len({int(r['n']) for r in group}) != 1:
            raise ValueError('A mask shuffle changed the scoring support')
        row = dict(zip(keys, key[:-1]), variant_family=family,
                   shuffle_seed_count=0 if family == 'original' else expected,
                   n=int(group[0]['n']), length_count=int(group[0]['length_count']),
                   aggregation='Equal registered lengths, then equal shuffle realizations WITHIN unit')
        for metric in METRICS:
            values = [r[metric] for r in group]
            row[metric] = fixed_mean(values)
            row[metric + '_finite_realizations'] = sum(finite(v) is not None for v in values)
        values = [finite(r['exact_accuracy']) for r in group]
        row['seed_accuracy_minimum'] = min(values) if all(v is not None for v in values) else None
        row['seed_accuracy_maximum'] = max(values) if all(v is not None for v in values) else None
        row['individual_accuracy_values'] = ';'.join(r['variant'] + ':' + ('undefined' if finite(r['exact_accuracy']) is None else repr(float(r['exact_accuracy']))) for r in group)
        output.append(row)
    return output


def descriptive(values, unit_names):
    vals = [finite(v) for v in values]
    good = [v for v in vals if v is not None]
    complete = len(good) == len(unit_names) == len(vals)
    return dict(registered_units=len(unit_names), finite_units=len(good),
                mean=statistics.mean(good) if complete and good else None,
                median=statistics.median(good) if complete and good else None,
                sample_sd=statistics.stdev(good) if complete and len(good) > 1 else None,
                minimum=min(good) if complete and good else None,
                maximum=max(good) if complete and good else None,
                positive_units=sum(v > 0 for v in good), negative_units=sum(v < 0 for v in good),
                zero_units=sum(v == 0 for v in good),
                per_unit_values=';'.join(u + ':' + ('undefined' if v is None else repr(v)) for u, v in zip(unit_names, vals)),
                undefined_reason='' if complete else 'Undefined required unit/length/realization; no silent reweighting')


def cohorts(config):
    names = units(config)
    historical = config.get('historical_unit', HISTORICAL)
    return [('new_units', [u for u in names if u != historical]), ('all_units', names)]


def cohort_summaries(config, conditions):
    keys = ('route', 'variant_family', 'task', 'support', 'stratum')
    index = defaultdict(dict)
    for r in conditions:index[tuple(r[k] for k in keys)][r['unit']] = r
    output = []
    for cohort, names in cohorts(config):
        for key, by_unit in index.items():
            task = key[2]
            for metric in COHORT_METRICS:
                if not applicable(task, metric):continue
                output.append(dict(zip(keys, key), cohort=cohort, metric=metric,
                    **descriptive([by_unit.get(u, {}).get(metric) for u in names], names)))
    return output


def paired_effects(config, conditions):
    index = {(r['unit'], r['route'], r['variant_family'], r['task'], r['support'], r['stratum']): r for r in conditions}
    individual = []
    for key, aligned in index.items():
        unit, route, variant, task, support, stratum = key
        if route != 'aligned_complement':continue
        baseline = index.get((unit, 'same_model_twice', variant, task, support, stratum))
        if baseline is None:raise ValueError('Missing prescribed same-model-twice baseline')
        a, b = finite(aligned['exact_accuracy']), finite(baseline['exact_accuracy'])
        individual.append(dict(unit=unit, variant_family=variant, task=task, support=support, stratum=stratum,
            aligned_exact_accuracy=a, same_model_exact_accuracy=b,
            aligned_minus_same_model=a-b if a is not None and b is not None else None,
            n=aligned['n']))
    keys = ('variant_family', 'task', 'support', 'stratum')
    groups = defaultdict(dict)
    for r in individual:groups[tuple(r[k] for k in keys)][r['unit']] = r
    summary = []
    for cohort, names in cohorts(config):
        for key, group in groups.items():
            summary.append(dict(zip(keys, key), cohort=cohort,
                **descriptive([group.get(u, {}).get('aligned_minus_same_model') for u in names], names)))
    return individual, summary


def shuffle_effects(config, conditions):
    index = {(r['unit'], r['route'], r['variant_family'], r['task'], r['support'], r['stratum']): r for r in conditions}
    individual = []
    for key, shuffled in index.items():
        unit, route, variant, task, support, stratum = key
        if variant == 'original':continue
        original = index.get((unit, route, 'original', task, support, stratum))
        if original is None:raise ValueError('Missing original masks for shuffle comparison')
        a, b = finite(original['exact_accuracy']), finite(shuffled['exact_accuracy'])
        individual.append(dict(unit=unit, route=route, variant_family=variant, task=task, support=support, stratum=stratum,
            original_exact_accuracy=a, shuffled_mean_exact_accuracy=b,
            accuracy_drop_original_minus_shuffle=a-b if a is not None and b is not None else None,
            shuffle_seed_count=shuffled['shuffle_seed_count'],
            shuffled_seed_minimum=shuffled['seed_accuracy_minimum'],shuffled_seed_maximum=shuffled['seed_accuracy_maximum'],n=shuffled['n']))
    keys = ('route', 'variant_family', 'task', 'support', 'stratum')
    groups = defaultdict(dict)
    for r in individual:groups[tuple(r[k] for k in keys)][r['unit']] = r
    summary = []
    for cohort, names in cohorts(config):
        for key, group in groups.items():
            summary.append(dict(zip(keys, key), cohort=cohort,
                **descriptive([group.get(u, {}).get('accuracy_drop_original_minus_shuffle') for u in names], names)))
    return individual, summary


def make_plots(config, conditions, paired, cohort_rows, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names = units(config)
    selected_routes = ['same_model_twice', 'aligned_complement', 'direct_transfer', 'own_probes']
    existing = {r['route'] for r in conditions}
    selected_routes = [r for r in selected_routes if r in existing]
    colors = dict(zip(selected_routes, ['C0', 'C1', 'C2', 'C3']))
    index = {(r['unit'],r['route'],r['variant_family'],r['task'],r['support'],r['stratum']):r for r in conditions}
    fig, axes = plt.subplots(1,3,figsize=(15,4.5),sharey=True,constrained_layout=True)
    for task, ax in zip(TASKS,axes):
        for route in selected_routes:
            values = [finite(index.get((u,route,'original',task,'all','all'),{}).get('exact_accuracy')) for u in names]
            ax.plot(range(len(names)),[np.nan if v is None else v for v in values],marker='o',label=route,color=colors[route])
        ax.set_title(TASK_LABELS[task],fontsize=9);ax.set_xticks(range(len(names)),names,rotation=15);ax.set_ylim(0,1);ax.set_ylabel('Exact answer accuracy');ax.grid(alpha=.2)
    axes[0].legend(fontsize=6);fig.suptitle('Frozen readouts, deterministic mask-only rules; historical unit shown separately at left')
    fig.savefig(out/'task_exact_accuracy_by_unit.png',dpi=170);plt.close(fig)
    fig, axes = plt.subplots(1,3,figsize=(15,4.5),constrained_layout=True)
    for task,ax in zip(TASKS,axes):
        for si,support in enumerate(['all','matched']):
            values=[]
            for u in names:
                rr=[r for r in paired if r['unit']==u and r['variant_family']=='original' and r['task']==task and r['support']==support and r['stratum']=='all']
                values.append(finite(rr[0]['aligned_minus_same_model']) if len(rr)==1 else None)
            ax.plot(np.arange(len(names))+(si-.5)*.04,[np.nan if v is None else v for v in values],marker='o',label=support)
        ax.axhline(0,color='.5',linestyle='--');ax.set_title(TASK_LABELS[task],fontsize=9);ax.set_xticks(range(len(names)),names,rotation=15);ax.set_ylabel('Aligned minus same-model-twice exact accuracy');ax.grid(alpha=.2)
    axes[0].legend();fig.suptitle('Paired unit differences; no condition or length is an extra independent replication')
    fig.savefig(out/'aligned_minus_same_model.png',dpi=170);plt.close(fig)
    cindex={(r['route'],r['variant_family'],r['task'],r['support'],r['stratum'],r['metric']):r for r in cohort_rows if r['cohort']=='new_units'}
    fig, axes = plt.subplots(1,3,figsize=(15,4.7),sharey=True,constrained_layout=True)
    families=['original','turn_shuffle_mean','fixed_shuffle_mean']
    for task,ax in zip(TASKS,axes):
        for j,route in enumerate(selected_routes):
            for i,family in enumerate(families):
                row=cindex.get((route,family,task,'all','all','exact_accuracy'),{})
                value=finite(row.get('mean'))
                if value is None:continue
                low,high=finite(row.get('minimum')),finite(row.get('maximum'))
                ax.errorbar(i+(j-(len(selected_routes)-1)/2)*.13,value,yerr=[[value-low],[high-value]],fmt='o',capsize=2,color=colors[route],label=route if i==0 else None)
        ax.set_xticks(range(3),['original','turns shuffled','fixed points shuffled'],rotation=12);ax.set_title(TASK_LABELS[task],fontsize=9);ax.set_ylim(0,1);ax.set_ylabel('Exact answer accuracy');ax.grid(alpha=.2)
    axes[0].legend(fontsize=6);fig.suptitle('Count-preserving mask interventions; seed means within unit, bars = four-unit min–max, not CIs')
    fig.savefig(out/'position_shuffle_effects.png',dpi=170);plt.close(fig)
    fig, ax = plt.subplots(figsize=(10,5),constrained_layout=True)
    strata=['all','fixed_present','constraint_changes']
    for j,route in enumerate(selected_routes):
        for i,stratum in enumerate(strata):
            row=cindex.get((route,'original','fixed_longest','all',stratum,'exact_accuracy'),{})
            value=finite(row.get('mean'))
            if value is None:continue
            low,high=finite(row.get('minimum')),finite(row.get('maximum'))
            ax.errorbar(i+(j-(len(selected_routes)-1)/2)*.13,value,yerr=[[value-low],[high-value]],fmt='o',capsize=3,color=colors[route],label=route if i==0 else None)
    ax.set_xticks(range(3),['All test anchors','True fixed point present','Constraint changes oracle interval']);ax.set_ylim(0,1);ax.set_ylabel('Constrained-interval exact accuracy');ax.grid(alpha=.2);ax.legend(fontsize=7)
    ax.set_title('Truth strata are scoring-only; bars = four-unit min–max, not confidence intervals')
    fig.savefig(out/'fixed_constraint_diagnostic.png',dpi=170);plt.close(fig)


def report(config, conditions, paired_summary, shuffle_summary, cohort_rows, out):
    fmt=lambda x:'undefined' if finite(x) is None else f'{float(x):.6f}'
    lines=['# Frozen position-readout composition results','',
        'The original masks and source-only thresholds are reused across all three downstream questions. Formal composers receive predicted Boolean masks and length only. New-task truths define scoring and diagnostic strata; they never select probes, maps, thresholds or routes. The historical unit is an engineering reference; four newer units are the repetition units.','',
        '## Aligned route versus the same-model-twice baseline','',
        '| Cohort | Task | Support | Stratum | Mean aligned-minus-baseline | Sample SD | Range | Positive units |','|---|---|---|---|---:|---:|---:|---:|']
    for r in paired_summary:
        if r['variant_family']!='original' or r['stratum'] not in ['all','constraint_changes']:continue
        lines.append('| '+' | '.join(str(r[k]) for k in ['cohort','task','support','stratum'])+f" | {fmt(r['mean'])} | {fmt(r['sample_sd'])} | {fmt(r['minimum'])} to {fmt(r['maximum'])} | {r['positive_units']}/{r['registered_units']} |")
    lines += ['', 'Every original unit value and contrast is retained in `aligned_minus_same_model.csv`. A high upstream position AUROC does not by itself establish better exact composition answers. Negative, small and mixed contrasts are retained.','',
        '## Exact-answer accuracy, four newer units','',
        '| Route | Task | Support | Mean | Sample SD | Range |','|---|---|---|---:|---:|---:|']
    for r in cohort_rows:
        if r['cohort']!='new_units' or r['variant_family']!='original' or r['metric']!='exact_accuracy' or r['stratum']!='all' or r['route'] not in ['same_model_twice','aligned_complement','direct_transfer','own_probes']:continue
        lines.append('| '+' | '.join(str(r[k]) for k in ['route','task','support'])+f" | {fmt(r['mean'])} | {fmt(r['sample_sd'])} | {fmt(r['minimum'])} to {fmt(r['maximum'])} |")
    lines += ['', '## Count-preserving position-shuffle drops','',
        'The table gives original exact accuracy minus the mean of the eight registered mask-shuffle realizations **within each unit**, then summarizes across four units. The seeds are not extra scientific repetitions. A negative value means the shuffled prediction scores better.','',
        '| Route | Task | Intervention | Stratum | Mean accuracy drop | Sample SD | Range |','|---|---|---|---|---:|---:|---:|']
    for r in shuffle_summary:
        if r['cohort']!='new_units' or r['support']!='all' or r['route'] not in ['same_model_twice','aligned_complement','own_probes'] or r['stratum'] not in ['all','fixed_present','constraint_changes']:continue
        lines.append('| '+' | '.join(str(r[k]) for k in ['route','task','variant_family','stratum'])+f" | {fmt(r['mean'])} | {fmt(r['sample_sd'])} | {fmt(r['minimum'])} to {fmt(r['maximum'])} |")
    lines += ['', 'Fixed-position shuffling must leave the unconstrained task bit-identical. Every count-preserving intervention must preserve predicted NONE decisions. On true-no-fixed-point examples the two fixed-point questions therefore have unchanged exact correctness; their fixed-present and constraint-change strata are reported to avoid treating this algebraic invariance as evidence of useless position information.','',
        '## Scoring and uncertainty','',
        'Primary accuracy is the fixed equal-1/21 mean of exact-answer rates within each registered length. All-test and the original count-matched support remain separate. Missing/empty length components propagate as undefined; they are not zero-filled or silently reweighted. Pooled sample-weighted accuracy is an explicitly separate supplementary column.','',
        'Intervals use inclusive 1-based endpoints. Raw start/end/endpoint MAE and length error are conditional on both true and predicted intervals being present, with coverage and presence mistakes recorded. Normalized unconditional errors divide finite interval errors by length, assign penalty 1 to a presence mismatch and 0 when both answers are NONE. Count MAE is similarly conditional on both finite answers; normalized count error uses the same explicit NONE convention.','',
        'The diagnostic constraint-change subset requires a true fixed point and a constrained oracle interval different from the unconstrained oracle interval; absent fixed points are not counted as a trivial constraint change. False/missed fixed-point presence can dominate aggregate scores, so true-present/true-absent results remain explicit. No new significance, equivalence test, bootstrap confidence interval, task-pair generality or neural causal-use claim is introduced. Mask interventions test the specified analytic composition pipeline.','',
        '## Files and figures','',
        '`per_unit_metrics.csv` retains every original and shuffled seed; `per_unit_condition_summary.csv` averages seeds only within each unit. `cohort_summary.csv` contains four-new/all-five means, sample SDs, ranges and raw unit values. Length tables are separate under `per_length/`. Per-sample predictions remain in their hash-bound source NPZ files.','',
        '![Unit task accuracy](task_exact_accuracy_by_unit.png)','',
        '![Aligned versus baseline](aligned_minus_same_model.png)','',
        '![Shuffle effects](position_shuffle_effects.png)','',
        '![Constraint diagnostic](fixed_constraint_diagnostic.png)','']
    (out/'report.md').write_text('\n'.join(lines))


def summarize(config, unit_aggregates, out, make_figures=True):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    conditions=condition_summaries(unit_aggregates,int(config.get('mask_shuffle_repetitions',8)))
    co=cohort_summaries(config,conditions)
    paired,pairsum=paired_effects(config,conditions)
    effects,effectsum=shuffle_effects(config,conditions)
    for name,data in [('per_unit_metrics.csv',unit_aggregates),('per_unit_condition_summary.csv',conditions),
                      ('cohort_summary.csv',co),('aligned_minus_same_model.csv',paired),('paired_summary.csv',pairsum),
                      ('shuffle_effects_by_unit.csv',effects),('shuffle_effects_summary.csv',effectsum)]:
        write_csv(out/name,data)
    if make_figures:make_plots(config,conditions,paired,co,out)
    report(config,conditions,pairsum,effectsum,co,out)
    return dict(status='completed_saved_composition_summary',units=units(config),
                per_unit_raw_rows=len(unit_aggregates),condition_rows=len(conditions),
                cohort_rows=len(co),paired_rows=len(paired),shuffle_effect_rows=len(effects),
                no_probe_fitting=True,no_neural_forward=True,no_downstream_tuning=True,
                fixed_length_weights=True,mask_shuffle_seeds_not_replications=True)


def prediction_source(protocol, unit, args):
    receipt_path=args.protocol.parent/'results'/unit/'completion.json'
    receipt=read(receipt_path)
    if receipt.get('status') not in ['completed','completed_and_stopped']:
        raise ValueError('Incomplete registered unit: '+unit)
    binding=receipt.get('protocol_sha256',receipt.get('protocol_SHA256'))
    if binding is not None and binding!=sha(args.protocol):raise ValueError('Prediction protocol binding mismatch')
    descriptor=receipt.get('predictions')
    if isinstance(descriptor,dict):path=Path(descriptor['path']);expected=descriptor.get('sha256')
    elif receipt.get('predictions_path'):
        path=Path(receipt['predictions_path']);expected=receipt.get('predictions_sha256')
    else:
        base=args.predictions_root or Path(protocol['output'])/'predictions'
        path=base/unit/'predictions.npz';expected=receipt.get('prediction_sha256')
    if not path.is_absolute():path=ROOT/path
    digest=sha(path)
    if expected is not None and digest!=expected:raise ValueError('Prediction SHA mismatch: '+str(path))
    return path,digest,str(receipt_path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',type=Path,default=HERE/'protocol.json')
    parser.add_argument('--predictions-root',type=Path)
    parser.add_argument('--out',type=Path,default=HERE/'summary')
    parser.add_argument('--synthetic-checks',action='store_true')
    args=parser.parse_args()
    if args.synthetic_checks:
        synthetic_checks();return
    config=read(args.protocol)
    weights=config.get('length_weights')
    if weights is not None:
        values=list(weights.values()) if isinstance(weights,dict) else list(weights)
        if len(values)!=len(config.get('lengths',list(range(10,31)))) or not np.allclose(values,1/len(values),rtol=0,atol=1e-15):
            raise ValueError('This frozen diagnostic requires equal registered length weights')
    freeze=HERE/'summary_helper_freeze.json'
    source={str(Path(__file__).relative_to(ROOT)):sha(__file__),
            'experiments/permworld_lemma_composition_20261110/metrics.py':sha(ROOT/'experiments/permworld_lemma_composition_20261110/metrics.py')}
    if freeze.exists():
        old=read(freeze)
        if old['source_sha256']!=source or old['protocol_sha256']!=sha(args.protocol):raise ValueError('Frozen summary helper changed')
    else:write_json(freeze,dict(source_sha256=source,protocol_sha256=sha(args.protocol),before_first_actual_prediction_read=True))
    aggregate=[];audits=[];sources=[];inventory=None
    for unit in units(config):
        path,digest,receipt=prediction_source(config,unit,args)
        with np.load(path,allow_pickle=False) as raw:arrays={k:raw[k] for k in raw.files}
        current=([str(x) for x in arrays['route_names']],[str(x) for x in arrays['variant_names']])
        if inventory is None:inventory=current
        elif current!=inventory:raise ValueError('Route/variant inventory differs across registered units')
        length_rows,unit_rows,audit=score_unit_arrays(unit,arrays,config.get('lengths',list(range(10,31))))
        write_csv(args.out/'per_length'/f'{unit}.csv',length_rows)
        aggregate.extend(unit_rows);audits.append(dict(unit=unit,**audit))
        sources.append(dict(unit=unit,path=str(path),sha256=digest,receipt=receipt))
        print(json.dumps(dict(unit=unit,per_length_rows=len(length_rows),unit_aggregate_rows=len(unit_rows),invariants='passed')),flush=True)
        del arrays,length_rows
    result=summarize(config,aggregate,args.out)
    result.update(completed_utc=datetime.now(timezone.utc).isoformat(),protocol_sha256=sha(args.protocol),
                  source_predictions=sources,invariants=audits)
    write_json(args.out/'completion.json',result);print(json.dumps(result),flush=True)


def synthetic_arrays(unit_shift=0):
    ns=np.repeat(np.arange(10,31),6);n=len(ns)
    j=np.tile(np.arange(6),21)
    present=j%3!=0
    longest=np.column_stack([np.ones(n,int),ns])
    constrained=np.where(present[:,None],np.column_stack([np.where(j%2,1,2),ns]),-1)
    count=np.where(present,j%3,-1)
    routes=['same_model_twice','aligned_complement','direct_transfer','own_probes']
    variants=['original',*[f'turn_shuffle_{s}' for s in range(611101,611109)],*[f'fixed_shuffle_{s}' for s in range(611101,611109)]]
    pl=np.broadcast_to(longest,(len(routes),len(variants),n,2)).copy()
    pf=np.broadcast_to(constrained,pl.shape).copy()
    pc=np.broadcast_to(count,(len(routes),len(variants),n)).copy()
    # Synthetic variation remains in valid answer space and never changes NONE.
    for ri in range(len(routes)):
        change=(j+ri+unit_shift)%4==0
        pl[ri,:,change,0]=2
        for vi,name in enumerate(variants):
            if name.startswith('turn_shuffle_'):
                pl[ri,vi,:,0]=2
                pf[ri,vi,present,0]=1
                pc[ri,vi,present]=0
            elif name.startswith('fixed_shuffle_'):
                pf[ri,vi,present,0]=1
                pc[ri,vi,present]=0
    return dict(lengths=ns,anchor_indices=np.arange(n),matched=j%2==0,
        truth_fixed_present=present,truth_constraint_changes=present & (constrained[:,0]!=1),
        truth_longest=longest,truth_fixed_longest=constrained,truth_turns_before_fixed=count,
        route_names=np.asarray(routes),variant_names=np.asarray(variants),
        pred_longest=pl,pred_fixed_longest=pf,pred_turns_before_fixed=pc)


def synthetic_checks():
    import tempfile
    original=synthetic_arrays()
    lr,ur,audit=score_unit_arrays('toy',original)
    assert len(lr)==4*17*16*21 and len(ur)==4*17*16
    for r in ur:
        if r['stratum']=='fixed_absent' and r['task']!='longest':
            assert r['exact_accuracy']==1
            assert r['joint_present_coverage']==0
        if r['task']=='fixed_longest' and r['stratum']=='fixed_absent':
            assert r['endpoint_mae'] is None and r['endpoint_mae_finite_lengths']==0
            assert r['normalized_endpoint_error']==0
    broken={k:np.array(v,copy=True) for k,v in original.items()}
    broken['pred_longest'][0,9,0,0]=3
    try:score_unit_arrays('bad',broken)
    except ValueError:pass
    else:raise AssertionError('FP-shuffle invariant was not enforced')
    broken={k:np.array(v,copy=True) for k,v in original.items()}
    broken['pred_turns_before_fixed'][0,1,0]=0
    try:score_unit_arrays('bad',broken)
    except ValueError:pass
    else:raise AssertionError('NONE invariant was not enforced')
    interval_truth=np.asarray([[1,10],[-1,-1],[2,6],[-1,-1]])
    interval_pred=np.asarray([[-1,-1],[1,3],[3,7],[-1,-1]])
    known=one_length_row({},sample_metrics('fixed_longest',interval_truth,interval_pred,np.full(4,10)),np.ones(4,bool))
    assert known['exact_accuracy']==.25 and known['joint_present_coverage']==.25
    assert known['endpoint_mae']==1 and known['length_mae']==0
    assert abs(known['normalized_endpoint_error']-.525)<1e-15
    assert known['normalized_length_error']==.5
    assert known['missed_presence_n']==known['false_presence_n']==1
    config=dict(units=[HISTORICAL,'unit02','unit03','unit04','unit05'],lengths=list(range(10,31)))
    all_rows=[]
    for i,unit in enumerate(units(config)):
        _,u,_=score_unit_arrays(unit,synthetic_arrays(i));all_rows.extend(u)
    with tempfile.TemporaryDirectory() as folder:
        result=summarize(config,all_rows,Path(folder),make_figures=True)
        assert len(list(Path(folder).glob('*.png')))==4
    receipt=dict(status='passed_synthetic_only',single_unit_length_rows=len(lr),single_unit_aggregate_rows=len(ur),
        synthetic_units=5,invariant_negative_checks=2,hand_calculated_interval_error_case=True,figures=4,actual_predictions_read=False,
        summary_helper_sha256=sha(__file__),metrics_source_sha256=sha(ROOT/'experiments/permworld_lemma_composition_20261110/metrics.py'))
    write_json(HERE/'summary_synthetic_verification.json',receipt);print(json.dumps(receipt))


if __name__=='__main__':main()
