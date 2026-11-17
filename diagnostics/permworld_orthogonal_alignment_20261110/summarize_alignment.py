"""Saved-table summaries for the frozen orthogonal-alignment repetition.

No activation extraction, alignment/probe fitting, or model construction occurs
here. Every registered length is required; maximum controls are taken after
the original equal-length aggregation, not by pooling or selecting lengths.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
UNITS=('archdev001','unit02','unit03','unit04','unit05')
ENCODINGS=('canonical_cycle','one_line')
PHASES=('initial','trained')
SUPPORTS=('matched','eligible')
LENGTHS=tuple(range(10,31))
SHUFFLES=tuple(range(1026,1034))
CORE_METHODS=('direct','means_only','within_peaks','within_valleys','oracle_count_position_prior')
PAIR_METHODS=tuple(f'fit_{a}__test_{suffix}' for a in ('identity','complement','inverse')
                   for suffix in ('complement','native','native_own_valley'))
SHUFFLE_METHODS=tuple(f'fit_shuffle_{s}__test_complement' for s in SHUFFLES)
METHODS=CORE_METHODS+PAIR_METHODS+SHUFFLE_METHODS
NUMERIC_METRICS=('direct_auc','aligned_auc','gain_vs_direct','means_only_auc','identity_map_auc',
    'inverse_map_auc','initial_aligned_auc','initial_direct_auc','aligned_minus_initial_aligned',
    'within_peaks_auc','within_valleys_auc','oracle_count_position_prior_auc',
    *(f'shuffle_{s}_auc' for s in SHUFFLES),'shuffle_mean_auc','shuffle_min_auc','shuffle_max_auc',
    'descents_direct_auc','descents_aligned_auc','recoils_direct_auc','recoils_aligned_auc',
    'best_map_control_auc','best_all_control_auc','aligned_minus_best_map','aligned_minus_best_all')
TOLERANCE=1e-12


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def read_csv(path):
    with Path(path).open(newline='') as stream:return list(csv.DictReader(stream))


def write_csv(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fields=list(dict.fromkeys(k for r in rows for k in r)) or ['status']
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)


def write_json(path,value):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def finite(value,context):
    answer=float(value)
    if not math.isfinite(answer):raise ValueError('Undefined required score: '+str(context))
    if not -TOLERANCE<=answer<=1+TOLERANCE:raise ValueError('AUROC outside range: '+str(context))
    return answer


def scalar_mean(values):return math.fsum(values)/len(values)


def validate_tables(unit,summary,detail,other,other_detail):
    """Independently reproduce every existing equal-length scalar summary."""
    keys=('encoding','phase','support','method')
    expected={(e,p,s,m) for e in ENCODINGS for p in PHASES for s in SUPPORTS for m in METHODS}
    index={tuple(r[k] for k in keys):r for r in summary}
    if len(index)!=len(summary) or set(index)!=expected:raise ValueError((unit,'Alignment score inventory differs'))
    groups=defaultdict(dict)
    all_detail=[]
    for original in detail:
        row=dict(original);key=tuple(row[k] for k in keys);n=int(row['length'])
        if key not in expected or n not in LENGTHS or n in groups[key]:raise ValueError((unit,'Duplicate/unregistered length',key,n))
        score=finite(row['auc'],(unit,key,n));count=int(row['n']);defined=int(row['finite'])
        if not count>0 or not 0<defined<=count:raise ValueError((unit,'Invalid support count',key,n))
        row.update(unit=unit,length=n,auc=score,n=count,finite=defined)
        groups[key][n]=row;all_detail.append(row)
    maximum=0.
    for key in sorted(expected):
        if set(groups[key])!=set(LENGTHS):raise ValueError((unit,'Missing required length',key))
        rows=[groups[key][n] for n in LENGTHS];value=scalar_mean([r['auc'] for r in rows]);target=index[key]
        observed=finite(target['auc'],(unit,key,'aggregate'));error=abs(value-observed);maximum=max(maximum,error)
        if error>TOLERANCE or int(target['lengths'])!=21:raise ValueError((unit,'Aggregate mismatch',key,error))
        if int(target['anchors'])!=sum(r['n'] for r in rows) or int(target['finite'])!=sum(r['finite'] for r in rows):raise ValueError((unit,'Aggregate support mismatch',key))
    # Main methods share original peak labels and complement test anchors.
    for phase in PHASES:
        for support in SUPPORTS:
            for n in LENGTHS:
                counts={(groups[e,phase,support,m][n]['n'],groups[e,phase,support,m][n]['finite'])
                        for e in ENCODINGS for m in ('direct','means_only',*SHUFFLE_METHODS,
                              'fit_identity__test_complement','fit_complement__test_complement','fit_inverse__test_complement')}
                if len(counts)!=1:raise ValueError((unit,'Main scoring support differs',phase,support,n,counts))
    okeys=('target','support','method')
    oexpected={(t,s,m) for t in ('descents','recoils') for s in SUPPORTS for m in ('direct','procrustes')}
    oi={tuple(r[k] for k in okeys):r for r in other}
    if len(oi)!=len(other) or set(oi)!=oexpected:raise ValueError((unit,'Task-control inventory differs'))
    og=defaultdict(dict);all_other=[]
    for original in other_detail:
        row=dict(original);key=tuple(row[k] for k in okeys);n=int(row['length'])
        if key not in oexpected or n not in LENGTHS or n in og[key]:raise ValueError((unit,'Task-control length inventory differs',key,n))
        row.update(unit=unit,encoding='one_line',phase='trained',length=n,auc=finite(row['auc'],(unit,key,n)))
        og[key][n]=row;all_other.append(row)
    for key in sorted(oexpected):
        if set(og[key])!=set(LENGTHS):raise ValueError((unit,'Missing task-control length',key))
        value=scalar_mean([og[key][n]['auc'] for n in LENGTHS]);target=oi[key]
        error=abs(value-finite(target['auc'],(unit,key,'aggregate')));maximum=max(maximum,error)
        if error>TOLERANCE or int(target['lengths'])!=21:raise ValueError((unit,'Task-control aggregate mismatch',key,error))
    return index,groups,oi,og,all_detail,all_other,maximum


def table_row(unit,encoding,support,get,task_get,length=None,anchors=None,defined=None):
    """A score getter spans phases; controls are fixed before outcomes."""
    c=get('trained','fit_complement__test_complement');direct=get('trained','direct')
    sh={s:get('trained',f'fit_shuffle_{s}__test_complement') for s in SHUFFLES}
    row=dict(unit=unit,encoding=encoding,support=support,direct_auc=direct,aligned_auc=c,
        gain_vs_direct=c-direct,means_only_auc=get('trained','means_only'),
        identity_map_auc=get('trained','fit_identity__test_complement'),inverse_map_auc=get('trained','fit_inverse__test_complement'),
        initial_aligned_auc=get('initial','fit_complement__test_complement'),initial_direct_auc=get('initial','direct'),
        within_peaks_auc=get('trained','within_peaks'),within_valleys_auc=get('trained','within_valleys'),
        oracle_count_position_prior_auc=get('trained','oracle_count_position_prior'),
        shuffle_mean_auc=scalar_mean(list(sh.values())),shuffle_min_auc=min(sh.values()),shuffle_max_auc=max(sh.values()))
    row.update({f'shuffle_{s}_auc':v for s,v in sh.items()})
    row['aligned_minus_initial_aligned']=c-row['initial_aligned_auc']
    map_controls={'means_only':row['means_only_auc'],'identity_map':row['identity_map_auc'],'inverse_map':row['inverse_map_auc'],
                  **{f'shuffle_{s}':v for s,v in sh.items()}}
    all_controls={**map_controls,'initial_aligned':row['initial_aligned_auc'],'oracle_count_position_prior':row['oracle_count_position_prior_auc']}
    for target in ('descents','recoils'):
        for method,label in [('direct','direct'),('procrustes','aligned')]:
            value=task_get(target,method) if encoding=='one_line' else None
            row[f'{target}_{label}_auc']=value
            if value is not None:all_controls[f'{target}_{label}']=value
    for label,controls in [('map',map_controls),('all',all_controls)]:
        best=max(controls.values());row[f'best_{label}_control_auc']=best
        row[f'best_{label}_control_names']=';'.join(k for k,v in controls.items() if v==best)
        row[f'aligned_minus_best_{label}']=c-best
        row[f'beats_best_{label}']=c>best;row[f'beats_direct_and_best_{label}']=c>direct and c>best
    row.update(beats_direct=c>direct,
        controls_scope='B_map plus step0/oracle-count; descents/recoils not evaluated in cycle' if encoding!='one_line' else 'B_map plus step0/oracle-count and both direct/aligned descents/recoils',
        best_control_definition='Maximum across EACH eight shuffle score, never shuffle mean; own-model readouts excluded',
        aggregation='Per-length component' if length is not None else 'Equal1/21 mean; controls maximized AFTER aggregation')
    if length is not None:row['length']=length
    if anchors is not None:row['anchors']=anchors
    if defined is not None:row['finite']=defined
    return row


def summarize_runs(runs,out,make_plots=True):
    unit_tables=[];per_length=[];scores=[];details=[];other_scores=[];other_details=[];fit_audits=[];checks=[]
    for unit in UNITS:
        run=runs[unit]
        ix,groups,oi,og,dd,od,error=validate_tables(unit,run['summary'],run['detail'],run['other'],run['other_detail'])
        checks.append(dict(unit=unit,max_absolute_aggregation_error=error,alignment_scores=len(ix),alignment_lengths=len(dd),task_control_scores=len(oi),task_control_lengths=len(od)))
        scores.extend(dict(r,unit=unit,auc=finite(r['auc'],(unit,'aggregate'))) for r in run['summary']);details.extend(dd)
        other_scores.extend(dict(r,unit=unit,encoding='one_line',phase='trained',auc=finite(r['auc'],(unit,'taskaggregate'))) for r in run['other']);other_details.extend(od)
        audit=run['fit_audit']
        if {(r['encoding'],r['phase']) for r in audit}!={(e,p) for e in ENCODINGS for p in PHASES} or len(audit)!=4:raise ValueError((unit,'Fit audit inventory'))
        for r in audit:
            if int(r['fit_pairs'])<=0 or not 0<=float(r['min_shuffled_fraction'])<=1:raise ValueError((unit,'Fit audit bounds'))
            fit_audits.append(dict(r,unit=unit))
        for e in ENCODINGS:
            for support in SUPPORTS:
                get=lambda p,m:finite(ix[e,p,support,m]['auc'],(unit,e,p,support,m))
                task=lambda t,m:finite(oi[t,support,m]['auc'],(unit,t,m))
                ref=ix[e,'trained',support,'fit_complement__test_complement']
                unit_tables.append(table_row(unit,e,support,get,task,anchors=int(ref['anchors']),defined=int(ref['finite'])))
                for n in LENGTHS:
                    get_n=lambda p,m:groups[e,p,support,m][n]['auc']
                    task_n=lambda t,m:og[t,support,m][n]['auc']
                    ref_n=groups[e,'trained',support,'fit_complement__test_complement'][n]
                    per_length.append(table_row(unit,e,support,get_n,task_n,length=n,anchors=ref_n['n'],defined=ref_n['finite']))
    cohorts=[];success=[];directions=[]
    for cohort,units in [('new_units',UNITS[1:]),('all_units',UNITS)]:
        for e in ENCODINGS:
            for s in SUPPORTS:
                chosen=[r for r in unit_tables if r['unit'] in units and r['encoding']==e and r['support']==s]
                assert len(chosen)==len(units)
                for metric in NUMERIC_METRICS:
                    values=[r[metric] for r in chosen];defined=[v for v in values if v is not None]
                    all_defined=len(defined)==len(units)
                    cohorts.append(dict(cohort=cohort,encoding=e,support=s,metric=metric,registered_units=len(units),finite_units=len(defined),
                        mean=scalar_mean(defined) if all_defined else None,sample_sd=statistics.stdev(defined) if all_defined and len(defined)>1 else None,
                        minimum=min(defined) if all_defined else None,maximum=max(defined) if all_defined else None,
                        per_unit_values=';'.join(r['unit']+':'+str(r[metric]) for r in chosen),undefined_reason='' if all_defined else 'task control not evaluated in this encoding'))
                success.append(dict(cohort=cohort,encoding=e,support=s,units=len(units),
                    beats_direct_units=sum(r['beats_direct'] for r in chosen),
                    beats_direct_and_best_map_units=sum(r['beats_direct_and_best_map'] for r in chosen),
                    beats_direct_and_best_all_units=sum(r['beats_direct_and_best_all'] for r in chosen),
                    primary_requested_rule='Aligned C > direct AND > B_all; within-model probes excluded',controls_scope=chosen[0]['controls_scope']))
    for r in unit_tables:
        selected=[x for x in per_length if x['unit']==r['unit'] and x['encoding']==r['encoding'] and x['support']==r['support']]
        assert len(selected)==21
        directions.append(dict(unit=r['unit'],encoding=r['encoding'],support=r['support'],required_lengths=21,
            gain_positive_lengths=sum(x['gain_vs_direct']>0 for x in selected),
            above_best_map_lengths=sum(x['aligned_minus_best_map']>0 for x in selected),
            above_best_all_lengths=sum(x['aligned_minus_best_all']>0 for x in selected),
            joint_direct_and_best_map_lengths=sum(x['beats_direct_and_best_map'] for x in selected),
            joint_direct_and_best_all_lengths=sum(x['beats_direct_and_best_all'] for x in selected),
            gain_minimum=min(x['gain_vs_direct'] for x in selected),gain_maximum=max(x['gain_vs_direct'] for x in selected),
            mean_lengthwise_margin_best_map=scalar_mean([x['aligned_minus_best_map'] for x in selected]),
            mean_lengthwise_margin_best_all=scalar_mean([x['aligned_minus_best_all'] for x in selected]),
            aggregate_margin_best_map=r['aligned_minus_best_map'],aggregate_margin_best_all=r['aligned_minus_best_all'],
            interpretation='Dependent length conditions; aggregate control maximum differs from mean of lengthwise control maxima'))
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    for name,data in [('unit_comparison',unit_tables),('cohort_summary',cohorts),('success_counts',success),('per_length_comparison',per_length),
                      ('per_length_directions',directions),('all_alignment_scores',scores),('all_alignment_by_length',details),
                      ('other_specialist_scores',other_scores),('other_specialists_by_length',other_details),('alignment_fit_audits',fit_audits),('aggregation_checks',checks)]:write_csv(out/(name+'.csv'),data)
    if make_plots:plots(unit_tables,per_length,out)
    report(unit_tables,cohorts,success,directions,out)
    return dict(status='completed_saved_alignment_summary',units=list(UNITS),new_units=list(UNITS[1:]),
        unit_comparison_rows=len(unit_tables),per_length_comparison_rows=len(per_length),alignment_score_rows=len(scores),
        alignment_length_rows=len(details),task_control_score_rows=len(other_scores),task_control_length_rows=len(other_details),
        required_lengths=list(LENGTHS),max_absolute_aggregation_error=max(r['max_absolute_aggregation_error'] for r in checks),
        no_alignment_or_probe_fitting=True,no_neural_forward=True,
        primary_success_rule='Aligned C > direct AND > B_all',
        statistical_scope='Four-new/all-five raw unit values, equal-unit mean/sampleSD/range; no new CI/p-values or length-as-unit inference')


def plots(rows,detail,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    selected=[r for r in rows if r['support']=='matched']
    fig,axes=plt.subplots(1,2,figsize=(13,4.8),sharey=True,constrained_layout=True)
    fields=['direct_auc','aligned_auc','best_all_control_auc','within_valleys_auc','initial_aligned_auc']
    for ax,e in zip(axes,ENCODINGS):
        rs=[next(r for r in selected if r['unit']==u and r['encoding']==e) for u in UNITS]
        for k in fields:ax.plot(range(5),[r[k] for r in rs],marker='o',label=k.replace('_auc','').replace('_',' '))
        ax.axhline(.5,color='.5',ls='--');ax.set_xticks(range(5),UNITS,rotation=15);ax.set_title(e);ax.set_ylabel('Matched-support position AUROC');ax.grid(alpha=.2)
    axes[1].legend(fontsize=7);fig.suptitle('Existing-cache orthogonal alignment; historical unit shown separately at left')
    fig.savefig(out/'alignment_recovery_by_unit.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(13,4.8),sharey=True,constrained_layout=True)
    for ax,e in zip(axes,ENCODINGS):
        for j,u in enumerate(UNITS):
            r=next(x for x in selected if x['unit']==u and x['encoding']==e)
            ax.plot(np.arange(3)+(j-2)*.025,[r[k] for k in ['gain_vs_direct','aligned_minus_best_map','aligned_minus_best_all']],marker='x' if j==0 else 'o',label=u)
        ax.axhline(0,color='.5',ls='--');ax.set_xticks(range(3),['Aligned−direct','Aligned−B_map','Aligned−B_all'],rotation=12);ax.set_title(e);ax.set_ylabel('AUROC difference');ax.grid(alpha=.2)
    axes[1].legend(fontsize=7);fig.suptitle('Paired contrasts; one point per unit, not independent lengths or shuffles')
    fig.savefig(out/'alignment_control_margins.png',dpi=170);plt.close(fig)
    fields=['direct_auc','aligned_auc','best_all_control_auc','within_valleys_auc']
    values=[r[k] for r in detail if r['support']=='matched' for k in fields]
    low,maxi=min(values),max(values);padding=max(.015,(maxi-low)*.07)
    for e in ENCODINGS:
        fig,axes=plt.subplots(2,3,figsize=(13,7),sharex=True,sharey=True,constrained_layout=True)
        for ax,u in zip(axes.flat,UNITS):
            rs=sorted([r for r in detail if r['unit']==u and r['encoding']==e and r['support']=='matched'],key=lambda r:r['length'])
            for k in fields:ax.plot(LENGTHS,[r[k] for r in rs],marker='.',markersize=3,label=k.replace('_auc','').replace('_',' '))
            ax.axhline(.5,color='.5',ls='--');ax.set_ylim(max(0,low-padding),min(1,maxi+padding));ax.set_title(u+(' (historical)' if u==UNITS[0] else ''));ax.grid(alpha=.2);ax.set_xlabel('Permutation length');ax.set_ylabel('Matched position AUROC')
        axes.flat[-1].axis('off');axes.flat[0].legend(fontsize=6);fig.suptitle(e+': all registered lengths, common vertical scale across encodings')
        fig.savefig(out/(e+'_per_length_alignment.png'),dpi=170);plt.close(fig)


def report(rows,cohorts,success,directions,out):
    fmt=lambda x:'not evaluated' if x is None else f'{x:.6f}'
    text=['# Orthogonal-alignment cached-unit results','','All original position-probe splits, scalers, source-only alpha selection, held-out complement inputs and peak-position labels are retained. A global target-to-source orthogonal map per phase/encoding/fit condition is fitted from validation-only raw final_norm features, with per-length means and equal-length covariance weighting. The source probe is fixed across mapping controls within each phase/length. No model forward or base training is performed.','','## Per-unit comparison (common-count matched support)','','| Unit | Encoding | Direct | Aligned C | Gain | B_map | B_all | Aligned−B_all | Initial aligned C | Within peaks | Within valleys |','|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        if r['support']=='matched':text.append('| '+r['unit']+' | '+r['encoding']+' | '+' | '.join(fmt(r[k]) for k in ['direct_auc','aligned_auc','gain_vs_direct','best_map_control_auc','best_all_control_auc','aligned_minus_best_all','initial_aligned_auc','within_peaks_auc','within_valleys_auc'])+' |')
    text+=['','The complete wide table contains both matched and eligible/full-test supports, each of the eight shuffle scores, their mean/minimum/maximum, means-only, identity and inverse maps, the oracle-count prior, and direct/aligned descents/recoils controls. Cycle task controls are not evaluated, rather than filled with zero.','',
        '**B_map** is the maximum of means-only, identity-map, inverse-map, and each of eight shuffled-complement maps. **B_all** additionally includes initial aligned complement, oracle-count position prior, and (for one-line) both direct and aligned descents/recoils. Within-model probes are positive readout references and are excluded. Native-transform/own-valley rows remain auxiliary; all main map comparisons score the same complement features and peak labels.','',
        'The control maximum is taken after each original equal-1/21 length aggregation. It is not the mean of per-length maxima. Both lengthwise margins and the aggregate margin are retained separately.','','## Four-new versus all-five summaries','','| Cohort | Encoding | Support | Metric | Mean | Sample SD | Minimum | Maximum |','|---|---|---|---|---:|---:|---:|---:|']
    for r in cohorts:
        if r['metric'] in ['direct_auc','aligned_auc','gain_vs_direct','aligned_minus_best_all']:
            text.append('| '+' | '.join(r[k] for k in ['cohort','encoding','support','metric'])+' | '+' | '.join(fmt(r[k]) for k in ['mean','sample_sd','minimum','maximum'])+' |')
    text+=['','## Prespecified success counts','','| Cohort | Encoding | Support | Units | Aligned > direct and B_map | Aligned > direct and B_all |','|---|---|---|---:|---:|---:|']
    for r in success:text.append('| '+' | '.join(str(r[k]) for k in ['cohort','encoding','support','units','beats_direct_and_best_map_units','beats_direct_and_best_all_units'])+' |')
    text+=['','The requested one-line count uses the stricter B_all rule, with the map-only count also reported. B_all contains different task-control availability across encodings; use B_map for a common control set. Four newer data/initialization/training bundles are the new repetition units. The historical exploratory pilot is retained separately. The 21 lengths and eight shuffle realizations are dependent conditions, not extra independent repetitions. No new significance/equivalence test, confidence interval or practical-zero threshold is introduced.','','## Per-length descriptions','','| Unit | Encoding | Support | Gain positive | Above B_map | Above B_all | Joint direct/B_all | Gain range |','|---|---|---|---:|---:|---:|---:|---:|']
    for r in directions:text.append('| '+' | '.join(str(r[k]) for k in ['unit','encoding','support'])+' | '+' | '.join(str(r[k])+'/21' for k in ['gain_positive_lengths','above_best_map_lengths','above_best_all_lengths','joint_direct_and_best_all_lengths'])+' | '+fmt(r['gain_minimum'])+' to '+fmt(r['gain_maximum'])+' |')
    text+=['','## Interpretation limits','','Recovery of a linear readout after fitting an isometry is a coordinate-recovery result. It does not show that the model discovered or uses the mathematical relation, nor that decoded positions are causally used in counting. Alignment itself uses paired validation features without position targets; source probes use position labels and shuffled controls use true peak counts. Initial and trained phases refit separate probes. The map is orthogonal in raw final_norm coordinates, not after source-standardization, and may include a reflection. The wrong maps intentionally share the complement centering at test time. The oracle-count prior uses the true test count. Descents/recoils are same-input specialist controls, not task/mastery-matched random controls.','','## Figures','','![Unit scores](alignment_recovery_by_unit.png)','','![Control margins](alignment_control_margins.png)','','![One-line lengths](one_line_per_length_alignment.png)','','![Cycle lengths](canonical_cycle_per_length_alignment.png)','']
    (out/'report.md').write_text('\n'.join(text))


def load_runs(results,protocol):
    c=json.loads(protocol.read_text());digest=sha(protocol);runs={};sources={}
    for unit in UNITS:
        base=results/unit;receipt=json.loads((base/'completion.json').read_text())
        if receipt['status']!='completed' or receipt['protocol_sha256']!=digest:raise ValueError((unit,'Completion binding'))
        paths={'summary':'summary.csv','detail':'by_length.csv','other':'other_specialists.csv','other_detail':'other_specialists_by_length.csv','fit_audit':'alignment_fit_audit.csv'}
        runs[unit]={}
        for key,name in paths.items():
            path=base/name;actual=sha(path)
            if receipt['result_sha256'][name]!=actual:raise ValueError((unit,'Saved result SHA',name))
            sources[str(path)]=actual;runs[unit][key]=read_csv(path)
    return runs,sources


def synthetic_smoke(out):
    """Synthetic scalar-only fixture; no real data, alignment or probe fit."""
    runs={}
    for u in UNITS:
        summary=[];detail=[];other=[];od=[]
        for e in ENCODINGS:
            for p in PHASES:
                for s in SUPPORTS:
                    for method in METHODS:
                        value=.5
                        if p=='trained':
                            if method=='fit_complement__test_complement':value=.65
                            elif method in SHUFFLE_METHODS:value=.64 if method==SHUFFLE_METHODS[-1] else .48
                            elif method=='within_peaks':value=.76
                            elif method=='within_valleys':value=.75
                            elif method=='means_only':value=.51
                        summary.append(dict(encoding=e,phase=p,support=s,method=method,auc=value,lengths=21,anchors=21*64,finite=21*64))
                        detail.extend(dict(encoding=e,phase=p,support=s,method=method,length=n,auc=value,n=64,finite=64) for n in LENGTHS)
        for t in ['descents','recoils']:
            for s in SUPPORTS:
                for m in ['direct','procrustes']:
                    value=.68 if t=='descents' and m=='procrustes' else .49
                    other.append(dict(target=t,support=s,method=m,auc=value,lengths=21))
                    od.extend(dict(target=t,support=s,method=m,auc=value,length=n) for n in LENGTHS)
        runs[u]=dict(summary=summary,detail=detail,other=other,other_detail=od,fit_audit=[dict(encoding=e,phase=p,fit_pairs=4200,min_shuffled_fraction=.997) for e in ENCODINGS for p in PHASES])
    result=summarize_runs(runs,out,make_plots=True);r=read_csv(Path(out)/'unit_comparison.csv')
    line=next(x for x in r if x['encoding']=='one_line' and x['support']=='matched')
    assert float(line['best_map_control_auc'])==.64 and float(line['best_all_control_auc'])==.68
    assert line['beats_direct_and_best_map']=='True' and line['beats_direct_and_best_all']=='False'
    broken={u:dict(x) for u,x in runs.items()};broken[UNITS[0]]['detail']=runs[UNITS[0]]['detail'][1:]
    try:summarize_runs(broken,Path(out)/'must_not_complete',make_plots=False)
    except ValueError:pass
    else:raise AssertionError('Missing required length not rejected')
    return dict(**result,synthetic_only=True,strict_maximum_control_test_passed=True,missing_length_guard_passed=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--protocol',type=Path,default=HERE/'protocol.json');parser.add_argument('--results',type=Path,default=HERE/'results');parser.add_argument('--out',type=Path,default=HERE/'summary');parser.add_argument('--synthetic-smoke',type=Path);args=parser.parse_args()
    if args.synthetic_smoke:print(json.dumps(synthetic_smoke(args.synthetic_smoke)));return
    # Register the exact helper before the first read of any actual outcomes.
    freeze=HERE/'summary_helper_freeze.json';signature=sha(Path(__file__))
    if freeze.exists():
        if json.loads(freeze.read_text())['helper_sha256']!=signature:raise ValueError('Summary helper changed after its first outcome read')
    else:write_json(freeze,dict(helper_sha256=signature,created_utc=datetime.now(timezone.utc).isoformat(),before_first_actual_score_read=True,
        best_map='max(means-only,identity,inverse,EACH eight shuffle)',best_all='B_map + step0 alignedC +oracle-count +one-line direct/aligned descents/recoils',within_model_probes_excluded=True))
    runs,sources=load_runs(args.results,args.protocol);result=summarize_runs(runs,args.out)
    result.update(completed_utc=datetime.now(timezone.utc).isoformat(),helper_sha256=signature,protocol_sha256=sha(args.protocol),source_table_sha256=sources)
    write_json(args.out/'completion.json',result);print(json.dumps(result))


if __name__=='__main__':main()
