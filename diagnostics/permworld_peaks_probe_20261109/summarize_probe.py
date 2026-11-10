"""Summarize completed, unchanged ridge-probe runs; never fits a probe/model."""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics

import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
ACTIONS=('identity','complement','inverse')
PHASES=('initial','trained')
KEYS=('encoding','phase','method','transform','label','support')


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def read_csv(path):
    with Path(path).open(newline='') as f:return list(csv.DictReader(f))


def write_csv(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fields=list(dict.fromkeys(k for r in rows for k in r)) or ['status']
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)


def fixed_mean(values,lengths):
    if set(values)!=set(lengths):raise ValueError('Missing or extra registered length')
    x=[float(values[n]) for n in lengths]
    if not all(math.isfinite(v) for v in x):raise ValueError('Undefined required length; no reweighting')
    return math.fsum(x)/len(lengths)


def contrast(c,i,v,c0):
    return dict(complement_minus_identity=c-i,complement_minus_inverse=c-v,
                complement_minus_best_alternative=c-max(i,v),trained_complement_minus_initial_complement=c-c0)


def describe(values):
    vals=list(values.values());n=len(vals)
    if not n or not all(math.isfinite(v) for v in vals):raise ValueError('Incomplete unit support')
    return dict(unit_count=n,mean=statistics.mean(vals),median=statistics.median(vals),
        sample_sd=statistics.stdev(vals) if n>1 else None,minimum=min(vals),maximum=max(vals),
        per_unit_values=';'.join(k+':'+repr(v) for k,v in values.items()))


def load_runs(config,base):
    complete=json.loads((base/'run_completion.json').read_text())
    assert complete['status']=='completed' and complete['stage']=='all'
    runs={};inputs={}
    for u in config['units']:
        unit=u['unit'];folder=base/'results'/unit
        receipt=json.loads((folder/'completion.json').read_text());assert receipt['status']=='completed'
        assert receipt['protocol_sha256']==sha(base/'protocol.json')
        for rel,digest in receipt['result_sha256'].items():
            p=folder/rel;assert sha(p)==digest;inputs[str(p)]=digest
        runs[unit]={k:read_csv(folder/(k+'.csv')) for k in ['summary','by_length','coverage','learning_by_length']}
    return runs,inputs


def summarize(config,runs,learning_rows,out,make_plots=True):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    units=[u['unit'] if isinstance(u,dict) else u for u in config['units']]
    historical=next((u['unit'] for u in config['units'] if isinstance(u,dict) and u.get('historical')),units[0])
    encodings=config['encodings'];ns=config['lengths'];score={};per_n={};all_summary=[];all_lengths=[];all_learning=[];coverage=[]
    for unit in units:
        run=runs[unit]
        for r in run['summary']:
            key=(unit,*[r[k] for k in KEYS]);assert key not in score
            assert int(r['finite_lengths'])==len(ns)
            score[key]=float(r['auc']);assert math.isfinite(score[key]);all_summary.append(r)
        for r in run['by_length']:
            key=(unit,*[r[k] for k in KEYS],int(r['length']));assert key not in per_n
            per_n[key]=r;all_lengths.append(r)
        for key,value in [(k,v) for k,v in score.items() if k[0]==unit]:
            parts={n:per_n[(*key,n)]['auc'] for n in ns}
            assert abs(fixed_mean(parts,ns)-value)<=1e-12,('Aggregate arithmetic mismatch',key)
        all_learning.extend(run['learning_by_length']);coverage.extend(run['coverage'])
    inventories=[{k[1:] for k in score if k[0]==u} for u in units]
    assert all(k==inventories[0] for k in inventories)
    def get(u,e,p,m,t,label='peak',support='matched'):return score[u,e,p,m,t,label,support]
    def length_get(u,e,p,m,t,n,label='peak',support='matched'):return float(per_n[u,e,p,m,t,label,support,n]['auc'])
    primary=[];paired=[];length_contrasts=[];directions=[];readouts=[]
    for unit in units:
        for e in encodings:
            c0=get(unit,e,'initial','transfer','complement')
            cc=get(unit,e,'trained','transfer','complement');ii=get(unit,e,'trained','transfer','identity');vv=get(unit,e,'trained','transfer','inverse')
            common=dict(unit=unit,encoding=e,support='matched',label='peak',length_count=len(ns))
            row=dict(**common,initial_complement_auc=c0,trained_complement_auc=cc,trained_identity_auc=ii,trained_inverse_auc=vv,**contrast(cc,ii,vv,c0))
            for name,method,action,label in [('source_peaks_within_auc','source_peaks','identity','peak'),('target_valleys_within_complement_auc','target_own','complement','own_valley'),('descents_complement_auc','descents','complement','peak'),('recoils_complement_auc','recoils','complement','peak'),('oracle_count_position_prior_auc','position_prior','identity','peak')]:
                row[name]=get(unit,e,'trained',method,action,label)
            row['source_within_minus_direct_complement']=row['source_peaks_within_auc']-cc
            row['target_within_minus_direct_complement']=row['target_valleys_within_complement_auc']-cc
            paired.append(row)
            for p in PHASES:
                for t in ACTIONS:
                    original=next(r for r in runs[unit]['summary'] if tuple(r[k] for k in KEYS)==(e,p,'transfer',t,'peak','matched'))
                    primary.append(dict(**common,phase=p,transform=t,auc=get(unit,e,p,'transfer',t),n=int(original['n']),defined_n=int(original['defined_n'])))
            local=[]
            for n in ns:
                cc=length_get(unit,e,'trained','transfer','complement',n);ii=length_get(unit,e,'trained','transfer','identity',n);vv=length_get(unit,e,'trained','transfer','inverse',n);c0=length_get(unit,e,'initial','transfer','complement',n)
                record=dict(unit=unit,encoding=e,length=n,initial_complement_auc=c0,trained_complement_auc=cc,trained_identity_auc=ii,trained_inverse_auc=vv,**contrast(cc,ii,vv,c0),matched_n=int(per_n[unit,e,'trained','transfer','complement','peak','matched',n]['n']))
                local.append(record);length_contrasts.append(record)
            dr=dict(**common,complement_above_identity_lengths=sum(r['complement_minus_identity']>0 for r in local),complement_above_inverse_lengths=sum(r['complement_minus_inverse']>0 for r in local),complement_above_both_lengths=sum(r['complement_minus_best_alternative']>0 for r in local),trained_above_initial_complement_lengths=sum(r['trained_complement_minus_initial_complement']>0 for r in local))
            dr['mean_lengthwise_C_minus_best']=statistics.mean(r['complement_minus_best_alternative'] for r in local)
            dr['aggregate_C_minus_best']=row['complement_minus_best_alternative'];directions.append(dr)
    for r in all_summary:
        if r['method']!='transfer' or r['label']=='own_valley':readouts.append(r)
    prior={(r['unit'],r['encoding'],r['task']):r for r in learning_rows if r.get('architecture')=='gru' and r.get('phase')=='trained' and r.get('split')=='test' and r.get('task') in ('peaks','valleys')}
    learn=[]
    for unit in units:
        for e in encodings:
            for task in ['peaks','valleys']:
                original=[r for r in runs[unit]['learning_by_length'] if r['encoding']==e and r['task']==task]
                acc=fixed_mean({int(r['length']):r['accuracy'] for r in original},ns);baseline=fixed_mean({int(r['length']):r['baseline'] for r in original},ns)
                reused=prior.get((unit,e,task));assert reused is not None,('Missing existing learning summary',unit,e,task)
                assert abs(acc-float(reused['exact_match_accuracy']))<=1e-12 and abs(baseline-float(reused['train_length_mode_accuracy']))<=1e-12
                learn.append(dict(unit=unit,encoding=e,task=task,accuracy=acc,length_mode_accuracy=baseline,accuracy_minus_length_mode=acc-baseline,full_nll_minus_length_prior_nll=float(reused['full_nll_minus_length_prior_nll']),expected_answer_spearman=float(reused['expected_answer_spearman']),reused_from='permworld_answer_controls_20261109/learning_summary.csv',learning_input='identity only',length_count=len(ns)))
    cohorts=[('new_units',[u for u in units if u!=historical]),('all_units',units)]
    measurements=[k for k in paired[0] if k not in ('unit','encoding','support','label','length_count')]
    cohort_rows=[]
    for cohort,selected in cohorts:
        for e in encodings:
            for field in measurements:
                vals={u:next(r[field] for r in paired if r['unit']==u and r['encoding']==e) for u in selected}
                ref=.5 if field.endswith('_auc') else 0.
                cohort_rows.append(dict(cohort=cohort,encoding=e,measure=field,**describe(vals),reference=ref,above_reference_units=sum(v>ref for v in vals.values()),below_reference_units=sum(v<ref for v in vals.values()),inference='Descriptive unit statistics; no length/anchor-as-replication test or CI'))
    learning_cohorts=[]
    for cohort,selected in cohorts:
        for e in encodings:
            for task in ['peaks','valleys']:
                for field in ['accuracy','length_mode_accuracy','accuracy_minus_length_mode','full_nll_minus_length_prior_nll','expected_answer_spearman']:
                    vals={u:next(r[field] for r in learn if r['unit']==u and r['encoding']==e and r['task']==task) for u in selected}
                    learning_cohorts.append(dict(cohort=cohort,encoding=e,task=task,measure=field,**describe(vals)))
    files={'all_scores.csv':all_summary,'all_by_length.csv':all_lengths,'all_learning_by_length.csv':all_learning,'coverage.csv':coverage,'primary_transfer_by_phase.csv':primary,'primary_contrasts_by_unit.csv':paired,'readout_controls.csv':readouts,'per_length_contrasts.csv':length_contrasts,'per_length_direction_counts.csv':directions,'learning_by_unit.csv':learn,'cohort_summary.csv':cohort_rows,'learning_cohort_summary.csv':learning_cohorts}
    for name,rows in files.items():write_csv(out/name,rows)
    if make_plots:plots(config,units,historical,paired,length_contrasts,out)
    report(units,historical,paired,cohort_rows,directions,learn,out)
    return dict(status='completed_saved_probe_summary',units=units,historical_unit=historical,registered_lengths=ns,summary_rows=len(all_summary),length_rows=len(all_lengths),primary_rows=len(primary),unit_contrast_rows=len(paired),no_probe_or_model_fitting=True,no_neural_forward=True,statistics='New-four and all-five mean/sampleSD/range; no new inferential CI or p-value')


def plots(config,units,historical,paired,per_length,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    encs=config['encodings'];ns=config['lengths'];colors=['C0','C1','C2','C3']
    def rows(e):return [next(r for r in paired if r['unit']==u and r['encoding']==e) for u in units]
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),constrained_layout=True)
    fields=['initial_complement_auc','trained_identity_auc','trained_complement_auc','trained_inverse_auc']
    for ax,e in zip(axes,encs):
        for j,f in enumerate(fields):ax.plot(np.arange(len(units))+(j-1.5)*.035,[r[f] for r in rows(e)],marker='o',label=f.replace('_auc','').replace('_',' '),color=colors[j])
        ax.axhline(.5,color='.5',linestyle='--');ax.set_xticks(range(len(units)),units,rotation=15);ax.set_title(e);ax.set_ylabel('Position AUROC, matched support');ax.grid(alpha=.2)
    axes[0].legend(fontsize=7);fig.suptitle('Same source probe across transforms within phase; initial phase refits its own probe')
    fig.savefig(out/'transfer_auroc_by_unit.png',dpi=170);plt.close(fig)
    fields=['complement_minus_identity','complement_minus_inverse','complement_minus_best_alternative','trained_complement_minus_initial_complement']
    fig,axes=plt.subplots(1,2,figsize=(13,4.5),constrained_layout=True)
    for ax,e in zip(axes,encs):
        for j,u in enumerate(units):
            row=next(r for r in rows(e) if r['unit']==u);ax.plot(np.arange(4)+(j-2)*.035,[row[f] for f in fields],marker='x' if u==historical else 'o',label=u,alpha=.8)
        ax.axhline(0,color='.5',linestyle='--');ax.set_xticks(range(4),['C−I','C−inverse','C−best alternative','Trained C−initial C'],rotation=14);ax.set_title(e);ax.set_ylabel('AUROC difference');ax.grid(alpha=.2)
    axes[1].legend(fontsize=7);fig.suptitle('Prespecified paired contrasts; each point is a unit, not an independent length')
    fig.savefig(out/'paired_contrasts.png',dpi=170);plt.close(fig)
    cycle='canonical_cycle';fig,axes=plt.subplots(2,3,figsize=(13,7),constrained_layout=True,sharex=True,sharey=True)
    for ax,u in zip(axes.flat,units):
        selected=sorted([r for r in per_length if r['unit']==u and r['encoding']==cycle],key=lambda r:r['length'])
        for j,f in enumerate(['initial_complement_auc','trained_identity_auc','trained_complement_auc','trained_inverse_auc']):ax.plot(ns,[r[f] for r in selected],marker='.',markersize=3,label=f.replace('_auc','').replace('_',' '),color=colors[j])
        ax.axhline(.5,color='.5',linestyle='--');ax.set_title(u+(' (historical)' if u==historical else ''));ax.grid(alpha=.2);ax.set_xlabel('Permutation length');ax.set_ylabel('Matched position AUROC')
    axes.flat[-1].axis('off');axes.flat[0].legend(fontsize=6);fig.suptitle('Canonical cycle: all registered lengths retained; lengths are dependent conditions')
    fig.savefig(out/'cycle_per_length.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),constrained_layout=True)
    selected=rows('one_line')
    for j,f in enumerate(['source_peaks_within_auc','target_valleys_within_complement_auc','trained_complement_auc']):axes[0].plot(range(len(units)),[r[f] for r in selected],marker='o',label=f.replace('_auc','').replace('_',' '),color=colors[j])
    axes[0].axhline(.5,color='.5',linestyle='--');axes[0].set_ylabel('Matched position AUROC');axes[0].legend(fontsize=6)
    for j,f in enumerate(['source_within_minus_direct_complement','target_within_minus_direct_complement']):axes[1].plot(range(len(units)),[r[f] for r in selected],marker='o',label=f.replace('_',' '),color=colors[j])
    axes[1].axhline(0,color='.5',linestyle='--');axes[1].set_ylabel('Within-model minus direct-transfer AUROC');axes[1].legend(fontsize=6)
    for ax in axes:ax.set_xticks(range(len(units)),units,rotation=15);ax.grid(alpha=.2)
    fig.suptitle('One-line: available within-model position readout versus unaligned cross-model transfer')
    fig.savefig(out/'one_line_readout_contrast.png',dpi=170);plt.close(fig)


def report(units,historical,paired,cohorts,directions,learning,out):
    fmt=lambda v:f'{float(v):.6f}'
    text=['# Saved position-probe results','','The historical pilot and four newer independent units use the supplied probe protocol unchanged. All 21 lengths are required. The primary quantity is an equal-length mean of within-example position AUROC on each unit\'s own common-count test support. Models and encodings share these anchors within a unit; they are not new independent observations.','', '| Unit | Encoding | Initial C | Trained C | Trained I | Trained inverse | C−best alternative | Trained C−initial C |','|---|---|---:|---:|---:|---:|---:|---:|']
    for r in paired:text.append('| '+r['unit']+' | '+r['encoding']+' | '+' | '.join(fmt(r[k]) for k in ['initial_complement_auc','trained_complement_auc','trained_identity_auc','trained_inverse_auc','complement_minus_best_alternative','trained_complement_minus_initial_complement'])+' |')
    text+=['','## Unit-level summaries','','| Cohort | Encoding | Contrast | Mean | Sample SD | Minimum | Maximum | Positive units |','|---|---|---|---:|---:|---:|---:|---:|']
    for r in cohorts:
        if r['measure'] not in ['complement_minus_best_alternative','trained_complement_minus_initial_complement']:continue
        text.append('| '+r['cohort']+' | '+r['encoding']+' | '+r['measure']+' | '+' | '.join(fmt(r[k]) for k in ['mean','sample_sd','minimum','maximum'])+f" | {r['above_reference_units']}/{r['unit_count']} |")
    text+=['','No significance/equivalence test or confidence interval is introduced. The four newer units are the new repetition units; the historical pilot is exploratory and is retained separately. One task pair cannot establish generality across mathematical relationships.','','## Per-length directions','','| Unit | Encoding | C>I | C>inverse | C>both | Trained C>initial C |','|---|---|---:|---:|---:|---:|']
    for r in directions:text.append('| '+r['unit']+' | '+r['encoding']+' | '+' | '.join(str(r[k])+'/21' for k in ['complement_above_identity_lengths','complement_above_inverse_lengths','complement_above_both_lengths','trained_above_initial_complement_lengths'])+' |')
    text+=['','The primary best-alternative contrast takes the maximum **after** equal-length aggregation. It is not the mean of lengthwise maxima; both values are explicitly retained in the direction table. Lengthwise direction counts are descriptive.','','## Identity-input specialist learning','','| Unit | Encoding | Task | Count accuracy | Train-length mode | Accuracy gap | Full NLL gap | Expected-answer Spearman |','|---|---|---|---:|---:|---:|---:|---:|']
    for r in learning:text.append('| '+r['unit']+' | '+r['encoding']+' | '+r['task']+' | '+' | '.join(fmt(r[k]) for k in ['accuracy','length_mode_accuracy','accuracy_minus_length_mode','full_nll_minus_length_prior_nll','expected_answer_spearman'])+' |')
    text+=['','Full NLL and expected-answer Spearman are reused from the completed existing-checkpoint diagnostic, not newly measured by this probe. These identity-input quantities distinguish modest input-associated predictive information from exact-mode-only behavior; neither near-mode accuracy nor a small NLL difference proves baseline equivalence or task mastery. NLL alone can reflect probability calibration.','','## Readout interpretation','','The same source probe is applied across target models/actions within each phase, length and encoding. Initial and trained phases independently refit and retune their probes. Weak direct transfer can coexist with strong within-model readout because transfer depends on the coordinate basis. Successful decoding does not show that the counting computation uses decoded position features. The position-prior baseline conditions on the true test count and is an oracle-count control, not a length-only baseline. Initial/source/target own-label controls, all-anchor support, non-designated specialist controls, alphas and every negative or mixed result remain in the CSVs. No target-space alignment was fitted.','', '## Figures','','![Transfer scores](transfer_auroc_by_unit.png)','','![Paired contrasts](paired_contrasts.png)','','![Cycle lengths](cycle_per_length.png)','','![One-line readout contrast](one_line_readout_contrast.png)','']
    (out/'report.md').write_text('\n'.join(text))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--protocol',type=Path,default=HERE/'protocol.json');parser.add_argument('--out',type=Path,default=HERE/'summary');args=parser.parse_args()
    config=json.loads(args.protocol.read_text());runs,inputs=load_runs(config,args.protocol.parent)
    previous=ROOT/'diagnostics/permworld_answer_controls_20261109/learning_summary.csv';learning=read_csv(previous)
    result=summarize(config,runs,learning,args.out)
    result.update(completed_utc=datetime.now(timezone.utc).isoformat(),summary_helper_sha256=sha(Path(__file__)),protocol_sha256=sha(args.protocol),source_table_sha256=inputs,reused_learning_sha256=sha(previous))
    (args.out/'completion.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
