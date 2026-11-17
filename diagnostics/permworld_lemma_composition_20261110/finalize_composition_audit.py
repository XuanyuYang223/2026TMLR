"""Bounded independent checks of frozen readouts, masks, answers and files.

This helper performs no fit, forward pass, calibration change or new inference
procedure. It verifies saved artifacts and named source identities only.
"""
from pathlib import Path
from datetime import datetime,timezone
from concurrent.futures import ThreadPoolExecutor,as_completed
import argparse
import csv
import hashlib
import inspect
import json
import math
import subprocess
import sys
import time

import numpy as np
from threadpoolctl import threadpool_limits

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT))
from experiments.permworld_lemma_composition_20261110.combiners import compose

TOL=1e-12
TASKS=('longest','fixed_longest','turns_before_fixed')
MAIN_ROUTES=('same_model_twice','aligned_complement','direct_transfer','own_probes')


def read(path):return json.loads(Path(path).read_text())
def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def array_sha(value):return hashlib.sha256(np.asarray(value).tobytes(order='C')).hexdigest()
def write(path,value):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)
def csv_rows(path):
    with Path(path).open(newline='') as stream:return list(csv.DictReader(stream))
def number(v):
    if v is None or v in ('','None','nan','NaN'):return None
    v=float(v);return v if math.isfinite(v) else None


class Checks:
    def __init__(self):self.count=0;self.failures=[];self.maximum=0.
    def ok(self,value,label,detail=None):
        self.count+=1
        if not bool(value):self.failures.append(dict(label=label,detail=detail))
    def close(self,actual,expected,label):
        a,b=number(actual),number(expected)
        self.count+=1
        if a is None or b is None:
            if a is not None or b is not None:self.failures.append(dict(label=label,actual=a,expected=b))
        else:
            error=abs(a-b);self.maximum=max(self.maximum,error)
            if error>TOL:self.failures.append(dict(label=label,actual=a,expected=b,error=error))


def independent_threshold(scores,truth):
    """Ascending candidate scan independent of the producer's descending scan."""
    x=np.asarray(scores,np.float64).ravel();y=np.asarray(truth,bool).ravel()
    if not y.any():return float('inf'),0.,0
    order=np.argsort(x,kind='stable');s=x[order];yy=y[order]
    values,first=np.unique(s,return_index=True)
    prefix=np.r_[0,np.cumsum(yy,dtype=np.int64)]
    positive=int(y.sum());predicted=len(x)-first;tp=positive-prefix[first]
    f1=2*tp/(positive+predicted)
    best=np.flatnonzero(f1==f1.max())[-1]
    return float(values[best]),float(f1[best]),int(predicted[best])


def orbit_key(row):
    p=np.asarray(row);n=len(p)
    half=[p,p[::-1],n+1-p,n+1-p[::-1]]
    orbit=half+[np.argsort(v)+1 for v in half]
    return min(np.asarray(v,np.uint8).tobytes() for v in orbit)


def split_indices(data,n):
    selections={s:np.flatnonzero(data[s+'_lengths']==n) for s in ('train','validation','test')}
    keys={s:[orbit_key(row[:n]) for row in data[s+'_permutations'][idx]] for s,idx in selections.items()}
    prohibited=set(keys['train'])|set(keys['test'])
    tuning=np.array([hashlib.sha256(b'peaks-probe-v1'+key).digest()[0]<51 for key in keys['validation']])
    valid=np.array([key not in prohibited for key in keys['validation']])
    return selections['validation'][valid&~tuning],selections['validation'][valid&tuning]


def source_labels(perms,task,n):
    p=np.asarray(perms)[:,:n]
    if task=='fixed_points':return p==np.arange(1,n+1)
    if task=='peaks':return (p[:,1:-1]>p[:,:-2])&(p[:,1:-1]>p[:,2:])
    return (p[:,1:-1]<p[:,:-2])&(p[:,1:-1]<p[:,2:])


def brute_oracle(permutation):
    """Enumerate every interval, independent of sign-run/mask algorithms."""
    p=list(map(int,permutation));n=len(p);fixed=[j for j,v in enumerate(p,1) if j==v]
    monotone=[]
    for a in range(n):
        for b in range(a,n):
            differences=[p[j+1]-p[j] for j in range(a,b)]
            if all(v>0 for v in differences) or all(v<0 for v in differences):monotone.append((a+1,b+1))
    key=lambda interval:(-(interval[1]-interval[0]+1),interval[0])
    longest=min(monotone,key=key)
    if not fixed:return dict(longest=longest,fixed_longest=None,turns_before_fixed=None)
    eligible=[v for v in monotone if any(v[0]<=j<=v[1] for j in fixed)]
    turning=sum((p[j]>p[j-1] and p[j]>p[j+1]) or (p[j]<p[j-1] and p[j]<p[j+1]) for j in range(1,min(fixed[0]-1,n-1)))
    return dict(longest=longest,fixed_longest=min(eligible,key=key),turns_before_fixed=turning)


def expected_sample_stats(task,truth,pred,ns,selection):
    t=truth[selection];p=pred[selection];n=np.asarray(ns)[selection];size=len(n)
    if task=='turns_before_fixed':tp=t>=0;pp=p>=0;exact=t==p
    else:tp=t[:,0]>0;pp=p[:,0]>0;exact=(t==p).all(1)
    joint=tp&pp;mismatch=tp!=pp
    result=dict(n=size,exact_n=int(exact.sum()),truth_none_n=int((~tp).sum()),prediction_none_n=int((~pp).sum()),
        both_none_n=int((~tp&~pp).sum()),missed_presence_n=int((tp&~pp).sum()),false_presence_n=int((~tp&pp).sum()),joint_present_n=int(joint.sum()),truth_present_n=int(tp.sum()))
    for name,vector in [('exact_accuracy',exact),('truth_none_rate',~tp),('prediction_none_rate',~pp),('joint_present_coverage',joint)]:result[name]=float(vector.mean()) if size else None
    result['prediction_present_given_truth_present']=float(joint.sum()/tp.sum()) if tp.any() else None
    if task=='turns_before_fixed':
        err=np.abs(p.astype(float)-t)
        result['count_mae']=float(err[joint].mean()) if joint.any() else None
        result['normalized_count_error']=float(np.where(mismatch,1.,np.where(joint,err/n,0.)).mean()) if size else None
    else:
        starts=np.abs(p[:,0].astype(float)-t[:,0]);ends=np.abs(p[:,1].astype(float)-t[:,1])
        length=np.abs((p[:,1]-p[:,0]).astype(float)-(t[:,1]-t[:,0]))
        for name,v in [('start_mae',starts),('end_mae',ends),('endpoint_mae',(starts+ends)/2),('length_mae',length)]:result[name]=float(v[joint].mean()) if joint.any() else None
        result['normalized_endpoint_error']=float(np.where(mismatch,1.,np.where(joint,(starts+ends)/(2*n),0.)).mean()) if size else None
        result['normalized_length_error']=float(np.where(mismatch,1.,np.where(joint,length/n,0.)).mean()) if size else None
    return result


def unit_checks(config,u,checks,expected):
    unit=u['unit'];base=Path(config['output'])/'units'/unit;folder=HERE/'results'/unit;receipt=read(folder/'completion.json')
    checks.ok(receipt['status']=='completed' and receipt['protocol_sha256']==sha(HERE/'protocol.json'),unit+'/completion_binding')
    expected[str(folder/'completion.json')]=sha(folder/'completion.json')
    for desc in receipt['artifacts'].values():expected[desc['path']]=desc['sha256']
    with np.load(u['data_path'],allow_pickle=False) as z:data={k:z[k].copy() for k in z.files if k.endswith(('_lengths','_permutations'))}
    with np.load(base/'signals.npz',allow_pickle=False) as z:signals=dict(z)
    with np.load(base/'frozen_readouts.npz',allow_pickle=False) as z:params=dict(z)
    with np.load(base/'prediction_only.npz',allow_pickle=False) as z:prediction=dict(z)
    with np.load(base/'truth.npz',allow_pickle=False) as z:truth=dict(z)
    with np.load(base/'predictions.npz',allow_pickle=False) as z:combined=dict(z)
    with np.load(base/'mask_shuffle_permutations.npz',allow_pickle=False) as z:saved_perms=dict(z)
    ns=signals['lengths'];routes=[str(v) for v in signals['route_names']];variants=[str(v) for v in prediction['variant_names']]
    checks.ok(routes==config['routes'] and variants==config['variants'],unit+'/route_variant_inventory')
    checks.ok(np.array_equal(ns,data['test_lengths']),unit+'/anchor_length_order')
    checks.ok(sha(base/'prediction_only.npz')==receipt['prediction_only_sha_before_truth'],unit+'/prediction_before_truth_SHA')
    checks.ok((base/'prediction_only.npz').stat().st_mtime_ns<=(base/'truth.npz').stat().st_mtime_ns<=(base/'predictions.npz').stat().st_mtime_ns,unit+'/documented_file_order')
    checks.ok(set(prediction).isdisjoint({'truth_longest','truth_fixed_longest','truth_turns_before_fixed','matched'}),unit+'/prediction_only_has_no_truth_fields')
    for key,value in prediction.items():checks.ok(np.array_equal(value,combined[key]),unit+'/frozen_prediction_copy/'+key)
    for key,value in truth.items():
        if key in combined:checks.ok(np.array_equal(value,combined[key]),unit+'/truth_copy/'+key)
    calibration=read(folder/'source_calibration.json');expected[str(folder/'source_calibration.json')]=sha(folder/'source_calibration.json')
    cal={(r['task'],int(r['length'])):r for r in calibration['records']};checks.ok(len(cal)==63,unit+'/63_source_calibrations')
    thresholds={};oracle_samples=[];composer_samples=0
    for n in config['lengths']:
        ids=np.flatnonzero(ns==n);checks.ok(len(ids)==512,unit+f'/n{n}/test_count')
        fit,dev=split_indices(data,n)
        for task in ['peaks','valleys','fixed_points']:
            r=cal[task,n];prefix=f'{task}_n{n}_';tuning=params[prefix+'source_tuning_scores'];target=params[prefix+'source_tuning_targets']
            checks.ok(np.array_equal(params[prefix+'source_fit_indices'],fit),unit+prefix+'source_fit_ids')
            checks.ok(np.array_equal(params[prefix+'source_tuning_indices'],dev),unit+prefix+'source_tuning_ids')
            checks.ok(np.array_equal(target,source_labels(data['validation_permutations'][dev],task,n)),unit+prefix+'source_own_identity_labels')
            checks.ok(array_sha(tuning)==r['source_tuning_scores_sha256'] and array_sha(target)==r['source_tuning_targets_sha256'],unit+prefix+'tuning_hashes')
            for name in ['coef','intercept','mean','scale']:checks.ok(array_sha(params[prefix+name])==r['probe_parameter_hashes'][name],unit+prefix+name+'_hash')
            tau,f1,predicted=independent_threshold(tuning,target);info=r['threshold_info'];reported=float('inf') if info['threshold']=='Infinity' else float(info['threshold'])
            checks.ok(tau==reported,unit+prefix+'independent_threshold_argmax')
            checks.close(info['source_micro_f1'],f1,unit+prefix+'threshold_F1')
            checks.ok(int(info['source_predicted_positions'])==predicted,unit+prefix+'threshold_source_count')
            checks.ok(r['source_validation_action']==0 and r['uses_downstream_answers'] is False,unit+prefix+'source_only_binding')
            thresholds[task,n]=tau
        checks.ok(np.array_equal(signals['peak_mask'][ids,1:n-1],signals['peak_scores'][ids,1:n-1]>=thresholds['peaks',n]),unit+f'/n{n}/peaks_shared_tau')
        checks.ok(np.array_equal(signals['fixed_mask'][ids,:n],signals['fixed_scores'][ids,:n]>=thresholds['fixed_points',n]),unit+f'/n{n}/fixed_shared_tau')
        checks.ok(not signals['peak_mask'][ids,0].any() and not signals['peak_mask'][ids,n-1:].any(),unit+f'/n{n}/peak_endpoints_padding')
        checks.ok(not signals['fixed_mask'][ids,n:].any(),unit+f'/n{n}/FP_padding')
        for ri,route in enumerate(routes):
            tau=thresholds['valleys' if route=='own_probes' else 'peaks',n]
            checks.ok(np.array_equal(signals['valley_masks'][ri,ids,1:n-1],signals['valley_scores'][ri,ids,1:n-1]>=tau),unit+f'/n{n}/{route}/frozen_tau')
            checks.ok(np.array_equal(signals['turn_masks'][ri,ids,:n],signals['peak_mask'][ids,:n]|signals['valley_masks'][ri,ids,:n]),unit+f'/n{n}/{route}/turn_union')
            checks.ok(not signals['turn_masks'][ri,ids,0].any() and not signals['turn_masks'][ri,ids,n-1:].any(),unit+f'/n{n}/{route}/real_interior_turns')
        for i in ids:
            actual=n+1-data['test_permutations'][i,:n]
            checks.ok(np.array_equal(truth['pi'][i,:n],actual),unit+f'/anchor{i}/pi_Cx')
        for seed in config['mask_shuffle_seeds']:
            for stream,kind,width in [(0,'turn',n-2),(1,'fixed',n)]:
                key=f'{kind}_n{n}_s{seed}';p=saved_perms[key]
                rng=np.random.default_rng(np.random.SeedSequence([seed,1 if unit=='archdev001' else int(unit[-2:]),n,stream]))
                independent=rng.permuted(np.broadcast_to(np.arange(width,dtype=np.int16),(len(ids),width)),axis=1)
                checks.ok(np.array_equal(p,independent),unit+'/'+key+'/fixed_RNG')
                checks.ok(np.array_equal(np.sort(p,axis=1),np.broadcast_to(np.arange(width),p.shape)),unit+'/'+key+'/row_bijections')
        samples=ids[:4];oracle_samples.extend(int(v) for v in samples)
        for ri,route in enumerate(routes):
            turn=signals['turn_masks'][ri,ids,:n];fixed=signals['fixed_mask'][ids,:n]
            for vi,variant in enumerate(variants):
                tm,fm=turn,fixed
                if variant.startswith('turn_shuffle_'):
                    p=saved_perms[f'turn_n{n}_s{variant.rsplit("_",1)[1]}'];tm=turn.copy();tm[:,1:-1]=np.take_along_axis(turn[:,1:-1],p,axis=1)
                    checks.ok(np.array_equal(tm.sum(1),turn.sum(1)),unit+f'/{n}/{route}/{variant}/predicted_turn_count')
                elif variant.startswith('fixed_shuffle_'):
                    p=saved_perms[f'fixed_n{n}_s{variant.rsplit("_",1)[1]}'];fm=np.take_along_axis(fixed,p,axis=1)
                    checks.ok(np.array_equal(fm.sum(1),fixed.sum(1)),unit+f'/{n}/{route}/{variant}/predicted_FP_count')
                    checks.ok(np.array_equal(prediction['pred_longest'][ri,vi,ids],prediction['pred_longest'][ri,0,ids]),unit+f'/{n}/{route}/{variant}/FPshuffle_ignores_E1')
                for local,i in enumerate(samples):
                    scalar=compose(tm[local],fm[local],int(n));composer_samples+=1
                    for task,value in scalar.items():
                        expected_answer=(-1 if task=='turns_before_fixed' else (-1,-1)) if value is None else value
                        checks.ok(np.array_equal(prediction['pred_'+task][ri,vi,i],expected_answer),unit+f'/{i}/{route}/{variant}/{task}/scalar_combiner')
    for i in oracle_samples:
        n=int(ns[i]);answer=brute_oracle(truth['pi'][i,:n])
        for task,value in answer.items():
            sentinel=(-1 if task=='turns_before_fixed' else (-1,-1))
            checks.ok(np.array_equal(truth['truth_'+task][i],sentinel if value is None else value),unit+f'/anchor{i}/{task}/brute_interval_oracle')
    none=prediction['pred_fixed_longest'][...,0]<0
    checks.ok(np.array_equal(none,prediction['pred_turns_before_fixed']<0),unit+'/both_FP_tasks_same_NONE')
    checks.ok(np.all(none==none[:,0:1,:]) and np.all(none[:,0]==none[0,0]),unit+'/allroutes_shuffles_same_FP_absence')
    absent=~truth['truth_fixed_present']
    for task in ['fixed_longest','turns_before_fixed']:
        p=prediction['pred_'+task];t=truth['truth_'+task]
        exact=(p==t).all(-1) if task=='fixed_longest' else p==t
        checks.ok(np.all(exact[...,absent]==exact[:,0:1,absent]),unit+'/'+task+'/truth_NONE_accuracy_invariant')
    upstream=read(folder/'upstream_reproduction.json');expected[str(folder/'upstream_reproduction.json')]=sha(folder/'upstream_reproduction.json')
    old={}
    for r in csv_rows(ROOT/'diagnostics/permworld_orthogonal_alignment_20261110/results'/unit/'by_length.csv'):
        if r['encoding']=='one_line' and r['phase']=='trained':old[int(r['length']),r['method'],r['support']]=float(r['auc'])
    methods={'same_model_twice':'within_peaks','aligned_complement':'fit_complement__test_complement','direct_transfer':'direct','means_only':'means_only','map_identity':'fit_identity__test_complement','map_inverse':'fit_inverse__test_complement','own_probes':'within_valleys',**{f'map_shuffle_{s}':f'fit_shuffle_{s}__test_complement' for s in range(1026,1034)}}
    checks.ok(upstream['status']=='passed' and len(upstream['checks'])==630,unit+'/630_reused_upstream_scores')
    for r in upstream['checks']:
        previous=old[int(r['length']),methods[r['route']],r['support']]
        checks.close(r['previous'],previous,unit+'/upstream_source_scalar')
        checks.close(r['actual'],previous,unit+'/reused_probe_map_AUROC')
    summary_checks(config,unit,combined,checks)
    return dict(unit=unit,test_anchors=len(ns),matched_anchors=int(combined['matched'].sum()),
        source_calibration_records=63,independent_thresholds_verified=63,
        fixed_oracle_anchors=len(oracle_samples),scalar_combiner_evaluations=composer_samples,
        shuffle_position_arrays=len(saved_perms),prediction_only_frozen_before_truth=True)


def summary_checks(config,unit,arrays,checks):
    ns=arrays['lengths'];route_names=list(map(str,arrays['route_names']));original=list(map(str,arrays['variant_names'])).index('original')
    needed={}
    file=HERE/'summary/per_length'/f'{unit}.csv'
    with file.open(newline='') as stream:
        for row in csv.DictReader(stream):
            if row['variant']=='original' and row['route'] in MAIN_ROUTES:
                needed[row['route'],row['task'],row['support'],row['stratum'],int(row['length'])]=row
    generated={}
    for route in MAIN_ROUTES:
        ri=route_names.index(route)
        for task in TASKS:
            truth=arrays['truth_'+task];pred=arrays['pred_'+task][ri,original]
            strata={'all':np.ones(len(ns),bool)}
            if task!='longest':strata.update(fixed_present=arrays['truth_fixed_present'],fixed_absent=~arrays['truth_fixed_present'])
            if task=='fixed_longest':strata['constraint_changes']=arrays['truth_constraint_changes']
            for support,supportmask in [('all',arrays.get('eligible',np.ones(len(ns),bool))),('matched',arrays['matched'])]:
                for stratum,mask in strata.items():
                    metric_rows=[]
                    for n in config['lengths']:
                        expected=expected_sample_stats(task,truth,pred,ns,(ns==n)&supportmask&mask)
                        row=needed[route,task,support,stratum,n]
                        for key,value in expected.items():checks.close(row[key],value,f'{unit}/{route}/{task}/{support}/{stratum}/{n}/{key}')
                        metric_rows.append(expected)
                    generated[route,task,support,stratum]=metric_rows
    aggregate={}
    with (HERE/'summary/per_unit_metrics.csv').open(newline='') as stream:
        for row in csv.DictReader(stream):
            if row['unit']==unit and row['variant']=='original' and row['route'] in MAIN_ROUTES:aggregate[row['route'],row['task'],row['support'],row['stratum']]=row
    for key,rr in generated.items():
        row=aggregate[key];checks.ok(int(row['length_count'])==21,unit+'/summary21lengths')
        for field in rr[0]:
            values=[r[field] for r in rr]
            expected=sum(values) if field=='n' or field.endswith('_n') else math.fsum(values)/21 if all(v is not None for v in values) else None
            checks.close(row[field],expected,unit+'/independent_summary/'+str(key)+'/'+field)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=2);args=parser.parse_args()
    config=read(HERE/'protocol.json');freeze=read(HERE/'freeze.json');start=read(HERE/'protection_start.json')
    if not (HERE/'summary/completion.json').exists():raise SystemExit('Saved summary is pending; run this audit after all five summaries complete.')
    assert read(HERE/'run_completion.json')['status']=='completed'
    assert read(HERE/'summary/completion.json')['status']=='completed_saved_composition_summary'
    began=datetime.now(timezone.utc).isoformat();clock=time.monotonic();checks=Checks();expected=dict(start['files'])
    checks.ok(list(inspect.signature(compose).parameters)==['turning_mask','fixed_mask','n'],'mask_only_scalar_composer_interface')
    expected[str(HERE/'protocol.json')]=freeze['protocol_sha256']
    for p,d in freeze['source_sha256'].items():expected[str(ROOT/p)]=d
    report=[]
    for u in config['units']:report.append(unit_checks(config,u,checks,expected))
    print(json.dumps(dict(event='numeric_checks_finished',checks=checks.count,failures=len(checks.failures))),flush=True)
    def hash_one(item):
        path,digest=item;p=Path(path);before=p.stat();actual=sha(p);after=p.stat()
        return dict(path=path,expected_sha256=digest,actual_sha256=actual,size_bytes=after.st_size,
            passed=actual==digest and (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns))
    files=[]
    with ThreadPoolExecutor(max_workers=max(1,args.workers)) as pool:
        pending=[pool.submit(hash_one,item) for item in sorted(expected.items())]
        for future in as_completed(pending):files.append(future.result())
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    source_files=[r for r in files if r['path'] in start['files']]
    protected=dict(status='passed' if all(r['passed'] for r in files) and head==start['git_head'] else 'failed',
        started_utc=began,completed_utc=datetime.now(timezone.utc).isoformat(),
        scope='Endpoint SHA of explicitly named prior inputs/results/CP/caches, frozen source and saved artifacts; not all files or continuous monitoring.',
        protected_start_files=len(source_files),files_hashed=len(files),bytes_hashed=sum(r['size_bytes'] for r in files),
        git_head_start=start['git_head'],git_head_end=head,git_head_unchanged=head==start['git_head'],
        failed_files=[r for r in files if not r['passed']],files=sorted(files,key=lambda r:r['path']),
        no_neural_forward=True,no_probe_fitting=True,no_alignment_fitting=True,no_old_file_writes=True)
    saved=dict(status='passed' if not checks.failures else 'failed',completed_utc=datetime.now(timezone.utc).isoformat(),
        scope='Independent source-only threshold/label checks, saved mask/answer algebra, fixed-subset scalar composers and brute-force oracle, bounded primary summary arithmetic. No new fits or inferential procedure.',
        checks=checks.count,maximum_arithmetic_difference=checks.maximum,arithmetic_tolerance=TOL,failures=checks.failures,units=report,
        primary_summary_scope='Original variants of same-model-twice, aligned, direct and own-probe routes; all registered supports/strata/lengths. Other routes/shuffles verified structurally and on fixed first4anchors of each length.',
        scalar_anchor_policy='First four test anchors at each registered length, fixed independently of predictions or correctness.',
        threshold_independent_scan='All unique source-tuning scores scanned ascending with higher-threshold ties; raw source validation identity labels verified.',
        no_neural_forward=True,no_probe_fitting=True,no_alignment_fitting=True,no_threshold_changes=True,no_hyperparameter_changes=True,
        elapsed_seconds=time.monotonic()-clock)
    write(HERE/'protection_end.json',protected);write(HERE/'saved_results_audit.json',saved)
    print(json.dumps(dict(protection_status=protected['status'],saved_results_status=saved['status'],checks=checks.count,failures=len(checks.failures),maximum_error=checks.maximum,files_hashed=len(files),elapsed_seconds=saved['elapsed_seconds'])),flush=True)
    if protected['status']!='passed' or saved['status']!='passed':raise SystemExit('Audit failed; inspect failures without changing scientific results or tolerances.')


if __name__=='__main__':
    with threadpool_limits(limits=2):main()
