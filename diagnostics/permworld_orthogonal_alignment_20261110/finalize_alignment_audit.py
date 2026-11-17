"""Bounded end audit of saved alignment results, maps, permutations and inputs.

No probe, map, neural model or activation metric is fitted/recomputed here.
The checks reconstruct only deterministic split masks, labels and shuffle
permutations; numerical scoring is verified from the saved scalar tables.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time
from zipfile import ZipFile

import numpy as np
from threadpoolctl import threadpool_limits

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
PACKAGE=ROOT/'experiments/permworld_orthogonal_alignment_20261110'
PRIOR=ROOT/'diagnostics/permworld_peaks_probe_20261109'
TOL=1e-12


def utc():return datetime.now(timezone.utc).isoformat()
def read(path):return json.loads(Path(path).read_text())
def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def array_sha(a):return hashlib.sha256(np.asarray(a).tobytes(order='C')).hexdigest()
def csv_rows(path):
    with Path(path).open(newline='') as stream:return list(csv.DictReader(stream))
def write(path,value):
    p=Path(path);tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(p)


class Checks:
    def __init__(self):self.total=0;self.failures=[];self.maximum_arithmetic_difference=0.
    def require(self,value,name,**context):
        self.total+=1
        if not bool(value):self.failures.append(dict(check=name,**context))
    def close(self,a,b,name,tol=TOL,**context):
        a,b=float(a),float(b)
        error=abs(a-b) if np.isfinite(a) and np.isfinite(b) else None
        if error is not None:self.maximum_arithmetic_difference=max(self.maximum_arithmetic_difference,error)
        self.require(error is not None and error<=tol,name,actual=a if np.isfinite(a) else None,
                     expected=b if np.isfinite(b) else None,error=error,tolerance=tol,**context)


def extrema(p):
    return ((p[:,1:-1]>p[:,:-2])&(p[:,1:-1]>p[:,2:]),
            (p[:,1:-1]<p[:,:-2])&(p[:,1:-1]<p[:,2:]))


def orbit_keys(p):
    n=p.shape[1];half=[p,p[:,::-1],n+1-p,n+1-p[:,::-1]]
    all_actions=half+[np.argsort(q,axis=1)+1 for q in half]
    packed=np.stack(all_actions,axis=1).astype(np.uint8)
    return [min(q.tobytes() for q in row) for row in packed]


def independent_splits(unit,lengths,check):
    with np.load(unit['data_path'],allow_pickle=False) as z:
        data={k:z[k] for k in z.files if k.endswith(('_lengths','_permutations'))}
    splits={}
    for n in lengths:
        ps={s:data[s+'_permutations'][data[s+'_lengths']==n,:n] for s in ['train','validation','test']}
        keys={s:orbit_keys(p) for s,p in ps.items()}
        train,test=set(keys['train']),set(keys['test'])
        valid=np.array([x not in train and x not in test for x in keys['validation']])
        dev=np.array([hashlib.sha256(b'peaks-probe-v1'+x).digest()[0]<51 for x in keys['validation']])
        fit=valid&~dev;dev=valid&dev
        eligible=np.array([x not in train for x in keys['test']])
        pf,vf=extrema(ps['validation']);peak,valley=extrema(ps['test'])
        vc=extrema(n+1-ps['test'])[1];vi=extrema(np.argsort(ps['test'],axis=1)+1)[1]
        count=peak.sum(1);matched=eligible&(valley.sum(1)==count)&(vi.sum(1)==count)
        check.require(np.array_equal(vc,peak),'complement_peak_position_identity',unit=unit['unit'],length=n)
        check.require(len(peak)==512 and len(pf)==256,'registered_length_anchor_count',unit=unit['unit'],length=n)
        check.require(min(int(fit.sum()),int(dev.sum()),int(matched.sum()))>0,'nonempty_fixed_support',unit=unit['unit'],length=n)
        splits[n]=dict(fit=fit,dev=dev,eligible=eligible,matched=matched,
            fit_counts=pf[fit].sum(1),peak=peak,valleys=[valley,vc,vi],
            eligible_n=int(eligible.sum()),matched_n=int(matched.sum()),fit_n=int(fit.sum()))
    return data,splits


def score_label(method,split):
    if method.endswith('__test_native_own_valley'):
        action=method.split('__')[0][len('fit_'):]
        return split['valleys'][['identity','complement','inverse'].index(action)]
    if method=='within_valleys':return split['valleys'][1]
    return split['peak']


def fixed_rows(summary,detail,keys,lengths,check,context,split_info=None):
    grouped=defaultdict(list)
    for r in detail:grouped[tuple(r[k] for k in keys)].append(r)
    check.require(len({tuple(r[k] for k in keys) for r in summary})==len(summary),'unique_summary_keys',**context)
    check.require(len(grouped)==len(summary),'detail_summary_inventory',**context)
    for r in summary:
        key=tuple(r[k] for k in keys);items=grouped.get(key,[])
        check.require(len(items)==21 and {int(x['length']) for x in items}==set(lengths),'exact21length_support',key=key,**context)
        if len(items)!=21:continue
        values=[float(x['auc']) for x in items]
        check.require(all(np.isfinite(v) for v in values),'finite_length_scores',key=key,**context)
        check.require(int(r['lengths'])==21,'finite_lengths_receipt',key=key,**context)
        check.close(r['auc'],math.fsum(values)/21,'independent_equal_length_aggregate',key=key,**context)
        if 'anchors' in r:
            check.require(int(r['anchors'])==sum(int(x['n']) for x in items),'aggregate_anchor_sum',key=key,**context)
            check.require(int(r['finite'])==sum(int(x['finite']) for x in items),'aggregate_finite_sum',key=key,**context)
        if split_info is not None:
            for d in items:
                n=int(d['length']);split=split_info[n];mask=split[d['support']]
                expected_n=int(mask.sum());labels=score_label(d['method'],split)
                positives=labels.sum(1);valid=(positives>0)&(positives<labels.shape[1])
                check.require(int(d['n'])==expected_n,'fixed_common_anchor_count',length=n,key=key,**context)
                check.require(int(d['finite'])==int((mask&valid).sum()),'label_defined_example_count',length=n,key=key,**context)
                check.require(-1e-14<=float(d['auc'])<=1+1e-14,'AUROC_range',length=n,key=key,**context)


def register(expected,scopes,path,digest,scope,check):
    path=str(Path(path));scopes[scope].add(path)
    if path in expected:check.require(expected[path]==digest,'consistent_expected_SHA',path=path)
    expected[path]=digest


def audit_parameters(unit,stage,meta,archive,splits,c,check):
    nmap=45 if stage=='alignment' else 13;nshuffle=672 if stage=='alignment' else 168
    context=dict(unit=unit,stage=stage)
    check.require(meta['status']=='completed' and meta['unit']==unit and meta['stage']==stage,'map_stage_completed_binding',**context)
    check.require(meta['map_calls']==len(meta['maps'])==nmap,'actual_map_count',**context)
    check.require(meta['shuffle_call_count']==len(meta['shuffle_calls'])==nshuffle,'actual_shuffle_count',**context)
    maximum=0.;shuffle_totals=defaultdict(lambda:dict(moved=0,pairs=0))
    with np.load(archive,allow_pickle=False) as arrays:
        expected_keys={f'R_{j:03d}' for j in range(nmap)}|{f'permutation_{j:03d}' for j in range(nshuffle)}
        check.require(set(arrays.files)==expected_keys,'parameter_archive_inventory',**context)
        for j,r in enumerate(meta['maps']):
            R=arrays[f'R_{j:03d}'];error=float(np.max(np.abs(R.T@R-np.eye(R.shape[0]))));maximum=max(maximum,error)
            check.require(r['call']==j,'map_call_sequence',call=j,**context)
            if not (stage=='alignment' and j==0):
                names=['identity','complement','inverse',*[f'shuffle_{s}' for s in range(1026,1034)]]
                index=j-1 if stage=='alignment' else j
                encoding=['canonical_cycle','one_line'][(index//11)//2] if stage=='alignment' else 'one_line'
                phase=['initial','trained'][(index//11)%2] if stage=='alignment' else 'trained'
                fit_condition=names[index%11] if stage=='alignment' or index<11 else ['descents','recoils'][index-11]
                check.require(r['dimension']==256 and (r['encoding'],r['phase'],r['fit_condition'])==(encoding,phase,fit_condition),'map_condition_binding',call=j,**context)
            check.require(R.dtype==np.float64 and R.shape==(r['dimension'],r['dimension']),'map_dtype_shape',call=j,**context)
            check.require(np.isfinite(R).all() and error<=c['numerical']['strict_orthogonality_absolute'],'strict_actual_R_orthogonality_rtol0',error=error,tolerance=c['numerical']['strict_orthogonality_absolute'],call=j,**context)
            check.require(array_sha(R)==r['map_sha256'],'actual_R_SHA',call=j,**context)
            check.close(error,r['maximum_orthogonality_error'],'orthogonality_metadata',call=j,**context)
            check.require(r['determinant_sign'] in (-1.,1.),'O_group_determinant_sign',call=j,**context)
            if stage=='alignment' and j==0:
                rng=np.random.default_rng(21);rng.normal(size=(50,8));q,_=np.linalg.qr(rng.normal(size=(8,8)))
                delta=float(np.max(np.abs(R-q.T)))
                check.require(R.shape==(8,8) and delta<=2e-12,'independent_known_basis_orientation',error=delta,**context)
                check.close(delta,r['synthetic_target_to_source_orientation_error'],'synthetic_orientation_receipt',**context)
        generators={}
        for j,r in enumerate(meta['shuffle_calls']):
            block=j//168;n=10+(j//8)%21;seed=1026+j%8
            enc=['canonical_cycle','one_line'][block//2] if stage=='alignment' else 'one_line'
            phase=['initial','trained'][block%2] if stage=='alignment' else 'trained'
            counts=splits[n]['fit_counts'];actual=arrays[f'permutation_{j:03d}'];m=len(counts)
            check.require((r['call'],r['length'],r['shuffle_seed'],r['encoding'],r['phase'])==(j,n,seed,enc,phase),'shuffle_context',call=j,**context)
            check.require(actual.shape==(m,) and np.issubdtype(actual.dtype,np.integer),'permutation_shape_type',call=j,**context)
            if actual.shape!=(m,):continue
            check.require(np.array_equal(np.sort(actual),np.arange(m)),'shuffle_bijection',call=j,**context)
            if not np.array_equal(np.sort(actual),np.arange(m)):continue
            check.require(np.array_equal(counts[actual],counts),'true_count_preservation',call=j,**context)
            _,inverse,freq=np.unique(counts,return_inverse=True,return_counts=True);single=freq[inverse]==1
            check.require(np.all(actual[~single]!=np.flatnonzero(~single)) and np.array_equal(actual[single],np.flatnonzero(single)),'derangement_except_singletons',call=j,**context)
            check.require(array_sha(actual)==r['permutation_sha256'] and array_sha(counts)==r['counts_sha256'],'shuffle_actual_array_SHA',call=j,**context)
            moved=int((actual!=np.arange(m)).sum())
            check.require((r['fit_pairs'],r['moved'],r['singleton_anchors'])==(m,moved,int(single.sum())),'shuffle_count_metadata',call=j,**context)
            rng=generators.setdefault((block,seed),np.random.default_rng(seed));expected=np.arange(m)
            for k in np.unique(counts):
                ids=np.flatnonzero(counts==k)
                if len(ids)>1:
                    order=rng.permutation(ids);expected[order]=np.roll(order,1)
            check.require(np.array_equal(expected,actual),'independent_fixed_seed_shuffle_reconstruction',call=j,**context)
            totals=shuffle_totals[enc,phase,seed];totals['moved']+=moved;totals['pairs']+=m
        check.close(maximum,meta['maximum_orthogonality_error'],'max_actual_orthogonality_receipt',**context)
    return maximum,shuffle_totals


def check_file(item):
    path,expected=item;p=Path(path);r=dict(path=path,expected_sha256=expected)
    try:
        before=p.stat();actual=sha(p);after=p.stat()
        r.update(actual_sha256=actual,size_bytes=after.st_size,
                 passed=actual==expected and (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns))
    except Exception as e:r.update(passed=False,error_type=type(e).__name__,error=str(e))
    return r


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=2);args=parser.parse_args()
    started=utc();timer=time.monotonic();c=read(HERE/'protocol.json');freeze=read(HERE/'freeze.json');start=read(HERE/'protection_start.json')
    out=Path(c['output']);check=Checks();expected={};scopes=defaultdict(set);unit_info=[];map_max=0.;map_count=0;permutation_count=0
    check.require(read(HERE/'run_completion.json')['status']=='completed','completed_registered_run')
    check.require(read(HERE/'historical_reproduction.json')['status']=='passed','historical_reproduction_passed')
    for path,digest in start['files'].items():register(expected,scopes,path,digest,'protected_old_files',check)
    register(expected,scopes,HERE/'protocol.json',freeze['protocol_sha256'],'frozen_sources',check)
    for path,digest in freeze['source_sha256'].items():register(expected,scopes,ROOT/path,digest,'frozen_sources',check)
    for u in c['units']:
        unit=u['unit'];dest=HERE/'results'/unit;receipt=read(dest/'completion.json');data,splits=independent_splits(u,c['lengths'],check)
        check.require(receipt['status']=='completed' and receipt['unit']==unit and receipt['protocol_sha256']==freeze['protocol_sha256'],'unit_receipt_binding',unit=unit)
        for name,digest in receipt['result_sha256'].items():register(expected,scopes,dest/name,digest,'saved_outputs',check)
        for path,digest in receipt['features']['files'].items():register(expected,scopes,path,digest,'reused_features',check)
        aggregate=csv_rows(dest/'summary.csv');detail=csv_rows(dest/'by_length.csv');other=csv_rows(dest/'other_specialists.csv');other_detail=csv_rows(dest/'other_specialists_by_length.csv')
        check.require((len(aggregate),len(detail),len(other),len(other_detail))==(176,3696,8,168),'full_saved_table_dimensions',unit=unit)
        fixed_rows(aggregate,detail,['encoding','phase','support','method'],c['lengths'],check,dict(unit=unit,table='main'),splits)
        fixed_rows(other,other_detail,['target','support','method'],c['lengths'],check,dict(unit=unit,table='other'))
        main_index={(r['encoding'],r['phase'],r['support'],r['method'],int(r['length'])):r for r in detail}
        previous=csv_rows(PRIOR/'results'/unit/'by_length.csv')
        previous_index={(r['encoding'],r['phase'],r['support'],r['method'],r['transform'],r['label'],int(r['length'])):r for r in previous}
        baseline={'direct':('transfer','complement','peak'),'within_peaks':('source_peaks','identity','peak'),'within_valleys':('target_own','complement','own_valley'),'oracle_count_position_prior':('position_prior','identity','peak')}
        for r in detail:
            if r['method'] not in baseline:continue
            old=previous_index[r['encoding'],r['phase'],'all' if r['support']=='eligible' else 'matched',*baseline[r['method']],int(r['length'])]
            check.close(r['auc'],old['auc'],'reuse_previous_per_length_baseline',unit=unit,method=r['method'],length=int(r['length']))
            check.require(int(r['n'])==int(old['n']) and int(r['finite'])==int(old['defined_n']),'previous_baseline_support_count',unit=unit,method=r['method'])
        for r in other_detail:
            if r['method']=='direct':
                old=previous_index['one_line','trained','all' if r['support']=='eligible' else 'matched',r['target'],'complement','peak',int(r['length'])]
                check.close(r['auc'],old['auc'],'reuse_other_specialist_direct_baseline',unit=unit,target=r['target'],length=int(r['length']))
        for enc in c['encodings']:
            for phase in ['initial','trained']:
                for support in ['eligible','matched']:
                    for n in c['lengths']:
                        common=(enc,phase,support);reference=main_index[*common,'fit_complement__test_complement',n]
                        for method in ['fit_complement__test_native','fit_complement__test_native_own_valley']:
                            check.close(main_index[*common,method,n]['auc'],reference['auc'],'complement_native_label_equivalence',unit=unit,length=n)
        for stage,desc in receipt['map_audits'].items():
            meta_path=Path(desc['path']);meta=read(meta_path);archive=Path(meta['parameter_archive_path'])
            register(expected,scopes,meta_path,desc['sha256'],'saved_parameters',check)
            register(expected,scopes,archive,meta['parameter_archive_sha256'],'saved_parameters',check)
            error,totals=audit_parameters(unit,stage,meta,archive,splits,c,check);map_max=max(map_max,error);map_count+=len(meta['maps']);permutation_count+=len(meta['shuffle_calls'])
            if stage=='alignment':
                fit_audit=csv_rows(dest/'alignment_fit_audit.csv');check.require(len(fit_audit)==4,'map_fit_audit_rows',unit=unit)
                for r in fit_audit:
                    values=[totals[r['encoding'],r['phase'],seed] for seed in c['shuffle_seeds']]
                    pairs=sum(s['fit_n'] for s in splits.values())
                    check.require(int(r['fit_pairs'])==pairs and all(v['pairs']==pairs for v in values),'equal_length_fit_pair_count',unit=unit)
                    check.close(r['min_shuffled_fraction'],min(v['moved']/v['pairs'] for v in values),'actual_shuffle_fraction_csv',unit=unit)
        for record in receipt['validation_adaptation']:
            adapted=Path(record['output_path']);source=Path(record['source_path'])
            register(expected,scopes,adapted,record['output_sha256'],'adapted_validation',check)
            register(expected,scopes,source,record['source_sha256'],'validation_sources',check)
            if source.suffix=='.npz':
                with np.load(source,allow_pickle=False) as z:h=z['prefix_final']
            else:h=np.load(source,allow_pickle=False)
            with np.load(adapted,allow_pickle=False) as z:
                check.require(str(z['checkpoint_sha256'])==record['checkpoint_sha256'] and np.array_equal(z['validation_indices'],np.arange(5376)),'adapted_CP_and_row_order',unit=unit,path=str(adapted))
                check.require(h.shape==(5376,3,256) and h.dtype==np.float32,'raw_validation_source_shape_dtype',unit=unit,path=str(source))
                for action in record['actions']:
                    check.require(z[f'validation_{action}_hidden'].dtype==np.float32 and np.array_equal(z[f'validation_{action}_hidden'],h[:,action]),'exact_validation_action_slice',unit=unit,path=str(adapted),action=action)
            check.require(record['data_sha256']==u['data_sha256'] and record['raw_states_no_additional_normalization'],'adaptation_data_binding',unit=unit)
        expected_match=sum(s['matched_n'] for s in splits.values());prior_coverage=csv_rows(PRIOR/'results'/unit/'coverage.csv')
        check.require(expected_match==sum(int(r['matched_n']) for r in prior_coverage),'previous_common_mask_anchor_count',unit=unit)
        unit_info.append(dict(unit=unit,main_summary_rows=len(aggregate),main_length_rows=len(detail),other_summary_rows=len(other),other_length_rows=len(other_detail),registered_lengths=21,matched_anchors=expected_match,eligible_anchors=sum(s['eligible_n'] for s in splits.values()),probe_fit_validation_pairs=sum(s['fit_n'] for s in splits.values())))
    supplied=read(HERE/'imported_archive_manifest.json');register(expected,scopes,supplied['archive_path'],supplied['archive_sha256'],'supplied_archive',check)
    member_checks=[]
    with ZipFile(supplied['archive_path']) as z:
        for name,desc in supplied['files'].items():
            b=z.read(name);check.require(hashlib.sha256(b).hexdigest()==desc['sha256'],'original_ZIP_member_SHA',member=name)
            rel=Path(name).relative_to('probe_alignment')
            target=PACKAGE/'vendor'/rel.name if rel.parts[0]=='code' else HERE/'supplied_historical'/rel
            register(expected,scopes,target,desc['sha256'],'supplied_copies',check)
            member_checks.append(dict(member=name,repository_copy=str(target),expected_sha256=desc['sha256']))
    results={}
    with ThreadPoolExecutor(max_workers=max(1,args.workers)) as pool:
        futures=[pool.submit(check_file,item) for item in sorted(expected.items())]
        for future in as_completed(futures):
            r=future.result();results[r['path']]=r
            if len(results)%50==0:print(json.dumps(dict(event='alignment_endpoint_hash',files=len(results),total=len(expected))),flush=True)
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();check.require(head==start['git_head'],'analysis_HEAD_unchanged')
    scientific_scopes=[s for s in scopes if s not in ['saved_outputs','saved_parameters','adapted_validation']]
    old_paths=set().union(*(scopes[s] for s in scientific_scopes));old_good=all(results[p]['passed'] for p in old_paths)
    payload_good=all(r['passed'] for r in results.values())
    protected=dict(status='passed' if old_good and head==start['git_head'] else 'failed',started_utc=started,completed_utc=utc(),
        scope='Endpoint SHA of the explicitly named original results/sources/CP/data/caches and supplied package. Not continuous monitoring and not every repository file.',
        protected_start_files=len(start['files']),scope_counts={k:len(v) for k,v in scopes.items()},
        unique_files_hashed=len(results),unique_bytes_hashed=sum(r.get('size_bytes',0) for r in results.values()),
        git_head_start=start['git_head'],git_head_end=head,git_head_unchanged=head==start['git_head'],
        source_hash_failures=[results[p] for p in sorted(old_paths) if not results[p]['passed']],
        files=[results[p] for p in sorted(old_paths)],supplied_member_checks=member_checks,
        no_probe_fits=True,no_alignment_fits=True,no_neural_forward=True,no_old_file_writes=True,elapsed_seconds=time.monotonic()-timer)
    write(HERE/'protection_end.json',protected)
    saved=dict(status='passed' if not check.failures and payload_good else 'failed',completed_utc=utc(),
        scope='Independent scalar-table arithmetic, strict saved-map geometry, deterministic shuffle/split checks, exact adapted feature slices and receipt hashes; not a new scientific replication.',
        units=unit_info,checks=check.total,failures=check.failures,maximum_arithmetic_difference=check.maximum_arithmetic_difference,
        arithmetic_tolerance=TOL,actual_saved_maps_checked=map_count,actual_saved_permutations_checked=permutation_count,
        maximum_actual_R_orthogonality_error=map_max,strict_orthogonality_atol=2e-12,strict_orthogonality_rtol=0,
        all_required_lengths=21,all_unit_equal_length_aggregates_checked=True,
        file_hash_failures=[r for r in results.values() if not r['passed']],
        saved_output_hash_checks=[results[p] for p in sorted(set().union(scopes['saved_outputs'],scopes['saved_parameters'],scopes['adapted_validation']))],
        no_probe_fits=True,no_alignment_fits=True,no_neural_forward=True,no_new_inferential_test=True)
    write(HERE/'saved_results_audit.json',saved)
    print(json.dumps(dict(protection_status=protected['status'],saved_results_status=saved['status'],checks=check.total,
        failures=len(check.failures),maps=map_count,permutations=permutation_count,maximum_actual_R_orthogonality_error=map_max,
        maximum_arithmetic_difference=check.maximum_arithmetic_difference,files_hashed=len(results),units=unit_info)),flush=True)
    if protected['status']!='passed' or saved['status']!='passed':raise SystemExit('Audit failed; inspect saved receipts without modifying science or tolerances.')


if __name__=='__main__':
    with threadpool_limits(limits=2):main()
