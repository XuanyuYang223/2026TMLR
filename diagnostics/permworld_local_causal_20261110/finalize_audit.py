"""Bounded independent saved-intervention audit and named endpoint hashes.

No fitting, model construction, forward, or intervention is run here. CSVs
are streamed; only predeclared input-order cases are retained. Large forward
state caches are memory-mapped at needed nodes, not rehashed or copied whole.
"""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import argparse
import csv
import hashlib
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
OUT=Path('/mnt/yangx-projects-f/20261110/permworld_local_causal')
TASKS=('peaks','valleys')
SEEDS=tuple(range(1026,1034))
CONDITIONS=('unedited','local_projection','same_label_projection',
    *(f'random_direction_{s}' for s in SEEDS),'full_state_swap','actual_counterfactual')
LENGTHS=(10,20,30)
CENTERS=(2,6)
TOL=1e-10


def read(path):return json.loads(Path(path).read_text())
def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def write(path,value):
    path=Path(path);temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temporary.replace(path)
def scalar(value):
    if value is None or str(value).lower() in ('','nan','none'):return np.nan
    return float(value)
def boolean(value):return str(value).lower() in ('true','1')


class Checks:
    def __init__(self):self.count=0;self.failures=[];self.maximum=0.
    def ok(self,value,label):
        self.count+=1
        if not bool(value):self.failures.append(dict(label=label))
    def close(self,actual,expected,label,tolerance=TOL):
        a=np.asarray(actual,dtype=float);b=np.asarray(expected,dtype=float);self.count+=1
        if a.shape!=b.shape or not np.array_equal(np.isnan(a),np.isnan(b)):
            self.failures.append(dict(label=label,reason='Shape or undefined support differs'));return
        finite=np.isfinite(a)&np.isfinite(b)
        error=float(np.max(np.abs(a[finite]-b[finite]))) if finite.any() else 0.
        self.maximum=max(self.maximum,error)
        if error>tolerance or not np.array_equal(np.isinf(a),np.isinf(b)):
            self.failures.append(dict(label=label,error=error,tolerance=tolerance))


def count_metrics(logits):
    """Independent NumPy FP64 softmax, full-vocabulary argmax and mean."""
    z=np.asarray(logits,np.float64)
    maximum=z.max(-1,keepdims=True)
    probabilities=np.exp(z-maximum);probabilities/=probabilities.sum(-1,keepdims=True)
    numeric=z[...,:31];conditional=np.exp(numeric-numeric.max(-1,keepdims=True))
    conditional/=conditional.sum(-1,keepdims=True)
    token=z.argmax(-1)
    return dict(argmax_token=token,argmax_count=np.where(token<=30,token,-1),
                numeric_mass=probabilities[...,:31].sum(-1),
                expectation=(conditional*np.arange(31)).sum(-1))


def linear(parameters,role,x):
    return np.asarray(x,np.float64)@parameters[role+'_weight']+parameters[role+'_bias']


def input_cases(selected,task,n,i,lookup):
    from experiments.permworld_local_causal_20261110.shared import transformed
    result=[]
    for family in selected:
        if family['split']!='test' or family['n']!=n or family['center']!=i:continue
        controls={r['control_for_pair_id']:r for r in family['pairs'] if r['kind']=='same_label'}
        for pair in family['pairs']:
            if pair['kind'] not in ('simple','count_matched'):continue
            find=lambda p:lookup[n,i,transformed(p,task)]
            same=controls.get(pair['pair_id'])
            result.append(dict(family=family,pair=pair,same=same,
                recipient=find(pair['recipient']),donor=find(pair['donor']),
                actual_cf=find(pair['actual_local_counterfactual']),
                same_node=None if same is None else find(same['donor'])))
    return result


def check_task(unit,task,protocol,task_receipt,checks):
    from experiments.permworld_local_causal_20261110.shared import families,catalog,transformed,ids_for
    name=unit['unit'];selected=families(name)
    nodes,lookup=catalog(selected,task,{'test'})
    fit_nodes,_=catalog(selected,task,{'fit','dev'})
    layer=int(protocol['selected_layer']);point=int(protocol['point']);dose=float(protocol['dose'])
    checks.ok(point==1 and dose==1 and protocol['dimension']==1,name+'/'+task+'/fixed selected intervention')
    parameter_path=Path(task_receipt['probe_file'])
    checks.ok(sha(parameter_path)==task_receipt['probe_sha256'],name+'/'+task+'/probe archive binding')
    with np.load(parameter_path,allow_pickle=False) as z:parameters=dict(z)
    for role in ('edit','checker','count'):
        weight=parameters[role+'_ridge_coef']/parameters[role+'_scaler_scale']
        bias=parameters[role+'_ridge_intercept']-parameters[role+'_scaler_mean']@weight
        checks.close(parameters[role+'_weight'],weight,name+'/'+task+'/'+role+'/serialized raw weight')
        checks.close(parameters[role+'_bias'],bias,name+'/'+task+'/'+role+'/serialized raw bias')
    direction=parameters['edit_weight']/np.linalg.norm(parameters['edit_weight'])
    checks.close(parameters['direction'],direction,name+'/'+task+'/direction from FIT_A editing weight')
    a=[r for r in fit_nodes if r['pool']=='fit' and r['fit_partition']=='A']
    b=[r for r in fit_nodes if r['pool']=='fit' and r['fit_partition']=='B']
    checks.ok(not ({r['component'] for r in a}&{r['component'] for r in b}),name+'/'+task+'/source fit halves component-disjoint')
    checks.ok(list(parameters['fit_a_node_ids'])==[r['node_id'] for r in a] and list(parameters['fit_b_node_ids'])==[r['node_id'] for r in b],name+'/'+task+'/source fit node identities')
    state_path=OUT/name/'states'/task/'test/states.npy'
    states=np.load(state_path,mmap_mode='r',allow_pickle=False)
    state_meta=read(state_path.parent/'completion.json')
    checks.ok(states.shape==(len(nodes),2,4,256) and states.dtype==np.float32,name+'/'+task+'/state cache header')
    checks.ok(state_meta['node_ids']==[r['node_id'] for r in nodes],name+'/'+task+'/state cache node order')
    checks.ok(state_meta['model_parameter_sha256']==unit['models'][task]['parameter_sha256'],name+'/'+task+'/state parameter identity')
    checks.ok(state_meta['maximum_original_vs_noop_logit_difference']<=1e-4 and state_meta['original_vs_noop_logit_tolerance']==1e-4,name+'/'+task+'/all-node original/noop guard')
    with np.load(state_path.parent/'native_logits.npz',allow_pickle=False) as z:
        native=z['native_logits'];noop=z['noop_logits']
    checks.ok(native.shape==noop.shape==(len(nodes),188),name+'/'+task+'/natural query logits shape')
    native_metrics=count_metrics(native);base_metrics=count_metrics(noop)
    task_index=TASKS.index(task);unit_index=0 if unit['historical'] else int(name[-2:])
    random=[]
    for seed in SEEDS:
        value=np.random.default_rng(np.random.SeedSequence([seed,unit_index,task_index])).normal(size=256)
        random.append(value/np.linalg.norm(value))
    random=np.stack(random)
    expected_rows={};case_count=0;small_file_count=0
    artifact_expected={item['path']:item['sha256'] for item in task_receipt['_intervention_files']}
    for n in LENGTHS:
        for i in CENTERS:
            cases=input_cases(selected,task,n,i,lookup)
            checks.ok(bool(cases),name+'/'+task+f'/n{n}i{i}/registered input cases')
            if not cases:continue
            path=OUT/name/'interventions'/task/f'n{n}_i{i}.npz'
            checks.ok(str(path) in artifact_expected and sha(path)==artifact_expected[str(path)],name+'/'+task+f'/n{n}i{i}/saved score archive SHA')
            with np.load(path,allow_pickle=False) as z:raw=dict(z)
            small_file_count+=1
            checks.ok(list(raw['condition_names'])==list(CONDITIONS),name+'/'+task+'/condition-major score order')
            checks.ok(list(raw['pair_ids'])==[r['pair']['pair_id'] for r in cases],name+'/'+task+'/input pair order')
            for field,key in [('recipient_node_ids','recipient'),('donor_node_ids','donor'),('actual_cf_node_ids','actual_cf')]:
                checks.ok(list(raw[field])==[nodes[c[key]]['node_id'] for c in cases],name+'/'+task+'/'+field)
            checks.ok(str(raw['source_checkpoint_sha256'])==unit['models'][task]['checkpoint_sha256'] and str(raw['data_sha256'])==unit['data_sha256'] and str(raw['protocol_sha256'])==sha(HERE/'protocol.json'),name+'/'+task+'/score source bindings')
            checks.close(raw['direction'],direction,name+'/'+task+'/saved intervention direction',0.)
            checks.close(raw['random_directions'],random,name+'/'+task+'/fixed random direction RNG',0.)
            # First input-order case of each registered group/sign. Nothing is
            # selected using scores, geometry, count accuracy, or effect size.
            chosen={}
            for index,case in enumerate(cases):chosen.setdefault((case['pair']['kind'],case['pair']['direction']),index)
            for index in chosen.values():
                case=cases[index];pair=case['pair'];ri,di,ci=case['recipient'],case['donor'],case['actual_cf']
                si=case['same_node'];hr=np.array(states[ri,point]);hd=np.array(states[di,point]);hc=np.array(states[ci,point]);hs=np.array(states[ri if si is None else si,point])
                case_count+=1
                for field,value in [('recipient_states',hr),('donor_states',hd),('same_label_states',hs)]:
                    checks.close(raw[field][index],value,name+'/'+task+'/'+pair['pair_id']+'/'+field,0.)
                coordinate=float((hd[layer].astype(np.float64)-hr[layer])@direction)
                same_coordinate=float((hs[layer].astype(np.float64)-hr[layer])@direction)
                checks.close(raw['coordinate'][index],coordinate,name+'/'+task+'/projection coordinate')
                changed=hr.copy();changed[layer]=(hr[layer].astype(np.float64)+dose*coordinate*direction).astype(np.float32)
                same_changed=hr.copy();same_changed[layer]=(hr[layer].astype(np.float64)+dose*same_coordinate*direction).astype(np.float32)
                carried={'unedited':hr,'local_projection':changed,'same_label_projection':same_changed,
                         'full_state_swap':hd,'actual_counterfactual':hc}
                for seed,q in zip(SEEDS,random):
                    h=hr.copy();h[layer]=(hr[layer].astype(np.float64)+dose*coordinate*q).astype(np.float32)
                    carried[f'random_direction_{seed}']=h
                treatment_norm=np.linalg.norm(changed.astype(float)-hr)
                for seed in SEEDS:
                    norm=np.linalg.norm(carried[f'random_direction_{seed}'].astype(float)-hr)
                    checks.ok(abs(norm-treatment_norm)<=1e-5+1e-5*treatment_norm,name+'/'+task+'/actual random norm matching')
                queries=ids_for([nodes[k]['permutation'] for k in (ri,di,ci)],task);cut=4+2*i
                checks.ok(np.array_equal(queries[0,cut+1:],queries[1,cut+1:]) and np.array_equal(queries[0,cut+1:],queries[2,cut+1:]),name+'/'+task+'/identical donor and localCF future suffix')
                logits=raw['logits'][:,index]
                checks.close(logits[CONDITIONS.index('unedited')],noop[ri],name+'/'+task+'/raw no-op query',0.)
                checks.close(logits[CONDITIONS.index('actual_counterfactual')],native[ci],name+'/'+task+'/raw actualCF query',0.)
                # This tolerance is the producer's fixed all-case wiring guard,
                # not a new criterion for causal or scientific success.
                checks.close(logits[CONDITIONS.index('full_state_swap')],native[di],name+'/'+task+'/full-state donor wiring',1e-4)
                metric=count_metrics(logits)
                for condition_index,condition in enumerate(CONDITIONS):
                    h=carried[condition];available=condition!='same_label_projection' or si is not None
                    donor=ri if condition=='unedited' else ci if condition=='actual_counterfactual' else si if condition=='same_label_projection' else di if condition in ('local_projection','full_state_swap') else None
                    full_count=pair['recipient_total_count'];desired=full_count+pair['direction']
                    oracle=pair['donor_total_count'] if condition=='full_state_swap' else full_count if condition in ('unedited','same_label_projection') else desired
                    norm=float(np.linalg.norm(h.astype(float)-hr))
                    plan=abs(coordinate)*dose if condition=='local_projection' or condition.startswith('random_direction_') else abs(same_coordinate)*dose if condition=='same_label_projection' else norm
                    expected_rows[task,pair['pair_id'],condition]=dict(
                        metadata=dict(group='main' if pair['kind']=='simple' else 'matched',sign=pair['direction'],
                            length=n,position=i,layer=layer+1,timing='after',cut_token_index=cut,
                            family_id=pair['family_id'],component_id=case['family']['component_id'],
                            recipient_node_id=nodes[ri]['node_id'],donor_id=nodes[di]['node_id'],pair_donor_node_id=nodes[di]['node_id'],
                            donor_fields_scope='fixed_label_changing_pair_donor',
                            intervention_donor_id='' if donor is None else nodes[donor]['node_id'],
                            recipient_local_label=pair['recipient_label'],desired_local_label=pair['donor_label'],
                            intervention_donor_local_label=-1 if donor is None else nodes[donor]['label'],
                            intervention_donor_resolved_count='' if donor is None else nodes[donor]['resolved_count'],
                            intervention_donor_full_true_count='' if donor is None else nodes[donor]['total_count'],
                            recipient_resolved_count=pair['recipient_prefix_count'],donor_resolved_count=pair['donor_prefix_count'],
                            recipient_true_count=full_count,desired_true_count=desired,donor_full_true_count=pair['donor_total_count'],condition_oracle_count=oracle),
                        available=available,numeric=dict(dose=dose,probe_boundary=.5,
                            recipient_local_score=float(linear(parameters,'checker',hr[layer])),
                            edited_local_score=float(linear(parameters,'checker',h[layer])) if available else np.nan,
                            checker_donor_score=float(linear(parameters,'checker',hd[layer])),
                            edit_probe_recipient_score=float(linear(parameters,'edit',hr[layer])),
                            edit_probe_edited_score=float(linear(parameters,'edit',h[layer])) if available else np.nan,
                            edit_probe_donor_score=float(linear(parameters,'edit',hd[layer])),
                            recipient_prefix_count_prediction=float(linear(parameters,'count',hr[layer])),
                            edited_prefix_count_prediction=float(linear(parameters,'count',h[layer])) if available else np.nan,
                            recipient_argmax_count=int(base_metrics['argmax_count'][ri]),
                            edited_argmax_count=int(metric['argmax_count'][condition_index]) if available else -1,
                            recipient_argmax_token=int(base_metrics['argmax_token'][ri]),
                            edited_argmax_token=int(metric['argmax_token'][condition_index]) if available else -1,
                            recipient_numeric_expectation=float(base_metrics['expectation'][ri]),
                            edited_numeric_expectation=float(metric['expectation'][condition_index]) if available else np.nan,
                            recipient_numeric_mass=float(base_metrics['numeric_mass'][ri]),
                            edited_numeric_mass=float(metric['numeric_mass'][condition_index]) if available else np.nan,
                            planned_edit_l2=plan if available else np.nan,edit_l2=norm if available else np.nan),
                        recipient_correct=int(base_metrics['argmax_count'][ri])==full_count)
            del raw
    del states,native,noop
    return expected_rows,dict(task=task,fixed_cases=case_count,selected_score_archives=small_file_count,
        original_noop_max_error=state_meta['maximum_original_vs_noop_logit_difference'],
        fullswap_all_case_max_error=task_receipt['maximum_fullswap_vs_native_donor_logit_difference'])


def check_unit(unit,protocol,checks):
    name=unit['unit'];receipt=read(HERE/'results'/name/'completion.json')
    checks.ok(receipt['status']=='completed' and receipt['protocol_sha256']==sha(HERE/'protocol.json'),name+'/formal completion binding')
    checks.ok(receipt['no_optimizer'] and receipt['no_backward'] and receipt['no_base_training'],name+'/frozen network execution scope')
    expected={};task_reports=[]
    for task in TASKS:
        item=dict(next(r for r in receipt['task_audits'] if r['task']==task))
        checks.ok(item['parameter_sha256_before']==item['parameter_sha256_after']==unit['models'][task]['parameter_sha256'],name+'/'+task+'/all forward parameter identity')
        checks.ok(item['maximum_fullswap_vs_native_donor_logit_difference']<=1e-4,name+'/'+task+'/all-case fullstate donor wiring')
        checks.ok(item['all_conditions_use_actual_recurrent_state'],name+'/'+task+'/actual state assertion')
        item['_intervention_files']=receipt['intervention_files']
        result,report=check_task(unit,task,protocol,item,checks);expected.update(result);task_reports.append(report)
    paths=list(receipt['files']);checks.ok(len(paths)==1,name+'/single complete pair table')
    path=Path(paths[0]);checks.ok(sha(path)==receipt['files'][str(path)],name+'/streamed full CSV SHA')
    seen=set();row_count=0
    with path.open(newline='') as stream:
        reader=csv.reader(stream);header=next(reader);columns={name:i for i,name in enumerate(header)}
        for values in reader:
            row_count+=1
            key=(values[columns['task']],values[columns['pair_id']],values[columns['condition']])
            if key not in expected:continue
            checks.ok(key not in seen,name+'/unique selected row');seen.add(key)
            row=dict(zip(header,values));wanted=expected[key];label=name+'/'+key[0]+'/'+key[1]+'/'+key[2]
            checks.ok(row['unit']==name,label+'/unit')
            for field,value in wanted['metadata'].items():checks.ok(row[field]==str(value),label+'/'+field)
            checks.ok(boolean(row['condition_available'])==wanted['available'],label+'/availability')
            checks.ok(boolean(row['recipient_correct'])==wanted['recipient_correct'],label+'/baseline correctness')
            for field,value in wanted['numeric'].items():checks.close(scalar(row[field]),value,label+'/'+field)
    checks.ok(row_count==receipt['rows'],name+'/complete CSV row inventory')
    checks.ok(seen==set(expected),name+'/all predeclared selected rows present')
    return dict(unit=name,total_rows_streamed=row_count,selected_rows_verified=len(seen),tasks=task_reports)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--workers',type=int,default=2);args=parser.parse_args()
    if not (HERE/'run_completion.json').exists():raise SystemExit('Wait until all five formal units finish.')
    started=time.monotonic();began=datetime.now(timezone.utc).isoformat()
    protocol=read(HERE/'protocol.json');source=read(HERE/'source_audit.json');start=read(HERE/'protection_start.json')
    formal=read(HERE/'freeze.json');engineering=read(HERE/'engineering_freeze.json');checks=Checks()
    complete=read(HERE/'run_completion.json');names=[r['unit'] for r in source['units']]
    checks.ok(complete['status']=='completed' and complete['units']==names,'complete fixed five source units')
    checks.ok(complete['stopped_after_within_model_stage'] and complete['no_base_training'],'authorized stop boundary')
    expected=dict(start['files']);scopes={p:'named_original_snapshot' for p in expected}
    for frozen,protocol_path,label in [(engineering,HERE/'engineering_protocol.json','engineering'),(formal,HERE/'protocol.json','formal')]:
        checks.ok(sha(protocol_path)==frozen['protocol_sha256'],label+'/protocol identity')
        expected[str(protocol_path)]=frozen['protocol_sha256'];scopes.setdefault(str(protocol_path),label+'_frozen_protocol')
        for path,digest in frozen['sources'].items():
            absolute=str(ROOT/path)
            if absolute in expected:checks.ok(expected[absolute]==digest,label+'/consistent source expectation')
            expected[absolute]=digest;scopes.setdefault(absolute,label+'_frozen_source')
    reports=[]
    for unit in source['units']:
        reports.append(check_unit(unit,protocol,checks))
        print(json.dumps(dict(event='unit_verified',unit=unit['unit'],checks=checks.count,failures=len(checks.failures))),flush=True)
    def hash_one(item):
        path,digest=item;file=Path(path);before=file.stat();actual=sha(file);after=file.stat()
        return dict(path=path,scope=scopes[path],expected_sha256=digest,actual_sha256=actual,
            size_bytes=after.st_size,passed=actual==digest and (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns))
    with ThreadPoolExecutor(max_workers=max(1,args.workers)) as pool:files=list(pool.map(hash_one,sorted(expected.items())))
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    tracked=subprocess.check_output(['git','status','--porcelain=v1','--untracked-files=no'],cwd=ROOT,text=True)
    checks.ok(head==start['git_head'],'Git HEAD unchanged')
    initial_tracked=[s for s in start['git_status'].splitlines() if not s.startswith('?? ')]
    checks.ok(not initial_tracked and not tracked,'Tracked files clean at recorded endpoints')
    protection=dict(status='passed' if all(r['passed'] for r in files) and head==start['git_head'] and not tracked else 'failed',
        started_utc=began,completed_utc=datetime.now(timezone.utc).isoformat(),
        scope='Endpoint SHA of the752 original named files and separately frozen engineering/formal dependencies; not every file or continuous monitoring. Snapshot follows the earlier authorized validation-only engine preflight.',
        original_snapshot_files=len(start['files']),unique_files_hashed=len(files),bytes_hashed=sum(r['size_bytes'] for r in files),
        files=files,failed_files=[r for r in files if not r['passed']],git_head_start=start['git_head'],git_head_end=head,
        git_head_unchanged=head==start['git_head'],start_tracked_status=initial_tracked,end_tracked_status=tracked,
        whole_forward_state_caches_rehashed=0,no_old_file_writes=True)
    verification=dict(status='passed' if not checks.failures and protection['status']=='passed' else 'failed',
        completed_utc=datetime.now(timezone.utc).isoformat(),checks=checks.count,failures=checks.failures,
        maximum_checked_difference=checks.maximum,FP64_score_metric_tolerance=TOL,
        original_noop_and_fullswap_tolerance=1e-4,actual_norm_tolerance='1e-5+1e-5*treatment_actual_l2',
        units=reports,fixed_subset='First input-order case per(group,sign) at n10/20/30,i2/6,both tasks/allfive. Expected180cases and2340rows. Selected solely from input identities before outcome access.',
        scope='Reconstructed saved rank-one/random carried states and independent probes, NumPy FP64 raw-logit metrics and condition metadata; full-state/noop/real-input positive controls; streaming CSV and explicit source endpoint verification. No new model computation.',
        limitations=['No neural forward was repeated for partial edits. Their recurrence wiring is supported by frozen code review,11engine tests and all-case producer no-op/fullstate guards.',
            'Source-state cache SHA values rely on creation receipts; only selected nodes were memory-mapped and whole caches were not rehashed.',
            'Count matching and the FIT_B prefix-count readout do not rule out generic count-axis information. Motif values/ranks and synthetic state distribution remain alternative explanations; no abstract mathematical mechanism is identified by this audit.',
            'Technical checks and selected pairs are not additional independent experimental repeats.'],
        no_neural_forward=True,no_model_construction=True,no_probe_fitting=True,no_base_training=True,
        no_new_hyperparameters=True,no_cross_model_exchange=True,
        provenance_snapshot={str(p):sha(p) for p in [HERE/'finalize_audit.py',HERE/'formal_code_review.json',HERE/'freeze.json',HERE/'engineering_freeze.json',HERE/'source_audit.json']},
        elapsed_seconds=time.monotonic()-started)
    write(HERE/'protection_end.json',protection);write(HERE/'final_verification.json',verification)
    write(HERE/'saved_results_audit.json',verification)
    print(json.dumps(dict(protection_status=protection['status'],verification_status=verification['status'],checks=checks.count,
        failures=len(checks.failures),maximum_difference=checks.maximum,files_hashed=len(files),elapsed_seconds=verification['elapsed_seconds'])),flush=True)
    if verification['status']!='passed':raise SystemExit('Audit failed; preserve scientific results and frozen tolerances.')


if __name__=='__main__':
    with threadpool_limits(limits=2):main()
