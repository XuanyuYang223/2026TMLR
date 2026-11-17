"""Bounded saved-result verification, independent GRU equations, end hashes.

Actual checkpoint weights are read as dictionaries on CPU. Selected upper
updates are independently recomputed with NumPy GRU equations; no production
model is instantiated or resumed and no readout is fitted. This is explicitly
arithmetic neural verification, not a claim of zero neural calculations.
"""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import argparse
import csv
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time

import numpy as np
import torch
from threadpoolctl import threadpool_limits

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT))
OUT=Path('/mnt/yangx-projects-f/20261110/permworld_local_timing')
OLD_OUT=Path('/mnt/yangx-projects-f/20261110/permworld_local_causal')
LENGTHS=(10,20,30)
CENTERS=(2,6)
TASKS=('peaks','valleys')
SEEDS=tuple(range(1026,1034))
COMMON=('unedited','local_projection','same_label_projection',*(f'random_direction_{s}' for s in SEEDS))
AUX=tuple(f'late_layer_swap_{k}' for k in range(1,5))
REPLAY=(*COMMON,'early_full_layer1')
ORDER=(*(('late',c) for c in COMMON),*(('early',c) for c in COMMON),
       *(('late',c) for c in AUX),('early','early_full_layer1'),
       ('diagnostic','full_state_swap'),('diagnostic','actual_counterfactual'))
TOL=1e-10
STATE_TOL=2e-5
LOGIT_TOL=1e-4


def read(path):return json.loads(Path(path).read_text())
def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def bits(array):return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()
def write(path,value):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)
def number(value):
    return np.nan if value is None or str(value).lower() in ('','none','nan') else float(value)
def boolean(value):return str(value).lower() in ('true','1')


class Checks:
    def __init__(self):self.count=0;self.failures=[];self.maxima={}
    def ok(self,value,label):
        self.count+=1
        if not bool(value):self.failures.append(dict(label=label))
    def close(self,actual,expected,label,tolerance=TOL,category='metric_probe_arithmetic'):
        a=np.asarray(actual,float);b=np.asarray(expected,float);self.count+=1
        if a.shape!=b.shape or not np.array_equal(np.isnan(a),np.isnan(b)):
            self.failures.append(dict(label=label,reason='Shape/undefined support differs'));return
        finite=np.isfinite(a)&np.isfinite(b)
        error=float(np.max(np.abs(a[finite]-b[finite]))) if finite.any() else 0.
        self.maxima[category]=max(error,self.maxima.get(category,0.))
        if error>tolerance or not np.array_equal(np.isinf(a),np.isinf(b)):
            self.failures.append(dict(label=label,error=error,tolerance=tolerance,category=category))


def sigmoid(value):
    value=np.asarray(value,np.float64);out=np.empty_like(value);positive=value>=0
    out[positive]=1/(1+np.exp(-value[positive]))
    exponential=np.exp(value[~positive]);out[~positive]=exponential/(1+exponential)
    return out


def numpy_gru_step(inputs,previous,weight_ih,weight_hh,bias_ih,bias_hh):
    """PyTorch reset/update/new ordering, with reset only on hidden-new term."""
    x=np.asarray(inputs,np.float64);h=np.asarray(previous,np.float64)
    input_affine=x@np.asarray(weight_ih,np.float64).T+np.asarray(bias_ih,np.float64)
    hidden_affine=h@np.asarray(weight_hh,np.float64).T+np.asarray(bias_hh,np.float64)
    ir,iz,inn=np.split(input_affine,3,axis=-1)
    hr,hz,hn=np.split(hidden_affine,3,axis=-1)
    reset=sigmoid(ir+hr);update=sigmoid(iz+hz)
    candidate=np.tanh(inn+reset*hn)
    return (1-update)*candidate+update*h


def upper_equations(previous,first_layer_after,state_dict):
    current=np.asarray(first_layer_after,np.float64);upper=[]
    for layer in range(1,4):
        prefix=f'recurrent_layers.{layer}.'
        current=numpy_gru_step(current,previous[:,layer],
            state_dict[prefix+'weight_ih_l0'],state_dict[prefix+'weight_hh_l0'],
            state_dict[prefix+'bias_ih_l0'],state_dict[prefix+'bias_hh_l0'])
        upper.append(current)
    return np.stack(upper,axis=1)


def synthetic_checks():
    """No real inputs: compare equations to a synthetic torch.GRU reference."""
    torch.manual_seed(921);rng=np.random.default_rng(921)
    layer=torch.nn.GRU(5,5,batch_first=True).double().eval()
    x=rng.normal(size=(7,5));h=rng.normal(size=(7,5))
    state={k:v.detach().numpy() for k,v in layer.state_dict().items()}
    reference=numpy_gru_step(x,h,**{k.replace('_l0',''):v for k,v in state.items()})
    with torch.inference_mode():_,actual=layer(torch.as_tensor(x)[:,None],torch.as_tensor(h)[None])
    error=float(np.max(np.abs(reference-actual[0].numpy())))
    assert error<=1e-12
    # A hand-built case detects both reset placement and hidden-new bias.
    result=numpy_gru_step([[0.]],[[0.]],np.zeros((3,1)),np.zeros((3,1)),
                         np.array([-np.log(3.),0.,1.]),np.array([0.,0.,2.]))
    np.testing.assert_allclose(result,.5*np.tanh(1.5),atol=1e-14,rtol=0)
    prev=rng.normal(size=(4,4,5));first=rng.normal(size=(4,5));fake={}
    current=torch.as_tensor(first,dtype=torch.float64)[:,None];expected=[]
    for index in range(1,4):
        gru=torch.nn.GRU(5,5,batch_first=True).double().eval()
        for key,val in gru.state_dict().items():fake[f'recurrent_layers.{index}.{key}']=val.detach().numpy()
        with torch.inference_mode():current,hn=gru(current,torch.as_tensor(prev[:,index])[None])
        expected.append(hn[0].numpy())
    combined=upper_equations(prev,first,fake)
    stack_error=float(np.max(np.abs(combined-np.stack(expected,1))))
    assert stack_error<=1e-12
    result=dict(status='passed',scope='Synthetic mathematical gate/order/seed verification only; no actual checkpoint or timing outcome accessed.',
        single_GRU_max_error=error,three_upper_GRU_max_error=stack_error,
        hand_checked_reset_and_hidden_bias=True,tolerance=1e-12,
        actual_checkpoint_model_modules_constructed=0,synthetic_torch_GRU_references=True)
    write(HERE/'equation_synthetic_verification.json',result);return result


def count_metrics(logits):
    z=np.asarray(logits,np.float64)
    p=np.exp(z-z.max(-1,keepdims=True));p/=p.sum(-1,keepdims=True)
    numeric=z[...,:31];q=np.exp(numeric-numeric.max(-1,keepdims=True));q/=q.sum(-1,keepdims=True)
    token=z.argmax(-1)
    return dict(token=token,count=np.where(token<=30,token,-1),mass=p[...,:31].sum(-1),expectation=(q*np.arange(31)).sum(-1))


def linear(probe,role,states):return np.asarray(states,float)@probe[role+'_weight']+probe[role+'_bias']


def checkpoint_arrays(descriptor,checks,label):
    checks.ok(sha(descriptor['checkpoint_path'])==descriptor['checkpoint_sha256'],label+'/checkpoint SHA before trusted dictionary load')
    checkpoint=torch.load(descriptor['checkpoint_path'],map_location='cpu',weights_only=False)
    checks.ok(checkpoint['architecture']=='gru' and checkpoint['step']==checkpoint['finite_gradient_updates']==20000,label+'/actual frozen20k checkpoint header')
    checks.ok(checkpoint['data_sha256']==descriptor['data_sha256'],label+'/checkpoint data binding')
    digest=hashlib.sha256();weights={}
    for name,tensor in sorted(checkpoint['model'].items()):
        array=tensor.detach().cpu().contiguous().numpy()
        for text in (name,str(array.dtype),str(array.shape)):digest.update(text.encode())
        digest.update(array.tobytes())
        if name.startswith(('recurrent_layers.1.','recurrent_layers.2.','recurrent_layers.3.')):weights[name]=array.copy()
    checks.ok(digest.hexdigest()==descriptor['parameter_sha256'],label+'/actual saved state_dict identity')
    del checkpoint
    return weights


def verify_task(unit,task,report,checks):
    from experiments.permworld_local_causal_20261110.shared import families,catalog
    name=unit['unit'];label=name+'/'+task
    selected=families(name);records={r['pair_id']:(f,r) for f in selected if f['split']=='test' for r in f['pairs'] if r['kind'] in ('simple','count_matched')}
    nodes,_=catalog(selected,task,{'test'});id_to_row={r['node_id']:j for j,r in enumerate(nodes)}
    old_state=np.load(unit['test_states'][task]['path'],mmap_mode='r',allow_pickle=False)
    with np.load(OLD_OUT/name/'states'/task/'test/native_logits.npz',allow_pickle=False) as z:native=z['native_logits']
    with np.load(unit['probes'][task]['path'],allow_pickle=False) as z:probe=dict(z)
    checks.ok(sha(unit['probes'][task]['path'])==unit['probes'][task]['sha256'],label+'/unchanged saved probe SHA')
    weights=checkpoint_arrays(unit['models'][task],checks,label)
    checks.ok(report['parameter_sha256_before']==report['parameter_sha256_after']==unit['models'][task]['parameter_sha256'],label+'/all production forward parameter identities')
    maxima=report['maxima']
    checks.ok(maxima['noedit_state']<=STATE_TOL,label+'/all-case unchanged replay state guard')
    for key in ('noedit_logits','early_noedit_native_logits','late_old_logits','fullswap_native_logits'):
        checks.ok(maxima[key]<=LOGIT_TOL,label+'/all-case '+key+' guard')
    checks.ok(report['identical_layer1_bytes_all_common_cases'] and report['no_probe_fitting'],label+'/all common byte identity/no-refit receipt')
    checks.ok(len(report['cells'])==len(unit['cells'][task])==357,label+'/all357 cell receipts')
    new_cells={(r['n'],r['i']):r for r in report['cells']}
    expected_rows={};case_count=0;equation_cases=0;small_files=0
    for old_desc in unit['cells'][task]:
        n,i=old_desc['n'],old_desc['i']
        if n not in LENGTHS or i not in CENTERS:continue
        new_desc=new_cells[n,i]
        checks.ok(sha(old_desc['path'])==old_desc['sha256'],label+f'/n{n}i{i}/source cell SHA')
        checks.ok(sha(new_desc['path'])==new_desc['sha256'],label+f'/n{n}i{i}/new cell SHA')
        with np.load(old_desc['path'],allow_pickle=False) as z:old=dict(z)
        with np.load(new_desc['path'],allow_pickle=False) as z:raw=dict(z)
        small_files+=1
        checks.ok(list(zip(map(str,raw['timings']),map(str,raw['conditions'])))==list(ORDER),label+'/29 condition order')
        checks.ok(list(map(str,raw['injected_conditions']))==list(REPLAY),label+'/12 current-upper replay order')
        checks.ok(np.array_equal(raw['pair_ids'],old['pair_ids']) and list(raw['pair_ids'])==old_desc['pair_ids'],label+'/same paired source ID order')
        for key in ('recipient_node_ids','donor_node_ids','actual_cf_node_ids'):
            checks.ok(np.array_equal(raw[key],old[key]),label+'/'+key)
        checks.ok(str(raw['source_cell_sha256'])==old_desc['sha256'] and str(raw['source_checkpoint_sha256'])==unit['models'][task]['checkpoint_sha256'] and str(raw['data_sha256'])==unit['data_sha256'] and str(raw['protocol_sha256'])==sha(HERE/'protocol.json'),label+'/new/source cell provenance')
        chosen={}
        for j,(group,sign) in enumerate(zip(old_desc['groups'],old_desc['signs'])):chosen.setdefault((group,sign),j)
        indices=np.array(list(chosen.values()),int)
        previous=raw['recipient_previous_states'][indices]
        ri=np.array([id_to_row[str(k)] for k in raw['recipient_node_ids'][indices]])
        checks.close(previous,old_state[ri,0],label+'/source recipient BEFORE states',0.,'exact_byte_values')
        lower=raw['injected_layer1_states'][:,indices]
        expected_upper=upper_equations(np.concatenate([previous]*len(REPLAY)),lower.reshape(-1,256),weights).reshape(len(REPLAY),len(indices),3,256)
        checks.close(raw['early_current_upper_states'][:,indices],expected_upper,label+'/independent NumPy current GRU equations',STATE_TOL,'GRU_equation_verification')
        equation_cases+=len(REPLAY)*len(indices)
        old_names=list(map(str,old['condition_names']))
        for j in indices:
            family,pair=records[str(old['pair_ids'][j])]
            recipient=old['recipient_states'][j];donor=old['donor_states'][j];same=old['same_label_states'][j]
            recipient_index=id_to_row[str(old['recipient_node_ids'][j])];donor_index=id_to_row[str(old['donor_node_ids'][j])];cf_index=id_to_row[str(old['actual_cf_node_ids'][j])]
            checks.close(recipient,old_state[recipient_index,1],label+'/source recipient AFTER states',0.,'exact_byte_values')
            lower_expected={'unedited':recipient[0],
                'local_projection':(recipient[0].astype(float)+old['coordinate'][j]*old['direction']).astype(np.float32)}
            same_coordinate=(same[0].astype(float)-recipient[0])@old['direction']
            lower_expected['same_label_projection']=(recipient[0].astype(float)+same_coordinate*old['direction']).astype(np.float32)
            for seed,q in zip(SEEDS,old['random_directions']):lower_expected[f'random_direction_{seed}']=(recipient[0].astype(float)+old['coordinate'][j]*q).astype(np.float32)
            carried={}
            for k,condition in enumerate(COMMON):
                injected=raw['injected_layer1_states'][k,j]
                checks.ok(bits(injected)==bits(lower_expected[condition]),label+'/'+condition+'/exact reused edited h1 bytes')
                late=recipient.copy();late[0]=injected
                early=np.concatenate([injected[None],raw['early_current_upper_states'][k,j]],axis=0)
                carried['late',condition]=late;carried['early',condition]=early
                checks.ok(bits(late[0])==bits(early[0]),label+'/'+condition+'/paired exact lower state')
                checks.close(raw['logits'][ORDER.index(('late',condition)),j],old['logits'][old_names.index(condition),j],label+'/'+condition+'/late original logits',LOGIT_TOL,'logit_wiring_guards')
            for layer,condition in enumerate(AUX):
                h=recipient.copy();h[layer]=donor[layer];carried['late',condition]=h
            injected=raw['injected_layer1_states'][-1,j]
            checks.ok(bits(injected)==bits(donor[0]),label+'/full-L1 exact donor lower state')
            carried['early','early_full_layer1']=np.concatenate([injected[None],raw['early_current_upper_states'][-1,j]],axis=0)
            carried['diagnostic','full_state_swap']=donor
            carried['diagnostic','actual_counterfactual']=np.array(old_state[cf_index,1])
            checks.close(carried['early','unedited'],recipient,label+'/unchanged current replay state',STATE_TOL,'noedit_state_guard')
            queries=raw['logits'][:,j];metrics=count_metrics(queries)
            baseline=metrics['count'][ORDER.index(('late','unedited'))]
            baseline_token=metrics['token'][ORDER.index(('late','unedited'))]
            baseline_expect=metrics['expectation'][ORDER.index(('late','unedited'))]
            baseline_mass=metrics['mass'][ORDER.index(('late','unedited'))]
            checks.close(queries[ORDER.index(('early','unedited'))],queries[ORDER.index(('late','unedited'))],label+'/early noedit versus late',LOGIT_TOL,'logit_wiring_guards')
            checks.close(queries[ORDER.index(('early','unedited'))],native[recipient_index],label+'/early noedit versus original native',LOGIT_TOL,'logit_wiring_guards')
            checks.close(queries[ORDER.index(('diagnostic','full_state_swap'))],native[donor_index],label+'/full donor wiring',LOGIT_TOL,'logit_wiring_guards')
            checks.close(queries[ORDER.index(('diagnostic','actual_counterfactual'))],old['logits'][old_names.index('actual_counterfactual'),j],label+'/actualCF byte reuse',0.,'exact_byte_values')
            target_norm=np.linalg.norm(carried['late','local_projection'][0].astype(float)-recipient[0])
            for condition in COMMON:
                if condition.startswith('random_direction_'):
                    actual=np.linalg.norm(carried['early',condition][0].astype(float)-recipient[0])
                    checks.ok(abs(actual-target_norm)<=1e-5+1e-5*target_norm,label+'/random injected lower norm')
            for k,key in enumerate(ORDER):
                timing,condition=key;h=carried[key];delta=h.astype(float)-recipient
                norm=float(np.linalg.norm(delta[0]));full_norm=float(np.linalg.norm(delta))
                plan=abs(float(old['coordinate'][j])) if condition=='local_projection' or condition.startswith('random_direction_') else norm
                layer=int(condition.rsplit('_',1)[1]) if condition in AUX else 1 if timing!='diagnostic' else 0
                dimension=1 if condition in COMMON and condition!='unedited' else 256 if condition in AUX or condition=='early_full_layer1' else 1024 if condition=='full_state_swap' else 0
                total=pair['recipient_total_count'];desired=total+pair['direction']
                oracle=total if condition in ('unedited','same_label_projection') else pair['donor_total_count'] if condition=='full_state_swap' else desired if condition=='actual_counterfactual' else ''
                before=float(linear(probe,'checker',recipient[0]));after=float(linear(probe,'checker',h[0]))
                numeric=baseline>=0 and metrics['count'][k]>=0
                expected_rows[task,pair['pair_id'],timing,condition]=dict(
                    metadata=dict(unit=name,task=task,group='main' if pair['kind']=='simple' else 'matched',sign=pair['direction'],
                        family_id=pair['family_id'],component_id=family['component_id'],length=n,position=i,layer=layer,probe_layer=1,
                        payload_dimension=dimension,cut_token_index=4+2*i,recipient_node_id=nodes[recipient_index]['node_id'],
                        pair_donor_id=nodes[donor_index]['node_id'],first_layer_state_sha256=bits(h[0]),
                        recipient_local_label=pair['recipient_label'],desired_local_label=pair['donor_label'],
                        recipient_resolved_count=pair['recipient_prefix_count'],donor_resolved_count=pair['donor_prefix_count'],
                        recipient_true_count=total,desired_true_count=desired,donor_full_true_count=pair['donor_total_count'],
                        condition_oracle_count=oracle,source_cell_sha256=old_desc['sha256']),
                    numeric=dict(dose=1.,probe_boundary=.5,recipient_local_score=before,edited_local_score=after,
                        checker_donor_score=float(linear(probe,'checker',donor[0])),
                        edit_probe_recipient_score=float(linear(probe,'edit',recipient[0])),edit_probe_edited_score=float(linear(probe,'edit',h[0])),edit_probe_donor_score=float(linear(probe,'edit',donor[0])),
                        recipient_prefix_count_prediction=float(linear(probe,'count',recipient[0])),edited_prefix_count_prediction=float(linear(probe,'count',h[0])),
                        recipient_argmax_count=int(baseline),edited_argmax_count=int(metrics['count'][k]),recipient_argmax_token=int(baseline_token),edited_argmax_token=int(metrics['token'][k]),
                        recipient_numeric_expectation=float(baseline_expect),edited_numeric_expectation=float(metrics['expectation'][k]),recipient_numeric_mass=float(baseline_mass),edited_numeric_mass=float(metrics['mass'][k]),
                        planned_edit_l2=plan,layer1_edit_l2=norm,edit_l2=norm,full_state_edit_l2=full_norm,carried_state_l2norm=full_norm),
                    derived=dict(relative_step_hit=float(numeric and metrics['count'][k]-baseline==pair['direction']),desired_answer_hit=float(metrics['count'][k]==desired),
                        reference_oracle_hit=np.nan if oracle=='' else float(metrics['count'][k]==oracle),numeric_argmax_coverage=float(numeric),
                        edited_nonnumeric_rate=float(metrics['count'][k]<0),signed_numeric_expectation_delta=float(pair['direction']*(metrics['expectation'][k]-baseline_expect)),
                        checker_boundary_crossing=float(int(before>=.5)==pair['recipient_label'] and int(after>=.5)==pair['donor_label'])),
                    recipient_correct=bool(baseline==total))
            case_count+=1
    del old_state,native,weights
    return expected_rows,dict(task=task,cases=case_count,selected_cells=small_files,manual_upper_equation_cases=equation_cases,
        all_case_maxima=report['maxima'],early_noop_argmax_mismatches=report['early_noop_argmax_mismatches'],rerun_late_old_argmax_mismatches=report['rerun_late_old_argmax_mismatches'])


def verify_unit(unit,checks,score_function):
    name=unit['unit'];receipt=read(HERE/'results'/name/'completion.json')
    checks.ok(receipt['status']=='completed' and receipt['protocol_sha256']==sha(HERE/'protocol.json'),name+'/completed protocol binding')
    for field in ('no_probe_fitting','no_new_data','no_base_training','no_DAS','no_cross_model','no_new_generalization_task'):
        checks.ok(receipt[field],name+'/'+field)
    expected={};reports=[]
    for task in TASKS:
        report=next(r for r in receipt['task_reports'] if r['task']==task)
        wanted,part=verify_task(unit,task,report,checks);expected.update(wanted);reports.append(part)
    files=list(receipt['files']);checks.ok(len(files)==1,name+'/one complete pair table')
    path=Path(files[0]);checks.ok(sha(path)==receipt['files'][str(path)],name+'/streamed timing CSV SHA')
    seen=set();row_count=0
    with path.open(newline='') as stream:
        reader=csv.reader(stream);header=next(reader);idx={field:k for k,field in enumerate(header)}
        for values in reader:
            row_count+=1
            key=(values[idx['task']],values[idx['pair_id']],values[idx['timing']],values[idx['condition']])
            if key not in expected:continue
            checks.ok(key not in seen,name+'/unique selected29 row');seen.add(key)
            row=dict(zip(header,values));wanted=expected[key];label=name+'/'+str(key)
            for field,value in wanted['metadata'].items():checks.ok(row[field]==str(value),label+'/'+field)
            checks.ok(boolean(row['condition_available']) and row['missing_reason']=='',label+'/input-audited available condition')
            checks.ok(boolean(row['recipient_correct'])==wanted['recipient_correct'],label+'/baseline correctness')
            for field,value in wanted['numeric'].items():checks.close(number(row[field]),value,label+'/'+field)
            derived=score_function(row)
            for field,value in wanted['derived'].items():checks.close(number(derived[field]),value,label+'/independent relative-versus-oracle/'+field)
    checks.ok(row_count==receipt['rows'],name+'/complete row inventory')
    checks.ok(seen==set(expected),name+'/all fixed selected rows')
    return dict(unit=name,rows_streamed=row_count,selected_rows_verified=len(seen),task_reports=reports)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--synthetic',action='store_true');parser.add_argument('--workers',type=int,default=2);args=parser.parse_args()
    if args.synthetic:
        print(json.dumps(synthetic_checks()),flush=True);return
    if not (HERE/'run_completion.json').exists():raise SystemExit('Wait for all five timing units to complete.')
    began=datetime.now(timezone.utc).isoformat();clock=time.monotonic();checks=Checks()
    protocol=read(HERE/'protocol.json');start=read(HERE/'protection_start.json');frozen=read(HERE/'freeze.json');audit=read(HERE/'input_audit.json')
    complete=read(HERE/'run_completion.json');names=[u['unit'] for u in audit['units']]
    checks.ok(complete['status']=='completed' and complete['units']==names,'all five timing units finished')
    checks.ok(complete['no_probe_fitting'] and complete['no_base_training'] and complete['no_extra_stage'],'stop boundary and no fitting/training')
    expected=dict(start['files']);scope={p:'original_timing_input_snapshot' for p in expected}
    expected[str(HERE/'protocol.json')]=frozen['protocol_sha256'];scope[str(HERE/'protocol.json')]='new_frozen_protocol'
    for relative,digest in frozen['sources'].items():
        path=str(ROOT/relative)
        if path in expected:checks.ok(expected[path]==digest,'consistent input/frozen expected SHA')
        expected[path]=digest;scope.setdefault(path,'new_frozen_source')
    spec=importlib.util.spec_from_file_location('frozen_timing_summary_for_audit',HERE/'summarize_results.py')
    summary=importlib.util.module_from_spec(spec);spec.loader.exec_module(summary)
    reports=[]
    for unit in audit['units']:
        reports.append(verify_unit(unit,checks,summary.score))
        print(json.dumps(dict(event='timing_unit_verified',unit=unit['unit'],checks=checks.count,failures=len(checks.failures))),flush=True)
    # One physical read per named input, preserving logical path/symlink scope.
    physical={};logical={}
    for path,digest in expected.items():
        target=str(Path(path).resolve());logical[path]=target
        if target in physical:checks.ok(physical[target]==digest,'consistent aliased physical SHA')
        physical[target]=digest
    def hash_one(item):
        path,digest=item;file=Path(path);before=file.stat();actual=sha(file);after=file.stat()
        return path,dict(expected_sha256=digest,actual_sha256=actual,size_bytes=after.st_size,
            passed=actual==digest and (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns))
    with ThreadPoolExecutor(max_workers=max(1,args.workers)) as pool:physical_records=dict(pool.map(hash_one,sorted(physical.items())))
    files=[dict(path=path,physical_path=logical[path],scope=scope[path],**physical_records[logical[path]]) for path in sorted(expected)]
    for path,desc in start['symlinks'].items():checks.ok(Path(path).is_symlink() and os.readlink(path)==desc['target'] and str(Path(path).resolve())==desc['physical_path'],'old named symlink target unchanged')
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    tracked=subprocess.check_output(['git','status','--porcelain=v1','--untracked-files=no'],cwd=ROOT,text=True)
    initial_tracked=[row for row in start['git_status'].splitlines() if not row.startswith('?? ')]
    checks.ok(head==start['git_head'],'Git HEAD unchanged');checks.ok(not initial_tracked and not tracked,'tracked files clean at recorded endpoints')
    protection=dict(status='passed' if all(r['passed'] for r in files) and head==start['git_head'] and not tracked else 'failed',
        started_utc=began,completed_utc=datetime.now(timezone.utc).isoformat(),
        scope='End SHA of4498 explicitly named timing inputs/old reports/used ten test-state payloads plus separately frozen new dependencies. Physical paths deduplicated. Endpoint evidence only, not every file or continuous monitoring.',
        original_input_logical_files=len(start['files']),logical_files_verified=len(files),unique_physical_files_hashed=len(physical_records),
        unique_physical_bytes_hashed=sum(r['size_bytes'] for r in physical_records.values()),files=files,failed_files=[r for r in files if not r['passed']],
        git_head_start=start['git_head'],git_head_end=head,git_head_unchanged=head==start['git_head'],end_tracked_status=tracked,
        unused_fit_dev_payloads_rehashed=0,unused40GB_token_payloads_read=0,no_old_file_writes=True)
    result=dict(status='passed' if not checks.failures and protection['status']=='passed' else 'failed',
        completed_utc=datetime.now(timezone.utc).isoformat(),checks=checks.count,failures=checks.failures,maximum_differences_by_category=checks.maxima,
        tolerances=dict(metric_probe_arithmetic=TOL,GRU_equation_verification=STATE_TOL,noedit_state_guard=STATE_TOL,logit_wiring_guards=LOGIT_TOL),
        units=reports,fixed_subset='Input-order first case per(group,sign), n10/20/30,i2/6, both tasks/allfive:180cases,5220CSVrows,60old/newcellpairs; no outcome selection.',
        GRU_equation_scope='NumPy float64 reset/update/new equations for L2-L4 using actual saved CPU checkpoint dictionaries, recipient previous states and injected current h1. Reset multiplies the hidden-new affine term including its bias, not the input-new term. This bounded independent arithmetic verifies saved upper states, not a new production model evaluation.',
        limitations=['Manual equations perform bounded neural arithmetic; no production torch model was constructed, resumed or fitted in the actual audit.',
            'This is implementation/provenance verification, not an independent scientific repeat or evidence for a natural abstract one-dimensional mechanism.',
            'Token-rank/value and generic count-axis explanations remain possible; early/late differences concern this intervention interface/current update.'],
        no_actual_checkpoint_model_construction=True,no_production_neural_forward=True,no_probe_fitting=True,no_new_base_training=True,
        no_new_conditions_or_stages=True,manual_GRU_equation_verification=True,
        provenance_snapshot={str(p):sha(p) for p in [HERE/'finalize_audit.py',HERE/'equation_synthetic_verification.json',HERE/'code_review.json',HERE/'freeze.json',HERE/'input_audit.json']},
        elapsed_seconds=time.monotonic()-clock)
    write(HERE/'protection_end.json',protection);write(HERE/'saved_results_audit.json',result)
    print(json.dumps(dict(protection_status=protection['status'],audit_status=result['status'],checks=checks.count,failures=len(checks.failures),
        maxima=checks.maxima,physical_files_hashed=len(physical_records),elapsed_seconds=result['elapsed_seconds'])),flush=True)
    if result['status']!='passed':raise SystemExit('Audit failed; retain frozen sources/results/tolerances and inspect failures.')


if __name__=='__main__':
    torch.set_num_threads(2)
    with threadpool_limits(limits=2):main()
