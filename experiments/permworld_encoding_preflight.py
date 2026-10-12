"""Real-learning-rate engineering preflight, excluded from confirmation.

Stages are explicit. No stage starts formal source training.
"""
import argparse
from collections import defaultdict
import copy
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

from experiments.permworld_encoding.core import (
    ACTIONS, ENCODINGS, RELATIONS, TASKS, WORKSPACE, TOKEN_TO_ID,
    batch, decode, encode, fingerprint, json_write, load_config, loss,
    make_model, orbit, orbit_key, parameter_hash, sha, transform, verify_frozen,
)
from experiments.permworld_encoding import train as engine
from experiments.permworld_encoding.data import contains, prepare
from experiments.permworld_encoding.evaluate import evaluate_unit, write_csv
from experiments.longrun_attention import accelerate

ROOT = WORKSPACE/'results/permworld_encoding_preflight_20261009'
CONFIG = ROOT/'protocol.json'
DEV_TASKS = ('descents','recoils','exceedances','deficiencies')


def setup():
    if CONFIG.exists():
        verify_frozen(CONFIG)
        return
    base=WORKSPACE/'results/permworld_encoding_plan_20261009'
    old=load_config(base/'freeze_manifest.json')
    assert sha(base/'protocol.json')==old['config_sha256']
    changed=[]
    for path,h in old['code_sha256'].items():
        if sha(WORKSPACE/path)!=h:changed.append(path)
    assert changed==['experiments/permworld_encoding/evaluate.py'], changed
    cfg=copy.deepcopy(load_config(base/'protocol.json'))
    unit=dict(unit='dev001',model_seed=94100001,data_seed=94200001,
              batch_seed=94300001,dropout_seed_base=94400000)
    cfg.update(protocol_version='permworld-encoding-engineering-preflight/v1',
        purpose='engineering_preflight_not_confirmation',status='frozen_before_real_development_training',
        output=str((ROOT/'development').relative_to(WORKSPACE)),
        freeze_manifest=str((ROOT/'freeze_manifest.json').relative_to(WORKSPACE)),
        units=[unit],relations=[r for r in RELATIONS if r['pair_id'] in ['descent_inverse','exceedance_inverse']],
        development_tasks=list(DEV_TASKS),training_manifest=[
            dict(run_id=f'dev001_{e}_{t}',unit='dev001',encoding=e,task=t,
                 model_seed=unit['model_seed'],data_seed=unit['data_seed'],
                 batch_seed=unit['batch_seed'],updates=20000,effective_batch=64,
                 shared_step0='dev001/step0.pt') for e in ENCODINGS for t in DEV_TASKS],
        source_formal_protocol_sha256=sha(base/'protocol.json'),
        engineering_checks={
            'overfit':{'samples':48,'answer_classes':4,'per_class':12,'updates':1000,
                'learning_rate':.0003,'micro_batch':16,'accumulation':3,
                'model_seed':94500001,'dropout_seed_base':94600000,
                'full_vocab_answer_accuracy_min':.95,'teacher_forced_eos_accuracy_min':.99},
            'resume':{'continuous_stop':2000,'restart_from':1000,
                'parameter_max_abs_tolerance':1e-5,'optimizer_max_abs_tolerance':1e-5,
                'fp32_logit_max_abs_tolerance':1e-4,'loss_tolerance':1e-4},
            'independent_cka_absolute_tolerance':5e-8,
            'independent_activation_absolute_tolerance':2e-5,
            'full_source_validation_gain_over_initial_min':.10,
            'all_updates_finite_gradients':True,
            'no_gamma_sign_or_pvalue_gate':True,
            'stop_after_preflight':True,
        },
        registered_utc=datetime.now(timezone.utc).isoformat())
    cfg['statistics']['source_count']=1
    cfg['statistics']['inference']='Development descriptives only, no confirmation p-values or confidence intervals.'
    json_write(CONFIG,cfg)
    hashes={p:sha(WORKSPACE/p) for p in old['code_sha256']}
    hashes[str(Path(__file__).relative_to(WORKSPACE))]=sha(__file__)
    json_write(ROOT/'freeze_manifest.json',{
        'config_sha256':sha(CONFIG),'code_sha256':dict(sorted(hashes.items())),
        'purpose':'engineering preflight only; formal v1 refrozen as an explicit revision after preflight',
        'formal_training_started':False})
    json_write(ROOT/'repair_log.json',{'changes':[{
        'file':'experiments/permworld_encoding/evaluate.py',
        'reason':'Evaluate configured task/relationship subset in a development unit; retain full16 label indices. No CKA equation or formal scope changes.',
        'old_sha256':sha(ROOT/'before/evaluate.py'),'new_sha256':sha(WORKSPACE/'experiments/permworld_encoding/evaluate.py')}],
        'original_formal_protocol_preserved':True,'formal_execution_not_started':True})
    print({'stage':'setup','models':8,'updates_per_model':20000,'formal_training_started':False},flush=True)


def configure_device():
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def world_root(cfg):return WORKSPACE/cfg['output']/'dev001'


def audit_data():
    cfg=verify_frozen(CONFIG);root=world_root(cfg)
    data=dict(np.load(root/'data.npz'))
    hindex=np.load(root.parent/'historical_input_hashes.npy',mmap_mode='r')
    seen=set();memberhashes=[];serializations=0;labels_checked=0
    for split in ['train','validation','test']:
        per=cfg['data']['examples_per_length'][split]
        ns=data[split+'_lengths'];assert set(ns.tolist())==set(cfg['lengths'])
        assert all(np.sum(ns==n)==per for n in cfg['lengths'])
        for i,n in enumerate(ns):
            p=tuple(map(int,data[split+'_permutations'][i,:n]));key=orbit_key(p)
            assert key not in seen;seen.add(key)
            members=orbit(p);assert len(set(members))==8
            memberhashes.extend(fingerprint(q) for q in members)
            if len(memberhashes)>=8192:
                assert not contains(hindex,memberhashes).any();memberhashes=[]
            acts=ACTIONS[:1] if split=='train' else ACTIONS
            for ai,action in enumerate(acts):
                q=transform(p,action);positions=np.argsort(q)
                independent=(sum(q[j]>q[j+1] for j in range(n-1)),
                    sum(positions[j]>positions[j+1] for j in range(n-1)),
                    sum(v>j+1 for j,v in enumerate(q)),sum(v<j+1 for j,v in enumerate(q)))
                for t,y in zip(DEV_TASKS,independent):
                    assert int(data[split+'_labels'][i,ai,TASKS.index(t)])==y
                    labels_checked+=1
                for enc in ENCODINGS:
                    assert decode(data[f'{split}_{enc}_input'][i,ai,:2*n+4],enc)==q
                    serializations+=1
        print({'audit_split':split,'anchors':len(ns)},flush=True)
    assert not contains(hindex,memberhashes).any()
    schedule=np.load(root/'batch_indices.npy');n_schedule=np.load(root/'batch_lengths.npy')
    assert schedule.shape==(20000,64)
    assert np.all(np.diff(np.sort(schedule,axis=1),axis=1)>0)
    assert np.all(data['train_lengths'][schedule]==n_schedule[:,None])
    counts=[int(np.sum(n_schedule==n)) for n in cfg['lengths']]
    assert max(counts)-min(counts)<=1
    rec={'status':'passed','orbit_count':len(seen),'orbit_members_historically_disjoint':True,
        'within_development_orbit_split_overlap':0,'roundtrip_serializations':serializations,
        'independent_four_property_label_checks':labels_checked,'length_update_counts':counts,
        'equal_batches_and_objects_across_encodings':True,'formal_output_absent':not (WORKSPACE/'results/permworld_encoding_confirmation_v1').exists()}
    json_write(ROOT/'data_validation.json',rec)
    engine.step0(cfg,cfg['units'][0],root,sha(CONFIG))
    return data


@torch.inference_mode()
def overfit_scores(model,data,enc,task,selected,device):
    model.eval();ns=data['train_lengths'][selected]
    prefix=data[f'train_{enc}_input'][selected,0]
    answers=data['train_labels'][selected,0,TASKS.index(task)]
    ids,mask,target,_,eq=batch(prefix,ns,task,answers,device)
    logits=model(ids,mask);rows=torch.arange(len(ids),device=device)
    answer=float((logits[rows,eq].argmax(-1)==torch.as_tensor(answers.astype('int64'),device=device)).float().mean())
    eos=float((logits[rows,eq+1].argmax(-1)==TOKEN_TO_ID['<EOS>']).float().mean())
    return {'answer_accuracy':answer,'eos_accuracy':eos,'answer_plus_eos_loss':float(loss(logits,target))}


def check_loss_mask(data,selected,enc,task):
    ns=data['train_lengths'][selected];answers=data['train_labels'][selected,0,TASKS.index(task)]
    ids,mask,target,end,eq=batch(data[f'train_{enc}_input'][selected,0],ns,task,answers)
    assert torch.all((target!=-100).sum(1)==2)
    rows=torch.arange(len(ids));answer=torch.as_tensor(answers.astype('int64'))
    assert torch.equal(target[rows,eq+1],answer)
    assert torch.all(target[rows,eq+2]==TOKEN_TO_ID['<EOS>'])
    logits=torch.randn(len(ids),ids.shape[1],188,requires_grad=True)
    actual=loss(logits,target)
    manual=.5*(F.cross_entropy(logits[rows,eq],answer)+F.cross_entropy(logits[rows,eq+1],
        torch.full_like(answer,TOKEN_TO_ID['<EOS>'])))
    assert abs(float((actual-manual).detach()))<1e-6
    actual.backward();nonzero=logits.grad.abs().sum(-1)>0
    expected=torch.zeros_like(nonzero);expected[rows,eq]=True;expected[rows,eq+1]=True
    assert torch.equal(nonzero,expected)


def overfit():
    cfg=verify_frozen(CONFIG);device=configure_device();root=world_root(cfg)
    assert load_config(ROOT/'data_validation.json')['status']=='passed'
    data=dict(np.load(root/'data.npz'));settings=cfg['engineering_checks']['overfit']
    out=ROOT/'overfit';out.mkdir(exist_ok=True);results=[]
    for ti,task in enumerate(DEV_TASKS):
        candidates=np.flatnonzero(data['train_lengths']==10)
        y=data['train_labels'][candidates,0,TASKS.index(task)]
        counts=np.bincount(y,minlength=31)
        classes=sorted(np.argsort(-counts,kind='stable')[:4].tolist())
        assert all(counts[c]>=12 for c in classes)
        selected=np.concatenate([candidates[y==c][:12] for c in classes])
        assert len(selected)==48 and len(np.unique(y[np.isin(candidates,selected)]))==4
        for enc in ENCODINGS:
            model=make_model(cfg,settings['model_seed']);initial_hash=parameter_hash(model)
            model=accelerate(model).to(device)
            torch.manual_seed(settings['dropout_seed_base']+ti)
            if device.type=='cuda':torch.cuda.manual_seed_all(settings['dropout_seed_base']+ti)
            opt=torch.optim.AdamW(model.parameters(),lr=settings['learning_rate'],weight_decay=.01,fused=device.type=='cuda')
            check_loss_mask(data,selected,enc,task)
            initial=overfit_scores(model,data,enc,task,selected,device);curve=[];min_norm=math.inf;max_norm=0.;started=time.monotonic()
            for step in range(1,settings['updates']+1):
                model.train();opt.zero_grad(set_to_none=True)
                for b in range(3):
                    use=selected[16*b:16*(b+1)]
                    ids,mask,target,_,_=batch(data[f'train_{enc}_input'][use,0],data['train_lengths'][use],task,
                        data['train_labels'][use,0,TASKS.index(task)],device)
                    with torch.autocast(device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'):
                        value=loss(model(ids,mask),target)/3
                    assert torch.isfinite(value);value.backward()
                norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True))
                min_norm=min(min_norm,norm);max_norm=max(max_norm,norm);opt.step()
                if step%100==0:
                    score=overfit_scores(model,data,enc,task,selected,device);curve.append({'step':step,**score})
                    print({'overfit':f'{enc}_{task}','step':step,**score},flush=True)
            final=curve[-1];assert initial_hash!=parameter_hash(model)
            assert all(torch.isfinite(p).all() for p in model.parameters())
            # Independently verify that appending gold answer/EOS cannot leak into answer logits.
            model.eval();use=selected[:8];ns=data['train_lengths'][use]
            with torch.inference_mode():
                ids,mask,_,_,eq=batch(data[f'train_{enc}_input'][use,0],ns,task,device=device)
                prompt=model(ids,mask)[torch.arange(8,device=device),eq]
                ids,mask,_,_,eq=batch(data[f'train_{enc}_input'][use,0],ns,task,data['train_labels'][use,0,TASKS.index(task)],device)
                supplied=model(ids,mask)[torch.arange(8,device=device),eq]
                causal_error=float((prompt-supplied).abs().max())
                assert causal_error<2e-5
            passed=final['answer_accuracy']>=settings['full_vocab_answer_accuracy_min'] and final['eos_accuracy']>=settings['teacher_forced_eos_accuracy_min']
            result={'task':task,'encoding':enc,'samples':48,'answer_classes':classes,'updates':1000,
                'initial':initial,'final':final,'finite_gradient_updates':1000,'minimum_gradient_norm':min_norm,
                'maximum_gradient_norm':max_norm,'parameter_changed':True,'initial_parameter_hash':initial_hash,
                'final_parameter_hash':parameter_hash(model),'manual_loss_and_mask_check':True,
                'future_answer_logit_max_difference':causal_error,'passed':passed,'elapsed_seconds':time.monotonic()-started,'curve':curve}
            engine.atomic_checkpoint(out/f'{enc}_{task}.pt',{'model':model.state_dict(),'engineering_only':True,'protocol_sha256':sha(CONFIG)})
            json_write(out/f'{enc}_{task}.json',result);results.append(result)
            del model,opt;torch.cuda.empty_cache()
    rec={'status':'passed' if all(r['passed'] for r in results) else 'failed','models':results,
        'formal_models_or_seeds_used':False,'gamma_not_used_for_gate':True}
    json_write(ROOT/'overfit_summary.json',rec)
    assert rec['status']=='passed','Real-learning-rate overfit gate failed; do not run full unit automatically.'


class CheckpointPause(Exception):pass


def train_job(run_id,phase=None):
    cfg=verify_frozen(CONFIG);root=world_root(cfg);out=root/run_id
    resume=ROOT/'resume';resume.mkdir(exist_ok=True)
    original_save=engine.atomic_checkpoint
    original_clip=torch.nn.utils.clip_grad_norm_
    stats={'updates':0,'norm_min':math.inf,'norm_max':0.}
    def clip(parameters,*args,**kwargs):
        kwargs['error_if_nonfinite']=True
        norm=original_clip(parameters,*args,**kwargs)
        value=float(norm);stats['updates']+=1;stats['norm_min']=min(stats['norm_min'],value);stats['norm_max']=max(stats['norm_max'],value)
        return norm
    def save(path,payload):
        original_save(path,payload)
        if Path(path).name!='checkpoint.pt' or not phase:return
        if phase=='continuous' and payload['step']==1000:
            shutil.copyfile(path,resume/'step1000.pt')
        if payload['step']==2000:
            shutil.copyfile(path,resume/f'{phase}_step2000.pt')
            raise CheckpointPause()
    if phase=='replayed':shutil.copyfile(resume/'step1000.pt',out/'checkpoint.pt')
    torch.nn.utils.clip_grad_norm_=clip;engine.atomic_checkpoint=save
    started=time.monotonic()
    try:engine.run(CONFIG,run_id)
    except CheckpointPause:print({'development_pause':phase,'step':2000},flush=True)
    finally:
        torch.nn.utils.clip_grad_norm_=original_clip;engine.atomic_checkpoint=original_save
    stats['elapsed_seconds']=time.monotonic()-started;stats['phase']=phase or 'full'
    assert stats['updates']>0 and stats['norm_max']>0
    json_write(out/f'gradient_audit_{phase or "full"}.json',stats)


def tensor_difference(a,b):
    if torch.is_tensor(a):
        assert a.shape==b.shape and a.dtype==b.dtype
        return float((a.double()-b.double()).abs().max()) if a.numel() else 0.
    if isinstance(a,dict):
        assert a.keys()==b.keys()
        return max([tensor_difference(a[k],b[k]) for k in a]+[0.])
    if isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        return max([tensor_difference(x,y) for x,y in zip(a,b)]+[0.])
    assert a==b,(a,b)
    return 0.


def check_resume():
    cfg=verify_frozen(CONFIG);settings=cfg['engineering_checks']['resume'];root=ROOT/'resume'
    a=torch.load(root/'continuous_step2000.pt',map_location='cpu',weights_only=False)
    b=torch.load(root/'replayed_step2000.pt',map_location='cpu',weights_only=False)
    weights=tensor_difference(a['model'],b['model']);optimizer=tensor_difference(a['optimizer'],b['optimizer'])
    assert weights<=settings['parameter_max_abs_tolerance']
    assert optimizer<=settings['optimizer_max_abs_tolerance']
    assert engine.rng_hash(a['rng'])==engine.rng_hash(b['rng'])
    le=abs(a['curve'][-1]['train_loss']-b['curve'][-1]['train_loss'])
    assert le<=settings['loss_tolerance']
    data=dict(np.load(world_root(cfg)/'data.npz'));device=configure_device();scores=[]
    for cp in [a,b]:
        model=accelerate(make_model(cfg,cfg['units'][0]['model_seed'])).to(device).eval();model.load_state_dict(cp['model'])
        ids,mask,_,_,eq=batch(data['validation_one_line_input'][:128,0],data['validation_lengths'][:128],DEV_TASKS[0],device=device)
        with torch.inference_mode():scores.append(model(ids,mask)[torch.arange(128,device=device),eq].cpu())
        del model
    logits=float((scores[0]-scores[1]).abs().max());assert logits<=settings['fp32_logit_max_abs_tolerance']
    rec={'status':'passed','separate_process_restart':True,'restart_from_step':1000,'compared_step':2000,
        'parameters_max_abs_difference':weights,'optimizer_max_abs_difference':optimizer,
        'rng_hash_identical':True,'training_loss_difference':le,'fp32_logit_max_difference':logits,
        'continuous_sha256':sha(root/'continuous_step2000.pt'),'replayed_sha256':sha(root/'replayed_step2000.pt')}
    json_write(ROOT/'resume_validation.json',rec);print(rec,flush=True)


def child(run_id,phase=None):
    command=[sys.executable,'-m','experiments.permworld_encoding_preflight','--stage','train-job','--run-id',run_id]
    if phase:command+=['--phase',phase]
    subprocess.run(command,cwd=WORKSPACE,check=True)


def full_unit():
    cfg=verify_frozen(CONFIG);assert load_config(ROOT/'overfit_summary.json')['status']=='passed'
    start=time.monotonic();first=cfg['training_manifest'][0]['run_id']
    if not (ROOT/'resume_validation.json').exists():
        child(first,'continuous');child(first,'replayed');check_resume()
    for job in cfg['training_manifest']:
        if (world_root(cfg)/job['run_id']/'completed.json').exists():continue
        child(job['run_id'])
    json_write(ROOT/'full_training_complete.json',{'models':8,'updates_per_model':20000,
        'elapsed_seconds':time.monotonic()-start,'formal_training_started':False})


def independent_center(x,keys):
    # Independent projection matrix, rather than the production group-subtraction loop.
    x=np.asarray(x,dtype=np.float64)
    if keys is None:return (np.eye(len(x))-np.ones((len(x),len(x)))/len(x))@x
    _,inv=np.unique(keys,axis=0,return_inverse=True)
    z=np.eye(int(inv.max())+1)[inv]
    return x-z@((z.T@x)/z.sum(0)[:,None])


def feature_cka(x,y,keys):
    x=independent_center(x,keys);y=independent_center(y,keys)
    den=np.linalg.norm(x.T@x)*np.linalg.norm(y.T@y)
    return float(np.linalg.norm(x.T@y)**2/den) if den>1e-20 else None


def independent_evaluation():
    cfg=verify_frozen(CONFIG);root=world_root(cfg);data=dict(np.load(root/'data.npz'))
    import csv
    with (root/'evaluation_length.csv').open() as f:rows=list(csv.DictReader(f))
    assert len(rows)==2*21*2*2*3
    features={};cache_root=root/'evaluation_cache';max_error=0.;comparisons=0
    for condition in ['initial','trained']:
        for enc in ENCODINGS:
            for task in DEV_TASKS:
                path=cache_root/f'{condition}_{enc}_{task}.npz'
                assert sha(path)==load_config(path.with_suffix('.json'))['sha256']
                features[condition,enc,task]=np.load(path)['prefix_final']
    for row in rows:
        if row['metric']!='ordinary':continue
        relation=next(r for r in cfg['relations'] if r['pair_id']==row['relation'])
        ns=data['test_lengths'];yl=data['test_labels'][:,0,TASKS.index(relation['left'])]
        yr=data['test_labels'][:,:,TASKS.index(relation['right'])]
        allkeys=np.column_stack([ns,yl,yr]);_,inv,count=np.unique(allkeys,axis=0,return_inverse=True,return_counts=True)
        use=ns==int(row['length'])
        if row['control']=='answer_strata':use&=count[inv]>=3
        keys=allkeys[use] if row['control']=='answer_strata' else None
        assert use.sum()==int(row['anchors'])
        for condition in ['initial','trained']:
            x=features[condition,row['encoding'],relation['left']][use,0]
            for role,ai in [('correct',2),('wrong',1),('identity',0)]:
                y=features[condition,row['encoding'],relation['right']][use,ai]
                independent=feature_cka(x,y,keys);recorded=row[f'{condition}_{role}']
                if independent is None:assert recorded==''
                else:
                    error=abs(independent-float(recorded));max_error=max(max_error,error)
                    assert error<cfg['engineering_checks']['independent_cka_absolute_tolerance']
                comparisons+=1
        if row['delta']!='':
            independent_delta=(float(row['trained_correct'])-float(row['trained_wrong']))-(float(row['initial_correct'])-float(row['initial_wrong']))
            assert abs(independent_delta-float(row['delta']))<1e-12
    # Recompute a fixed subset using original attention and a prefix-only forward.
    device=configure_device();activation_error=0.;actcomp=0
    for condition in ['initial','trained']:
        for enc in ENCODINGS:
            for task in DEV_TASKS:
                cpfile=root/'step0.pt' if condition=='initial' else root/f'dev001_{enc}_{task}'/'checkpoint.pt'
                cp=torch.load(cpfile,map_location='cpu',weights_only=False)
                model=make_model(cfg,cfg['units'][0]['model_seed']).to(device).eval();model.load_state_dict(cp['model'])
                captures=[];hook=model.final_norm.register_forward_hook(lambda m,i,o:captures.append(o))
                try:
                    for n in cfg['lengths']:
                        chosen=np.flatnonzero(data['test_lengths']==n)[:8]
                        p=data[f'test_{enc}_input'][chosen,:,:2*n+4].reshape(24,2*n+4).astype('int64')
                        ids=torch.as_tensor(p,device=device);captures.clear()
                        with torch.inference_mode():model(ids,ids.ne(TOKEN_TO_ID['<PAD>']))
                        actual=captures[-1][:,-1].cpu().numpy().reshape(8,3,256)
                        expected=features[condition,enc,task][chosen]
                        error=float(np.max(np.abs(actual-expected)));activation_error=max(activation_error,error)
                        assert error<cfg['engineering_checks']['independent_activation_absolute_tolerance']
                        actcomp+=24
                finally:hook.remove()
                del cp,model;torch.cuda.empty_cache()
    json_write(ROOT/'independent_evaluation.json',{'status':'passed','scalar_cka_comparisons':comparisons,
        'max_cka_absolute_difference':max_error,'independent_paths':'Feature cross-products versus Gram alignment; separate stratification projection. Original attention prefix-only forward versus saved SDPA with task prompt.',
        'activation_vectors_checked':actcomp,'max_activation_absolute_difference':activation_error,
        'same_fixed_length_weights':True,'all21_lengths_kept':True})


def report():
    import csv
    cfg=verify_frozen(CONFIG);root=world_root(cfg)
    with (root/'task_accuracy_length.csv').open() as f:acc=list(csv.DictReader(f))
    models=[]
    data=dict(np.load(root/'data.npz'));device=configure_device()
    initial_cp=torch.load(root/'step0.pt',map_location='cpu',weights_only=False)
    initial_model=accelerate(make_model(cfg,cfg['units'][0]['model_seed'])).to(device)
    initial_model.load_state_dict(initial_cp['model'])
    for enc in ENCODINGS:
        for task in DEV_TASKS:
            initial=float(np.mean([float(r['full_vocab_accuracy']) for r in acc if r['encoding']==enc and r['task']==task and r['condition']=='initial' and r['action']=='identity']))
            job=f'dev001_{enc}_{task}';completed=load_config(root/job/'completed.json')
            trained=float(np.mean([float(r['full_vocab_accuracy']) for r in acc if r['encoding']==enc and r['task']==task and r['condition']=='trained' and r['action']=='identity']))
            grad=load_config(root/job/'gradient_audit_full.json')
            final_val=completed['curve'][-1]['validation_accuracy']
            initial_val=engine.accuracy(initial_model,data,enc,task,device)['validation_accuracy']
            passed=final_val-initial_val>=cfg['engineering_checks']['full_source_validation_gain_over_initial_min']
            models.append({'encoding':enc,'task':task,'updates':completed['step'],
                'initial_test_accuracy':initial,'initial_validation_accuracy':initial_val,
                'final_test_accuracy':trained,'final_validation_accuracy':final_val,
                'real_training_seconds':completed['elapsed_seconds'],'finite_gradients':True,
                'source_learning_gate_passed':passed,'gradient_updates_in_last_process':grad['updates']})
    write_csv(ROOT/'source_learning.csv',models)
    with (root/'evaluation_length.csv').open() as f:lengthrows=list(csv.DictReader(f))
    grouped=defaultdict(list)
    for row in lengthrows:
        if row['metric']=='ordinary' and row['control']=='answer_strata':grouped[row['relation'],row['encoding']].append(row)
    interactions=[]
    for relation in cfg['relations']:
        encmeans={}
        for enc in ENCODINGS:
            rows=grouped[relation['pair_id'],enc]
            assert sorted(int(r['length']) for r in rows)==cfg['lengths']
            fields=['initial_preference','trained_preference','change_correct','change_wrong','delta']
            encmeans[enc]={field:float(np.mean([float(r[field]) for r in rows])) if all(r[field]!='' for r in rows) else None for field in fields}
        row={'relation':relation['pair_id'],'development_only':True}
        for field,name in [('initial_preference','gamma_initial'),('trained_preference','gamma_trained'),('change_correct','gamma_correct'),('change_wrong','gamma_wrong'),('delta','gamma_delta')]:
            a=encmeans['one_line'][field];b=encmeans['canonical_cycle'][field];row[name]=b-a if a is not None and b is not None else None
        interactions.append(row)
    write_csv(ROOT/'development_interactions.csv',interactions)
    passed=all(m['source_learning_gate_passed'] for m in models)
    checks={name:load_config(ROOT/name)['status'] for name in ['overfit_summary.json','data_validation.json','resume_validation.json','independent_evaluation.json']}
    passed&=all(value=='passed' for value in checks.values())
    rec={'status':'passed' if passed else 'failed','checks':checks,'full_models':8,'updates_per_model':20000,
        'source_learning_passed':all(m['source_learning_gate_passed'] for m in models),
        'gamma_not_used_for_gate':True,'development_excluded_from_confirmation':True,
        'formal_training_started':False,'user_selected_stop_after_preflight':True}
    json_write(ROOT/'engineering_complete.json',rec)
    lines=['PermWorld 编码工程预跑（开发实验，不计入正式确认）','',f'工程通过状态：{rec["status"]}',
        '四性质×两编码，各20,000更新；正式模型、数据世界和种子均未使用。',
        '小样本真实学习率拟合、单独答案/EOS、有限梯度、掩码和因果端点检查见overfit_summary.json。',
        '独立进程从1000恢复并与连续2000步比较，见resume_validation.json。',
        '全部逐长度普通CKA由激活矩阵另一条路径重算，另用原始注意力/无任务前缀复核激活，见independent_evaluation.json。','',
        '编码 / 性质 / 初始测试准确率 / 最终测试准确率 / 最终验证准确率']
    for m in models:lines.append(f'{m["encoding"]} / {m["task"]}: {100*m["initial_test_accuracy"]:.2f}% / {100*m["final_test_accuracy"]:.2f}% / {100*m["final_validation_accuracy"]:.2f}%')
    lines+=['','Gamma0 / Gamma_t / Gamma_Delta：仅开发描述，不作正式推断，不作为通过标准。']
    for row in interactions:lines.append(str(row))
    lines+=['','本轮完成后停止，不启动固定正式清单。F17、旧诊断和发布产物另行核验保持不变。']
    (ROOT/'report.txt').write_text('\n'.join(lines)+'\n')
    print(rec,flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',required=True,
        choices=['setup','prepare','audit-data','overfit','full-unit','train-job','evaluate','report'])
    p.add_argument('--run-id');p.add_argument('--phase',choices=['continuous','replayed'])
    a=p.parse_args()
    if a.stage=='setup':setup()
    elif a.stage=='prepare':prepare(CONFIG)
    elif a.stage=='audit-data':audit_data()
    elif a.stage=='overfit':overfit()
    elif a.stage=='full-unit':full_unit()
    elif a.stage=='train-job':train_job(a.run_id,a.phase)
    elif a.stage=='evaluate':evaluate_unit(CONFIG,'dev001');independent_evaluation()
    elif a.stage=='report':report()


if __name__=='__main__':main()
