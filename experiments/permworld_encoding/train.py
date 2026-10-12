"""Explicitly invoked training only; real shared step-0 and strict provenance."""
import argparse
import copy
from datetime import datetime,timezone
import json
import math
import os
import random
import time
from pathlib import Path
import numpy as np
import torch
from .core import *
from experiments.longrun_attention import accelerate


def rng_state():
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),
        cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(s):
    random.setstate(s['python']);np.random.set_state(s['numpy']);torch.set_rng_state(s['torch'])
    if torch.cuda.is_available() and s['cuda']:torch.cuda.set_rng_state_all(s['cuda'])


def rng_hash(s):
    import hashlib
    h=hashlib.sha256();h.update(repr(s['python']).encode())
    h.update(s['numpy'][0].encode());h.update(s['numpy'][1].tobytes());h.update(repr(s['numpy'][2:]).encode())
    h.update(s['torch'].numpy().tobytes())
    for state in s['cuda']:h.update(state.cpu().numpy().tobytes())
    return h.hexdigest()


def atomic_checkpoint(path,payload):
    path=Path(path);temporary=path.with_name(path.name+'.tmp');torch.save(payload,temporary);os.replace(temporary,path)


def step0(config,unit,root,protocol_sha):
    path=root/'step0.pt';meta=path.with_suffix('.json')
    if path.exists():
        rec=json.loads(meta.read_text());assert sha(path)==rec['checkpoint_sha256'] and rec['protocol_sha256']==protocol_sha
        saved=torch.load(path,map_location='cpu',weights_only=False)
        model=make_model(config,unit['model_seed']);model.load_state_dict(saved['model'],strict=True)
        assert parameter_hash(model)==rec['parameter_sha256'];return saved,rec
    model=make_model(config,unit['model_seed']);h=parameter_hash(model)
    saved=dict(model=copy.deepcopy(model.state_dict()),rng=rng_state(),config=config['model'],vocabulary=dict(TOKEN_TO_ID),
        model_seed=unit['model_seed'],unit=unit['unit'],protocol_sha256=protocol_sha,stage='post_constructor_and_weight_tying_cpu_before_optimizer_or_training_forward',
        torch_version=torch.__version__,numpy_version=np.__version__,cuda_version=torch.version.cuda,
        code_hashes=json.loads((WORKSPACE/config['freeze_manifest']).read_text())['code_sha256'],parameter_sha256=h)
    atomic_checkpoint(path,saved)
    rec=dict(checkpoint_sha256=sha(path),parameter_sha256=h,protocol_sha256=protocol_sha,unit=unit['unit'],
        actual_original_initial_identity_verified=True,stage=saved['stage'],created_utc=datetime.now(timezone.utc).isoformat())
    json_write(meta,rec);return saved,rec


@torch.inference_mode()
def accuracy(model,data,encoding,task,device):
    model.eval();correct=0;numeric_correct=0;count=0;taskid=TASKS.index(task)
    for start in range(0,len(data['validation_lengths']),128):
        sl=slice(start,start+128);ns=data['validation_lengths'][sl];prefix=data[f'validation_{encoding}_input'][sl,0]
        ids,mask,_,_,eq=batch(prefix,ns,task,device=device);scores=model(ids,mask)[torch.arange(len(ns),device=device),eq]
        truth=torch.as_tensor(data['validation_labels'][sl,0,taskid].astype(np.int64),device=device)
        correct+=int((scores.argmax(-1)==truth).sum());numeric_correct+=int((scores[:,:31].argmax(-1)==truth).sum());count+=len(ns)
    return dict(validation_accuracy=correct/count,validation_numeric_conditional_accuracy=numeric_correct/count,validation_examples=count)


def run(config_path,run_id):
    config=verify_frozen(config_path);job=next(j for j in config['training_manifest'] if j['run_id']==run_id)
    unit=next(u for u in config['units'] if u['unit']==job['unit']);root=WORKSPACE/config['output']/unit['unit']
    protocol_sha=sha(config_path);audit=json.loads((root/'data_audit.json').read_text())
    assert sha(root/'data.npz')==audit['data_sha256'];assert sha(root/'batch_indices.npy')==audit['batch_indices_sha256']
    saved0,meta0=step0(config,unit,root,protocol_sha)
    out=root/run_id;out.mkdir(parents=True,exist_ok=True);checkpoint=out/'checkpoint.pt';completed=out/'completed.json'
    if completed.exists():
        m=json.loads(completed.read_text());assert m['checkpoint_sha256']==sha(checkpoint) and m['protocol_sha256']==protocol_sha;print('Already complete; no repeated training');return
    data=dict(np.load(root/'data.npz'));schedule=np.load(root/'batch_indices.npy',mmap_mode='r')
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model=make_model(config,unit['model_seed']);model.load_state_dict(saved0['model'],strict=True)
    assert parameter_hash(model)==meta0['parameter_sha256'];model=accelerate(model).to(device)
    train=config['training'];opt=torch.optim.AdamW(model.parameters(),lr=train['learning_rate'],weight_decay=train['weight_decay'],
        betas=tuple(train['betas']),eps=train['epsilon'],fused=device.type=='cuda')
    dropout_seed=unit['dropout_seed_base']+TASKS.index(job['task']);random.seed(dropout_seed);np.random.seed(dropout_seed);torch.manual_seed(dropout_seed)
    if device.type=='cuda':torch.cuda.manual_seed_all(dropout_seed)
    # Each task's paired encodings share the exact pre-forward stochastic state as well.
    source_rng=rng_state();source_rng_digest=rng_hash(source_rng);rngpath=root/f'start_rng_{job["task"]}.pt'
    if rngpath.exists():
        prior=torch.load(rngpath,map_location='cpu',weights_only=False)
        assert prior['rng_hash']==source_rng_digest and prior['initial_parameter_sha256']==meta0['parameter_sha256']
    else:atomic_checkpoint(rngpath,dict(rng=source_rng,rng_hash=source_rng_digest,initial_parameter_sha256=meta0['parameter_sha256'],
        stage='device transfer and optimizer construction complete, before first stochastic training forward',dropout_seed=dropout_seed,
        batch_indices_sha256=audit['batch_indices_sha256']))
    start_step=0;curve=[]
    if checkpoint.exists():
        cp=torch.load(checkpoint,map_location='cpu',weights_only=False)
        assert cp['protocol_sha256']==protocol_sha and cp['data_sha256']==audit['data_sha256'] and cp['step0_parameter_sha256']==meta0['parameter_sha256']
        model.load_state_dict(cp['model']);opt.load_state_dict(cp['optimizer']);start_step=cp['step'];curve=cp['curve'];restore_rng(cp['rng'])
    taskid=TASKS.index(job['task']);started=time.monotonic()
    for i in range(start_step,train['updates']):
        model.train();opt.zero_grad(set_to_none=True);step=i+1
        if step<=train['warmup_updates']:factor=step/train['warmup_updates']
        else:factor=train['minimum_lr_ratio']+(1-train['minimum_lr_ratio'])*.5*(1+math.cos(math.pi*(step-train['warmup_updates'])/(train['updates']-train['warmup_updates'])))
        for g in opt.param_groups:g['lr']=train['learning_rate']*factor
        indices=schedule[i];lossvalue=0.
        for k in range(train['accumulation']):
            chosen=indices[k*train['micro_batch']:(k+1)*train['micro_batch']]
            ids,mask,target,_,_=batch(data[f'train_{job["encoding"]}_input'][chosen,0],data['train_lengths'][chosen],job['task'],data['train_labels'][chosen,0,taskid],device=device)
            with torch.autocast(device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'):
                value=loss(model(ids,mask),target)/train['accumulation']
            value.backward();lossvalue+=float(value.detach())
        torch.nn.utils.clip_grad_norm_(model.parameters(),train['gradient_clip']);opt.step()
        if step%train['validation_every']==0 or step==train['updates']:
            record=dict(step=step,train_loss=lossvalue,**accuracy(model,data,job['encoding'],job['task'],device));curve.append(record)
            print({'run':run_id,**record},flush=True)
        if step%train['checkpoint_every']==0 or step==train['updates']:
            atomic_checkpoint(checkpoint,dict(model=model.state_dict(),optimizer=opt.state_dict(),step=step,curve=curve,rng=rng_state(),
                run_id=run_id,protocol_sha256=protocol_sha,data_sha256=audit['data_sha256'],step0_checkpoint_sha256=meta0['checkpoint_sha256'],
                step0_parameter_sha256=meta0['parameter_sha256'],initial_identity_verified=True,dropout_seed=dropout_seed))
    json_write(completed,dict(run_id=run_id,unit=unit['unit'],task=job['task'],encoding=job['encoding'],step=train['updates'],
        checkpoint_sha256=sha(checkpoint),protocol_sha256=protocol_sha,step0_checkpoint_sha256=meta0['checkpoint_sha256'],step0_parameter_sha256=meta0['parameter_sha256'],
        data_sha256=audit['data_sha256'],sample_exposures=train['updates']*train['effective_batch'],supervised_tokens=2*train['updates']*train['effective_batch'],
        source_start_rng_sha256=sha(rngpath),source_start_rng_hash=source_rng_digest,
        elapsed_seconds=time.monotonic()-started,curve=curve))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--run-id',required=True);a=p.parse_args();run(a.config,a.run_id)
