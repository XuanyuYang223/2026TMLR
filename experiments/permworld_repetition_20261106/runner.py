"""Explicit, resumable execution of four additional shared-initialization units.

The old experiment is read-only. All large artifacts live in a new external
directory. No accuracy/geometry gate, adaptive seed selection, or extra world
is allowed. Run ``python -m experiments.permworld_repetition_20261106.runner -h``.
"""
from __future__ import annotations

import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import copy
from datetime import datetime, timezone
import fcntl
import hashlib
import itertools
import json
import math
from pathlib import Path
import platform
import random
import subprocess
import sys
import time
import traceback

import numpy as np
import torch

from experiments.permworld_encoding.core import (
    TASKS, RELATIONS, ACTIONS, ENCODINGS, TOKEN_TO_ID, WORKSPACE, sha,
    parameter_hash, batch,
)
from experiments.permworld_encoding.data import generate_world
from experiments.permworld_encoding.train import (
    atomic_checkpoint, rng_state, rng_hash, restore_rng,
)
from experiments.permworld_encoding_archdev_20261009.common import (
    ARCHITECTURES, DEV_TASKS, model, device_setup,
)
from experiments.permworld_encoding_archdev_20261009.evaluation import extract, manual_final
from experiments.permworld_eight_relations_20261010.train import update, optimizer, accuracy

DIAG = WORKSPACE / 'diagnostics/permworld_repetition_20261106'
DEFAULT_CONFIG = DIAG / 'protocol.json'
DEFAULT_ROOT = Path('/mnt/yangx-projects-f/20261106/permworld_repetition')
SOURCE_BASE = Path('/mnt/yangx-projects-f/20261009/permworld_new_artifacts/encoding_archdev')


def utc():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    os.replace(temporary, path)


def event(root, **record):
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    with (root / 'events.jsonl').open('a') as f:
        f.write(json.dumps(dict(utc=utc(), **record), allow_nan=False) + '\n')


def source_files():
    from . import analysis  # include future analysis imports in the same freeze
    paths = list((WORKSPACE / 'experiments/permworld_repetition_20261106').glob('*.py'))
    for name in ['permworld_encoding/core.py', 'permworld_encoding/data.py',
                 'permworld_encoding/train.py', 'permworld_encoding/evaluate.py',
                 'permworld_encoding_archdev_20261009/common.py',
                 'permworld_encoding_archdev_20261009/evaluation.py',
                 'permworld_eight_relations_20261010/common.py',
                 'permworld_eight_relations_20261010/train.py',
                 'permworld_eight_relations_20261010/analysis_helpers.py',
                 'longrun_attention.py', 'permutation_audit.py']:
        paths.append(WORKSPACE / 'experiments' / name)
    paths.extend((WORKSPACE / 'external/neurips/src/neurips_permutations').rglob('*.py'))
    paths.extend([WORKSPACE / 'external/neurips/configs/property_task_geometry.toml',
                  WORKSPACE / 'diagnostics/permworld_scope_inventory_20261009/metrics.py'])
    # Some old helpers import other local modules even when their fitting entry
    # points are unused. Bind that actual transitive import closure as well.
    for module in list(sys.modules.values()):
        file=getattr(module,'__file__',None)
        if file:
            p=Path(file).resolve()
            if p.suffix=='.py' and p.is_relative_to(WORKSPACE) and p.is_file():paths.append(p)
    return {str(p.relative_to(WORKSPACE)): sha(p) for p in sorted(set(paths))}


def freeze(root=DEFAULT_ROOT):
    """Register fixed seeds before new sampling or training, never overwrite."""
    if DEFAULT_CONFIG.exists():
        return verify(DEFAULT_CONFIG)
    base = read(SOURCE_BASE / 'protocol.json')
    old = read(WORKSPACE / 'diagnostics/permworld_eight_relations_20261010/protocol.json')
    assert list(TASKS) == old['tasks'] and len(TASKS) == 16
    # Fixed public numeric sequence; no pilot-result-dependent selection.
    units = [dict(unit=f'unit{i:02d}', data_seed=96200000+i,
                  batch_seed=96300000+i, dropout_seed_base=96400000+10000*i,
                  architecture_model_seeds={'transformer': 96100000+2*i,
                                             'gru': 96100001+2*i},
                  evaluation_rng_seed=None,
                  evaluation_anchor_source='same sequential data_seed RNG as train/validation; no separate evaluation sampler')
             for i in range(2, 6)]
    history = SOURCE_BASE / 'worlds/historical_input_hashes.npy'
    history_meta = read(history.with_suffix('.json'))
    assert sha(history) == history_meta['sha256']
    config = dict(version='permworld-independent-repetition/v1', registered_utc=utc(),
        purpose='Four fixed additional units; original observed unit is the fifth. Joint random-factor robustness, not factor-isolated causal effects.',
        output=str(Path(root).resolve()), historical_unit='archdev001', units=units,
        tasks=list(TASKS), label_task_names=list(TASKS), relations=list(RELATIONS),
        architectures=list(ARCHITECTURES), encodings=list(ENCODINGS), actions=list(ACTIONS),
        lengths=base['lengths'], length_weights=base['evaluation']['length_weights'],
        model=base['model'], training=base['training'], data=base['data'],
        evaluation=base['evaluation'],
        history_index=dict(path=str(history), sha256=sha(history), metadata=history_meta),
        original_protocols={str(p):sha(p) for p in [SOURCE_BASE/'protocol.json',
            WORKSPACE/'diagnostics/permworld_eight_relations_20261010/protocol.json',
            WORKSPACE/'diagnostics/representation_pairing_review_20261011/protocol.json']},
        source_git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=WORKSPACE,text=True).strip(),
        source_worktree_status=subprocess.check_output(['git','status','--porcelain'],cwd=WORKSPACE,text=True),
        software=dict(python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,
                      cuda=torch.version.cuda,device=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None),
        training_randomness='For original four tasks, dropout_base+1000*architecture_index+index(descents,recoils,peaks,valleys). Other twelve use descents stream. Each encoding pair shares actual RNG. Fresh base per unit.',
        shared_initialization='One actual constructor state_dict per architecture/unit, copied exactly across 16 tasks and both encodings. Architectures do not share parameters.',
        sampling_scope='Exact original rejection sampler and fixed historical exclusion index. Sequential train/validation/test draws, within-world orbit-disjoint; cross-world overlap allowed, not artificially rejected. No new mathematical system/task distribution.',
        analysis_scope='Original final_norm three metrics plus debiased CKA and output probabilities, raw/joint-answer. Comparison B raw and candidate-specific answer ordinary CKA. Layer/input-pooling diagnostics are secondary historical analyses, not rerun.',
        ranking='rank=1+count(other>designated); unique first also requires zero exact ties; no tolerance-defined ties',
        primary_counts=dict(comparison_A_positive_Bt_denominator=32,comparison_A_positive_Delta_denominator=32,
                            comparison_B_above_mean_denominator=32,comparison_B_unique_first_denominator=32),
        failure_policy='Technical failures resume exact saved optimizer/RNG/config; keep attempt logs. Nonfinite training failures stop that job and remain reported. Weak source learning never excludes a fixed20k endpoint.',
        stop_rule='Only units02..05, 64 endpoints/unit, fixed20k, original final-state main analyses and reporting. No extra seeds, independent-specialist initialization experiment, Graph/Group/Matrix/F17 work or publication.')
    assert all(u['data_seed'] != base['units'][0]['data_seed'] for u in units)
    hashes = source_files()
    write(DEFAULT_CONFIG, config)
    write(DIAG/'freeze.json',dict(protocol_sha256=sha(DEFAULT_CONFIG),source_sha256=hashes,
                               registered_utc=utc(),prior_results_seen=True,exploratory_repetition=True))
    write(Path(root)/'protocol.json',config)
    return config


def verify(path=DEFAULT_CONFIG):
    path=Path(path); c=read(path); f=read(path.with_name('freeze.json'))
    assert sha(path)==f['protocol_sha256'], 'Protocol changed after freeze'
    for p,h in f['source_sha256'].items():
        assert sha(WORKSPACE/p)==h, f'Frozen execution dependency changed: {p}'
    return c


def unit_config(c, unit):
    return next(u for u in c['units'] if u['unit']==unit)


def jobs(c, unit):
    # Interleave paired encodings for early paired coverage; no model shares
    # runtime RNG with another job, so execution ordering cannot change weights.
    for arch in c['architectures']:
        for task in list(DEV_TASKS)+[t for t in c['tasks'] if t not in DEV_TASKS]:
            for enc in c['encodings']:
                yield dict(run_id=f'{unit}_{arch}_{enc}_{task}',unit=unit,
                           architecture=arch,encoding=enc,task=task)


def prepare(c, unit):
    u=unit_config(c,unit); root=Path(c['output'])/unit
    history=c['history_index']; assert sha(history['path'])==history['sha256']
    index=np.load(history['path'],mmap_mode='r')
    if (root/'data.npz').exists() and not (root/'data_audit.json').exists():
        # A generator interruption before its audit receipt is a technical
        # failure, not a new seed. Preserve the orphan for diagnosis.
        orphan=root/'failed_attempts'/str(time.time_ns())
        orphan.mkdir(parents=True,exist_ok=True)
        for name in ['data.npz','batch_indices.npy','batch_lengths.npy']:
            if (root/name).exists():os.replace(root/name,orphan/name)
        event(root,kind='incomplete_data_generation_preserved',path=str(orphan))
    generate_world(c,u,index,history['metadata'],root)
    audit=read(root/'data_audit.json')
    assert audit['unit']==unit and audit['data_seed']==u['data_seed'] and audit['batch_seed']==u['batch_seed']
    assert audit['historical_index']['sha256']==history['sha256']
    assert sha(root/'data.npz')==audit['data_sha256']
    assert sha(root/'batch_indices.npy')==audit['batch_indices_sha256']
    assert sha(root/'batch_lengths.npy')==audit['batch_lengths_sha256']
    with np.load(root/'data.npz',allow_pickle=False) as data:
        for split,count in c['data']['examples_per_length'].items():
            ns=data[split+'_lengths']
            assert all(int((ns==n).sum())==count for n in c['lengths'])
            assert len(ns)==len(c['lengths'])*count
            assert data[split+'_labels'].shape==(len(ns),1 if split=='train' else 3,len(c['tasks']))
    schedule=np.load(root/'batch_indices.npy',mmap_mode='r')
    assert schedule.shape==(c['training']['updates'],c['training']['effective_batch'])
    # Verify all actual starts now, before any model is optimized.
    for arch in c['architectures']:
        initial(c,u,arch)
    write(root/'unit_config.json',dict(**u,protocol_sha256=sha(DEFAULT_CONFIG),data_audit=audit))
    return root


def initial(c,u,arch):
    root=Path(c['output'])/u['unit']/'initializations'/arch
    root.mkdir(parents=True,exist_ok=True); path=root/'step0.pt'
    if path.exists():
        meta=read(path.with_suffix('.json')); assert sha(path)==meta['checkpoint_sha256']
        assert meta['protocol_sha256']==sha(DEFAULT_CONFIG)
        saved=torch.load(path,map_location='cpu',weights_only=False)
        return saved,meta
    net=model(c,arch,u['architecture_model_seeds'][arch],'cpu')
    saved=dict(model=copy.deepcopy(net.state_dict()),rng=rng_state(),
        parameter_sha256=parameter_hash(net),unit=u['unit'],architecture=arch,
        model_seed=u['architecture_model_seeds'][arch],model_config=c['model'],
        vocabulary=dict(TOKEN_TO_ID),protocol_sha256=sha(DEFAULT_CONFIG),
        stage='actual_constructor_output_before_optimizer_or_training_forward')
    atomic_checkpoint(path,saved)
    meta=dict(checkpoint_sha256=sha(path),parameter_sha256=saved['parameter_sha256'],
              protocol_sha256=sha(DEFAULT_CONFIG),actual_initial_identity_verified=True,
              parameters=sum(p.numel() for p in net.parameters()))
    write(path.with_suffix('.json'),meta)
    return saved,meta


def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)


def start_state(c,u,job,init_meta):
    task=job['task'] if job['task'] in DEV_TASKS else 'descents'
    seed=u['dropout_seed_base']+1000*c['architectures'].index(job['architecture'])+DEV_TASKS.index(task)
    seed_all(seed); state=rng_state(); digest=rng_hash(state)
    path=Path(c['output'])/u['unit']/'initializations'/job['architecture']/f'start_rng_{task}.pt'
    if path.exists():
        old=torch.load(path,map_location='cpu',weights_only=False)
        assert old['rng_hash']==digest and old['parameter_sha256']==init_meta['parameter_sha256']
    else:
        atomic_checkpoint(path,dict(rng=state,rng_hash=digest,seed=seed,
                                   parameter_sha256=init_meta['parameter_sha256']))
    return state,seed,digest


def load_data(root):
    audit=read(root/'data_audit.json')
    assert sha(root/'data.npz')==audit['data_sha256']
    assert sha(root/'batch_indices.npy')==audit['batch_indices_sha256']
    return dict(np.load(root/'data.npz',allow_pickle=False)),np.load(root/'batch_indices.npy',mmap_mode='r'),audit


def train_job(c,job,stop_at=None,engineering_tag=None):
    u=unit_config(c,job['unit']); root=Path(c['output'])/job['unit']
    out=root/'models'/job['run_id'] if engineering_tag is None else root/'engineering'/engineering_tag
    out.mkdir(parents=True,exist_ok=True); path=out/'checkpoint.pt'; done=out/'completed.json'
    maximum=c['training']['updates'] if stop_at is None else stop_at
    assert 0<maximum<=c['training']['updates']
    signature=dict(protocol_sha256=sha(DEFAULT_CONFIG),source_sha256=read(DIAG/'freeze.json')['source_sha256'])
    if done.exists():
        d=read(done); assert d['signature']==signature and d['step']==20000
        assert sha(path)==d['checkpoint_sha256']
        return d
    data,schedule,audit=load_data(root); saved,im=initial(c,u,job['architecture'])
    device=device_setup(); net=model(c,job['architecture'],u['architecture_model_seeds'][job['architecture']],device)
    net.load_state_dict(saved['model']); actual=parameter_hash(net)
    assert actual==im['parameter_sha256']
    opt=optimizer(net,c['training'],device)
    start,training_seed,start_hash=start_state(c,u,job,im)
    binding=dict(**job,signature=signature,data_sha256=audit['data_sha256'],
                 batch_schedule_sha256=audit['batch_indices_sha256'],
                 initial_parameter_sha256=actual,initial_checkpoint_sha256=im['checkpoint_sha256'],
                 actual_loaded_initial_state_dict_sha256=actual,
                 training_seed=training_seed,starting_rng_hash=start_hash)
    step=0; curve=[];elapsed=0.;finite=0;low=math.inf;high=0.
    if path.exists():
        cp=torch.load(path,map_location='cpu',weights_only=False)
        for key,value in binding.items(): assert cp[key]==value, ('Resume binding mismatch',key)
        net.load_state_dict(cp['model']);opt.load_state_dict(cp['optimizer']);restore_rng(cp['rng'])
        step=cp['step'];curve=cp['curve'];elapsed=cp['elapsed_seconds'];finite=cp['finite_gradient_updates']
        low=cp['minimum_gradient_norm'];high=cp['maximum_gradient_norm']
    else:restore_rng(start)
    assert step<=maximum
    started=time.monotonic(); event(root,kind='training_invocation',run_id=job['run_id'],start_step=step,stop_step=maximum,engineering=engineering_tag)
    for idx in range(step,maximum):
        try:
            value,norm=update(net,opt,data,schedule,job,idx,c['training'],device)
        except AssertionError as error:
            if 'assert torch.isfinite(value)' in traceback.format_exc():
                raise FloatingPointError(f'Nonfinite training loss at update {idx+1}') from error
            raise
        step=idx+1;finite+=1;low=min(low,norm);high=max(high,norm)
        if step%c['training']['validation_every']==0 or step==maximum:
            rec=dict(step=step,train_loss=value,**accuracy(net,data,job['encoding'],job['task'],device))
            curve.append(rec)
            print(json.dumps(dict(run_id=job['run_id'],elapsed_seconds=elapsed+time.monotonic()-started,**rec)),flush=True)
        if step%c['training']['checkpoint_every']==0 or step==maximum:
            atomic_checkpoint(path,dict(**binding,model=net.state_dict(),optimizer=opt.state_dict(),rng=rng_state(),
                step=step,curve=curve,elapsed_seconds=elapsed+time.monotonic()-started,
                finite_gradient_updates=finite,minimum_gradient_norm=low,maximum_gradient_norm=high))
            write(out/'progress.json',dict(run_id=job['run_id'],step=step,planned_updates=c['training']['updates'],
                elapsed_seconds=elapsed+time.monotonic()-started,status='checkpointed',updated_utc=utc()))
    result=dict(**binding,step=step,checkpoint_sha256=sha(path),parameter_sha256=parameter_hash(net),
        finite_gradient_updates=finite,elapsed_seconds=elapsed+time.monotonic()-started,curve=curve,
        parameters=sum(v.numel() for v in net.parameters()),exposures=step*c['training']['effective_batch'],
        supervised_tokens=2*step*c['training']['effective_batch'],actual_initial_identity_verified=True)
    if maximum==c['training']['updates'] and engineering_tag is None:
        assert finite==20000 and result['parameter_sha256']!=actual
        write(done,result)
        event(root,kind='endpoint_completed',run_id=job['run_id'],step=step,elapsed_seconds=result['elapsed_seconds'])
    del net,opt,data;torch.cuda.empty_cache()
    return result


def feature_path(root,job,phase,split):
    return root/'features'/f'{job["architecture"]}_{job["encoding"]}_{job["task"]}_{phase}_{split}.npz'


def extract_job(c,job,engineering=False):
    root=Path(c['output'])/job['unit']; data,_,audit=load_data(root); u=unit_config(c,job['unit'])
    saved,im=initial(c,u,job['architecture']);device=device_setup();records=[]
    if engineering:
        data={key:(value[:128] if key.startswith(('test_','validation_')) else value) for key,value in data.items()}
    for phase in ['initial','trained']:
        cp_path=root/'initializations'/job['architecture']/'step0.pt' if phase=='initial' else root/'models'/job['run_id']/'checkpoint.pt'
        if engineering and phase=='trained':cp_path=root/'engineering'/f'{job["architecture"]}_{job["encoding"]}_{job["task"]}_resumed'/'checkpoint.pt'
        checkpoint_hash=sha(cp_path)
        if phase=='trained':
            cp=torch.load(cp_path,map_location='cpu',weights_only=False)
            assert cp['run_id']==job['run_id'] and cp['unit']==job['unit']
            assert cp['signature']['protocol_sha256']==sha(DEFAULT_CONFIG)
            assert cp['data_sha256']==audit['data_sha256']
            assert cp['initial_parameter_sha256']==im['parameter_sha256']
            if not engineering:
                done=read(cp_path.with_name('completed.json'))
                assert cp['step']==cp['finite_gradient_updates']==20000
                assert done['checkpoint_sha256']==checkpoint_hash
                assert cp['signature']['source_sha256']==read(DIAG/'freeze.json')['source_sha256']
                assert cp['batch_schedule_sha256']==audit['batch_indices_sha256']
            del cp
        net=None
        for split in ['validation','test']:
            path=feature_path(root,job,phase,split)
            if engineering:path=root/'engineering/features'/path.name
            path.parent.mkdir(parents=True,exist_ok=True)
            signature=dict(checkpoint_sha256=checkpoint_hash,data_sha256=audit['data_sha256'],
                protocol_sha256=sha(DEFAULT_CONFIG),architecture=job['architecture'],encoding=job['encoding'],
                task=job['task'],phase=phase,split=split,point='input end before task token, final_norm',
                parameter_dtype='float32',forward_autocast=False,engineering=engineering)
            if path.exists():
                if path.with_suffix('.json').exists():
                    meta=read(path.with_suffix('.json'));assert meta['signature']==signature and sha(path)==meta['sha256']
                    records.append(meta);continue
                orphan=path.with_name(path.stem+f'.orphan_{time.time_ns()}.npz')
                os.replace(path,orphan)
                event(root,kind='incomplete_feature_cache_preserved',path=str(orphan),run_id=job['run_id'])
            if net is None:
                cp=torch.load(cp_path,map_location='cpu',weights_only=False)
                net=model(c,job['architecture'],u['architecture_model_seeds'][job['architecture']],device)
                net.load_state_dict(cp['model']);net.eval();before=parameter_hash(net)
                assert phase!='initial' or before==im['parameter_sha256']
            arrays=extract(net,data,job['encoding'],job['task'],device,split)
            # Independent pure-prefix/manual forward, first eight anchors of
            # fixed lengths. No training-time/test-selected subset gate.
            checks=[]
            with torch.inference_mode():
                for n in ([10] if engineering else [10,20,30]):
                    selected=np.flatnonzero(data[split+'_lengths']==n)[:8]
                    prefixes=data[f'{split}_{job["encoding"]}_input'][selected,:,:2*n+4]
                    ids=torch.as_tensor(prefixes.reshape(-1,2*n+4).astype(np.int64),device=device)
                    pred=manual_final(net,ids,ids.ne(TOKEN_TO_ID['<PAD>']),job['architecture'])[:,-1]
                    manual=pred.cpu().numpy().reshape(len(selected),3,c['model']['d_model'])
                    error=float(np.max(np.abs(manual-arrays['prefix_final'][selected])))
                    assert error<=2e-5,(job,phase,split,n,error)
                    checks.append(dict(length=n,samples=len(selected),max_absolute_error=error,tolerance=2e-5))
            assert before==parameter_hash(net), 'Forward pass changed parameters'
            temp=path.with_name(path.stem+'.tmp.npz');np.savez(temp,**arrays);os.replace(temp,path)
            meta=dict(signature=signature,sha256=sha(path),arrays={k:dict(shape=list(v.shape),dtype=str(v.dtype)) for k,v in arrays.items()},
                      parameter_sha256_before=before,parameter_sha256_after=parameter_hash(net),manual_checks=checks)
            write(path.with_suffix('.json'),meta);records.append(meta)
        del net;torch.cuda.empty_cache()
    return records


def engineering(c,unit):
    """Actual updates + on-disk resume + extraction; excluded from formal jobs."""
    existing=DIAG/'engineering_audit.json'
    if existing.exists():
        passed=read(existing)
        assert passed['protocol_sha256']==sha(DEFAULT_CONFIG)
        assert passed['status']=='passed'
        return passed
    root=prepare(c,unit); records=[]; begun=time.monotonic()
    for arch in c['architectures']:
        for enc,task in itertools.product(c['encodings'],DEV_TASKS):
            job=next(j for j in jobs(c,unit) if j['architecture']==arch and j['encoding']==enc and j['task']==task)
            prefix=f'{arch}_{enc}_{task}'
            continuous=train_job(c,job,stop_at=40,engineering_tag=prefix+'_continuous')
            middle=root/'engineering'/f'{prefix}_resumed/checkpoint.pt'
            if not middle.exists() or torch.load(middle,map_location='cpu',weights_only=False)['step']<20:
                train_job(c,job,stop_at=20,engineering_tag=prefix+'_resumed')
            resumed=train_job(c,job,stop_at=40,engineering_tag=prefix+'_resumed')
            a=torch.load(root/'engineering'/f'{prefix}_continuous/checkpoint.pt',map_location='cpu',weights_only=False)
            b=torch.load(root/'engineering'/f'{prefix}_resumed/checkpoint.pt',map_location='cpu',weights_only=False)
            difference=max(float((a['model'][k]-b['model'][k]).abs().max()) for k in a['model'])
            assert difference<=1e-5 and rng_hash(a['rng'])==rng_hash(b['rng'])
            optimizer_difference=0.
            for k,state in a['optimizer']['state'].items():
                for key,value in state.items():
                    other=b['optimizer']['state'][k][key]
                    if torch.is_tensor(value):optimizer_difference=max(optimizer_difference,float((value-other).abs().max()))
                    else:assert value==other
            assert optimizer_difference<=1e-5
            features=extract_job(c,job,engineering=True)
            # CKA independently computed in feature and sample-Gram forms.
            feature=feature_path(root,job,'trained','test').name
            x=np.load(root/'engineering/features'/feature)['prefix_final'].astype(np.float64)
            x=x[:,0]-x[:,0].mean(0); y=np.roll(x,1,axis=0)
            cross=x.T@y; s1=float(np.sum(cross*cross)/np.sqrt(np.sum((x.T@x)**2)*np.sum((y.T@y)**2)))
            k=x@x.T;l=y@y.T;s2=float(np.sum(k*l)/np.sqrt(np.sum(k*k)*np.sum(l*l)))
            assert abs(s1-s2)<=1e-10
            records.append(dict(architecture=arch,encoding=enc,task=task,updates=40,resume_step=20,
                max_parameter_difference=difference,max_optimizer_difference=optimizer_difference,
                identical_final_rng=True,parameters_changed=continuous['parameter_sha256']!=continuous['initial_parameter_sha256'],
                feature_records=len(features),cka_independent_error=abs(s1-s2),
                continuous_process_seconds=continuous['elapsed_seconds'],
                no_accuracy_or_geometry_sign_gate=True))
    # Real model -> disk -> extraction -> both original comparisons, with a
    # bounded four-task/n10 engineering dataset. These are never formal models.
    data,_,_=load_data(root); miniature={}
    columns=[TASKS.index(t) for t in DEV_TASKS]
    for key,value in data.items():
        if key.startswith('train_'): selected=value[:128]
        else:selected=value[:128]
        miniature[key]=selected[:,:,columns] if key.endswith('_labels') else selected
    np.savez(root/'engineering/data.npz',**miniature)
    mini=copy.deepcopy(c);mini['tasks']=list(DEV_TASKS);mini['label_task_names']=list(DEV_TASKS)
    mini['lengths']=[10];mini['length_weights']={'10':1.0}
    mini['relations']=[r for r in c['relations'] if r['left'] in DEV_TASKS and r['right'] in DEV_TASKS]
    from .analysis import analyze_unit
    measured=analyze_unit(mini,root/'engineering',root/'engineering/analysis')
    assert measured['status']=='completed'
    record=dict(status='passed',protocol_sha256=sha(DEFAULT_CONFIG),engineering_only=True,formal_endpoints=0,records=records,
                miniature_analysis=measured,small_end_to_end_complete=True,
                elapsed_seconds=time.monotonic()-begun,updated_utc=utc())
    write(DIAG/'engineering_audit.json',record);write(root/'engineering_audit.json',record)
    return record


def status(c):
    units=[]
    for u in c['units']:
        root=Path(c['output'])/u['unit']; completed=[]; partial=[]
        for j in jobs(c,u['unit']):
            d=root/'models'/j['run_id']; done=d/'completed.json'; progress=d/'progress.json'
            if done.exists():
                rec=read(done);completed.append(dict(run_id=j['run_id'],step=rec['step'],elapsed_seconds=rec['elapsed_seconds']))
            elif progress.exists():partial.append(read(progress))
        analysis=root/'analysis/completion.json'
        units.append(dict(unit=u['unit'],planned_models=64,completed_models=len(completed),
            completed=completed,partial=partial,data_prepared=(root/'data_audit.json').exists(),
            main_analysis_complete=analysis.exists() and read(analysis).get('status')=='completed'))
    result=dict(updated_utc=utc(),historical_completed_units=1,additional_units=units,
                planned_total_units=5,planned_additional_models=256,
                completed_additional_models=sum(u['completed_models'] for u in units),
                complete=all(u['main_analysis_complete'] for u in units))
    write(DIAG/'execution_status.json',result);write(Path(c['output'])/'execution_status.json',result)
    return result


def run_all(c,unit_filter=None):
    root=Path(c['output']);root.mkdir(parents=True,exist_ok=True)
    with (root/'execution.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        write(root/'active_process.json',dict(pid=os.getpid(),host=platform.node(),started_utc=utc(),
              config=str(DEFAULT_CONFIG),protocol_sha256=sha(DEFAULT_CONFIG)))
        assert read(DIAG/'engineering_audit.json')['status']=='passed', 'Engineering check required'
        for u in c['units']:
            unit=u['unit']
            if unit_filter and unit!=unit_filter:continue
            prepare(c,unit)
            for job in jobs(c,unit):
                try:
                    train_job(c,job); extract_job(c,job);status(c)
                except BaseException as error:
                    event(root,kind='failure',run_id=job['run_id'],error_type=type(error).__name__,
                          error=str(error),traceback=traceback.format_exc(),
                          classification='nonfinite_training' if isinstance(error,FloatingPointError) or 'non-finite' in str(error).lower() else 'technical_or_unclassified',
                          policy='No automatic seed replacement; resume same configuration after diagnosis.')
                    status(c);raise
            from .analysis import analyze_unit
            analyze_unit(c,root/unit,root/unit/'analysis')
            refresh_summary(c)
        refresh_summary(c);status(c)
        write(root/'pipeline_exit.json',dict(status='finished_requested_manifest',utc=utc()))


def refresh_summary(c):
    from .analysis import summarize_units, import_historical
    historical=DIAG/'historical_analysis'
    import_historical(WORKSPACE,historical,unit='archdev001')
    records=[dict(unit='archdev001',analysis_dir=str(historical),status='completed')]
    for u in c['units']:
        path=Path(c['output'])/u['unit']/'analysis'
        records.append(dict(unit=u['unit'],analysis_dir=str(path),status='completed' if (path/'completion.json').exists() and read(path/'completion.json')['status']=='completed' else 'pending'))
    result=summarize_units(records,DIAG/'summary')
    from .reporting import update_paper_status
    update_paper_status(DIAG/'summary',WORKSPACE)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['freeze','prepare','engineering','run','job','extract','analyze','summary','status'])
    p.add_argument('--config',type=Path,default=DEFAULT_CONFIG)
    p.add_argument('--root',type=Path,default=DEFAULT_ROOT)
    p.add_argument('--unit',default=None);p.add_argument('--run-id')
    a=p.parse_args()
    assert a.config.resolve()==DEFAULT_CONFIG.resolve(), 'This registered run uses one fixed protocol path'
    if a.stage=='freeze':freeze(a.root);return
    c=verify(a.config)
    if a.stage=='prepare':prepare(c,a.unit or 'unit02')
    elif a.stage=='engineering':engineering(c,a.unit or 'unit02')
    elif a.stage=='status':print(json.dumps(status(c),indent=2))
    elif a.stage=='summary':refresh_summary(c)
    elif a.stage=='run':run_all(c,a.unit)
    elif a.stage=='analyze':
        from .analysis import analyze_unit
        root=Path(c['output'])/(a.unit or 'unit02');analyze_unit(c,root,root/'analysis')
    elif a.stage in ['job','extract']:
        j=next(j for u in c['units'] for j in jobs(c,u['unit']) if j['run_id']==a.run_id)
        train_job(c,j) if a.stage=='job' else extract_job(c,j)


if __name__=='__main__':main()
