"""Read-only, resumable caches for the fixed eight-relation cohort.

The first16 preserve their original separately saved primary final-norm arrays.
New initial prefix states alias the verified original shared hidden arrays, but
each new task's output probabilities/prediction/CE comes from its actual step0.
No optimizer, backward, update, task selection, or new input generation occurs.
"""
from __future__ import annotations

import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import time

import numpy as np
import torch
from torch.nn import functional as F

from .common import (
    ROOT, DIAG, CONFIG, SOURCE_ROOT, OLD_ANALYSIS, TASKS, TOKEN_TO_ID,
    verify, initial, model, device_setup, parameter_hash, sha, js,
    source_version as training_source_version,
)

OLD_EXTRACT = OLD_ANALYSIS/'extraction_review/extract_layers.py'
spec = importlib.util.spec_from_file_location('eight_relations_frozen_capture', OLD_EXTRACT)
original = importlib.util.module_from_spec(spec)
spec.loader.exec_module(original)
LAYERS = original.LAYERS
TOLERANCE = original.TOLERANCE
CACHE = ROOT/'evaluation'
INDEX = CACHE/'index.json'
SPLITS = ('validation', 'test')
PHASES = ('initial', 'trained')
N_SAMPLES = {'validation':5376, 'test':10752}
EXPECTED_SHAPES = {'layers':(3,5,256), 'prefix_final':(3,256),
                   'output_prob':(3,31), 'prediction':(3,), 'answer_ce':(3,)}


def write_json(path, obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    os.replace(temporary,path)


def array_hash(value):
    a=np.ascontiguousarray(value)
    h=hashlib.sha256();h.update(str(a.dtype).encode());h.update(str(a.shape).encode())
    view=memoryview(a).cast('B')
    for start in range(0,len(view),4*1024*1024):h.update(view[start:start+4*1024*1024])
    return h.hexdigest()


def describe(path, layer_order=None):
    path=Path(path).resolve();a=np.load(path,mmap_mode='r',allow_pickle=False)
    result=dict(path=str(path),sha256=sha(path),array_sha256=array_hash(a),
                shape=list(a.shape),dtype=str(a.dtype))
    if layer_order is not None:result['layer_order']=list(layer_order)
    assert not a.flags.writeable
    return result


def validate_array(description, name, N, hashed):
    path=Path(description['path'])
    assert path.is_absolute() and path.suffix=='.npy' and path.is_file()
    if str(path) not in hashed:
        a=np.load(path,mmap_mode='r',allow_pickle=False)
        assert not a.flags.writeable
        assert sha(path)==description['sha256'],('Array file changed',str(path))
        assert array_hash(a)==description['array_sha256'],('Array content changed',str(path))
        hashed[str(path)]=dict(shape=list(a.shape),dtype=str(a.dtype))
    assert hashed[str(path)]==dict(shape=description['shape'],dtype=description['dtype'])
    assert tuple(description['shape'])==(N,*EXPECTED_SHAPES[name]),(name,description['shape'],N)
    if name in ('layers','prefix_final','output_prob'):assert description['dtype']=='float32'
    elif name=='prediction':assert np.dtype(description['dtype']).kind in 'iu'
    elif name=='answer_ce':assert description['dtype']=='float64'


def key(job,condition,split):return '|'.join((job['run_id'],condition,split))


def record_path(job,condition,split):return CACHE/key(job,condition,split).replace('|','__')/'record.json'


def analysis_verify():
    c=verify();path=DIAG/'analysis_freeze.json'
    assert path.is_file(),'Freeze analysis sources before any reuse/extraction/evaluation'
    frozen=js(path);assert frozen['protocol_sha256']==sha(CONFIG)
    assert frozen['source_sha256'].get(str(Path(__file__).resolve()))==sha(__file__),'Extractor itself is not frozen'
    assert frozen['source_sha256'].get(str(OLD_EXTRACT))==sha(OLD_EXTRACT),'Original capture/manual implementation is not frozen'
    for p,h in frozen['source_sha256'].items():assert sha(p)==h,('Analysis source changed',p)
    assert c['actions']==['identity','complement','inverse'] and c['lengths']==list(range(10,31))
    assert len(c['checkpoint_inventory'])==64 and len(c['training_manifest'])==48
    assert Path(c['evaluation_index_path']).resolve()==INDEX.resolve()
    version=hashlib.sha256(json.dumps(frozen['source_sha256'],sort_keys=True,separators=(',',':')).encode()).hexdigest()
    return c,version


def old_sources(c):
    path=Path(c['old_layer_index_path']);index=js(path)
    assert index['status']=='completed' and index['initial_shared_exact'] is True
    protection=js(DIAG/'protected_files.json')['files']
    assert str(path) in protection and sha(path)==protection[str(path)]
    assert index['source_data_sha256']==c['data_sha256']
    assert len(index['records'])==40 and len(index['job_condition_split_map'])==64
    physical={r['source_key']:r for r in index['records']}
    return index,physical,sha(path)


def checkpoint_source(c,job,condition):
    if condition=='initial':
        ref=c['initializations'][job['architecture']]
        return Path(ref['path']),ref['sha256'],ref['parameter_sha256']
    if job['task'] in ('descents','recoils','peaks','valleys'):
        return Path(job['checkpoint_path']),job['checkpoint_sha256'],job.get('parameter_sha256')
    path=ROOT/'models'/job['run_id']/'checkpoint.pt';completed=js(path.with_name('completed.json'))
    assert Path(job['checkpoint_path']).resolve()==path.resolve()
    assert completed['run_id']==job['run_id'] and completed['step']==completed['finite_gradient_updates']==20000
    assert completed['protocol_sha256']==sha(CONFIG) and completed['data_sha256']==c['data_sha256']
    assert completed['execution_source_version_sha256']==training_source_version()
    assert completed['batch_schedule_sha256']==c['batch_schedule_sha256']
    assert completed['initial_parameter_sha256']==c['initializations'][job['architecture']]['parameter_sha256']
    assert completed['initial_checkpoint_sha256']==c['initializations'][job['architecture']]['sha256']
    assert completed['starting_rng_hash']==c['new_task_start_rng'][job['architecture']]['rng_hash']
    assert completed['exposures']==1280000 and completed['supervised_tokens']==2560000
    return path,completed['checkpoint_sha256'],completed['parameter_sha256']


def blank_index(c,version):
    return dict(schema_version='permworld-eight-relations-cache/v1',status='incomplete',
        protocol_sha256=sha(CONFIG),analysis_source_version_sha256=version,
        source_data_sha256=c['data_sha256'],actions=c['actions'],layers=LAYERS,
        records=[],initial_shared_exact=False,expected_logical_records=256,
        old_initial_CE_missing_records=32,new_initial_CE_evaluated=True,
        primary_final_norm='separate prefix_final array; never replaced by packed layer norm',
        row_order='Original validation/test anchor index; action identity/complement/inverse',
        forward_batch_examples=128,prompt_shape='Original full prompt contains task and equals; sequence width2*max(n)+6, not padding to128',
        source_updates=0,new_training=False,optimizer_constructed=False,backward=False,
        new_anchors=False,old_artifacts_modified=False)


def load_index(c,version):
    if not INDEX.exists():return blank_index(c,version)
    obj=js(INDEX)
    assert obj['protocol_sha256']==sha(CONFIG) and obj['analysis_source_version_sha256']==version
    assert obj['source_data_sha256']==c['data_sha256']
    assert len(obj['records'])==len({(r['run_id'],r['condition'],r['split']) for r in obj['records']})
    return obj


def base_record(c,version,job,condition,split,cp_path,cp_sha):
    return dict(run_id=job['run_id'],architecture=job['architecture'],encoding=job['encoding'],
        task=job['task'],condition=condition,split=split,protocol_sha256=sha(CONFIG),
        analysis_source_version_sha256=version,source_data_sha256=c['data_sha256'],
        source_checkpoint_path=str(cp_path),source_checkpoint_sha256=cp_sha,
        actions=list(c['actions']),row_order='Original anchor index',arrays={},
        source_updates=0,source_weights_modified=False,EOS_evaluated=False)


def validate_record(c,version,job,record,condition,split,hashed,cp_sha=None):
    assert record['run_id']==job['run_id'] and record['condition']==condition and record['split']==split
    for f in ('architecture','encoding','task'):assert record[f]==job[f]
    assert record['protocol_sha256']==sha(CONFIG) and record['analysis_source_version_sha256']==version
    assert record['source_data_sha256']==c['data_sha256']
    if cp_sha is not None:assert record['source_checkpoint_sha256']==cp_sha
    names=set(record['arrays']);assert {'layers','prefix_final','output_prob','prediction'}<=names
    if condition=='trained' or job['task'] not in ('descents','recoils','peaks','valleys'):assert 'answer_ce' in names
    else:assert 'answer_ce' not in names and record.get('initial_CE_missing_reason')
    assert record['arrays']['layers']['layer_order']==LAYERS[job['architecture']]
    for name,description in record['arrays'].items():
        assert name in EXPECTED_SHAPES
        validate_array(description,name,N_SAMPLES[split],hashed)
    if record.get('reused_original'):
        checks=record['original_array_conversion_checks']
        assert set(checks)=={'prefix_final','output_prob','prediction'}
        for name,check in checks.items():
            assert check['conversion_bitwise_equal'] is True
            assert check['original_array_sha256']==check['converted_array_sha256']==record['arrays'][name]['array_sha256']
    elif split=='test':
        checks=record['independent_manual_checks']
        assert {(r['length'],r['action']) for r in checks}=={(n,a) for n in (10,20,30) for a in c['actions']}
        assert len(checks)==9
        for check in checks:
            assert check['split']=='test' and check['checked_samples']==8 and check['full_prompt_samples']==128
            assert check['saved_layer_max_absolute_difference']<=TOLERANCE
            assert check['saved_primary_norm_max_absolute_difference']<=TOLERANCE
            assert check['saved_packed_vs_primary_norm_max_absolute_difference']==0.
            assert check['saved_layers_path']==record['arrays']['layers']['path']
            assert check['saved_layers_sha256']==record['arrays']['layers']['sha256']
            assert check['saved_prefix_final_path']==record['arrays']['prefix_final']['path']
            assert check['saved_prefix_final_sha256']==record['arrays']['prefix_final']['sha256']
            if condition=='initial':assert check['saved_initial_alias_fields_verified'] is True


def save_record(index,record,job,condition,split):
    write_json(record_path(job,condition,split),record)
    existing={(r['run_id'],r['condition'],r['split']):r for r in index['records']}
    existing[job['run_id'],condition,split]=record
    index['records']=sorted(existing.values(),key=lambda r:(r['run_id'],r['condition'],r['split']))
    index['completed_logical_records']=len(index['records'])
    write_json(INDEX,index)


def atomic_npy(path,value):
    path=Path(path);temporary=path.with_name(path.stem+'.partial.npy')
    np.save(temporary,value,allow_pickle=False);os.replace(temporary,path)


def convert_old_array(path,original_array):
    """Decode each NPZ field once and prove the saved mmap preserves every value."""
    original_sha=array_hash(original_array)
    atomic_npy(path,original_array)
    description=describe(path)
    saved=np.load(path,mmap_mode='r',allow_pickle=False)
    exact=bool(np.array_equal(saved,original_array))
    assert exact and description['array_sha256']==original_sha
    return description,dict(original_array_sha256=original_sha,
        converted_array_sha256=description['array_sha256'],conversion_bitwise_equal=exact)


def reuse():
    c,version=analysis_verify();CACHE.mkdir(parents=True,exist_ok=True)
    index=load_index(c,version);old_index,physical,index_sha=old_sources(c)
    hashed={};metadata={};started=time.monotonic();npz_decoded=0
    for job in c['checkpoint_inventory']:
        if job['task'] not in ('descents','recoils','peaks','valleys'):continue
        for condition in PHASES:
            cp_path,cp_sha,_=checkpoint_source(c,job,condition)
            for split in SPLITS:
                path=record_path(job,condition,split)
                if path.exists():
                    rec=js(path);validate_record(c,version,job,rec,condition,split,hashed,cp_sha)
                    save_record(index,rec,job,condition,split);continue
                logical=key(job,condition,split);source_key=old_index['job_condition_split_map'][logical]
                packed=physical[source_key];raw=old_index['logical_old_cache_map'][logical]
                assert packed['checkpoint_sha256']==cp_sha
                source_metadata=js(Path(raw['path']).with_suffix('.json'))
                signature=source_metadata['signature']
                assert signature['checkpoint_sha256']==cp_sha and signature['data_sha256']==c['data_sha256']
                assert sha(raw['path'])==raw['sha256']==source_metadata['sha256']
                out=path.parent;out.mkdir(parents=True,exist_ok=True)
                rec=base_record(c,version,job,condition,split,cp_path,cp_sha)
                rec.update(reused_original=True,old_logical_key=logical,old_layer_source_key=source_key,
                    original_layer_index_path=c['old_layer_index_path'],original_layer_index_sha256=index_sha,
                    original_NPZ_path=raw['path'],original_NPZ_sha256=raw['sha256'])
                feature=Path(packed['features_path']);assert sha(feature)==packed['features_sha256']
                if str(feature) not in metadata:metadata[str(feature)]=describe(feature,LAYERS[job['architecture']])
                rec['arrays']['layers']=metadata[str(feature)]
                # Each old logical NPZ is opened once. Primary norm is its exact
                # saved prefix_final, never the tolerance-equal packed norm.
                with np.load(raw['path'],allow_pickle=False) as ar:
                    rec['original_array_conversion_checks']={}
                    for name in ('prefix_final','output_prob','prediction'):
                        target=out/(name+'.npy')
                        rec['arrays'][name],rec['original_array_conversion_checks'][name]=convert_old_array(target,ar[name])
                npz_decoded+=1
                if condition=='trained':
                    ce=Path(packed['answer_ce_path']);assert sha(ce)==packed['answer_ce_sha256']
                    rec['arrays']['answer_ce']=describe(ce)
                else:
                    rec['initial_CE_missing_reason']='Not saved in original first16 cache; no original task-specific initial CE newly evaluated.'
                validate_record(c,version,job,rec,condition,split,hashed,cp_sha)
                save_record(index,rec,job,condition,split)
    records={(r['run_id'],r['condition'],r['split']):r for r in index['records']}
    old_jobs=[j for j in c['checkpoint_inventory'] if j['task'] in ('descents','recoils','peaks','valleys')]
    assert len(old_jobs)==16
    equality=[]
    for a in c['architectures']:
        for e in c['encodings']:
            group=[j for j in old_jobs if j['architecture']==a and j['encoding']==e]
            representative=next(j for j in group if j['task']=='descents')
            for split in SPLITS:
                base=records[representative['run_id'],'initial',split]
                for job in group:
                    other=records[job['run_id'],'initial',split]
                    assert base['arrays']['prefix_final']['array_sha256']==other['arrays']['prefix_final']['array_sha256']
                    assert base['arrays']['layers']['array_sha256']==other['arrays']['layers']['array_sha256']
                    equality.append(dict(run_id=job['run_id'],split=split,representative_run_id=representative['run_id'],bitwise_equal=True))
    index['old_initial_equality']=equality;index['old_shared_initial_verified']=True
    index['old_sources_reused']=64;index['reuse_wall_seconds']=time.monotonic()-started
    if len(index['records'])<256:index['status']='old_reuse_completed_new_sources_pending'
    write_json(INDEX,index)
    write_json(DIAG/'extraction_reuse_audit.json',dict(status='passed',logical_original_splits=64,
        old_NPZ_archives_decoded_this_invocation=npz_decoded,old_NPZ_archives_redecoded_for_relations=0,
        original_norm_retained=True,old_initial_CE_missing=32,weights_loaded=False,
        old_array_conversion_original_to_saved_hash_and_bitwise_verified=True,
        forward=False,training=False,index_sha256=sha(INDEX),elapsed_wall_seconds=index['reuse_wall_seconds']))
    return index


def eval_data(c):
    assert sha(c['data_path'])==c['data_sha256']
    with np.load(c['data_path'],allow_pickle=False) as ar:
        data={f'{s}_{k}':ar[f'{s}_{k}'].copy() for s in SPLITS
              for k in ('lengths','labels','one_line_input','canonical_cycle_input')}
    for split in SPLITS:
        assert len(data[split+'_lengths'])==N_SAMPLES[split]
        assert data[split+'_labels'].shape==(N_SAMPLES[split],3,16)
        assert all(int((data[split+'_lengths']==n).sum())==(256 if split=='validation' else 512) for n in c['lengths'])
    return data


def load_model(c,job,condition,device):
    path,digest,parameter=checkpoint_source(c,job,condition)
    assert sha(path)==digest
    cp=torch.load(path,map_location='cpu',weights_only=False)
    if condition=='trained':
        completed=js(path.with_name('completed.json'))
        assert cp['run_id']==job['run_id'] and cp['step']==cp['finite_gradient_updates']==20000
        assert cp['data_sha256']==c['data_sha256'] and cp['initial_parameter_sha256']==c['initializations'][job['architecture']]['parameter_sha256']
        assert cp['batch_schedule_sha256']==c['batch_schedule_sha256']
        assert cp['initial_checkpoint_sha256']==c['initializations'][job['architecture']]['sha256']
        assert cp['protocol_sha256']==sha(CONFIG) and cp['execution_source_version_sha256']==completed['execution_source_version_sha256']
    else:assert cp['parameter_sha256']==c['initializations'][job['architecture']]['parameter_sha256']
    net=model(c,job['architecture'],job['model_seed'],device)
    net.load_state_dict(cp['model'],strict=True);net.eval()
    assert all(p.dtype==torch.float32 and p.grad is None for p in net.parameters())
    assert parameter_hash(net)==parameter
    del cp
    return net,path,digest,parameter


@torch.inference_mode()
def answer_outputs(logits,labels):
    """Conditional numeric probabilities, full188 prediction and full188 CE."""
    assert logits.dtype==torch.float32 and logits.shape[1]==188 and torch.isfinite(logits).all()
    truth=torch.as_tensor([TOKEN_TO_ID[f'{int(v):02d}'] for v in labels],device=logits.device)
    return (F.softmax(logits[:,:31],dim=-1).cpu().numpy(),
            logits.argmax(-1).cpu().numpy(),
            F.cross_entropy(logits,truth,reduction='none').cpu().double().numpy())


@torch.inference_mode()
def independent_manual(c,job,condition,net,capture,data,device,reference):
    checks=[];references=[]
    for n in (10,20,30):
        use=np.flatnonzero(data['test_lengths']==n)[:128]
        assert len(use)==128
        # The first fixed128 examples must be exactly one normal extraction
        # batch, so the manual and stored paths use the same FP32 shapes.
        assert use[0]%128==0 and np.array_equal(use,np.arange(use[0],use[0]+128))
        for action in range(3):
            prefixes=data[f'test_{job["encoding"]}_input'][use,action]
            hidden,_,ids,mask,ends=original.forward(net,capture,prefixes,data['test_lengths'][use],job['task'],device,job['architecture'])
            hook_norm=capture.states['final_norm'].copy()
            manual=original.manual(net,ids,mask,ends,job['architecture'])
            discrepancy=float(np.max(np.abs(hidden[:8]-manual[:8])))
            assert discrepancy<=TOLERANCE
            norm=hook_norm[:8]
            norm_discrepancy=float(np.max(np.abs(hidden[:8,-1]-norm)))
            assert norm_discrepancy==0.
            row=dict(run_id=job['run_id'],architecture=job['architecture'],encoding=job['encoding'],condition=condition,
                split='test',length=n,action=c['actions'][action],full_prompt_samples=128,checked_samples=8,
                manual_layer_max_absolute_difference=discrepancy,packed_norm_vs_separate_norm=norm_discrepancy)
            if condition=='initial':
                shared=np.load(reference['arrays']['prefix_final']['path'],mmap_mode='r',allow_pickle=False)
                error=float(np.max(np.abs(norm-shared[use[:8],action])))
                assert error<=TOLERANCE
                row['initial_task_prefix_vs_original_norm_max_absolute_difference']=error
            checks.append(row)
            references.append(dict(indices=use[:8].copy(),action_index=action,manual_layers=manual[:8].copy()))
    return checks,references


def verify_saved_manual(record,checks,references):
    """Independent manual states are compared directly with the final mmap files."""
    assert record['split']=='test' and len(checks)==len(references)
    layers_desc=record['arrays']['layers'];prefix_desc=record['arrays']['prefix_final']
    saved_layers=np.load(layers_desc['path'],mmap_mode='r',allow_pickle=False)
    saved_prefix=np.load(prefix_desc['path'],mmap_mode='r',allow_pickle=False)
    result=[]
    for check,reference in zip(checks,references):
        use=reference['indices'];action=reference['action_index'];manual=reference['manual_layers']
        assert len(use)==8 and manual.shape==(8,5,256)
        packed=saved_layers[use,action];prefix=saved_prefix[use,action]
        layer_error=float(np.max(np.abs(packed-manual)))
        norm_error=float(np.max(np.abs(prefix-manual[:,-1])))
        same_forward_error=float(np.max(np.abs(packed[:,-1]-prefix)))
        assert layer_error<=TOLERANCE and norm_error<=TOLERANCE and same_forward_error==0.
        row=dict(check,anchor_indices=use.tolist(),saved_layer_max_absolute_difference=layer_error,
            saved_primary_norm_max_absolute_difference=norm_error,
            saved_packed_vs_primary_norm_max_absolute_difference=same_forward_error,
            saved_layers_path=layers_desc['path'],saved_layers_sha256=layers_desc['sha256'],
            saved_prefix_final_path=prefix_desc['path'],saved_prefix_final_sha256=prefix_desc['sha256'],
            direct_manual_to_saved_arrays=True,tolerance=TOLERANCE)
        if record['condition']=='initial':
            assert record['initial_hidden_alias_verified_all_anchors'] is True
            assert record['initial_hidden_alias_bitwise'] is True
            assert record['initial_hidden_source_condition']=='initial'
            assert record['initial_hidden_source_run_id']
            source=record['initial_hidden_alias_source_arrays']
            assert source['layers']==layers_desc and source['prefix_final']==prefix_desc
            row['saved_initial_alias_fields_verified']=True
        result.append(row)
    return result


def extract():
    c,version=analysis_verify();index=load_index(c,version)
    assert index.get('old_shared_initial_verified') and index.get('old_sources_reused')==64,'Run CPU reuse before extraction'
    old_records={(r['architecture'],r['encoding'],r['task'],r['condition'],r['split']):r for r in index['records'] if r.get('reused_original')}
    new_jobs=[j for j in c['checkpoint_inventory'] if not j.get('reused_existing',True)]
    assert len(new_jobs)==48
    # Fail before CUDA/forward when any fixed final endpoint is still absent.
    missing=[j['run_id'] for j in new_jobs if not (ROOT/'models'/j['run_id']/'completed.json').is_file()]
    assert not missing,('Fixed20k sources incomplete; no new forward started',missing)
    for job in new_jobs:checkpoint_source(c,job,'trained')
    data=eval_data(c);device=device_setup();hashed={};started_all=time.monotonic()
    for job in new_jobs:
        for condition in PHASES:
            paths=[record_path(job,condition,s) for s in SPLITS]
            cp_path,cp_sha,_=checkpoint_source(c,job,condition)
            if all(p.exists() for p in paths):
                for split,path in zip(SPLITS,paths):
                    rec=js(path);validate_record(c,version,job,rec,condition,split,hashed,cp_sha)
                    save_record(index,rec,job,condition,split)
                continue
            net,cp_path,cp_sha,before=load_model(c,job,condition,device)
            capture=original.Capture(net,job['architecture'])
            representative=old_records[job['architecture'],job['encoding'],'descents','initial','test']
            try:
                checks,manual_references=independent_manual(c,job,condition,net,capture,data,device,representative)
                for split,path in zip(SPLITS,paths):
                    if path.exists():
                        rec=js(path);validate_record(c,version,job,rec,condition,split,hashed,cp_sha)
                        save_record(index,rec,job,condition,split);continue
                    out=path.parent;out.mkdir(parents=True,exist_ok=True);N=N_SAMPLES[split]
                    shared=old_records[job['architecture'],job['encoding'],'descents','initial',split]
                    targets={'output_prob':(np.float32,(N,3,31)),'prediction':(np.int64,(N,3)),'answer_ce':(np.float64,(N,3))}
                    if condition=='trained':targets.update(layers=(np.float32,(N,3,5,256)),prefix_final=(np.float32,(N,3,256)))
                    values={name:np.lib.format.open_memmap(out/(name+'.partial.npy'),mode='w+',dtype=dtype,shape=shape)
                            for name,(dtype,shape) in targets.items()}
                    initial_layers=np.load(shared['arrays']['layers']['path'],mmap_mode='r',allow_pickle=False) if condition=='initial' else None
                    initial_norm=np.load(shared['arrays']['prefix_final']['path'],mmap_mode='r',allow_pickle=False) if condition=='initial' else None
                    seen=np.zeros((N,3),dtype=bool);started=time.monotonic();max_alias_error=0.;alias_bitwise=True
                    for start in range(0,N,128):
                        use=np.arange(start,min(start+128,N));ns=data[split+'_lengths'][use]
                        for action in range(3):
                            hidden,logits,_,_,_=original.forward(net,capture,data[f'{split}_{job["encoding"]}_input'][use,action],ns,job['task'],device,job['architecture'])
                            separate_norm=capture.states['final_norm']
                            assert np.array_equal(hidden[:,-1],separate_norm),'Same-forward final norm capture differs'
                            if condition=='initial':
                                alias_bitwise&=bool(np.array_equal(hidden,initial_layers[use,action]) and np.array_equal(separate_norm,initial_norm[use,action]))
                                max_alias_error=max(max_alias_error,float(np.max(np.abs(hidden-initial_layers[use,action]))),float(np.max(np.abs(separate_norm-initial_norm[use,action]))))
                            else:
                                values['layers'][use,action]=hidden
                                values['prefix_final'][use,action]=separate_norm
                            labels=data[split+'_labels'][use,action,TASKS.index(job['task'])]
                            prob,pred,ce=answer_outputs(logits,labels)
                            values['output_prob'][use,action]=prob;values['prediction'][use,action]=pred;values['answer_ce'][use,action]=ce
                            assert not seen[use,action].any();seen[use,action]=True
                    original.sync(device)
                    assert seen.all() and parameter_hash(net)==before and all(p.grad is None for p in net.parameters())
                    if condition=='initial':assert alias_bitwise,('Initial hidden alias is not exact',job['run_id'],split,max_alias_error)
                    for name in list(values):
                        values[name].flush();del values[name]
                        os.replace(out/(name+'.partial.npy'),out/(name+'.npy'))
                    rec=base_record(c,version,job,condition,split,cp_path,cp_sha)
                    rec.update(reused_original=False,initial_output_forward=condition=='initial',initial_CE_evaluated=condition=='initial',
                        parameter_sha256_before=before,parameter_sha256_after=parameter_hash(net),
                        forward_and_store_seconds=time.monotonic()-started,
                        independent_manual_scope='Fixed test n10/20/30, first8 from each128-example full prompt, all3 actions; validation arrays are not independently sampled.',
                        same_forward_packed_and_primary_norm_exact=True,
                        full_vocab_size=188,output_probability_definition='softmax(logits[:,0:31]) conditional on legal numeric answers0..30',
                        prediction_definition='argmax over full188 vocabulary',answer_CE_definition='CE over full188 vocabulary with actual source task label; no relation offset applied to labels')
                    if condition=='initial':
                        rec['arrays'].update(layers=shared['arrays']['layers'],prefix_final=shared['arrays']['prefix_final'])
                        rec.update(initial_hidden_alias_verified_all_anchors=True,initial_hidden_alias_bitwise=True,
                            initial_hidden_alias_max_absolute_difference=max_alias_error,
                            initial_hidden_source_run_id=shared['run_id'],initial_hidden_source_condition='initial',
                            initial_hidden_alias_source_arrays={name:shared['arrays'][name] for name in ('layers','prefix_final')})
                    for name in targets:rec['arrays'][name]=describe(out/(name+'.npy'),LAYERS[job['architecture']] if name=='layers' else None)
                    if split=='test':rec['independent_manual_checks']=verify_saved_manual(rec,checks,manual_references)
                    else:rec['independent_manual_test_record_path']=str(record_path(job,condition,'test'))
                    validate_record(c,version,job,rec,condition,split,hashed,cp_sha)
                    save_record(index,rec,job,condition,split)
                    print(dict(extracted=key(job,condition,split),seconds=rec['forward_and_store_seconds'],initial_exact_alias=condition=='initial'),flush=True)
            finally:capture.close()
            del net;torch.cuda.empty_cache() if device.type=='cuda' else None
    assert len(index['records'])==256
    index['initial_shared_exact']=all(r.get('initial_hidden_alias_bitwise',False) for r in index['records'] if r['condition']=='initial' and not r.get('reused_original')) and index['old_shared_initial_verified']
    assert index['initial_shared_exact']
    index['status']='completed';index['extraction_pipeline_wall_seconds']=time.monotonic()-started_all
    index['timing_scope']='Current invocation pipeline wall includes forward, IO, model loading and hashes; no sum of process times is called actual GPU kernel hours.'
    write_json(INDEX,index)
    return validate()


def validate():
    c,version=analysis_verify();index=load_index(c,version);hashed={}
    expected={(j['run_id'],phase,split):j for j in c['checkpoint_inventory'] for phase in PHASES for split in SPLITS}
    actual={(r['run_id'],r['condition'],r['split']):r for r in index['records']}
    assert len(actual)==len(index['records'])==256 and set(actual)==set(expected)
    for k,job in expected.items():
        path,cp_sha,_=checkpoint_source(c,job,k[1]);validate_record(c,version,job,actual[k],k[1],k[2],hashed,cp_sha)
        assert sha(path)==cp_sha
    assert index['initial_shared_exact'] and index['status']=='completed'
    old_initial=[r for r in actual.values() if r['condition']=='initial' and r.get('reused_original')]
    new_initial=[r for r in actual.values() if r['condition']=='initial' and not r.get('reused_original')]
    assert len(old_initial)==32 and all('answer_ce' not in r['arrays'] for r in old_initial)
    assert len(new_initial)==96 and all('answer_ce' in r['arrays'] and r['initial_hidden_alias_verified_all_anchors'] for r in new_initial)
    record=dict(status='passed',index_path=str(INDEX),index_sha256=sha(INDEX),logical_records=256,
        source_models=64,old_models_reused=16,new_fixed20k_models=48,
        actual_unique_mmap_files=len(hashed),old_initial_CE_missing_records=32,
        new_task_initial_outputs_forwarded=True,initial_hidden_alias_all_anchors_bitwise=True,
        primary_norm_separate=True,layer_order=LAYERS,full_vocab_CE188=True,conditional_numeric_probability31=True,
        fixed_manual_lengths=[10,20,30],manual_first8_from_full128_prompt=True,
        manual_scope='Each of48 new sources, both initial/trained, fixed test anchors only, all3 actions',
        direct_manual_to_saved_packed_and_primary_norm_verified=True,
        saved_initial_alias_fields_verified=True,old_conversion_exact_arrays=64*3,
        source_updates=0,optimizer_constructed=False,backward=False,new_inputs=False,old_artifacts_modified=False)
    write_json(DIAG/'extraction_validation.json',record)
    return record


def fixture():
    """CPU-only schema/normalization/hash checks without experiment inputs."""
    checks={}
    with tempfile.TemporaryDirectory(prefix='eight-relations-extract-') as temporary:
        p=Path(temporary);N=8
        prefix=np.arange(N*3*256,dtype=np.float32).reshape(N,3,256)
        packed=np.repeat((prefix+1)[:,:,None,:],5,axis=2)
        np.savez_compressed(p/'old.npz',prefix_final=prefix,output_prob=np.ones((N,3,31),np.float32)/31,prediction=np.zeros((N,3),np.int16))
        conversions={}
        with np.load(p/'old.npz',allow_pickle=False) as ar:
            for name in ('prefix_final','output_prob','prediction'):
                _,conversions[name]=convert_old_array(p/(name+'.npy'),ar[name])
        assert all(r['conversion_bitwise_equal'] and r['original_array_sha256']==r['converted_array_sha256'] for r in conversions.values())
        checks['old_arrays_original_to_saved_hash_and_bitwise_equal']=True
        atomic_npy(p/'layers.npy',packed)
        descriptions={name:describe(p/(name+'.npy'),LAYERS['transformer'] if name=='layers' else None) for name in ('layers','prefix_final','output_prob','prediction')}
        hashed={}
        for name,d in descriptions.items():validate_array(d,name,N,hashed)
        assert np.array_equal(np.load(p/'prefix_final.npy'),prefix)
        assert not np.array_equal(np.load(p/'prefix_final.npy'),packed[:,:,-1])
        checks['primary_original_norm_not_packed_norm']=True
        raw=np.ascontiguousarray(prefix);h=hashlib.sha256();h.update(str(raw.dtype).encode());h.update(str(raw.shape).encode());h.update(raw.tobytes())
        assert array_hash(prefix)==h.hexdigest();checks['array_hash_matches_frozen_convention']=True
        assert descriptions['prediction']['dtype']=='int16';checks['original_prediction_dtype_preserved']=True
        logits=torch.zeros((2,188),dtype=torch.float32);logits[:,TOKEN_TO_ID['<EOS>']]=10
        prob,pred,ce=answer_outputs(logits,[0,30])
        assert np.allclose(prob.sum(-1),1) and prob.shape==(2,31)
        assert (pred==TOKEN_TO_ID['<EOS>']).all()
        assert np.allclose(ce,F.cross_entropy(logits,torch.tensor([0,30]),reduction='none').double().numpy())
        assert ce.dtype==np.float64
        checks.update(numeric_conditioning_not_full_vocab_prediction=True,CE_full188_including_outside_numeric_tokens=True,
                      labels_zero_to_thirty_no_offset=True,answer_ce_float64=True,readonly_mmap_metadata=True)
        atomic_npy(p/'answer_ce.npy',ce[:,None].repeat(3,axis=1));validate_array(describe(p/'answer_ce.npy'),'answer_ce',2,{})
        # Verify the same saved-array routine used after real extraction. The
        # reference is synthetic here; it does not constitute real manual proof.
        faithful=np.repeat(prefix[:,:,None,:],5,axis=2)
        atomic_npy(p/'faithful.npy',faithful)
        rec=dict(split='test',condition='trained',arrays={
            'layers':describe(p/'faithful.npy',LAYERS['transformer']),
            'prefix_final':describe(p/'prefix_final.npy')})
        reference=[dict(indices=np.arange(8),action_index=1,manual_layers=faithful[:,1].copy())]
        manual_checks=[dict(split='test',length=10,action='complement',full_prompt_samples=128,checked_samples=8)]
        verified=verify_saved_manual(rec,manual_checks,reference)
        assert verified[0]['saved_layer_max_absolute_difference']==verified[0]['saved_primary_norm_max_absolute_difference']==0.
        assert verified[0]['direct_manual_to_saved_arrays'] is True
        checks['direct_manual_to_saved_arrays_path_verified']=True
        bad=dict(rec,arrays=dict(rec['arrays'],layers=descriptions['layers']))
        try:verify_saved_manual(bad,manual_checks,reference)
        except AssertionError:checks['incorrect_saved_arrays_rejected']=True
        else:raise AssertionError('A changed stored layer array was accepted')
        rec.update(condition='initial',initial_hidden_alias_verified_all_anchors=True,
            initial_hidden_alias_bitwise=True,initial_hidden_source_condition='initial',initial_hidden_source_run_id='synthetic-original',
            initial_hidden_alias_source_arrays=dict(rec['arrays']))
        assert verify_saved_manual(rec,manual_checks,reference)[0]['saved_initial_alias_fields_verified'] is True
        checks['saved_initial_alias_metadata_verified']=True
        bad=dict(rec,initial_hidden_alias_source_arrays={})
        try:verify_saved_manual(bad,manual_checks,reference)
        except (AssertionError,KeyError):checks['incorrect_initial_alias_metadata_rejected']=True
        else:raise AssertionError('Mismatched initial alias metadata was accepted')
    record=dict(status='passed',checks=checks,extractor_sha256=sha(__file__),
        original_capture_sha256=sha(OLD_EXTRACT),CPU_synthetic_only=True,real_inputs_read=False,
        checkpoints_loaded=False,GPU_started=False,training_started=False,new_metrics=False)
    write_json(DIAG/'extract_synthetic_fixture.json',record)
    return record


if __name__=='__main__':
    p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--stage',choices=['reuse','extract','validate']);g.add_argument('--fixture',action='store_true')
    a=p.parse_args()
    result=fixture() if a.fixture else {'reuse':reuse,'extract':extract,'validate':validate}[a.stage]()
    print(json.dumps({k:v for k,v in result.items() if k!='records'},ensure_ascii=False,indent=2))
