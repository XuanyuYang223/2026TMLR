"""Independent CPU provenance audit; no models/forward/optimizers are constructed.

--pending reads only small sidecars. --run refuses incomplete phases BEFORE
hashing weights/caches. Only an all-passed run can produce final_audit.json.
This audit does not compute scientific metrics or overwrite their artifacts.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
import tempfile
import time

import numpy as np

WORKSPACE=Path(__file__).resolve().parents[3]
DIAG=Path(__file__).resolve().parents[1]
REVIEW=Path(__file__).resolve().parent
GRAPH_DIAG=WORKSPACE/'diagnostics/graphworld_pilot_20261010'
GRAPH_ROOT=Path('/mnt/yangx-projects-f/20261010/graphworld_pilot')
FREEZE=REVIEW/'audit_freeze.json'
EXPECTED_HEAD='e4e9ded9648aa2dc92a1cceef929cc9521546418'
EMPTY_SHA=hashlib.sha256(b'').hexdigest()
OLD_TASKS=('descents','recoils','peaks','valleys')
PERM_SPLITS=('validation','test')
GRAPH_SPLITS=('validation','analysis')
CONDITIONS=('initial','trained')
EXTRA_READOUT_PHASES={'perm_result_readout':'permworld','graph_result_readout':'graphworld'}


def require(value,message):
    if not value:raise AssertionError(message)


def read(path):return json.loads(Path(path).read_text())


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    temp.replace(path)


def csv_write(path,records):
    require(records,'Refuse empty manifest')
    fields=list(dict.fromkeys(k for r in records for k in r))
    with Path(path).open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(records)


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def array_hash(value):
    a=np.ascontiguousarray(value);h=hashlib.sha256()
    h.update(str(a.dtype).encode());h.update(str(a.shape).encode())
    view=memoryview(a).cast('B')
    for start in range(0,len(view),4*1024*1024):h.update(view[start:start+4*1024*1024])
    return h.hexdigest()


def state_hash(state):
    h=hashlib.sha256()
    for name,value in sorted(state.items()):
        require(value.device.type=='cpu','Checkpoint tensor was not CPU-loaded')
        a=value.detach().contiguous().numpy()
        for part in (name,str(a.dtype),str(a.shape)):h.update(part.encode())
        h.update(a.tobytes())
    return h.hexdigest()


def rng_hash(state):
    h=hashlib.sha256();h.update(repr(state['python']).encode())
    h.update(state['numpy'][0].encode());h.update(state['numpy'][1].tobytes())
    h.update(repr(state['numpy'][2:]).encode());h.update(state['torch'].cpu().numpy().tobytes())
    for value in state['cuda']:h.update(value.cpu().numpy().tobytes())
    return h.hexdigest()


class Ledger:
    """One SHA scan per resolved physical file, even with many logical aliases."""
    def __init__(self):self.files={};self.arrays={}

    def file(self,path,expected=None,group='metadata'):
        path=Path(path).resolve();key=str(path)
        require(path.is_file(),'Missing file: '+key)
        before=path.stat();signature=(before.st_size,before.st_mtime_ns,before.st_ino)
        if key not in self.files:
            digest=sha(path)
            after=path.stat();require(signature==(after.st_size,after.st_mtime_ns,after.st_ino),'File changed while hashing: '+key)
            self.files[key]=dict(path=key,sha256=digest,bytes=before.st_size,signature=signature,groups=set())
        item=self.files[key]
        require(signature==item['signature'],'File changed during audit: '+key)
        if expected is not None:require(item['sha256']==expected,'SHA mismatch: '+key)
        item['groups'].add(group)
        return item['sha256']

    def array(self,desc,group='perm_cache'):
        path=Path(desc['path']).resolve();key=str(path)
        require(Path(desc['path']).is_absolute() and path.suffix=='.npy','Cache must be absolute .npy')
        self.file(path,desc['sha256'],group)
        if key not in self.arrays:
            a=np.load(path,mmap_mode='r',allow_pickle=False)
            require(not a.flags.writeable,'Expected readonly mmap')
            self.arrays[key]=dict(array_sha256=array_hash(a),shape=list(a.shape),dtype=str(a.dtype))
        actual=self.arrays[key]
        for field in ('array_sha256','shape','dtype'):require(actual[field]==desc[field],'Array metadata mismatch: '+key+'/'+field)
        return actual

    def storage(self):
        groups={}
        for group in sorted({g for item in self.files.values() for g in item['groups']}):
            values=[x for x in self.files.values() if group in x['groups']]
            groups[group]=dict(unique_physical_files=len(values),physical_bytes=sum(v['bytes'] for v in values))
        return dict(unique_hashed_physical_files=len(self.files),unique_physical_bytes=sum(x['bytes'] for x in self.files.values()),
                    groups=groups,group_totals_overlap_not_additive=True,
                    scope='Referenced files hashed by this audit; unique resolved paths, not logical references or total disk usage.')

    def verify_stable(self):
        for key,item in self.files.items():
            info=Path(key).stat()
            require(item['signature']==(info.st_size,info.st_mtime_ns,info.st_ino),'File changed after its SHA scan: '+key)


def freeze():
    policy=dict(version='permworld-eight-relations-final-provenance/v1',
        expected_perm_models=64,expected_reused=16,expected_perm_new=48,expected_graph_new=4,
        fixed_updates=20000,exposures_per_model=1280000,supervised_tokens_per_model=2560000,
        successful_perm_engineering_updates=240,failed_preformal_perm_engineering_updates=40,
        perm_cache_logical_records=256,graph_cache_logical_records=16,protected_perm_files=1223,
        expected_git_HEAD=EXPECTED_HEAD,pre_run_git_diff_byte_snapshot_available=False,
        git_claim='Final tracked files/index equal stated HEAD; protected old files match frozen SHA. No pre-run diff snapshot claim.',
        no_model_construction=True,no_forward=True,no_training=True,no_scientific_metrics=True,
        GPU_kernel_time='unknown unless independently measured; never set to0 from CPU-only audit',
        incomplete_phases='--run writes pending/failure only, never a completed audit')
    policy['required_extra_CPU_readout_phases']=EXTRA_READOUT_PHASES
    value=dict(registered_utc=datetime.now(timezone.utc).isoformat(),source_sha256={str(Path(__file__).resolve()):sha(__file__)},
        policy=policy,protocol_sha256=sha(DIAG/'protocol.json'),graph_protocol_sha256=sha(GRAPH_DIAG/'protocol.json'),
        input_manifests_sha256={str(p):sha(p) for p in [DIAG/'training_freeze.json',DIAG/'analysis_freeze.json',DIAG/'metadata_writer_freeze.json',
            DIAG/'protected_files.json',GRAPH_DIAG/'freeze_manifest.json',GRAPH_DIAG/'verification_freeze.json',GRAPH_DIAG/'protected_files.json',REVIEW/'readout_freeze.json']})
    if FREEZE.exists():
        old=read(FREEZE)
        for field in ('source_sha256','policy','protocol_sha256','graph_protocol_sha256','input_manifests_sha256'):require(old[field]==value[field],'Refuse changing frozen final-audit policy')
        return old
    write(FREEZE,value);return value


def verify_freeze():
    frozen=read(FREEZE)
    require(frozen['source_sha256']=={str(Path(__file__).resolve()):sha(__file__)},'Final auditor source changed after freeze')
    require(frozen['protocol_sha256']==sha(DIAG/'protocol.json'),'Perm protocol changed')
    require(frozen['graph_protocol_sha256']==sha(GRAPH_DIAG/'protocol.json'),'Graph protocol changed')
    for p,digest in frozen['input_manifests_sha256'].items():require(sha(p)==digest,'Frozen final-audit input manifest changed: '+p)
    return frozen


def require_json(path,statuses,missing,problems,extra=None):
    path=Path(path)
    if not path.exists():missing.append(str(path));return None
    try:value=read(path)
    except (ValueError,OSError) as error:problems.append(str(path)+': '+str(error));return None
    if value.get('status') not in statuses:missing.append(str(path)+' status='+str(value.get('status')))
    if extra is not None and not extra(value):missing.append(str(path)+' completeness fields not yet satisfied')
    return value


def pending(write_result=True):
    """Read current sidecars only: never open a weight, NPZ or mmap array."""
    c=read(DIAG/'protocol.json');g=read(GRAPH_DIAG/'protocol.json');root=Path(c['root'])
    missing=[];problems=[];completed=[]
    require(len(c['checkpoint_inventory'])==64 and len(c['training_manifest'])==48,'Wrong fixed Perm source manifest')
    require(len(g['training_manifest'])==4,'Wrong graph manifest')
    for domain,jobs,base in [('permworld',c['checkpoint_inventory'],root),('graphworld',g['training_manifest'],GRAPH_ROOT)]:
        for job in jobs:
            cp=Path(job.get('checkpoint_path',base/'models'/job['run_id']/'checkpoint.pt'))
            sidecar=Path(job.get('completed_path',cp.with_name('completed.json')))
            if not cp.exists():missing.append(str(cp))
            if not sidecar.exists():missing.append(str(sidecar));continue
            value=read(sidecar)
            if value.get('step')!=20000 or value.get('finite_gradient_updates')!=20000:
                missing.append(str(sidecar)+' not fixed20k/20000finite');continue
            if value.get('run_id')!=job['run_id']:problems.append(str(sidecar)+' run_id mismatch')
            completed.append(dict(domain=domain,run_id=job['run_id'],step=value['step']))
    require_json(root/'queue_state.json',{'completed'},missing,problems,lambda v:v.get('completed_models')==48)
    idx=require_json(root/'evaluation/index.json',{'completed'},missing,problems)
    if idx is not None and len(idx.get('records',[]))!=256:missing.append('Perm cache records: '+str(len(idx.get('records',[])))+'/256')
    required_perm={
        'math_label_audit.json':{'passed'},'engineering_audit.json':{'passed'},
        'extraction_reuse_audit.json':{'passed'},'extraction_validation.json':{'passed'},
        'metric_review/independent_metric_actual.json':{'passed'},'metric_review/final_reader_audit.json':{'passed'},
        'report_audit.json':{'passed'}}
    for name,statuses in required_perm.items():require_json(DIAG/name,statuses,missing,problems)
    graph_training=require_json(GRAPH_ROOT/'training_complete.json',{'completed'},missing,problems)
    if graph_training is not None and len(graph_training.get('records',[]))!=4:missing.append('Graph training records not4')
    gi=require_json(GRAPH_ROOT/'evaluation_cache/index.json',{'completed'},missing,problems)
    if gi is not None and len(gi.get('records',[]))!=16:missing.append('Graph cache records not16')
    for name,statuses in {
        'mathematics_audit.json':{'passed'},'pretraining_data_validation.json':{'passed'},
        'gpu_engineering.json':{'passed'},'evaluation_complete.json':{'completed'},
        'independent_metrics.json':{'passed'},'independent_actual_small_batch.json':{'passed'},
        'report_reader_audit.json':{'passed'},'completion.json':{'completed_and_stopped'}}.items():
        v=require_json(GRAPH_DIAG/name,statuses,missing,problems)
        if name=='report_reader_audit.json' and v is not None and v.get('source_count')!=4:missing.append('Graph reader source_count not4')
    for path in [DIAG/'report.html',DIAG/'report.txt',GRAPH_DIAG/'report.html',GRAPH_DIAG/'report.txt']:
        if not path.exists():missing.append(str(path))
    for stage in ('permworld','graphworld'):
        require_json(REVIEW/stage/'readout_status.json',{'completed_readout'},missing,problems)
    phase_path=DIAG/'phase_timing.json'
    if not phase_path.exists():missing.append(str(phase_path))
    else:
        phases=read(phase_path)['phases'];orchestration=read(DIAG/'orchestration_freeze.json')
        expected_phases={p[0] for p in orchestration['perm_stages']+orchestration['graph_stages']}|set(EXTRA_READOUT_PHASES)
        present={p['phase'] for p in phases if p.get('exit_code')==0}
        if present!=expected_phases:missing.append('Instrumented phases incomplete: '+','.join(sorted(expected_phases-present)))
    result=dict(status='ready_for_full_hash_audit' if not missing and not problems else 'pending' if not problems else 'failed',
        registered_utc=datetime.now(timezone.utc).isoformat(),missing=sorted(set(missing)),problems=problems,
        completed_metadata_endpoints=len(completed),expected_endpoints=68,completed_metadata=completed,
        metadata_only=True,weights_or_raw_cache_opened=False,GPU_started=False,training=False,
        final_complete_written=False)
    if write_result:write(REVIEW/'pending.json',result)
    return result


def frozen_sources(ledger,path,protocol_sha):
    value=read(path);ledger.file(path,group='freezes')
    require(value['protocol_sha256']==protocol_sha,'Source freeze protocol mismatch: '+str(path))
    for source,digest in value['source_sha256'].items():ledger.file(source,digest,'frozen_source')
    return canonical(value['source_sha256']),value


def git_audit():
    def call(*args):return subprocess.check_output(['git',*args],cwd=WORKSPACE)
    head=call('rev-parse','HEAD').decode().strip()
    unstaged=call('diff','--binary','--no-ext-diff','HEAD','--')
    staged=call('diff','--cached','--binary','--no-ext-diff','HEAD','--')
    require(head==EXPECTED_HEAD,'Git HEAD differs from stated starting HEAD')
    require(not unstaged and not staged,'Tracked working tree/index differs from HEAD')
    return dict(HEAD=head,working_tree_vs_HEAD_bytes=len(unstaged),working_tree_vs_HEAD_sha256=hashlib.sha256(unstaged).hexdigest(),
        index_vs_HEAD_bytes=len(staged),index_vs_HEAD_sha256=hashlib.sha256(staged).hexdigest(),
        final_tracked_files_equal_HEAD=True,pre_run_tracked_diff_byte_snapshot_available=False,
        limitation='Starting HEAD was retained; no pre-run tracked diff byte snapshot was located. This does not prove the tracked diff stayed constant throughout execution.',
        untracked_new_experiment_outputs_excluded_from_tracked_equality=True)


def initial_sources(c,g,ledger):
    import torch
    result=[];vocab=None;parameters={}
    sources=[(a,r,c['model']) for a,r in c['initializations'].items()]
    sources.append(('graph_transformer',dict(path=g['initial_checkpoint_path'],sha256=g['initial_checkpoint_sha256'],parameter_sha256=g['initial_parameter_sha256']),g['model']))
    for arch,ref,config in sources:
        ledger.file(ref['path'],ref['sha256'],'initial_weights')
        saved=torch.load(ref['path'],map_location='cpu',weights_only=False)
        actual=state_hash(saved['model']);require(actual==ref['parameter_sha256']==saved['parameter_sha256'],'Actual initial parameter hash mismatch')
        require(saved['model_config']==config,'Actual initial model config mismatch')
        vocabulary=saved['vocabulary'];require(len(vocabulary)==188 and all(vocabulary[f'{n:02d}']==n for n in range(31)),'Numeric vocabulary mismatch')
        if vocab is None:vocab=vocabulary
        require(vocab==vocabulary,'Shared vocabulary mismatch')
        parameters[arch]=actual
        result.append(dict(architecture=arch,path=ref['path'],checkpoint_sha256=ref['sha256'],parameter_sha256=actual,
            vocabulary_sha256=canonical(vocabulary),actual_container_stage=saved.get('stage'),actual_container_protocol_sha256=saved['protocol_sha256'],
            evidence='CPU-loaded saved tensor state; no constructor or seed reconstruction'))
        del saved
    require(parameters['transformer']==parameters['graph_transformer'],'Graph does not reuse the original TF initial parameters')
    rng_refs={}
    for job in c['checkpoint_inventory']:
        rng_refs[job['starting_rng_path']]=(job['starting_rng_checkpoint_sha256'],job['starting_rng_hash'],job['initial_parameter_sha256'])
    rng_refs[g['starting_rng_path']]=(g['starting_rng_sha256'],g['starting_rng_hash'],g['initial_parameter_sha256'])
    rngs=[]
    for path,(file_sha,expected_rng,expected_param) in rng_refs.items():
        ledger.file(path,file_sha,'saved_start_RNG');value=torch.load(path,map_location='cpu',weights_only=False)
        actual=rng_hash(value['rng']);require(actual==expected_rng==value['rng_hash'],'Actual start RNG hash mismatch')
        require(value['parameter_sha256']==expected_param,'Start RNG associated with different initial parameters')
        rngs.append(dict(path=path,checkpoint_sha256=file_sha,rng_hash=actual,initial_parameter_sha256=expected_param));del value
    return result,rngs,canonical(vocab)


def checkpoint_rows(c,g,ledger,versions,vocab_sha):
    import torch
    rows=[];lookup={};root=Path(c['root'])
    planned_new={j['run_id']:j for j in c['training_manifest']}
    require(len(planned_new)==len(c['training_manifest'])==48,'Fixed new48 training manifest differs')
    world_seeds=read(Path(c['data_path']).with_name('data_audit.json'))
    for domain,jobs,base in [('permworld',c['checkpoint_inventory'],root),('graphworld',g['training_manifest'],GRAPH_ROOT)]:
        for job in jobs:
            old=domain=='permworld' and job['task'] in OLD_TASKS
            cp_path=Path(job.get('checkpoint_path',base/'models'/job['run_id']/'checkpoint.pt'))
            sidecar=Path(job.get('completed_path',cp_path.with_name('completed.json')))
            rec=read(sidecar);cp_sha=ledger.file(cp_path,rec['checkpoint_sha256'],'reused_weights' if old else 'new_formal_weights')
            ledger.file(sidecar,job.get('completed_sha256') if old else None,'completed_metadata')
            if old:require(cp_sha==job['checkpoint_sha256'],'Original reused checkpoint changed')
            require(rec['run_id']==job['run_id'] and rec['step']==rec['finite_gradient_updates']==20000,'Source is not fixed20k')
            for field in ('architecture','encoding','task'):require(rec[field]==job[field],'Source job identity differs: '+field)
            if domain=='permworld':
                require(rec['model_seed']==job['model_seed'],'Source nominal model seed differs')
                require(rec['data_seed']==world_seeds['data_seed'] and rec['batch_seed']==world_seeds['batch_seed'],'Perm source nominal world/batch seed differs')
                require(rec['parameters']==int(job['parameters']),'Recorded parameter count differs from frozen inventory')
                if not old:
                    planned=planned_new[job['run_id']]
                    for field in ('architecture','encoding','task','model_seed','checkpoint_path','initial_checkpoint_sha256',
                                  'initial_parameter_sha256','starting_rng_hash','starting_rng_path'):
                        require(planned[field]==job[field],'Frozen new inventory/manifest disagreement: '+field)
                    for field in ('data_seed','batch_seed','updates','parameters','exposures','supervised_tokens'):
                        require(int(rec[field])==int(planned[field]),'New source receipt/manifest differs: '+field)
                    require(planned['vocabulary_sha256']==rec['vocabulary_sha256']==vocab_sha,'New source vocabulary differs')
                    require(planned['data_sha256']==c['data_sha256'] and planned['batch_schedule_sha256']==c['batch_schedule_sha256'],'New source world/batch plan differs')
            require(rec['exposures']==1280000 and rec['supervised_tokens']==2560000,'Source exposure differs')
            require(rec['actual_initial_identity_verified'] is True,'Actual initial identity not verified')
            version=versions['old'] if old else versions['perm_training'] if domain=='permworld' else versions['graph_training']
            if old and job.get('original_execution_source_version_sha256'):require(job['original_execution_source_version_sha256']==version,'Old job source version differs')
            expected_protocol=(c['source_protocol_sha256'] if old else sha(DIAG/'protocol.json')) if domain=='permworld' else sha(GRAPH_DIAG/'protocol.json')
            source_field='execution_source_version_sha256' if domain=='permworld' else 'source_version_sha256'
            require(rec[source_field]==version and rec['protocol_sha256']==expected_protocol,'Completed source version mismatch')
            data_sha=c['data_sha256'] if domain=='permworld' else read(GRAPH_ROOT/'world/data_audit.json')['data_sha256']
            init_sha=job['initial_parameter_sha256'] if domain=='permworld' else g['initial_parameter_sha256']
            init_cp_sha=job['initial_checkpoint_sha256'] if domain=='permworld' else g['initial_checkpoint_sha256']
            expected_rng=job['starting_rng_hash'] if domain=='permworld' else g['starting_rng_hash']
            require(rec['data_sha256']==data_sha and rec['initial_parameter_sha256']==init_sha,'Source data or init differs')
            if not old:require(rec['initial_checkpoint_sha256']==init_cp_sha and rec['starting_rng_hash']==expected_rng,'Source container or RNG differs')
            if domain=='permworld':
                if not old:
                    require(rec['batch_schedule_sha256']==c['batch_schedule_sha256'],'Source batch schedule differs')
                    require(rec['metadata_repair_source_updates']==0 and rec['checkpoint_bytes_unmodified'] is True,'Writer repair changed checkpoint/update budget')
                    ledger.file(DIAG/'metadata_writer_freeze.json',rec['metadata_writer_freeze_sha256'],'freezes')
                    ledger.file(WORKSPACE/'experiments/permworld_eight_relations_20261010/runner.py',rec['metadata_writer_sha256'],'frozen_source')
            saved=torch.load(cp_path,map_location='cpu',weights_only=False)
            for field,expected in {'run_id':job['run_id'],'step':20000,'finite_gradient_updates':20000,'protocol_sha256':expected_protocol,
                source_field:version,'data_sha256':data_sha,'initial_parameter_sha256':init_sha,
                'initial_checkpoint_sha256':init_cp_sha,'starting_rng_hash':expected_rng}.items():require(saved[field]==expected,'Raw checkpoint binding mismatch: '+str(cp_path)+'/'+field)
            if domain=='permworld' and not old:require(saved['batch_schedule_sha256']==c['batch_schedule_sha256'],'Raw checkpoint batch differs')
            actual=state_hash(saved['model']);expected_param=job['parameter_sha256'] if old else rec['parameter_sha256'] if domain=='permworld' else rec['final_parameter_sha256']
            require(actual==expected_param,'Final tensor parameter hash mismatch')
            if old:require(actual==job['parameter_sha256'],'Original parameter hash differs')
            if old:require(job['vocabulary_sha256']==vocab_sha,'Original job vocabulary identity differs')
            row=dict(domain=domain,run_id=job['run_id'],architecture=job['architecture'],encoding=job['encoding'],task=job['task'],
                model_seed=rec['model_seed'] if domain=='permworld' else g['model_seed'],
                data_seed=rec['data_seed'] if domain=='permworld' else g['data_seed'],batch_seed=rec['batch_seed'] if domain=='permworld' else g['batch_seed'],
                reused_existing=old,step=20000,finite_gradient_updates=20000,checkpoint_path=str(cp_path),checkpoint_sha256=cp_sha,
                parameter_sha256=actual,completed_path=str(sidecar),completed_sha256=ledger.files[str(sidecar.resolve())]['sha256'],
                initial_checkpoint_path=job['initial_checkpoint_path'] if domain=='permworld' else g['initial_checkpoint_path'],
                initial_checkpoint_sha256=init_cp_sha,initial_parameter_sha256=init_sha,starting_rng_hash=expected_rng,
                starting_rng_path=job['starting_rng_path'] if domain=='permworld' else g['starting_rng_path'],
                starting_rng_checkpoint_sha256=job['starting_rng_checkpoint_sha256'] if domain=='permworld' else g['starting_rng_sha256'],
                vocabulary_sha256=vocab_sha,protocol_sha256=expected_protocol,execution_source_version_sha256=version,
                data_sha256=data_sha,batch_schedule_sha256=c['batch_schedule_sha256'] if domain=='permworld' else read(GRAPH_ROOT/'world/data_audit.json')['batch_indices_sha256'],
                batch_binding='Explicit checkpoint+receipt SHA' if domain=='permworld' and not old else 'Frozen load_data verifies persisted world batch SHA; original/graph checkpoint binds world data SHA, not a separate batch-SHA field',
                exposures=1280000,supervised_tokens=2560000,parameters=rec['parameters'],elapsed_process_wall_seconds=rec['elapsed_seconds'],
                status='completed_fixed20k_verified',
                actual_initial_identity_verified=True,source_origin='original first20k' if old else 'new fixed20k',
                receipt_inventory_status_not_completion_gate=rec.get('status'))
            rows.append(row);lookup[domain,job['run_id']]=row;del saved
    require(sum(r['reused_existing'] for r in rows)==16 and len(rows)==68,'Endpoint counts differ from16reuse+52new')
    return rows,lookup


def perm_caches(c,ledger,lookup,version):
    index_path=Path(c['evaluation_index_path']);index=read(index_path);ledger.file(index_path,group='cache_index')
    require(index['status']=='completed' and index['initial_shared_exact'] is True,'Perm extraction incomplete')
    require(index['analysis_source_version_sha256']==version and index['protocol_sha256']==sha(DIAG/'protocol.json'),'Perm cache freeze mismatch')
    require(index['source_data_sha256']==c['data_sha256'] and index['source_updates']==0,'Perm cache source mismatch')
    expected={(j['run_id'],condition,split):j for j in c['checkpoint_inventory'] for condition in CONDITIONS for split in PERM_SPLITS}
    keys=[(r['run_id'],r['condition'],r['split']) for r in index['records']]
    require(len(keys)==len(set(keys))==256 and set(keys)==set(expected),'Perm256 logical coverage incomplete')
    array_refs=0;old_missing=0;manual_checks=0;alias_records=0
    for rec in index['records']:
        k=rec['run_id'],rec['condition'],rec['split'];job=expected[k]
        phase,split=k[1:];N=5376 if split=='validation' else 10752
        checkpoint=lookup['permworld',job['run_id']]
        cp_sha=checkpoint['initial_checkpoint_sha256'] if phase=='initial' else checkpoint['checkpoint_sha256']
        require(rec['source_checkpoint_sha256']==cp_sha and rec['protocol_sha256']==sha(DIAG/'protocol.json'),'Perm cache checkpoint differs')
        require(rec['analysis_source_version_sha256']==version and rec['source_data_sha256']==c['data_sha256'],'Perm record version differs')
        require(rec['source_updates']==0 and rec['source_weights_modified'] is False,'Cache changed source weights')
        shapes={'layers':[N,3,5,256],'prefix_final':[N,3,256],'output_prob':[N,3,31],'prediction':[N,3],'answer_ce':[N,3]}
        required={'layers','prefix_final','output_prob','prediction'}
        old=job['task'] in OLD_TASKS
        if not (old and phase=='initial'):required.add('answer_ce')
        else:old_missing+=1;require(rec.get('initial_CE_missing_reason'),'Original initial CE lacks missing declaration')
        require(set(rec['arrays'])==required,'Perm cache fields differ')
        for name,desc in rec['arrays'].items():
            require(desc['shape']==shapes[name],'Perm cache shape differs: '+name)
            dtype=np.dtype(desc['dtype'])
            require(dtype==np.dtype('float32') if name in ('layers','prefix_final','output_prob') else dtype==np.dtype('float64') if name=='answer_ce' else dtype.kind in 'iu','Unexpected cache dtype')
            ledger.array(desc);array_refs+=1
            origin='perm_cache_new_physical' if Path(desc['path']).resolve().is_relative_to(Path(c['root']).resolve()) else 'perm_cache_reused_physical'
            ledger.file(desc['path'],desc['sha256'],origin)
            if not old and phase=='initial' and name in ('layers','prefix_final'):
                ledger.file(desc['path'],desc['sha256'],'perm_cache_shared_alias_targets')
        layer_names=[f'block_{i:02d}' if job['architecture']=='transformer' else f'layer_{i:02d}' for i in range(1,5)]+['final_norm']
        require(rec['arrays']['layers']['layer_order']==layer_names,'Cache layer order differs')
        if old:
            require(rec['reused_original'] is True,'Original source was not reused')
            ledger.file(rec['original_NPZ_path'],rec['original_NPZ_sha256'],'original_cache')
            conversion=rec['original_array_conversion_checks']
            require(set(conversion)=={'prefix_final','output_prob','prediction'},'Old array conversion evidence incomplete')
            for name,item in conversion.items():require(item['conversion_bitwise_equal'] is True and item['original_array_sha256']==item['converted_array_sha256']==rec['arrays'][name]['array_sha256'],'Old conversion did not preserve array')
        else:
            require(rec['parameter_sha256_before']==rec['parameter_sha256_after'],'New cache altered parameter hash')
            expected_param=checkpoint['initial_parameter_sha256'] if phase=='initial' else checkpoint['parameter_sha256']
            require(rec['parameter_sha256_before']==expected_param,'New cache parameter hash differs from actual checkpoint tensors')
            if phase=='initial':
                require(rec['initial_hidden_alias_verified_all_anchors'] and rec['initial_hidden_alias_bitwise'],'Initial hidden alias unverified')
                require(rec['initial_hidden_alias_source_arrays']=={name:rec['arrays'][name] for name in ('layers','prefix_final')},'Initial saved alias descriptor mismatch')
                alias_records+=1
            if split=='test':
                checks=rec['independent_manual_checks'];require(len(checks)==9,'Fixed manual test checks missing')
                require({(x['length'],x['action']) for x in checks}=={(n,a) for n in (10,20,30) for a in c['actions']},'Manual check subset differs')
                for x in checks:
                    require(x['direct_manual_to_saved_arrays'] and x['split']=='test' and x['full_prompt_samples']==128 and x['checked_samples']==8,'Manual evidence does not cover saved arrays')
                    require(x['saved_layer_max_absolute_difference']<=2e-5 and x['saved_primary_norm_max_absolute_difference']<=2e-5 and x['saved_packed_vs_primary_norm_max_absolute_difference']==0,'Saved manual tolerance breached')
                    for n,dn in [('layers','layers'),('prefix_final','prefix_final')]:
                        require(x[f'saved_{n}_path']==rec['arrays'][dn]['path'] and x[f'saved_{n}_sha256']==rec['arrays'][dn]['sha256'],'Manual evidence binds other saved array')
                    if phase=='initial':require(x['saved_initial_alias_fields_verified'],'Manual initial alias not verified')
                manual_checks+=len(checks)
    require(old_missing==32 and alias_records==96 and manual_checks==864,'Perm missing/init/manual counts differ')
    return dict(logical_records=256,logical_array_references=array_refs,unique_physical_arrays=len({d['path'] for r in index['records'] for d in r['arrays'].values()}),
        original_initial_CE_missing_records=old_missing,new_initial_alias_records=alias_records,
        saved_manual_comparisons=manual_checks,saved_manual_scope='Each48new×2conditions×test n10/20/30×3actions, first8fromsame128prompt; no validation sampling claim')


def graph_caches(g,ledger,lookup,version):
    path=GRAPH_ROOT/'evaluation_cache/index.json';index=read(path);ledger.file(path,group='cache_index')
    expected={(j['task'],condition,split):j for j in g['training_manifest'] for condition in CONDITIONS for split in GRAPH_SPLITS}
    keys=[(r['task'],r['condition'],r['split']) for r in index['records']]
    require(index['status']=='completed' and len(keys)==len(set(keys))==16 and set(keys)==set(expected),'Graph16 logical coverage incomplete')
    require(index['protocol_sha256']==sha(GRAPH_DIAG/'protocol.json') and index['source_version_sha256']==version,'Graph cache version differs')
    require(index['actual_initial_identity_verified'] and index['initial_hidden_task_independent_bitwise'] and index['source_updates']==0,'Graph cache initial/update scope differs')
    contents=[];initial_hidden=defaultdict(set)
    for rec in index['records']:
        task,condition,split=rec['task'],rec['condition'],rec['split'];job=expected[task,condition,split]
        file=Path(rec['path']);meta=read(file.with_suffix('.json'));require(meta==rec['metadata'],'Graph index metadata differs from sidecar')
        ledger.file(file,meta['sha256'],'graph_cache');ledger.file(file.with_suffix('.json'),group='cache_metadata')
        cp=lookup['graphworld',job['run_id']];cp_sha=g['initial_checkpoint_sha256'] if condition=='initial' else cp['checkpoint_sha256']
        expected_sig=dict(protocol_sha256=sha(GRAPH_DIAG/'protocol.json'),source_version_sha256=version,
            data_sha256=index['data_sha256'],checkpoint_sha256=cp_sha,task=task,split=split,condition=condition)
        require(meta['signature']==expected_sig,'Graph cache signature differs')
        require(meta['source_updates']==0 and meta['parameter_sha256_before']==meta['parameter_sha256_after'],'Graph cache changed weights')
        expected_param=g['initial_parameter_sha256'] if condition=='initial' else cp['parameter_sha256']
        require(meta['parameter_sha256_before']==expected_param,'Graph cache parameter hash differs from actual checkpoint tensors')
        N=768 if split=='validation' else 1536
        shapes={'prefix_final':[N,5,256],'output_prob':[N,5,31],'prediction':[N,5],'answer_ce':[N,5]}
        with np.load(file,allow_pickle=False) as archive:
            require(set(archive.files)==set(shapes),'Graph cache fields differ')
            hashes={}
            for name,shape in shapes.items():
                a=archive[name];require(list(a.shape)==shape==meta['shape'][name],'Graph cache shape differs')
                dtype='float32' if name in ('prefix_final','output_prob') else 'int16' if name=='prediction' else 'float64'
                require(a.dtype==np.dtype(dtype),'Graph production cache dtype differs: '+name)
                hashes[name]=array_hash(a)
            if condition=='initial':initial_hidden[split].add(hashes['prefix_final'])
        contents.append(dict(task=task,condition=condition,split=split,path=str(file),sha256=meta['sha256'],array_sha256=hashes))
    require(set(initial_hidden)==set(GRAPH_SPLITS) and all(len(x)==1 for x in initial_hidden.values()),'Graph actual initial hidden states differ across tasks')
    write(REVIEW/'graph_cache_content_manifest.json',dict(status='passed',records=contents,logical_cache_count=16,unique_physical_NPZ_files=len({x['path'] for x in contents})))
    return dict(logical_records=16,unique_physical_cache_files=len({x['path'] for x in contents}),all_cache_file_SHA_and_raw_array_hashes_verified=True,
        hidden_initial_task_independent_content_SHA_equal=True,manual_scope='Graph production checks compare live manual/forward fixed validation samples; this audit does not rerun a model')


def protection(ledger):
    counts={}
    for label,path in [('permworld',DIAG/'protected_files.json'),('graphworld',GRAPH_DIAG/'protected_files.json')]:
        meta=read(path);ledger.file(path,group='protection_manifest')
        if label=='permworld':require(len(meta['files'])==1223,'Expected1223 protected old artifacts')
        for p,digest in meta['files'].items():ledger.file(p,digest,'protected_old')
        counts[label]=len(meta['files'])
    return dict(manifest_file_counts=counts,unique_protected_files=len([x for x in ledger.files.values() if 'protected_old' in x['groups']]),all_SHA_unchanged=True)


def timing(c,rows,ledger):
    groups=defaultdict(list)
    for r in rows:groups['old_reused' if r['reused_existing'] else r['domain']].append(r)
    result=dict(recorded_training_process_wall_seconds={name:math.fsum(x['elapsed_process_wall_seconds'] for x in rr) for name,rr in groups.items()},
        GPU_kernel_seconds=None,GPU_active_seconds=None,GPU_measurement_status='Not independently measured; unknown, not zero.',
        process_wall_scope='Source receipt elapsed_seconds includes validation/IO and is not active GPU time. Old16 timings are historical, not new training cost.')
    queue=read(Path(c['root'])/'queue_state.json')
    require(queue['status']=='completed' and queue['completed_models']==48,'Queue incomplete')
    require(abs(queue['source_process_wall_seconds']-result['recorded_training_process_wall_seconds']['permworld'])<1e-6,'Queue process wall differs from actual48receipts')
    result['permworld_queue_wall_seconds']=queue['queue_wall_seconds']
    result['permworld_queue_wall_scope']='Recorded queue wall includes its saved resume accounting; does not imply unique GPU kernel duration.'
    phases=[]
    path=DIAG/'phase_timing.json'
    if path.exists():
        ledger.file(path,group='phase_timing');value=read(path);raw=value if isinstance(value,list) else value.get('phases',value.get('records',[]))
        require(isinstance(raw,list),'Unsupported phase timing schema')
        orchestration=read(DIAG/'orchestration_freeze.json')
        expected={p[0] for p in orchestration['perm_stages']+orchestration['graph_stages']}|set(EXTRA_READOUT_PHASES)
        require(len(raw)==len(expected) and {p['phase'] for p in raw}==expected,'Missing/duplicate instrumented phases')
        for phase in raw:
            require(phase['wall_seconds']>=0,'Negative phase wall duration')
            require(phase.get('exit_code',0)==0,'Recorded phase failed: '+str(phase.get('phase')))
            if phase['phase'] in EXTRA_READOUT_PHASES:
                expected_cmd=[str(WORKSPACE/'.venv/bin/python'),'-B',
                    'diagnostics/permworld_eight_relations_20261010/final_review/result_readout.py',
                    '--stage',EXTRA_READOUT_PHASES[phase['phase']]]
                require(phase['command']==expected_cmd,'Extra readout command differs from fixed CPU-only scope')
            phases.append(dict(phase,interpretation='Command process wall/user/system CPU observations; GPU kernel duration is unknown.'))
    result['instrumented_phase_timings']=phases
    result['phase_timing_missing']=not bool(phases)
    result['perm_math_recorded_CPU_pipeline_wall_seconds']=read(DIAG/'math_label_audit.json')['elapsed_cpu_wall_seconds']
    result['perm_successful_engineering_recorded_pipeline_wall_seconds']=math.fsum(r['elapsed_wall_seconds'] for r in read(DIAG/'engineering_audit.json')['records'])
    result['perm_failed_preformal_engineering_wall_seconds']=None
    result['perm_failed_preformal_engineering_timing_reason']='The first40-update mkdir failure is preserved, but no independent phase duration is present in its sidecar.'
    for key,path,field in [('perm_reuse',DIAG/'extraction_reuse_audit.json','elapsed_wall_seconds'),
                           ('perm_extract',DIAG/'extraction_validation.json','extraction_pipeline_wall_seconds'),
                           ('graph_evaluate',GRAPH_DIAG/'evaluation_complete.json','wall_seconds'),
                           ('graph_engineering',GRAPH_DIAG/'gpu_engineering.json','wall_seconds')]:
        value=read(path);result[key+'_recorded_pipeline_wall_seconds']=value.get(field)
    result['perm_extract_recorded_pipeline_wall_seconds']=read(Path(c['evaluation_index_path'])).get('extraction_pipeline_wall_seconds')
    require(result['perm_extract_recorded_pipeline_wall_seconds'] is not None,'Final extraction pipeline timing absent')
    return result


def engineering(ledger):
    p=read(DIAG/'engineering_audit.json');g=read(GRAPH_DIAG/'gpu_engineering.json')
    require(p['status']=='passed' and len(p['records'])==4,'Perm engineering incomplete')
    require(g['status']=='passed' and g['source_endpoints_produced']==0 and len(g['records'])==4,'Graph engineering incomplete')
    require(all(r['real_recipe_updates']==40 and r['optimizer_resume_from']==20 for r in p['records']),'Perm engineering budget differs')
    # Perm4config: reference40+replayed20 each. Graph4gradient32,
    # plus independentcontinuous32 and split16+16: actual updates192.
    repair=read(DIAG/'engineering_repair_01/repair.json')
    require(repair['before_any_formal_training'] and repair['scientific_protocol_unchanged'],'Engineering repair scope changed')
    ledger.file(DIAG/'engineering_repair_01/train.before.py',repair['before_train_sha256'],'engineering_metadata')
    ledger.file(DIAG/'engineering_repair_01/failed_engineering.log',group='engineering_metadata')
    ledger.file(DIAG/'engineering_repair_01/repair.json',group='engineering_metadata')
    result=dict(permworld_successful_engineering_updates=4*(40+20),permworld_failed_preformal_engineering_updates=40,
        permworld_actual_engineering_updates=4*(40+20)+40,graphworld_actual_engineering_updates=4*32+32+16+16,
        formal_new_models=52,formal_new_updates=52*20000,reused_original_models=16,reused_updates_this_round=0,
        engineering_paths_not_formal_endpoints=True,includes_previous_toy_checks=False,
        scope='Successful bounded engineering plus preserved first40-update mkdir failure, derived from failed log/control flow; prior development and external CUDA toy fixtures are separate historical artifacts.')
    require(read(GRAPH_DIAG/'completion.json')['engineering_updates']==result['graphworld_actual_engineering_updates'],'Graph reported engineering count differs')
    for path in [DIAG/'engineering_audit.json',GRAPH_DIAG/'gpu_engineering.json']:ledger.file(path,group='engineering_metadata')
    return result


def storage_inventory(c,ledger):
    """Count actual F-root files by unique path; never sum logical aliases."""
    root_inventory={};union={}
    for name,root in [('permworld_new_root',Path(c['root'])),('graphworld_new_root',GRAPH_ROOT)]:
        files={str(p.resolve()):p.stat().st_size for p in root.rglob('*') if p.is_file()}
        union.update(files)
        root_inventory[name]=dict(path=str(root),unique_physical_files=len(files),physical_bytes=sum(files.values()),
            unhashed_files_stat_only=sum(p not in ledger.files for p in files),
            leftover_temporary_paths=[p for p in files if p.endswith(('.tmp','.partial.npy','.pending'))])
    return dict(audited_references=ledger.storage(),actual_new_F_roots=root_inventory,
        new_F_roots_unique_physical_files=len(union),new_F_roots_unique_physical_bytes=sum(union.values()),
        scope='F-root totals include logs, engineering checkpoints and metadata; stat-only totals are separate from SHA-verified scientific references.',
        initial_alias_targets_do_not_add_storage=True,unique_path_convention='Resolved absolute path; no content-deduplication assumption about distinct files.')


def run():
    frozen=verify_freeze();pre=pending()
    if pre['status']!='ready_for_full_hash_audit':
        print(json.dumps(dict(status='refused_incomplete',pending_path=str(REVIEW/'pending.json'),missing_count=len(pre['missing']),problems=pre['problems'])));return 2
    begun=time.monotonic();ledger=Ledger();c=read(DIAG/'protocol.json');g=read(GRAPH_DIAG/'protocol.json')
    require(len({j['run_id'] for j in c['checkpoint_inventory']})==64,'Duplicate inventory run IDs')
    require({(j['architecture'],j['encoding'],j['task']) for j in c['checkpoint_inventory']}==
            {(a,e,t) for a in c['architectures'] for e in c['encodings'] for t in c['tasks']},'Incomplete16properties×4configurations')
    require({p.parent.name for p in (Path(c['root'])/'models').glob('*/checkpoint.pt')}=={j['run_id'] for j in c['training_manifest']},'Additional/missing formal Perm checkpoint directories')
    require({p.parent.name for p in (GRAPH_ROOT/'models').glob('*/checkpoint.pt')}=={j['run_id'] for j in g['training_manifest']},'Additional/missing formal Graph checkpoint directories')
    git=git_audit();ledger.file(DIAG/'protocol.json',frozen['protocol_sha256'],'protocol');ledger.file(GRAPH_DIAG/'protocol.json',frozen['graph_protocol_sha256'],'protocol')
    versions={}
    versions['perm_training'],training_freeze=frozen_sources(ledger,DIAG/'training_freeze.json',sha(DIAG/'protocol.json'))
    ledger.file(DIAG/'protected_files.json',training_freeze['protected_manifest_sha256'],'protection_manifest')
    ledger.file(DIAG/'training_manifest.csv',training_freeze['training_manifest_sha256'],'training_manifest')
    versions['perm_analysis'],_=frozen_sources(ledger,DIAG/'analysis_freeze.json',sha(DIAG/'protocol.json'))
    _,writer=frozen_sources(ledger,DIAG/'metadata_writer_freeze.json',sha(DIAG/'protocol.json'))
    require(writer['frozen_training_source_version_sha256']==versions['perm_training'] and writer['source_updates']==0,'Metadata writer training binding differs')
    versions['graph_training'],_=frozen_sources(ledger,GRAPH_DIAG/'freeze_manifest.json',sha(GRAPH_DIAG/'protocol.json'))
    _,graph_verification=frozen_sources(ledger,GRAPH_DIAG/'verification_freeze.json',sha(GRAPH_DIAG/'protocol.json'))
    require(graph_verification['main_source_version_sha256']==versions['graph_training'],'Graph verification main version differs')
    old_freeze=read(Path(c['source_root'])/'freeze_manifest.json');versions['old']=canonical(old_freeze['source_sha256'])
    for path,digest in old_freeze['source_sha256'].items():ledger.file(WORKSPACE/path,digest,'frozen_source')
    ledger.file(c['source_protocol_path'],c['source_protocol_sha256'],'protocol')
    ledger.file(c['data_path'],c['data_sha256'],'world_data');ledger.file(c['batch_schedule_path'],c['batch_schedule_sha256'],'batch_schedule')
    old_data_path=Path(c['data_path']).with_name('data_audit.json');old_data=read(old_data_path)
    ledger.file(old_data_path,group='data_metadata')
    require(old_data['data_sha256']==c['data_sha256'] and old_data['batch_indices_sha256']==c['batch_schedule_sha256'],'Old world audit identity differs')
    require(old_data['within_world_orbit_overlap']==0 and old_data['across_encodings_identical_objects_labels_splits_batches'],'Original data pairing/orbit policy differs')
    label_audit=read(DIAG/'math_label_audit.json');require(label_audit['status']=='passed' and label_audit['data_sha256']==c['data_sha256'],'Perm label audit mismatches world')
    ledger.file(DIAG/'math_label_audit.json',group='data_metadata')
    for j in c['training_manifest']:ledger.file(j['batch_lengths_path'],j['batch_lengths_sha256'],'batch_schedule')
    graph_data=read(GRAPH_ROOT/'world/data_audit.json')
    ledger.file(GRAPH_ROOT/'world/data.npz',graph_data['data_sha256'],'world_data')
    ledger.file(GRAPH_ROOT/'world/batch_indices.npy',graph_data['batch_indices_sha256'],'batch_schedule')
    ledger.file(GRAPH_ROOT/'world/length_schedule.npy',graph_data['length_schedule_sha256'],'batch_schedule')
    require(graph_data['full_orbit_cross_split_overlap']==0,'Graph cross-split orbit overlap nonzero')
    graph_validation=read(GRAPH_DIAG/'pretraining_data_validation.json')
    require(graph_validation['status']=='passed' and graph_validation['data_sha256']==graph_data['data_sha256'],'Graph pretraining world verification mismatch')
    ledger.file(GRAPH_DIAG/'pretraining_data_validation.json',group='data_metadata')
    ledger.file(GRAPH_DIAG/'mathematics_audit.json',g['mathematics_audit_sha256'],'data_metadata')
    for path,digest in g['source_artifacts_sha256'].items():ledger.file(path,digest,'graph_source_dependency')
    orchestration=read(DIAG/'orchestration_freeze.json')
    ledger.file(DIAG/'orchestration_freeze.json',group='freezes')
    ledger.file(DIAG/'run_fixed_analysis.py',orchestration['source_sha256'],'frozen_source')
    for path,digest in orchestration['dependent_freeze_sha256'].items():ledger.file(path,digest,'freezes')
    readout=read(REVIEW/'readout_freeze.json');ledger.file(REVIEW/'readout_freeze.json',group='freezes')
    ledger.file(readout['source_path'],readout['source_sha256'],'frozen_source')
    for stage,protocol_sha in readout['protocol_sha256'].items():
        require(protocol_sha==sha(DIAG/'protocol.json' if stage=='permworld' else GRAPH_DIAG/'protocol.json'),'Readout protocol binding mismatch')
        output=read(REVIEW/stage/'readout_status.json')
        require(output['reader_sha256']==readout['source_sha256'] and output['readout_freeze_sha256']==sha(REVIEW/'readout_freeze.json'),'Readout source identity differs')
        ledger.file(REVIEW/stage/'readout_status.json',group='verification_metadata')
    initial,rngs,vocab_sha=initial_sources(c,g,ledger)
    sources,lookup=checkpoint_rows(c,g,ledger,versions,vocab_sha)
    perm_cache=perm_caches(c,ledger,lookup,versions['perm_analysis']);graph_cache=graph_caches(g,ledger,lookup,versions['graph_training'])
    protected=protection(ledger);cost=timing(c,sources,ledger);eng=engineering(ledger)
    for path in [DIAG/'extraction_validation.json',DIAG/'metric_review/independent_metric_actual.json',DIAG/'metric_review/final_reader_audit.json',DIAG/'report_audit.json',
                 GRAPH_DIAG/'independent_actual_small_batch.json',GRAPH_DIAG/'report_reader_audit.json',GRAPH_DIAG/'completion.json']:
        ledger.file(path,group='verification_metadata')
    require(read(GRAPH_DIAG/'independent_actual_small_batch.json')['failure_count']==0,'Graph actual metric verification failed')
    require(read(GRAPH_DIAG/'report_reader_audit.json')['problems']==[],'Graph reader audit found problems')
    reader=read(DIAG/'metric_review/final_reader_audit.json');require(reader['per_length_rows']==24192 and reader['source_learning_rows']==16128,'Perm table coverage differs')
    require(reader['no_training'] and read(DIAG/'report_audit.json')['all_relations_and_conditions_retained'],'Report/audit scope differs')
    for name,digest in reader['input_sha256'].items():ledger.file(DIAG/name,digest,'scientific_table')
    for path in [DIAG/'report.html',DIAG/'report.txt',GRAPH_DIAG/'report.html',GRAPH_DIAG/'report.txt']:ledger.file(path,group='reports')
    ledger.verify_stable();require(git_audit()==git,'Tracked Git identity changed during final audit')
    perm_rows=[r for r in sources if r['domain']=='permworld']
    csv_write(DIAG/'completed_checkpoint_manifest.csv',perm_rows)
    write(DIAG/'completed_checkpoint_manifest.json',dict(status='passed',models=64,reused=16,new=48,
        protocol_sha256=frozen['protocol_sha256'],original_frozen_inventory_unchanged=True,records=perm_rows))
    write(REVIEW/'graph_completed_checkpoint_manifest.json',dict(status='passed',models=4,records=[r for r in sources if r['domain']=='graphworld']))
    result=dict(status='passed',registered_utc=datetime.now(timezone.utc).isoformat(),auditor_sha256=sha(__file__),audit_freeze_sha256=sha(FREEZE),
        formal_sources=eng,actual_saved_initializations=initial,saved_RNG_streams=rngs,
        source_versions=versions,permworld_cache=perm_cache,graphworld_cache=graph_cache,
        old_artifact_protection=protected,tracked_git=git,timing=cost,physical_storage=storage_inventory(c,ledger),
        CPU_audit_pipeline_wall_seconds=time.monotonic()-begun,models_constructed=False,forward=False,
        optimizers_constructed=False,training=False,scientific_metrics_recalculated=False,GPU_started=False,
        CPU_only_does_not_imply_original_GPU_kernel_seconds_zero=True,all_fixed_stages_complete=True,
        old_artifacts_modified=False,stopping_boundary='Fixed48 PermWorld new +4 GraphWorld pilot complete; no extra training, metrics, tasks, seeds or publication.')
    serial=[{k:v for k,v in item.items() if k!='signature'}|dict(groups=sorted(item['groups'])) for item in ledger.files.values()]
    write(REVIEW/'physical_file_manifest.json',dict(status='passed',records=serial,one_hash_scan_per_resolved_file=True))
    write(REVIEW/'final_audit.json',result)
    (REVIEW/'final_audit.txt').write_text('Final CPU provenance audit passed:64 PermWorld=16 original20k reused+48 new20k;GraphWorld4 new20k.\n'
        'Formal additions52 models/1,040,000 updates;Perm engineering240 successful+40 failed updates and Graph engineering192 updates are separate.256+16 logical caches are verified by unique physical paths;1223 earlier protected-file hashes remain identical.\n'
        'Genuine step-0,saved RNG,world,batches,vocabulary,source code,weight/cache hashes verified;no model construction or forward passes.\n'
        'Final tracked files match the retained HEAD;the pre-run diff-byte snapshot is missing,so unchanged diffs throughout execution cannot be claimed.\n'
        'Costs separate cumulative process time and phase wall time;GPU-kernel/active time is Unknown.Stop after the fixed scope.\n')
    print(json.dumps(dict(status='passed',path=str(REVIEW/'final_audit.json'),sources=68,unique_physical_files=len(ledger.files),wall_seconds=result['CPU_audit_pipeline_wall_seconds'])));return 0


def source_schema_fixture():
    """Mock CPU-load/schema paths; no actual checkpoint or model is opened."""
    from types import SimpleNamespace
    from unittest.mock import patch
    import sys
    versions=dict(old='old-code',perm_training='new-code',graph_training='graph-code')
    cp_map={};metadata={};vocab='shared-vocabulary'
    c=dict(root='/synthetic/perm',source_protocol_sha256='old-protocol',data_path='/synthetic/world/data.npz',data_sha256='perm-data',batch_schedule_sha256='perm-batches',checkpoint_inventory=[],training_manifest=[])
    g=dict(model_seed=1,data_seed=2,batch_seed=3,initial_parameter_sha256='transformer-init',initial_checkpoint_sha256='transformer-init-file',
        initial_checkpoint_path='/synthetic/initial/transformer.pt',starting_rng_hash='graph-rng',starting_rng_path='/synthetic/graph-rng.pt',starting_rng_sha256='graph-rng-file',training_manifest=[])
    metadata[str(GRAPH_ROOT/'world/data_audit.json')]=dict(data_sha256='graph-data',batch_indices_sha256='graph-batches')
    metadata['/synthetic/world/data_audit.json']=dict(data_seed=2,batch_seed=3)
    def fake_sha(path):
        if Path(path)==DIAG/'protocol.json':return 'new-protocol'
        if Path(path)==GRAPH_DIAG/'protocol.json':return 'graph-protocol'
        return canonical(str(path))
    class FakeLedger:
        def __init__(self):self.files={}
        def file(self,path,expected=None,group='metadata'):
            actual=expected or fake_sha(path);self.files[str(Path(path).resolve())]=dict(sha256=actual)
            return actual
    def add(job,domain,old=False):
        base=Path('/synthetic/perm') if domain=='permworld' else GRAPH_ROOT
        cp_path=base/'models'/job['run_id']/'checkpoint.pt'
        job['checkpoint_path']=str(cp_path)
        if domain=='permworld':
            job.update(initial_checkpoint_path=f'/synthetic/initial/{job["architecture"]}.pt',initial_checkpoint_sha256=job['architecture']+'-init-file',
                initial_parameter_sha256=job['architecture']+'-init',starting_rng_hash=job['run_id']+'-rng',starting_rng_path='/synthetic/'+job['run_id']+'-rng.pt',
                starting_rng_checkpoint_sha256=job['run_id']+'-rng-file',vocabulary_sha256=vocab,
                checkpoint_sha256=job['run_id']+'-cp',parameter_sha256=job['run_id']+'-parameters')
        protocol='old-protocol' if old else 'new-protocol' if domain=='permworld' else 'graph-protocol'
        version=versions['old'] if old else versions['perm_training'] if domain=='permworld' else versions['graph_training']
        field='execution_source_version_sha256' if domain=='permworld' else 'source_version_sha256'
        init=job['initial_parameter_sha256'] if domain=='permworld' else g['initial_parameter_sha256']
        init_cp=job['initial_checkpoint_sha256'] if domain=='permworld' else g['initial_checkpoint_sha256']
        rng=job['starting_rng_hash'] if domain=='permworld' else g['starting_rng_hash']
        row=dict(run_id=job['run_id'],architecture=job['architecture'],encoding=job['encoding'],task=job['task'],model_seed=1,data_seed=2,batch_seed=3,updates=20000,step=20000,finite_gradient_updates=20000,exposures=1280000,supervised_tokens=2560000,
            actual_initial_identity_verified=True,protocol_sha256=protocol,initial_parameter_sha256=init,
            data_sha256='perm-data' if domain=='permworld' else 'graph-data',checkpoint_sha256=job['run_id']+'-cp',
            parameters=3240448,elapsed_seconds=10,**{field:version})
        saved=dict(row,initial_checkpoint_sha256=init_cp,starting_rng_hash=rng,model={'synthetic_SHA':job['run_id']+'-parameters'})
        if not old:
            row.update(initial_checkpoint_sha256=init_cp,starting_rng_hash=rng)
            row['parameter_sha256' if domain=='permworld' else 'final_parameter_sha256']=job['run_id']+'-parameters'
        if domain=='permworld' and not old:
            row.update(batch_schedule_sha256=c['batch_schedule_sha256'],metadata_repair_source_updates=0,
                checkpoint_bytes_unmodified=True,metadata_writer_freeze_sha256='writer-freeze',metadata_writer_sha256='writer-source',status='missing_authorized',vocabulary_sha256=vocab,
                starting_rng_path=job['starting_rng_path'])
            saved['batch_schedule_sha256']=c['batch_schedule_sha256']
            planned=dict(job,data_seed=2,batch_seed=3,updates=20000,exposures=1280000,supervised_tokens=2560000,
                data_sha256=c['data_sha256'],batch_schedule_sha256=c['batch_schedule_sha256'])
            c['training_manifest'].append(planned)
            # Match the real sparse future-checkpoint inventory: vocabulary
            # belongs to the frozen training manifest, not its inventory row.
            del job['vocabulary_sha256']
        metadata[str(cp_path.with_name('completed.json'))]=row;cp_map[str(cp_path)]=saved
    tasks=OLD_TASKS+tuple('new_task_'+str(i) for i in range(12))
    for arch in ('transformer','gru'):
        for enc in ('one_line','canonical_cycle'):
            for task in tasks:
                job=dict(run_id='_'.join((arch,enc,task)),architecture=arch,encoding=enc,task=task,model_seed=1,parameters=3240448)
                add(job,'permworld',task in OLD_TASKS);c['checkpoint_inventory'].append(job)
    for task in ('max_out','min_out','max_in','min_in'):
        job=dict(run_id='graph_'+task,architecture='transformer',encoding='graph',task=task)
        add(job,'graphworld');g['training_manifest'].append(job)
    fake_torch=SimpleNamespace(load=lambda path,**kwargs:cp_map[str(Path(path))])
    with patch.dict(sys.modules,{'torch':fake_torch}),patch.dict(globals(),{
        'read':lambda path:metadata[str(Path(path))],'sha':fake_sha,'state_hash':lambda value:value['synthetic_SHA']}):
        rows,_=checkpoint_rows(c,g,FakeLedger(),versions,vocab)
        require(len(rows)==68 and sum(r['reused_existing'] for r in rows)==16,'Representative source schemas failed')
        require(all(r['step']==20000 for r in rows),'Endpoint schema failed')
        first=str(Path(c['checkpoint_inventory'][0]['checkpoint_path']))
        cp_map[first]['initial_parameter_sha256']='wrong-init'
        try:checkpoint_rows(c,g,FakeLedger(),versions,vocab)
        except AssertionError:rejected=True
        else:rejected=False
        require(rejected,'Wrong raw initial identity accepted')
    return dict(all64_perm_and4_graph_receipt_schemas_accepted=True,
        old_receipt_missing_identity_fields_verified_in_raw_checkpoint=True,
        new_sparse_inventory_vocabulary_bound_from_frozen_training_manifest=True,
        stale_inventory_status_not_used_as_endpoint_evidence=True,wrong_raw_checkpoint_initial_identity_rejected=True)


def fixture():
    checks={}
    with tempfile.TemporaryDirectory(prefix='eightrel-final-audit-') as td:
        path=Path(td)/'data.npy';a=np.arange(24,dtype=np.float32).reshape(8,3);np.save(path,a)
        desc=dict(path=str(path),sha256=sha(path),array_sha256=array_hash(a),shape=list(a.shape),dtype=str(a.dtype))
        ledger=Ledger();ledger.array(desc);ledger.array(desc);ledger.file(path,desc['sha256'],'aliased_other_group')
        require(len(ledger.files)==len(ledger.arrays)==1,'Logical alias counted as separate physical file')
        checks['logical_alias_deduplicated_to_one_physical_hash_scan']=True
        require(ledger.storage()['unique_physical_bytes']==path.stat().st_size,'Physical byte count incorrect')
        checks['physical_size_not_logical_sum']=True
        bad=dict(desc,array_sha256='0'*64)
        try:ledger.array(bad)
        except AssertionError:checks['wrong_content_SHA_rejected']=True
        else:raise AssertionError('Wrong array SHA accepted')
        bad=dict(desc,shape=[24])
        try:ledger.array(bad)
        except AssertionError:checks['wrong_array_shape_rejected']=True
        else:raise AssertionError('Wrong shape accepted')
        path.write_bytes(b'changed')
        try:ledger.file(path)
        except AssertionError:checks['mutation_during_audit_rejected']=True
        else:raise AssertionError('Mutated file accepted')
        records=[dict(run_id=str(i),step=20000) for i in range(255)]
        require(len({r['run_id'] for r in records})!=256,'Partial cache accepted')
        checks['partial256_is_not_complete']=True
        require(52*20000==1040000 and 4*(40+20)==240 and 4*32+32+16+16==192,'Engineering separation arithmetic failed')
        checks['formal_and_engineering_update_counts_separate']=True
        require(EMPTY_SHA=='e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855','Empty Git diff SHA wrong')
        checks['git_final_equality_does_not_fabricate_pre_diff_snapshot']=True
        # Exercise the real early-refusal branch, with filesystem/ledger/model
        # work blocked. It must return2 without claiming completion.
        from unittest.mock import patch
        import contextlib,io
        def forbidden():raise AssertionError('Full hashes started despite incomplete phases')
        with patch.dict(globals(),{'verify_freeze':lambda:{},'pending':lambda:{'status':'pending','missing':['synthetic-incomplete'],'problems':[]},'Ledger':forbidden}):
            with contextlib.redirect_stdout(io.StringIO()):require(run()==2,'Incomplete run did not refuse')
        checks['incomplete_run_refuses_before_hash_ledger_or_completion_write']=True
    checks.update(source_schema_fixture())
    result=dict(status='passed',auditor_sha256=sha(__file__),checks=checks,
        real_models_or_raw_data_or_caches_opened=False,GPU_started=False,training=False)
    write(REVIEW/'synthetic_fixture.json',result);print(json.dumps(result));return 0


def main():
    p=argparse.ArgumentParser(description=__doc__);group=p.add_mutually_exclusive_group(required=True)
    for mode in ('freeze','fixture','pending','run'):group.add_argument('--'+mode,action='store_true')
    args=p.parse_args()
    if args.freeze:print(json.dumps(freeze()));return 0
    if args.fixture:return fixture()
    if args.pending:print(json.dumps(pending()));return 0
    try:return run()
    except (AssertionError,KeyError,FileNotFoundError,ValueError,OSError) as error:
        write(REVIEW/'failed_audit.json',dict(status='failed',reason=str(error),final_complete_written=False,
            auditor_sha256=sha(__file__),registered_utc=datetime.now(timezone.utc).isoformat(),new_training=False))
        print(json.dumps(dict(status='failed',error=str(error))));return 1


if __name__=='__main__':raise SystemExit(main())
