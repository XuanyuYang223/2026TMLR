"""Explicit frozen eight-job source run and requested inference only."""
from __future__ import annotations
import argparse
import datetime
import time
import traceback
import numpy as np
from . import budget
from .common import DIAG,ROOT,CONFIG,read,sha,json_write,source_version,verify,gpu_deadline

def load_training(c):
    ready=read(ROOT/'data_ready.json');by_unit={}
    assert sha(ROOT/'data_ready.json')==c['freeze']['data_ready_sha256']
    frozen={u['unit']:u for u in c['data_units']}
    for u in ready['units']:
        for k,v in frozen[u['unit']].items():assert u[k]==v,('Data-ready binding changed',k)
        assert sha(u['data_path'])==u['data_sha256'] and sha(u['schedule_path'])==u['schedule_sha256']
        with np.load(u['data_path'],allow_pickle=False) as z:
            d={k:z[k] for k in z.files if k.startswith(('train_','validation_'))}
        assert not any(k.startswith(('analysis_','test_')) for k in d)
        with np.load(u['schedule_path'],allow_pickle=False) as z:s={k:z[k] for k in z.files}
        by_unit[u['unit']]=(d,s,u)
    return by_unit

def load_evaluation(c,deadline=None):
    from .evaluation import prepare_evaluation_data
    ready=read(ROOT/'data_ready.json');out={}
    assert sha(ROOT/'data_ready.json')==c['freeze']['data_ready_sha256']
    for u in ready['units']:
        p=ROOT/'worlds'/u['unit']/'analysis_labels.npz';meta=read(p.parent/'analysis_label_seal.json')
        assert meta['status']=='sealed' and meta['protocol_sha256']==sha(CONFIG)
        assert meta['labels_sha256']==sha(p) and meta['data_sha256']==u['data_sha256']==sha(u['data_path'])
        assert meta['freeze_manifest_sha256']==sha(DIAG/'freeze_manifest.json')
        with np.load(u['data_path'],allow_pickle=False) as z:d={k:z[k] for k in z.files}
        with np.load(p,allow_pickle=False) as z:d['analysis_labels']=z['labels']
        d['_provenance']=dict(data_path=u['data_path'],data_sha256=u['data_sha256'],
            analysis_labels_path=str(p),analysis_labels_sha256=meta['labels_sha256'])
        out[u['unit']]=prepare_evaluation_data(d,cpu_deadline=deadline)
    return out

def state(status,**kwargs):
    r=dict(status=status,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        protocol_sha256=sha(CONFIG),source_version_sha256=source_version(),**kwargs)
    json_write(DIAG/'execution_state.json',r);print(r,flush=True)

def run_gpu():
    from .training import run_job
    from .evaluation import evaluate
    c=verify(); deadline=gpu_deadline(); refs=read(DIAG/'initialization_inventory.json')
    assert refs==c['initializations'],'Saved initialization inventory changed after formal freeze'
    engineering=read(DIAG/'gpu_engineering.json')
    assert engineering['status']=='passed' and engineering['cost_estimate']['estimated_remaining_within_budget']
    data=load_training(c);receipts=[]
    try:
        for job in c['training_manifest']:
            if time.monotonic()+30>=deadline:
                state('GPU_budget_stop',completed_models=sum(r['formal_completed'] for r in receipts),receipts=receipts)
                return 3
            state('source_training',active_run=job['run_id'],completed_models=len(receipts))
            d,s,u=data[job['unit']]
            ref=refs[job['unit']]
            for k in ('initial_checkpoint_path','initial_checkpoint_sha256','initial_parameter_sha256','starting_rng_path','starting_rng_sha256','starting_rng_hash'):
                assert job[k]==ref[k],('Job initial binding changed',k)
            assert job['data_sha256']==u['data_sha256'] and job['batch_schedule_sha256']==u['schedule_sha256']
            binding=dict(protocol_sha256=sha(CONFIG),source_version_sha256=source_version(),
                         data_sha256=u['data_sha256'],batch_schedule_sha256=u['schedule_sha256'])
            receipt=run_job(c,job,d,s,refs[job['unit']],deadline,ROOT/'models'/job['run_id'],binding)
            receipts.append(receipt);budget.snapshot()
            if not receipt['formal_completed']:
                state('source_interrupted',active_run=job['run_id'],receipts=receipts)
                return 3
        assert len(receipts)==8 and all(r['step']==20000 and r['formal_completed'] for r in receipts)
        json_write(DIAG/'training_complete.json',dict(status='completed',models=8,formal_updates=160000,receipts=receipts))
        state('prepare_frozen_evaluation',completed_models=8)
        # This occurs within the conservative serial GPU stage span. No model update.
        prepared=load_evaluation(c,deadline)
        state('feature_extraction',completed_models=8)
        index=evaluate(c,prepared,deadline=deadline,device='cuda')
        state('GPU_completed',completed_models=8,evaluation_index=str(ROOT/'evaluation_cache/index.json'),
            evaluation_cache_count=len(index['records']))
        return 0
    except Exception as e:
        json_write(DIAG/'GPU_failure.json',dict(type=type(e).__name__,message=str(e),traceback=traceback.format_exc(),receipts=receipts))
        state('GPU_failure',completed_models=sum(r.get('formal_completed',False) for r in receipts),error=str(e))
        raise
    finally:
        if 'end_monotonic' not in read(DIAG/'gpu_budget.json'):budget.finish_gpu('Frozen source/inference stage ended; inspect execution_state for completeness')
        budget.snapshot()

def run_cpu():
    from .analysis import analyze
    from .verification import verify_actual
    budget.guard_cpu(reserve=180);c=verify();deadline=budget.cpu_deadline()-120
    assert read(DIAG/'execution_state.json')['status']=='GPU_completed'
    prepared=load_evaluation(c,deadline)
    index=read(ROOT/'evaluation_cache/index.json')
    state('CPU_analysis',completed_models=8)
    tables=analyze(c,prepared,index,cpu_deadline=deadline)
    state('independent_saved_feature_verification',completed_models=8)
    check=verify_actual(c,prepared,index,cpu_deadline=deadline)
    state('analysis_completed',completed_models=8,analysis_status=read(DIAG/'analysis_completion.json')['status'],verification_status=check.get('status'),analysis_completion_sha256=sha(DIAG/'analysis_completion.json'))
    budget.snapshot();return 0

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',choices=['gpu','cpu'],required=True)
    a=p.parse_args();raise SystemExit(run_gpu() if a.stage=='gpu' else run_cpu())
