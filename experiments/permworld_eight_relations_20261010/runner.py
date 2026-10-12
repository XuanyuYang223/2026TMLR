"""Keep the frozen training recipe; repair its terminal metadata writer only."""
import argparse
from pathlib import Path
import time
from .common import *
from . import train

def writer_verify():
    f=js(DIAG/'metadata_writer_freeze.json')
    assert f['protocol_sha256']==sha(CONFIG)
    assert f['frozen_training_source_version_sha256']==source_version()
    for p,h in f['source_sha256'].items():assert sha(p)==h

def finalize(run_id):
    writer_verify();c=verify();job=next(j for j in c['training_manifest'] if j['run_id']==run_id)
    path=ROOT/'models'/run_id/'checkpoint.pt';before=sha(path)
    cp=torch.load(path,map_location='cpu',weights_only=False);arch=job['architecture'];ref=c['initializations'][arch]
    for k,v in dict(run_id=run_id,architecture=arch,step=20000,finite_gradient_updates=20000,
                    protocol_sha256=sha(CONFIG),execution_source_version_sha256=source_version(),
                    data_sha256=c['data_sha256'],batch_schedule_sha256=c['batch_schedule_sha256'],
                    initial_parameter_sha256=ref['parameter_sha256'],initial_checkpoint_sha256=ref['sha256'],
                    starting_rng_hash=c['new_task_start_rng'][arch]['rng_hash']).items():assert cp[k]==v,(k,cp[k],v)
    assert len(cp['curve'])==20 and [r['step'] for r in cp['curve']]==list(range(1000,20001,1000))
    assert math.isfinite(cp['minimum_gradient_norm']) and math.isfinite(cp['maximum_gradient_norm']) and cp['maximum_gradient_norm']>0
    net=model(c,arch,job['model_seed'],'cpu');net.load_state_dict(cp['model'],strict=True)
    ph=parameter_hash(net);assert ph!=ref['parameter_sha256']
    assert all(torch.isfinite(p).all() for p in net.parameters())
    rec=dict(job)
    rec.update(step=20000,checkpoint_sha256=before,parameter_sha256=ph,
               protocol_sha256=sha(CONFIG),execution_source_version_sha256=source_version(),
               data_sha256=c['data_sha256'],batch_schedule_sha256=c['batch_schedule_sha256'],
               initial_parameter_sha256=ref['parameter_sha256'],initial_checkpoint_sha256=ref['sha256'],
               starting_rng_hash=cp['starting_rng_hash'],actual_initial_identity_verified=True,
               finite_gradient_updates=20000,elapsed_seconds=cp['elapsed_seconds'],curve=cp['curve'],
               parameters=sum(p.numel() for p in net.parameters()),exposures=1280000,supervised_tokens=2560000,
               peak_cuda_allocated_bytes=cp['peak_cuda_allocated_bytes'],
               metadata_writer_sha256=sha(__file__),metadata_writer_freeze_sha256=sha(DIAG/'metadata_writer_freeze.json'),
               metadata_repair='Frozen train.py terminal dict(**job, checkpoint_sha256=...) has duplicate keyword; only sidecar writer replaced.',
               metadata_repair_source_updates=0,checkpoint_bytes_unmodified=True)
    assert sha(path)==before
    json_write(path.with_name('completed.json'),rec)
    print(dict(run_id=run_id,metadata_finalized=True,step=20000,new_updates=0,checkpoint_sha256=before),flush=True)

def run(run_id):
    writer_verify();p=ROOT/'models'/run_id/'checkpoint.pt'
    if p.exists() and not p.with_name('completed.json').exists():
        cp=torch.load(p,map_location='cpu',weights_only=False)
        step=cp['step'];del cp
        if step==20000:return finalize(run_id)
    try:train.run_job(run_id)
    except TypeError as error:
        tb=error.__traceback__
        while tb.tb_next is not None:tb=tb.tb_next
        assert str(error)=="dict() got multiple values for keyword argument 'checkpoint_sha256'"
        assert Path(tb.tb_frame.f_code.co_filename).resolve()==Path(train.__file__).resolve() and tb.tb_lineno==91
        finalize(run_id)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-id',required=True)
    run(p.parse_args().run_id)
