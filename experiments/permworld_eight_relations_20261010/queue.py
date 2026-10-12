"""Bounded serial execution of the 48 frozen missing sources, nothing else."""
import argparse
import datetime
import os
from pathlib import Path
import subprocess
import sys
import time
from .common import CONFIG, DIAG, ROOT, csv_write, js, json_write, sha, verify

def main():
    c=verify()
    assert js(DIAG/'engineering_audit.json')['status']=='passed'
    jobs=c['training_manifest'];assert len(jobs)==48
    logs=ROOT/'logs';logs.mkdir(parents=True,exist_ok=True)
    started=time.time();elapsed_before=0.;state_path=ROOT/'queue_state.json'
    if state_path.exists():
        prior=js(state_path)
        assert prior['protocol_sha256']==sha(CONFIG)
        elapsed_before=prior.get('queue_wall_seconds',0.)
    def state(status,current=None,error=None):
        completed=[]
        for j in jobs:
            p=ROOT/'models'/j['run_id']/'completed.json'
            if p.exists():
                r=js(p);assert r['step']==20000 and r['protocol_sha256']==sha(CONFIG)
                completed.append(r)
        result=dict(status=status,protocol_sha256=sha(CONFIG),queue_source_sha256=sha(__file__),
                    authorized_missing_models=48,completed_models=len(completed),current_run=current,
                    updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    queue_wall_seconds=elapsed_before+time.time()-started,
                    source_process_wall_seconds=sum(r['elapsed_seconds'] for r in completed),
                    no_metric_based_selection=True,no_extra_training=True,model_budget_unchanged=True)
        if error:result['error']=error
        json_write(state_path,result)
        if completed:csv_write(DIAG/'actual_training_cost.csv',[
            {k:r.get(k) for k in ('run_id','architecture','encoding','task','step','elapsed_seconds',
                                'peak_cuda_allocated_bytes','exposures','checkpoint_sha256')}
            for r in completed])
        return result
    state('running')
    for i,j in enumerate(jobs,1):
        run=j['run_id'];state('running',run)
        print(dict(job=i,total=48,run_id=run,status='starting'),flush=True)
        with (logs/(run+'.log')).open('a') as log:
            rc=subprocess.call([sys.executable,'-m','experiments.permworld_eight_relations_20261010.runner',
                                '--run-id',run],stdout=log,stderr=subprocess.STDOUT,
                               cwd=str(DIAG.parents[1]),env=os.environ.copy())
        if rc:
            state('failed',run,f'child exit code {rc}; see {logs/(run+".log")}')
            raise SystemExit(rc)
        r=state('running')
        print(dict(job=i,total=48,run_id=run,status='completed',
                   completed_models=r['completed_models'],source_process_wall_seconds=r['source_process_wall_seconds']),flush=True)
    final=state('completed')
    assert final['completed_models']==48
    print(final,flush=True)

if __name__=='__main__':
    argparse.ArgumentParser(description=__doc__).parse_args();main()
