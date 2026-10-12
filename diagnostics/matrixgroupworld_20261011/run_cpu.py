"""Preflight frozen metadata, then the prescribed saved-feature analysis."""
from experiments.matrixgroupworld_20261011.common import DIAG,ROOT,read,sha,json_write,verify
from experiments.matrixgroupworld_20261011.driver import run_cpu

def main():
    c=verify();checks=[]
    for name,field in [('candidate_manifest.csv','candidate_manifest_sha256'),('candidate_classification_audit.json','candidate_classification_audit_sha256')]:
        actual=sha(DIAG/name);expected=c['freeze'][field]
        assert actual==expected,('Prospectively frozen candidate metadata changed',name)
        checks.append(dict(path=str(DIAG/name),sha256=actual,expected=expected,passed=True))
    assert read(DIAG/'execution_state.json')['status']=='GPU_completed'
    assert 'end_monotonic' in read(DIAG/'gpu_budget.json')
    for j in c['training_manifest']:
        r=read(ROOT/'models'/j['run_id']/'completed.json')
        assert r['formal_completed'] and r['step']==r['finite_gradient_updates']==20000
        assert r['checkpoint_sha256']==sha(j['checkpoint_path'])
    json_write(DIAG/'CPU_preflight.json',dict(status='passed',checks=checks,actual_20k_endpoints=8,new_GPU_forward=0,new_training=0))
    return run_cpu()

if __name__=='__main__':raise SystemExit(main())
