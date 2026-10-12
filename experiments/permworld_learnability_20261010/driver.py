"""One wall-clock-limited run; no metric-driven extensions or extra jobs."""
from datetime import datetime, timezone
import json
import subprocess
import sys
import time

from .common import *


def develop():
    config = verify()
    cpu = json.loads((ROOT / 'cpu_check.json').read_text())
    assert cpu['status'] == 'passed' and cpu['protocol_sha256'] == sha(CONFIG)
    budget_path = ROOT / 'gpu_budget.json'
    if budget_path.exists():
        budget = json.loads(budget_path.read_text())
        assert budget['protocol_sha256'] == sha(CONFIG)
        assert budget['limit_seconds'] == config['budget']['active_gpu_wall_seconds']
        assert abs(budget['deadline_epoch']-budget['start_epoch']-budget['limit_seconds']) < 1e-5
        assert abs(budget['stop_updates_epoch']-budget['deadline_epoch']+config['budget']['finalization_reserve_seconds']) < 1e-5
    else:
        now = time.time()
        budget = dict(start_epoch=now, deadline_epoch=now+config['budget']['active_gpu_wall_seconds'],
            stop_updates_epoch=now+config['budget']['active_gpu_wall_seconds']-config['budget']['finalization_reserve_seconds'],
            finalization_reserve_seconds=config['budget']['finalization_reserve_seconds'],
            limit_seconds=config['budget']['active_gpu_wall_seconds'], protocol_sha256=sha(CONFIG),
            start_utc=datetime.now(timezone.utc).isoformat(), deadline_not_reset_on_resume=True)
        json_write(budget_path, budget)
    if time.time() < budget['stop_updates_epoch']:
        subprocess.run([sys.executable, '-m', 'experiments.permworld_learnability_20261010',
                        '--stage', 'resume'], cwd=WORKSPACE, check=True)
    if not (ROOT / 'resume_check.json').exists() or json.loads((ROOT / 'resume_check.json').read_text())['status'] == 'budget_exhausted':
        json_write(ROOT / 'development_execution.json', dict(status='budget_exhausted_before_resume_gate',
            protocol_sha256=sha(CONFIG), no_source_jobs_started=True))
        return
    assert json.loads((ROOT / 'resume_check.json').read_text())['status'] == 'passed'
    invocations = []
    exhausted = False
    for stage in (2000, 4000, 8000):
        for job in config['training_manifest']:
            if time.time() >= budget['stop_updates_epoch']:
                exhausted = True; break
            start = time.time()
            subprocess.run([sys.executable, '-m', 'experiments.permworld_learnability_20261010',
                '--stage', 'train-one', '--run-id', job['run_id'], '--stop-step', str(stage)],
                cwd=WORKSPACE, check=True)
            invocations.append(dict(run_id=job['run_id'], requested_stage=stage,
                invocation_wall_seconds=time.time()-start))
            json_write(ROOT / 'development_execution.json', dict(status='running',
                protocol_sha256=sha(CONFIG), invocations=invocations, formal_training_started=False,
                geometry_evaluated=False, test_evaluated=False))
        if exhausted: break
    expected = {job['run_id']:job for job in config['training_manifest']}
    complete = []
    data_audit = json.loads((ROOT / 'worlds' / config['units'][0]['unit'] / 'data_audit.json').read_text())
    for path in (ROOT / 'models').glob('*/completed.json'):
        item = json.loads(path.read_text()); run_id = path.parent.name
        assert run_id in expected and item['run_id'] == run_id
        assert item['step'] == 8000 and item['protocol_sha256'] == sha(CONFIG)
        assert item['execution_source_version_sha256'] == source_version()
        assert item['data_sha256'] == data_audit['data_sha256']
        complete.append(run_id)
    json_write(ROOT / 'development_execution.json', dict(status='completed' if len(complete)==16 else 'budget_exhausted',
        protocol_sha256=sha(CONFIG), completed_fixed8k_models=len(complete),
        invocations=invocations, training_wall_seconds=time.time()-budget['start_epoch'],
        budget_seconds=budget['limit_seconds'], budget_deadline_not_reset=True,
        formal_training_started=False, GroupWorld_started=False,
        geometry_evaluated=False, test_evaluated=False, cost_extension=False))
