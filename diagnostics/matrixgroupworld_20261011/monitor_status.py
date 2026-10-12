"""Tiny read-only metadata status for the minute-scale progress ticker."""
import json,time
from experiments.matrixgroupworld_20261011.common import DIAG,ROOT,read

state=read(DIAG/'execution_state.json');g=read(DIAG/'gpu_budget.json')
out=dict(status=state['status'],active_run=state.get('active_run'),completed_models=state.get('completed_models',0),
    GPU_stage_minutes=round((g.get('end_monotonic',time.monotonic())-g['start_monotonic'])/60,1),
    deadline_minutes_remaining=round(max(0.,g['deadline_monotonic']-time.monotonic())/60,1))
if out['active_run']:
    p=ROOT/'models'/out['active_run'];files=list(p.glob('stage_*.json'))
    if files:
        r=read(max(files,key=lambda f:int(f.stem.split('_')[1])))
        v=r['validation'];out.update(step=r['step'],answer_accuracy=v['answer_accuracy'],
        per_n_accuracy={str(t['n']):t['answer_accuracy'] for t in v['per_n']})
print(json.dumps(out))
