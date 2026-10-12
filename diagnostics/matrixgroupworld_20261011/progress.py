"""Read-only small-metadata status, without reading model or final-label arrays."""
from pathlib import Path
import json
from experiments.matrixgroupworld_20261011.common import DIAG,ROOT,read
from experiments.matrixgroupworld_20261011.budget import snapshot

def main():
    state=read(DIAG/'execution_state.json') if (DIAG/'execution_state.json').exists() else {}
    jobs=[]
    for directory in sorted((ROOT/'models').glob('*')):
        if not directory.is_dir():continue
        stages=list(directory.glob('stage_*.json'))
        if not stages:continue
        p=max(stages,key=lambda p:int(p.stem.split('_')[1]));r=read(p);v=r['validation']
        jobs.append(dict(run_id=directory.name,step=r['step'],answer_accuracy=v['answer_accuracy'],
            per_n=v['per_n'],completed=(directory/'completed.json').exists()))
    value=dict(execution=state,jobs=jobs,cost=snapshot())
    print(json.dumps(value,ensure_ascii=False))

if __name__=='__main__':main()
