"""One explicit, bounded engineering launch; no formal training transition."""
from pathlib import Path
import time
import numpy as np
from experiments.matrixgroupworld_20261011 import budget, engineering, model_runtime
from experiments.matrixgroupworld_20261011.common import DIAG, ROOT, SOURCE, WORKSPACE, read, sha, payload_sha, json_write

def main():
    c=read(DIAG/'protocol_draft.json'); ready=read(ROOT/'data_ready.json')
    assert ready['status']=='passed'
    assert read(DIAG/'math_audit.json')['status']=='passed'
    assert read(DIAG/'protection_start.json')['status']=='passed'
    assert not any((ROOT/'worlds'/u['unit']/'analysis_labels.npz').exists() for u in c['units'])
    paths=[SOURCE/(x+'.py') for x in ('core','data','math_checks','common','budget','model_runtime','training','engineering')]
    paths.extend(WORKSPACE/x for x in ('experiments/permworld_encoding/core.py','experiments/longrun_attention.py','external/neurips/src/neurips_permutations/models.py','external/neurips/src/neurips_permutations/passage.py'))
    paths.append(Path(__file__).resolve())
    sources={str(p):sha(p) for p in paths}
    bind={'protocol_sha256':sha(DIAG/'protocol_draft.json'),'source_version_sha256':payload_sha(sources),
          'data_sha256':ready['units'][0]['data_sha256'],'batch_schedule_sha256':ready['units'][0]['schedule_sha256']}
    json_write(DIAG/'engineering_source_bindings.json',dict(source_sha256=sources,binding=bind,data_ready_sha256=sha(ROOT/'data_ready.json')))
    g=budget.start_gpu()
    refs=model_runtime.create_initializations(ROOT,c['units'],c)
    json_write(DIAG/'initialization_inventory.json',refs)
    u=ready['units'][0]
    with np.load(u['data_path'],allow_pickle=False) as z:
        formal={k:z[k] for k in z.files if k.startswith(('train_','validation_'))}
    with np.load(u['schedule_path'],allow_pickle=False) as z: schedule={k:z[k] for k in z.files}
    with np.load(ready['engineering']['data_path'],allow_pickle=False) as z: raw={k:z[k] for k in z.files}
    result=engineering.run(c,raw,refs[u['unit']],g['deadline_monotonic'],formal_data=formal,formal_schedule=schedule,source_binding=bind,output_dir=ROOT/'engineering')
    result['actual_GPU_stage_wall_seconds_at_receipt']=time.monotonic()-g['start_monotonic']
    result['engineering_audit_path']=str(ROOT/'engineering/engineering_audit.json')
    result['engineering_audit_sha256']=sha(ROOT/'engineering/engineering_audit.json')
    json_write(DIAG/'gpu_engineering.json',result)
    print({'status':result['status'],'cost_estimate':result['cost_estimate'],'problems':result['problems']},flush=True)
    budget.snapshot()
    return 0 if result['status']=='passed' else 2

if __name__=='__main__': raise SystemExit(main())
