"""Freeze the authorized fixed manifest after the one engineering/cost gate."""
import datetime
import json
import platform
import importlib.metadata
import subprocess
import time
from experiments.matrixgroupworld_20261011.common import DIAG,ROOT,SOURCE,WORKSPACE,read,sha,json_write,csv_write,payload_sha
from experiments.matrixgroupworld_20261011.data import config_signature,seal_analysis_labels

def main():
    assert not (DIAG/'protocol.json').exists() and not (DIAG/'freeze_manifest.json').exists()
    c=read(DIAG/'protocol_draft.json'); ready=read(ROOT/'data_ready.json')
    e=read(DIAG/'gpu_engineering.json');g=read(DIAG/'gpu_budget.json')
    assert e['status']=='passed' and e['cost_estimate']['estimated_remaining_within_budget']
    required=e['cost_estimate']['remaining_unscaled_wall_seconds']*e['cost_estimate']['headroom_multiplier']
    assert time.monotonic()+required<g['deadline_monotonic'],'Cost fits only at stale receipt; stop without formal launch'
    assert read(DIAG/'numerical_fixture.json')['status']=='passed'
    refs=read(DIAG/'initialization_inventory.json'); data={u['unit']:u for u in ready['units']}
    assert not list((ROOT/'models').glob('*/checkpoint.pt'))
    assert not list((ROOT/'worlds').glob('*/analysis_labels.npz'))
    assert config_signature(c)==ready['preparation_config_signature']
    c['protocol_version']='matrixgroupworld-third-domain-bounded/v1-frozen'
    c['evaluation']['batch_size']=64
    c['initializations']=refs
    c['data_units']=[{k:u[k] for k in ('unit','data_seed','batch_seed','model_seed','data_path','data_sha256','schedule_path','schedule_sha256')} for u in ready['units']]
    c['freeze']={
        'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'scope':'Exploratory third-domain pilot. Freeze is prospective for these new endpoints/geometry; earlier PermWorld/GraphWorld outcomes have been viewed.',
        'engineering_path':str(DIAG/'gpu_engineering.json'),'engineering_sha256':sha(DIAG/'gpu_engineering.json'),
        'data_ready_sha256':sha(ROOT/'data_ready.json'),'draft_sha256':sha(DIAG/'protocol_draft.json'),
        'candidate_manifest_sha256':sha(DIAG/'candidate_manifest.csv'),
        'candidate_classification_audit_sha256':sha(DIAG/'candidate_classification_audit.json'),
        'protected_old_files':2664,'old_frozen_research_modified':False,
        'formal_training_started':False,'final_analysis_labels_generated':False,
        'learning_criterion':'Report all fixed20k endpoints and full-vocabulary accuracy versus train-derived mode. No source-accuracy-based selection or retraining; near-baseline endpoints limit interpretation.',
        'independent_units':2,'relation_types':2,'new_formal_conditions':8,
        'budget_origin_is_once_saved':True,'stop_after_requested_report':True}
    for job in c['training_manifest']:
        u=data[job['unit']];ref=refs[job['unit']]
        job.update({k:ref[k] for k in ('model_seed','initial_checkpoint_path','initial_checkpoint_sha256','initial_parameter_sha256','starting_rng_path','starting_rng_sha256','starting_rng_hash')})
        job.update(data_path=u['data_path'],data_sha256=u['data_sha256'],batch_schedule_path=u['schedule_path'],
                   batch_schedule_sha256=u['schedule_sha256'],checkpoint_path=str(ROOT/'models'/job['run_id']/'checkpoint.pt'))
    assert config_signature(c)==ready['preparation_config_signature'],'Base data/training/unit/engineering settings may not change after data preparation'
    sources={str(p):sha(p) for p in sorted(SOURCE.glob('*.py'))}
    inherited=['experiments/permworld_encoding/core.py','experiments/longrun_attention.py',
               'external/neurips/src/neurips_permutations/models.py','external/neurips/src/neurips_permutations/passage.py',
               'diagnostics/permworld_scope_inventory_20261009/metrics.py']
    for name in inherited:sources[str(WORKSPACE/name)]=sha(WORKSPACE/name)
    sources[str(DIAG/'freeze.py')]=sha(DIAG/'freeze.py')
    json_write(DIAG/'protocol.json',c)
    csv_write(DIAG/'training_manifest.csv',c['training_manifest'])
    json_write(DIAG/'training_manifest.json',c['training_manifest'])
    f=dict(status='frozen',utc=c['freeze']['utc'],protocol_sha256=sha(DIAG/'protocol.json'),
        source_sha256=sources,source_version_sha256=payload_sha(sources),candidate_manifest_sha256=sha(DIAG/'candidate_manifest.csv'),
        training_manifest_sha256=sha(DIAG/'training_manifest.csv'),
        git_HEAD=subprocess.check_output(['git','rev-parse','HEAD'],cwd=WORKSPACE,text=True).strip(),
        git_status=subprocess.check_output(['git','status','--porcelain=v1'],cwd=WORKSPACE,text=True),
        Python=platform.python_version(),environment={name:importlib.metadata.version(name) for name in ('torch','numpy','matplotlib','threadpoolctl')},
        formal_training_started=False,geometry_computed=False,final_labels_generated=False,
        known_previous_results=True,no_confirmatory_relabel=True,
        initial_parameter_sha256={u:r['initial_parameter_sha256'] for u,r in refs.items()},
        GPU_deadline_monotonic=g['deadline_monotonic'],cost_rechecked_remaining_seconds=g['deadline_monotonic']-time.monotonic(),
        predicted_remaining_seconds_with_headroom=required)
    json_write(DIAG/'freeze_manifest.json',f)
    seals=seal_analysis_labels(c,ROOT,DIAG/'freeze_manifest.json',DIAG/'protocol.json')
    json_write(DIAG/'analysis_label_seals.json',seals)
    print(dict(status='frozen_and_final_labels_sealed',protocol_sha256=f['protocol_sha256'],source_version_sha256=f['source_version_sha256']),flush=True)

if __name__=='__main__':main()
