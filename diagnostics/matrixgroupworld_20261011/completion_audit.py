"""Seal complete fixed tables while retaining the original numerical failures.

This delivery audit does not override the frozen driver's failed verification
step and does not recompute or revise scientific results.
"""
import csv
import datetime
from pathlib import Path
from experiments.matrixgroupworld_20261011.common import DIAG,ROOT,SOURCE,CONFIG,read,sha,json_write,verify,source_version
from experiments.matrixgroupworld_20261011 import budget

def main():
    c=verify()
    driver_state=read(DIAG/'execution_state.json')
    assert driver_state['status']=='independent_saved_feature_verification'
    assert read(DIAG/'analysis_completion.json')['status']=='completed'
    assert read(DIAG/'evaluation_completion.json')['status']=='completed'
    numerical=read(DIAG/'actual_numerical_verification.json')
    assert numerical['status']=='failed'
    assert numerical['status_counts']==dict(passed=64,failed=2,undefined_consistent=6)
    failures=[r for r in numerical['records'] if r['status']=='failed']
    assert len(failures)==2 and {r['role'] for r in failures}=={'correct','wrong'}
    for r in failures:
        assert (r['unit'],r['relation'],r['n'],r['control'],r['branch'],r['phase'],r['metric'])==(
            'unit001','R1',6,'raw_within_n','output_prob','trained','knn_overlap')
        assert r['tolerance']==1e-8 and abs(r['absolute_error']-1/640)<1e-15
    assert numerical['selection_check_count']==288
    assert read(DIAG/'protection_end.json')['status']=='passed'
    report=read(DIAG/'report_audit.json')
    assert report['tables_complete'] and report['verification_caveat']
    assert not report['scientific_analysis_complete'] and report['completed20k_models']==8
    assert report['run_mode']=='completed_with_verification_caveat'
    assert report['reader_input_problems']==[]
    for path,h in report['input_sha256'].items():
        assert sha(path)==h,('Report input changed after reading',path)
    for path,h in report['outputs'].items():
        assert sha(path)==h,('Report output changed',path)
    arithmetic=read(DIAG/'table_arithmetic_audit.json')
    assert arithmetic['status']=='passed_table_arithmetic_only' and arithmetic['failure_count']==0
    assert arithmetic['all_input_bytes_unchanged'] and arithmetic['existing_numerical_verification_preserved']
    diagnosis=read(DIAG/'knn_failure_diagnostic.json')
    assert diagnosis['protected_hashes_unchanged'] and not diagnosis['failed_status_replaced']
    assert not diagnosis['tolerance_relaxed'] and diagnosis['new_forward']==diagnosis['training_updates']==0
    receipts=[]
    for j in c['training_manifest']:
        r=read(ROOT/'models'/j['run_id']/'completed.json')
        assert r['formal_completed'] and r['step']==r['finite_gradient_updates']==20000
        assert r['checkpoint_sha256']==sha(j['checkpoint_path'])
        assert r['initial_parameter_sha256']==j['initial_parameter_sha256']
        assert r['actual_initial_identity_verified']
        receipts.append(dict(run_id=j['run_id'],unit=j['unit'],task=j['task'],step=r['step'],
            actual_initial_identity_verified=r['actual_initial_identity_verified'],
            checkpoint_path=j['checkpoint_path'],checkpoint_sha256=r['checkpoint_sha256'],
            parameter_sha256=r['parameter_sha256'],initial_parameter_sha256=r['initial_parameter_sha256'],
            validation_accuracy=r['final_validation']['answer_accuracy']))
    for filename,expected in {'geometry_per_n.csv':432,'geometry_summary.csv':144,'pair_per_n.csv':144,'pair_summary.csv':48,'source_learning.csv':816,'W_actual_per_anchor.csv':12288,
        'validation_curves_per_n.csv':480,'validation_curves_equal_n.csv':160,'validation_curve_endpoint_checks.csv':72}.items():
        with (DIAG/filename).open(newline='') as f:assert sum(1 for _ in csv.DictReader(f))==expected
    with (DIAG/'validation_curve_endpoint_checks.csv').open(newline='') as f:
        assert all(r['exact_equal']=='True' and float(r['signed_difference'])==0 for r in csv.DictReader(f))
    # Keep the report's bound snapshot unchanged. A later closure snapshot is
    # recorded here, with no mutation of any report input or frozen artifact.
    cost=dict(read(DIAG/'cost_snapshot.json'))
    cost.update(cpu_external_wall_seconds=budget.cpu_elapsed(),new_apparent_bytes=budget.apparent_bytes(),
        snapshot_scope='Closure-time external wall and new-file size; report retains its earlier bound cost snapshot.')
    assert cost['GPU_stage_wall_seconds']<cost['GPU_stage_limit_seconds']
    assert cost['cpu_external_wall_seconds']<cost['cpu_limit_seconds']
    assert cost['new_apparent_bytes']<cost['new_apparent_limit_bytes']
    closure=dict(status='completed_with_verification_caveat',tables_complete=True,
        scientific_analysis_complete=False,original_driver_state=driver_state['status'],
        original_numerical_status=numerical['status'],original_failure_count=2,
        original_verification_sha256=sha(DIAG/'actual_numerical_verification.json'),
        original_CPU_error_log_sha256=sha(DIAG/'CPU_analysis_run.log'),
        numerical_diagnosis_sha256=sha(DIAG/'knn_failure_diagnostic.json'),
        table_arithmetic_audit_sha256=sha(DIAG/'table_arithmetic_audit.json'),
        interpretation='All fixed models, caches and tables exist. Two output-probability kNN checks remain failed at the frozen tolerance. Numerical cause localization does not count as passing independent verification. Ordinary CKA, Procrustes and hidden-state kNN checks passed only within the prescribed saved-feature subset.',
        stopped=True,no_new_training_after_fixed_manifest=True,no_new_forward_after_extraction=True)
    json_write(DIAG/'diagnostic_closure.json',closure)
    bindings={}
    excluded={str(DIAG/'completion.json'),str(DIAG/'artifact_manifest.json')}
    for base in (DIAG,SOURCE,ROOT):
        for p in sorted(base.rglob('*')):
            if p.is_file() and str(p) not in excluded and '__pycache__' not in p.parts:
                bindings[str(p)]=dict(sha256=sha(p),apparent_bytes=p.stat().st_size)
    json_write(DIAG/'artifact_manifest.json',dict(status='sealed',scope='All named new files found in the isolated source, delivery and large-artifact directories; no claim about all old files or filesystem history.',files=bindings))
    value=dict(status='completed_with_verification_caveat',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        protocol_sha256=sha(CONFIG),source_version_sha256=source_version(),
        artifact_manifest_sha256=sha(DIAG/'artifact_manifest.json'),
        report_audit_sha256=sha(DIAG/'report_audit.json'),protection_end_sha256=sha(DIAG/'protection_end.json'),
        formal_models=8,independent_data_initialization_units=2,formal_updates=160000,
        engineering_overfit_updates=2000,engineering_resume_updates=64,receipts=receipts,
        tables_complete=True,scientific_analysis_complete=False,
        numerical_verification_status=numerical['status'],numerical_failure_count=2,
        numerical_status_counts=numerical['status_counts'],original_failures_preserved=failures,
        diagnostic_closure_sha256=sha(DIAG/'diagnostic_closure.json'),cost=cost,
        protected_scope='Start/end SHA checks of the 2664 explicitly named protected files and tracked/index git state; not every file or intermediate event.',
        stopped=True,automatic_next_stage=False,new_worlds_after_this_pilot=False,
        uploads=0,publishing=0,F17_edits=0,
        interpretation='Exploratory geometry and pairing selectivity, not functional or group-mechanism identification; fixed positive, negative and undefined results retained.')
    json_write(DIAG/'completion.json',value)
    assert sha(DIAG/'artifact_manifest.json')==value['artifact_manifest_sha256']
    for path,h in read(DIAG/'artifact_manifest.json')['files'].items():assert sha(path)==h['sha256'],('Delivery changed during closure',path)
    print({k:value[k] for k in ('status','formal_models','formal_updates','numerical_verification_status','stopped')},flush=True)

if __name__=='__main__':main()
