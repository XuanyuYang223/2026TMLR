"""Read-only provenance/stop audit after the fixed development driver exits."""
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORKSPACE))
from experiments.permworld_learnability_20261010.common import (
    ROOT, DIAG, CONFIG, verify, source_version, initial, sha, json_write,
)


def main():
    config = verify()
    version = source_version()
    protected = json.loads((DIAG / 'protected_legacy_before.json').read_text())
    differences = []
    for name, digest in protected.items():
        path = Path(name)
        path = path if path.is_absolute() else WORKSPACE / path
        if not path.is_file() or sha(path) != digest:
            differences.append(name)
    assert not differences, ('Protected legacy artifacts changed', differences)
    assert sha(ROOT / 'protocol.json') == sha(CONFIG)
    assert json.loads((ROOT / 'training_manifest.json').read_text()) == config['training_manifest']
    execution = json.loads((ROOT / 'development_execution.json').read_text())
    assert execution['status'] in ('completed', 'budget_exhausted', 'budget_exhausted_before_resume_gate')
    budget = json.loads((ROOT / 'gpu_budget.json').read_text())
    assert budget['limit_seconds'] == config['budget']['active_gpu_wall_seconds'] == 7200
    assert abs(budget['deadline_epoch']-budget['start_epoch']-7200) < 1e-5
    assert abs(budget['deadline_epoch']-budget['stop_updates_epoch']-30) < 1e-5
    cpu = json.loads((ROOT / 'cpu_check.json').read_text())
    assert cpu['status'] == 'passed' and len(cpu['records']) == 16
    resume_path = ROOT / 'resume_check.json'
    if resume_path.exists():
        resume = json.loads(resume_path.read_text())
        assert resume['status'] in ('passed', 'budget_exhausted')
    else:
        assert execution['status'] == 'budget_exhausted_before_resume_gate'
        resume = dict(status='not_reached')
    if resume['status'] == 'passed':
        assert len(resume['records']) == 4
        for item in resume['records']:
            assert item['parameter_max_difference'] <= 1e-5
            assert item['optimizer_max_difference'] <= 1e-5 and item['RNG_identical']
    init = {}
    for architecture in config['architectures']:
        # initial() can construct missing weights; this audit must never do that.
        assert (ROOT / 'initializations' / architecture / 'step0.pt').is_file()
        assert (ROOT / 'initializations' / architecture / 'step0.json').is_file()
        saved, metadata = initial(config, architecture)
        init[architecture] = metadata
        assert saved['stage'] == 'actual_constructor_output_before_training_or_optimizer'
    world = ROOT / 'worlds' / config['units'][0]['unit']
    data_audit = json.loads((world / 'data_audit.json').read_text())
    assert sha(world / 'data.npz') == data_audit['data_sha256']
    assert sha(world / 'batch_indices.npy') == data_audit['batch_indices_sha256']
    stage_records, latest, checkpoint_hashes, intervals = [], {}, {}, []
    for job in config['training_manifest']:
        directory = ROOT / 'models' / job['run_id']
        stages = []
        for path in directory.glob('stage_*.json'):
            record = json.loads(path.read_text())
            assert record['run_id'] == job['run_id'] and record['source_model_run'] is True
            assert record['protocol_sha256'] == sha(CONFIG)
            assert record['execution_source_version_sha256'] == version
            assert record['data_sha256'] == data_audit['data_sha256']
            assert record['actual_initial_identity_verified']
            metadata = init[job['architecture']]
            assert record['initial_checkpoint_sha256'] == metadata['checkpoint_sha256']
            assert record['initial_parameter_sha256'] == metadata['parameter_sha256']
            assert record['finite_gradient_updates'] == record['step']
            assert record['exposures'] == 64*record['step']
            assert record['supervised_tokens'] == 128*record['step']
            start, end = record['invocation_start_epoch'], record['invocation_end_epoch']
            assert budget['start_epoch'] <= start < budget['stop_updates_epoch']
            assert start <= end
            # This is measured process wall time, not a claimed GPU-kernel time.
            assert abs((end-start)-record['invocation_elapsed_seconds']) < 1.
            assert abs(record['cleanup_after_deadline_seconds']-max(0.,end-budget['deadline_epoch'])) < 1e-5
            intervals.append(dict(run_id=record['run_id'], step=record['step'],
                start_epoch=start, end_epoch=end,
                ended_within_budget=end <= budget['deadline_epoch']))
            for flag in ('test_evaluated', 'geometry_evaluated', 'formal_training_started'):
                assert record[flag] is False
            cp = Path(record['checkpoint_path'])
            assert cp.parent == directory
            if str(cp) not in checkpoint_hashes:
                checkpoint_hashes[str(cp)] = sha(cp)
            assert checkpoint_hashes[str(cp)] == record['checkpoint_sha256']
            if record['fixed_endpoint_completed']:
                assert record['step'] == 8000 and record['validation'] is not None
            if record['validation'] is not None:
                validation = record['validation']
                assert validation['split'] == 'validation' and validation['action'] == 'identity'
                assert [r['length'] for r in validation['per_length']] == config['lengths']
                assert all(r['samples'] == 1024 for r in validation['per_length'])
            stages.append(record)
            stage_records.append(dict(run_id=record['run_id'], step=record['step'],
                checkpoint_sha256=record['checkpoint_sha256'], status=record['status']))
        if stages:
            latest[job['run_id']] = max(stages, key=lambda r:r['step'])
        marker = directory / 'completed.json'
        if marker.exists():
            item = json.loads(marker.read_text())
            assert job['run_id'] in latest
            stage = latest[job['run_id']]
            assert item['status'] == 'completed' and item['step'] == 8000
            assert {k:v for k,v in item.items() if k != 'status'} == {k:v for k,v in stage.items() if k != 'status'}
        if job['run_id'] in latest and latest[job['run_id']]['fixed_endpoint_completed']:
            assert marker.is_file()
    paired = []
    for architecture in config['architectures']:
        for task in config['tasks']:
            jobs = [j for j in config['training_manifest'] if
                    j['architecture'] == architecture and j['task'] == task]
            available = [latest[j['run_id']] for j in jobs if j['run_id'] in latest]
            if len(available) == 2:
                assert available[0]['initial_parameter_sha256'] == available[1]['initial_parameter_sha256']
                assert available[0]['starting_rng_hash'] == available[1]['starting_rng_hash']
            paired.append(dict(architecture=architecture, task=task,
                available=len(available), initial_and_starting_rng_matched=len(available)==2,
                completed_steps=[r['step'] for r in available]))
    complete = sum(r['fixed_endpoint_completed'] for r in latest.values())
    if execution['status'] == 'completed':
        assert complete == len(config['training_manifest']) == 16
        assert execution['training_wall_seconds'] <= budget['limit_seconds']
        assert all(r['ended_within_budget'] for r in intervals)
    process_list = subprocess.run(['ps', '-eo', 'pid,args'], check=True,
        text=True, capture_output=True).stdout.splitlines()
    own_workers = [line for line in process_list if
        'experiments.permworld_learnability_20261010' in line and
        any(marker in line for marker in ('--stage develop', '--stage train-one', '--stage resume'))]
    assert not own_workers, ('Development GPU workers remain', own_workers)
    result = dict(status='passed', generated_utc=datetime.now(timezone.utc).isoformat(),
        protocol_sha256=sha(CONFIG), execution_source_version_sha256=version,
        protected_legacy_files=len(protected), legacy_hash_differences=differences,
        frozen_source_files=len(json.loads((DIAG/'freeze.json').read_text())['source_sha256']),
        actual_initializations=init, stage_checkpoint_hashes=checkpoint_hashes,
        stage_records=stage_records, paired_initial_and_rng=paired,
        fixed8k_complete_models=complete, available_models=len(latest),
        fixed8k_completion_is_not_learning_qualification=True,
        learning_qualification_computed_by='Frozen report.py from validation-only criteria',
        execution_status=execution['status'], active_gpu_budget_seconds=7200,
        execution_wall_seconds=execution.get('training_wall_seconds'),
        measured_source_process_intervals=intervals,
        all_source_intervals_ended_within_budget=all(r['ended_within_budget'] for r in intervals),
        own_training_workers=own_workers, automatic_continuation_started=False,
        test_labels_loaded_by_this_audit=False, geometry_evaluated=False,
        GroupWorld_started=False, F17_changed=False,
        statistical_independence='One new data world and one initializer per architecture; 16 models are not 16 independent worlds.')
    json_write(DIAG / 'completion_audit.json', result)
    print(json.dumps({k:v for k,v in result.items() if k not in
        ('stage_checkpoint_hashes','stage_records','actual_initializations','paired_initial_and_rng',
         'measured_source_process_intervals')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
