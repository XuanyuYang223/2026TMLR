"""Run disjoint tail jobs from the fixed development manifest, never add jobs.

The main driver retains responsibility for completion/evaluation/report.  Each
job is a separate process/RNG context.  Start only with a large manifest gap to
the serial driver, and stop taking jobs once that gap narrows.  No metric or
accuracy determines orchestration.  Source training functions stay unchanged.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

WORKSPACE = Path(__file__).resolve().parents[2]
ROOT = Path('/mnt/yangx-projects-f/20261009/permworld_new_artifacts/encoding_archdev')


def now():
    return datetime.now(timezone.utc).isoformat()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', choices=['tail', 'middle'], default='tail')
    args = parser.parse_args()
    manifest = json.loads((ROOT / 'training_manifest.json').read_text())
    assert len(manifest) == 16
    for name in ['cpu_check.json', 'overfit_summary.json', 'resume_summary.json']:
        assert json.loads((ROOT / name).read_text())['status'] == 'passed'
    state = json.loads((ROOT / 'preflight_driver_state.json').read_text())
    assert state['stage'] == 'train' and state['formal_training_started'] is False
    audit = dict(started_utc=now(), development_only=True, formal_training_started=False,
                 unchanged_fixed_manifest=True, models_added=0, source_code_modified=False,
                 purpose='Independent fixed tail jobs overlap serial driver to reduce wall time',
                 preassigned_manifest_indices=([15, 14, 13, 12] if args.worker == 'tail' else [11, 10, 9, 8]),
                 worker=args.worker, minimum_start_gap=6, jobs=[])
    path = ROOT / ('parallel_execution.json' if args.worker == 'tail' else 'parallel_middle_execution.json')
    assert not path.exists(), 'Only one tail worker is allowed'
    path.write_text(json.dumps(audit, indent=2) + '\n')
    for index in audit['preassigned_manifest_indices']:
        job = manifest[index]
        done = ROOT / 'models' / job['run_id'] / 'completed.json'
        if done.exists():
            continue
        serial = next((i for i, candidate in enumerate(manifest)
                       if not (ROOT / 'models' / candidate['run_id'] / 'completed.json').exists()), 16)
        if index - serial < audit['minimum_start_gap']:
            audit['stopped_taking_jobs_reason'] = 'Fixed manifest gap narrowed; serial driver completes remaining jobs'
            break
        assert not (done.parent / 'checkpoint.pt').exists(), 'Tail job already active or previously started'
        record = dict(run_id=job['run_id'], manifest_index=index, serial_first_pending_index=serial,
                      started_utc=now(), updates=20000)
        audit['jobs'].append(record)
        path.write_text(json.dumps(audit, indent=2) + '\n')
        environment = dict(os.environ, OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
        command = [sys.executable, '-m', 'experiments.permworld_encoding_archdev_20261009',
                   '--stage', 'train-one', '--run-id', job['run_id']]
        with (ROOT / ('parallel_' + args.worker + '.log')).open('a') as log:
            result = subprocess.run(command, cwd=WORKSPACE, env=environment,
                                    stdout=log, stderr=subprocess.STDOUT)
        record.update(finished_utc=now(), returncode=result.returncode)
        path.write_text(json.dumps(audit, indent=2) + '\n')
        if result.returncode:
            raise RuntimeError(f'Tail development job failed: {job["run_id"]}')
        completion = json.loads(done.read_text())
        assert completion['step'] == completion['finite_gradient_updates'] == 20000
    audit['finished_utc'] = now()
    audit['status'] = 'completed'
    path.write_text(json.dumps(audit, indent=2) + '\n')
    print(json.dumps(audit))


if __name__ == '__main__':
    main()
