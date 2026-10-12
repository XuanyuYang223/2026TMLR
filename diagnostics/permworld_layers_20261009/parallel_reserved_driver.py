"""Execute fixed manifest job 8 early; never add or select scientific conditions.

The serial driver is at job 3 and must complete five Transformer jobs before
reaching job 8. The middle worker skips this job once completed, or stops taking
jobs when its six-position guard narrows. No source code or budget is modified.
"""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path('/mnt/yangx-projects-f/20261009/permworld_new_artifacts/encoding_archdev')
WORKSPACE = Path(__file__).resolve().parents[2]


def now():
    return datetime.now(timezone.utc).isoformat()


def main():
    manifest = json.loads((ROOT / 'training_manifest.json').read_text())
    assert len(manifest) == 16
    for name in ('cpu_check.json', 'overfit_summary.json', 'resume_summary.json'):
        assert json.loads((ROOT / name).read_text())['status'] == 'passed'
    state = json.loads((ROOT / 'preflight_driver_state.json').read_text())
    assert state['stage'] == 'train' and state['formal_training_started'] is False
    index = 8
    job = manifest[index]
    assert (job['architecture'], job['encoding'], job['task']) == ('gru', 'one_line', 'descents')
    serial = next(i for i, candidate in enumerate(manifest)
                  if not (ROOT / 'models' / candidate['run_id'] / 'completed.json').exists())
    assert index - serial >= 4
    done = ROOT / 'models' / job['run_id'] / 'completed.json'
    assert not done.exists() and not (done.parent / 'checkpoint.pt').exists()
    audit_path = ROOT / 'parallel_reserved_execution.json'
    assert not audit_path.exists()
    audit = dict(started_utc=now(), status='running', development_only=True,
                 formal_training_started=False, fixed_manifest_index=index,
                 run_id=job['run_id'], serial_first_pending_index=serial,
                 minimum_start_gap=4, models_added=0, updates=20000,
                 source_code_modified=False, hyperparameters_modified=False,
                 purpose='Overlap one remaining fixed job; no score-based scheduling')
    audit_path.write_text(json.dumps(audit, indent=2) + '\n')
    env = dict(os.environ, OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
    command = [sys.executable, '-m', 'experiments.permworld_encoding_archdev_20261009',
               '--stage', 'train-one', '--run-id', job['run_id']]
    with (ROOT / 'parallel_reserved.log').open('a') as log:
        result = subprocess.run(command, cwd=WORKSPACE, env=env,
                                stdout=log, stderr=subprocess.STDOUT)
    audit.update(finished_utc=now(), returncode=result.returncode)
    if result.returncode:
        audit['status'] = 'failed'
        audit_path.write_text(json.dumps(audit, indent=2) + '\n')
        raise RuntimeError('Fixed reserved development job failed')
    completion = json.loads(done.read_text())
    assert completion['step'] == completion['finite_gradient_updates'] == 20000
    audit['status'] = 'completed'
    audit_path.write_text(json.dumps(audit, indent=2) + '\n')
    print(json.dumps(audit))


if __name__ == '__main__':
    main()
