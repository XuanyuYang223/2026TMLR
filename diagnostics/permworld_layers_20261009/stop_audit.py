"""Verify finished fixed jobs and absence of owned training commands; no training."""
import ast
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = Path('/mnt/yangx-projects-f/20261009/permworld_new_artifacts/encoding_archdev')


def js(path):
    return json.loads(Path(path).read_text())


def main():
    manifest = js(ROOT / 'training_manifest.json')
    expected = {job['run_id'] for job in manifest}
    assert len(manifest) == len(expected) == 16
    engineering = js(ROOT / 'engineering_complete.json')
    assert engineering['models'] == 16
    assert engineering['formal_training_started'] is False
    assert engineering['stop_after_report'] is True
    sessions = js(HERE / 'session_exit_results.json')
    assert set(sessions) == {'main', 'tail', 'middle', 'reserved', 'recoils'}
    assert all(item['exit_code'] == 0 for item in sessions.values())
    owners = []
    for name in ['parallel_execution.json', 'parallel_middle_execution.json',
                 'parallel_reserved_execution.json', 'parallel_recoils_execution.json']:
        audit = js(ROOT / name)
        assert audit['status'] == 'completed'
        assert audit['models_added'] == 0
        assert audit['formal_training_started'] is False
        jobs = audit.get('jobs', [audit])
        for job in jobs:
            assert job['returncode'] == 0
            owners.append(dict(run_id=job['run_id'], execution_owner=name))
    # Only real full-source records reach20k; overfit/resume gates have lower
    # budgets. Already-completed skip records do not contain this 'run' key.
    for line in (ROOT / 'preflight_run.log').read_text().splitlines():
        try:
            item = ast.literal_eval(line)
        except (ValueError, SyntaxError):
            continue
        if isinstance(item, dict) and item.get('run') in expected and item.get('step') == 20000:
            owners.append(dict(run_id=item['run'], execution_owner='serial_main'))
    counts = Counter(row['run_id'] for row in owners)
    assert set(counts) == expected
    assert all(count == 1 for count in counts.values()), 'Duplicate full-source execution owner'
    completed = [js(ROOT / 'models' / job['run_id'] / 'completed.json') for job in manifest]
    assert all(row['step'] == row['finite_gradient_updates'] == 20000 for row in completed)
    assert js(HERE / 'final_audit.json')['status'] == 'passed'
    commands = subprocess.check_output(['ps', '-eo', 'args'], text=True).splitlines()
    pattern = re.compile(r'python(?:\d+(?:\.\d+)?)?\s+-m\s+experiments\.permworld_encoding_archdev_20261009\s+--stage\s+(?:train(?:-one)?|overfit|resume)\b')
    active = [command for command in commands if pattern.search(command)]
    assert not active, ('Owned source training still active', active)
    last_completion = max((ROOT / 'models' / job['run_id'] / 'completed.json').stat().st_mtime
                          for job in manifest)
    start = js(HERE / 'development_training_start.json')['observed_training_driver_state']['started_utc']
    start_time = datetime.fromisoformat(start.replace('Z', '+00:00')).timestamp()
    result = dict(status='passed', created_utc=datetime.now(timezone.utc).isoformat(),
                  completed_development_models=16, updates_per_model=20000,
                  active_owned_training_commands=active, formal_training_started=False,
                  stop_after_report=True, source_execution_owners=owners,
                  source_execution_owner_count_per_model=dict(counts),
                  training_stage_observed_start_utc=start,
                  final_source_completed_utc=datetime.fromtimestamp(last_completion, timezone.utc).isoformat(),
                  source_stage_wall_seconds=last_completion-start_time,
                  sum_model_loop_elapsed_seconds=sum(row['elapsed_seconds'] for row in completed),
                  timing_scope='Wall time from recorded training driver stage start to last completion file; per-model loop elapsed includes validation/checkpoint costs and overlaps across processes. Neither measures GPU compute hours.',
                  gpu_compute_hours_measured=False,
                  protected_f17_files_unchanged=js(HERE / 'final_audit.json')['protected_f17_files_unchanged'])
    (HERE / 'final_stop_audit.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
