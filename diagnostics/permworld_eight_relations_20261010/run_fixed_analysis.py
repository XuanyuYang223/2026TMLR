"""Run only the frozen analysis stages after the bounded source queue ends.

This orchestration adds process resource accounting, never a scientific choice.
GraphWorld remains a separate explicit command after the PermWorld report.
"""
import argparse
import ast
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

WORKSPACE = Path('/home/yangx/ICML')
DIAG = WORKSPACE / 'diagnostics/permworld_eight_relations_20261010'
ROOT = Path('/mnt/yangx-projects-f/20261010/permworld_eight_relations')
PYTHON = str(WORKSPACE / '.venv/bin/python')

PERM_STAGES = [
    ('perm_extract', ['-B', '-m', 'experiments.permworld_eight_relations_20261010.extract', '--stage', 'extract']),
    ('perm_extraction_validation', ['-B', '-m', 'experiments.permworld_eight_relations_20261010.extract', '--stage', 'validate']),
    ('perm_geometry', ['-B', '-m', 'experiments.permworld_eight_relations_20261010.analysis', '--run']),
    ('perm_report', ['-B', '-m', 'experiments.permworld_eight_relations_20261010.report']),
    ('perm_reader_audit', ['-B', 'diagnostics/permworld_eight_relations_20261010/metric_review/final_reader_audit.py']),
]
GRAPH_STAGES = [
    ('graph_verify', ['-B', '-m', 'experiments.graphworld_pilot_20261010', '--stage', 'verify']),
    ('graph_engineering', ['-B', '-m', 'experiments.graphworld_pilot_20261010', '--stage', 'engineering']),
    ('graph_train_four', ['-B', '-m', 'experiments.graphworld_pilot_20261010', '--stage', 'train-all']),
    ('graph_evaluate', ['-B', '-m', 'experiments.graphworld_pilot_20261010', '--stage', 'evaluate']),
    ('graph_independent_metrics', ['-B', '-m', 'experiments.graphworld_pilot_20261010.validate', '--stage', 'actual']),
    ('graph_report', ['-B', '-m', 'experiments.graphworld_pilot_20261010', '--stage', 'report']),
    ('graph_reader_audit', ['-B', '-m', 'experiments.graphworld_pilot_20261010.validate', '--stage', 'audit-report']),
]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.pending')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def verify_orchestration():
    freeze = read(DIAG / 'orchestration_freeze.json')
    assert freeze['source_sha256'] == sha(__file__)
    # JSON stores stage tuples as lists; compare the exact serialized structure.
    assert freeze['perm_stages'] == json.loads(json.dumps(PERM_STAGES))
    assert freeze['graph_stages'] == json.loads(json.dumps(GRAPH_STAGES))
    for name, expected in freeze['dependent_freeze_sha256'].items():
        assert sha(name) == expected, ('Freeze changed', name)
    for name in freeze['dependent_freeze_sha256']:
        f = read(name)
        for source, expected in f.get('source_sha256', {}).items():
            assert sha(source) == expected, ('Frozen source changed', source)


def latest_source_progress(state):
    result = {k: state.get(k) for k in ('status', 'completed_models', 'authorized_missing_models', 'current_run')}
    run = state.get('current_run')
    if run:
        path = ROOT / 'logs' / (run + '.log')
        if path.exists():
            with path.open('rb') as f:
                f.seek(max(0, path.stat().st_size - 65536))
                lines = f.read().decode(errors='replace').splitlines()
            for line in reversed(lines):
                try:
                    row = ast.literal_eval(line)
                except (ValueError, SyntaxError):
                    continue
                if isinstance(row, dict) and 'step' in row:
                    result.update(step=row['step'], last_logged_elapsed_seconds=row.get('elapsed_seconds'))
                    break
    return result


def wait_queue():
    begun = time.monotonic()
    while True:
        state = read(ROOT / 'queue_state.json')
        assert state['authorized_missing_models'] == 48
        if state['status'] == 'failed':
            raise RuntimeError(state)
        if state['status'] == 'completed':
            assert state['completed_models'] == 48
            break
        assert state['status'] == 'running', state
        print(json.dumps(dict(event='source_queue_progress', utc=utc(), **latest_source_progress(state))), flush=True)
        time.sleep(45)
    write(DIAG / 'orchestration_wait.json', dict(status='passed', queue_completed_utc=utc(),
        orchestration_wait_seconds=time.monotonic() - begun, wait_overlaps_training=True,
        cost_not_added_to_queue_or_source_process_wall=True, queue_sha256=sha(ROOT / 'queue_state.json')))


def run_stage(name, argv):
    logs = ROOT / 'pipeline_logs'
    logs.mkdir(parents=True, exist_ok=True)
    log = logs / (name + '.log')
    resources = logs / (name + '.resources.json')
    records_path = DIAG / 'phase_timing.json'
    timing = read(records_path) if records_path.exists() else dict(
        phases=[], CPU_scope='GNU time process user/system including children; host CPU, not device kernels',
        GPU_kernel_seconds=None, GPU_kernel_seconds_reason='No CUDA kernel-time instrumentation',
        phase_wall_times_must_not_be_added_to_overlapping_queue_wait=True)
    assert all(r['phase'] != name for r in timing['phases']), ('Refusing an unrecorded repeated stage', name)
    cmd = [PYTHON] + argv
    fmt = '{"user_cpu_seconds":%U,"system_cpu_seconds":%S,"max_rss_kib":%M,"exit_code":%x}'
    start_utc = utc()
    begun = time.monotonic()
    print(json.dumps(dict(event='phase_start', phase=name, utc=start_utc, log=str(log))), flush=True)
    write(DIAG / 'orchestration_state.json', dict(status='running', phase=name, start_utc=start_utc, command=cmd))
    with log.open('w') as f:
        process = subprocess.Popen(['/usr/bin/time', '-f', fmt, '-o', str(resources)] + cmd,
            stdout=f, stderr=subprocess.STDOUT, cwd=WORKSPACE)
        while True:
            try:
                result = process.wait(timeout=45)
                break
            except subprocess.TimeoutExpired:
                print(json.dumps(dict(event='phase_progress', phase=name, elapsed_seconds=time.monotonic()-begun,
                    utc=utc(), log=str(log))), flush=True)
    elapsed = time.monotonic() - begun
    try:
        measured = read(resources)
    except (OSError, json.JSONDecodeError):
        measured = dict(user_cpu_seconds=None, system_cpu_seconds=None, max_rss_kib=None, timing_parse_failed=True)
    row = dict(phase=name, command=cmd, start_utc=start_utc, finish_utc=utc(), wall_seconds=elapsed,
        log_path=str(log), resource_path=str(resources), timing_scope='nonoverlapping orchestrated process phase',
        **measured)
    row['exit_code'] = result
    timing['phases'].append(row)
    write(records_path, timing)
    print(json.dumps(dict(event='phase_finished', **row)), flush=True)
    if result:
        write(DIAG / 'orchestration_state.json', dict(status='failed', phase=name, exit_code=result, log=str(log)))
        raise SystemExit(result)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', choices=['permworld', 'graphworld'], required=True)
    args = p.parse_args()
    verify_orchestration()
    if args.stage == 'permworld':
        wait_queue()
        stages = PERM_STAGES
    else:
        assert read(DIAG / 'metric_review/final_reader_audit.json')['status'] == 'passed'
        assert read(DIAG / 'report_audit.json')['status'] == 'passed'
        assert read(ROOT / 'queue_state.json')['status'] == 'completed'
        stages = GRAPH_STAGES
    for name, argv in stages:
        verify_orchestration()
        run_stage(name, argv)
    write(DIAG / 'orchestration_state.json', dict(status=args.stage + '_stages_completed', finish_utc=utc(),
        stages=[x[0] for x in stages], additional_training_started=False if args.stage=='permworld' else 'only_four_frozen_graph_jobs',
        automatic_scope_extension=False))


if __name__ == '__main__':
    main()
