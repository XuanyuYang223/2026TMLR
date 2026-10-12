"""Finish only the authorized four-model graph pilot after PermWorld audits.

No geometry, accuracy, significance, or model-selection gate is introduced.
This bridge stops before the final human-readable synthesis/provenance review.
"""
import importlib.util
import json
from pathlib import Path
import subprocess
import time

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location('fixed_stage_accounting', HERE / 'run_fixed_analysis.py')
RUN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUN)
STATE = HERE / 'scope_bridge_state.json'
GRAPH_ROOT = Path('/mnt/yangx-projects-f/20261010/graphworld_pilot')
READOUT = ['-B', 'diagnostics/permworld_eight_relations_20261010/final_review/result_readout.py']


def verify():
    f = RUN.read(HERE / 'scope_bridge_freeze.json')
    assert RUN.sha(__file__) == f['source_sha256']
    for name, expected in f['dependent_sha256'].items():
        assert RUN.sha(name) == expected, ('Changed bridge dependency', name)
    RUN.verify_orchestration()


def wait_perm():
    begun = time.monotonic()
    while True:
        path = HERE / 'orchestration_state.json'
        row = RUN.read(path) if path.exists() else dict(status='waiting_for48_source_endpoints')
        if row['status'] == 'permworld_stages_completed':
            assert RUN.read(HERE / 'metric_review/final_reader_audit.json')['status'] == 'passed'
            assert RUN.read(HERE / 'report_audit.json')['status'] == 'passed'
            return time.monotonic() - begun
        if row['status'] == 'failed':
            raise RuntimeError(row)
        queue = RUN.read(RUN.ROOT / 'queue_state.json')
        if queue['status'] == 'failed':
            raise RuntimeError(queue)
        progress = RUN.latest_source_progress(queue)
        print(json.dumps(dict(event='waiting_for_completed_perm_audits', utc=RUN.utc(),
            analysis_state=row, source_queue=progress)), flush=True)
        time.sleep(45)


def graph_driver():
    verify()
    cmd = [RUN.PYTHON, '-B', str(HERE / 'run_fixed_analysis.py'), '--stage', 'graphworld']
    path = RUN.ROOT / 'pipeline_logs/graph_pipeline_driver.log'
    path.parent.mkdir(parents=True, exist_ok=True)
    begun = time.monotonic()
    print(json.dumps(dict(event='fixed_graph_pilot_start', utc=RUN.utc(), command=cmd)), flush=True)
    with path.open('w') as log:
        child = subprocess.Popen(cmd, cwd=RUN.WORKSPACE, stdout=log, stderr=subprocess.STDOUT)
        while True:
            try:
                exit_code = child.wait(timeout=45)
                break
            except subprocess.TimeoutExpired:
                phase = RUN.read(HERE / 'orchestration_state.json')
                sources = list((GRAPH_ROOT / 'models').glob('*/completed.json'))
                print(json.dumps(dict(event='fixed_graph_pilot_progress', utc=RUN.utc(),
                    elapsed_seconds=time.monotonic()-begun, current_phase=phase,
                    fixed_source_receipts=len(sources), authorized_sources=4)), flush=True)
    if exit_code:
        raise RuntimeError(dict(graph_driver_exit_code=exit_code, log=str(path)))
    return dict(driver_wall_seconds=time.monotonic()-begun, log_path=str(path),
        cost_scope='Overlaps seven graph phase times; do not add again', exit_code=exit_code)


def main():
    verify()
    assert not STATE.exists(), 'A bridge state exists; no implicit repeated execution'
    RUN.write(STATE, dict(status='waiting_for_permworld', start_utc=RUN.utc(), additional_scope=False))
    try:
        wait = wait_perm()
        verify()
        RUN.run_stage('perm_result_readout', READOUT + ['--stage', 'permworld'])
        RUN.write(STATE, dict(status='only_fixed_graph_pilot_running', started_utc=RUN.utc(),
            new_graph_limit=4, perm_wait_wall_seconds=wait, wait_overlaps_source_queue_and_analysis=True))
        graph = graph_driver()
        verify()
        RUN.run_stage('graph_result_readout', READOUT + ['--stage', 'graphworld'])
        g = RUN.read(RUN.WORKSPACE / 'diagnostics/graphworld_pilot_20261010/independent_actual_small_batch.json')
        assert g['status'] == 'passed' and g['failure_count'] == 0
        audit = RUN.read(RUN.WORKSPACE / 'diagnostics/graphworld_pilot_20261010/report_reader_audit.json')
        assert audit['status'] == 'passed' and audit['source_count'] == 4
        RUN.write(STATE, dict(status='scientific_scope_completed_report_review_pending', finish_utc=RUN.utc(),
            graph_driver=graph, perm_wait_wall_seconds=wait, fixed_perm_new=48, reused_perm=16,
            fixed_graph_new=4, automatic_followup=False, geometry_based_selection=False))
        RUN.write(HERE / 'orchestration_state.json', dict(status='all_fixed_scientific_stages_completed',
            finish_utc=RUN.utc(), remaining='Final provenance audit and readable synthesis; no more training',
            automatic_training=False))
        print(json.dumps(RUN.read(STATE)), flush=True)
    except BaseException as exc:
        RUN.write(STATE, dict(status='failed', utc=RUN.utc(), error=repr(exc), automatic_retry=False))
        raise


if __name__ == '__main__':
    main()
