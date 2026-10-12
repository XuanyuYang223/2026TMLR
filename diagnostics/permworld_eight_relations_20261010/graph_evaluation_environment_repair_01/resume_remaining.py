"""Explicit bounded recovery of dependency failure; no source training."""
import hashlib
import importlib.util
import json
from pathlib import Path
import time

BASE = Path(__file__).resolve().parent.parent
REPAIR = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location('frozen_driver', BASE / 'run_fixed_analysis.py')
RUN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUN)


def main():
    RUN.verify_orchestration()
    scope = RUN.read(BASE / 'scope_bridge_freeze.json')
    for path, expected in scope['dependent_sha256'].items():
        assert RUN.sha(path) == expected
    repair = RUN.read(REPAIR / 'repair.json')
    assert repair['status'] == 'environment_repaired_fixed_evaluation_pending'
    stages = RUN.GRAPH_STAGES[3:]
    assert [x[0] for x in stages] == repair['remaining_stages'][:-1]
    saved = RUN.read(REPAIR / 'artifacts_before.json')
    for group in saved.values():
        for path, expected in group.items():
            assert RUN.sha(path) == expected
    begun = time.monotonic()
    try:
        for name, argv in stages:
            RUN.verify_orchestration()
            RUN.run_stage(name, argv)
        RUN.run_stage('graph_result_readout', ['-B', 'diagnostics/permworld_eight_relations_20261010/final_review/result_readout.py', '--stage', 'graphworld'])
        graph = BASE.parent / 'graphworld_pilot_20261010'
        assert RUN.read(graph / 'independent_actual_small_batch.json')['failure_count'] == 0
        reader = RUN.read(graph / 'report_reader_audit.json')
        assert reader['status'] == 'passed' and reader['source_count'] == 4
        for group in saved.values():
            for path, expected in group.items():
                assert RUN.sha(path) == expected
        repair.update(status='completed_fixed_remaining_stages', finished_utc=RUN.utc(),
                      resume_wall_seconds=time.monotonic()-begun, all_checkpoint_and_cache_bytes_unchanged=True,
                      duplicate_extraction_forward=False, remaining_training_updates=0)
        RUN.write(REPAIR / 'repair.json', repair)
        RUN.write(BASE / 'scope_bridge_state.json', dict(status='scientific_scope_completed_report_review_pending',
                  finish_utc=RUN.utc(), fixed_perm_new=48, reused_perm=16, fixed_graph_new=4,
                  automatic_followup=False, geometry_based_selection=False,
                  explicit_environment_recovery=str(REPAIR / 'repair.json'),
                  original_failed_bridge_preserved=str(REPAIR / 'scope_bridge_state.failed.json')))
        RUN.write(BASE / 'orchestration_state.json', dict(status='all_fixed_scientific_stages_completed',
                  finish_utc=RUN.utc(), remaining='Final provenance audit and readable synthesis; no more training', automatic_training=False))
        print(json.dumps(dict(status='completed_fixed_remaining_stages', source_updates=0)), flush=True)
    except BaseException as exc:
        RUN.write(REPAIR / 'resume_failure.json', dict(status='failed', utc=RUN.utc(), error=repr(exc), automatic_retry=False))
        raise


if __name__ == '__main__':
    main()
