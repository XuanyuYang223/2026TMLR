"""Close metadata only, explicitly after root review and logical archive.

No training, model loading, activation extraction, metrics, file deletion,
publication, or mutation of any scientific result is performed here.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
PERM = HERE.parent
WORKSPACE = PERM.parent.parent
GRAPH = PERM.parent / 'graphworld_pilot_20261010'


def read(path):
    return json.loads(Path(path).read_text())


def require(value, message):
    if not value:
        raise RuntimeError(message)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def run():
    target = PERM / 'completion.json'
    require(not target.exists(), 'Completion already exists; do not refresh completed artifacts')
    audit = read(HERE / 'final_audit.json')
    archive = read(HERE / 'archive_update_audit.json')
    root = read(HERE / 'root_result_review.json')
    require(audit['status'] == archive['status'] == 'passed', 'Final provenance/archive not passed')
    for field, path in (
        ('final_audit_sha256', HERE / 'final_audit.json'),
        ('updated_archive_sha256', WORKSPACE / 'archive_index.csv'),
        ('scope_manifest_sha256', HERE / 'scope_artifact_manifest.json'),
    ):
        require(archive[field] == sha(path), 'Archive reverse binding differs: ' + field)
    combined = read(HERE / 'combined_input_hashes.json')
    require(combined['source_sha256'] == read(HERE / 'combined_reader_freeze.json')['source_sha256'],
            'Combined writer identity differs')
    for name in ('combined_report.html', 'combined_report.txt'):
        require(combined['outputs'][name]['sha256'] == sha(HERE / name), 'Combined output differs: ' + name)
    require(audit['all_fixed_stages_complete'] and not audit['old_artifacts_modified'], 'Scope/protection not passed')
    require(root['status'] == 'read_review_passed' and root['actual_results_read'], 'Root actual result review absent')
    recovery_dir = PERM / 'graph_evaluation_environment_repair_01'
    recovery = read(recovery_dir / 'repair.json')
    require(recovery['status'] == 'completed_fixed_remaining_stages' and recovery['all_checkpoint_and_cache_bytes_unchanged'],
            'Dependency recovery did not close without changing checkpoints/caches')
    require(recovery['training_updates'] == recovery['remaining_training_updates'] == 0,
            'Dependency recovery unexpectedly trained')
    require(read(PERM / 'scope_bridge_state.json')['status'] == 'scientific_scope_completed_report_review_pending', 'Bridge not closed')
    require(read(PERM / 'orchestration_state.json')['status'] == 'all_fixed_scientific_stages_completed', 'Scientific phases not closed')
    phases = read(PERM / 'phase_timing.json')['phases']
    require(len(phases) == 14 and all(p['exit_code'] == 0 and p['finish_utc'] for p in phases), 'Incomplete phase ledger')
    manifest = read(PERM / 'completed_checkpoint_manifest.json')
    graph_manifest = read(HERE / 'graph_completed_checkpoint_manifest.json')
    require(manifest['status'] == graph_manifest['status'] == 'passed' and
            (manifest['models'], manifest['reused'], manifest['new'], graph_manifest['models']) == (64, 16, 48, 4),
            'Actual source counts differ from bounded scope')
    reports = {}
    for domain in ('permworld', 'graphworld'):
        reports[domain] = read(HERE / domain / 'readout_status.json')
        require(reports[domain]['status'] == 'completed_readout' and not reports[domain].get('fixture_only', False), 'Incomplete readout')
    perm_reader = read(PERM / 'metric_review/final_reader_audit.json')
    graph_reader = read(GRAPH / 'report_reader_audit.json')
    require(perm_reader['status'] == graph_reader['status'] == 'passed' and graph_reader['source_count'] == 4,
            'Actual reader audits not passed')
    gpu_peaks = []
    for row in manifest['records'] + graph_manifest['records']:
        receipt = read(row['completed_path'])
        field = 'peak_cuda_allocated_bytes' if row['domain'] == 'permworld' else 'peak_cuda_memory_bytes'
        gpu_peaks.append(dict(domain=row['domain'], run_id=row['run_id'], reused_existing=row['reused_existing'],
                              receipt_field=field, bytes=receipt.get(field),
                              interpretation=('PyTorch allocated-memory peak retained across recorded resume invocations; absent historical measurements remain unknown.'
                                              if row['domain'] == 'permworld' else
                                              'PyTorch allocated-memory peak for this recorded invocation; not full-device or reserved memory.')))
    paths = [
        PERM / 'protocol.json', PERM / 'training_freeze.json', PERM / 'analysis_freeze.json',
        PERM / 'metadata_writer_freeze.json', PERM / 'protected_files.json',
        GRAPH / 'protocol.json', GRAPH / 'freeze_manifest.json', GRAPH / 'verification_freeze.json', GRAPH / 'protected_files.json',
        PERM / 'orchestration_freeze.json', PERM / 'scope_bridge_freeze.json', PERM / 'phase_timing.json',
        HERE / 'readout_freeze.json', HERE / 'permworld/readout_status.json', HERE / 'graphworld/readout_status.json',
        HERE / 'audit_freeze.json', HERE / 'final_audit.json', HERE / 'final_audit.txt', HERE / 'physical_file_manifest.json',
        PERM / 'completed_checkpoint_manifest.csv', PERM / 'completed_checkpoint_manifest.json',
        HERE / 'graph_completed_checkpoint_manifest.json', HERE / 'graph_cache_content_manifest.json',
        PERM / 'report.html', PERM / 'report.txt', PERM / 'report_audit.json', PERM / 'metric_review/final_reader_audit.json',
        GRAPH / 'report.html', GRAPH / 'report.txt', GRAPH / 'completion.json', GRAPH / 'report_reader_audit.json',
        GRAPH / 'independent_actual_small_batch.json', GRAPH / 'answer_control_decomposition.png',
        HERE / 'combined_reader_freeze.json', HERE / 'combined_report.html', HERE / 'combined_report.txt', HERE / 'combined_input_hashes.json',
        HERE / 'root_result_review.txt', HERE / 'root_result_review.json',
        HERE / 'actual_perm_interpretation_review.txt', HERE / 'actual_perm_reader_notes.txt', HERE / 'actual_perm_reader_notes.json',
        HERE / 'actual_graph_scope_review.txt', HERE / 'actual_graph_scope_review.json',
        HERE / 'completion_writer_review.txt',
        HERE / 'timing_discrepancy_review.txt', HERE / 'timing_discrepancy_review.json', HERE / 'resource_summary.json',
        HERE / 'final_audit_resource.json', HERE / 'final_audit_resource.txt', HERE / 'final_audit_process.log',
        HERE / 'archive_update_freeze.json', HERE / 'archive_update_audit.json', HERE / 'scope_artifact_manifest.json', HERE / 'current_entry_manifest.csv',
        WORKSPACE / 'CURRENT_RESEARCH.txt', WORKSPACE / 'archive_index.csv', WORKSPACE / 'cleanup_candidates.csv', Path(__file__),
    ]
    paths += [PERM / rel for rel in read(PERM / 'report_audit.json')['figures_sha256']]
    paths += [p for p in recovery_dir.iterdir() if p.is_file()]
    paths += [p for p in (PERM / 'archive_metadata_repair_01').iterdir() if p.is_file()]
    paths += [PERM / 'scope_bridge_state.json']
    bindings = {}
    for path in paths:
        require(path.is_file(), 'Delivery artifact absent: ' + str(path))
        bindings[str(path.resolve())] = dict(sha256=sha(path), bytes=path.stat().st_size)
    result = dict(status='completed_and_stopped', finished_utc=datetime.now(timezone.utc).isoformat(),
                  all_fixed_stages_complete=True, stopped=True, automatic_followup=False,
                  publication_performed=False, deleted_files=0, moved_files=0,
                  formal_sources=audit['formal_sources'], source_counts=dict(permworld=64, graphworld=4, new=52, reused=16),
                  actual_saved_initializations=audit['actual_saved_initializations'], saved_RNG_streams=audit['saved_RNG_streams'],
                  source_versions=audit['source_versions'], permworld_cache=audit['permworld_cache'], graphworld_cache=audit['graphworld_cache'],
                  report_coverage=reports, independent_reader_audits=dict(permworld=perm_reader, graphworld=graph_reader),
                  old_artifact_protection=audit['old_artifact_protection'], tracked_git=audit['tracked_git'],
                  old_artifacts_modified=False, F17_protected_artifacts_unchanged=True,
                  timing=audit['timing'], final_CPU_audit_resource=read(HERE / 'final_audit_resource.json'),
                  dependency_recovery=recovery,
                  failed_stage_cost_scope='The failed 17.26-second first Graph evaluation is retained separately from the 14 successful distinct stages; its source extraction caches were reused unchanged.',
                  CPU_audit_pipeline_wall_seconds=audit['CPU_audit_pipeline_wall_seconds'],
                  timing_discrepancy_review=read(HERE / 'timing_discrepancy_review.json'),
                  GPU_allocated_memory_peaks=gpu_peaks, physical_storage=audit['physical_storage'],
                  storage_snapshot_scope='Stat apparent bytes at final CPU audit; overlapping groups are not summed; later delivery metadata is outside that snapshot.',
                  inference_scope=root['inference_scope'], actual_result_review=root['findings'],
                  undefined_values_preserved=True, source_learning_failures_preserved=True,
                  archive_update=archive, artifact_bindings=bindings,
                  stop_boundary='Completed eight PermWorld relations and four-model graph pilot; no new tasks, seeds, training, metrics, tuning or publication.')
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(status=result['status'], path=str(target), new_models=52, reused_models=16, bound_artifacts=len(bindings))))


if __name__ == '__main__':
    run()
