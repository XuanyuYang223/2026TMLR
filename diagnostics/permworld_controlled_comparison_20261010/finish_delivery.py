"""Delivery metadata and a reader annotation; does not recompute any metric."""
from common import *
from datetime import datetime, timezone
from collections import Counter
import argparse
import html
import os
import shutil


def active_scientific_jobs():
    jobs = []
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            args = (entry / 'cmdline').read_bytes().split(b'\0')
            args = [a.decode(errors='replace') for a in args if a]
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        # Inspect actual executable/script arguments, never search our shell's text.
        scripts = [a for a in args[1:] if a.endswith('.py')]
        for script in scripts:
            if ('permworld_controlled_comparison_20261010' in script and
                    Path(script).name in ['analyze.py', 'extract_layers.py', 'source_learning.py']):
                jobs.append(dict(pid=int(entry.name), argv=args))
            if ('permworld_encoding_archdev_20261009' in script and
                    Path(script).name in ['training.py', 'train.py', 'run.py']):
                jobs.append(dict(pid=int(entry.name), argv=args))
    return jobs


def prepare():
    validation = js(DIAG / 'validation_audit.json')
    assert validation['status'] == 'passed'
    assert not active_scientific_jobs(), 'Scientific calculation still running'
    cost_path = DIAG / 'cost_estimate.json'
    original = DIAG / 'cost_estimate_before_execution.json'
    if not original.exists():
        shutil.copyfile(cost_path, original)
    cost = js(original)
    index = js(FROOT / 'layers/index.json')
    metric = js(DIAG / 'metric_review/analysis_audit.json')
    cost.update(status='completed_actual_stage_costs_with_original_estimate',
        original_estimate_path=str(original), original_estimate_sha256=sha(original),
        actual_inventory_wall_seconds=js(DIAG / 'inventory/source_index.json')['inventory_elapsed_seconds'],
        actual_setup_wall_seconds=js(DIAG / 'setup_audit.json')['elapsed_seconds'],
        actual_gpu_extraction_pipeline_wall_seconds=index['actual_extraction_wall_seconds'],
        actual_forward_and_store_seconds_sum=sum(r['forward_and_store_seconds'] for r in index['records']),
        actual_cpu_metric_pipeline_wall_seconds=metric['elapsed_cpu_wall_seconds'],
        actual_extraction_readonly_audit_wall_seconds=js(DIAG / 'extraction_review/index_reader_audit.json')['reader_audit_wall_seconds'],
        actual_validation_wall_seconds=validation['elapsed_seconds'],
        actual_new_cache_bytes=sum(Path(r[k + '_path']).stat().st_size for r in index['records']
            for k in ['features', 'lengths', 'answer_ce', 'answer_argmax'] if k + '_path' in r),
        timing_interpretation='GPU pipeline wall includes encoding, transfer and IO; CPU pipeline wall includes hashes, cache IO, preprocessing and metrics. Neither is device-kernel time or CPU-core hours. Parallel stage times must not be added as an elapsed-turn duration.',
        cuda_kernel_seconds_measured=False,
        real_costs_exclude_agent_reasoning_and_report_generation=True)
    json_write(cost_path, cost)
    manifest = rows(DIAG / 'comparison_manifest.csv')
    completion = dict(status='scientific_analysis_complete_report_pending',
        generated_utc=datetime.now(timezone.utc).isoformat(),
        primary_cohort='First16, fixed20k, n10..30, one saved initial per architecture',
        models=16, lengths=21, relation_instances=2,
        comparisons=len(manifest), comparison_status_counts=dict(Counter(r['match_status'] for r in manifest)),
        available_comparisons_computed=416,
        structurally_undefined_embedding_comparisons=48,
        missing_checkpoints=0, missing_original_inputs=0,
        per_length_rows=validation['per_length_rows'],
        condition_summary_rows=validation['condition_summary_rows'],
        cross_arch_same_task_rows=validation['cross_arch_same_task_rows'],
        source_learning_rows=4032,
        mathematically_undefined_Delta_rows=validation['undefined_Delta_rows'],
        auxiliary_missing=[dict(item='initial_answer_CE', rows=2016,
            reason='Not originally cached. Exact shared initial hidden states do not establish shared task-specific CE; left missing as declared before extraction.')],
        actual_costs=cost,
        all_scientific_calculations_ended=True, active_scientific_jobs=[],
        report_complete=False, stopped=False,
        new_training=False, optimizer_constructed=False, backward=False,
        new_samples=False, updates_added=0, GroupWorld_started=False, F17_changed=False,
        original768_confirmation_started=False, second8k_cohort_used=False,
        old_protected_files_unchanged=validation['protected_files'],
        automatic_followup_planned=False)
    json_write(DIAG / 'completion.json', completion)


def complete():
    for name in ['reader_report.html', 'four_part_main.png', 'report_export_audit.json',
                 'inventory/report_reader_review.txt', 'metric_review/final_reader_audit.json']:
        assert (DIAG / name).is_file(), name
    assert js(DIAG / 'report_export_audit.json')['status'] == 'passed'
    assert js(DIAG / 'metric_review/final_reader_audit.json')['status'] == 'passed'
    assert not active_scientific_jobs()
    result = js(DIAG / 'completion.json')
    result.update(status='completed_and_stopped', report_complete=True, stopped=True,
        all_scientific_calculations_ended=True, active_scientific_jobs=[],
        completed_utc=datetime.now(timezone.utc).isoformat(),
        stop_boundary='No new training or downstream experiment. Final HTML refresh/annotation and file-hash packaging only.')
    json_write(DIAG / 'completion.json', result)


def annotate():
    report = DIAG / 'reader_report.html'
    exported = js(DIAG / 'report_export_audit.json')
    assert sha(report) == exported['output_sha256']['reader_report.html']
    assert js(DIAG / 'completion.json')['status'] == 'completed_and_stopped'
    raw = report.read_text()
    summaries = rows(DIAG / 'contrast_summary.csv')
    lookup = {tuple(r[k] for k in ['architecture', 'encoding', 'relation', 'control', 'branch', 'layer', 'metric']):r
        for r in summaries if r['contrast_type'] == 'condition_summary'}
    def cell(a,e,r,c='answer_strata',m='linear_cka'):
        return lookup[a,e,r,c,'hidden','final_norm',m]
    tf_inv = cell('transformer','one_line','descent_inverse')
    gru_inv = cell('gru','one_line','descent_inverse')
    tf_comp = cell('transformer','one_line','peak_complement')
    raw_tf_comp = cell('transformer','one_line','peak_complement','raw_within_length')
    knn_gru_comp = cell('gru','one_line','peak_complement',m='knn_overlap')
    def f(row,key): return f'{number(row[key]):+.4f}'
    direction_rows = [r for r in summaries if r['contrast_type'] == 'metric_direction' and r['control'] == 'answer_strata']
    assert len(direction_rows) == 8
    assert sum(r['Bt_direction_consistent'] == 'True' for r in direction_rows) == 8
    assert sum(r['Delta_direction_consistent'] == 'True' for r in direction_rows) == 7
    assert number(knn_gru_comp['Delta']) < 0 < number(cell('gru','one_line','peak_complement')['Delta'])
    assert all(number(r['dw']) < 0 < number(r['dc']) for r in [tf_inv,gru_inv])
    findings = f'''<aside style="padding:14px;border:1px solid #b7cbd9;background:#f6faff"><b>How to interpret this round of results</b>
    <p>All three primary metrics agree on Bt direction for all eight endpoint conditions and on Δ direction for seven.
    The only Δ disagreement is GRU / one-line / peaks–valleys: neighborhood-overlap Δ={f(knn_gru_comp,'Delta')}, whereas CKA and held-out Procrustes are positive.
    This describes sign agreement, not significance, and does not guarantee agreement at every one of the 21 lengths.</p>
    <p>Initialization preferences reverse across encodings in both configurations: one-line initially favors complement more than inverse, whereas canonical cycle shows the opposite preference.
    Among four CKA encoding contrasts, two reverse Δ without reversing Bt (Transformer / peaks–valleys and GRU / descents–inverse descents).
    For Transformer / descents–inverse descents, Bt does change from positive to negative across encodings. See the full Γ decomposition below; Δ alone cannot describe learned preferences.</p>
    <p>Positive Δ for one-line / inverse comes mainly from falling wrong-pair scores: Transformer dc={f(tf_inv,'dc')}, dw={f(tf_inv,'dw')},
    GRU dc={f(gru_inv,'dc')}, dw={f(gru_inv,'dw')}. These support reshaping of pairing selectivity, not substantial convergence of correct-pair geometry.</p>
    <p>Layer choice also matters: GRU / one-line / inverse and GRU / canonical cycle / complement have negative Bt in early layers and positive Bt later; length-aggregated Bt retains its sign across layers for the other six conditions.
    Answer control changes the increment interpretation: Transformer / one-line / peaks–valleys raw CKA Δ={f(raw_tf_comp,'Delta')},
    after answer stratification Δ={f(tf_comp,'Delta')}. U-centered CKA preserves ordinary-CKA Bt and Δ directions in all eight final primary conditions; this does not rule out output association.</p>
    <p>Canonical-cycle source tasks remain near the length baseline fixed from training labels, and one-line mastery also varies by configuration and task.
    The findings support conditional dependence on encoding, configuration, and layer under a fixed budget. One world, one initialization per configuration, and two relation instances cannot establish a universal mathematical mechanism.</p></aside>'''
    annotation = '''<aside style="padding:14px;border:1px solid #b7cbd9;background:#f6faff"><b>Delivery scope and audit notes</b>
    <p>This round is complete and stopped. No training, updates, or samples were added; F17 and original results remain unchanged.</p>
    <p>The 2,016 initial answer-CE entries are explicitly missing because the original cache did not save this metric. Shared hidden initialization does not imply shared task logits. Final-model answer CE uses the full vocabulary.
    Original <a href="source_validation_eos_original.csv">teacher-forced validation answer/EOS scores for 16 models</a>are retained separately; new prompt-only answer accuracies match original answer scores item by item. EOS does not establish task learning.</p>
    <p>Output-probability geometry reuses the original conditional distribution over numbers 0..30, not full-vocabulary probabilities. Answer CE uses the full vocabulary; distinguish these quantities.
    The embedding at the logical end token is constant for a fixed length, making similarity undefined rather than a zero effect.</p>
    <p>Independent checks: <a href="metric_review/final_reader_audit.txt">Complete-table reader check</a>,
    <a href="extraction_review/index_reader_audit.txt">New-layer cache elementwise/hash check</a>,
    <a href="inventory/report_reader_review.txt">Report check</a>.
    <a href="inventory/model_inventory.csv">Actual 16-model and initialization manifest</a>,
    <a href="inventory/match_audit.txt">Fixed-item audit</a> and
    <a href="cross_arch_same_task_summary.csv">Four-task cross-configuration similarity summaries</a>are retained separately.
    The small-batch repair only changed independent intermediate-layer validation to matching full-prompt/batch shapes. The original 2e−5 tolerance, extraction formulas, and primary results were unchanged.
    <a href="extraction_review/engineering_repair.json">Repair record</a>retains the previous failure; numerical intermediate-layer differences for prefix-only execution also remain fully recorded.
    <a href="REPRODUCE.txt">Reproduction instructions</a>list actual data, initializations, frozen versions, large caches, and commands.</p></aside>'''
    needle = '<h1>PermWorld controlled representation comparison</h1>'
    assert raw.count(needle) == 1
    shutil.copyfile(report, DIAG / 'reader_report_frozen_writer.html')
    report.write_text(raw.replace(needle, needle + findings + annotation))
    json_write(DIAG / 'reader_annotation_audit.json', dict(status='passed',
        frozen_report_writer_sha256=exported['reporter_sha256'],
        original_writer_report_sha256=exported['output_sha256']['reader_report.html'],
        preserved_writer_report_sha256=sha(DIAG / 'reader_report_frozen_writer.html'),
        annotated_report_sha256=sha(report),
        annotation_helper_sha256=sha(__file__), metric_values_changed=False,
        actual_main_Bt_direction_agreement=8, actual_main_Delta_direction_agreement=7,
        source_CE_missing_rows=2016, teacher_forced_EOS_separate=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['prepare', 'complete', 'annotate'])
    args = parser.parse_args()
    {'prepare': prepare, 'complete': complete, 'annotate': annotate}[args.stage]()
    print(args.stage, 'passed', flush=True)
