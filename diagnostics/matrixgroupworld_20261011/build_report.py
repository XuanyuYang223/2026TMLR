"""Read saved MatrixGroupWorld artifacts; no model, labels, cache or metric calls.

The root may invoke this after an engineering/budget stop or after complete
analysis. It is a reader, not a stopping rule, science implementation or selector.
"""
from __future__ import annotations
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import html
import json
import math
from pathlib import Path

DEFAULT_DIAG = Path('/home/yangx/ICML/diagnostics/matrixgroupworld_20261011')
DEFAULT_ROOT = Path('/mnt/yangx-projects-f/20261011/matrixgroupworld')
OLD = Path('/home/yangx/ICML/diagnostics/permworld_eight_relations_20261010/final_review')
TABLE_COUNTS = {'geometry_per_n.csv': 432, 'geometry_summary.csv': 144,
                'selection_masks.csv': 96, 'source_learning.csv': 816,
                'pair_per_n.csv': 144, 'pair_summary.csv': 48,
                'category_summary.csv': 96, 'designated_category_comparisons.csv': 64,
                'common_support_training_comparisons.csv': 48, 'designated_rank.csv': 32,
                'W_actual_per_n.csv': 48, 'W_actual_per_anchor.csv': 12288,
                'W_actual_summary.csv': 16}
METRICS = ('linear_cka', 'knn_overlap', 'procrustes')
CONTROLS = ('raw_within_n', 'answer_strata')
COMPONENTS = ('initial_correct', 'initial_wrong', 'initial_identity',
              'trained_correct', 'trained_wrong', 'trained_identity',
              'B0', 'Bt', 'dc', 'du', 'di', 'Delta')
NUMERIC = set(COMPONENTS) | {'n', 'step', 'samples', 'answer_ce', 'eos_accuracy', 'eos_ce',
    'full_vocab_answer_accuracy', 'combined_two_token_ce', 'training_mode_accuracy',
    'fixed_transform_score_initial', 'fixed_transform_score_trained', 'training_score_change',
    'designated_score', 'designated_rank', 'candidate_mean', 'candidate_median',
    'nondesignated_mean', 'nondesignated_median', 'designated_minus_all_mean',
    'designated_minus_nondesignated_mean', 'exact_score_tie_count', 'candidate_finite',
    'candidate_total', 'fit_count', 'analysis_count', 'selected_count', 'stratum_count',
    'residual_df', 'fit_residual_df', 'analysis_residual_df', 'fit_strata', 'analysis_strata',
    'wrong_formula_satisfied_count', 'correct_formula_satisfied_count', 'retained_count',
    'wrong_formula_fraction', 'equal_n_fraction', 'pooled_numerator', 'pooled_denominator',
    'pooled_fraction', 'Gcat', 'mean', 'median', 'total_count', 'finite_count', 'undefined_count',
    'common_initial_mean', 'common_trained_mean', 'designated_minus_common_initial',
    'designated_minus_common_trained', 'designated_minus_category_mean', 'designated_minus_category_median'}


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def number(value):
    if value is None or str(value).strip() in ('', 'None', 'null', 'nan', 'NaN'):
        return None
    value = float(value)
    if not math.isfinite(value):
        raise ValueError('Unexpected infinite saved value, rather than a structural undefined cell')
    return value


def json_read(path, inputs):
    path = Path(path)
    if not path.exists():
        return None
    inputs[str(path)] = sha(path)
    return json.loads(path.read_text())


def csv_read(path, inputs):
    path = Path(path)
    if not path.exists():
        return []
    inputs[str(path)] = sha(path)
    with path.open(newline='') as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        for key in row:
            if key in NUMERIC or key.endswith('_error'):
                row[key] = number(row[key])
    return rows


def fixed_mean(values):
    return math.fsum(values) / len(values) if len(values) == 3 and all(v is not None for v in values) else None


def sign(value):
    return None if value is None else 1 if value > 0 else -1 if value < 0 else 0


def counts(rows, field):
    c = Counter(sign(r[field]) for r in rows)
    return {'positive': c[1], 'negative': c[-1], 'zero': c[0], 'undefined': c[None], 'total': len(rows)}


def formatted(value, percent=False):
    if value is None:
        return 'NaN / undefined or not applicable'
    if isinstance(value, bool):
        return 'yes' if value else 'no'
    if isinstance(value, (float, int)):
        return f'{100 * value:.2f}%' if percent else f'{value:.6g}'
    return str(value)


def source_summary(rows):
    """Presentation-only fixed 1/3 aggregation of existing identity score rows."""
    groups = {}
    for row in rows:
        if row['action'] != 'identity':
            continue
        key = row['unit'], row['task'], row['condition'], row['split']
        groups.setdefault(key, []).append(row)
    result = []
    for (unit, task, condition, split), rr in sorted(groups.items()):
        rr.sort(key=lambda r: r['n'])
        if len({r['n'] for r in rr}) != len(rr):
            raise ValueError('Repeated saved source score for one scale')
        complete = [r['n'] for r in rr] == [6, 8, 10]
        s = dict(unit=unit, task=task, condition=condition, split=split, n_count=len(rr),
                 complete_fixed_three_scales=complete, weighting='1/3; missing scales retain undefined')
        for field in ('full_vocab_answer_accuracy', 'answer_ce', 'eos_accuracy', 'eos_ce',
                      'combined_two_token_ce', 'training_mode_accuracy'):
            s[field] = fixed_mean([r[field] for r in rr]) if complete else None
        s['minimum_n_answer_accuracy'] = min(r['full_vocab_answer_accuracy'] for r in rr) if complete and all(r['full_vocab_answer_accuracy'] is not None for r in rr) else None
        s['gain_over_train_mode_pp'] = 100 * (s['full_vocab_answer_accuracy'] - s['training_mode_accuracy']) if s['full_vocab_answer_accuracy'] is not None and s['training_mode_accuracy'] is not None else None
        result.append(s)
    return result


def initial_candidate_support(rows):
    """Describe saved raw-init scores; never infer unobserved equality.

    A common actual initialization produces task-independent pre-query states
    and query probabilities, checked by the evaluator. With fixed f/T and raw
    support, candidate initial scores should therefore coincide. Task-specific
    answer strata change both support and centering even with identical states.
    """
    groups = {}
    for row in rows:
        if row['control'] == 'raw_within_n':
            groups.setdefault((row['unit'], row['relation']), []).append(row)
    summaries = []
    for (unit, relation), rr in sorted(groups.items()):
        values = [r['fixed_transform_score_initial'] for r in rr]
        finite = [v for v in values if v is not None]
        lo, hi = (min(finite), max(finite)) if finite else (None, None)
        summaries.append(dict(unit=unit, relation=relation, control='raw_within_n',
                              expected_candidate_count=3, saved_candidate_count=len(rr), finite_count=len(finite),
                              initial_score_min=lo, initial_score_max=hi,
                              initial_score_range=hi-lo if finite else None,
                              exact_equal_complete_three=len(rr) == len(finite) == 3 and lo == hi,
                              evidence_scope='Saved three-size equal-weight initial scores only; no cache/metric recomputation.'))
    return summaries


def execution_mode(tables_complete, numerical_ok, engineering, cost_estimate, statuses):
    if tables_complete:
        return 'complete_analysis' if numerical_ok else 'completed_with_verification_caveat'
    if engineering.get('status') == 'failed':
        return 'engineering_failed'
    if cost_estimate.get('estimated_remaining_within_budget') is False:
        return 'estimated_cost_exceeds_bound'
    if any('interrupted' in status or 'budget' in status or 'stop' in status or 'failed' in status for status in statuses):
        return 'stopped_incomplete'
    return 'pending'


def model_inventory(config, root, inputs):
    rows = []
    for job in config.get('training_manifest', []):
        directory = Path(job['checkpoint_path']).parent if job.get('checkpoint_path') else root / 'models' / job['run_id']
        completed_path, partial_path = directory / 'completed.json', directory / 'partial.json'
        receipt = json_read(completed_path, inputs)
        completed_file = receipt is not None
        if receipt is None:
            receipt = json_read(partial_path, inputs)
        receipt = receipt or {}
        valid = (completed_file and receipt.get('step') == 20000 and receipt.get('formal_completed') is True
                 and receipt.get('finite_gradient_updates') == 20000 and receipt.get('run_id') == job['run_id']
                 and receipt.get('unit') == job['unit'] and receipt.get('task') == job['task'])
        rows.append(dict(run_id=job['run_id'], unit=job['unit'], task=job['task'],
                         actual_updates=receipt.get('step', 0), formal20k_completed=valid,
                         status=receipt.get('status', 'not_started'),
                         elapsed_process_wall_seconds=receipt.get('elapsed_seconds'),
                         sample_exposures=receipt.get('sample_exposures'),
                         supervised_tokens=receipt.get('supervised_tokens'),
                         initial_parameter_sha256=receipt.get('initial_parameter_sha256', job.get('initial_parameter_sha256')),
                         checkpoint_sha256=receipt.get('checkpoint_sha256'),
                         completed_receipt=str(completed_path) if completed_file else '',
                         receipt=receipt))
    return rows


def collect(diag, root):
    inputs = {}; diag, root = Path(diag), Path(root)
    protocol_path = diag / 'protocol.json'
    formal_protocol = protocol_path.exists()
    if not formal_protocol:
        protocol_path = diag / 'protocol_draft.json'
    config = json_read(protocol_path, inputs)
    if config is None:
        raise ValueError('No saved protocol or protocol draft')
    objects = {}
    for name in ('completion.json', 'execution_state.json', 'training_complete.json', 'gpu_engineering.json', 'freeze_manifest.json',
                 'gpu_budget.json', 'cost_snapshot.json', 'cost_estimate.json', 'analysis_completion.json',
                 'evaluation_completion.json', 'actual_numerical_verification.json', 'protection_start.json',
                 'protection_end.json', 'math_audit.json', 'math_audit_input_overlap.json', 'task_start.json',
                 'knn_failure_diagnostic.json', 'table_arithmetic_audit.json'):
        objects[name] = json_read(diag / name, inputs)
    diagnostic_text_path = diag / 'knn_failure_diagnostic.txt'
    objects['knn_failure_diagnostic.txt'] = diagnostic_text_path.read_text() if diagnostic_text_path.exists() else None
    if diagnostic_text_path.exists():
        inputs[str(diagnostic_text_path)] = sha(diagnostic_text_path)
    for name in ('training_complete.json',):
        if objects[name] is None:
            objects[name] = json_read(root / name, inputs)
    objects['engineering_audit.json'] = json_read(root / 'engineering/engineering_audit.json', inputs)
    objects['data_ready.json'] = json_read(root / 'data_ready.json', inputs)
    objects['cache_index.json'] = json_read(root / 'evaluation_cache/index.json', inputs)
    tables = {name: csv_read(diag / name, inputs) for name in TABLE_COUNTS}
    inventory = model_inventory(config, root, inputs)
    problems = []
    frozen = objects['freeze_manifest.json'] or {}
    if formal_protocol:
        if frozen.get('protocol_sha256') != sha(protocol_path):
            problems.append('Formal protocol does not match its saved freeze')
        for objname in ('analysis_completion.json', 'evaluation_completion.json', 'actual_numerical_verification.json'):
            obj = objects[objname]
            if obj is not None and (obj.get('protocol_sha256') != sha(protocol_path) or obj.get('source_version_sha256') != frozen.get('source_version_sha256')):
                problems.append('Protocol/source binding mismatch: ' + objname)
        for row in inventory:
            if row['formal20k_completed']:
                receipt = row['receipt']; job = next(j for j in config['training_manifest'] if j['run_id'] == row['run_id'])
                checks = dict(protocol_sha256=sha(protocol_path), source_version_sha256=frozen.get('source_version_sha256'),
                              **{k: job[k] for k in ('data_sha256', 'batch_schedule_sha256', 'initial_parameter_sha256', 'initial_checkpoint_sha256', 'starting_rng_hash')})
                if any(receipt.get(k) != expected for k, expected in checks.items()):
                    problems.append('Saved20k receipt identity mismatch: ' + row['run_id'])
    diagnostic = objects['knn_failure_diagnostic.json']
    if diagnostic is not None and (diagnostic.get('protocol_sha256') != sha(protocol_path) or
                                   diagnostic.get('original_verification_sha256') != inputs.get(str(diag / 'actual_numerical_verification.json'))):
        problems.append('kNN localization diagnostic does not bind the current original verification/protocol')
    for name, expected in TABLE_COUNTS.items():
        if tables[name] and len(tables[name]) != expected:
            problems.append(f'{name}: {len(tables[name])} rows instead of {expected}')
    analysis = objects['analysis_completion.json'] or {}
    if analysis.get('status') == 'completed':
        for filename, expected in analysis.get('output_sha256', {}).items():
            if not Path(filename).exists() or sha(filename) != expected:
                problems.append(f'Analysis table digest mismatch: {filename}')
        for filename, expected in TABLE_COUNTS.items():
            if len(tables[filename]) != expected:
                problems.append(f'Completed analysis missing {filename}')
    numerical = objects['actual_numerical_verification.json'] or {}
    num_ok = numerical.get('status') in ('passed', 'passed_with_unavailable_checks') and numerical.get('failure_count') == 0
    eval_ok = (objects['evaluation_completion.json'] or {}).get('status') == 'completed'
    all_models = len(inventory) == 8 and all(r['formal20k_completed'] for r in inventory)
    tables_complete = formal_protocol and all_models and eval_ok and analysis.get('status') == 'completed' and not problems
    full = tables_complete and num_ok
    verification_caveat = tables_complete and not num_ok
    eng = objects['engineering_audit.json'] or {}
    cost_estimate = eng.get('cost_estimate') or objects['cost_estimate.json'] or {}
    statuses = [str((objects[n] or {}).get('status', '')) for n in ('completion.json', 'execution_state.json')]
    mode = execution_mode(tables_complete, num_ok, eng, cost_estimate, statuses)
    old = {}
    for name, path in (
        ('PermWorld32', OLD / 'permworld/main32_final_cka_readout.csv'),
        ('PermWorldEncoding16', OLD / 'permworld/encoding16_decomposition.csv'),
        ('PermWorldSource64', OLD / 'permworld/source64_learning_and_baseline.csv'),
        ('GraphWorld4', OLD / 'graphworld/graph_main4_readout.csv'),
        ('GraphWorldSource4', OLD / 'graphworld/graph_source4_learning.csv')):
        old[name] = csv_read(path, inputs)
    return dict(diag=diag, root=root, config=config, formal_protocol=formal_protocol, inputs=inputs,
                objects=objects, tables=tables, inventory=inventory, problems=problems, mode=mode,
                full_analysis=full, tables_complete=tables_complete, verification_caveat=verification_caveat,
                historical=old, source_summary=source_summary(tables['source_learning.csv']),
                initial_candidate_support=initial_candidate_support(tables['pair_summary.csv']))


def results_text(data):
    mode = data['mode']; models = data['inventory']; done = sum(r['formal20k_completed'] for r in models)
    zh = [f'Current state: {mode}; actual complete 20k endpoints {done}/8.']
    en = [f'Run state: {mode}; {done}/8 fixed 20,000-update endpoints are complete.']
    if not data.get('tables_complete', data['full_analysis']):
        engineering = data['objects']['engineering_audit.json'] or {}
        for problem in engineering.get('problems', []):
            zh.append('Engineering stop reason: ' + problem); en.append('Engineering stop reason: ' + problem)
        if mode == 'estimated_cost_exceeds_bound':
            zh.append('The fixed plan exceeds the authorized cost estimate; the third-domain scientific comparison is incomplete, not a negative scientific result.')
            en.append('The fixed plan exceeds the authorized cost estimate; the third-domain scientific comparison is incomplete, not a negative scientific result.')
        else:
            zh.append('Without complete formal analysis, no conclusion is drawn about relation geometry, task-pair rankings or cross-domain reproduction. Engineering overfit checks implementation rather than generalization.')
            en.append('Without complete formal analysis, no conclusion is drawn about relation geometry, task-pair rankings or cross-domain reproduction. Engineering overfit checks implementation rather than generalization.')
        return zh, en
    if data.get('verification_caveat'):
        numerical = data['objects']['actual_numerical_verification.json'] or {}
        failed = [r for r in numerical.get('records', []) if r['status'] == 'failed']
        zh.append('All fixed endpoints, caches and result tables are available, but numerical verification is not fully passed. The saved results are reported with a verification caveat; scientific_analysis_complete remains false. Verification covers the prescribed first-unit R1/n6 subset of 64 inputs, not every production cell.')
        en.append('All fixed endpoints, caches and result tables are available, but numerical verification is not fully passed. The saved results are reported with a verification caveat; scientific_analysis_complete remains false. Verification covers the prescribed first-unit R1/n6 subset of 64 inputs, not every production cell.')
        zh.append(f'Saved verification counts: {numerical.get("status_counts")}; the 288 selector checks retain their original records. No failures were recorded for the planned ordinary CKA, Procrustes or hidden-state kNN checks; consistently undefined checks remain undefined.')
        en.append(f'Saved verification counts: {numerical.get("status_counts")}; the 288 selector checks retain their original records. No failures were recorded for the planned ordinary CKA, Procrustes or hidden-state kNN checks; consistently undefined checks remain undefined.')
        for r in failed:
            label = '/'.join(str(r[k]) for k in ('unit', 'relation', 'n', 'control', 'branch', 'phase', 'role', 'metric'))
            zh.append(f'Failed verification: {label}; production={r["production"]:.9g}, independent={r["independent"]:.9g}, absolute difference={r["absolute_error"]:.9g}, frozen tolerance={r["tolerance"]:.9g}. Methods, data, tolerance and the original verification result are unchanged.')
            en.append(f'Failed verification: {label}; production={r["production"]:.9g}, independent={r["independent"]:.9g}, absolute difference={r["absolute_error"]:.9g}, frozen tolerance={r["tolerance"]:.9g}. Methods, data, tolerance and the original verification result are unchanged.')
        diagnostic = data['objects'].get('knn_failure_diagnostic.json') or {}
        if diagnostic.get('status') == 'cause_localized_original_verification_remains_failed':
            zh.append('The read-only diagnosis reproduced both original arithmetic paths. For near-onehot, not exactly saturated vectors, Gram-subtraction roundoff of about 2.9e-15 at squared distances near 1e-13 reverses neighborhood boundaries; tiny preceding centering roundoff also affects a Gram boundary tie in the wrong branch. Each score discrepancy is one neighbor count out of 64×10. This localization does not replace the frozen method or turn the original failed verification into a pass.')
            en.append('The read-only diagnosis reproduced both original arithmetic paths. For near-onehot, not exactly saturated vectors, Gram-subtraction roundoff of about 2.9e-15 at squared distances near 1e-13 reverses neighborhood boundaries; tiny preceding centering roundoff also affects a Gram boundary tie in the wrong branch. Each score discrepancy is one neighbor count out of 64×10. This localization does not replace the frozen method or turn the original failed verification into a pass.')
        else:
            zh.append('Both discrepancies affect trained raw output-probability kNN. Without a bound saved diagnostic, their cause is not inferred. Output-kNN results retain this limitation; primary CKA claims are restricted to the completed validation scope.')
            en.append('Both discrepancies affect trained raw output-probability kNN. Without a bound saved diagnostic, their cause is not inferred. Output-kNN results retain this limitation; primary CKA claims are restricted to the completed validation scope.')
    source = [r for r in data['source_summary'] if r['condition'] == 'trained' and r['split'] == 'analysis']
    finite = [r for r in source if r['full_vocab_answer_accuracy'] is not None]
    if finite:
        lo = min(r['full_vocab_answer_accuracy'] for r in finite); hi = max(r['full_vocab_answer_accuracy'] for r in finite)
        gains = [r['gain_over_train_mode_pp'] for r in finite]
        zh.append(f'On identity analysis inputs, full-vocabulary answer accuracy spans {lo:.2%}–{hi:.2%}, with gains of {min(gains):+.2f}–{max(gains):+.2f} percentage points over the train-defined per-size mode. Per-size accuracy, CE and teacher-conditioned EOS are retained without adding a mastery threshold.')
        en.append(f'On identity analysis inputs, full-vocabulary answer accuracy spans {lo:.2%}–{hi:.2%}, with gains of {min(gains):+.2f}–{max(gains):+.2f} percentage points over the train-defined per-size mode. Per-size accuracy, CE and teacher-conditioned EOS are retained without adding a mastery threshold.')
    summary = data['tables']['geometry_summary.csv']
    for control, label in (('raw_within_n', 'raw'), ('answer_strata', 'answer control')):
        all_relations = [r for r in summary if r['branch'] == 'prefix_final' and r['metric'] == 'linear_cka' and r['control'] == control]
        b, d = counts(all_relations, 'Bt'), counts(all_relations, 'Delta')
        zh.append(f'Across all two-unit/four-relation prefix-state ordinary-CKA cells ({label}), Bt is positive in {b["positive"]}/{b["total"]} and Delta in {d["positive"]}/{d["total"]}. Pairing preference and increment over initialization are separate quantities.')
        en.append(f'Across all two-unit/four-relation prefix-state ordinary-CKA cells ({control}), Bt is positive in {b["positive"]}/{b["total"]} and Delta in {d["positive"]}/{d["total"]}. Pairing preference and increment over initialization are separate quantities.')
    main = [r for r in summary if r['branch'] == 'prefix_final' and r['metric'] == 'linear_cka' and r['control'] == 'answer_strata']
    for names, label, enlabel in ((('R1', 'R3'), 'primaryR1/R3', 'Primary R1/R3'), (('R2', 'R4'), 'Supplementary R2/R4', 'Supplementary R2/R4')):
        rr = [r for r in main if r['relation'] in names]
        b, d = counts(rr, 'Bt'), counts(rr, 'Delta')
        zh.append(f'{label}, answer-controlled ordinary CKA cells: {len(rr)}; Bt positive/negative/zero/undefined={b["positive"]}/{b["negative"]}/{b["zero"]}/{b["undefined"]};Delta={d["positive"]}/{d["negative"]}/{d["zero"]}/{d["undefined"]}. These descriptive cells share tasks and relations and are not independent replications.')
        en.append(f'{enlabel}: controlled ordinary-CKA Bt positive/negative/zero/undefined counts are {b["positive"]}/{b["negative"]}/{b["zero"]}/{b["undefined"]}; Delta counts are {d["positive"]}/{d["negative"]}/{d["zero"]}/{d["undefined"]}. These coupled cells are descriptive counts, not independent replications.')
    for relation in ('R1', 'R3'):
        rr = sorted([r for r in main if r['relation'] == relation], key=lambda r: r['unit'])
        components = '; '.join(f'{r["unit"]}: Bt={r["Bt"]:+.6f}, Delta={r["Delta"]:+.6f}, dc={r["dc"]:+.6f}, du={r["du"]:+.6f}' if all(r[k] is not None for k in ('Bt', 'Delta', 'dc', 'du')) else r['unit'] + ': undefined components' for r in rr)
        zh.append(f'primary{relation}: {components}.')
        en.append(f'Controlled decomposition for primary {relation}: {components}.')
    f3_n10 = sorted([r for r in data['tables']['source_learning.csv'] if r['condition'] == 'trained' and r['action'] == 'identity' and r['split'] == 'analysis' and r['task'] == 'F3' and r['n'] == 10], key=lambda r: r['unit'])
    if f3_n10:
        values = '; '.join(f'{r["unit"]}: accuracy={r["full_vocab_answer_accuracy"]:.2%}, train-mode baseline={r["training_mode_accuracy"]:.2%}, CE={r["answer_ce"]:.6f}' for r in f3_n10)
        zh.append('The F3 source for R3 remains insufficiently learned at n10, limiting cross-size mechanism interpretation; identity analysis performance is ' + values + '. Positive R3 Delta must be reported alongside these source-learning differences, not as universal convergence of correct geometry.')
        en.append('F3 remains weak at n10, limiting the interpretation of R3 across scales: ' + values + '. Positive R3 Delta must be read alongside these source-learning differences, rather than as universal convergence of the correct geometry.')
    full_n10 = [r for r in data['tables']['source_learning.csv'] if r['condition'] == 'trained' and r['action'] == 'identity'
                and r['task'] in ('F3', 'F4') and r['n'] == 10]
    if full_n10:
        for unit in (u['unit'] for u in data['config']['units']):
            for task in ('F3', 'F4'):
                rr = {r['split']: r for r in full_n10 if r['unit'] == unit and r['task'] == task}
                values = ', '.join(f'{split}={rr[split]["full_vocab_answer_accuracy"]:.2%} (CE={rr[split]["answer_ce"]:.6f})' for split in ('train', 'validation', 'analysis') if split in rr)
                zh.append(f'Complete n10 source evaluation {unit}/{task}:{values}.')
                en.append(f'Complete n10 source evaluation {unit}/{task}: {values}.')
        zh.append('At n10, both F3 units and the second F4 unit also have low full-training-set accuracy. The weakness cannot be attributed solely to validation/analysis generalization; its cause is not identified and the fixed budget is not extended.')
        en.append('At n10, both F3 units and the second F4 unit also have low full-training-set accuracy. The weakness cannot be attributed solely to validation/analysis generalization; its cause is not identified and the fixed budget is not extended.')
    rankings = [r for r in data['tables']['designated_rank.csv'] if r['phase'] == 'trained' and r['control'] == 'answer_strata']
    available = [r for r in rankings if r['designated_score'] is not None]
    first = sum(r['designated_rank'] == 1 for r in available)
    unique = sum(r['designated_rank'] == 1 and r['exact_score_tie_count'] == 1 and r['candidate_finite'] == 3 for r in available)
    above = sum(r['designated_minus_nondesignated_mean'] is not None and r['designated_minus_nondesignated_mean'] > 0 for r in available)
    zh.append(f'For the separate fixed-T task comparison, the designated task ranks first in {first}/{len(available)} finite controlled trained cells ({unique} complete, uniquely first), and exceeds the mean of the other candidates in {above}. Candidate rankings are distinct from Bt and Delta.')
    en.append(f'For the separate fixed-T task comparison, the designated task ranks first in {first}/{len(available)} finite controlled trained cells ({unique} complete, uniquely first), and exceeds the mean of the other candidates in {above}. Candidate rankings are distinct from Bt and Delta.')
    raw_rankings = [r for r in data['tables']['designated_rank.csv'] if r['phase'] == 'trained' and r['control'] == 'raw_within_n']
    raw_available = [r for r in raw_rankings if r['designated_score'] is not None]
    raw_first = sum(r['designated_rank'] == 1 for r in raw_available)
    zh.append(f'The raw fixed-T comparison ranks the designated task first in {raw_first}/{len(raw_available)}, versus {first}/{len(available)} after answer control and {above}/{len(available)} above the alternative mean. These are not counts of correct-versus-wrong Bt or Delta.')
    en.append(f'The raw fixed-T comparison ranks the designated task first in {raw_first}/{len(raw_available)}, versus {first}/{len(available)} after answer control and {above}/{len(available)} above the alternative mean. These are not counts of correct-versus-wrong Bt or Delta.')
    for field in ('Bt', 'Delta'):
        aligned = 0; total = 0
        for row in main:
            others = [r for r in summary if r['unit'] == row['unit'] and r['relation'] == row['relation'] and r['branch'] == 'prefix_final' and r['control'] == 'answer_strata']
            ss = [sign(r[field]) for r in others]
            if len(ss) == 3 and None not in ss:
                total += 1; aligned += len(set(ss)) == 1
        by_relation = {}
        for r in main:
            by_relation.setdefault(r['relation'], []).append(sign(r[field]))
        ufinite = [v for v in by_relation.values() if len(v) == 2 and None not in v]
        uagree = sum(v[0] == v[1] for v in ufinite)
        zh.append(f'{field} directions agree across the three metrics in {aligned}/{total} complete cells; ordinary CKA agrees across the two units for {uagree}/{len(ufinite)} finite relations. Near-zero signs have no significance interpretation; undefined values are not agreement.')
        en.append(f'{field} directions agree across the three metrics in {aligned}/{total} complete cells; ordinary CKA agrees across the two units for {uagree}/{len(ufinite)} finite relations. Near-zero signs have no significance interpretation; undefined values are not agreement.')
    zh.append('The two units independently repeat only this third domain. GL inputs and transpose/fixed multiplication actions do not establish learned group multiplication; the weight identities also hold for general binary square matrices. This study does not test encoding interactions, search layers or independently reproduce the previous domains.')
    en.append('The two units independently repeat only this third domain. GL inputs and transpose/fixed multiplication actions do not establish learned group multiplication; the weight identities also hold for general binary square matrices. This study does not test encoding interactions, search layers or independently reproduce the previous domains.')
    historical = data.get('historical', {})
    comparisons = []
    for name in ('PermWorld32', 'GraphWorld4'):
        rr = historical.get(name, [])
        if rr:
            b, d = counts(rr, 'Bt'), counts(rr, 'Delta')
            comparisons.append(f'{name}: Bt+ {b["positive"]}/{b["total"]}, Delta+ {d["positive"]}/{d["total"]}')
    b, d = counts(main, 'Bt'), counts(main, 'Delta')
    comparisons.append(f'MatrixGroupWorld8: Bt+ {b["positive"]}/{b["total"]}, Delta+ {d["positive"]}/{d["total"]}')
    zh.append('Saved designated primary-table counts under answer control by domain: ' + '; '.join(comparisons) + '. Graph includes only its four original answer-controlled cells, excluding answer-plus-exact-edge-count sensitivity. Bt, Delta, and controls must be distinguished; sign distributions and R1 metric directions differ. Conditions, relations, controls, and learning differ, so these proportions are not cross-domain effects or pooled independent samples.')
    en.append('Saved designated answer-controlled endpoint counts: ' + '; '.join(comparisons) + '. Graph counts use only its original four answer-controlled cells, excluding the answer-plus-exact-edge-count sensitivity. Bt, Delta and control effects require separate interpretation in every domain; sign distributions and the present R1 metric disagreement differ. Conditions, relations, controls and source learning differ, so these proportions are neither cross-domain effect estimates nor pooled independent samples.')
    return zh, en


def table(rows, fields=None):
    if not rows:
        return '<p>No saved result is available for this item.</p>'
    fields = fields or list(dict.fromkeys(k for r in rows for k in r))
    body = '<tr>' + ''.join(f'<th>{html.escape(k)}</th>' for k in fields) + '</tr>'
    for row in rows:
        body += '<tr>' + ''.join(f'<td>{html.escape(formatted(row.get(k)))}</td>' for k in fields) + '</tr>'
    return '<div class="scroll"><table>' + body + '</table></div>'


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def saved_validation_curves(data):
    """Copy the actual twenty validations per source; never choose an endpoint.

    CE train fields are the final training batch at that step, not complete
    training-set evaluations. No Torch or feature archive is read.
    """
    rows, means, endpoint = [], [], []
    expected_steps = list(range(1000, 20001, 1000))
    validation_source = {(r['run_id'], int(r['n'])): r for r in data['tables']['source_learning.csv']
                         if r['condition'] == 'trained' and r['split'] == 'validation' and r['action'] == 'identity'}
    if not data['tables_complete']:
        return rows, means, endpoint
    for model in data['inventory']:
        receipt = model['receipt']; path = model['completed_receipt']; curve = receipt['curve']
        if [r['step'] for r in curve] != expected_steps:
            raise ValueError('Saved source curve lacks one of its twenty fixed validation points')
        binding = dict(run_id=model['run_id'], unit=model['unit'], task=model['task'],
                       completed_receipt_path=path, completed_receipt_sha256=data['inputs'][path],
                       final_checkpoint_sha256=model['checkpoint_sha256'],
                       final_parameter_sha256=receipt['parameter_sha256'],
                       initial_parameter_sha256=model['initial_parameter_sha256'],
                       protocol_sha256=receipt['protocol_sha256'], source_version_sha256=receipt['source_version_sha256'])
        for item in curve:
            val = item['validation']; per = val.get('per_n', val.get('per_length'))
            if val.get('validation_complete') is not True or sorted(r['n'] for r in per) != [6, 8, 10] or len(per) != 3:
                raise ValueError('Incomplete saved three-scale validation point')
            training = {f'last_training_batch_{k}': item.get(k) for k in ('train_combined_ce', 'train_answer_ce', 'train_eos_ce')}
            for original in per:
                row = dict(**binding, step=item['step'], n=original['n'], **training,
                           last_training_batch_scheduled_n=item.get('length'),
                           training_CE_scope='The single scheduled training batch at this update; not a full training-set evaluation.',
                           validation_source='Copied original completed.json curve.validation.per_n',
                           **{f'validation_{k}': v for k, v in original.items()})
                rows.append(row)
                if item['step'] == 20000:
                    source = validation_source[model['run_id'], original['n']]
                    for curve_field, source_field in (('answer_accuracy', 'full_vocab_answer_accuracy'),
                                                       ('eos_accuracy', 'eos_accuracy'),
                                                       ('length_majority_accuracy', 'training_mode_accuracy')):
                        difference = original[curve_field] - source[source_field]
                        endpoint.append(dict(run_id=model['run_id'], unit=model['unit'], task=model['task'], step=20000,
                                             n=original['n'], metric=curve_field, original_curve_value=original[curve_field],
                                             source_learning_value=source[source_field], signed_difference=difference,
                                             exact_equal=difference == 0., completed_receipt_sha256=binding['completed_receipt_sha256'],
                                             source_learning_sha256=data['inputs'][str(data['diag'] / 'source_learning.csv')]))
                        if difference != 0.:
                            raise ValueError('Saved20k source accuracy does not exactly reproduce the identity validation CSV')
            means.append(dict(**binding, step=item['step'], n_count=3, fixed_n_weights='1/3 each', **training,
                              training_CE_scope='Last single training batch only',
                              **{f'validation_{k}': val[k] for k in ('answer_accuracy', 'eos_accuracy', 'answer_ce', 'eos_ce', 'combined_ce', 'length_majority_accuracy')},
                              mean_source='Copied original saved equal-n validation summary, not a selected checkpoint'))
    if len(rows) != 480 or len(means) != 160 or len(endpoint) != 72:
        raise ValueError('Incomplete full source curve/endpoint comparison grid')
    write_csv(data['diag'] / 'validation_curves_per_n.csv', rows)
    write_csv(data['diag'] / 'validation_curves_equal_n.csv', means)
    write_csv(data['diag'] / 'validation_curve_endpoint_checks.csv', endpoint)
    return rows, means, endpoint


def plot_data(data):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figures = {}; diag = data['diag']
    if not data['tables_complete']:
        records = (data['objects']['engineering_audit.json'] or {}).get('overfit_records', [])
        if not records:
            return figures
        fig, axs = plt.subplots(1, 2, figsize=(12, 4))
        for record in records:
            curve = record['receipt'].get('curve', [])
            points = [(r['step'], r.get('validation')) for r in curve if r.get('validation')]
            if points:
                axs[0].plot([s for s, v in points], [v['answer_accuracy'] for s, v in points], 'o-', label=record['task'])
                axs[1].plot([s for s, v in points], [v['answer_ce'] for s, v in points], 'o-', label=record['task'])
        axs[0].set(ylabel='Full188 answer accuracy', ylim=(0, 1.02), xlabel='Engineering updates')
        axs[1].set(ylabel='Answer CE (EOS separate)', xlabel='Engineering updates')
        for ax in axs: ax.grid(alpha=.2); ax.legend()
        fig.suptitle('Engineering fixed32 n6 samples; not formal generalization')
        fig.tight_layout(); path = diag / 'engineering_curves.png'; fig.savefig(path, dpi=150); plt.close(fig)
        figures['engineering'] = str(path); return figures
    labels = [f'{u["unit"]}/{task}' for u in data['config']['units'] for task in data['config']['tasks']]
    ss = {(r['unit'] + '/' + r['task'], r['split']): r for r in data['source_summary'] if r['condition'] == 'trained'}
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    for split, offset in (('train', -.22), ('validation', 0), ('analysis', .22)):
        use = [ss.get((label, split), {}) for label in labels]
        xs = [i + offset for i in range(len(labels))]
        axs[0].bar(xs, [r['full_vocab_answer_accuracy'] if r.get('full_vocab_answer_accuracy') is not None else float('nan') for r in use], width=.2, label=split)
        axs[1].plot(range(len(labels)), [r.get('answer_ce') if r.get('answer_ce') is not None else float('nan') for r in use], 'o-', label=split)
    mode = [ss.get((label, 'analysis'), {}).get('training_mode_accuracy') for label in labels]
    axs[0].plot(range(len(labels)), mode, 'kx', label='Train-defined size mode on analysis')
    axs[0].set(ylabel='Answer accuracy, equal n weights', ylim=(0, 1.04)); axs[1].set(ylabel='Answer CE, equal n weights')
    for ax in axs: ax.set_xticks(range(len(labels)), labels, rotation=35, ha='right'); ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.suptitle('All eight fixed20k models; identity inputs; EOS reported separately')
    fig.tight_layout(); path = diag / 'source_learning.png'; fig.savefig(path, dpi=150); plt.close(fig); figures['source'] = str(path)
    curves = data.get('validation_curves', [])
    if curves:
        fig, axs = plt.subplots(2, 4, figsize=(18, 8), sharex=True, sharey=True)
        colors = {6: '#3973a6', 8: '#d17c22', 10: '#399660'}
        for ax, label in zip(axs.flat, labels):
            unit, task = label.split('/')
            for n in (6, 8, 10):
                rr = sorted([r for r in curves if r['unit'] == unit and r['task'] == task and r['n'] == n], key=lambda r: r['step'])
                ax.plot([r['step'] / 1000 for r in rr], [r['validation_answer_accuracy'] for r in rr], 'o-', ms=2,
                        lw=1.1, color=colors[n], label=f'n{n}')
                if rr:
                    ax.axhline(rr[0]['validation_length_majority_accuracy'], ls='--', lw=.7, color=colors[n], alpha=.6,
                               label=f'n{n} train-mode baseline')
            ax.set_title(label); ax.grid(alpha=.2); ax.set(ylim=(0, 1.03), xlabel='Saved update / 1000', ylabel='Full188 validation answer accuracy')
        axs[0, 0].legend(fontsize=7, loc='lower right')
        fig.suptitle('All eight sources / all three sizes / all 20 fixed checkpoints; validation only, no checkpoint selection')
        fig.tight_layout(); path = diag / 'validation_accuracy_curves.png'; fig.savefig(path, dpi=150); plt.close(fig); figures['validation_curves'] = str(path)
    summary = data['tables']['geometry_summary.csv']; cells = [(u['unit'], r['relation']) for u in data['config']['units'] for r in data['config']['relations']]
    clabel = [u + '/' + r for u, r in cells]
    fig, axs = plt.subplots(2, 3, figsize=(16, 8), sharex=True)
    for ci, control in enumerate(CONTROLS):
        rr = {(r['unit'], r['relation']): r for r in summary if r['metric'] == 'linear_cka' and r['branch'] == 'prefix_final' and r['control'] == control}
        for ax, fields, title in ((axs[ci, 0], ('B0', 'Bt'), 'Pairing preference'),
                                 (axs[ci, 1], ('dc', 'du'), 'Correct/wrong score changes'),
                                 (axs[ci, 2], ('Delta',), 'Training increment')):
            for field in fields:
                ax.plot(range(8), [rr[k][field] if rr[k][field] is not None else float('nan') for k in cells], 'o-', label=field)
            ax.axhline(0, color='k', lw=.6); ax.set_title(control + ': ' + title, fontsize=10); ax.legend(); ax.grid(alpha=.2)
            ax.set_xticks(range(8), clabel, rotation=35, ha='right')
    fig.suptitle('Ordinary CKA: B0, Bt, dc, du and Delta kept separate; all four relations / two units')
    fig.tight_layout(); path = diag / 'geometry_decomposition.png'; fig.savefig(path, dpi=150); plt.close(fig); figures['decomposition'] = str(path)
    fig, axs = plt.subplots(2, 3, figsize=(16, 8), sharex=True)
    for ci, control in enumerate(CONTROLS):
        for mi, metric in enumerate(METRICS):
            ax = axs[ci, mi]
            rr = {(r['unit'], r['relation']): r for r in summary if r['metric'] == metric and r['branch'] == 'prefix_final' and r['control'] == control}
            for field in ('Bt', 'Delta'):
                ax.plot(range(8), [rr[k][field] if rr[k][field] is not None else float('nan') for k in cells], 'o-', label=field)
            ax.axhline(0, color='k', lw=.6); ax.set_title(control + ': ' + metric, fontsize=10); ax.grid(alpha=.2); ax.legend()
            ax.set_xticks(range(8), clabel, rotation=35, ha='right')
    fig.suptitle('Fixed sensitivity metrics; independent axes; gaps are undefined, never zero-filled')
    fig.tight_layout(); path = diag / 'metric_sensitivity.png'; fig.savefig(path, dpi=150); plt.close(fig); figures['sensitivity'] = str(path)
    pairs = data['tables']['pair_summary.csv']; fig, axs = plt.subplots(2, 2, figsize=(15, 8), sharex=True)
    for ci, control in enumerate(CONTROLS):
        for pi, phase in enumerate(('initial', 'trained')):
            ax = axs[ci, pi]
            for ix, key in enumerate(cells):
                rr = [r for r in pairs if (r['unit'], r['relation']) == key and r['control'] == control]
                for j, row in enumerate(rr):
                    value = row['fixed_transform_score_' + phase]
                    if value is not None:
                        ax.scatter(ix + (j - 1) * .13, value, marker='*' if row['category'] == 'designated' else 'o',
                                   s=100 if row['category'] == 'designated' else 22, color='#bd3434' if row['category'] == 'designated' else '#3973a6')
            ax.set_title(control + ': ' + phase); ax.set_ylabel('Fixed-T ordinary CKA'); ax.grid(alpha=.2)
            ax.set_xticks(range(8), clabel, rotation=35, ha='right')
    fig.suptitle('All 24 candidate task pairs: red star designated, blue circles other tasks; not Bt/Delta')
    fig.tight_layout(); path = diag / 'fixed_transform_candidates.png'; fig.savefig(path, dpi=150); plt.close(fig); figures['candidates'] = str(path)
    return figures


def build(diag=DEFAULT_DIAG, root=DEFAULT_ROOT, plots=True):
    data = collect(diag, root)
    validation_curves, validation_means, validation_checks = saved_validation_curves(data)
    data['validation_curves'] = validation_curves
    zh, en = results_text(data); figpaths = plot_data(data) if plots else {}
    diag = data['diag']; objects = data['objects']; tables = data['tables']; config = data['config']
    sections = []
    def section(title, content): sections.append('<h2>' + html.escape(title) + '</h2>' + content)
    section('Current conclusions', ''.join('<p>' + html.escape(p) + '</p>' for p in zh))
    section('English results', ''.join('<p>' + html.escape(p) + '</p>' for p in en))
    section('Scope and fixed interpretation', '<p>GL(n,F2), n=6/8/10; four single tasks and two independent data-and-initialization units; 8 models at 20k each, a 256-dimensional 4-layer 8-head Transformer, shared 188-token vocabulary, complete row-major binary encoding, and common query token. The main extraction point is final norm at input end, before the query. R1/R3 are primary and R2/R4 prespecified supplements. Transposition is an antiautomorphism and left/right multiplication are group actions, not group homomorphisms. Only transposition and fixed-action relation types are covered.</p><p>Record 160,000 formal updates separately from engineering updates: small-sample engineering at most 500/task across four tasks, and resume engineering 0→32 plus 0→16→32, 64 updates in total; these are not extra formal models. Do not overwrite previous Perm32/Graph4 tables or protected artifacts.</p><p>B0=S_correct_initial−S_wrong_initial; Bt=S_correct_trained−S_wrong_trained; dc and du are changes in the correct and wrong branches; Delta=Bt−B0=dc−du. Positive Delta establishes neither positive Bt nor an increase in the correct branch. Identical initial weights do not ensure identical initial geometry for differently transformed inputs. Retain undefined values, without replacing them by 0 or reallocating the three 1/3 size weights.</p><p>Answer control uses pair-specific joint[n,left(identity),right(identity),right(correct),right(wrong)]≥3 strata and within-group centering. Retained samples can differ by candidate, so residual geometric differences do not identify a pure causal effect of the mathematical relation. Output probabilities are full 188-token softmax vectors; EOS is scored separately given the true answer. Do not subtract output Delta from hidden Delta to estimate an additional mechanism. Ground-truth one-hot vectors are a static baseline without initialization or training increments.</p>')
    shown_models = [{k: v for k, v in r.items() if k != 'receipt'} for r in data['inventory']]
    section('Actual model endpoints and missing artifacts', table(shown_models))
    if validation_curves:
        section('Source accuracy curves and endpoint checks', '<p>All 8 models×20 saved points×3 sizes give 480 original validation rows; another 160 original equal-weight means are retained. No best point was selected and no forward pass was repeated. All 72 checks of 20k answer accuracy, EOS accuracy, and training-mode baselines exactly match validation identity rows in source_learning.csv. CSV files bind each model receipt, endpoint weights, and parameter SHA. Training CE records individual training batches at those steps, not full-training-set fit; see source_learning.csv for full-training-set endpoint evaluation.</p><p><a href="validation_curves_per_n.csv">480 per-step/per-n validation rows</a> · <a href="validation_curves_equal_n.csv">160 equal-weight validation summaries</a> · <a href="validation_curve_endpoint_checks.csv">72 exact endpoint-accuracy checks</a></p>')
    initial_cache = objects['cache_index.json'] or {}
    clone_evidence = dict(cache_index_status=initial_cache.get('status'),
                         initial_hidden_and_answer_probabilities_task_independent_bitwise=initial_cache.get('initial_hidden_and_answer_probabilities_task_independent_bitwise'),
                         evidence_source='Saved evaluator index metadata, not a new array comparison',
                         raw_initial_candidate_scores=data['initial_candidate_support'])
    section('Cloned initialization and candidate sample support', '<p>The four specialists in a unit share a genuine initialization, and the evaluator checks pre-query prefix and answer-probability caches element by element. With fixed left f, T, and raw inputs, initial candidate CKA should coincide. The table shows actual min/max/range of saved scores only; missing CSV files do not imply completed verification. Initial answer-stratified candidates can differ because right-specific joint groups, retained samples, and within-group centering differ. These are control/support differences, not task-specific geometry among different initial models.</p><p>Within each unit, exact cloned initial models have task-independent pre-query states and query probabilities, with bitwise checks recorded by the evaluator. Fixed-f/T raw initial candidate scores should coincide. Answer-controlled scores can differ through right-specific sample support and centering, even for identical initial states; this does not establish task-specific initial model geometry.</p><pre>' + html.escape(json.dumps(clone_evidence, ensure_ascii=False, indent=2)) + '</pre>')
    engineering = objects['engineering_audit.json'] or {}
    eng_rows = []
    for task in config['tasks']:
        record = next((r for r in engineering.get('overfit_records', []) if r['task'] == task), None)
        receipt = (record or {}).get('receipt', {}); val = receipt.get('final_validation') or {}
        eng_rows.append(dict(task=task, attempted=record is not None, passed=(record or {}).get('passed'),
                             actual_updates=receipt.get('step'), full188_answer_accuracy=val.get('answer_accuracy'),
                             answer_ce=val.get('answer_ce'), teacher_conditioned_eos_accuracy=val.get('eos_accuracy'),
                             EOS_ce=val.get('eos_ce'), combined_two_token_ce=val.get('combined_ce')))
    section('Engineering gate and resume checks', table(eng_rows) + '<p>Use 32 fixed n6 engineering samples and at most 500 real-learning-rate updates per task. Unexecuted items remain missing. Stopping after failure is not a formal negative result. Continuous 0→32 and independent-process 0→16→32 resume checks and budget estimates:</p><pre>' + html.escape(json.dumps({k: engineering.get(k) for k in ('status', 'problems', 'resume', 'cost_estimate', 'elapsed_seconds')}, ensure_ascii=False, indent=2)) + '</pre>')
    if objects.get('knn_failure_diagnostic.json') is not None:
        section('Read-only localization of output-kNN verification discrepancies', '<p>This localization does not replace the frozen production metric or change the original failed status. For the original 64 rows, Gram/direct-distance paths and boundary-roundoff evidence, see <a href="knn_failure_diagnostic.json">JSON</a>, <a href="knn_failure_diagnostic.txt">Diagnostic text</a>; SHA values of both files are bound in the reader-report audit.</p><pre>' + html.escape(objects.get('knn_failure_diagnostic.txt') or 'No diagnostic text is available') + '</pre>')
    if objects.get('table_arithmetic_audit.json') is not None:
        audit = objects['table_arithmetic_audit.json']
        section('Independent saved-table arithmetic review', '<p>This review checks existing CSV arithmetic, classification, coverage, and hashes only. It does not recompute activation metrics or override the two original output-kNN failures.</p><pre>' + html.escape(json.dumps({k: audit.get(k) for k in ('status', 'checks_total', 'failure_count', 'maximum_absolute_differences', 'candidate_categories', 'scope', 'existing_independent_numerical_verification_status', 'existing_independent_numerical_verification_failure_count')}, ensure_ascii=False, indent=2)) + '</pre>')
    section('Actual costs, limits, and unknown accounting quantities', '<pre>' + html.escape(json.dumps({k: objects[k] for k in ('gpu_budget.json', 'cost_snapshot.json', 'gpu_engineering.json')}, ensure_ascii=False, indent=2)) + '</pre><p>Authorized limits: 4 hours total GPU-stage wall time, 2 hours external CPU wall time outside the GPU stage, and 20GiB apparent new-artifact size. Per-model elapsed_seconds is process wall time including validation/saving, not active GPU time, and must not be added again to overall GPU-stage wall time. CPU and GPU durations must not be summed as GPU hours. Actual active kernel duration is unknown, not 0; sizes follow the deduplicated root-snapshot accounting.</p>')
    section('Mathematics, orbits, identity, and protection', '<pre>' + html.escape(json.dumps({'math': objects['math_audit.json'],
      'audit_input_actual_overlap': objects['math_audit_input_overlap.json'],
      'data_summary': {k: (objects['data_ready.json'] or {}).get(k) for k in ('status', 'cross_unit_unique_orbit_overlap', 'historical_exclusions', 'elapsed_process_wall_seconds')},
      'protection_start_status': (objects['protection_start.json'] or {}).get('status'),
      'protection_end_status': (objects['protection_end.json'] or {}).get('status'),
      'formal_protocol_saved': data['formal_protocol'], 'actual_numerical': objects['actual_numerical_verification.json']}, ensure_ascii=False, indent=2)) + '</pre><p>H embeds GL2 in the first two coordinates. Complete closed LAR / LAᵀR orbits have at most 72 members, and formal splits deduplicate packed canonical orbits. Accepted canonical orbits are not claimed equiprobable. The previous-matrix search covers only located metadata and explicitly listed archives, not all historical storage. Four models share the genuine step0 and starting RNG in each unit; initializations are independent across units. Protocols, saved hashes, and the root audit bind exact weight identities.</p>')
    if data['problems']:
        section('Report-input issues', '<pre>' + html.escape('\n'.join(data['problems'])) + '</pre>')
    for name, path in figpaths.items():
        section('Fixed figure: ' + name, '<img src="' + html.escape(Path(path).name) + '" alt="' + name + '">')
    if data['tables_complete']:
        geometry = tables['geometry_summary.csv']
        for relation_names, title in ((('R1', 'R3'), 'Primary relations R1/R3: all branch values and decompositions'), (('R2', 'R4'), 'Supplementary relations R2/R4: all branch values and decompositions')):
            section(title, table([r for r in geometry if r['relation'] in relation_names]))
        section('Complete per-size results', table(tables['geometry_per_n.csv']))
        section('Source learning: fixed 1/3 identity summaries', table(data['source_summary']))
        section('Source learning: all 816 rows, retaining every action and EOS', table(tables['source_learning.csv']))
        section('24 fixed-T task pairs: two controls, initial and trained', table(tables['pair_summary.csv']) + table(tables['pair_per_n.csv']))
        section('Designated ranks, all candidate distributions, and common support', table(tables['designated_rank.csv']) + table(tables['category_summary.csv']) + table(tables['designated_category_comparisons.csv']) + table(tables['common_support_training_comparisons.csv']) + '<p>Keep the fixed other_known_exact category empty when empty; not_designated does not prove the absence of other mathematical relations. Gcat is a relative change on common support of fixed candidates, not original Delta; exceeding the mean does not imply exceeding every candidate.</p>')
        section('Accidental wrong-formula matches and retained control samples', table(tables['W_actual_per_n.csv']) + table(tables['W_actual_summary.csv']) + table(tables['selection_masks.csv']) + '<p>Original per-anchor records are in W_actual_per_anchor.csv. Retain all accidental matches; raw and answer controls have different denominators, and equal-weight and pooled rates are reported separately.</p>')
    historical = data['historical']; historical_counts = {}
    for name in ('PermWorld32', 'GraphWorld4'):
        rr = historical[name]
        if rr:
            historical_counts[name] = dict(rows=len(rr), Bt=counts(rr, 'Bt'), Delta=counts(rr, 'Delta'))
    section('Comparison with saved PermWorld / GraphWorld results', '<p>Only previous results are reused here; no previous metrics, forward passes, or labels are recomputed. Domains, sizes, encodings, tasks, sampling distributions, wrong transformations, and controls differ; subtraction across domains is not a causal effect. The two new units belong only to MatrixGroupWorld; previous Perm/Graph independence and learning limitations remain. Previous graph-weight tasks relate to matrix row/column weights; GL inputs and GF2 mixing alone do not establish group-specific effects.</p><pre>' + html.escape(json.dumps(historical_counts, ensure_ascii=False, indent=2)) + '</pre>' +
      '<details><summary>Previous Perm32 primary cells</summary>' + table(historical['PermWorld32']) + '</details>' +
      '<details><summary>Previous encoding16 cells: Gamma0 / Gammat / GammaDelta separately</summary>' + table(historical['PermWorldEncoding16']) + '</details>' +
      '<details><summary>Previous Graph4 primary cells and source learning</summary>' + table(historical['GraphWorld4']) + table(historical['GraphWorldSource4']) + '</details>' +
      '<details><summary>Previous Perm64 source performance and mode baselines</summary>' + table(historical['PermWorldSource64']) + '</details>' +
      '<p>There is only one encoding in this round, hence no new encoding Gamma. Historical Delta reversals without Bt reversals are not reversals of learned pairing preference. Two new independent units, four relation instances, three sizes, three metrics, and 24 task pairs are not additive independent samples; no significance, equivalence, or power guarantees are claimed.</p>')
    section('Artifact links and stopping boundary', '<p>This read-only report performs no training, re-extraction, scientific metric calls, or final-analysis-label generation/reading. If stopped by engineering or budget limits, deliver completed scope only; stop after the fixed formal scope without adding models/worlds, uploading automatically, or modifying F17 or previous artifacts.</p>' +
      '<ul>' + ''.join('<li><a href="' + html.escape(name) + '">' + html.escape(name) + '</a></li>' for name in TABLE_COUNTS if (diag / name).exists()) + '</ul>' +
      '<p>See also <a href="table_arithmetic_audit.json">Independent table-arithmetic review</a>(executed separately from this report; only files actually saved and read appear in the input-SHA manifest).</p>' +
      '<pre>' + html.escape(json.dumps({'input_sha256': data['inputs']}, ensure_ascii=False, indent=2)) + '</pre>')
    document = '<!doctype html><html lang="en"><meta charset="utf-8"><title>MatrixGroupWorld study report</title><style>body{font:15px/1.6 system-ui,sans-serif;max-width:1450px;margin:30px auto;padding:0 18px}h1,h2{color:#263b55}.scroll{overflow:auto;max-height:650px}table{border-collapse:collapse;font-size:12px;white-space:nowrap}td,th{border:1px solid #ddd;padding:5px 8px}th{background:#edf1f5;position:sticky;top:0}pre{white-space:pre-wrap;font-size:12px;overflow-wrap:anywhere;background:#f4f5f7;padding:12px}img{max-width:100%}details{margin:12px 0}</style><h1>MatrixGroupWorld: fixed third-domain study</h1><p>' + html.escape(datetime.now(timezone.utc).isoformat()) + '</p>' + ''.join(sections) + '</html>\n'
    (diag / 'report.html').write_text(document)
    txt = '\n\n'.join(zh + ['English results'] + en) + '\n\n' + '\n'.join(('Scope: two data worlds plus genuine initializations, 8×20k; R1/R3 primary, R2/R4 supplementary.',
      'Report B0, Bt, dc, du, Delta and raw/answer controls separately; do not replace NaN by 0 or change weights.',
      'Fixed-T task-pair ranks are separate from correct-minus-wrong Bt/Delta.',
      'With cloned initial states within a unit, raw initial candidates at fixed f/T should coincide. Answer-stratified initial differences may reflect right-specific sample support/centering, not task-specific geometry of different initial models. See HTML for actual min/max and bitwise cache-check metadata.',
      'Output probabilities cover all 188 classes; EOS is conditioned on the true answer. hiddenDelta-outputDelta is not a mechanism estimate.',
      'Only two independent third-domain units are available; 4 relations/3 n values/3 metrics/24 candidates are not independent replications.',
      'Incomplete engineering/budget execution is not a negative scientific result. Account separately for GPU-stage wall time, external CPU wall time, and source-process wall time; active kernel time is unknown.',
      'Read saved results only; after stopping, do not extend training, publish to GitHub, or modify previous results or F17.',
      'See report.html for complete tables, engineering, audits, and historical scope.')) + '\n'
    (diag / 'report.txt').write_text(txt)
    result = {'status': 'report_written', 'run_mode': data['mode'], 'scientific_analysis_complete': data['full_analysis'],
              'tables_complete': data['tables_complete'], 'verification_caveat': data['verification_caveat'],
              'completed20k_models': sum(r['formal20k_completed'] for r in data['inventory']),
              'formal_models_planned': 8, 'units': 2, 'input_table_row_counts': {k: len(v) for k, v in tables.items()},
              'reader_input_problems': data['problems'], 'input_sha256': data['inputs'], 'reporter_sha256': sha(__file__),
              'figures': figpaths, 'no_model_calls': True, 'no_metric_calls': True, 'no_analysis_labels_read': True,
              'initial_clone_reader_evidence': clone_evidence,
              'validation_curve_export': {'per_n_rows': len(validation_curves), 'equal_n_rows': len(validation_means),
                                          'endpoint_checks': len(validation_checks), 'endpoint_all_exact_equal': all(r['exact_equal'] for r in validation_checks) if validation_checks else None,
                                          'source_receipt_sha256': {r['completed_receipt']: data['inputs'][r['completed_receipt']] for r in data['inventory'] if r['completed_receipt']},
                                          'no_model_or_forward_calls': True, 'last_training_batch_CE_is_not_full_train_CE': True},
              'outputs': {str(p): sha(p) for p in [diag / 'report.html', diag / 'report.txt', *map(Path, figpaths.values()),
                            *[diag / name for name in ('validation_curves_per_n.csv', 'validation_curves_equal_n.csv', 'validation_curve_endpoint_checks.csv') if (diag / name).exists()]]},
              'zh_results': zh, 'en_results': en, 'next_training_started': False}
    (diag / 'report_audit.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    return result


def fixture(diag):
    tests = []
    def check(name, value):
        assert value, name; tests.append({'check': name, 'passed': True})
    check('undefined aggregation does not reweight', fixed_mean([1., None, 2.]) is None)
    check('fixed_three_equal_weights', fixed_mean([1., 2., 3.]) == 2.)
    check('zero is a defined value', number('0') == 0 and sign(0.) == 0)
    check('structural NaN retained', number('nan') is None)
    try: number('inf')
    except ValueError: check('unexpected infinity rejected', True)
    else: check('unexpected infinity rejected', False)
    fake = [dict(unit='unit001', task='F1', condition='trained', split='analysis', action='identity', n=n,
                 full_vocab_answer_accuracy=.2, answer_ce=3., eos_accuracy=1., eos_ce=0.,
                 combined_two_token_ce=1.5, training_mode_accuracy=.2) for n in (6, 8, 10)]
    summary = source_summary(fake)[0]
    check('equal model and mode does not become learning gain', summary['gain_over_train_mode_pp'] == 0.)
    check('missing source scale stays undefined', source_summary(fake[:2])[0]['full_vocab_answer_accuracy'] is None)
    candidates = [dict(unit='unit001', relation='R1', control='raw_within_n', fixed_transform_score_initial=.25) for _ in range(3)]
    candidates.append(dict(unit='unit001', relation='R1', control='answer_strata', fixed_transform_score_initial=.9))
    support = initial_candidate_support(candidates)
    check('raw cloned-score agreement uses only actual three saved candidates', len(support) == 1 and support[0]['exact_equal_complete_three'] and support[0]['initial_score_range'] == 0.)
    check('missing raw candidate is not exact equality evidence', not initial_candidate_support(candidates[:2])[0]['exact_equal_complete_three'])
    check('absent candidate results remain absent', initial_candidate_support([]) == [])
    check('complete tables with numerical failure are caveated rather than pending', execution_mode(True, False, {}, {}, ['failed']) == 'completed_with_verification_caveat')
    check('complete table presentation does not imply verification pass', execution_mode(True, True, {}, {}, ['completed']) == 'complete_analysis')
    check('engineering failure without formal tables stays engineering stop', execution_mode(False, False, {'status': 'failed'}, {}, []) == 'engineering_failed')
    data = {'mode': 'engineering_failed', 'inventory': [dict(formal20k_completed=False)] * 8,
            'full_analysis': False, 'objects': {'engineering_audit.json': {'problems': ['synthetic failure']}}}
    zh, en = results_text(data)
    check('failed engineering yields no scientific conclusion', 'Without complete formal analysis' in '\n'.join(zh))
    check('report emits both languages', bool(en) and 'Run state' in en[0])
    result = {'status': 'passed', 'count': len(tests), 'checks': tests, 'synthetic_only': True,
              'source_sha256': sha(__file__), 'no_model_or_metric_or_analysis_label_calls': True}
    (Path(diag) / 'report_fixture.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--diag', type=Path, default=DEFAULT_DIAG)
    p.add_argument('--root', type=Path, default=DEFAULT_ROOT); p.add_argument('--fixture', action='store_true')
    p.add_argument('--no-plots', action='store_true'); args = p.parse_args()
    result = fixture(args.diag) if args.fixture else build(args.diag, args.root, not args.no_plots)
    print(json.dumps({k: result[k] for k in ('status',) + (('count',) if args.fixture else ('run_mode', 'completed20k_models', 'scientific_analysis_complete'))}, ensure_ascii=False))
