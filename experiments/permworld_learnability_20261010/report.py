"""Frozen, validation-only reporting for the one bounded learning preset.

This module intentionally does not import torch, the training driver, or any
representation metric. It never reads a model checkpoint or a test array.
``report()`` consumes only the fixed manifest, stage JSON records, initializer
metadata, and train/validation labels. ``--fixture`` checks the reporting gates
on synthetic CPU-only data before any real development result is observed.
"""
from __future__ import annotations

import argparse
import copy
import csv
from datetime import datetime, timezone
import hashlib
import html
import json
import math
from pathlib import Path
import tempfile

import numpy as np


WORKSPACE = Path(__file__).resolve().parents[2]
ROOT = Path('/mnt/yangx-projects-f/20261010/permworld_learnability')
DIAG = WORKSPACE / 'diagnostics/permworld_learnability_20261010'
TASKS = (
    'anti_fixed_points', 'components', 'deficiencies', 'descents', 'exceedances',
    'fixed_points', 'global_descents', 'lds_length', 'left_to_right_maxima',
    'left_to_right_minima', 'lis_length', 'longest_decreasing_run',
    'longest_increasing_run', 'peaks', 'recoils', 'valleys',
)
DEVELOPMENT_TASKS = ('descents', 'recoils', 'peaks', 'valleys')
ENCODINGS = ('one_line', 'canonical_cycle')
ARCHITECTURES = ('transformer', 'gru')
ACCURACY_FIELDS = ('full_vocab_accuracy', 'numeric_accuracy', 'eos_accuracy')
LOSS_FIELDS = ('answer_ce', 'eos_ce', 'combined_ce')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True,
                                    allow_nan=False) + '\n')


def write_csv(path, rows, fields):
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def equal_mean(records, field):
    return float(np.mean([float(row[field]) for row in records]))


def validate_measurement(validation, config):
    """Validate the complete identity validation schema, including all lengths."""
    if validation is None:
        return ['validation_missing']
    errors = []
    if validation.get('split') != 'validation' or validation.get('action') != 'identity':
        errors.append('validation_is_not_identity_validation')
    if validation.get('test_evaluated') is not False or validation.get('geometry_evaluated') is not False:
        errors.append('validation_scope_flags_invalid')
    if validation.get('eos_teacher_forced') is not True:
        errors.append('eos_measurement_not_teacher_forced')
    rows = validation.get('per_length', [])
    lengths = config['lengths']
    if [row.get('length') for row in rows] != lengths:
        errors.append('validation_lengths_do_not_match_fixed_window')
        return errors
    expected = int(config['data']['examples_per_length']['validation'])
    for row in rows:
        if row.get('samples') != expected:
            errors.append(f'validation_sample_count_n{row["length"]}')
        for field in (*ACCURACY_FIELDS, *LOSS_FIELDS):
            value = row.get(field)
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                errors.append(f'invalid_{field}_n{row["length"]}')
            elif field in ACCURACY_FIELDS and not 0 <= value <= 1:
                errors.append(f'out_of_range_{field}_n{row["length"]}')
            elif field in LOSS_FIELDS and value < 0:
                errors.append(f'negative_{field}_n{row["length"]}')
        if all(isinstance(row.get(field), (int, float)) for field in LOSS_FIELDS):
            if abs(row['combined_ce'] - (row['answer_ce'] + row['eos_ce']) / 2) > 1e-6:
                errors.append(f'combined_ce_equation_n{row["length"]}')
        if row.get('answer_accuracy') is not None and row['answer_accuracy'] != row.get('full_vocab_accuracy'):
            errors.append(f'answer_accuracy_alias_n{row["length"]}')
    total = sum(row.get('samples', 0) for row in rows)
    if total != len(lengths) * expected or validation.get('samples') != total:
        errors.append('validation_total_sample_count')
    aggregate_map = dict(full_vocab_accuracy='overall_full_vocab',
                         numeric_accuracy='overall_numeric', eos_accuracy='overall_eos',
                         answer_ce='answer_ce', eos_ce='eos_ce', combined_ce='combined_ce')
    if not errors:
        for field, overall in aggregate_map.items():
            expected_value = sum(row['samples'] * row[field] for row in rows) / total
            value = validation.get(overall)
            if value is None or not math.isfinite(value) or abs(value - expected_value) > 1e-6:
                errors.append('validation_aggregate_' + overall)
    return errors


def baseline_from_labels(path, config):
    """Decode only train/validation identity labels; test and input arrays stay unread."""
    with np.load(path, allow_pickle=False) as archive:
        data = {name: archive[name].copy() for name in (
            'train_lengths', 'train_labels', 'validation_lengths', 'validation_labels')}
    baselines = {}
    rows = []
    for split in ('train', 'validation'):
        lengths, labels = data[split + '_lengths'], data[split + '_labels']
        assert lengths.ndim == 1 and labels.ndim == 3
        expected_actions = 1 if split == 'train' else 3
        assert labels.shape[0] == len(lengths) and labels.shape[1] == expected_actions
        assert labels.shape[2] == len(TASKS)
        assert sorted(map(int, np.unique(lengths))) == config['lengths']
        expected = int(config['data']['examples_per_length'][split])
        assert all(int(np.sum(lengths == n)) == expected for n in config['lengths'])
    for task in config['tasks']:
        baselines[task] = {}
        index = TASKS.index(task)
        for length in config['lengths']:
            train = data['train_labels'][data['train_lengths'] == length, 0, index].astype(int)
            validation = data['validation_labels'][data['validation_lengths'] == length, 0, index].astype(int)
            assert np.all((0 <= train) & (train <= 30)) and np.all((0 <= validation) & (validation <= 30))
            counts = np.bincount(train, minlength=31)
            majority = int(np.argmax(counts))  # Fixed lowest-answer tie break.
            accuracy = float(np.mean(validation == majority))
            baselines[task][length] = accuracy
            rows.append(dict(task=task, length=length, training_samples=len(train),
                majority_answer=majority, majority_training_count=int(counts[majority]),
                majority_training_fraction=float(counts[majority] / len(train)),
                validation_samples=len(validation), validation_baseline_accuracy=accuracy,
                rule='Train-only identity-label mode; ties choose lowest numeric answer.'))
    return baselines, rows


def model_gates(stage, completed, baseline, config, audit_errors=()):
    """Fixed endpoint only; neither a partial run nor a good earlier score qualifies."""
    final = int(config['training']['updates'])
    complete = bool(stage and completed and stage.get('step') == final and
                    completed.get('step') == final and stage.get('fixed_endpoint_completed') is True and
                    completed.get('status') == 'completed' and stage.get('source_model_run') is True and
                    completed.get('source_model_run') is True)
    validation = stage.get('validation') if stage else None
    validation_errors = validate_measurement(validation, config)
    gate = config['learning_gate']
    values = dict(equal_length_accuracy=None, minimum_length_accuracy=None,
                  equal_length_baseline=float(np.mean(list(baseline.values()))),
                  gain_over_length_baseline_pp=None, equal_length_eos_accuracy=None,
                  equal_length_answer_ce=None, equal_length_combined_ce=None)
    failures = list(audit_errors)
    if not complete:
        failures.append('fixed_8k_endpoint_not_completed')
    failures.extend(validation_errors)
    if not validation_errors:
        rows = validation['per_length']
        values.update(equal_length_accuracy=equal_mean(rows, 'full_vocab_accuracy'),
            minimum_length_accuracy=min(row['full_vocab_accuracy'] for row in rows),
            equal_length_eos_accuracy=equal_mean(rows, 'eos_accuracy'),
            equal_length_answer_ce=equal_mean(rows, 'answer_ce'),
            equal_length_combined_ce=equal_mean(rows, 'combined_ce'))
        values['gain_over_length_baseline_pp'] = 100 * (
            values['equal_length_accuracy'] - values['equal_length_baseline'])
        comparisons = (
            ('equal_length_accuracy', gate['per_model_equal_length_accuracy_min']),
            ('minimum_length_accuracy', gate['per_length_accuracy_min']),
            ('gain_over_length_baseline_pp', gate['per_model_gain_over_training_length_majority_min_pp']),
            ('equal_length_eos_accuracy', gate['per_model_eos_accuracy_min']),
        )
        for name, threshold in comparisons:
            if values[name] + 1e-10 < threshold:
                failures.append(name + '_below_fixed_gate')
    return dict(**values, endpoint_completed=complete,
                validation_complete=not validation_errors,
                gate_pass=not failures, failures=sorted(set(failures)))


def pair_gates(first, second, config):
    """Encoding equality is assessed at two complete, equal-budget 8k endpoints."""
    rows = dict(mean_accuracy_gap_pp=None, maximum_length_accuracy_gap_pp=None,
                per_length_gaps_pp={}, same_updates=False, paired_endpoint_complete=False,
                gate_pass=False, failures=[])
    if first is None or second is None:
        rows['failures'] = ['encoding_model_missing']
        return rows
    stages = [item['stage'] for item in (first, second)]
    rows['same_updates'] = bool(all(stages) and stages[0]['step'] == stages[1]['step'])
    rows['paired_endpoint_complete'] = all(item['gates']['endpoint_completed'] and
                                           item['gates']['validation_complete'] for item in (first, second))
    if not rows['paired_endpoint_complete'] or not rows['same_updates']:
        rows['failures'] = ['paired_fixed_8k_endpoints_incomplete_or_unequal_budget']
        return rows
    values = [{r['length']: r['full_vocab_accuracy'] for r in s['validation']['per_length']} for s in stages]
    rows['per_length_gaps_pp'] = {n: 100 * abs(values[0][n] - values[1][n]) for n in config['lengths']}
    rows['mean_accuracy_gap_pp'] = 100 * abs(float(np.mean(list(values[0].values()))) -
                                           float(np.mean(list(values[1].values()))))
    rows['maximum_length_accuracy_gap_pp'] = max(rows['per_length_gaps_pp'].values())
    if rows['mean_accuracy_gap_pp'] > config['learning_gate']['paired_encoding_accuracy_gap_max_pp'] + 1e-10:
        rows['failures'].append('mean_encoding_gap_above_fixed_gate')
    if rows['maximum_length_accuracy_gap_pp'] > config['learning_gate']['paired_encoding_per_length_accuracy_gap_max_pp'] + 1e-10:
        rows['failures'].append('length_encoding_gap_above_fixed_gate')
    if first['audit_errors'] or second['audit_errors']:
        rows['failures'].append('encoding_pair_metadata_audit_failed')
    if not first['gates']['gate_pass'] or not second['gates']['gate_pass']:
        rows['failures'].append('paired_model_learning_gate_failed')
    rows['gate_pass'] = not rows['failures']
    return rows


def audit_stage(record, job, config, expected, initial):
    errors = []
    for field in ('run_id', 'unit', 'architecture', 'encoding', 'task', 'model_seed',
                  'data_seed', 'batch_seed'):
        if record.get(field) != job.get(field):
            errors.append('stage_manifest_' + field)
    references = dict(protocol_sha256=expected['protocol_sha256'],
        execution_source_version_sha256=expected['source_version_sha256'],
        data_sha256=expected['data_sha256'],
        initial_parameter_sha256=initial['parameter_sha256'],
        initial_checkpoint_sha256=initial['checkpoint_sha256'])
    for field, value in references.items():
        if record.get(field) != value:
            errors.append('stage_' + field)
    if record.get('actual_initial_identity_verified') is not True:
        errors.append('initial_identity_not_verified')
    if record.get('source_model_run') is not True:
        errors.append('engineering_override_cannot_count_as_source_model')
    if record.get('test_evaluated') is not False or record.get('geometry_evaluated') is not False:
        errors.append('stage_scope_flags_invalid')
    if record.get('formal_training_started') is not False:
        errors.append('stage_formal_training_flag_invalid')
    final = int(config['training']['updates'])
    step = record.get('step')
    if not isinstance(step, int) or not 0 <= step <= final:
        errors.append('stage_step_invalid')
    elif (record.get('fixed_endpoint_step') != final or
          record.get('fixed_endpoint_completed') != (step == final and record.get('validation') is not None)):
        errors.append('stage_fixed_endpoint_metadata_invalid')
    if isinstance(step, int):
        if ('fixed_endpoint_updates_completed' in record and
                record['fixed_endpoint_updates_completed'] != (step == final)):
            errors.append('stage_endpoint_update_completion_invalid')
        exposures = step * int(config['training']['effective_batch'])
        if record.get('exposures') != exposures or record.get('supervised_tokens') != 2 * exposures:
            errors.append('stage_exposure_count_invalid')
    if record.get('parameters') != initial['parameters']:
        errors.append('stage_parameter_count_invalid')
    for field in ('checkpoint_sha256', 'starting_rng_hash'):
        value = record.get(field)
        if not isinstance(value, str) or len(value) != 64:
            errors.append('stage_missing_' + field)
    if not record.get('checkpoint_path'):
        errors.append('stage_missing_checkpoint_reference')
    return errors


def histories(stages, config):
    """Use every saved validation point, never choose a best checkpoint."""
    points = {}
    for stage in sorted(stages, key=lambda r: r['step']):
        candidates = list(stage.get('curve', []))
        if stage.get('validation') is not None:
            candidates.append(dict(step=stage['step'], validation=stage['validation']))
        for point in candidates:
            if point.get('validation') is None:
                continue
            assert 0 <= point['step'] <= stage['step'], 'A curve contains a future validation point'
            errors = validate_measurement(point['validation'], config)
            assert not errors, ('Invalid saved validation curve', errors)
            if point['step'] in points:
                old = points[point['step']]['validation']
                new = point['validation']
                for field in ('overall_full_vocab', 'overall_eos', 'answer_ce', 'combined_ce'):
                    assert abs(old[field] - new[field]) < 1e-6, 'A frozen validation point changed'
            previous = points.get(point['step'], {})
            points[point['step']] = {**previous, **point}
    return [points[step] for step in sorted(points)]


def union_seconds(intervals):
    intervals = sorted((float(a), float(b)) for a, b in intervals if b >= a)
    if not intervals:
        return 0.
    total, start, end = 0., *intervals[0]
    for a, b in intervals[1:]:
        if a > end:
            total += end - start
            start, end = a, b
        else:
            end = max(end, b)
    return total + end - start


def collect(root, output, *, fixture=False):
    config = read_json(root / 'protocol.json')
    assert config['tasks'] == list(DEVELOPMENT_TASKS)
    assert config['encodings'] == list(ENCODINGS) and config['architectures'] == list(ARCHITECTURES)
    assert config['lengths'] == [10, 11, 12] and config['training']['updates'] == 8000
    assert config['scope']['geometry_used_for_selection'] is False
    assert config['scope']['test_used_for_selection'] is False
    assert config['development_choice']['candidate_count'] == 1
    inputs = {}
    def register(path):
        inputs[str(path)] = sha(path)
        return read_json(path)
    protocol_digest = sha(root / 'protocol.json')
    inputs[str(root / 'protocol.json')] = protocol_digest
    frozen = register(output / 'freeze.json')
    assert frozen['protocol_sha256'] == protocol_digest
    assert sha(output / 'protocol.json') == protocol_digest
    inputs[str(output / 'protocol.json')] = protocol_digest
    for relative, digest in frozen['source_sha256'].items():
        assert sha(WORKSPACE / relative) == digest, ('Frozen source changed', relative)
    own_relative = str(Path(__file__).resolve().relative_to(WORKSPACE))
    assert own_relative in frozen['source_sha256'], 'Reporter was not frozen before results'
    source_version = hashlib.sha256(json.dumps(frozen['source_sha256'], sort_keys=True,
                                               separators=(',', ':')).encode()).hexdigest()
    manifest = config['training_manifest']
    assert register(root / 'training_manifest.json') == manifest
    assert len(manifest) == 16 and len({job['run_id'] for job in manifest}) == 16
    assert {(j['architecture'], j['encoding'], j['task']) for j in manifest} == {
        (a, e, t) for a in ARCHITECTURES for e in ENCODINGS for t in DEVELOPMENT_TASKS}
    world = root / 'worlds' / config['units'][0]['unit']
    data_audit = register(world / 'data_audit.json')
    data_path = world / 'data.npz'
    assert sha(data_path) == data_audit['data_sha256'], 'Learning data changed'
    inputs[str(data_path)] = data_audit['data_sha256']
    baseline, baseline_rows = baseline_from_labels(data_path, config)
    expected = dict(protocol_sha256=protocol_digest, source_version_sha256=source_version,
                    data_sha256=data_audit['data_sha256'])
    initial = {a: register(root / 'initializations' / a / 'step0.json') for a in ARCHITECTURES}
    for a, meta in initial.items():
        assert meta['architecture'] == a and meta['protocol_sha256'] == protocol_digest
        assert meta['actual_initial_identity_verified'] is True
        assert meta['model_seed'] == config['architecture_model_seeds'][a]
    items, length_rows, curve_rows, summary_rows, intervals = [], [], [], [], []
    for job in manifest:
        directory = root / 'models' / job['run_id']
        stages, errors = [], []
        for path in sorted(directory.glob('stage_*.json')):
            stage = register(path)
            assert path.stem == f'stage_{stage["step"]}', 'Stage filename is not its actual saved step'
            errors.extend(audit_stage(stage, job, config, expected, initial[job['architecture']]))
            stages.append(stage)
            if 'invocation_start_epoch' in stage and 'invocation_end_epoch' in stage:
                intervals.append((stage['invocation_start_epoch'], stage['invocation_end_epoch']))
        stages.sort(key=lambda row: row['step'])
        stage = stages[-1] if stages else None
        completed_path = directory / 'completed.json'
        completed = register(completed_path) if completed_path.exists() else None
        if completed:
            errors.extend(audit_stage(completed, job, config, expected, initial[job['architecture']]))
            if stage is None or stage['step'] != 8000:
                errors.append('completed_marker_without_stage_8000')
            else:
                for field in ('checkpoint_sha256', 'initial_checkpoint_sha256', 'data_sha256',
                              'execution_source_version_sha256', 'step', 'validation'):
                    if completed.get(field) != stage.get(field):
                        errors.append('completed_marker_mismatch_' + field)
        gates = model_gates(stage, completed, baseline[job['task']], config, errors)
        history = histories(stages, config)
        item = dict(job=job, stage=stage, completed=completed, gates=gates,
                    history=history, audit_errors=sorted(set(errors)))
        items.append(item)
        for point in history:
            validation = point['validation']
            curve_rows.append(dict(run_id=job['run_id'], architecture=job['architecture'],
                encoding=job['encoding'], task=job['task'], step=point['step'],
                equal_length_accuracy=equal_mean(validation['per_length'], 'full_vocab_accuracy'),
                equal_length_eos_accuracy=equal_mean(validation['per_length'], 'eos_accuracy'),
                equal_length_answer_ce=equal_mean(validation['per_length'], 'answer_ce'),
                equal_length_combined_ce=equal_mean(validation['per_length'], 'combined_ce'),
                equal_length_baseline=float(np.mean(list(baseline[job['task']].values()))),
                learning_rate=point.get('learning_rate'),
                train_loss_last_update=point.get('train_loss_last_update')))
            for row in validation['per_length']:
                length_rows.append(dict(run_id=job['run_id'], architecture=job['architecture'],
                    encoding=job['encoding'], task=job['task'], step=point['step'], **row,
                    length_baseline_accuracy=baseline[job['task']][row['length']],
                    gain_over_length_baseline_pp=100 * (row['full_vocab_accuracy'] - baseline[job['task']][row['length']]),
                    split='validation', action='identity'))
        summary_rows.append(dict(**job, actual_step=stage['step'] if stage else 0,
            fixed_endpoint_updates_completed=bool(stage and stage['step'] == config['training']['updates']),
            source_status=stage['status'] if stage else 'not_started_or_no_saved_stage',
            parameters=initial[job['architecture']]['parameters'],
            exposures=stage['exposures'] if stage else 0,
            supervised_tokens=stage['supervised_tokens'] if stage else 0,
            elapsed_process_wall_seconds=stage.get('elapsed_seconds', 0) if stage else 0,
            finite_gradient_updates=stage.get('finite_gradient_updates') if stage else None,
            minimum_gradient_norm=stage.get('minimum_gradient_norm') if stage else None,
            maximum_gradient_norm=stage.get('maximum_gradient_norm') if stage else None,
            validation_missing_due_to_budget=stage.get('validation_missing_due_to_budget', False) if stage else False,
            **{k: v for k, v in gates.items() if k != 'failures'},
            failures=';'.join(gates['failures']), audit_errors=';'.join(item['audit_errors'])))
    lookup = {(i['job']['architecture'], i['job']['task'], i['job']['encoding']): i for i in items}
    pairs = []
    for architecture in ARCHITECTURES:
        for task in DEVELOPMENT_TASKS:
            result = pair_gates(*(lookup[(architecture, task, e)] for e in ENCODINGS), config)
            pairs.append(dict(architecture=architecture, task=task,
                first_run_id=lookup[(architecture, task, ENCODINGS[0])]['job']['run_id'],
                second_run_id=lookup[(architecture, task, ENCODINGS[1])]['job']['run_id'],
                **{k: v for k, v in result.items() if k not in ('failures', 'per_length_gaps_pp')},
                failures=';'.join(result['failures']),
                **{f'gap_n{n}_pp': result['per_length_gaps_pp'].get(n) for n in config['lengths']}))
    acceptance = {a: all(i['gates']['gate_pass'] for i in items if i['job']['architecture'] == a)
                  and all(p['gate_pass'] for p in pairs if p['architecture'] == a) for a in ARCHITECTURES}
    execution = register(root / 'development_execution.json') if (root / 'development_execution.json').exists() else None
    budget = register(root / 'gpu_budget.json') if (root / 'gpu_budget.json').exists() else None
    for record, label in ((execution, 'execution'), (budget, 'budget')):
        if record is not None:
            assert record['protocol_sha256'] == protocol_digest, label + ' protocol mismatch'
    cost = dict(
        fixed_manifest_models=16, fixed_steps_per_model=8000,
        actual_updates=sum(row['actual_step'] for row in summary_rows),
        actual_anchor_exposures=sum(row['exposures'] for row in summary_rows),
        actual_supervised_answer_plus_eos_tokens=sum(row['supervised_tokens'] for row in summary_rows),
        sum_model_process_wall_seconds=sum(row['elapsed_process_wall_seconds'] for row in summary_rows),
        source_invocation_interval_union_seconds=union_seconds(intervals),
        interval_union_scope='Saved source invocation metadata only; excludes engineering, gaps, and unsaved crashed invocations.',
        sum_process_wall_scope='Per-model cumulative process wall time; overlapping execution is counted repeatedly. This is not GPU utilization hours or project elapsed time.',
        budget_limit_seconds=config['budget']['active_gpu_wall_seconds'],
        finalization_reserve_seconds=config['budget']['finalization_reserve_seconds'],
        orchestrator_active_wall_seconds=execution.get('training_wall_seconds') if execution else None,
        orchestrator_wall_scope='From persistent GPU-budget start through orchestrator finish, including engineering and validation/save overhead; not measured GPU utilization.',
        budget_deadline_epoch=budget.get('deadline_epoch') if budget else None,
        update_cutoff_epoch=budget.get('stop_updates_epoch') if budget else None,
        budget_reset=False, additional_budget_authorized=False,
        architecture_parameters={a: initial[a]['parameters'] for a in ARCHITECTURES})
    return dict(config=config, input_sha256=inputs, source_version_sha256=source_version,
        protocol_sha256=protocol_digest, items=items, length_rows=length_rows,
        curve_rows=curve_rows, summary_rows=summary_rows, baseline_rows=baseline_rows,
        pair_rows=pairs, architecture_accepted=acceptance, overall_16_accepted=all(acceptance.values()),
        execution=execution, budget=budget, cost=cost, fixture=fixture)


def plots(result, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    directory = output / 'learnability_plots'
    directory.mkdir(exist_ok=True)
    colors = dict(one_line='#2366a8', canonical_cycle='#db7226')
    styles = {10: '-', 11: '--', 12: ':'}
    specs = [
        ('equal_length_accuracy', 'Equal-length full-vocabulary answer accuracy', 'full_vocab_accuracy', False),
        ('per_length_accuracy', 'Full-vocabulary answer accuracy by length', 'full_vocab_accuracy', True),
        ('equal_length_answer_ce', 'Equal-length answer cross-entropy', 'answer_ce', False),
        ('per_length_answer_ce', 'Answer cross-entropy by length', 'answer_ce', True),
        ('equal_length_combined_ce', 'Equal-length answer-plus-EOS cross-entropy', 'combined_ce', False),
        ('per_length_combined_ce', 'Answer-plus-EOS cross-entropy by length', 'combined_ce', True),
        ('equal_length_eos_accuracy', 'Teacher-forced EOS accuracy', 'eos_accuracy', False),
    ]
    names = []
    for name, title, field, per_length in specs:
        figure, axes = plt.subplots(2, 4, figsize=(16, 7), squeeze=False, sharex=True)
        for row, architecture in enumerate(ARCHITECTURES):
            for col, task in enumerate(DEVELOPMENT_TASKS):
                axis = axes[row, col]
                for item in result['items']:
                    job = item['job']
                    if job['architecture'] != architecture or job['task'] != task:
                        continue
                    points = item['history']
                    x = [p['step'] for p in points]
                    if per_length:
                        for n in result['config']['lengths']:
                            y = [next(r[field] for r in p['validation']['per_length'] if r['length'] == n) for p in points]
                            axis.plot(x, y, styles[n], color=colors[job['encoding']], linewidth=1.2,
                                      label=f'{job["encoding"]}, n={n}')
                    else:
                        y = [equal_mean(p['validation']['per_length'], field) for p in points]
                        axis.plot(x, y, color=colors[job['encoding']], label=job['encoding'], linewidth=1.5)
                if field == 'full_vocab_accuracy':
                    if per_length:
                        axis.axhline(result['config']['learning_gate']['per_length_accuracy_min'], color='#666', linestyle='-.', linewidth=.8)
                        for n in result['config']['lengths']:
                            base = next(r['validation_baseline_accuracy'] for r in result['baseline_rows'] if r['task'] == task and r['length'] == n)
                            axis.axhline(base, color='#999', linestyle=styles[n], linewidth=.6)
                    else:
                        axis.axhline(result['config']['learning_gate']['per_model_equal_length_accuracy_min'], color='#666', linestyle='-.', linewidth=.8)
                        base = next(i['gates']['equal_length_baseline'] for i in result['items'] if i['job']['task'] == task)
                        axis.axhline(base, color='#999', linestyle=':', linewidth=.8, label='Train-only length mode')
                if field == 'eos_accuracy':
                    axis.axhline(result['config']['learning_gate']['per_model_eos_accuracy_min'], color='#666', linestyle='-.', linewidth=.8)
                if field in ACCURACY_FIELDS:
                    axis.set_ylim(0, 1.02)
                axis.set_title(architecture + ' / ' + task, fontsize=10)
                axis.set_xlim(0, 8000)
                axis.grid(alpha=.2)
                axis.set_xlabel('Actual update')
                if col == 0:
                    axis.set_ylabel('Accuracy' if field in ACCURACY_FIELDS else 'Cross-entropy')
                axis.legend(fontsize=6, loc='best')
        figure.suptitle(title + ' (validation identity only; all fixed models)')
        figure.tight_layout(rect=(0, 0, 1, .95))
        figure.savefig(directory / (name + '.png'), dpi=130)
        plt.close(figure)
        names.append('learnability_plots/' + name + '.png')
    figure, axes = plt.subplots(2, 4, figsize=(16, 7), squeeze=False, sharex=True)
    lookup = {(i['job']['architecture'], i['job']['task'], i['job']['encoding']): i for i in result['items']}
    for row, architecture in enumerate(ARCHITECTURES):
        for col, task in enumerate(DEVELOPMENT_TASKS):
            axis = axes[row, col]
            histories_by_encoding = [{p['step']: p['validation'] for p in lookup[(architecture, task, e)]['history']} for e in ENCODINGS]
            steps = sorted(set(histories_by_encoding[0]) & set(histories_by_encoding[1]))
            for n in result['config']['lengths']:
                gaps = [100 * abs(next(r['full_vocab_accuracy'] for r in histories_by_encoding[0][step]['per_length'] if r['length'] == n) -
                                  next(r['full_vocab_accuracy'] for r in histories_by_encoding[1][step]['per_length'] if r['length'] == n)) for step in steps]
                axis.plot(steps, gaps, styles[n], label=f'n={n}', linewidth=1.2)
            means = [100 * abs(equal_mean(histories_by_encoding[0][step]['per_length'], 'full_vocab_accuracy') -
                               equal_mean(histories_by_encoding[1][step]['per_length'], 'full_vocab_accuracy')) for step in steps]
            axis.plot(steps, means, color='black', label='Equal-length mean', linewidth=1.5)
            axis.axhline(result['config']['learning_gate']['paired_encoding_accuracy_gap_max_pp'], color='#666', linestyle='-.', linewidth=.8)
            axis.set_title(architecture + ' / ' + task, fontsize=10)
            axis.set_xlim(0, 8000)
            axis.grid(alpha=.2)
            axis.set_xlabel('Same actual update')
            if col == 0:
                axis.set_ylabel('Absolute encoding gap (pp)')
            axis.legend(fontsize=7)
    figure.suptitle('Encoding gaps at common saved update counts; descriptive before fixed 8k')
    figure.tight_layout(rect=(0, 0, 1, .95))
    figure.savefig(directory / 'encoding_validation_gaps.png', dpi=130)
    plt.close(figure)
    names.append('learnability_plots/encoding_validation_gaps.png')
    return names


def fmt(value, percentage=False):
    if value is None:
        return 'missing'
    if isinstance(value, bool):
        return 'yes' if value else 'no'
    if isinstance(value, (int, float)):
        return f'{100 * value:.2f}%' if percentage else f'{value:.3f}'
    return str(value)


def render(result, output):
    output.mkdir(parents=True, exist_ok=True)
    summary, pairs, length = result['summary_rows'], result['pair_rows'], result['length_rows']
    write_csv(output / 'model_learning_summary.csv', summary, list(summary[0]))
    write_csv(output / 'pair_learning_gates.csv', pairs, list(pairs[0]))
    write_csv(output / 'validation_length.csv', length, [
        'run_id', 'architecture', 'encoding', 'task', 'step', 'length', 'samples',
        *ACCURACY_FIELDS, 'answer_accuracy', *LOSS_FIELDS, 'length_baseline_accuracy',
        'gain_over_length_baseline_pp', 'split', 'action'])
    write_csv(output / 'validation_curves.csv', result['curve_rows'], [
        'run_id', 'architecture', 'encoding', 'task', 'step',
        'equal_length_accuracy', 'equal_length_eos_accuracy', 'equal_length_answer_ce',
        'equal_length_combined_ce', 'equal_length_baseline', 'learning_rate', 'train_loss_last_update'])
    write_csv(output / 'training_length_majority.csv', result['baseline_rows'], list(result['baseline_rows'][0]))
    images = plots(result, output)
    completed = sum(row['endpoint_completed'] and row['validation_complete'] for row in summary)
    errors = {i['job']['run_id']: i['audit_errors'] for i in result['items'] if i['audit_errors']}
    text = [
        'PermWorld bounded learning development: validation-only report',
        f'Generated UTC: {datetime.now(timezone.utc).isoformat()}',
        'Synthetic fixture only.' if result['fixture'] else 'Exploratory development; one preset, one new world, one initializer per architecture.',
        f'Complete fixed-8k models with complete validation: {completed}/16.',
        f'Models with all 8000 updates, regardless of missing final validation: {sum(row["fixed_endpoint_updates_completed"] for row in summary)}/16.',
        'Overall 16-model learning acceptance: ' + ('PASS' if result['overall_16_accepted'] else 'FAIL / INCOMPLETE'),
        'Architecture joint acceptance: ' + ', '.join(f'{a}={"PASS" if ok else "FAIL / INCOMPLETE"}' for a, ok in result['architecture_accepted'].items()),
        '',
        'Only the fixed 8000-update endpoint can qualify. Partial 2k/4k/actual-budget-stop results remain visible; earlier maxima do not replace the endpoint.',
        'Each architecture requires all four tasks, both encodings, and all paired encoding gates. A partial subset is not a completed development unit.',
        'Primary accuracy is the full-vocabulary answer argmax on identity validation; numeric accuracy is additional description. EOS is teacher-forced at the true answer token, not free generation.',
        'The training-length majority baseline is fixed from identity training labels only, with the smallest numeric label breaking ties, then scored on validation.',
        'All three lengths have fixed equal weights of 1/3. Paired encoding comparisons require equal actual update counts; unequal-budget partial runs do not estimate an encoding effect.',
        'Loss curves show all saved validation points and last-update training loss in CSV; no best-checkpoint selection or additional loss threshold.',
        '', 'Fixed learning thresholds: ' + json.dumps(result['config']['learning_gate'], sort_keys=True),
        '', 'Model endpoints (accuracy/baseline/EOS in %, gaps in percentage points):',
    ]
    for row in summary:
        text.append(f'{row["run_id"]}: step={row["actual_step"]}, answer={fmt(row["equal_length_accuracy"], True)}, '
            f'min-length={fmt(row["minimum_length_accuracy"], True)}, baseline={fmt(row["equal_length_baseline"], True)}, '
            f'gain={fmt(row["gain_over_length_baseline_pp"])}pp, EOS={fmt(row["equal_length_eos_accuracy"], True)}, '
            f'answer-CE={fmt(row["equal_length_answer_ce"])}, combined-CE={fmt(row["equal_length_combined_ce"])}, '
            f'gate={"PASS" if row["gate_pass"] else "FAIL"}, failures={row["failures"] or "none"}.')
    text.extend(['', 'Paired encoding gates (fixed equal-budget 8k only):'])
    for row in pairs:
        text.append(f'{row["architecture"]}/{row["task"]}: mean-gap={fmt(row["mean_accuracy_gap_pp"])}pp, '
                    f'max-length-gap={fmt(row["maximum_length_accuracy_gap_pp"])}pp, '
                    f'gate={"PASS" if row["gate_pass"] else "FAIL"}; {row["failures"] or "no failures"}.')
    text.extend([
        '', 'Budget and scope:', json.dumps(result['cost'], indent=2, sort_keys=True),
        'Summed model process wall time is not actual GPU hours or project elapsed time. The persistent orchestrator wall-clock budget includes engineering, validation, and save overhead; GPU utilization was not measured.',
        'One exploratory data world and one initializer per architecture do not provide independent world replication, statistical power, confirmation p-values, or an equivalence claim.',
        'Transformer and GRU have different parameter counts and different initial parameters. Within each architecture the two encodings and four tasks share an exact saved initializer.',
        'The one development preset changes six training factors together. Improved learning cannot be attributed to a single factor or to mathematical geometry.',
        'No test arrays, CKA, Delta, hidden activations, probabilities, or checkpoint tensors were read. Old artifacts and F17 were not modified; no additional candidate or training was started by this reporter.',
        'Passing supports only readiness for an independently frozen learning/geometry verification. Failing identifies the learning/budget gate; no task, length, or checkpoint is silently dropped.',
        'The budget may leave a partial manifest due to its fixed job order. Such an outcome is incomplete, rather than evidence for a favorable architecture or encoding.',
        '', 'Metadata audit errors: ' + json.dumps(errors, sort_keys=True),
        'Independent-verification proposal: ' + ('generated; does not execute anything' if result['overall_16_accepted'] else 'not generated because the complete 16-model unit did not qualify'),
    ])
    (output / 'learnability_report.txt').write_text('\n'.join(text) + '\n')
    cards = ''.join(f'<figure><img src="{html.escape(name)}" alt="{html.escape(Path(name).stem)}"><figcaption>{html.escape(Path(name).stem)}</figcaption></figure>' for name in images)
    (output / 'learnability_report.html').write_text(
        '<!doctype html><meta charset="utf-8"><title>PermWorld validation learning development</title>'
        '<style>body{max-width:1250px;margin:30px auto;padding:0 20px;font:15px/1.5 system-ui}pre{white-space:pre-wrap;overflow-wrap:anywhere}img{width:100%;height:auto}figure{margin:30px 0}a{color:#2469a1}</style>'
        '<h1>PermWorld validation learning development</h1>'
        '<p><a href="model_learning_summary.csv">All 16 endpoints</a> · <a href="validation_length.csv">Every saved length</a> · '
        '<a href="pair_learning_gates.csv">All 8 encoding pairs</a> · <a href="validation_curves.csv">Learning curves</a> · '
        '<a href="training_length_majority.csv">Train-only baseline</a> · <a href="report_audit.json">Audit</a></p>'
        '<pre>' + html.escape('\n'.join(text)) + '</pre>' + cards)
    proposal_path = output / 'independent_verification_proposal.json'
    if result['overall_16_accepted']:
        write_json(proposal_path, dict(status='proposal_only_no_execution',
            protocol_sha256=result['protocol_sha256'], source_version_sha256=result['source_version_sha256'],
            accepted_development_manifest=result['config']['training_manifest'],
            accepted_training=result['config']['training'], accepted_lengths=result['config']['lengths'],
            requirements=[
                'Freeze independent new data-world and model-seed manifest before outcomes.',
                'Save true step-0 parameters and random states, clone the two encodings within architecture.',
                'Retain all four tasks and both encodings in each architecture; preserve full-vocabulary answer and EOS supervision.',
                'Verify historical orbit exclusion and train/validation/test orbit separation.',
                'Freeze source-learning and representation evaluation before independent results; define replication precision separately.',
                'Do not use this single-world development to claim confirmation, power, equivalence, or a causal architecture effect.',
                'Independent execution requires a separate bounded instruction; this file launches nothing.',
            ], test_read=False, geometry_used_for_selection=False, training_started=False))
    elif proposal_path.exists():
        # Refuse to leave a stale passing proposal after rerendering a partial unit.
        proposal_path.unlink()
    audit = dict(status='passed' if not errors else 'metadata_failed',
        generated_utc=datetime.now(timezone.utc).isoformat(), synthetic_fixture=result['fixture'],
        protocol_sha256=result['protocol_sha256'], source_version_sha256=result['source_version_sha256'],
        reporter_sha256=sha(__file__), input_sha256=result['input_sha256'],
        fixed_manifest_models=16, reported_manifest_models=len(summary), paired_conditions=len(pairs),
        completed_fixed_endpoint_with_validation=completed, model_gate_pass_count=sum(r['gate_pass'] for r in summary),
        fixed_endpoint_updates_completed_count=sum(r['fixed_endpoint_updates_completed'] for r in summary),
        architecture_accepted=result['architecture_accepted'], overall_16_accepted=result['overall_16_accepted'],
        actual_saved_validation_rows=len(length), figures=len(images),
        scope=dict(decoded_npz_arrays=['train_lengths', 'train_labels', 'validation_lengths', 'validation_labels'],
            test_decoded=False, representation_metrics_read=False, checkpoint_tensors_loaded=False,
            gpu_used=False, training_started=False, old_artifacts_modified=False, F17_modified=False,
            source_count_uses_fixed_manifest_only=True, engineering_override_excluded=True,
            partial_endpoint_cannot_qualify=True, no_best_checkpoint_selection=True),
        metadata_errors=errors, cost=result['cost'],
        output_sha256={str(path.relative_to(output)): sha(path) for path in output.glob('*.csv') if path.name in (
            'model_learning_summary.csv', 'pair_learning_gates.csv', 'validation_length.csv',
            'validation_curves.csv', 'training_length_majority.csv')},
        figure_sha256={name: sha(output / name) for name in images})
    write_json(output / 'report_audit.json', audit)
    return audit


def report():
    """Read the real frozen development only when explicitly called by its driver."""
    return render(collect(ROOT, DIAG), DIAG)


def fixture():
    """CPU synthetic tests; no real new source results or held-out test arrays."""
    checks = {}
    config = dict(tasks=list(DEVELOPMENT_TASKS), encodings=list(ENCODINGS),
        architectures=list(ARCHITECTURES), lengths=[10, 11, 12],
        units=[dict(unit='fixture')], training=dict(updates=8000, effective_batch=64),
        architecture_model_seeds=dict(transformer=101, gru=102),
        scope=dict(geometry_used_for_selection=False, test_used_for_selection=False),
        development_choice=dict(candidate_count=1),
        data=dict(examples_per_length=dict(train=100, validation=100)),
        learning_gate=dict(per_model_equal_length_accuracy_min=.9, per_length_accuracy_min=.85,
            per_model_gain_over_training_length_majority_min_pp=20.,
            paired_encoding_accuracy_gap_max_pp=5., paired_encoding_per_length_accuracy_gap_max_pp=5.,
            per_model_eos_accuracy_min=.99),
        budget=dict(active_gpu_wall_seconds=7200, finalization_reserve_seconds=30))
    config['training_manifest'] = [dict(run_id=f'fixture_{a}_{e}_{t}', unit='fixture',
        architecture=a, encoding=e, task=t, model_seed=config['architecture_model_seeds'][a],
        data_seed=103, batch_seed=104, updates=8000)
        for a in ARCHITECTURES for t in DEVELOPMENT_TASKS for e in ENCODINGS]
    def validation(values=(.95, .95, .95), eos=1.):
        rows = [dict(length=n, samples=100, full_vocab_accuracy=v, answer_accuracy=v,
                     numeric_accuracy=v, eos_accuracy=eos, answer_ce=.2, eos_ce=.01,
                     combined_ce=.105) for n, v in zip(config['lengths'], values)]
        return dict(split='validation', action='identity', samples=300, per_length=rows,
            overall_full_vocab=equal_mean(rows, 'full_vocab_accuracy'),
            overall_numeric=equal_mean(rows, 'numeric_accuracy'), overall_eos=eos,
            answer_ce=.2, eos_ce=.01, combined_ce=.105,
            eos_teacher_forced=True, test_evaluated=False, geometry_evaluated=False)
    def stage(values=(.95, .95, .95), step=8000, eos=1.):
        return dict(step=step, fixed_endpoint_completed=step == 8000,
                    source_model_run=True, validation=validation(values, eos))
    completed = dict(step=8000, status='completed', source_model_run=True)
    baseline = dict.fromkeys(config['lengths'], .6)
    def test(name, condition):
        assert condition, 'Synthetic report fixture failed: ' + name
        checks[name] = True
    test('complete_and_above_baseline_passes', model_gates(stage(), completed, baseline, config)['gate_pass'])
    test('equal_encodings_at_majority_fail', not model_gates(stage(), completed, dict.fromkeys(config['lengths'], .95), config)['gate_pass'])
    test('partial_2k_cannot_pass', not model_gates(stage(step=2000), None, baseline, config)['gate_pass'])
    test('minimum_length_gate_independent', not model_gates(stage((.99, .99, .84)), completed, baseline, config)['gate_pass'])
    test('teacher_forced_eos_gate', not model_gates(stage(eos=.98), completed, baseline, config)['gate_pass'])
    missing = stage(); missing['validation'] = None
    test('missing_budget_validation_cannot_pass', not model_gates(missing, completed, baseline, config)['gate_pass'])
    engineer = stage(); engineer['source_model_run'] = False
    test('engineering_override_cannot_qualify', not model_gates(engineer, completed, baseline, config)['gate_pass'])
    def item(s):
        return dict(stage=s, gates=model_gates(s, completed, baseline, config), audit_errors=[])
    equal_mean_different_lengths = pair_gates(item(stage((.99, .90, .90))), item(stage((.90, .99, .90))), config)
    test('equal_mean_does_not_bypass_length_gap', not equal_mean_different_lengths['gate_pass'] and equal_mean_different_lengths['mean_accuracy_gap_pp'] == 0)
    high_baseline = item(stage())
    high_baseline['gates'] = model_gates(stage(), completed, dict.fromkeys(config['lengths'], .95), config)
    baseline_pair = pair_gates(high_baseline, high_baseline, config)
    test('equally_high_majority_models_fail_joint_pair', not baseline_pair['gate_pass'] and
         baseline_pair['mean_accuracy_gap_pp'] == 0 and
         'paired_model_learning_gate_failed' in baseline_pair['failures'])
    test('unequal_update_counts_cannot_be_encoding_effect', not pair_gates(item(stage()), item(stage(step=4000)), config)['gate_pass'])
    test('interval_union_removes_overlap', union_seconds([(1, 3), (2, 5), (6, 7)]) == 5.)
    with tempfile.TemporaryDirectory(prefix='permworld-learning-report-fixture-') as temporary:
        directory = Path(temporary)
        source, output = directory / 'source', directory / 'diagnostic'
        source.mkdir(); output.mkdir()
        write_json(source / 'protocol.json', config); write_json(output / 'protocol.json', config)
        hashes = {str(Path(__file__).resolve().relative_to(WORKSPACE)): sha(__file__)}
        digest = sha(source / 'protocol.json')
        version = hashlib.sha256(json.dumps(hashes, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        write_json(output / 'freeze.json', dict(protocol_sha256=digest, source_sha256=hashes))
        write_json(source / 'training_manifest.json', config['training_manifest'])
        world = source / 'worlds' / 'fixture'; world.mkdir(parents=True)
        lengths = np.repeat(config['lengths'], 100)
        labels = np.zeros((300, 3, 16), dtype=np.int16)
        for i in range(3):
            labels[i * 100 + 60:(i + 1) * 100, :, :] = 1
        np.savez(world / 'data.npz', train_lengths=lengths, train_labels=labels[:, :1, :],
                 validation_lengths=lengths, validation_labels=labels,
                 test_guard=np.array(['report must not access this array']))
        data_digest = sha(world / 'data.npz')
        write_json(world / 'data_audit.json', dict(data_sha256=data_digest))
        for a in ARCHITECTURES:
            init = source / 'initializations' / a; init.mkdir(parents=True)
            write_json(init / 'step0.json', dict(architecture=a, protocol_sha256=digest,
                actual_initial_identity_verified=True, model_seed=config['architecture_model_seeds'][a],
                parameters=3240448 if a == 'transformer' else 1660416,
                parameter_sha256=('1' if a == 'transformer' else '2') * 64,
                checkpoint_sha256=('3' if a == 'transformer' else '4') * 64))
        for job in config['training_manifest']:
            target = source / 'models' / job['run_id']; target.mkdir(parents=True)
            init = read_json(source / 'initializations' / job['architecture'] / 'step0.json')
            record = dict(**job, **stage(), status='stage_complete', fixed_endpoint_step=8000,
                parameters=init['parameters'], exposures=512000, supervised_tokens=1024000,
                protocol_sha256=digest, execution_source_version_sha256=version, data_sha256=data_digest,
                initial_parameter_sha256=init['parameter_sha256'], initial_checkpoint_sha256=init['checkpoint_sha256'],
                actual_initial_identity_verified=True, starting_rng_hash='5' * 64,
                checkpoint_sha256='6' * 64, checkpoint_path=str(target / 'not_loaded.pt'),
                test_evaluated=False, geometry_evaluated=False, formal_training_started=False,
                curve=[dict(step=2000, validation=validation()), dict(step=4000, validation=validation())],
                elapsed_seconds=10., invocation_start_epoch=100., invocation_end_epoch=110.)
            write_json(target / 'stage_8000.json', record)
            write_json(target / 'completed.json', dict(record, status='completed'))
        result = collect(source, output, fixture=True)
        audit = render(result, output)
        test('synthetic_full_manifest_reports_all_16', audit['reported_manifest_models'] == 16)
        test('synthetic_full_manifest_reports_all_8_pairs', audit['paired_conditions'] == 8)
        test('synthetic_complete_unit_passes', audit['overall_16_accepted'])
        test('eight_fixed_learning_plots', audit['figures'] == 8)
        test('saved_three_lengths_all_models_all_points', audit['actual_saved_validation_rows'] == 16 * 3 * 3)
        test('no_model_checkpoint_was_required', not list(source.rglob('*.pt')) and audit['scope']['checkpoint_tensors_loaded'] is False)
        test('proposal_is_only_written_after_joint_pass', (output / 'independent_verification_proposal.json').is_file())
        weak = copy.deepcopy(result)
        weak['items'][0]['gates']['gate_pass'] = False
        accepted = all(i['gates']['gate_pass'] for i in weak['items'] if i['job']['architecture'] == 'transformer')
        test('one_failed_task_prevents_architecture_acceptance', not accepted)
        test('cost_does_not_call_summed_process_time_gpu_hours', audit['cost']['sum_model_process_wall_seconds'] == 160 and
             audit['cost']['source_invocation_interval_union_seconds'] == 10 and
             'GPU utilization' in audit['cost']['sum_process_wall_scope'])
    DIAG.mkdir(parents=True, exist_ok=True)
    answer = dict(status='passed', generated_utc=datetime.now(timezone.utc).isoformat(),
        reporter_sha256=sha(__file__), checks=checks, synthetic_cpu_only=True,
        real_new_development_results_read=False, test_arrays_decoded=False,
        checkpoint_tensors_loaded=False, GPU_used=False, training_started=False,
        fixtures_removed=True)
    write_json(DIAG / 'report_fixture.json', answer)
    return answer


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fixture', action='store_true', required=True,
                        help='Run synthetic CPU report tests only; real report is an explicit driver stage.')
    args = parser.parse_args()
    print(json.dumps(fixture(), indent=2, sort_keys=True))
