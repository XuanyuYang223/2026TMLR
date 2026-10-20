"""Supplementary read-only arithmetic audit of every saved validation point.

This adds curve-table checks requested after the original endpoint auditor was
fixed; it does not edit that auditor or select any model, update, or metric.
"""
import copy
from datetime import datetime, timezone
import json
import math
from pathlib import Path

import independent_learning_audit as audit


def run():
    config, frozen, digest, task_names = audit.protocol_and_schema(audit.ROOT, audit.DIAG)
    audit.preflight(audit.ROOT, audit.DIAG, config)
    curve_csv = audit.DIAG/'validation_curves.csv'
    audit.require(curve_csv.is_file(), 'Curve CSV has not been generated')
    data_path = audit.ROOT/'worlds'/config['units'][0]['unit']/'data.npz'
    data, decoded = audit.labels_only(data_path)
    baseline, _ = audit.independent_baselines(data, config, task_names)
    curves, lengths, stages = [], [], []
    settings = config['training']
    saved_steps = list(range(settings['validation_every'], settings['updates']+1,
                             settings['validation_every']))
    for job in config['training_manifest']:
        path = audit.ROOT/'models'/job['run_id']/'stage_8000.json'
        stage = audit.read_json(path)
        points = stage['curve']
        audit.require([p['step'] for p in points] == saved_steps,
                      job['run_id']+': missing, repeated, or selected saved validation point')
        audit.require(points[-1]['validation'] == stage['validation'],
                      job['run_id']+': final curve/endpoint validation differ')
        stages.append(path)
        for point in points:
            require_step = int(point['step'])
            sample = copy.copy(stage)
            sample['validation'] = point['validation']
            values, rows = audit.model_values(sample, baseline, config)
            warmup = settings['warmup_updates']
            factor = require_step/warmup if require_step <= warmup else (
                settings['minimum_lr_ratio'] + (1-settings['minimum_lr_ratio'])*
                (1+math.cos(math.pi*(require_step-warmup)/(settings['updates']-warmup)))/2)
            expected_lr = settings['learning_rate']*factor
            audit.close(point['learning_rate'], expected_lr, job['run_id']+': global LR schedule')
            audit.require(math.isfinite(point['train_loss_last_update']) and
                          point['train_loss_last_update'] >= 0, job['run_id']+': last-update loss')
            curves.append(dict(run_id=job['run_id'], architecture=job['architecture'],
                encoding=job['encoding'], task=job['task'], step=require_step,
                equal_length_accuracy=values['equal_length_accuracy'],
                equal_length_eos_accuracy=values['equal_length_eos_accuracy'],
                equal_length_answer_ce=values['equal_length_answer_ce'],
                equal_length_combined_ce=values['equal_length_combined_ce'],
                equal_length_baseline=values['equal_length_baseline'],
                learning_rate=point['learning_rate'],
                train_loss_last_update=point['train_loss_last_update']))
            for row in rows:
                row['step'] = require_step
                lengths.append(row)
    comparisons = dict(
        validation_curves=audit.table_compare(curve_csv, curves, ('run_id', 'step')),
        validation_length=audit.table_compare(audit.DIAG/'validation_length.csv', lengths,
                                              ('run_id', 'step', 'length')))
    result = dict(status='passed', generated_utc=datetime.now(timezone.utc).isoformat(),
        auditor_sha256=audit.sha(__file__), original_endpoint_auditor_sha256=audit.sha(audit.__file__),
        protocol_sha256=digest, source_version_sha256=audit.source_version(frozen),
        models=16, saved_steps=saved_steps, saved_curve_rows=len(curves),
        saved_length_rows=len(lengths), CSV_comparisons=comparisons,
        input_sha256={str(p): audit.sha(p) for p in stages+[data_path, curve_csv,
                                                       audit.DIAG/'validation_length.csv']},
        scope=dict(decoded_npz_arrays=decoded, test_arrays_decoded=False,
            checkpoint_tensors_loaded=False, GPU_used=False, training_started=False,
            representation_metrics_read=False, best_checkpoint_selection=False,
            earlier_points_do_not_replace_fixed8k_endpoint=True,
            accuracies_derived_from_saved_validation_records_not_new_predictions=True,
            supplementary_rule_is_table_arithmetic_not_new_hypothesis=True))
    audit.write_json(audit.DIAG/'independent_curve_audit.json', result)
    (audit.DIAG/'independent_curve_audit.txt').write_text(
        'Independent arithmetic review of all saved validation curves: passed.\n'
        f'16 models, each at fixed{saved_steps} update points;{len(curves)} equal-weight curve rows and{len(lengths)} per-length rows.\n'
        'All answer/NLL/EOS/training-length-mode aggregates match report CSVs; global learning rates also match the fixed 8k schedule.\n'
        'Final curve points match 8k endpoints. Earlier results cannot replace fixed 8k gates; no maximum-accuracy or minimum-loss selection.\n'
        'Read only train/validation labels and saved validation records; no test arrays, checkpoint tensors, or CKA were read; no inference or training.\n'
        'This supplementary check was implemented after report generation to verify every table’s arithmetic, not as a new confirmatory test.\n')
    return result


if __name__ == '__main__':
    result = run()
    print(json.dumps({k: result[k] for k in ('status', 'models', 'saved_steps',
                                           'saved_curve_rows', 'saved_length_rows')}))
