"""Independent CPU audit of full-set, fixed-checkpoint train/validation inference.

This file never imports torch, loads checkpoint tensors, launches inference or
training, or decodes a test/input array. --fixture is wholly synthetic;
--run-real requires all sixteen predictions and complete primary CSVs first.
Saved FP32 CE remains the primary measure. Float64 logsumexp is an independent
numerical check under the protocol’s predeclared per-example tolerance.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import csv
import hashlib
import json
import math
from pathlib import Path
import tempfile
import tomllib

import numpy as np


WORKSPACE = Path(__file__).resolve().parents[2]
DIAG = Path(__file__).resolve().parent
SPLITS = ('train', 'validation')
LENGTHS = (10, 11, 12)
METRICS = ('answer_accuracy', 'answer_ce', 'eos_accuracy', 'eos_ce', 'combined_ce')
ARCHITECTURES = ('transformer', 'gru')
ENCODINGS = ('one_line', 'canonical_cycle')
TASKS = ('descents', 'recoils', 'peaks', 'valleys')
ARRAY_NAMES = ('lengths', 'sample_index', 'gold_answer_token',
               'predicted_answer_token', 'answer_ce', 'eos_predicted_token',
               'eos_ce', 'answer_logits', 'eos_logits')
SOURCE_ARRAYS = ('train_lengths', 'train_labels', 'validation_lengths', 'validation_labels')
EXPECTED_PREDICTION_ARRAYS = {f'{split}_{name}' for split in SPLITS for name in ARRAY_NAMES}
ARITHMETIC_TOLERANCE = 5e-12  # Reporting-roundoff only, never a scientific margin.


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(2**20), b''):
            value.update(chunk)
    return value.hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text())


def finite(value, context):
    value = float(value)
    require(math.isfinite(value), context + ': nonfinite value')
    return value


def close(observed, expected, context):
    observed, expected = finite(observed, context), finite(expected, context)
    require(abs(observed-expected) <= ARITHMETIC_TOLERANCE,
            f'{context}: observed={observed!r}; expected={expected!r}')


def boolean(value):
    if isinstance(value, bool):
        return value
    require(value in ('True', 'False', 'true', 'false'), 'Invalid CSV boolean: ' + repr(value))
    return value in ('True', 'true')


def read_csv(path):
    with Path(path).open(newline='') as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows, fields):
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def json_write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def protocol_check(diag, synthetic=False):
    protocol_path, freeze_path = diag/'protocol.json', diag/'freeze.json'
    protocol, frozen = load_json(protocol_path), load_json(freeze_path)
    require(sha(protocol_path) == frozen['protocol_sha256'], 'Protocol no longer matches freeze')
    require(bool(frozen.get('training_forbidden')), 'Freeze does not forbid training')
    source_hashes = frozen['source_sha256']
    require(bool(source_hashes), 'Frozen source hash map is empty')
    for name, expected in source_hashes.items():
        path = Path(name) if Path(name).is_absolute() else WORKSPACE/name
        require(path.is_file() and sha(path) == expected, 'Frozen source mismatch: ' + str(path))
    require(protocol['lengths'] == list(LENGTHS), 'Length set changed')
    require(set(protocol['length_weights']) == {'10', '11', '12'}, 'Length weights incomplete')
    for n in LENGTHS:
        require(float(protocol['length_weights'][str(n)]) == 1/3, 'Length weight is not exactly 1/3')
    require(protocol['validation_reproduction_tolerances'] == {
        'accuracy_absolute': 0.0, 'ce_absolute': 1e-5, 'ce_relative': 1e-6},
        'Validation-reproduction tolerance differs from predeclared budget')
    require(protocol['independent_logit_ce_tolerance'] == {'absolute': 2e-6, 'relative': 2e-6},
            'Independent logit-CE tolerance differs from predeclared budget')
    evaluation = protocol['evaluation']
    for key, expected in {'model_mode': 'eval', 'autocast': False,
                          'parameter_dtype': 'float32', 'logits_dtype': 'float32',
                          'TF32': False, 'batch_size': 128, 'vocabulary_size': 188,
                          'optimizer_constructed': False, 'training_forward': False,
                          'backward_called': False}.items():
        require(evaluation[key] == expected, f'Forward configuration mismatch: {key}')
    require(evaluation['actions'] == ['identity'], 'Evaluation includes nonidentity actions')
    scope = protocol['scope']
    for key in ('new_training', 'source_files_changed', 'test_arrays_decoded',
                'new_CKA', 'GroupWorld', 'F17_changes', 'auto_continuation'):
        require(scope[key] is False, 'Out-of-scope protocol setting: ' + key)
    require(protocol['decoded_arrays'] == [f'{s}_{name}' for s in SPLITS for name in
                ('lengths', 'labels', 'one_line_input', 'canonical_cycle_input')],
            'Primary inference decoded-array whitelist changed')
    require(set(protocol['samples_per_length']) == set(SPLITS), 'Unexpected split set')
    if not synthetic:
        require(protocol['samples_per_length'] == {'train': 2048, 'validation': 1024},
                'Full-set per-length sample counts changed')
        require(protocol['expected_model_sample_pairs'] == 147456 and
                protocol['expected_length_cells'] == 96, 'Expected full-set coverage changed')
    source_protocol_path = Path(protocol['source_protocol_path'])
    require(sha(source_protocol_path) == protocol['source_protocol_sha256'], 'Source protocol hash changed')
    source = load_json(source_protocol_path)
    require(source['lengths'] == list(LENGTHS), 'Source length set differs')
    manifest = protocol['training_manifest']
    require(manifest == source['training_manifest'], 'Model manifest differs from original source protocol')
    expected = {(a, e, t) for a in ARCHITECTURES for e in ENCODINGS for t in TASKS}
    require(len(manifest) == 16 and len({j['run_id'] for j in manifest}) == 16 and
            {(j['architecture'], j['encoding'], j['task']) for j in manifest} == expected,
            'Audit requires all sixteen fixed conditions exactly once')
    require(all(j['updates'] == 8000 for j in manifest), 'Checkpoint endpoint is not fixed at 8k')
    require(all('/' not in j['run_id'] and '..' not in j['run_id'] for j in manifest),
            'Unsafe run_id path')
    checkpoints = protocol['checkpoint_manifest']
    require(len(checkpoints) == 16 and {j['run_id'] for j in checkpoints} == {j['run_id'] for j in manifest},
            'Checkpoint manifest incomplete')
    source_data = Path(protocol['source_data_path'])
    require(source_data.is_file() and sha(source_data) == protocol['source_data_sha256'],
            'Source data hash changed')
    return protocol, source, checkpoints


def source_labels(protocol):
    """Decode only four named train/validation arrays, regardless of archive keys."""
    decoded = []
    with np.load(protocol['source_data_path'], allow_pickle=False) as archive:
        values = {}
        for name in SOURCE_ARRAYS:
            require(name in SOURCE_ARRAYS and not name.startswith('test_'), 'Forbidden source-array access')
            values[name] = archive[name].copy()
            decoded.append(name)
    if 'label_task_names' in protocol:
        task_names = protocol['label_task_names']
    else:
        relation_path = WORKSPACE/'external/neurips/configs/property_task_geometry.toml'
        relations = tomllib.loads(relation_path.read_text())['relation_pairs']
        task_names = sorted({r[side] for r in relations for side in ('left', 'right')})
    require(len(task_names) == 16 and len(set(task_names)) == 16 and
            all(t in task_names for t in TASKS), 'Source label-task schema invalid')
    token_map = protocol.get('numeric_answer_token_ids', {str(n): n for n in range(31)})
    require(token_map == {str(n): n for n in range(31)}, 'Frozen numeric answer-token mapping changed')
    for split, actions in (('train', 1), ('validation', 3)):
        ns, labels = values[split+'_lengths'], values[split+'_labels']
        require(ns.ndim == 1 and np.issubdtype(ns.dtype, np.integer), split+': length schema invalid')
        require(labels.shape == (len(ns), actions, 16) and np.issubdtype(labels.dtype, np.integer),
                split+': source labels schema invalid')
        require(set(map(int, np.unique(ns))) == set(LENGTHS), split+': source lengths incomplete')
        require(all(int(np.count_nonzero(ns == n)) == protocol['samples_per_length'][split]
                    for n in LENGTHS), split+': source length sample count changed')
    return values, task_names, token_map, decoded


def preflight(diag, protocol, checkpoints):
    required = ('prediction_manifest.json', 'metrics_by_length.csv', 'metrics_summary.csv',
                'validation_reproduction.csv', 'evaluation_execution.json')
    require(all((diag/name).is_file() for name in required), 'Refuse incomplete evaluation: missing summary artifact')
    manifest = load_json(diag/'prediction_manifest.json')
    expected = {j['run_id'] for j in protocol['training_manifest']}
    require(len(manifest) == 16 and len({j['run_id'] for j in manifest}) == 16 and
            {j['run_id'] for j in manifest} == expected, 'Refuse partial or extra prediction models')
    output_root = Path(protocol['output']).resolve()
    for job in manifest:
        path = Path(job['path'])
        require(path.resolve() == output_root/job['run_id']/'predictions.npz', 'Prediction path outside exact model directory')
        require(path.is_file() and sha(path) == job['sha256'], 'Prediction hash mismatch: '+job['run_id'])
        if 'parameter_sha256_before' in job or 'parameter_sha256_after' in job:
            require(job['parameter_sha256_before'] == job['parameter_sha256_after'], 'Inference changed model parameters')
    for cp in checkpoints:
        for prefix in ('checkpoint', 'stage'):
            path = Path(cp[prefix+'_path'])
            require(path.is_file() and sha(path) == cp[prefix+'_sha256'], 'Bound source '+prefix+' changed: '+cp['run_id'])
    execution = load_json(diag/'evaluation_execution.json')
    require(execution['status'] in ('completed', 'completed_with_reproduction_difference') and
            execution['completed_models'] == 16, 'Primary evaluation has not completed all sixteen models')
    require(execution['sample_model_pairs'] == protocol['expected_model_sample_pairs'] and
            execution['length_cells'] == 96, 'Execution coverage incomplete')
    for key in ('optimizer_constructed', 'backward_called', 'parameters_changed',
                'test_arrays_decoded', 'geometry_evaluated', 'new_training'):
        require(execution[key] is False, 'Forbidden execution activity: ' + key)
    require(execution['decoded_arrays'] == protocol['decoded_arrays'], 'Execution decoded arrays differ from protocol')
    return manifest, execution


def stable_ce64(logits, truth):
    scores = logits.astype(np.float64)
    maximum = scores.max(axis=1)
    partition = maximum + np.log(np.exp(scores-maximum[:, None]).sum(axis=1, dtype=np.float64))
    return partition-scores[np.arange(len(truth)), truth]


def review_split(arrays, split, job, protocol, source, task_names, token_map):
    ns = arrays[split+'_lengths']
    indices = arrays[split+'_sample_index']
    size = 3*protocol['samples_per_length'][split]
    context = job['run_id']+'/'+split
    for name in ('lengths', 'sample_index', 'gold_answer_token', 'predicted_answer_token', 'eos_predicted_token'):
        values = arrays[split+'_'+name]
        require(values.shape == (size,) and np.issubdtype(values.dtype, np.integer), context+': integer array schema '+name)
    require(np.array_equal(np.sort(indices), np.arange(size)), context+': missing, repeated or out-of-range sample indices')
    require(np.array_equal(ns, source[split+'_lengths'][indices]), context+': source length/index mismatch')
    gold = arrays[split+'_gold_answer_token']
    raw_gold = source[split+'_labels'][indices, 0, task_names.index(job['task'])]
    require(bool(np.all((raw_gold >= 0) & (raw_gold <= 30))), context+': source answer out of numeric domain')
    source_gold = np.array([token_map[str(int(x))] for x in raw_gold], dtype=np.int64)
    require(np.array_equal(gold, source_gold), context+': saved gold differs from source identity label')
    ce_checks, argmax_checks = {}, {}
    for kind in ('answer', 'eos'):
        ce = arrays[split+'_'+kind+'_ce']
        logits = arrays[split+'_'+kind+'_logits']
        predicted_name = 'predicted_answer_token' if kind == 'answer' else 'eos_predicted_token'
        pred = arrays[split+'_'+predicted_name]
        require(ce.shape == (size,) and ce.dtype == np.float64 and np.isfinite(ce).all() and bool(np.all(ce >= 0)),
                context+': invalid saved float64 '+kind+' CE')
        require(logits.shape == (size, 188) and logits.dtype == np.float32 and np.isfinite(logits).all(),
                context+': invalid all-vocabulary FP32 '+kind+' logits')
        require(bool(np.all((pred >= 0) & (pred < 188))), context+': token predictions outside vocabulary')
        independent_pred = logits.argmax(axis=1)
        require(np.array_equal(independent_pred, pred), context+': independent full-vocabulary '+kind+' argmax mismatch')
        truth = gold if kind == 'answer' else np.full(size, protocol['eos_token_id'], dtype=np.int64)
        require(bool(np.all((truth >= 0) & (truth < 188))), context+': gold token outside vocabulary')
        ce64 = stable_ce64(logits, truth)
        difference = np.abs(ce64-ce)
        rules = protocol['independent_logit_ce_tolerance']
        tolerance = rules['absolute']+rules['relative']*np.abs(ce)
        worst = int(np.argmax(difference/tolerance))
        worst_example = dict(run_id=job['run_id'], split=split, kind=kind,
            sample_index=int(indices[worst]), length=int(ns[worst]), saved_fp32_ce=float(ce[worst]),
            independent_float64_ce=float(ce64[worst]), signed_difference=float(ce64[worst]-ce[worst]),
            absolute_difference=float(difference[worst]), tolerance=float(tolerance[worst]),
            tolerance_ratio=float(difference[worst]/tolerance[worst]))
        require(bool(np.all(difference <= tolerance)), 'Independent per-sample logit-CE budget failed: '+json.dumps(worst_example))
        ce_checks[kind] = dict(samples=size, max_absolute_difference=float(difference.max()),
            maximum_tolerance_ratio=float((difference/tolerance).max()), worst_sample=worst_example,
            passed=True, primary_ce_preserved=True)
        argmax_checks[kind] = size
    answers = arrays[split+'_predicted_answer_token'] == gold
    eos = arrays[split+'_eos_predicted_token'] == protocol['eos_token_id']
    length_rows = []
    for n in LENGTHS:
        chosen = ns == n
        count = int(np.count_nonzero(chosen))
        require(count == protocol['samples_per_length'][split], context+': length sample count mismatch')
        answer_hits, eos_hits = int(np.count_nonzero(answers[chosen])), int(np.count_nonzero(eos[chosen]))
        a_ce = math.fsum(map(float, arrays[split+'_answer_ce'][chosen]))/count
        e_ce = math.fsum(map(float, arrays[split+'_eos_ce'][chosen]))/count
        length_rows.append(dict(run_id=job['run_id'], architecture=job['architecture'],
            encoding=job['encoding'], task=job['task'], split=split, length=n, samples=count,
            answer_correct=answer_hits, eos_correct=eos_hits, answer_accuracy=answer_hits/count,
            answer_ce=a_ce, eos_accuracy=eos_hits/count, eos_ce=e_ce, combined_ce=(a_ce+e_ce)/2))
    summary = dict(run_id=job['run_id'], architecture=job['architecture'], encoding=job['encoding'],
        task=job['task'], split=split, samples=size,
        **{field: math.fsum(row[field] for row in length_rows)/3 for field in METRICS})
    return length_rows, summary, dict(run_id=job['run_id'], split=split,
        samples=size, argmax_comparisons=argmax_checks, independent_ce=ce_checks,
        exact_source_identity_gold=True, complete_unique_sample_coverage=True)


def check_metrics_csv(path, expected, by_length):
    rows = read_csv(path)
    key_fields = ('run_id', 'split', 'length') if by_length else ('run_id', 'split')
    def key(row):
        return tuple(str(row[field]) for field in key_fields)
    expected_map = {key(row): row for row in expected}
    require(len(rows) == len(expected) and len(expected_map) == len(expected), str(path)+': row count mismatch')
    require(len({key(row) for row in rows}) == len(rows) and {key(row) for row in rows} == set(expected_map),
            str(path)+': row keys missing, extra or repeated')
    checked = 0
    for row in rows:
        target = expected_map[key(row)]
        for field in ('architecture', 'encoding', 'task'):
            require(row[field] == target[field], str(path)+': metadata mismatch '+field)
            checked += 1
        require(int(row['samples']) == target['samples'], str(path)+': sample count mismatch')
        checked += 1
        for field in METRICS:
            close(row[field], target[field], str(path)+'/'+repr(key(row))+'/'+field)
            checked += 1
    return dict(path=str(path), rows=len(rows), fields_compared=checked, passed=True)


def expected_reproduction(checkpoints, length_rows, summaries, protocol):
    length_lookup = {(row['run_id'], row['length']): row for row in length_rows if row['split'] == 'validation'}
    summary_lookup = {row['run_id']: row for row in summaries if row['split'] == 'validation'}
    result, exceeded = [], []
    for job in checkpoints:
        stage_path = Path(job['stage_path'])
        stage = load_json(stage_path)
        for field in ('run_id', 'architecture', 'encoding', 'task'):
            require(stage[field] == job[field], 'Bound source stage metadata mismatch: '+field)
        require(stage['step'] == 8000 and stage['status'] == 'stage_complete' and stage['fixed_endpoint_completed'],
                'Bound source stage is not complete fixed 8k')
        original = stage['validation']
        original_lengths = {int(row['length']): row for row in original['per_length']}
        require(len(original['per_length']) == 3 and set(original_lengths) == set(LENGTHS), 'Source validation lengths incomplete')
        for length in (*LENGTHS, 'equal_length'):
            old = original if length == 'equal_length' else original_lengths[length]
            current = summary_lookup[job['run_id']] if length == 'equal_length' else length_lookup[job['run_id'], length]
            if 'full_vocab_accuracy' in old:
                close(old['answer_accuracy'], old['full_vocab_accuracy'], 'Original answer/full-vocabulary accuracy alias')
            for metric in METRICS:
                before, after = finite(old[metric], 'Original '+metric), float(current[metric])
                difference = after-before
                rules = protocol['validation_reproduction_tolerances']
                tolerance = rules['accuracy_absolute'] if 'accuracy' in metric else rules['ce_absolute']+rules['ce_relative']*abs(before)
                record = dict(run_id=job['run_id'], architecture=job['architecture'], encoding=job['encoding'],
                    task=job['task'], length=str(length), metric=metric, original=before, recomputed=after,
                    signed_difference=difference, tolerance=tolerance, passed=abs(difference) <= tolerance,
                    reference_path=str(stage_path))
                result.append(record)
                if not record['passed']:
                    exceeded.append(record)
    return result, exceeded


def check_reproduction_csv(path, expected):
    rows = read_csv(path)
    def key(row):
        return row['run_id'], str(row['length']), row['metric']
    target = {key(row): row for row in expected}
    require(len(rows) == len(expected) == 320 and len({key(row) for row in rows}) == 320 and
            {key(row) for row in rows} == set(target), 'Reproduction table must retain all320 original/new comparisons')
    checked = 0
    for row in rows:
        original = target[key(row)]
        for field in ('architecture', 'encoding', 'task', 'reference_path'):
            require(row[field] == original[field], 'Reproduction metadata mismatch '+field)
            checked += 1
        for field in ('original', 'recomputed', 'signed_difference', 'tolerance'):
            close(row[field], original[field], 'Reproduction '+repr(key(row))+'/'+field)
            checked += 1
        require(boolean(row['passed']) == original['passed'], 'Reproduction pass/fail differs from fixed budget')
        checked += 1
    return dict(path=str(path), rows=320, fields_compared=checked, passed=True)


def audit(diag, synthetic=False):
    protocol, source_config, checkpoints = protocol_check(diag, synthetic)
    manifests, execution = preflight(diag, protocol, checkpoints)
    labels, task_names, token_map, decoded = source_labels(protocol)
    jobs = {row['run_id']: row for row in protocol['training_manifest']}
    length_rows, summaries, prediction_checks = [], [], []
    for record in manifests:
        job = jobs[record['run_id']]
        with np.load(record['path'], allow_pickle=False) as archive:
            require(set(archive.files) == EXPECTED_PREDICTION_ARRAYS,
                    job['run_id']+': prediction schema has missing, extra or forbidden arrays')
            for split in SPLITS:
                arrays = {split+'_'+name: archive[split+'_'+name] for name in ARRAY_NAMES}
                by_n, overall, review = review_split(arrays, split, job, protocol, labels, task_names, token_map)
                length_rows.extend(by_n)
                summaries.append(overall)
                prediction_checks.append(review)
    require(len(length_rows) == 96 and len(summaries) == 32 and
            sum(row['samples'] for row in summaries) == protocol['expected_model_sample_pairs'],
            'Independent full-set coverage incomplete')
    table_checks = [check_metrics_csv(diag/'metrics_by_length.csv', length_rows, True),
                    check_metrics_csv(diag/'metrics_summary.csv', summaries, False)]
    reproduction, exceeded = expected_reproduction(checkpoints, length_rows, summaries, protocol)
    table_checks.append(check_reproduction_csv(diag/'validation_reproduction.csv', reproduction))
    require(execution['validation_reproduction_passed'] == (not exceeded), 'Execution reproduction flag disagrees with fixed tolerance')
    require(execution['validation_reproduction_failed_rows'] == exceeded, 'Execution omits or changes reproduction failures')
    worst_ce = max((check['independent_ce'][kind]['worst_sample'] for check in prediction_checks
                    for kind in ('answer', 'eos')), key=lambda value: value['tolerance_ratio'])
    max_ce_abs = max(check['independent_ce'][kind]['max_absolute_difference'] for check in prediction_checks
                     for kind in ('answer', 'eos'))
    input_hashes = {str(diag/name): sha(diag/name) for name in
        ('protocol.json', 'freeze.json', 'prediction_manifest.json', 'evaluation_execution.json',
         'metrics_by_length.csv', 'metrics_summary.csv', 'validation_reproduction.csv')}
    result = dict(status='passed' if not exceeded else 'failed_validation_reproduction',
        generated_utc=datetime.now(timezone.utc).isoformat(), synthetic_fixture=synthetic,
        auditor_sha256=sha(__file__), input_sha256=input_hashes, models=16, splits=list(SPLITS),
        lengths=list(LENGTHS), length_weights=[1/3, 1/3, 1/3], length_rows=96, summary_rows=32,
        samples_model_pairs=sum(row['samples'] for row in summaries),
        all_vocabulary_argmax_comparisons=2*sum(row['samples'] for row in summaries),
        per_sample_ce_comparisons=2*sum(row['samples'] for row in summaries),
        independent_ce_float64=True, primary_saved_fp32_ce_preserved=True,
        independent_logit_ce_tolerance=protocol['independent_logit_ce_tolerance'],
        max_independent_ce_absolute_difference=max_ce_abs, worst_independent_ce_sample=worst_ce,
        validation_reproduction_tolerances=protocol['validation_reproduction_tolerances'],
        validation_reproduction_comparisons=320, validation_reproduction_passed=not exceeded,
        validation_reproduction_failed_rows=exceeded, table_checks=table_checks,
        source_arrays_decoded=decoded, prediction_arrays_decoded=sorted(EXPECTED_PREDICTION_ARRAYS),
        declared_fp32_eval_configuration_verified=True, full_fp32_forward_not_inferred_from_logits_dtype_alone=True,
        trained=False, evaluated_models=False, checkpoint_tensors_loaded=False, test_arrays_decoded=False,
        geometry_evaluated=False, F17_modified=False, independent_world_count=1,
        interpretation='Numerical audit of one existing development world; not confirmation, TOST, causal mechanism or independent-world replication.',
        models_summary=summaries, per_model_logit_checks=prediction_checks)
    return result


def make_fixture(directory):
    """Small synthetic all16 crossing; no real checkpoint or result access."""
    directory.mkdir(parents=True, exist_ok=True)
    output = directory/'predictions'
    source_root = directory/'source'
    source_root.mkdir()
    rng = np.random.default_rng(101017)
    task_names = ['anti_fixed_points', 'components', 'deficiencies', 'descents', 'exceedances',
        'fixed_points', 'global_descents', 'lds_length', 'left_to_right_maxima',
        'left_to_right_minima', 'lis_length', 'longest_decreasing_run', 'longest_increasing_run',
        'peaks', 'recoils', 'valleys']
    values, n_per_split = {}, {'train': 7, 'validation': 5}
    for split, actions in (('train', 1), ('validation', 3)):
        values[split+'_lengths'] = np.repeat(LENGTHS, n_per_split[split]).astype(np.int16)
        values[split+'_labels'] = rng.integers(0, 10, (3*n_per_split[split], actions, 16), dtype=np.int16)
    # Object sentinel proves the loader does not decode a forbidden test array.
    values['test_labels'] = np.array([{'must_not_decode': True}], dtype=object)
    data_path = source_root/'data.npz'
    np.savez(data_path, **values)
    jobs = [dict(run_id='fixture_'+a+'_'+e+'_'+t, architecture=a, encoding=e, task=t, updates=8000)
            for a in ARCHITECTURES for e in ENCODINGS for t in TASKS]
    source_protocol_path = source_root/'protocol.json'
    json_write(source_protocol_path, dict(training_manifest=jobs, lengths=list(LENGTHS)))
    protocol = dict(training_manifest=jobs, lengths=list(LENGTHS),
        length_weights={str(n): 1/3 for n in LENGTHS}, samples_per_length=n_per_split,
        expected_model_sample_pairs=16*sum(3*n for n in n_per_split.values()), expected_length_cells=96,
        source_protocol_path=str(source_protocol_path), source_protocol_sha256=sha(source_protocol_path),
        source_data_path=str(data_path), source_data_sha256=sha(data_path), output=str(output), eos_token_id=103,
        label_task_names=task_names, numeric_answer_token_ids={str(n): n for n in range(31)},
        decoded_arrays=[f'{s}_{name}' for s in SPLITS for name in
                        ('lengths', 'labels', 'one_line_input', 'canonical_cycle_input')],
        validation_reproduction_tolerances=dict(accuracy_absolute=0.0, ce_absolute=1e-5, ce_relative=1e-6),
        independent_logit_ce_tolerance=dict(absolute=2e-6, relative=2e-6),
        evaluation=dict(model_mode='eval', autocast=False, parameter_dtype='float32', logits_dtype='float32',
            TF32=False, batch_size=128, vocabulary_size=188, optimizer_constructed=False,
            training_forward=False, backward_called=False, actions=['identity']),
        scope=dict(new_training=False, source_files_changed=False, test_arrays_decoded=False,
                   new_CKA=False, GroupWorld=False, F17_changes=False, auto_continuation=False))
    prediction_manifest, checkpoint_manifest, length_rows, summaries = [], [], [], []
    for job in jobs:
        arrays = {}
        for split in SPLITS:
            ns = values[split+'_lengths']
            gold = values[split+'_labels'][:, 0, task_names.index(job['task'])]
            count = len(ns)
            prefix = split+'_'
            arrays[prefix+'lengths'] = ns
            arrays[prefix+'sample_index'] = np.arange(count, dtype=np.int32)
            arrays[prefix+'gold_answer_token'] = gold
            for kind in ('answer', 'eos'):
                logits = rng.normal(0, 1, (count, 188)).astype(np.float32)
                truth = gold if kind == 'answer' else np.full(count, 103, dtype=np.int64)
                logits[np.arange(count), truth] += np.linspace(-2, 7, count).astype(np.float32)
                arrays[prefix+kind+'_logits'] = logits
                arrays[prefix+('predicted_answer_token' if kind == 'answer' else 'eos_predicted_token')] = logits.argmax(axis=1).astype(np.int16)
                arrays[prefix+kind+'_ce'] = stable_ce64(logits, truth).astype(np.float32).astype(np.float64)
            rows, summary, _ = review_split(arrays, split, job, protocol, values, task_names, protocol['numeric_answer_token_ids'])
            length_rows.extend(rows)
            summaries.append(summary)
        model_path = output/job['run_id']
        model_path.mkdir(parents=True)
        prediction_path = model_path/'predictions.npz'
        np.savez(prediction_path, **arrays)
        prediction_manifest.append(dict(run_id=job['run_id'], path=str(prediction_path), sha256=sha(prediction_path),
                                        parameter_sha256_before='unchanged', parameter_sha256_after='unchanged'))
        stage_directory = source_root/job['run_id']
        stage_directory.mkdir()
        stage_path = stage_directory/'stage_8000.json'
        valid = next(r for r in summaries if r['run_id'] == job['run_id'] and r['split'] == 'validation')
        validation = {field: valid[field] for field in METRICS}
        validation['per_length'] = [{k:r[k] for k in ('length', 'samples', *METRICS)} for r in length_rows
                                    if r['run_id'] == job['run_id'] and r['split'] == 'validation']
        json_write(stage_path, dict(**job, status='stage_complete', step=8000, fixed_endpoint_completed=True,
                                   validation=validation))
        checkpoint = stage_directory/'checkpoint_step_8000.pt'
        checkpoint.write_bytes(b'fixture only; never deserialize')
        checkpoint_manifest.append(dict(**job, stage_path=str(stage_path), stage_sha256=sha(stage_path),
            checkpoint_path=str(checkpoint), checkpoint_sha256=sha(checkpoint)))
    protocol['checkpoint_manifest'] = checkpoint_manifest
    json_write(directory/'protocol.json', protocol)
    json_write(directory/'freeze.json', dict(protocol_sha256=sha(directory/'protocol.json'),
        source_sha256={str(Path(__file__).resolve()): sha(__file__)}, training_forbidden=True))
    json_write(directory/'prediction_manifest.json', prediction_manifest)
    fields = ['run_id', 'architecture', 'encoding', 'task', 'split', 'samples', *METRICS]
    write_csv(directory/'metrics_summary.csv', summaries, fields)
    write_csv(directory/'metrics_by_length.csv', [{field:r[field] for field in fields+['length']} for r in length_rows],
              fields[:5]+['length']+fields[5:])
    reproduction, exceeded = expected_reproduction(checkpoint_manifest, length_rows, summaries, protocol)
    require(not exceeded, 'Synthetic reproduction should be exact')
    write_csv(directory/'validation_reproduction.csv', reproduction, ['run_id', 'architecture', 'encoding', 'task',
        'length', 'metric', 'original', 'recomputed', 'signed_difference', 'tolerance', 'passed', 'reference_path'])
    json_write(directory/'evaluation_execution.json', dict(status='completed', completed_models=16,
        sample_model_pairs=protocol['expected_model_sample_pairs'], length_cells=96,
        optimizer_constructed=False, backward_called=False, parameters_changed=False,
        test_arrays_decoded=False, geometry_evaluated=False, new_training=False,
        decoded_arrays=protocol['decoded_arrays'], validation_reproduction_passed=True,
        validation_reproduction_failed_rows=[]))
    return protocol


def fixture():
    checks = []
    with tempfile.TemporaryDirectory(prefix='permworld_fullset_independent_') as temporary:
        directory = Path(temporary)
        protocol = make_fixture(directory)
        result = audit(directory, synthetic=True)
        require(result['status'] == 'passed' and result['source_arrays_decoded'] == list(SOURCE_ARRAYS),
                'Synthetic positive audit failed or forbidden source array decoded')
        checks.append('positive all16/full188 logits/CE/counts/equal-length weighting/reproduction')
        manifest = load_json(directory/'prediction_manifest.json')
        first_path = Path(manifest[0]['path'])
        with np.load(first_path, allow_pickle=False) as archive:
            pristine = {name: archive[name].copy() for name in archive.files}
        def prediction_error(label, mutate):
            changed = {name: value.copy() for name, value in pristine.items()}
            mutate(changed)
            np.savez(first_path, **changed)
            manifest[0]['sha256'] = sha(first_path)
            json_write(directory/'prediction_manifest.json', manifest)
            try:
                audit(directory, synthetic=True)
            except AssertionError:
                checks.append(label)
            else:
                raise AssertionError('Negative fixture was accepted: '+label)
            np.savez(first_path, **pristine)
            manifest[0]['sha256'] = sha(first_path)
            json_write(directory/'prediction_manifest.json', manifest)
        prediction_error('reject forbidden prediction test array', lambda a:a.update(test_logits=np.zeros((1,188),np.float32)))
        prediction_error('reject missing or repeated sample index', lambda a:a['train_sample_index'].__setitem__(0,1))
        prediction_error('reject changed source gold label', lambda a:a['train_gold_answer_token'].__setitem__(0,30))
        prediction_error('reject saved argmax inconsistency', lambda a:a['train_predicted_answer_token'].__setitem__(0,
            (int(a['train_predicted_answer_token'][0])+1)%188))
        prediction_error('reject independent CE beyond frozen numerical budget', lambda a:a['validation_answer_ce'].__setitem__(0,
            float(a['validation_answer_ce'][0])+.01))
        prediction_error('reject float64 logits despite identical values', lambda a:a.update(train_answer_logits=a['train_answer_logits'].astype(np.float64)))
        summary_path = directory/'metrics_summary.csv'
        summary_bytes = summary_path.read_bytes()
        rows = read_csv(summary_path)
        rows[0]['answer_accuracy'] = float(rows[0]['answer_accuracy'])+.01
        write_csv(summary_path, rows, list(rows[0]))
        try:
            audit(directory, synthetic=True)
        except AssertionError:
            checks.append('reject corrupted primary summary')
        else:
            raise AssertionError('Corrupted summary accepted')
        summary_path.write_bytes(summary_bytes)
        manifest_bytes = (directory/'prediction_manifest.json').read_bytes()
        json_write(directory/'prediction_manifest.json', manifest[:-1])
        try:
            audit(directory, synthetic=True)
        except AssertionError:
            checks.append('reject partial model inventory before reading predictions')
        else:
            raise AssertionError('Partial inventory accepted')
        (directory/'prediction_manifest.json').write_bytes(manifest_bytes)
        require(audit(directory, synthetic=True)['status'] == 'passed', 'Fixture restoration failed')
        result['fixture_checks'] = checks
        result['fixture_check_count'] = len(checks)
    return result


def review_text(result):
    lines = ['PermWorld full training/validation evaluation: independent statistical review',
        'Status:'+result['status'], 'Synthetic fixture:'+str(result.get('synthetic_fixture', False)),
        'This review reads saved numerical artifacts and four source train/validation length/label arrays only; no models, training, checkpoint-tensor deserialization, test access, or CKA.',
        'First verify frozen protocol and source/data/prediction hashes, then independently recompute argmax and float64 logsumexp CE from every full188-dimensional logit vector.',
        'Primary-report CE retains original FP32 per-sample CE; float64 is numerical validation, changing no primary endpoint or original value.',
        'Aggregate per length first; n10/n11/n12 each have fixed1/3 weight. EOS and answers remain separate; combined CE averages the two CEs.',
        'Validation reproduction retains old/new values and signed differences; accuracy tolerance0 and CE budget1e-5+1e-6×|old value|.',
        'Independent logit-CE budget2e-6+2e-6×|original FP32CE|, not enlarged after observation.',
        'Forward precision is also checked against frozen evaluation settings; saved float32 logits alone do not establish no internal autocast.',
        'This is a numerical audit of one existing development data world;16 models are not16 independent worlds, with no equivalence/mechanism conclusions.',
        'If numerical limits fail, retain original artifacts and report failures. No additional training or automatic representation work.',
        '', 'Machine-audit results:', json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)]
    return '\n'.join(lines)+'\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--fixture', action='store_true')
    modes.add_argument('--run-real', action='store_true')
    args = parser.parse_args()
    try:
        result = fixture() if args.fixture else audit(DIAG)
    except Exception as error:
        result = dict(status='failed', synthetic_fixture=args.fixture, auditor_sha256=sha(__file__),
            generated_utc=datetime.now(timezone.utc).isoformat(), error_type=type(error).__name__, error=str(error),
            trained=False, evaluated_models=False, checkpoint_tensors_loaded=False, test_arrays_decoded=False,
            geometry_evaluated=False, F17_modified=False)
        (DIAG/'independent_audit_review.txt').write_text(review_text(result))
        print(json.dumps(result, ensure_ascii=False), flush=True)
        raise
    (DIAG/'independent_audit_review.txt').write_text(review_text(result))
    print(json.dumps({key:result[key] for key in ('status', 'synthetic_fixture', 'models',
        'samples_model_pairs', 'length_rows', 'summary_rows', 'per_sample_ce_comparisons',
        'max_independent_ce_absolute_difference', 'validation_reproduction_passed')}, ensure_ascii=False), flush=True)
    if result['status'] != 'passed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
