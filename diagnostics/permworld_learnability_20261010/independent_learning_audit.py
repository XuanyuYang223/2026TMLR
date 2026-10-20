"""CPU-only audit of fixed-8k validation reporting, never a training launcher.

--fixture reads synthetic objects only. --run-real refuses to inspect partial
endpoints: every manifest model and the four reporter CSVs must already exist.
No torch import, checkpoint tensors, test arrays, metrics, or model inference.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import tempfile
import tomllib

import numpy as np


WORKSPACE = Path(__file__).resolve().parents[2]
DIAG = Path(__file__).resolve().parent
ROOT = Path('/mnt/yangx-projects-f/20261010/permworld_learnability')
LABEL_ARRAYS = ('train_lengths', 'train_labels', 'validation_lengths', 'validation_labels')
TABLES = ('model_learning_summary.csv', 'pair_learning_gates.csv',
          'validation_length.csv', 'training_length_majority.csv')
NUMERICAL_TOLERANCE = 5e-12
GATE_TOLERANCE = 1e-10  # The reporter’s fixed boundary convention, not a pp margin.


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(2**20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                    allow_nan=False) + '\n')


def close(a, b, label):
    require(math.isfinite(float(a)) and math.isfinite(float(b)), label + ': nonfinite')
    require(abs(float(a)-float(b)) <= NUMERICAL_TOLERANCE,
            f'{label}: {a!r} != {b!r}')


def mean(values):
    values = list(values)
    require(bool(values), 'Empty equal-weight average')
    return math.fsum(map(float, values))/len(values)


def source_version(frozen):
    return hashlib.sha256(json.dumps(frozen['source_sha256'], sort_keys=True,
                                     separators=(',', ':')).encode()).hexdigest()


def protocol_and_schema(root, diag):
    config = read_json(diag/'protocol.json')
    frozen = read_json(diag/'freeze.json')
    digest = sha(diag/'protocol.json')
    require(digest == frozen['protocol_sha256'] == sha(root/'protocol.json'),
            'Frozen protocol or ROOT mirror mismatch')
    for relative, expected in frozen['source_sha256'].items():
        require(sha(WORKSPACE/relative) == expected, 'Frozen source mismatch: '+relative)
    require(config['lengths'] == [10, 11, 12], 'Unexpected fixed lengths')
    require(config['training']['updates'] == 8000, 'Unexpected fixed endpoint')
    require(config['tasks'] == ['descents', 'recoils', 'peaks', 'valleys'], 'Unexpected tasks')
    require(config['encodings'] == ['one_line', 'canonical_cycle'], 'Unexpected encodings')
    require(config['architectures'] == ['transformer', 'gru'], 'Unexpected architectures')
    require(config['scope']['geometry_used_for_selection'] is False and
            config['scope']['test_used_for_selection'] is False, 'Selection scope changed')
    require(config['development_choice']['candidate_count'] == 1, 'Candidate count changed')
    require(len(config['units']) == 1, 'This audit is for one exploratory world')
    metadata = WORKSPACE/'external/neurips/configs/property_task_geometry.toml'
    relations = tomllib.loads(metadata.read_text())['relation_pairs']
    task_names = sorted({r[side] for r in relations for side in ('left', 'right')})
    require(len(task_names) == 16, 'Unexpected data label schema')
    manifest = config['training_manifest']
    require(len(manifest) == 16 and len({j['run_id'] for j in manifest}) == 16,
            'Manifest does not contain 16 distinct models')
    require({(j['architecture'], j['task'], j['encoding']) for j in manifest} ==
            {(a, t, e) for a in config['architectures'] for t in config['tasks']
             for e in config['encodings']}, 'Manifest is not the fixed full crossing')
    return config, frozen, digest, task_names


def preflight(root, diag, config):
    """Require completion before reading any actual stage or label contents."""
    paths = [root/'models'/j['run_id']/name for j in config['training_manifest']
             for name in ('stage_8000.json', 'completed.json')]
    paths += [diag/name for name in TABLES]
    missing = [str(p) for p in paths if not p.is_file()]
    require(not missing, 'Refuse partial-development review; missing: '+', '.join(missing))
    return paths


def labels_only(path):
    """Never decode a test/input array, even if one is present in the archive."""
    decoded = []
    with np.load(path, allow_pickle=False) as archive:
        def get(name):
            require(name in LABEL_ARRAYS and not name.startswith('test'),
                    'Forbidden NPZ array access: '+name)
            decoded.append(name)
            return archive[name].copy()
        values = {name: get(name) for name in LABEL_ARRAYS}
    require(tuple(decoded) == LABEL_ARRAYS, 'Unexpected decoded arrays')
    return values, decoded


def independent_baselines(values, config, task_names):
    for split, actions in (('train', 1), ('validation', 3)):
        ns, ys = values[split+'_lengths'], values[split+'_labels']
        require(ns.ndim == 1 and ys.ndim == 3 and ys.shape == (len(ns), actions, len(task_names)),
                split+': unexpected label schema')
        require(sorted(map(int, np.unique(ns))) == config['lengths'], split+': length window changed')
        require(np.issubdtype(ys.dtype, np.integer), split+': noninteger labels')
        expected = config['data']['examples_per_length'][split]
        require(all(int(np.count_nonzero(ns == n)) == expected for n in config['lengths']),
                split+': per-length sample count changed')
    baseline, rows = {}, []
    for task in config['tasks']:
        column = task_names.index(task)
        for n in config['lengths']:
            train = values['train_labels'][values['train_lengths'] == n, 0, column]
            valid = values['validation_labels'][values['validation_lengths'] == n, 0, column]
            require(bool(np.all((train >= 0) & (train <= 30))) and
                    bool(np.all((valid >= 0) & (valid <= 30))), 'Answer domain violation')
            histogram = Counter(map(int, train))
            largest = max(histogram.values())
            mode = min(k for k, count in histogram.items() if count == largest)
            hits = sum(int(y) == mode for y in valid)
            base = hits/len(valid)
            baseline[(task, n)] = base
            rows.append(dict(task=task, length=n, training_samples=len(train), majority_answer=mode,
                majority_training_count=largest, majority_training_fraction=largest/len(train),
                validation_samples=len(valid), validation_baseline_accuracy=base))
    return baseline, rows


def validate_endpoint(stage, completed, job, config, expected):
    for record, required_status in ((stage, 'stage_complete'), (completed, 'completed')):
        require(record.get('status') == required_status, 'Not a completed endpoint: '+job['run_id'])
        for field, value in job.items():
            require(record.get(field) == value, job['run_id']+': manifest '+field)
        for field, value in expected.items():
            require(record.get(field) == value, job['run_id']+': reference '+field)
        require(record.get('step') == record.get('fixed_endpoint_step') == 8000,
                job['run_id']+': not fixed 8k')
        require(record.get('source_model_run') is True and record.get('fixed_endpoint_completed') is True,
                job['run_id']+': partial or engineering record')
        require(record.get('validation_missing_due_to_budget') is False,
                job['run_id']+': final validation missing')
        require(record.get('actual_initial_identity_verified') is True,
                job['run_id']+': initial identity flag')
        require(record.get('test_evaluated') is False and record.get('geometry_evaluated') is False and
                record.get('formal_training_started') is False, job['run_id']+': scope violation')
        require(record.get('finite_gradient_updates') == 8000, job['run_id']+': update count')
        require(record.get('exposures') == 8000*config['training']['effective_batch'] and
                record.get('supervised_tokens') == 16000*config['training']['effective_batch'],
                job['run_id']+': exposure count')
        for field in ('initial_parameter_sha256', 'initial_checkpoint_sha256', 'starting_rng_hash',
                      'checkpoint_sha256'):
            token = record.get(field)
            require(isinstance(token, str) and len(token) == 64 and
                    all(c in '0123456789abcdef' for c in token), job['run_id']+': hash '+field)
    a, b = dict(stage), dict(completed)
    a.pop('status'); b.pop('status')
    require(a == b, job['run_id']+': stage/completed payloads differ')


def model_values(stage, baseline, config):
    validation = stage['validation']
    require(validation is not None and validation['split'] == 'validation' and
            validation['action'] == 'identity', 'Not identity validation')
    require(validation['test_evaluated'] is False and validation['geometry_evaluated'] is False and
            validation['eos_teacher_forced'] is True, 'Validation scope or EOS semantics changed')
    rows = validation['per_length']
    require([r['length'] for r in rows] == config['lengths'], 'Missing/reordered final lengths')
    expected_samples = config['data']['examples_per_length']['validation']
    for r in rows:
        require(r['samples'] == expected_samples, 'Per-length validation N differs')
        for field in ('full_vocab_accuracy', 'numeric_accuracy', 'eos_accuracy'):
            v = r[field]
            require(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and
                    0 <= v <= 1, 'Invalid '+field)
            require(abs(v*expected_samples-round(v*expected_samples)) < 1e-8,
                    field+': accuracy cannot arise from integer sample counts')
        require(r['numeric_accuracy']+NUMERICAL_TOLERANCE >= r['full_vocab_accuracy'],
                'Conditional numeric accuracy below full-vocab accuracy')
        for field in ('answer_ce', 'eos_ce', 'combined_ce'):
            require(math.isfinite(r[field]) and r[field] >= 0, 'Invalid '+field)
        close(r['combined_ce'], (r['answer_ce']+r['eos_ce'])/2, 'Combined CE per length')
        close(r['answer_accuracy'], r['full_vocab_accuracy'], 'Per-length answer alias')
    require(validation['samples'] == expected_samples*len(rows), 'Final sample total')
    for name, column in (('overall_full_vocab', 'full_vocab_accuracy'),
                         ('overall_numeric', 'numeric_accuracy'), ('overall_eos', 'eos_accuracy'),
                         ('answer_ce', 'answer_ce'), ('eos_ce', 'eos_ce'), ('combined_ce', 'combined_ce')):
        close(validation[name], mean(r[column] for r in rows), 'Validation aggregate '+name)
    for alias, column in (('answer_accuracy', 'full_vocab_accuracy'),
                          ('numeric_accuracy', 'numeric_accuracy'), ('eos_accuracy', 'eos_accuracy')):
        close(validation[alias], mean(r[column] for r in rows), 'Validation alias '+alias)
    task = stage['task']
    values = dict(equal_length_accuracy=mean(r['full_vocab_accuracy'] for r in rows),
        minimum_length_accuracy=min(r['full_vocab_accuracy'] for r in rows),
        equal_length_baseline=mean(baseline[(task, n)] for n in config['lengths']),
        equal_length_eos_accuracy=mean(r['eos_accuracy'] for r in rows),
        equal_length_answer_ce=mean(r['answer_ce'] for r in rows),
        equal_length_combined_ce=mean(r['combined_ce'] for r in rows),
        endpoint_completed=True, validation_complete=True)
    values['gain_over_length_baseline_pp'] = 100*(values['equal_length_accuracy']-values['equal_length_baseline'])
    gate = config['learning_gate']
    names = dict(equal_length_accuracy='per_model_equal_length_accuracy_min',
                 minimum_length_accuracy='per_length_accuracy_min',
                 gain_over_length_baseline_pp='per_model_gain_over_training_length_majority_min_pp',
                 equal_length_eos_accuracy='per_model_eos_accuracy_min')
    failures = sorted(name+'_below_fixed_gate' for name, setting in names.items()
                      if values[name]+GATE_TOLERANCE < gate[setting])
    values.update(gate_pass=not failures, failures=';'.join(failures))
    length_rows = [dict(run_id=stage['run_id'], architecture=stage['architecture'],
        encoding=stage['encoding'], task=task, step=8000, **r,
        length_baseline_accuracy=baseline[(task, r['length'])],
        gain_over_length_baseline_pp=100*(r['full_vocab_accuracy']-baseline[(task, r['length'])]),
        split='validation', action='identity') for r in rows]
    return values, length_rows


def encoding_pairs(items, config):
    lookup = {(j['architecture'], j['task'], j['encoding']): (j, s, v) for j, s, v in items}
    rows = []
    gate = config['learning_gate']
    for architecture in config['architectures']:
        for task in config['tasks']:
            left, right = [lookup[(architecture, task, e)] for e in config['encodings']]
            gaps = {n: 100*abs(next(r['full_vocab_accuracy'] for r in left[1]['validation']['per_length'] if r['length'] == n)-
                              next(r['full_vocab_accuracy'] for r in right[1]['validation']['per_length'] if r['length'] == n))
                    for n in config['lengths']}
            mean_gap = 100*abs(left[2]['equal_length_accuracy']-right[2]['equal_length_accuracy'])
            failures = []
            if mean_gap > gate['paired_encoding_accuracy_gap_max_pp']+GATE_TOLERANCE:
                failures.append('mean_encoding_gap_above_fixed_gate')
            if max(gaps.values()) > gate['paired_encoding_per_length_accuracy_gap_max_pp']+GATE_TOLERANCE:
                failures.append('length_encoding_gap_above_fixed_gate')
            if not all(x[2]['gate_pass'] for x in (left, right)):
                failures.append('paired_model_learning_gate_failed')
            require(left[1]['step'] == right[1]['step'] == 8000, 'Unequal-budget encoding pair')
            for name in ('initial_parameter_sha256', 'initial_checkpoint_sha256', 'starting_rng_hash',
                         'data_sha256', 'execution_source_version_sha256'):
                require(left[1][name] == right[1][name], architecture+'/'+task+': unpaired '+name)
            rows.append(dict(architecture=architecture, task=task, first_run_id=left[0]['run_id'],
                second_run_id=right[0]['run_id'], same_updates=True, paired_endpoint_complete=True,
                mean_accuracy_gap_pp=mean_gap, maximum_length_accuracy_gap_pp=max(gaps.values()),
                gate_pass=not failures, failures=';'.join(sorted(failures)),
                **{f'gap_n{n}_pp': gaps[n] for n in config['lengths']}))
    return rows


def csv_rows(path):
    with Path(path).open(newline='') as stream:
        return list(csv.DictReader(stream))


def table_compare(path, expected, keys, *, endpoint_only=False):
    supplied = csv_rows(path)
    if endpoint_only:
        supplied = [r for r in supplied if r.get('step') == '8000']
    require(len(supplied) == len(expected), Path(path).name+': row count')
    def key(row):
        return tuple(str(row[k]) for k in keys)
    provided = {key(r): r for r in supplied}
    require(len(provided) == len(supplied), Path(path).name+': duplicate keys')
    comparisons = 0
    for row in expected:
        k = key(row)
        require(k in provided, Path(path).name+': missing '+str(k))
        actual = provided[k]
        for field, value in row.items():
            require(field in actual, Path(path).name+': missing column '+field)
            if isinstance(value, bool):
                require(actual[field] in ('True', 'False') and (actual[field] == 'True') == value,
                        Path(path).name+': boolean '+field+' '+str(k))
            elif isinstance(value, (int, float)):
                close(actual[field], value, Path(path).name+': '+field+' '+str(k))
            elif field == 'failures':
                require(sorted(filter(None, actual[field].split(';'))) == sorted(filter(None, value.split(';'))),
                        Path(path).name+': gate failure reasons '+str(k))
            else:
                require(actual[field] == str(value), Path(path).name+': '+field+' '+str(k))
            comparisons += 1
    return dict(rows=len(expected), fields_compared=comparisons, sha256=sha(path))


def audit(root=ROOT, diag=DIAG):
    config, frozen, protocol_digest, task_names = protocol_and_schema(root, diag)
    inputs = preflight(root, diag, config)  # No partial source contents or labels before this guard.
    data_path = root/'worlds'/config['units'][0]['unit']/'data.npz'
    data_digest = sha(data_path)  # Raw archive checksum; no test-array decoding.
    values, decoded = labels_only(data_path)
    baselines, baseline_rows = independent_baselines(values, config, task_names)
    expected = dict(protocol_sha256=protocol_digest, execution_source_version_sha256=source_version(frozen),
                    data_sha256=data_digest)
    models, per_length, items = [], [], []
    for job in config['training_manifest']:
        directory = root/'models'/job['run_id']
        stage, completed = read_json(directory/'stage_8000.json'), read_json(directory/'completed.json')
        validate_endpoint(stage, completed, job, config, expected)
        values, rows = model_values(stage, baselines, config)
        models.append(dict(**job, actual_step=8000, fixed_endpoint_updates_completed=True,
            source_status='stage_complete', validation_missing_due_to_budget=False,
            exposures=stage['exposures'], supervised_tokens=stage['supervised_tokens'],
            finite_gradient_updates=8000, parameters=stage['parameters'], **values))
        per_length.extend(rows)
        items.append((job, stage, values))
    for architecture in config['architectures']:
        states = [s for j, s, _ in items if j['architecture'] == architecture]
        for name in ('initial_parameter_sha256', 'initial_checkpoint_sha256', 'parameters'):
            require(len({s[name] for s in states}) == 1, architecture+': initial identity/size varies across tasks')
    pairs = encoding_pairs(items, config)
    comparisons = {
        'training_length_majority.csv': table_compare(diag/'training_length_majority.csv', baseline_rows, ('task', 'length')),
        'model_learning_summary.csv': table_compare(diag/'model_learning_summary.csv', models, ('run_id',)),
        'validation_length.csv': table_compare(diag/'validation_length.csv', per_length, ('run_id', 'length'), endpoint_only=True),
        'pair_learning_gates.csv': table_compare(diag/'pair_learning_gates.csv', pairs, ('architecture', 'task')),
    }
    accepted = {a: all(r['gate_pass'] for r in models if r['architecture'] == a) and
                   all(r['gate_pass'] for r in pairs if r['architecture'] == a) for a in config['architectures']}
    return dict(status='passed', generated_utc=datetime.now(timezone.utc).isoformat(),
        auditor_sha256=sha(__file__), protocol_sha256=protocol_digest,
        source_version_sha256=source_version(frozen), input_sha256={str(p): sha(p) for p in inputs+[data_path]},
        CSV_comparisons=comparisons, baseline_rows=baseline_rows, model_rows=models,
        length_rows=per_length, pair_rows=pairs, model_gate_pass_count=sum(r['gate_pass'] for r in models),
        pair_gate_pass_count=sum(r['gate_pass'] for r in pairs), architecture_accepted=accepted,
        overall_16_accepted=all(accepted.values()),
        scope=dict(read_stage_steps=[8000], decoded_npz_arrays=decoded, test_arrays_decoded=False,
            torch_imported=False, checkpoint_tensors_loaded=False, GPU_used=False, training_started=False,
            representation_metrics_read=False, earlier_checkpoints_selected=False,
            audit_status_is_arithmetic_consistency_not_learning_acceptance=True,
            validation_accuracies_from_saved_stage_records=True, independent_predictions_recomputed=False,
            independent_worlds=1, initializer_count_per_architecture=1,
            sixteen_models_are_not_independent_worlds=True, five_pp_is_development_gate_not_TOST=True))


def fixture():
    """Exercise full I/O against a synthetic reporter, including fail controls."""
    config = copy.deepcopy(read_json(DIAG/'protocol.json'))
    config['data']['examples_per_length'] = dict(train=100, validation=100, test=1)
    frozen = read_json(DIAG/'freeze.json')
    task_names = sorted({r[k] for r in tomllib.loads((WORKSPACE/'external/neurips/configs/property_task_geometry.toml').read_text())['relation_pairs']
                         for k in ('left', 'right')})
    checked = {}
    with tempfile.TemporaryDirectory(prefix='permworld-independent-learning-audit-') as temporary:
        root, diag = Path(temporary)/'sources', Path(temporary)/'reports'
        root.mkdir(); diag.mkdir()
        for path in (root/'protocol.json', diag/'protocol.json'):
            write_json(path, config)
        frozen['protocol_sha256'] = sha(diag/'protocol.json')
        write_json(diag/'freeze.json', frozen)
        write_json(root/'training_manifest.json', config['training_manifest'])
        world = root/'worlds'/config['units'][0]['unit']; world.mkdir(parents=True)
        lengths = np.repeat(config['lengths'], 100)
        train = np.ones((300, 1, 16), dtype=np.uint8)
        valid = np.ones((300, 3, 16), dtype=np.uint8)
        for block in range(3):
            train[100*block:100*block+50] = 0  # Tie: train mode must be zero.
            valid[100*block:100*block+60] = 0
        np.savez_compressed(world/'data.npz', train_lengths=lengths, train_labels=train,
            validation_lengths=lengths, validation_labels=valid,
            test_forbidden=np.array([{'must_never_decode': True}], dtype=object))
        data_digest = sha(world/'data.npz')
        write_json(world/'data_audit.json', dict(data_sha256=data_digest))
        for architecture in config['architectures']:
            directory = root/'initializations'/architecture; directory.mkdir(parents=True)
            write_json(directory/'step0.json', dict(architecture=architecture,
                protocol_sha256=frozen['protocol_sha256'], actual_initial_identity_verified=True,
                model_seed=config['architecture_model_seeds'][architecture],
                parameters=3240448 if architecture == 'transformer' else 1660416,
                parameter_sha256=('a' if architecture == 'transformer' else 'b')*64,
                checkpoint_sha256=('c' if architecture == 'transformer' else 'd')*64))
        for job in config['training_manifest']:
            directory = root/'models'/job['run_id']; directory.mkdir(parents=True)
            initial = read_json(root/'initializations'/job['architecture']/'step0.json')
            records = []
            for n, score in zip(config['lengths'], (.94, .95, .96)):
                records.append(dict(length=n, samples=100, full_vocab_accuracy=score,
                    answer_accuracy=score, numeric_accuracy=score, eos_accuracy=1.,
                    answer_ce=.5, eos_ce=.01, combined_ce=.255))
            validation = dict(split='validation', action='identity', samples=300, per_length=records,
                overall_full_vocab=.95, overall_numeric=.95, overall_eos=1., answer_accuracy=.95,
                numeric_accuracy=.95, eos_accuracy=1., answer_ce=.5, eos_ce=.01, combined_ce=.255,
                eos_teacher_forced=True, test_evaluated=False, geometry_evaluated=False)
            stage = dict(**job, step=8000, status='stage_complete', validation=validation,
                fixed_endpoint_step=8000, fixed_endpoint_updates_completed=True, fixed_endpoint_completed=True,
                source_model_run=True, actual_initial_identity_verified=True, validation_missing_due_to_budget=False,
                finite_gradient_updates=8000, exposures=512000, supervised_tokens=1024000,
                parameters=initial['parameters'], protocol_sha256=frozen['protocol_sha256'],
                execution_source_version_sha256=source_version(frozen), data_sha256=data_digest,
                initial_parameter_sha256=initial['parameter_sha256'], initial_checkpoint_sha256=initial['checkpoint_sha256'],
                starting_rng_hash='e'*64, checkpoint_sha256='f'*64, checkpoint_path=str(directory/'never_loaded.pt'),
                test_evaluated=False, geometry_evaluated=False, formal_training_started=False, curve=[],
                elapsed_seconds=1., invocation_start_epoch=100., invocation_end_epoch=101.)
            write_json(directory/'stage_8000.json', stage)
            write_json(directory/'completed.json', dict(stage, status='completed'))
        # Reporter used solely to produce synthetic CSVs; audit calculation does
        # not import or reuse its baseline, model-gate, or pair-gate functions.
        spec = importlib.util.spec_from_file_location('synthetic_frozen_learning_report',
            WORKSPACE/'experiments/permworld_learnability_20261010/report.py')
        reporter = importlib.util.module_from_spec(spec); spec.loader.exec_module(reporter)
        report = reporter.collect(root, diag, fixture=True)
        for name, rows in (('model_learning_summary.csv', report['summary_rows']),
                           ('pair_learning_gates.csv', report['pair_rows']),
                           ('validation_length.csv', report['length_rows']),
                           ('training_length_majority.csv', report['baseline_rows'])):
            with (diag/name).open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        result = audit(root, diag)
        checked['all16_and8pairs_match_frozen_synthetic_reporter'] = result['overall_16_accepted'] and len(result['pair_rows']) == 8
        checked['train_tie_uses_lowest_answer'] = all(r['majority_answer'] == 0 for r in result['baseline_rows'])
        checked['no_test_object_decoded'] = result['scope']['decoded_npz_arrays'] == list(LABEL_ARRAYS)
        sample = read_json(root/'models'/config['training_manifest'][0]['run_id']/'stage_8000.json')
        baseline, _ = independent_baselines(labels_only(world/'data.npz')[0], config, task_names)
        weak = copy.deepcopy(sample)
        weak['validation']['per_length'][-1].update(full_vocab_accuracy=.84, answer_accuracy=.84, numeric_accuracy=.84)
        weak['validation'].update(overall_full_vocab=.91, overall_numeric=.91, answer_accuracy=.91, numeric_accuracy=.91)
        got, _ = model_values(weak, baseline, config)
        checked['weak_length_fails_despite_mean_above90'] = not got['gate_pass'] and got['equal_length_accuracy'] > .9
        altered_items = []
        for job in config['training_manifest']:
            stage = read_json(root/'models'/job['run_id']/'stage_8000.json')
            if job['architecture'] == 'transformer' and job['task'] == 'descents' and job['encoding'] == 'canonical_cycle':
                for row, score in zip(stage['validation']['per_length'], (1., .89, .96)):
                    row.update(full_vocab_accuracy=score, answer_accuracy=score, numeric_accuracy=score)
                stage['validation'].update(overall_full_vocab=.95, overall_numeric=.95,
                                           answer_accuracy=.95, numeric_accuracy=.95)
            if job['architecture'] == 'transformer' and job['task'] == 'recoils':
                score = .9 if job['encoding'] == 'one_line' else .95
                for row in stage['validation']['per_length']:
                    row.update(full_vocab_accuracy=score, answer_accuracy=score, numeric_accuracy=score)
                stage['validation'].update(overall_full_vocab=score, overall_numeric=score,
                                           answer_accuracy=score, numeric_accuracy=score)
            value, _ = model_values(stage, baseline, config)
            altered_items.append((job, stage, value))
        altered_pairs = encoding_pairs(altered_items, config)
        length_gap = next(r for r in altered_pairs if r['architecture'] == 'transformer' and r['task'] == 'descents')
        checked['length_gap_rejected_even_when_means_match'] = (not length_gap['gate_pass'] and
            length_gap['mean_accuracy_gap_pp'] < NUMERICAL_TOLERANCE and length_gap['maximum_length_accuracy_gap_pp'] > 5)
        boundary = next(r for r in altered_pairs if r['architecture'] == 'transformer' and r['task'] == 'recoils')
        checked['exact_five_pp_development_boundary_allowed'] = boundary['gate_pass'] and abs(boundary['mean_accuracy_gap_pp']-5) < NUMERICAL_TOLERANCE
        path = diag/'model_learning_summary.csv'
        text = path.read_text()
        altered = csv_rows(path)
        altered[0]['equal_length_accuracy'] = '0.94'
        with path.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(altered[0]))
            writer.writeheader(); writer.writerows(altered)
        try:
            audit(root, diag)
        except AssertionError:
            checked['incorrect_report_number_rejected'] = True
        else:
            checked['incorrect_report_number_rejected'] = False
        path.write_text(text)
        endpoint = root/'models'/config['training_manifest'][-1]['run_id']/'completed.json'
        endpoint.unlink()
        try:
            preflight(root, diag, config)
        except AssertionError:
            checked['partial_manifest_rejected_before_labels'] = True
        else:
            checked['partial_manifest_rejected_before_labels'] = False
    require(all(checked.values()), 'Synthetic audit fixture failed: '+repr(checked))
    result = dict(status='passed', generated_utc=datetime.now(timezone.utc).isoformat(),
        auditor_sha256=sha(__file__), checks=checked, synthetic_CPU_only=True,
        real_model_results_read=False, real_train_or_validation_labels_read=False,
        test_arrays_decoded=False, GPU_used=False, training_started=False, temporary_sources_removed=True)
    write_json(DIAG/'independent_learning_audit_fixture.json', result)
    return result


def save_real(result):
    write_json(DIAG/'independent_learning_audit.json', result)
    text = [
        'Independent PermWorld learning-report arithmetic review:'+result['status'],
        '16 fixed 8k models and8 encoding pairs, with equal-three-length weighting; no earlier/best-checkpoint selection.',
        f'Models passing learning gates {result["model_gate_pass_count"]}/16; encoding pairs passing {result["pair_gate_pass_count"]}/8.',
        'Joint architecture qualification:'+json.dumps(result['architecture_accepted'], ensure_ascii=False),
        'Complete 16-condition qualification:'+str(result['overall_16_accepted']),
        'status=passed means four report tables match independent recomputation, not that all models pass learning gates.',
        'Training length modes use train identity labels only, with ties to smaller answers, then score validation.',
        'Answer accuracy comes from validation records saved at fixed checkpoints; no inference rerun or independent per-sample prediction recomputation.',
        'Only train/validation lengths/labels are decoded; no test arrays, checkpoint tensors, CKA, or GPU operations.',
        'One data world and one initialization per architecture;16 models are not16 independent worlds.5pp is a development gate, not TOST/equivalence proof.',
        'Gate rules were fixed before actual review; the script was implemented during training. This reviews report validity, not a new confirmation experiment.',
    ]
    for name, compared in result['CSV_comparisons'].items():
        text.append(f'{name}: {compared["rows"]} rows,{compared["fields_compared"]} fields match.')
    (DIAG/'independent_learning_audit.txt').write_text('\n'.join(text)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--fixture', action='store_true', help='Synthetic CPU audit only; no real endpoints or labels.')
    modes.add_argument('--run-real', action='store_true', help='Require all16 completed8k and CSVs; then audit read-only.')
    arguments = parser.parse_args()
    if arguments.fixture:
        result = fixture()
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        try:
            result = audit()
            save_real(result)
        except Exception as error:
            write_json(DIAG/'independent_learning_audit_failure.json', dict(status='failed',
                generated_utc=datetime.now(timezone.utc).isoformat(), auditor_sha256=sha(__file__),
                error_type=type(error).__name__, error=str(error), GPU_used=False, training_started=False))
            raise
        print(json.dumps({k: result[k] for k in ('status', 'model_gate_pass_count', 'pair_gate_pass_count',
                                               'architecture_accepted', 'overall_16_accepted')}, ensure_ascii=False))
