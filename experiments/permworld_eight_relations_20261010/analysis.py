"""Eight exact PermWorld relations: fixed cache-only CPU measurements.

--fixture is synthetic and safe before source training. --run requires the
protocol freeze and the completed 64-model cache. No training or inference.
All earlier two-relation scientific rows are reused rather than remeasured.
"""
from __future__ import annotations

import os
os.environ['OPENBLAS_NUM_THREADS'] = '2'
os.environ['OMP_NUM_THREADS'] = '2'

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.permworld_eight_relations_20261010.analysis_helpers import *

REVIEW = DIAG / 'metric_review'
LAYER_NAMES = {f'block{i}': f'block_{i:02d}' for i in range(1, 5)}
LAYER_NAMES.update({f'layer{i}': f'layer_{i:02d}' for i in range(1, 5)})
LAYER_NAMES['final_norm'] = 'final_norm'


class Inputs:
    """Hash-checked read-only mmap arrays and frozen shared anchor selections."""
    def __init__(self, protocol):
        self.p = protocol
        self.m = metrics(protocol)
        self.jobs = {(j['architecture'], j['encoding'], j['task']): j for j in protocol['checkpoint_inventory']}
        require(len(self.jobs) == 64, 'Expected 64 distinct models, not 64 independent replicates')
        require(sha(protocol['source_data_path']) == protocol['source_data_sha256'], 'Original fixed-world data changed')
        with np.load(protocol['source_data_path'], allow_pickle=False) as archive:
            self.data = {f'{split}_{name}': archive[f'{split}_{name}'].copy()
                         for split in ('validation', 'test') for name in ('lengths', 'labels', 'permutations')}
        self.names = list(protocol['label_task_names'])
        self.label_alignment = []
        for split in ('validation', 'test'):
            require(self.data[split + '_labels'].shape == (len(self.data[split + '_lengths']), 3, 16), 'Original label schema changed')
            for r in protocol['relations']:
                action = protocol['actions'].index(r['input_transform'])
                yl = self.data[split + '_labels'][:, 0, self.names.index(r['left'])]
                yr = self.data[split + '_labels'][:, action, self.names.index(r['right'])]
                require(np.array_equal(yl + int(r['right_label_offset']), yr), 'Exact relation/offset violated: ' + r['pair_id'])
                for n in protocol['lengths']:
                    use = self.data[split + '_lengths'] == n
                    for role, ai in [('correct', action), ('wrong', 1 if action == 2 else 2), ('identity', 0)]:
                        mapped = self.data[split + '_labels'][use, ai, self.names.index(r['right'])].astype(np.int64) - int(r['right_label_offset'])
                        matches = int((mapped == yl[use]).sum())
                        self.label_alignment.append(dict(relation=r['pair_id'], split=split, length=n, role=role,
                            samples=int(use.sum()), mapped_answer_matches=matches,
                            mapped_answer_match_rate=matches/int(use.sum()), right_label_offset=int(r['right_label_offset']),
                            wrong_accidental_matches_retained=True, no_model_information_used=True))
        require(sha(protocol['selection_path']) == protocol['selection_sha256'], 'Frozen common anchor selections changed')
        with np.load(protocol['selection_path'], allow_pickle=False) as archive:
            self.selection = {k: archive[k].copy() for k in archive.files}
        self.mask_cache, self.mmaps, self.hashed = {}, {}, set()
        root = Path(protocol.get('cache_root', protocol['root']))
        path = Path(protocol.get('evaluation_index_path', root / 'evaluation/index.json'))
        self.index_path = path
        self.index = js(path)
        require(self.index['status'] == 'completed', '64-model extraction is incomplete')
        require(self.index['protocol_sha256'] == sha(DIAG / 'protocol.json'), 'Cache protocol binding changed')
        require(self.index['source_data_sha256'] == protocol['source_data_sha256'], 'Cache is bound to a different world')
        require(self.index['initial_shared_exact'] is True, 'Unverified initial hidden sharing')
        self.records = {(r['run_id'], r['condition'], r['split']): r for r in self.index['records']}
        require(len(self.records) == 256 == len(self.index['records']), 'Logical cache inventory incomplete')
        for job in self.jobs.values():
            completed = js(job['completed_path'])
            require(int(completed['step']) == 20000, 'Model endpoint changed')
            for phase in PHASES:
                for split in ('validation', 'test'):
                    record = self.records[job['run_id'], phase, split]
                    for k in ('architecture', 'encoding', 'task'):
                        require(record[k] == job[k], 'Cache assigned to the wrong model: ' + k)
                    require(record['condition'] == phase and record['split'] == split, 'Cache role mismatch')
                    require(record['protocol_sha256'] == sha(DIAG / 'protocol.json') and
                            record['source_data_sha256'] == protocol['source_data_sha256'], 'Logical record binding changed')
                    expected_path = job['initial_checkpoint_path'] if phase == 'initial' else job['checkpoint_path']
                    expected_sha = job['initial_checkpoint_sha256'] if phase == 'initial' else completed['checkpoint_sha256']
                    require(Path(record['source_checkpoint_path']).resolve() == Path(expected_path).resolve() and
                            record['source_checkpoint_sha256'] == expected_sha, 'Cache bound to a different checkpoint')
                    for name, description in record['arrays'].items():
                        require(Path(description['path']).is_file(), 'Missing mmap: ' + description['path'])
                        if description['path'] not in self.hashed:
                            require(sha(description['path']) == description['sha256'], 'Mmap bytes changed: ' + description['path'])
                            self.hashed.add(description['path'])
        self.old_diag = Path(protocol.get('old_analysis_dir', OLD_DIAG))
        self.old_tables = {}
        for filename in ('per_length_results.csv', 'cross_arch_same_task.csv', 'gold_baseline_length.csv',
                         'source_learning.csv', 'source_validation_eos_original.csv'):
            path = self.old_diag / filename
            require(path.is_file(), 'Original proven table missing: ' + str(path))
            expected = protocol.get('old_table_sha256', {}).get(filename)
            require(expected is not None and sha(path) == expected, 'Original table must be bound in protocol: ' + filename)
            self.old_tables[filename] = (rows(path), sha(path))

    def record(self, architecture, encoding, task, phase, split):
        return self.records[self.jobs[architecture, encoding, task]['run_id'], phase, split]

    def load(self, record, name):
        metadata = record['arrays'][name]
        path = metadata['path']
        if path not in self.mmaps:
            array = np.load(path, mmap_mode='r', allow_pickle=False)
            require(list(array.shape) == metadata['shape'] and str(array.dtype) == metadata['dtype'], 'Mmap schema mismatch')
            require(not array.flags.writeable, 'Scientific source mmap must be read-only')
            self.mmaps[path] = array
        return self.mmaps[path]

    def feature(self, a, e, t, phase, split, layer, branch):
        record = self.record(a, e, t, phase, split)
        if branch == 'output_prob':
            array = self.load(record, 'output_prob')
            require(array.shape == (len(self.data[split + '_lengths']), 3, 31), 'Output baseline dimension must stay numeric31')
            return array
        if layer == 'final_norm':
            # The earlier packed hook path is only tolerance-equal to the
            # original NPZ prefix_final. Exact-score reuse requires original
            # prefix_final bytes; never silently substitute packed norm.
            array = self.load(record, 'prefix_final')
            require(array.shape == (len(self.data[split + '_lengths']), 3, 256), 'Final endpoint schema changed')
            return array
        packed = self.load(record, 'layers')
        order = record['arrays']['layers']['layer_order']
        require(packed.shape == (len(self.data[split + '_lengths']), 3, 5, 256), 'All four layers and separate norm required')
        name = LAYER_NAMES[layer]
        return packed[:, :, order.index(name), :]

    def mask(self, relation, control, n, split):
        key = relation['pair_id'], control, n, split
        if key not in self.mask_cache:
            prefix = f'{relation["pair_id"]}_{control}_{n}_{split}'
            ids, keys = selections(self.data, relation, split, n, control, self.names)
            require(np.array_equal(ids, self.selection[prefix + '_indices']), 'Frozen retained anchors differ from the original >=3 rule')
            saved = self.selection.get(prefix + '_keys')
            require((keys is None and saved is None) or np.array_equal(keys, saved), 'Frozen affine answer keys differ')
            self.mask_cache[key] = ids, keys
        return self.mask_cache[key]

    def pair_arrays(self, a, e, relation, n, control, layer, branch):
        fit, fk = self.mask(relation, control, n, 'validation')
        evaluation, ek = self.mask(relation, control, n, 'test')
        correct = self.p['actions'].index(relation['input_transform'])
        roles = dict(correct=correct, wrong=1 if correct == 2 else 2, identity=0)
        centered, hashes = {}, {}
        for phase in PHASES:
            centered[phase], hashes[phase] = {}, {}
            for role, action in roles.items():
                centered[phase][role], hashes[phase][role] = {}, {}
                for split, ids, keys in [('validation', fit, fk), ('test', evaluation, ek)]:
                    x = self.feature(a, e, relation['left'], phase, split, layer, branch)[ids, 0]
                    y = self.feature(a, e, relation['right'], phase, split, layer, branch)[ids, action]
                    require(np.isfinite(x).all() and np.isfinite(y).all(), 'Nonfinite raw scientific activations')
                    centered[phase][role][split] = self.m.center(x, keys), self.m.center(y, keys)
                    hashes[phase][role][split] = dict(left=array_sha(x), right=array_sha(y), row_ids=array_sha(ids),
                                                    keys='none' if keys is None else array_sha(keys))
        return centered, hashes, len(fit), len(evaluation)


def inherit_rows(inputs):
    original, digest = inputs.old_tables['per_length_results.csv']
    require(len(original) == 6048, 'Original complete two-relation analysis missing')
    out, checked, selection_hashes, key_audits = [], set(), {}, []
    relations = {r['pair_id']: r for r in inputs.p['relations']}
    for saved in original:
        require(saved['relation'] in OLD_RELATIONS, 'Old scope changed')
        key = tuple(saved[k] for k in ('architecture', 'encoding', 'relation', 'control', 'layer', 'branch')) + (int(saved['length']),)
        # Every reused metric score retains its original raw arrays and rows.
        if key not in checked and saved['layer'] != 'embedding':
            a, e, r, c, l, b, n = key
            _, hashes, nf, ne = inputs.pair_arrays(a, e, relations[r], n, c, l, b)
            comparison_hashes = json.loads(json.dumps(hashes))
            selection_hashes[key] = {}
            # Old joint keys inherited int16 from the length array; the new
            # explicit affine mapping uses int64. Coordinate dtype changes
            # must be audited rather than mistaken for scientific changes.
            for split in ('validation', 'test'):
                ids, current_keys = inputs.mask(relations[r], c, n, split)
                if current_keys is None:
                    selection_hashes[key][split] = 'none'
                    continue
                raw = inputs.data[split + '_labels']
                old_keys = np.column_stack((inputs.data[split + '_lengths'],
                    raw[:, 0, inputs.names.index(relations[r]['left'])],
                    raw[:, :, inputs.names.index(relations[r]['right'])]))[ids]
                require(int(relations[r]['right_label_offset']) == 0 and np.array_equal(old_keys, current_keys),
                        'Old relation label key values changed; exact scientific reuse no longer valid')
                old_groups = np.unique(old_keys, axis=0, return_inverse=True)[1]
                new_groups = np.unique(current_keys, axis=0, return_inverse=True)[1]
                require(np.array_equal(old_groups, new_groups), 'Key coordinate conversion changed the group partition')
                old_hash, current_hash = array_sha(old_keys), array_sha(current_keys)
                selection_hashes[key][split] = current_hash
                for phase in PHASES:
                    for role in ROLES:
                        comparison_hashes[phase][role][split]['keys'] = old_hash
                # Same raw arrays, same sorted group assignments, same mean
                # summation order: this is exact centering invariance, not a
                # fresh CKA/neighbor/PCA scientific computation.
                x = inputs.feature(a, e, relations[r]['left'], 'initial', split, l, b)[ids, 0]
                require(np.array_equal(inputs.m.center(x, old_keys), inputs.m.center(x, current_keys)),
                        'Affine-safe integer dtype conversion changed centered features')
                key_audits.append(dict(architecture=a, encoding=e, relation=r, control=c, layer=l, branch=b,
                    length=n, split=split, original_key_dtype=str(old_keys.dtype), current_key_dtype=str(current_keys.dtype),
                    original_key_sha256=old_hash, current_key_sha256=current_hash,
                    values_equal=True, partitions_equal=True, initial_left_centered_values_bitwise_equal=True,
                    original_CSV_key_hash_preserved=True, no_scientific_metric_recomputed=True))
            require(comparison_hashes == json.loads(saved['raw_feature_array_hashes']), 'Reused activations/anchors/original keys changed')
            require(nf == int(saved['fit_samples']) and ne == int(saved['test_samples']), 'Reused sample counts changed')
            checked.add(key)
        row = dict(saved)
        row.update(inherited_original_values=True, source_table_path=str(inputs.old_diag / 'per_length_results.csv'),
                   source_table_sha256=digest, reused_analysis_scope='Original fixed20k two-relation study; scientific numbers unchanged')
        row['current_selection_keys_sha256'] = json.dumps(selection_hashes.get(key, {}), sort_keys=True)
        out.append(row)
    csv_write(DIAG / 'reused_key_coordinate_audit.csv', key_audits)
    return out


def measured_rows(inputs):
    out = []
    for a in inputs.p['architectures']:
        layers = [f'block{i}' if a == 'transformer' else f'layer{i}' for i in range(1, 5)]
        for e in inputs.p['encodings']:
            for relation in inputs.p['relations']:
                if relation['pair_id'] in OLD_RELATIONS:
                    continue
                for n in inputs.p['lengths']:
                    for control in inputs.p['controls']:
                        ids, _ = inputs.mask(relation, control, n, 'test')
                        duplicates = inputs.data['test_permutations'][ids]
                        branches = [('final_norm', 'hidden', ALL_METRICS), ('output', 'output_prob', ALL_METRICS)]
                        branches += [(layer, 'hidden', LAYER_METRICS) for layer in layers]
                        for layer, branch, metric_names in branches:
                            arrays, hashes, nf, ne = inputs.pair_arrays(a, e, relation, n, control, layer, branch)
                            measured = {m: {p: {} for p in PHASES} for m in metric_names}
                            undefined = {m: {} for m in metric_names}
                            for phase in PHASES:
                                for role in ROLES:
                                    xf, yf = arrays[phase][role]['validation']
                                    xe, ye = arrays[phase][role]['test']
                                    if metric_names == ALL_METRICS:
                                        scores, rank = inputs.m.scores(xf, yf, xe, ye, ids=ids, duplicate_keys=duplicates)
                                        require(rank == min(8, xf.shape[1], yf.shape[1]), 'Fixed Procrustes rank changed')
                                    else:
                                        scores = {m: (inputs.m.cka(xe, ye, debiased=m == 'debiased_cka'), None) for m in metric_names}
                                    for m in metric_names:
                                        value = scores[m][0]
                                        measured[m][phase][role] = number(value)
                                        if value is None:
                                            undefined[m][phase + '_' + role] = undefined_reason(m, xf, yf, xe, ye, ids, duplicates, inputs.m)
                            for metric_name in metric_names:
                                values = measured[metric_name]
                                error = {f'error_{p}_{r}': -values[p][r] if metric_name == 'procrustes' else float('nan')
                                         for p in PHASES for r in ROLES}
                                out.append(dict(architecture=a, encoding=e, relation=relation['pair_id'], control=control,
                                    layer=layer, cache_layer_name=LAYER_NAMES.get(layer, layer), branch=branch, metric=metric_name,
                                    pooling=inputs.p.get('pooling', 'input_end_before_task_at2n+3'), length=n,
                                    fit_samples=nf, test_samples=ne, **component_values(values), **error,
                                    nan_reason=json.dumps(undefined[metric_name], sort_keys=True),
                                    inherited_original_values=False, source_table_path='', source_table_sha256='',
                                    raw_feature_array_hashes=json.dumps(hashes, sort_keys=True),
                                    right_label_offset=int(relation['right_label_offset']),
                                    metric_preprocessing='Original frozen within-split/length common answer>=3 centering; fixed metric source unmodified'))
                        for m in LAYER_METRICS:
                            blank = {p: {r: float('nan') for r in ROLES} for p in PHASES}
                            out.append(dict(architecture=a, encoding=e, relation=relation['pair_id'], control=control,
                                layer='embedding', cache_layer_name='embedding', branch='hidden', metric=m,
                                pooling=inputs.p.get('pooling', 'input_end_before_task_at2n+3'), length=n,
                                fit_samples=len(inputs.mask(relation, control, n, 'validation')[0]), test_samples=len(ids),
                                **component_values(blank), **{f: float('nan') for f in ERROR_FIELDS},
                                nan_reason=json.dumps({p+'_'+r: 'structurally_constant_end_token_embedding_within_length'
                                    for p in PHASES for r in ROLES}, sort_keys=True), inherited_original_values=False,
                                source_table_path='', source_table_sha256='', raw_feature_array_hashes='not_extracted_structural_constant',
                                right_label_offset=int(relation['right_label_offset']),
                                metric_preprocessing='Constant end token plus fixed position; floating residuals are not geometry'))
                print(json.dumps(dict(phase='relation_complete', architecture=a, encoding=e,
                                      relation=relation['pair_id'], new_rows=len(out))), flush=True)
    require(len(out) == 18144, 'Six-relation row coverage incomplete')
    return out


def cross_arch_rows(inputs):
    old, digest = inputs.old_tables['cross_arch_same_task.csv']
    require(len(old) == 672, 'Original same-task cross-configuration table incomplete')
    result = [dict(r, inherited_original_values=True, source_table_sha256=digest) for r in old]
    old_tasks = {r['task'] for r in old}
    require(len(old_tasks) == 4, 'Old task set changed')
    for task in inputs.names:
        if task in old_tasks:
            continue
        column = inputs.names.index(task)
        for encoding in inputs.p['encodings']:
            for n in inputs.p['lengths']:
                for control in inputs.p['controls']:
                    selected = {}
                    for split in ('validation', 'test'):
                        ns = inputs.data[split + '_lengths']
                        labels = inputs.data[split + '_labels'][:, :, column]
                        keys = np.column_stack((ns, labels))
                        use = ns == n
                        if control == 'answer_strata':
                            _, group, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
                            use &= counts[group] >= 3
                        selected[split] = np.flatnonzero(use), keys[use] if control == 'answer_strata' else None
                    ids, keys = selected['test']
                    values, hashes, undefined = {}, {}, {}
                    for phase in PHASES:
                        x = inputs.feature('transformer', encoding, task, phase, 'test', 'final_norm', 'hidden')[ids, 0]
                        y = inputs.feature('gru', encoding, task, phase, 'test', 'final_norm', 'hidden')[ids, 0]
                        values[phase] = inputs.m.center(x, keys), inputs.m.center(y, keys)
                        hashes[phase] = dict(left=array_sha(x), right=array_sha(y), row_ids=array_sha(ids),
                                             keys='none' if keys is None else array_sha(keys))
                    for metric_name in LAYER_METRICS:
                        scores, undefined = {}, {}
                        for phase in PHASES:
                            x, y = values[phase]
                            score = inputs.m.cka(x, y, debiased=metric_name == 'debiased_cka')
                            scores[phase] = number(score)
                            if score is None:
                                undefined[phase] = undefined_reason(metric_name, x, y, x, y, ids, None, inputs.m)
                        result.append(dict(architecture_left='transformer', architecture_right='gru', encoding=encoding,
                            task=task, layer='final_norm', pooling=inputs.p.get('pooling', 'input_end_before_task_at2n+3'),
                            pairing='identity--identity', control=control, metric=metric_name, length=n,
                            fit_samples=len(selected['validation'][0]), test_samples=len(ids),
                            initial_similarity=scores['initial'], trained_similarity=scores['trained'],
                            similarity_change=scores['trained']-scores['initial'], nan_reason=json.dumps(undefined, sort_keys=True),
                            raw_feature_array_hashes=json.dumps(hashes, sort_keys=True), inherited_original_values=False,
                            centering_key='[n,same_task identity,complement,inverse]; >=3 per split; distinct from relation joint key',
                            interpretation='32 direct same-task configuration pairs; not correct/wrong relations or pure architecture causality'))
    require(len(result) == 2688, 'Complete 32-pair cross-architecture table missing')
    return result


def gold_rows(inputs):
    old, digest = inputs.old_tables['gold_baseline_length.csv']
    require(len(old) == 8064, 'Original static answer table incomplete')
    result = [dict(r, inherited_original_values=True, source_table_sha256=digest) for r in old]
    for relation in inputs.p['relations']:
        if relation['pair_id'] in OLD_RELATIONS:
            continue
        left_col, right_col = inputs.names.index(relation['left']), inputs.names.index(relation['right'])
        correct = inputs.p['actions'].index(relation['input_transform'])
        for n in inputs.p['lengths']:
            for control in inputs.p['controls']:
                fit, fk = inputs.mask(relation, control, n, 'validation')
                test, tk = inputs.mask(relation, control, n, 'test')
                duplicates = inputs.data['test_permutations'][test]
                # Static true answers do not depend on architecture/encoding.
                # Compute once, retain all original coordinate labels explicitly.
                for baseline in ('true_answer_onehot', 'true_answer_scalar'):
                    for role, action in [('correct', correct), ('wrong', 1 if correct == 2 else 2), ('identity', 0)]:
                        arrays = []
                        for split, ids, keys in [('validation', fit, fk), ('test', test, tk)]:
                            labels = inputs.data[split + '_labels']
                            yl = labels[ids, 0, left_col]
                            yr = labels[ids, action, right_col]
                            x = np.eye(31)[yl] if baseline.endswith('onehot') else yl[:, None]
                            y = np.eye(31)[yr] if baseline.endswith('onehot') else yr[:, None]
                            arrays.extend((inputs.m.center(x, keys), inputs.m.center(y, keys)))
                        values, rank = inputs.m.scores(*arrays, ids=test, duplicate_keys=duplicates)
                        for a in inputs.p['architectures']:
                            for e in inputs.p['encodings']:
                                for name, (score, error) in values.items():
                                    reason = '' if score is not None else undefined_reason(name, *arrays, test, duplicates, inputs.m)
                                    result.append(dict(architecture=a, encoding=e, relation=relation['pair_id'], length=n,
                                        control=control, baseline=baseline, role=role, metric=name,
                                        value=number(score), error=number(error), fit_samples=len(fit), test_samples=len(test),
                                        static_no_training_delta=True, right_label_offset=int(relation['right_label_offset']),
                                        answer_mapping='raw true labels retained; offset applied only to grouping coordinates',
                                        nan_reason=reason, inherited_original_values=False))
    require(len(result) == 32256, 'Eight-relation static gold coverage incomplete')
    return result


def source_learning_rows(inputs):
    """All fixed endpoint source accuracies, CE, and train-defined modes.

    Old numeric rows are exact copies. A missing initial CE remains missing;
    initial hidden sharing never implies shared task-specific output scores.
    """
    old, old_hash = inputs.old_tables['source_learning.csv']
    require(len(old) == 4032, 'Original source-learning table incomplete')
    result = [dict(r, inherited_original_values=True, source_table_sha256=old_hash) for r in old]
    old_jobs = {r['run_id'] for r in old}
    old_eos, old_eos_hash = inputs.old_tables['source_validation_eos_original.csv']
    eos_records = [dict(r, inherited_original_values=True, source_table_sha256=old_eos_hash) for r in old_eos]
    with np.load(inputs.p['source_data_path'], allow_pickle=False) as archive:
        train_ns = archive['train_lengths'].copy()
        train_labels = archive['train_labels'][:, 0, :].copy()
    modes = {}
    for task in inputs.names:
        column = inputs.names.index(task)
        for n in inputs.p['lengths']:
            counts = np.bincount(train_labels[train_ns == n, column].astype(np.int64), minlength=31)
            require(counts.sum() == 4096, 'Train-defined mode does not use the original4096/n anchors')
            modes[task, n] = int(counts.argmax())  # smallest numeric label wins ties
    for job in inputs.jobs.values():
        if job['run_id'] in old_jobs:
            continue
        a, e, task = job['architecture'], job['encoding'], job['task']
        column = inputs.names.index(task)
        completion = js(job['completed_path'])
        curve = completion['curve'][-1]
        require(int(curve['step']) == 20000, 'Source endpoint must stay fixed20k')
        for phase in PHASES:
            for split in ('validation', 'test'):
                record = inputs.record(a, e, task, phase, split)
                pred = inputs.load(record, 'prediction')
                ce = inputs.load(record, 'answer_ce') if 'answer_ce' in record['arrays'] else None
                ns = inputs.data[split + '_lengths']
                labels = inputs.data[split + '_labels'][:, :, column]
                require(pred.shape == labels.shape, 'Full-vocabulary prediction schema changed')
                if ce is not None:
                    require(ce.shape == labels.shape and np.isfinite(ce).all(), 'Full-vocabulary source CE is invalid')
                for n in inputs.p['lengths']:
                    use = ns == n
                    count = int(use.sum())
                    require(count == (256 if split == 'validation' else 512), 'Source evaluation anchor count changed')
                    for action, name in enumerate(inputs.p['actions']):
                        acc = int((pred[use, action] == labels[use, action]).sum()) / count
                        mode = modes[task, n]
                        baseline = int((labels[use, action] == mode).sum()) / count
                        loss = math.fsum(map(float, ce[use, action])) / count if ce is not None else float('nan')
                        result.append(dict(run_id=job['run_id'], architecture=a, encoding=e, task=task,
                            condition=phase, split=split, action=name, length=n, samples=count,
                            answer_accuracy=acc, answer_ce=loss,
                            answer_ce_reason='' if ce is not None else 'No saved task-specific initial CE; not filled from another task',
                            length_majority_accuracy=baseline, accuracy_minus_length_mode=acc-baseline,
                            training_majority_label=mode, majority_source='original identity train labels4096/n; smallest-label tie',
                            eos_accuracy=float('nan'), eos_reason='Prompt-only source evaluation; original teacher-forced EOS separate',
                            evaluated_token='answer at2n+5; all188 vocabulary argmax and CE',
                            parameters=job['parameters'], inherited_original_values=False))
                if phase == 'trained' and split == 'validation':
                    acc = float(np.mean(pred[:, 0] == labels[:, 0]))
                    eos_records.append(dict(run_id=job['run_id'], new_original_validation_answer_accuracy=acc,
                        old_validation_answer_accuracy=curve['answer_accuracy'],
                        signed_accuracy_difference_from_teacher_forced_old_curve=acc-curve['answer_accuracy'],
                        old_curve_accuracy_exactly_reproduced=acc == curve['answer_accuracy'],
                        old_validation_teacher_forced_eos_accuracy=curve['eos_accuracy'],
                        old_validation_samples=curve['samples'], eos_not_newly_evaluated=True,
                        inherited_original_values=False))
    require(len(result) == 16128 and len(eos_records) == 64, 'Complete source-learning coverage missing')
    groups = defaultdict(list)
    for row in result:
        groups[tuple(row[k] for k in ('architecture', 'encoding', 'task', 'condition', 'split', 'action'))].append(row)
    summary = []
    for key, group in groups.items():
        require(sorted(int(row['length']) for row in group) == inputs.p['lengths'], 'Source learning has incomplete lengths')
        out = dict(zip(('architecture', 'encoding', 'task', 'condition', 'split', 'action'), key))
        out.update(length_count=21, weights='fixed1/21; undefined CE remains undefined')
        for field in ('answer_accuracy', 'answer_ce', 'length_majority_accuracy', 'accuracy_minus_length_mode'):
            out[field] = equal_length([r[field] for r in group])
        out['minimum_length_accuracy'] = min(number(r['answer_accuracy']) for r in group)
        summary.append(out)
    return result, summary, eos_records


def independent_checks(inputs):
    independent = module(OLD_DIAG / 'metric_review/independent_metrics.py', 'eight_relations_independent_metrics')
    rule = inputs.p['validation']['metric_fixture']
    require(rule['architecture'] == 'transformer' and rule['encoding'] == 'one_line' and rule['relation'] == 'descent_inverse'
            and rule['length'] == 10 and rule['fit_rows'] == rule['eval_rows'] == 64, 'Original fixed fixture changed')
    tolerance = float(rule.get('metric_abs_tolerance', 1e-8))
    require(tolerance == 1e-8, 'Independent numerical tolerance changed')
    # Keep the original48, then add a fixed affine-offset relation64-row check.
    records = []
    for relation_id in ('descent_inverse', 'decomposition_complement'):
        relation = next(r for r in inputs.p['relations'] if r['pair_id'] == relation_id)
        for control in inputs.p['controls']:
            arrays, hashes, nf, ne = inputs.pair_arrays('transformer', 'one_line', relation, 10, control, 'final_norm', 'hidden')
            if nf < 64 or ne < 64:
                # Preserve the fixed fixture rather than resizing sparse strata.
                require(relation_id != 'descent_inverse', 'Original48 fixture no longer has64 fixed rows')
                records.append(dict(relation=relation_id, control=control, status='unavailable_fixed64',
                                    fit_samples=nf, eval_samples=ne, no_adaptive_row_reduction=True))
                continue
            ids = inputs.mask(relation, control, 10, 'test')[0][:64]
            duplicates = inputs.data['test_permutations'][ids]
            for phase in PHASES:
                for role in ROLES:
                    xf, yf = (x[:64] for x in arrays[phase][role]['validation'])
                    xe, ye = (x[:64] for x in arrays[phase][role]['test'])
                    primary, rank = inputs.m.scores(xf, yf, xe, ye, ids=ids, duplicate_keys=duplicates)
                    error, alternative_rank = independent.independent_procrustes(xf, yf, xe, ye)
                    require(rank == alternative_rank == 8, 'Fixed independent PCA rank changed')
                    other = dict(linear_cka=independent.covariance_cka(xe, ye),
                        debiased_cka=independent.independent_u_cka(xe, ye),
                        knn_overlap=independent.brute_overlap(xe, ye, ids=ids, duplicate_keys=duplicates),
                        procrustes=-error if error is not None else None)
                    for name in ALL_METRICS:
                        first, second = primary[name][0], other[name]
                        require((first is None) == (second is None), 'Independent undefined classification differs')
                        difference = abs(first - second) if first is not None else None
                        require(difference is None or difference <= tolerance, 'Independent metric failed: ' + name)
                        records.append(dict(relation=relation_id, control=control, condition=phase, role=role, metric=name,
                            primary=first, independent=second, absolute_error=difference, passed=True,
                            preprocessing='Frozen full selected split centering, then first64 retained rows',
                            eval_anchor_sha256=array_sha(ids), fit_samples=nf, eval_samples=ne,
                            fit_left_sha256=array_sha(xf), fit_right_sha256=array_sha(yf),
                            eval_left_sha256=array_sha(xe), eval_right_sha256=array_sha(ye)))
    original = [r for r in records if r['relation'] == 'descent_inverse']
    require(len(original) == 48 and all(r.get('passed') for r in original), 'Original48 independent fixture incomplete')
    result = dict(status='passed', records=records, original_fixture_checks=48,
        affine_offset_fixture_fixed_before_new_geometry=True,
        maximum_absolute_error=max((r.get('absolute_error') or 0) for r in records),
        independent_source_sha256=sha(OLD_DIAG / 'metric_review/independent_metrics.py'),
        no_scientific_scores_replaced=True, no_training=True)
    write_json(REVIEW / 'independent_metric_actual.json', result)
    return result


def run():
    start = time.monotonic()
    protocol = verify_protocol()
    inputs = Inputs(protocol)
    independent = independent_checks(inputs)
    inherited = inherit_rows(inputs)
    new = measured_rows(inputs)
    all_rows = inherited + new
    require(len(all_rows) == 24192, 'Eight-relation table coverage incomplete')
    summaries, index = summarize(all_rows, protocol['lengths'])
    require(len(index) == 1152, 'Expected 1152 full21 groups')
    cross = cross_arch_rows(inputs)
    gold = gold_rows(inputs)
    learning, learning_summary, eos = source_learning_rows(inputs)
    csv_write(DIAG / 'per_length_results.csv', all_rows)
    csv_write(DIAG / 'contrast_summary.csv', summaries)
    csv_write(DIAG / 'cross_arch_same_task.csv', cross)
    csv_write(DIAG / 'cross_arch_same_task_summary.csv', cross_summary(cross, protocol['lengths']))
    csv_write(DIAG / 'gold_baseline_length.csv', gold)
    csv_write(DIAG / 'source_learning.csv', learning)
    csv_write(DIAG / 'source_learning_summary.csv', learning_summary)
    csv_write(DIAG / 'source_validation_eos_original.csv', eos)
    csv_write(DIAG / 'relation_label_alignment.csv', inputs.label_alignment)
    audit = dict(status='passed', protocol_sha256=sha(DIAG / 'protocol.json'),
        elapsed_cpu_wall_seconds=time.monotonic()-start, per_length_rows=len(all_rows), original_rows_reused=len(inherited),
        new_rows=len(new), full21_groups=len(index), same_task_cross_arch_rows=len(cross), cross_arch_pairs=32,
        gold_rows=len(gold), source_learning_rows=len(learning), all_original_scientific_values_reused=True,
        fixed21_no_missing_reweighting=True, independent_metric_max_error=independent['maximum_absolute_error'],
        mmap_file_hashes_verified=len(inputs.hashed), original_NPZ_arrays_decoded_by_analysis=0,
        original_key_coordinate_audit_rows=len(rows(DIAG / 'reused_key_coordinate_audit.csv')),
        one_data_world=True, source_models=64, source_models_not_independent_replicates=True,
        relation_instances=8, exact_transform_types=2, no_training=True, no_F17_changes=True,
        source_data_sha256=protocol['source_data_sha256'], evaluation_index_sha256=sha(inputs.index_path))
    write_json(REVIEW / 'analysis_audit.json', audit)
    print(json.dumps(audit), flush=True)


def fixture():
    """Synthetic numeric, offset-grouping, constant, and aggregation checks."""
    original = metrics()
    independent = module(OLD_DIAG / 'metric_review/independent_metrics.py', 'eight_relation_fixture_independent')
    fixed = independent.run_fixture()
    rng = np.random.default_rng(2026101081)
    names = ['left', 'right']
    labels = rng.integers(0, 4, size=(96, 3, 2))
    ns = np.full(96, 10)
    r0 = dict(left='left', right='right', right_label_offset=0)
    r1 = dict(left='left', right='right', right_label_offset=1)
    data = dict(validation_lengths=ns, validation_labels=labels)
    ids0, keys0 = selections(data, r0, 'validation', 10, 'answer_strata', names)
    ids1, keys1 = selections(data, r1, 'validation', 10, 'answer_strata', names)
    require(np.array_equal(ids0, ids1), 'Affine offset changed retained group counts')
    require(np.array_equal(keys0[:, :2], keys1[:, :2]) and np.array_equal(keys0[:, 2:] - 1, keys1[:, 2:]), 'Offset direction changed')
    x = rng.normal(size=(len(ids0), 8))
    require(np.array_equal(original.center(x, keys0), original.center(x, keys1)), 'Bijective affine grouping changed centering')
    require(math.isnan(equal_length([1.0]*20+[float('nan')])), 'Undefined length was reweighted')
    require(equal_length([1.0]*21) == 1.0, 'Equal length arithmetic failed')
    blank = component_values({p: {r: float('nan') for r in ROLES} for p in PHASES})
    require(all(math.isnan(blank[k]) for k in FIELDS), 'Structural constant should remain undefined')
    result = dict(status='passed', original_independent_fixture=fixed,
        offset_plus1_maps_all_right_roles=True, offset_bijective_groups_unchanged=True,
        group_centered_values_bitwise_equal=True, all21_required_no_NaN_reweight=True,
        constant_all_components_undefined=True, no_real_geometry_inputs_loaded=True,
        no_training=True, analysis_sha256=sha(__file__), helpers_sha256=sha(Path(__file__).with_name('analysis_helpers.py')))
    REVIEW.mkdir(parents=True, exist_ok=True)
    write_json(REVIEW / 'synthetic_fixture.json', result)
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument('--fixture', action='store_true')
    choice.add_argument('--run', action='store_true')
    args = parser.parse_args()
    fixture() if args.fixture else run()


if __name__ == '__main__':
    main()
