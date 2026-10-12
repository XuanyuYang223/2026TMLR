"""Fixed20k controlled comparison: cache-only CPU analysis, never training.

--benchmark uses only predetermined synthetic arrays. --run requires the root
protocol freeze and complete extraction index. Final-layer/output measures are
the original CSV values; only four middle layers and the isolated same-task
cross-architecture comparison receive new CKA/U-centered calculations.
"""
from __future__ import annotations

import os
os.environ['OPENBLAS_NUM_THREADS'] = '2'
os.environ['OMP_NUM_THREADS'] = '2'

import argparse
from collections import OrderedDict
import csv
import importlib.util
import json
import math
from pathlib import Path
import shutil
import sys
import time

import numpy as np


HERE = Path(__file__).resolve().parent
DIAG = HERE.parent
WORKSPACE = DIAG.parents[1]
sys.path.insert(0, str(DIAG))
import common as fixed

CONDITIONS = ('initial', 'trained')
ROLES = ('correct', 'wrong', 'identity')
COMPONENTS = ('initial_correct', 'initial_wrong', 'initial_identity',
              'trained_correct', 'trained_wrong', 'trained_identity',
              'B0', 'Bt', 'dc', 'dw', 'di', 'Delta')
LAYER_TO_CACHE = {'block'+str(i): f'block_{i:02d}' for i in range(1, 5)}
LAYER_TO_CACHE.update({'layer'+str(i): f'layer_{i:02d}' for i in range(1, 5)})
LAYER_TO_CACHE.update(final_norm='final_norm', embedding='embedding')


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def number(value):
    return fixed.number(value)


def original_metrics():
    return fixed.metrics


def components(values):
    result = {condition+'_'+role: number(values[condition][role])
              for condition in CONDITIONS for role in ROLES}
    result['B0'] = result['initial_correct']-result['initial_wrong']
    result['Bt'] = result['trained_correct']-result['trained_wrong']
    for role, name in (('correct', 'dc'), ('wrong', 'dw'), ('identity', 'di')):
        result[name] = result['trained_'+role]-result['initial_'+role]
    result['Delta'] = result['Bt']-result['B0']
    if all(math.isfinite(result[field]) for field in ('Delta', 'dc', 'dw')):
        require(abs(result['Delta']-(result['dc']-result['dw'])) <= 1e-10, 'Delta decomposition failed')
    result.update(change_correct=result['dc'], change_wrong=result['dw'], change_identity=result['di'])
    return result


def inherited_components(row):
    result = {field: number(row[field]) for field in COMPONENTS
              if field not in ('dc', 'dw', 'di')}
    for name, source in (('dc', 'change_correct'), ('dw', 'change_wrong'), ('di', 'change_identity')):
        result[name] = number(row[source])
        result[source] = result[name]
    for left, right in (('B0', result['initial_correct']-result['initial_wrong']),
                        ('Bt', result['trained_correct']-result['trained_wrong']),
                        ('Delta', result['Bt']-result['B0']),
                        ('Delta', result['dc']-result['dw'])):
        if math.isfinite(result[left]) and math.isfinite(right):
            require(abs(result[left]-right) <= 1e-10, 'Inherited arithmetic mismatch: '+left)
    return result


def errors(values, metric):
    return {'error_'+condition+'_'+role:
                -number(values[condition][role]) if metric == 'procrustes' else float('nan')
            for condition in CONDITIONS for role in ROLES}


def criterion_undefined(metric, x, y):
    if len(x) < (4 if metric == 'debiased_cka' else 2):
        return 'insufficient_evaluation_samples_for_'+metric
    if np.square(x).sum() <= 1e-20 or np.square(y).sum() <= 1e-20:
        return 'zero_centered_feature_energy'
    return ('zero_U_centered_Gram_norm' if metric == 'debiased_cka'
            else 'linear_Gram_norm_product_at_most_1e-20')


class Inputs:
    """Fixed existing anchors, original masks, and hash-checked read-only caches."""
    def __init__(self, protocol, load_layers=True):
        self.protocol = protocol
        self.metrics = original_metrics()
        self.jobs = {(j['architecture'], j['encoding'], j['task']): j
                     for j in protocol['checkpoint_inventory']}
        require(len(self.jobs) == 16, 'Fixed source model crossing incomplete')
        require(fixed.sha(protocol['source_data_path']) == protocol['source_data_sha256'], 'Source data hash mismatch')
        with np.load(protocol['source_data_path'], allow_pickle=False) as archive:
            names = [f'{split}_{name}' for split in ('validation', 'test')
                     for name in ('lengths', 'labels', 'permutations')]
            self.data = {name: archive[name].copy() for name in names}
        self.source_arrays_decoded = names
        require(fixed.sha(protocol['selection_path']) == protocol['selection_sha256'], 'Fixed selection archive hash mismatch')
        with np.load(protocol['selection_path'], allow_pickle=False) as archive:
            self.selections = {name: archive[name].copy() for name in archive.files}
        self.selection_index = {(row['relation'], row['control'], int(row['length']), row['split']): row
                                for row in fixed.rows(DIAG/'selection_index.csv')}
        require(len(self.selection_index) == 168, 'Fixed relation selection coverage incomplete')
        self.old = {(row['architecture'], row['encoding'], row['task'], row['condition'], row['split']): row
                    for row in protocol['source_cache_inventory']}
        require(len(self.old) == 64, 'Original final cache coverage incomplete')
        self.old_arrays = OrderedDict()
        self.old_hashed = set()
        self.packed = {}
        self.selected_hashes = {}
        self.old_geometry_path = Path(protocol['source_root'])/'geometry_length.csv'
        self.old_geometry_sha256 = fixed.sha(self.old_geometry_path)
        self.old_gold_path = Path(protocol['source_root'])/'gold_baseline_length.csv'
        self.old_gold_sha256 = fixed.sha(self.old_gold_path)
        self.layer_index_path = Path(protocol['cache_root'])/'layers/index.json'
        self.mask_checks = 0
        if not load_layers:
            return
        require(self.layer_index_path.is_file(), 'Complete new layer extraction index missing')
        self.layer_index = fixed.js(self.layer_index_path)
        require(self.layer_index['status'] == 'completed', 'Layer extraction has not completed')
        require(self.layer_index['protocol_sha256'] == fixed.sha(DIAG/'protocol.json'), 'Layer index protocol binding mismatch')
        self.records = {row['source_key']: row for row in self.layer_index['records']}
        require(len(self.records) == len(self.layer_index['records']), 'Repeated physical layer source key')
        self.layer_map = self.layer_index['job_condition_split_map']
        expected = {j['run_id']+'|'+condition+'|'+split for j in protocol['checkpoint_inventory']
                    for condition in CONDITIONS for split in ('validation', 'test')}
        require(set(self.layer_map) == expected and set(self.layer_map.values()) == set(self.records),
                'Layer logical/physical source mapping incomplete')
        for row in self.records.values():
            feature_path = Path(row['features_path'])
            require(feature_path.is_file() and fixed.sha(feature_path) == row['features_sha256'],
                    'New layer feature file hash mismatch: '+row['source_key'])
            require(float(row['max_final_norm_difference']) <= protocol['validation']['activation_abs_tolerance'],
                    'New path disagrees with existing final activation cache')
            require(row['features_shape'] == [len(self.data[row['split']+'_lengths']), 3, 5, 256],
                    'Packed layer feature shape metadata mismatch')
        self.new_features_hashed = len(self.records)

    def mask(self, relation, control, n, split):
        name = f'{relation["pair_id"]}_{control}_{n}_{split}'
        indices = self.selections[name+'_indices']
        keys = self.selections.get(name+'_keys')
        saved = self.selection_index[relation['pair_id'], control, n, split]
        require(fixed.array_sha(indices) == saved['indices_sha256'], 'Selected anchor IDs changed')
        require(('none' if keys is None else fixed.array_sha(keys)) == saved['keys_sha256'],
                'Selected strata keys changed')
        original, original_keys = fixed.selections(self.data, relation, split, n, control)
        require(np.array_equal(indices, np.flatnonzero(original)), 'Frozen selection differs from original rule')
        require((keys is None and original_keys is None) or np.array_equal(keys, original_keys),
                'Frozen centering keys differ from original rule')
        self.mask_checks += 1
        return indices, keys

    def old_array(self, architecture, encoding, task, condition, split, branch):
        row = self.old[architecture, encoding, task, condition, split]
        source_key = 'prefix_final' if branch == 'hidden' else 'output_prob'
        key = row['path'], source_key
        if key in self.old_arrays:
            self.old_arrays.move_to_end(key)
            return self.old_arrays[key]
        if row['path'] not in self.old_hashed:
            require(fixed.sha(row['path']) == row['sha256'], 'Original cache raw hash mismatch')
            self.old_hashed.add(row['path'])
        with np.load(row['path'], allow_pickle=False) as archive:
            result = archive[source_key].copy()
        expected = (len(self.data[split+'_lengths']), 3, 256 if branch == 'hidden' else 31)
        require(result.shape == expected and result.dtype == np.float32 and np.isfinite(result).all(),
                'Original raw feature array schema mismatch')
        self.old_arrays[key] = result
        while len(self.old_arrays) > 12:
            self.old_arrays.popitem(last=False)
        return result

    def layer_array(self, architecture, encoding, task, condition, split, layer):
        job = self.jobs[architecture, encoding, task]
        logical = job['run_id']+'|'+condition+'|'+split
        physical = self.layer_map[logical]
        row = self.records[physical]
        require(row['architecture'] == architecture and row['encoding'] == encoding and
                row['condition'] == condition and row['split'] == split,
                'Mapped layer record mismatches logical model context')
        if condition == 'trained':
            require(row['task'] == task, 'Trained layer source belongs to another task')
        elif row['task'] != task:
            require(bool(self.layer_index['initial_shared_exact']), 'Unverified initial hidden sharing')
        if physical not in self.packed:
            features = np.load(row['features_path'], allow_pickle=False, mmap_mode='r')
            require(features.shape == tuple(row['features_shape']) and features.dtype == np.float32,
                    'New packed array dtype/shape mismatch')
            self.packed[physical] = features
        cache_layer = LAYER_TO_CACHE[layer]
        require(cache_layer in row['layer_order'], 'Requested layer missing from packed order')
        return self.packed[physical][:, :, row['layer_order'].index(cache_layer), :]

    def feature(self, architecture, encoding, task, condition, split, layer, branch):
        if layer in ('final_norm', 'output'):
            return self.old_array(architecture, encoding, task, condition, split, branch)
        return self.layer_array(architecture, encoding, task, condition, split, layer)

    def relation_arrays(self, architecture, encoding, relation, n, control, layer, branch):
        fit, fit_keys = self.mask(relation, control, n, 'validation')
        evaluation, evaluation_keys = self.mask(relation, control, n, 'test')
        correct = self.protocol['actions'].index(relation['input_transform'])
        wrong = 1 if correct == 2 else 2
        roles = {'correct': correct, 'wrong': wrong, 'identity': 0}
        arrays, hashes = {}, {}
        for condition in CONDITIONS:
            arrays[condition], hashes[condition] = {}, {}
            for role, action in roles.items():
                values, digests = {}, {}
                for split, ids, keys in (('validation', fit, fit_keys), ('test', evaluation, evaluation_keys)):
                    x = self.feature(architecture, encoding, relation['left'], condition, split, layer, branch)[ids, 0]
                    y = self.feature(architecture, encoding, relation['right'], condition, split, layer, branch)[ids, action]
                    require(np.isfinite(x).all() and np.isfinite(y).all(), 'Nonfinite selected raw features')
                    values[split] = (self.metrics.center(x, keys), self.metrics.center(y, keys))
                    digests[split] = dict(left=fixed.array_sha(x), right=fixed.array_sha(y),
                                         row_ids=fixed.array_sha(ids), keys='none' if keys is None else fixed.array_sha(keys))
                arrays[condition][role] = values
                hashes[condition][role] = digests
        return arrays, hashes, len(fit), len(evaluation)


def final_rows(inputs, protocol):
    original = fixed.rows(inputs.old_geometry_path)
    require(len(original) == 2688, 'Original final geometry table incomplete')
    expected = {(a, e, r['pair_id'], n, c, b, metric)
                for a in protocol['architectures'] for e in protocol['encodings']
                for r in protocol['relations'] for n in protocol['lengths'] for c in protocol['controls']
                for b in ('prefix_final', 'output_prob')
                for metric in ('linear_cka', 'debiased_cka', 'knn_overlap', 'procrustes')}
    observed = [(row['architecture'], row['encoding'], row['relation'], int(row['length']), row['control'],
                 row['branch'], row['metric']) for row in original]
    require(len(set(observed)) == len(original) and set(observed) == expected, 'Original final table row coverage differs')
    relations = {r['pair_id']: r for r in protocol['relations']}
    selected_cache, result = {}, []
    for old in original:
        a, e, r = old['architecture'], old['encoding'], old['relation']
        n, control, metric = int(old['length']), old['control'], old['metric']
        branch = 'hidden' if old['branch'] == 'prefix_final' else 'output_prob'
        layer = 'final_norm' if branch == 'hidden' else 'output'
        cache_key = a, e, r, n, control, layer
        if cache_key not in selected_cache:
            selected_cache[cache_key] = inputs.relation_arrays(a, e, relations[r], n, control, layer, branch)
            # Keep only one centered feature group in memory; retain raw hashes
            # separately when later metric rows revisit this exact cell.
        arrays, hashes, fit_samples, test_samples = selected_cache[cache_key]
        require(fit_samples == int(old['fit_samples']) and test_samples == int(old['test_samples']),
                'Original final table does not use frozen common selected anchors')
        values = {condition: {role: number(old[condition+'_'+role]) for role in ROLES} for condition in CONDITIONS}
        undefined = {}
        for condition in CONDITIONS:
            for role in ROLES:
                value = values[condition][role]
                if not math.isfinite(value):
                    if metric == 'procrustes':
                        xf, yf = arrays[condition][role]['validation']
                        xe, ye = arrays[condition][role]['test']
                        rank = min(8, xf.shape[1], yf.shape[1])
                        if len(xf) <= rank or len(yf) <= rank:
                            reason = 'fit_samples_not_greater_than_fixed_rank'
                        elif inputs.metrics.pca_fit(xf, rank) is None or inputs.metrics.pca_fit(yf, rank) is None:
                            reason = 'fit_energy_or_relative_1e-6_rank_below_fixed_rank'
                        elif len(xe) < 2:
                            reason = 'insufficient_evaluation_samples_for_procrustes'
                        else:
                            reason = 'zero_right_projected_evaluation_energy_under_original_rule'
                    else:
                        xe, ye = arrays[condition][role]['test']
                        reason = criterion_undefined(metric, xe, ye)
                    undefined[condition+'_'+role] = reason
        row = dict(architecture=a, encoding=e, relation=r, layer=layer, cache_layer_name=layer,
            pooling=protocol['pooling'], control=control, branch=branch, metric=metric, length=n,
            fit_samples=fit_samples, test_samples=test_samples, **inherited_components(old), **errors(values, metric),
            nan_reason=json.dumps(undefined, sort_keys=True), inherited_original_values=True,
            source_table_path=str(inputs.old_geometry_path), source_table_sha256=inputs.old_geometry_sha256,
            raw_feature_array_hashes=json.dumps(hashes, sort_keys=True),
            metric_preprocessing='own split/length centering; common >=3 answer strata when requested; '+
                ('fit-only side PCA/rank8/RMS/map, target-energy NRMSE' if metric == 'procrustes' else
                 'Euclidean k10 stable anchor-ID ties/no row normalization/self+duplicate exclusion' if metric == 'knn_overlap' else
                 'U-centered offdiagonal Grams sensitivity' if metric == 'debiased_cka' else 'ordinary centered linear Grams'))
        for condition in CONDITIONS:
            original_error = number(old[condition+'_correct_error'])
            restored = row['error_'+condition+'_correct']
            if math.isfinite(original_error) or math.isfinite(restored):
                require(math.isfinite(original_error) and math.isfinite(restored) and abs(original_error-restored) <= 1e-12,
                        'Inherited Procrustes score/error sign mismatch')
        result.append(row)
        # Original CSV has four consecutive metric rows per raw feature cell.
        if metric == 'procrustes':
            selected_cache.pop(cache_key)
    require(not selected_cache, 'Unexpected inherited metric row ordering left feature groups unclosed')
    return result


def middle_rows(inputs, protocol):
    result = []
    for architecture in protocol['architectures']:
        layers = [layer for layer in protocol['layer_rules'][architecture] if layer not in ('final_norm', 'embedding')]
        require(len(layers) == 4, 'All four middle layers must be retained')
        for encoding in protocol['encodings']:
            for relation in protocol['relations']:
                for n in protocol['lengths']:
                    for control in protocol['controls']:
                        fit, _ = inputs.mask(relation, control, n, 'validation')
                        evaluation, _ = inputs.mask(relation, control, n, 'test')
                        for layer in layers:
                            arrays, hashes, fit_samples, test_samples = inputs.relation_arrays(
                                architecture, encoding, relation, n, control, layer, 'hidden')
                            for metric in protocol['metrics']['new_layer_metrics']:
                                require(metric in ('linear_cka', 'debiased_cka'), 'No new layer metric search')
                                values, undefined = {}, {}
                                for condition in CONDITIONS:
                                    values[condition] = {}
                                    for role in ROLES:
                                        x, y = arrays[condition][role]['test']
                                        score = inputs.metrics.cka(x, y, debiased=metric == 'debiased_cka')
                                        values[condition][role] = number(score)
                                        if score is None:
                                            undefined[condition+'_'+role] = criterion_undefined(metric, x, y)
                                result.append(dict(architecture=architecture, encoding=encoding,
                                    relation=relation['pair_id'], layer=layer, cache_layer_name=LAYER_TO_CACHE[layer],
                                    pooling=protocol['pooling'], control=control, branch='hidden', metric=metric,
                                    length=n, fit_samples=fit_samples, test_samples=test_samples,
                                    **components(values), **errors(values, metric), nan_reason=json.dumps(undefined, sort_keys=True),
                                    inherited_original_values=False, source_table_path='', source_table_sha256='',
                                    raw_feature_array_hashes=json.dumps(hashes, sort_keys=True),
                                    metric_preprocessing='own split/length centering; common >=3 answer strata when requested; '+
                                        ('U-centered offdiagonal Grams sensitivity' if metric == 'debiased_cka' else 'ordinary centered linear Grams')))
                        for metric in protocol['metrics']['new_layer_metrics']:
                            undefined = {condition+'_'+role: 'structurally_constant_end_token_embedding_within_length'
                                         for condition in CONDITIONS for role in ROLES}
                            values = {condition: {role: float('nan') for role in ROLES} for condition in CONDITIONS}
                            result.append(dict(architecture=architecture, encoding=encoding,
                                relation=relation['pair_id'], layer='embedding', cache_layer_name='embedding',
                                pooling=protocol['pooling'], control=control, branch='hidden', metric=metric,
                                length=n, fit_samples=len(fit), test_samples=len(evaluation),
                                **components(values), **errors(values, metric), nan_reason=json.dumps(undefined, sort_keys=True),
                                inherited_original_values=False, source_table_path='', source_table_sha256='',
                                raw_feature_array_hashes='not_extracted_structural_constant',
                                metric_preprocessing='undefined by constant end-token plus fixed-position embedding; floating residuals not treated as geometry'))
                print(json.dumps(dict(phase='middle_layers', architecture=architecture, encoding=encoding,
                                      relation=relation['pair_id'], rows=len(result))), flush=True)
    require(len(result) == 3360, 'Middle/constant layer output coverage incomplete')
    return result


def cross_arch_rows(inputs, protocol):
    result = []
    metrics = protocol['cross_arch_same_task']['metrics']
    require(metrics == ['linear_cka', 'debiased_cka'], 'Cross-architecture metric scope changed')
    for task in protocol['tasks']:
        column = fixed.TASKS.index(task)
        for encoding in protocol['encodings']:
            for n in protocol['lengths']:
                for control in protocol['cross_arch_same_task']['controls']:
                    selected = {}
                    for split in ('validation', 'test'):
                        ns, labels = inputs.data[split+'_lengths'], inputs.data[split+'_labels']
                        all_keys = np.column_stack((ns, labels[:, :, column]))
                        use = ns == n
                        if control == 'answer_strata':
                            _, group, counts = np.unique(all_keys, axis=0, return_inverse=True, return_counts=True)
                            use &= counts[group] >= 3
                            keys = all_keys[use]
                        else:
                            keys = None
                        selected[split] = np.flatnonzero(use), keys
                    ids, keys = selected['test']
                    arrays, hashes = {}, {}
                    for condition in CONDITIONS:
                        x = inputs.old_array('transformer', encoding, task, condition, 'test', 'hidden')[ids, 0]
                        y = inputs.old_array('gru', encoding, task, condition, 'test', 'hidden')[ids, 0]
                        hashes[condition] = dict(left=fixed.array_sha(x), right=fixed.array_sha(y),
                            row_ids=fixed.array_sha(ids), keys='none' if keys is None else fixed.array_sha(keys))
                        arrays[condition] = inputs.metrics.center(x, keys), inputs.metrics.center(y, keys)
                    for metric in metrics:
                        scores, undefined = {}, {}
                        for condition in CONDITIONS:
                            x, y = arrays[condition]
                            value = inputs.metrics.cka(x, y, debiased=metric == 'debiased_cka')
                            scores[condition] = number(value)
                            if value is None:
                                undefined[condition] = criterion_undefined(metric, x, y)
                        result.append(dict(architecture_left='transformer', architecture_right='gru',
                            encoding=encoding, task=task, layer='final_norm', pooling=protocol['pooling'],
                            pairing='identity--identity', control=control, metric=metric, length=n,
                            fit_samples=len(selected['validation'][0]), test_samples=len(ids),
                            initial_similarity=scores['initial'], trained_similarity=scores['trained'],
                            similarity_change=scores['trained']-scores['initial'],
                            nan_reason=json.dumps(undefined, sort_keys=True),
                            raw_feature_array_hashes=json.dumps(hashes, sort_keys=True),
                            centering_key='[n,same_task identity,complement,inverse]; >=3 per split; differs from relation-task joint key',
                            interpretation='Direct same-task cross-configuration CKA; not mathematical correct/wrong pairing or pure architecture effect'))
    require(len(result) == 672, 'Cross-architecture same-task coverage incomplete')
    return result


def length_coverage(rows):
    groups = {}
    for row in rows:
        key = tuple(row[field] for field in ('architecture', 'encoding', 'relation', 'layer', 'control', 'branch', 'metric'))
        groups.setdefault(key, []).append(row)
    incomplete, undefined = [], []
    for key, values in groups.items():
        require(sorted(row['length'] for row in values) == list(range(10, 31)), 'Incomplete/duplicate 21-length group '+repr(key))
        for field in COMPONENTS:
            defined = [float(row[field]) for row in values]
            aggregate = fixed.equal_length(defined)
            if any(not math.isfinite(v) for v in defined):
                require(math.isnan(aggregate), 'Incomplete length group was silently renormalized')
                undefined.append(dict(group=list(key), field=field,
                                      undefined_lengths=[row['length'] for row in values if not math.isfinite(float(row[field]))]))
    return dict(groups=len(groups), all_complete_21=True, incomplete_groups=incomplete,
                undefined_full_length_aggregates=undefined, no_missing_length_reweighting=True)


def independent_module():
    spec = importlib.util.spec_from_file_location('controlled_fixed_independent_metrics', HERE/'independent_metrics.py')
    independent = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(independent)
    return independent


def actual_independent_check(inputs, protocol):
    """Predetermined64-row numerical check, not a newly selected endpoint."""
    rule = protocol['validation']['metric_fixture']
    for key, expected in {'architecture': 'transformer', 'encoding': 'one_line',
                          'relation': 'descent_inverse', 'length': 10,
                          'fit_rows': 64, 'eval_rows': 64}.items():
        require(rule[key] == expected, 'Actual fixed metric fixture changed: '+key)
    controls = rule.get('controls', protocol['controls'])
    require(controls == ['raw_within_length', 'answer_strata'], 'Actual fixture controls incomplete')
    tolerance = float(rule.get('metric_abs_tolerance', protocol['validation']['metric_abs_tolerance']))
    require(tolerance == 1e-8, 'Actual independent metric tolerance changed')
    independent = independent_module()
    relation = next(r for r in protocol['relations'] if r['pair_id'] == rule['relation'])
    records = []
    for control in controls:
        arrays, hashes, fit_samples, test_samples = inputs.relation_arrays(
            rule['architecture'], rule['encoding'], relation, rule['length'], control, 'final_norm', 'hidden')
        require(fit_samples >= 64 and test_samples >= 64, 'Fixed actual64-row fixture insufficient; do not reduce rows')
        eval_ids, _ = inputs.mask(relation, control, rule['length'], 'test')
        ids = eval_ids[:64]
        duplicates = inputs.data['test_permutations'][ids]
        for condition in CONDITIONS:
            for role in ROLES:
                full = arrays[condition][role]
                xf, yf = (x[:64] for x in full['validation'])
                xe, ye = (x[:64] for x in full['test'])
                primary, rank = inputs.metrics.scores(xf, yf, xe, ye, ids=ids, duplicate_keys=duplicates)
                proc_error, independent_rank = independent.independent_procrustes(xf, yf, xe, ye)
                require(rank == independent_rank == 8, 'Actual fixture rank differs from fixed8')
                alternative = dict(linear_cka=independent.covariance_cka(xe, ye),
                    debiased_cka=independent.independent_u_cka(xe, ye),
                    knn_overlap=independent.brute_overlap(xe, ye, ids=ids, duplicate_keys=duplicates),
                    procrustes=-proc_error if proc_error is not None else None)
                for metric in ('linear_cka', 'debiased_cka', 'knn_overlap', 'procrustes'):
                    original, other = primary[metric][0], alternative[metric]
                    if original is None or other is None:
                        require(original is None and other is None, 'Actual independent undefined status differs')
                        error = None
                    else:
                        error = abs(original-other)
                        require(error <= tolerance, 'Actual independent metric check failed: '+json.dumps(
                            dict(control=control, condition=condition, role=role, metric=metric,
                                 primary=original, independent=other, absolute_error=error, tolerance=tolerance)))
                    records.append(dict(control=control, condition=condition, role=role, metric=metric,
                        fit_rows=64, eval_rows=64, preprocessing='full frozen selected split centering first, then first64 rows',
                        eval_anchor_ids_sha256=fixed.array_sha(ids), duplicate_keys_sha256=fixed.array_sha(duplicates),
                        centered_fit_left_sha256=fixed.array_sha(xf), centered_fit_right_sha256=fixed.array_sha(yf),
                        centered_eval_left_sha256=fixed.array_sha(xe), centered_eval_right_sha256=fixed.array_sha(ye),
                        original=original, independent=other, absolute_error=error, tolerance=tolerance, passed=True))
    require(len(records) == 48, 'Actual independent numerical fixture coverage incomplete')
    result = dict(status='passed', fixed_rule=rule, controls=controls, records=records,
        checks_count=48, maximum_absolute_error=max(row['absolute_error'] or 0. for row in records),
        error_metrics_report_negative_NRMSE=True, subset_chosen_before_new_analysis=True,
        no_endpoint_values_replaced=True, no_new_inputs=True, new_training=False,
        independent_source_sha256=fixed.sha(HERE/'independent_metrics.py'))
    fixed.json_write(HERE/'independent_metric_actual.json', result)
    return result


def run():
    started = time.monotonic()
    protocol = fixed.verify()
    require(protocol['lengths'] == list(range(10, 31)) and protocol['no_new_training'] and protocol['no_new_inputs'],
            'Fixed existing20k scope changed')
    inputs = Inputs(protocol)
    print(json.dumps(dict(phase='preflight_passed', physical_layer_records=len(inputs.records),
                          original_final_values_reused=True)), flush=True)
    independent_actual = actual_independent_check(inputs, protocol)
    inherited = final_rows(inputs, protocol)
    extra = middle_rows(inputs, protocol)
    combined = inherited+extra
    coverage = length_coverage(combined)
    same_task = cross_arch_rows(inputs, protocol)
    fixed.csv_write(DIAG/'per_length_results.csv', combined)
    fixed.csv_write(DIAG/'cross_arch_same_task.csv', same_task)
    gold_target = DIAG/'gold_baseline_length.csv'
    shutil.copyfile(inputs.old_gold_path, gold_target)
    require(fixed.sha(gold_target) == inputs.old_gold_sha256, 'Static gold copy differs from original bytes')
    require(fixed.sha(inputs.old_geometry_path) == inputs.old_geometry_sha256 and
            fixed.sha(inputs.old_gold_path) == inputs.old_gold_sha256, 'Original metric table changed during analysis')
    require(len(inputs.old_hashed) == 64, 'Not every original logical cache raw hash checked')
    audit = dict(status='passed', elapsed_cpu_wall_seconds=time.monotonic()-started,
        original_final_rows=len(inherited), new_layer_rows=len(extra), total_per_length_rows=len(combined),
        cross_arch_same_task_rows=len(same_task), static_gold_copy_sha256=inputs.old_gold_sha256,
        original_geometry_table_sha256=inputs.old_geometry_sha256, all_original_final_values_reused=True,
        procrustes_all_role_errors_restored_as_negative_original_scores=True,
        original_cache_files_raw_sha256_checked=len(inputs.old_hashed),
        new_feature_files_raw_sha256_checked=inputs.new_features_hashed,
        raw_features_same_across_final_metric_variants=True, source_selected_value_hashes_saved=True,
        frozen_common_mask_checks=inputs.mask_checks, complete_length_coverage=coverage,
        source_arrays_decoded=inputs.source_arrays_decoded, new_test_created=False, new_training=False,
        old_results_modified=False, F17_modified=False, optimizer_constructed=False,
        pooling=protocol['pooling'], primary_layer='final_norm',
        other_layers_only_CKA_and_U_centered=True, embedding_all_structural_undefined=True,
        cross_arch_same_task_grouping_separately_specified=True,
        actual_independent_metric_checks=independent_actual['checks_count'],
        actual_independent_metric_maximum_absolute_error=independent_actual['maximum_absolute_error'],
        actual_independent_metric_audit_path=str(HERE/'independent_metric_actual.json'),
        actual_independent_metric_audit_sha256=fixed.sha(HERE/'independent_metric_actual.json'),
        metrics_source_sha256=fixed.sha(WORKSPACE/'diagnostics/permworld_scope_inventory_20261009/metrics.py'),
        protocol_sha256=fixed.sha(DIAG/'protocol.json'), layer_index_sha256=fixed.sha(inputs.layer_index_path),
        output_sha256={name: fixed.sha(DIAG/name) for name in
                      ('per_length_results.csv', 'cross_arch_same_task.csv', 'gold_baseline_length.csv')},
        exploratory=True, independent_worlds=1, confirmation_pvalues=False)
    fixed.json_write(HERE/'analysis_audit.json', audit)
    print(json.dumps({key:audit[key] for key in ('status', 'elapsed_cpu_wall_seconds', 'total_per_length_rows',
                                              'cross_arch_same_task_rows', 'original_cache_files_raw_sha256_checked')}), flush=True)


def benchmark():
    independent = independent_module()
    fixture = independent.run_fixture()
    require(fixture['status'] == 'passed', 'Independent fixed fixture failed')
    rng = np.random.default_rng(202610104112)
    left = [rng.normal(size=(512, 256)) for _ in range(4)]
    right = [[rng.normal(size=(512, 256)) for _ in range(3)] for _ in range(4)]
    started = time.monotonic()
    calls = 0
    for state in range(4):
        for role in range(3):
            for debiased in (False, True):
                original_metrics().cka(left[state], right[state][role], debiased=debiased)
                calls += 1
    seconds = time.monotonic()-started
    planned = 4*2*21*2*4*2*6 + 4*2*21*2*2*2
    result = dict(status='passed', synthetic_only=True, fixed_seed=202610104112,
        benchmark_samples=512, benchmark_features=256, calls=calls, seconds=seconds,
        seconds_per_CKA_call=seconds/calls, planned_new_CKA_calls=planned,
        estimated_metric_seconds=seconds/calls*planned,
        estimate_excludes_cache_IO_hashing_feature_extraction_and_final_reuse_bookkeeping=True,
        final_layer_original_metrics_reused_not_recomputed=True,
        old_result_values_loaded=False, new_result_values_computed=False,
        cpu_threads=2, independent_fixture_checks=fixture['checks_count'],
        three_metric_fixed_fixture_status=fixture['status'])
    fixed.json_write(HERE/'benchmark.json', result)
    print(json.dumps(result), flush=True)


def benchmark_real():
    """Frozen n10/20/30 cost timing only; no metric-based protocol selection."""
    protocol = fixed.verify()
    inputs = Inputs(protocol, load_layers=False)
    relation = next(r for r in protocol['relations'] if r['pair_id'] == 'descent_inverse')
    records = []
    for n in (10, 20, 30):
        for control in protocol['controls']:
            arrays, _, fit_samples, test_samples = inputs.relation_arrays(
                'transformer', 'one_line', relation, n, control, 'final_norm', 'hidden')
            for metric in ('linear_cka', 'debiased_cka'):
                seconds = []
                for condition in CONDITIONS:
                    for role in ROLES:
                        x, y = arrays[condition][role]['test']
                        start = time.monotonic()
                        inputs.metrics.cka(x, y, debiased=metric == 'debiased_cka')
                        seconds.append(time.monotonic()-start)
                records.append(dict(length=n, control=control, metric=metric, fit_samples=fit_samples,
                    test_samples=test_samples, fixed_pair_calls=6, pair_seconds=seconds,
                    median_pair_seconds=float(np.median(seconds))))
    estimate = 0.
    for control in protocol['controls']:
        for metric in ('linear_cka', 'debiased_cka'):
            rate = math.fsum(row['median_pair_seconds'] for row in records
                             if row['control'] == control and row['metric'] == metric)/3
            estimate += rate*4368  # Four middles+isolated cross-architecture;6/2 state scores.
    result = dict(status='passed', cost_only=True, fixed_architecture='transformer', fixed_encoding='one_line',
        fixed_relation='descent_inverse', fixed_lengths=[10,20,30], controls=protocol['controls'], records=records,
        planned_new_CKA_calls=17472, estimated_CPU_metric_seconds=estimate,
        estimate_excludes_IO_hashing_bookkeeping_and_GPU_forward=True,
        source_old_cache_hashes_checked=len(inputs.old_hashed), protocol_sha256=fixed.sha(DIAG/'protocol.json'),
        final_scores_recomputed_for_timing_only_not_replaced=True, no_selection_by_scores=True,
        new_inputs=False, new_training=False, optimizer_constructed=False, cached_existing_test_only=True)
    fixed.json_write(HERE/'real_cost_benchmark.json', result)
    print(json.dumps({key:result[key] for key in ('status','cost_only','planned_new_CKA_calls','estimated_CPU_metric_seconds',
                                               'source_old_cache_hashes_checked')}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--benchmark', action='store_true')
    mode.add_argument('--run', action='store_true')
    mode.add_argument('--benchmark-real', action='store_true')
    args = parser.parse_args()
    if args.benchmark:
        benchmark()
    elif args.benchmark_real:
        benchmark_real()
    else:
        run()
