"""Frozen, cache-only fixed-transform search; ordinary final-layer CKA only.

The 492 candidate correspondences and their three mathematical categories are
read from the protocol-bound manifest. No model, Torch, forward pass, training,
new anchors, or alternative metric is used. Execution requires a new freeze.
"""
from __future__ import annotations

import os
os.environ['OPENBLAS_NUM_THREADS'] = '2'
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'

import argparse
from collections import defaultdict
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
from threadpoolctl import threadpool_limits

DIAG = Path(__file__).resolve().parent
METRIC_SHA = 'c2a384a8212e661d1917c084a74699297bbef354f06dac42ac75ea90ce3d0b5c'
CATEGORIES = ('designated', 'other_known_exact', 'not_designated')
PHASES = ('initial', 'trained')
PAIR_FIELDS = ('domain', 'relation', 'architecture', 'encoding', 'left_task',
               'right_task', 'transform', 'category', 'designated_right_task',
               'pair_id', 'original_right_label_mapping')


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(2 ** 20), b''):
            h.update(chunk)
    return h.hexdigest()


def array_sha(a):
    a = np.ascontiguousarray(a)
    h = hashlib.sha256()
    h.update(str(a.dtype).encode())
    h.update(str(a.shape).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def csv_read(path):
    with Path(path).open(newline='') as stream:
        return list(csv.DictReader(stream))


def csv_write(path, records):
    require(bool(records), 'Empty table requires an explicit status, not an empty CSV')
    fields = list(dict.fromkeys(k for r in records for k in r))
    with Path(path).open('w', newline='') as stream:
        w = csv.DictWriter(stream, fieldnames=fields)
        w.writeheader()
        w.writerows(records)


def json_write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def number(value):
    if value in (None, '', 'None', 'nan', 'NaN'):
        return None
    x = float(value)
    require(math.isfinite(x), 'Nonfinite numeric value must be explicitly undefined')
    return x


def fixed_mean(values, lengths):
    require(len(values) == len(lengths), 'Incomplete fixed length grid')
    return math.fsum(values) / len(lengths) if all(v is not None for v in values) else None


def finite_stats(values):
    a = [float(v) for v in values if v is not None]
    a.sort()
    median = None if not a else a[len(a)//2] if len(a) % 2 else (a[len(a)//2-1] + a[len(a)//2]) / 2
    return dict(defined_count=len(a), total_count=len(values),
                finite_mean=math.fsum(a)/len(a) if a else None,
                finite_median=median)


def equal_optional(a, b, tolerance):
    return (a is None) == (b is None) and (a is None or abs(a-b) <= tolerance)


def frozen_inputs(diag):
    p = read(diag / 'protocol.json')
    f = read(diag / 'freeze.json')
    require(f['protocol_sha256'] == sha(diag / 'protocol.json'), 'New protocol is not frozen')
    require(f['source_sha256'].get(str(Path(__file__).resolve())) == sha(__file__), 'This source is not frozen')
    for path, expected in f['source_sha256'].items():
        require(sha(path) == expected, 'Frozen source changed: ' + path)
    require(p['cpu_threads'] == 2, 'CPU thread count must remain two')
    require(p['existing_tolerance'] == 1e-8, 'Original endpoint tolerance is fixed at 1e-8')
    metric_path = Path(p['metric_source_path'])
    require(sha(metric_path) == p['metric_source_sha256'] == METRIC_SHA, 'Original ordinary-CKA source changed')
    spec = importlib.util.spec_from_file_location('pairing_original_frozen_metrics', metric_path)
    metric = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(metric)
    require(hasattr(metric, 'center') and hasattr(metric, 'cka'), 'Original metric interface unavailable')
    pair_path = Path(p['pair_manifest_path'])
    require(sha(pair_path) == p['pair_manifest_sha256'], 'Fixed candidate classifications changed')
    pairs = csv_read(pair_path)
    require(len(pairs) == 492, 'Expected 480 PermWorld and 12 GraphWorld candidate pairs')
    require(len({r['pair_id'] for r in pairs}) == 492, 'Pair IDs must be globally unique')
    require(all(all(k in r for k in PAIR_FIELDS) for r in pairs), 'Manifest field missing')
    require(all(r['category'] in CATEGORIES for r in pairs), 'Unfrozen mathematical category')
    require(sum(r['domain'] == 'permworld' for r in pairs) == 480 and
            sum(r['domain'] == 'graphworld' for r in pairs) == 12, 'Domain candidate scope changed')
    require(all(r['left_task'] != r['right_task'] for r in pairs), 'Candidate search excludes same-task comparisons')
    require(all((r['category'] == 'designated') == (r['right_task'] == r['designated_right_task'])
                for r in pairs), 'Designated classification does not match the exact original endpoint')
    groups = defaultdict(list)
    for r in pairs:
        groups[tuple(r[k] for k in ('domain', 'relation', 'architecture', 'encoding'))].append(r)
    require(len(groups) == 36, 'Expected 32 PermWorld and four GraphWorld coupled settings')
    for key, rr in groups.items():
        require(len(rr) == (15 if key[0] == 'permworld' else 3), 'Incomplete candidate task grid')
        require(len({r['right_task'] for r in rr}) == len(rr), 'Candidate right task repeated')
        require(sum(r['category'] == 'designated' for r in rr) == 1, 'Exactly one original designated endpoint required')
    return p, f, metric, pairs


class SavedSystem:
    """Only metadata plus the original prefix_final endpoint is consumed.

    Large checkpoint/cache byte protection belongs to the root's bound audit.
    We check index/protocol/data bindings here and do not invoke old Inputs or
    hash every large cached field. Perm prefix_final is a read-only mmap;
    Graph NPZ loading selects prefix_final alone.
    """
    def __init__(self, domain, config, metric):
        self.domain, self.config, self.metric = domain, config, metric
        self.p = read(config['protocol_path'])
        require(sha(config['protocol_path']) == config['protocol_sha256'], 'Original protocol changed: ' + domain)
        require(sha(config['index_path']) == config['index_sha256'], 'Original cache index changed: ' + domain)
        require(sha(config['data_path']) == config['data_sha256'], 'Original fixed-world data changed: ' + domain)
        self.index = read(config['index_path'])
        require(self.index['status'] == 'completed' and self.index['protocol_sha256'] == config['protocol_sha256'], 'Cache not complete/bound: ' + domain)
        data_binding = self.index['source_data_sha256'] if domain == 'permworld' else self.index['data_sha256']
        require(data_binding == config['data_sha256'], 'Cache world mismatch: ' + domain)
        self.split = config['analysis_split']
        require(self.split == ('test' if domain == 'permworld' else 'analysis'), 'Original endpoint split changed')
        self.lengths = list(self.p['lengths'])
        require(self.lengths == (list(range(10, 31)) if domain == 'permworld' else [6, 7, 8]), 'Original length grid changed')
        self.names = list(self.p['label_task_names'] if domain == 'permworld' else self.p['tasks'])
        self.actions = list(self.p['actions'])
        self.controls = ['raw_within_length', 'answer_strata'] if domain == 'permworld' else ['raw_within_n', 'answer_strata', 'answer_density_strata']
        expected_controls = config.get('controls', self.controls)
        require(expected_controls == self.controls, 'Unfrozen control scope')
        with np.load(config['data_path'], allow_pickle=False) as ar:
            self.ns = ar[self.split + '_lengths'].astype(np.int64)
            self.labels = ar[self.split + '_labels'].astype(np.int64)
            self.edges = ar[self.split + '_edges'][:, 0].astype(np.int64) if domain == 'graphworld' else None
        require(self.labels.shape == (len(self.ns), len(self.actions), len(self.names)), 'Original label columns changed')
        require(all(int((self.ns == n).sum()) == 512 for n in self.lengths), 'Original 512 anchors per length changed')
        self.relations = {r['pair_id'] if domain == 'permworld' else r['relation']: r for r in self.p['relations']}
        records = [r for r in self.index['records'] if r['split'] == self.split]
        self.records = {(r['architecture'], r['encoding'], r['task'], r['condition']): r for r in records} if domain == 'permworld' else {(r['task'], r['condition']): r for r in records}
        require(len(self.records) == (128 if domain == 'permworld' else 8), 'Source endpoint cache coverage changed')
        self.cache, self.loaded_sources = {}, {}
        require(sha(config['original_results_path']) == config['original_results_sha256'], 'Original comparison table changed')
        old = csv_read(config['original_results_path'])
        self.original = {}
        for r in old:
            if domain == 'permworld':
                if r['branch'] != 'hidden' or r['layer'] != 'final_norm' or r['metric'] != 'linear_cka':
                    continue
                key = (r['relation'], r['architecture'], r['encoding'], r['control'], int(r['length']))
            else:
                if r['branch'] != 'prefix_final' or r['metric'] != 'linear_cka':
                    continue
                key = (r['relation'], r['control'], int(r['n']))
            require(key not in self.original, 'Original comparison is not unique')
            self.original[key] = r

    def descriptor(self, architecture, encoding, task, phase):
        r = self.records[architecture, encoding, task, phase] if self.domain == 'permworld' else self.records[task, phase]
        if self.domain == 'permworld':
            require(r['protocol_sha256'] == self.config['protocol_sha256'] and r['source_data_sha256'] == self.config['data_sha256'], 'Logical source record is unbound')
            meta = r['arrays']['prefix_final']
            source = dict(path=meta['path'], source_file_sha256=meta['sha256'], array_sha256=meta['array_sha256'],
                          shape=meta['shape'], dtype=meta['dtype'], checkpoint_sha256=r['source_checkpoint_sha256'],
                          record_run_id=r['run_id'], exact_endpoint='prefix_final; not packed norm')
        else:
            meta = r['metadata']; sig = meta['signature']
            require(sig['protocol_sha256'] == self.config['protocol_sha256'] and sig['data_sha256'] == self.config['data_sha256'], 'Graph logical source is unbound')
            require((sig['task'], sig['condition'], sig['split']) == (task, phase, self.split), 'Graph logical identity changed')
            source = dict(path=r['path'], source_file_sha256=meta['sha256'], shape=meta['shape']['prefix_final'],
                          dtype='float32', checkpoint_sha256=sig['checkpoint_sha256'], exact_endpoint='prefix_final only from NPZ')
        return r, source

    def feature(self, architecture, encoding, task, phase):
        r, desc = self.descriptor(architecture, encoding, task, phase)
        path = desc['path']
        if path not in self.cache:
            if self.domain == 'permworld':
                a = np.load(path, mmap_mode='r', allow_pickle=False)
                require(not a.flags.writeable, 'PermWorld source mmap must be read-only')
            else:
                with np.load(path, allow_pickle=False) as ar:
                    a = ar['prefix_final']
                a.setflags(write=False)
            require(a.shape == (len(self.ns), len(self.actions), 256) and a.dtype == np.float32, 'Original final endpoint schema changed')
            require(list(a.shape) == desc['shape'], 'Index endpoint shape differs')
            self.cache[path] = a
            self.loaded_sources[path] = desc
        return self.cache[path]

    def selection(self, pair, control, n):
        relation = self.relations[pair['relation']]
        require(relation['left'] == pair['left_task'] and relation['right'] == pair['designated_right_task'], 'Manifest relation identity differs')
        correct = relation['input_transform'] if self.domain == 'permworld' else relation['correct']
        require(pair['transform'] == correct, 'Fixed transformation differs from original relation')
        ai = self.actions.index(correct)
        left = self.labels[:, 0, self.names.index(pair['left_task'])]
        right = self.labels[:, :, self.names.index(pair['right_task'])]
        if self.domain == 'permworld':
            # Offset changes coordinates bijectively. It is applied to all
            # candidate right-role labels; no gold labels or features change.
            right = right - int(relation['right_label_offset'])
            keys = np.column_stack((self.ns, left, right))
        else:
            right = right[:, [0, ai, self.actions.index(relation['wrong'])]]
            if relation['mapping'] == 'n_minus_1_minus_y':
                right = self.ns[:, None] - 1 - right
            else:
                require(relation['mapping'] == 'identity', 'Unknown original graph mapping')
            keys = np.column_stack((self.ns, left, right))
            if control == 'answer_density_strata':
                keys = np.column_stack((keys, self.edges))
        use = self.ns == n
        raw = control in ('raw_within_length', 'raw_within_n')
        if not raw:
            require(control in ('answer_strata', 'answer_density_strata'), 'Unknown common-answer control')
            _, group, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
            use &= counts[group] >= 3
        ids = np.flatnonzero(use).astype(np.int64)
        selected_keys = None if raw else keys[ids].astype(np.int64, copy=False)
        group_count = (1 if len(ids) else 0) if raw else len(np.unique(selected_keys, axis=0))
        audit = dict(anchor_count=512, selected_count=len(ids), excluded_count=512-len(ids),
                     groups=group_count, residual_df=len(ids)-group_count, selection_ids_sha256=array_sha(ids),
                     grouping_keys_sha256=array_sha(selected_keys) if selected_keys is not None else '',
                     grouping_rule='global mean within n' if raw else 'original joint-answer keys; groups>=3',
                     group_threshold=0 if raw else 3, grouping_key_dtype='' if raw else 'int64')
        return ids, selected_keys, ai, audit

    def original_row(self, pair, control, n):
        key = (pair['relation'], pair['architecture'], pair['encoding'], control, n) if self.domain == 'permworld' else (pair['relation'], control, n)
        return self.original[key]


def score(metric, x, y):
    result = metric.cka(x, y, debiased=False)
    if result is not None:
        require(math.isfinite(result), 'Original CKA returned nonfinite value')
        return result, ''
    return None, ('selected_samples_below_2' if len(x) < 2 else 'centered_Gram_norm_product_at_most_1e-20')


def independent_covariance(x, y):
    # Original cka centers globally after optional joint-answer residualizing.
    x, y = x - x.mean(0), y - y.mean(0)
    xx, yy, xy = x.T @ x, y.T @ y, x.T @ y
    den = np.linalg.norm(xx) * np.linalg.norm(yy)
    return float(np.square(xy).sum()/den) if len(x) >= 2 and den > 1e-20 else None


def aggregate(per_length, systems):
    groups = defaultdict(list)
    for r in per_length:
        groups[r['pair_id'], r['control']].append(r)
    summaries = []
    for group in groups.values():
        group.sort(key=lambda r: r['length'])
        first = group[0]; lengths = systems[first['domain']].lengths
        require([r['length'] for r in group] == lengths, 'Missing or duplicate pair-length component')
        out = {k: first[k] for k in PAIR_FIELDS}
        out.update(control=first['control'], metric='ordinary_linear_cka', endpoint='final_norm prefix_final',
                   length_count=len(lengths), weights='1/' + str(len(lengths)) + '; undefined length leaves aggregate undefined')
        for phase in PHASES:
            values = [r['fixed_transform_score_' + phase] for r in group]
            out['fixed_transform_score_' + phase] = fixed_mean(values, lengths)
            out[phase + '_defined_lengths'] = sum(v is not None for v in values)
            out[phase + '_undefined_lengths'] = ','.join(str(r['length']) for r in group if r['fixed_transform_score_' + phase] is None)
        out['training_score_change'] = out['fixed_transform_score_trained'] - out['fixed_transform_score_initial'] if all(out['fixed_transform_score_' + p] is not None for p in PHASES) else None
        out['minimum_selected_count'] = min(r['selected_count'] for r in group)
        out['maximum_selected_count'] = max(r['selected_count'] for r in group)
        summaries.append(out)
    return summaries


def category_tables(summaries):
    groups = defaultdict(list)
    for r in summaries:
        groups[tuple(r[k] for k in ('domain', 'relation', 'architecture', 'encoding', 'control'))].append(r)
    categories, comparisons, changes = [], [], []
    for key, rr in groups.items():
        identity = dict(zip(('domain', 'relation', 'architecture', 'encoding', 'control'), key))
        designated = [r for r in rr if r['category'] == 'designated']
        require(len(designated) == 1, 'Missing designated candidate in summarized control')
        designated = designated[0]
        for category in CATEGORIES:
            cand = [r for r in rr if r['category'] == category]
            for phase in PHASES:
                stats = finite_stats([r['fixed_transform_score_' + phase] for r in cand])
                categories.append(dict(**identity, phase=phase, category=category, **stats,
                    status='empty_fixed_category' if not cand else 'no_defined_candidates' if not stats['defined_count'] else 'finite_candidate_description',
                    finite_support_right_tasks=';'.join(r['right_task'] for r in cand if r['fixed_transform_score_' + phase] is not None),
                    finite_support_pair_ids=';'.join(r['pair_id'] for r in cand if r['fixed_transform_score_' + phase] is not None),
                    candidate_support='complete fixed-length per-pair aggregates; category finite subset explicit',
                    independent_replications=False))
                if category != 'designated':
                    s = designated['fixed_transform_score_' + phase]; mean, median = stats['finite_mean'], stats['finite_median']
                    comparisons.append(dict(**identity, phase=phase, comparison_category=category,
                        designated_pair_id=designated['pair_id'], designated_score=s, **stats,
                        designated_minus_category_mean_state=s-mean if s is not None and mean is not None else None,
                        finite_support_right_tasks=';'.join(r['right_task'] for r in cand if r['fixed_transform_score_' + phase] is not None),
                        finite_support_pair_ids=';'.join(r['pair_id'] for r in cand if r['fixed_transform_score_' + phase] is not None),
                        designated_minus_category_median=s-median if s is not None and median is not None else None,
                        interpretation='descriptive score difference; candidate-specific strata/support; no significance or mechanism'))
            common = [r for r in cand if all(r['fixed_transform_score_' + p] is not None for p in PHASES)]
            initial = finite_stats([r['fixed_transform_score_initial'] for r in common])['finite_mean']
            trained = finite_stats([r['fixed_transform_score_trained'] for r in common])['finite_mean']
            individual = finite_stats([r['training_score_change'] for r in common])
            d0, dt = designated['fixed_transform_score_initial'], designated['fixed_transform_score_trained']
            r0 = d0-initial if d0 is not None and initial is not None else None
            rt = dt-trained if dt is not None and trained is not None else None
            changes.append(dict(**identity, category=category, candidate_total=len(cand), finite_common_pair_count=len(common),
                common_initial_mean=initial, common_trained_mean=trained, common_mean_training_score_change=individual['finite_mean'],
                common_median_training_score_change=individual['finite_median'],
                designated_initial_score=d0, designated_trained_score=dt,
                designated_minus_common_category_initial=r0, designated_minus_common_category_trained=rt,
                Gcat=rt-r0 if r0 is not None and rt is not None else None,
                common_support_pair_ids=';'.join(r['pair_id'] for r in common),
                common_support_right_tasks=';'.join(r['right_task'] for r in common),
                designated_both_phases_finite=d0 is not None and dt is not None,
                formula='(S_designated_trained - mean_common S_candidate_trained) - (S_designated_initial - mean_common S_candidate_initial)',
                support='same candidate pairs finite at both phases; each uses all fixed lengths and its own identical phase mask',
                not_original_Delta=True, no_nan_length_reweighting=True))
    return categories, comparisons, changes


def run(diag):
    require(not (diag / 'analysis_completion.json').exists(), 'Existing completed analysis retained; no implicit rerun')
    begun = time.monotonic()
    p, freeze, metric, pairs = frozen_inputs(diag)
    require(set(p['systems']) == {'permworld', 'graphworld'}, 'Fixed two-system scope changed')
    systems = {d: SavedSystem(d, c, metric) for d, c in p['systems'].items()}
    per_length, masks, designated_checks, aggregate_checks, covariance_checks = [], [], [], [], []
    fixture_pairs = set()
    for domain in systems:
        # Fixed first original relation, designated plus lexicographically
        # first non-designated task, all four Perm configurations. Graph uses
        # its only fixed configuration. Chosen before reading scores.
        system = systems[domain]; relation = next(iter(system.relations))
        candidates = [r for r in pairs if r['domain'] == domain and r['relation'] == relation]
        for config in sorted(set((r['architecture'], r['encoding']) for r in candidates)):
            configured = [r for r in candidates if (r['architecture'], r['encoding']) == config]
            desired = [r for r in configured if r['category'] == 'designated']
            desired += sorted((r for r in configured if r['category'] != 'designated'), key=lambda r: r['right_task'])[:1]
            fixture_pairs.update(r['pair_id'] for r in desired)
    require(len(fixture_pairs) == 10, 'Expected ten fixed covariance-verification pair configurations')
    for i, pair in enumerate(pairs):
        system = systems[pair['domain']]
        for control in system.controls:
            for n in system.lengths:
                ids, keys, ai, mask = system.selection(pair, control, n)
                base = {k: pair[k] for k in PAIR_FIELDS}
                base.update(control=control, length=n)
                masks.append(dict(**base, **mask))
                row = dict(**base, **mask, metric='ordinary_linear_cka', endpoint='final_norm prefix_final')
                for phase in PHASES:
                    x = system.feature(pair['architecture'], pair['encoding'], pair['left_task'], phase)[ids, 0]
                    y = system.feature(pair['architecture'], pair['encoding'], pair['right_task'], phase)[ids, ai]
                    x, y = metric.center(x, keys), metric.center(y, keys)
                    value, reason = score(metric, x, y)
                    row['fixed_transform_score_' + phase], row[phase + '_undefined_reason'] = value, reason
                    if pair['pair_id'] in fixture_pairs and n == system.lengths[0] and control != 'answer_density_strata':
                        independent = independent_covariance(x, y) if len(x) >= 2 else None
                        passed = equal_optional(value, independent, 1e-8)
                        covariance_checks.append(dict(**base, phase=phase, production_score=value,
                            independent_covariance_score=independent,
                            absolute_error=abs(value-independent) if value is not None and independent is not None else None,
                            tolerance=1e-8, passed=passed, selection_ids_sha256=mask['selection_ids_sha256'],
                            designated=pair['category'] == 'designated', selection_fixed_before_scores=True))
                    if pair['category'] == 'designated':
                        original = number(system.original_row(pair, control, n)[phase + '_correct'])
                        designated_checks.append(dict(**base, phase=phase, original_score=original, recomputed_score=value,
                            absolute_error=abs(value-original) if value is not None and original is not None else None,
                            tolerance=1e-8, passed=equal_optional(value, original, 1e-8)))
                row['training_score_change'] = row['fixed_transform_score_trained']-row['fixed_transform_score_initial'] if all(row['fixed_transform_score_' + q] is not None for q in PHASES) else None
                per_length.append(row)
        if (i+1) % 20 == 0:
            print(json.dumps(dict(candidate_pairs_done=i+1, candidate_pairs_total=len(pairs), wall_seconds=time.monotonic()-begun)), flush=True)
    summaries = aggregate(per_length, systems)
    for row in summaries:
        if row['category'] != 'designated':
            continue
        system = systems[row['domain']]
        for phase in PHASES:
            expected = fixed_mean([number(system.original_row(row, row['control'], n)[phase + '_correct'])
                                   for n in system.lengths], system.lengths)
            got = row['fixed_transform_score_' + phase]
            aggregate_checks.append(dict(**{k:row[k] for k in PAIR_FIELDS}, control=row['control'], phase=phase,
                original_fixed_length_mean=expected, recomputed_fixed_length_mean=got,
                absolute_error=abs(got-expected) if got is not None and expected is not None else None,
                fixed_length_count=len(system.lengths), tolerance=1e-8, passed=equal_optional(got, expected, 1e-8),
                reference='Original initial/trained_correct per-length CSV, same complete fixed equal-length weights'))
    categories, comparisons, changes = category_tables(summaries)
    problems = [r for r in designated_checks + aggregate_checks + covariance_checks if not r['passed']]
    expected_rows = 480*21*2 + 12*3*3
    require(len(per_length) == len(masks) == expected_rows, 'Full pair-length-control grid incomplete')
    require(len(summaries) == 480*2 + 12*3, 'Fixed pair summary grid incomplete')
    require(len(designated_checks) == 32*21*2*2 + 4*3*3*2, 'Original designated endpoints not fully rechecked')
    require(len(aggregate_checks) == 32*2*2 + 4*3*2, 'Original designated fixed-length aggregates not fully rechecked')
    require(len(covariance_checks) == 10*2*2, 'Fixed forty-cell independent numerical fixture incomplete')
    table_paths = []
    for filename, records in [('pair_per_length.csv', per_length), ('pair_masks.csv', masks), ('pair_summary.csv', summaries),
                              ('category_summary.csv', categories), ('designated_category_comparisons.csv', comparisons),
                              ('common_support_training_comparisons.csv', changes),
                              ('designated_endpoint_verification.csv', designated_checks),
                              ('designated_aggregate_verification.csv', aggregate_checks),
                              ('independent_covariance_verification.csv', covariance_checks)]:
        path = diag / filename; csv_write(path, records); table_paths.append(path)
    json_write(diag / 'loaded_endpoint_manifest.json', {domain: s.loaded_sources for domain, s in systems.items()})
    audit = dict(status='passed' if not problems else 'failed', protocol_sha256=sha(diag / 'protocol.json'),
        freeze_sha256=sha(diag / 'freeze.json'), source_sha256=sha(__file__), candidate_pairs=492,
        permworld_pairs=480, graphworld_pairs=12, pair_length_control_rows=len(per_length), pair_summary_rows=len(summaries),
        original_designated_per_length_phase_checks=len(designated_checks),
        original_designated_aggregate_phase_checks=len(aggregate_checks), independent_covariance_checks=len(covariance_checks),
        designated_maximum_absolute_error=max((r['absolute_error'] for r in designated_checks if r['absolute_error'] is not None), default=None),
        designated_aggregate_maximum_absolute_error=max((r['absolute_error'] for r in aggregate_checks if r['absolute_error'] is not None), default=None),
        independent_maximum_absolute_error=max((r['absolute_error'] for r in covariance_checks if r['absolute_error'] is not None), default=None),
        failure_count=len(problems), failures=problems, input_sha256={
            str(diag/'protocol.json'):sha(diag/'protocol.json'), str(diag/'freeze.json'):sha(diag/'freeze.json'),
            p['pair_manifest_path']:p['pair_manifest_sha256'], p['metric_source_path']:p['metric_source_sha256'],
            **{c[k]:c[k.replace('_path','_sha256')] for c in p['systems'].values()
               for k in ('protocol_path','index_path','data_path','original_results_path')}},
        output_sha256={str(path):sha(path) for path in table_paths + [diag/'loaded_endpoint_manifest.json']},
        wall_seconds=time.monotonic()-begun, cpu_threads=2, new_training=False, new_forward=False, torch_imported='torch' in sys.modules,
        other_metrics=False, no_old_artifacts_modified=True, large_cache_protection='Delegated to root-bound protection audit; not rehashed here',
        controls='Original pair-specific joint-answer strata and original anchors; not new pair-only keys',
        length_weights='All21/3 required for every pair-phase aggregate; no undefined-length reweighting',
        categories='Manifest-fixed designated / other_known_exact / not_designated; empty categories retained',
        common_support_formula='(S_d,t - mean_common S_c,t) - (S_d,0 - mean_common S_c,0); not original relation Delta',
        independence='492 dependent candidate cells, not 492 independent replications; task/transform/world/initialization shared',
        classification_limit='not_designated means no designated exact correspondence in this fixed manifest, not absence of all mathematical relations')
    require(not audit['torch_imported'], 'Torch must not be imported by cache-only analysis')
    json_write(diag/'analysis_audit.json', audit)
    if problems:
        raise AssertionError('Fixed designated/independent verification failed; see analysis_audit.json')
    json_write(diag/'analysis_completion.json', dict(status='completed_cache_only_fixed_pairing_analysis',
        analysis_audit_sha256=sha(diag/'analysis_audit.json'), source_sha256=sha(__file__), protocol_sha256=sha(diag/'protocol.json'),
        no_automatic_followup=True, new_training=False, new_forward=False))
    print(json.dumps({k:audit[k] for k in ('status','candidate_pairs','pair_length_control_rows','pair_summary_rows',
        'original_designated_per_length_phase_checks','independent_covariance_checks','failure_count','wall_seconds')}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true', help='Requires protocol.json plus matching freeze.json')
    parser.add_argument('--diag', type=Path, default=DIAG)
    args = parser.parse_args()
    require(args.run, 'No implicit computation. Run only after root freezes the complete protocol/source.')
    require(args.diag.resolve() == DIAG, 'All new outputs stay in the isolated diagnostic directory')
    with threadpool_limits(limits=2):
        run(args.diag.resolve())


if __name__ == '__main__':
    main()
