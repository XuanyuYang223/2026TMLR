"""Read-only analysis primitives for the eight-relation PermWorld extension.

No optimizer, Torch, checkpoint construction, or adaptive metric selection.
The old primary metrics are imported unchanged from their frozen local file.
"""
from __future__ import annotations

from collections import defaultdict
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path

import numpy as np

WORKSPACE = Path(__file__).resolve().parents[2]
DIAG = WORKSPACE / 'diagnostics/permworld_eight_relations_20261010'
OLD_DIAG = WORKSPACE / 'diagnostics/permworld_controlled_comparison_20261010'
METRIC_PATH = WORKSPACE / 'diagnostics/permworld_scope_inventory_20261009/metrics.py'
METRIC_SHA256 = 'c2a384a8212e661d1917c084a74699297bbef354f06dac42ac75ea90ce3d0b5c'
PHASES = ('initial', 'trained')
ROLES = ('correct', 'wrong', 'identity')
ALL_METRICS = ('linear_cka', 'debiased_cka', 'knn_overlap', 'procrustes')
LAYER_METRICS = ('linear_cka', 'debiased_cka')
FIELDS = tuple(f'{p}_{r}' for p in PHASES for r in ROLES) + ('B0', 'Bt', 'dc', 'dw', 'di', 'Delta')
ERROR_FIELDS = tuple(f'error_{p}_{r}' for p in PHASES for r in ROLES)
GROUP_KEYS = ('architecture', 'encoding', 'relation', 'control', 'branch', 'layer', 'metric')
OLD_RELATIONS = ('descent_inverse', 'peak_complement')


def require(ok, message):
    if not ok:
        raise AssertionError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def array_sha(value):
    value = np.ascontiguousarray(value)
    h = hashlib.sha256()
    h.update(str(value.dtype).encode())
    h.update(str(value.shape).encode())
    h.update(value.tobytes())
    return h.hexdigest()


def js(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def rows(path):
    with Path(path).open(newline='') as stream:
        return list(csv.DictReader(stream))


def csv_write(path, values):
    require(bool(values), 'An empty table must be reported explicitly, not silently written')
    fields = list(dict.fromkeys(k for row in values for k in row))
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(values)


def number(value):
    return float(value) if value not in (None, '', 'None', 'NaN', 'nan') else float('nan')


def equal_length(values):
    require(len(values) == 21, 'The primary aggregate requires all 21 fixed lengths')
    values = [number(v) for v in values]
    return math.fsum(values) / 21 if all(map(math.isfinite, values)) else float('nan')


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def metrics(protocol=None):
    p = protocol or {}
    path = Path(p.get('metric_source_path', p.get('metrics_source_path', METRIC_PATH)))
    expected = p.get('metric_source_sha256', p.get('metrics_source_sha256', METRIC_SHA256))
    require(sha(path) == expected == METRIC_SHA256, 'The original frozen metric implementation changed')
    return module(path, 'eight_relations_original_metrics')


def verify_protocol():
    p = js(DIAG / 'protocol.json')
    for filename in ('training_freeze.json', 'analysis_freeze.json'):
        f = js(DIAG / filename)
        require(sha(DIAG / 'protocol.json') == f['protocol_sha256'], 'Protocol bytes do not match ' + filename)
        for path, digest in f['source_sha256'].items():
            require(sha(path) == digest, 'Frozen source bytes changed: ' + path)
    require(p['lengths'] == list(range(10, 31)), 'Length grid changed')
    require(p['actions'] == ['identity', 'complement', 'inverse'], 'Action index changed')
    require(p['controls'] == ['raw_within_length', 'answer_strata'], 'Control scope changed')
    require(len(p['relations']) == 8 and len(p['label_task_names']) == 16, 'Eight relationships require 16 task labels')
    require(len(set(p['label_task_names'])) == 16, 'Task columns are not unique')
    require(p['architectures'] == ['transformer', 'gru'], 'Configuration scope changed')
    require(p['encodings'] == ['one_line', 'canonical_cycle'], 'Encoding scope changed')
    return p


def selections(data, relation, split, n, control, task_names):
    """Original joint strata with an explicit bijective affine answer mapping.

    For offset +1, subtract 1 from *every* right answer in the key. This is
    only a label-coordinate change and preserves exactly the raw-label groups;
    no test/training label is changed or equated merely by its numerical value.
    """
    ns = data[f'{split}_lengths']
    labels = data[f'{split}_labels']
    left = labels[:, 0, task_names.index(relation['left'])].astype(np.int64)
    right = labels[:, :, task_names.index(relation['right'])].astype(np.int64)
    offset = int(relation.get('right_label_offset', 0))
    key = np.column_stack((ns, left, right - offset))
    use = ns == n
    if control == 'answer_strata':
        _, group, counts = np.unique(key, axis=0, return_inverse=True, return_counts=True)
        use &= counts[group] >= 3
        return np.flatnonzero(use), key[use]
    require(control == 'raw_within_length', 'Unknown control')
    return np.flatnonzero(use), None


def component_values(values):
    out = {f'{p}_{r}': number(values[p][r]) for p in PHASES for r in ROLES}
    out['B0'] = out['initial_correct'] - out['initial_wrong']
    out['Bt'] = out['trained_correct'] - out['trained_wrong']
    for role, field in zip(ROLES, ('dc', 'dw', 'di')):
        out[field] = out['trained_' + role] - out['initial_' + role]
        out['change_' + role] = out[field]
    out['Delta'] = out['Bt'] - out['B0']
    if math.isfinite(out['Delta']):
        require(abs(out['Delta'] - (out['dc'] - out['dw'])) <= 1e-10, 'Additive decomposition failed')
    return out


def undefined_reason(name, xf, yf, xe, ye, ids, duplicates, original):
    if name == 'procrustes':
        rank = min(8, xf.shape[1], yf.shape[1])
        if len(xf) <= rank:
            return 'fit_samples_not_greater_than_fixed_rank'
        if original.pca_fit(xf, rank) is None or original.pca_fit(yf, rank) is None:
            return 'fit_energy_or_relative_1e-6_rank_below_fixed_rank'
        if len(xe) < 2:
            return 'insufficient_evaluation_samples'
        return 'zero_right_projected_evaluation_energy_under_original_rule'
    if len(xe) < (4 if name == 'debiased_cka' else 11 if name == 'knn_overlap' else 2):
        return 'insufficient_evaluation_samples_for_' + name
    if np.square(original.center(xe)).sum() <= 1e-20 or np.square(original.center(ye)).sum() <= 1e-20:
        return 'zero_centered_feature_energy'
    if name == 'knn_overlap':
        return 'fewer_than_k10_eligible_distinct_anchors_under_original_rule'
    return 'zero_U_centered_Gram_norm' if name == 'debiased_cka' else 'linear_Gram_norm_product_at_most_1e-20'


def summarize(source, lengths):
    groups = defaultdict(list)
    for row in source:
        groups[tuple(row[k] for k in GROUP_KEYS)].append(row)
    table, index = [], {}
    for key, group in groups.items():
        require(sorted(int(row['length']) for row in group) == lengths, 'Incomplete/duplicate length group: ' + repr(key))
        record = dict(zip(GROUP_KEYS, key))
        record.update(contrast_type='condition_summary', left_condition='', right_condition='', length_count=21,
                      weights='1/21 fixed; any undefined length leaves that component undefined')
        for field in FIELDS + ERROR_FIELDS:
            record[field] = equal_length([row.get(field) for row in group])
        record['undefined_lengths'] = sum(not math.isfinite(number(row['Delta'])) for row in group)
        record['undefined_reason'] = '; '.join(sorted({row.get('nan_reason', '') for row in group
                                                     if row.get('nan_reason', '') not in ('', '{}')}))
        record['Bt_positive_lengths'] = sum(number(row['Bt']) > 0 for row in group)
        record['Delta_positive_lengths'] = sum(number(row['Delta']) > 0 for row in group)
        for field in ('B0', 'Bt', 'dc', 'dw', 'Delta'):
            record[field + '_length_signs'] = ''.join('?' if not math.isfinite(number(row[field])) else
                '+' if number(row[field]) > 0 else '-' if number(row[field]) < 0 else '0'
                for row in sorted(group, key=lambda x: int(x['length'])))
        table.append(record)
        index[key] = record

    def diff(kind, left, right, **identity):
        row = {k: right[k] for k in GROUP_KEYS}
        row.update(identity, contrast_type=kind,
                   left_condition=json.dumps({k: left[k] for k in GROUP_KEYS}, sort_keys=True),
                   right_condition=json.dumps({k: right[k] for k in GROUP_KEYS}, sort_keys=True),
                   length_count=21, weights='Difference of corresponding complete fixed1/21 summaries')
        for field in FIELDS:
            row[field] = number(right[field]) - number(left[field])
        table.append(row)
        return row

    enc = {}
    for key, left in index.items():
        a, e, r, c, b, l, m = key
        if e == 'one_line':
            right = index[a, 'canonical_cycle', r, c, b, l, m]
            out = diff('encoding_difference', left, right, encoding='cycle-minus-line')
            for field, name in [('B0', 'Gamma0'), ('Bt', 'Gammat'), ('dc', 'Gammac'), ('dw', 'Gammaw'), ('Delta', 'GammaDelta')]:
                out[name] = out[field]
            enc[a, r, c, b, l, m] = out
        if a == 'transformer' and l in ('final_norm', 'embedding', 'output'):
            diff('configuration_difference', left, index['gru', e, r, c, b, l, m], architecture='GRU-minus-Transformer')
        if b == 'hidden' and m in LAYER_METRICS:
            last = 'block4' if a == 'transformer' else 'layer4'
            if l != last:
                diff('layer_difference', index[a, e, r, c, b, last, m], left,
                     layer=l + '-minus-' + last, norm_comparison=l == 'final_norm',
                     intended_changed_fields='normalization only' if l == 'final_norm' else 'extraction layer')
    for key, left in enc.items():
        a, r, c, b, l, m = key
        if a == 'transformer' and l in ('final_norm', 'embedding', 'output'):
            out = diff('interaction', left, enc['gru', r, c, b, l, m],
                       architecture='GRU-minus-Transformer', encoding='cycle-minus-line')
            out.update(I0=out['B0'], It=out['Bt'], I=out['Delta'],
                       interpretation='Exploratory configuration by encoding; capacities and source learning unmatched')
    direction = defaultdict(dict)
    for key, value in index.items():
        a, e, r, c, b, l, m = key
        if b == 'hidden' and l == 'final_norm' and m in ('linear_cka', 'knn_overlap', 'procrustes'):
            direction[(a, e, r, c, b, l)][m] = value
    for key, values in direction.items():
        require(len(values) == 3, 'Missing one of the fixed three principal metrics')
        out = dict(zip(GROUP_KEYS[:-1], key))
        out.update(contrast_type='metric_direction', metric='direction_only', length_count=21)
        for field in ('B0', 'Bt', 'dc', 'dw', 'Delta'):
            signs = {m: '?' if not math.isfinite(v[field]) else '+' if v[field] > 0 else '-' if v[field] < 0 else '0'
                     for m, v in values.items()}
            out[field + '_directions'] = json.dumps(signs, sort_keys=True)
            out[field + '_direction_consistent'] = len(set(signs.values())) == 1 and '?' not in signs.values()
        table.append(out)
    return table, index


def cross_summary(source, lengths):
    groups = defaultdict(list)
    for row in source:
        groups[tuple(row[k] for k in ('task', 'encoding', 'control', 'metric'))].append(row)
    result = []
    for key, group in groups.items():
        require(sorted(int(r['length']) for r in group) == lengths, 'Incomplete same-task lengths')
        out = dict(zip(('task', 'encoding', 'control', 'metric'), key))
        out.update(length_count=21, weights='fixed1/21',
                   interpretation='32 distinct same-task configuration pairs, not independent trials')
        for dest, field in [('initial', 'initial_similarity'), ('trained', 'trained_similarity'), ('change', 'similarity_change')]:
            out[dest] = equal_length([r[field] for r in group])
        result.append(out)
    return result
