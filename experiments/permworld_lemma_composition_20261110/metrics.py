"""Score saved mask-composition answers; never fit or inspect hidden states.

NONE is (-1,-1) for intervals and -1 for the count. Lengths, rather than
numbers of selected examples, receive fixed equal weight in macro summaries.
"""
from __future__ import annotations

from collections import defaultdict
import math
import numpy as np

TASKS = ('longest', 'fixed_longest', 'turns_before_fixed')
COUNT_FIELDS = ('n', 'exact_n', 'truth_none_n', 'prediction_none_n',
                'both_none_n', 'missed_presence_n', 'false_presence_n',
                'joint_present_n')
METRICS = ('exact_accuracy', 'truth_none_rate', 'prediction_none_rate',
           'joint_present_coverage', 'prediction_present_given_truth_present',
           'start_mae', 'end_mae', 'endpoint_mae', 'length_mae',
           'normalized_endpoint_error', 'normalized_length_error',
           'count_mae', 'normalized_count_error')
COHORT_METRICS = ('exact_accuracy', 'endpoint_mae', 'length_mae',
                  'normalized_endpoint_error', 'normalized_length_error',
                  'count_mae', 'normalized_count_error',
                  'joint_present_coverage', 'prediction_none_rate')


def finite(value):
    if value is None or value in ('', 'None', 'nan', 'NaN'):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def fixed_mean(values):
    values = [finite(v) for v in values]
    return math.fsum(values) / len(values) if values and all(v is not None for v in values) else None


def validate_arrays(arrays, registered_lengths):
    ns = np.asarray(arrays['lengths'])
    routes = [str(x) for x in arrays['route_names']]
    variants = [str(x) for x in arrays['variant_names']]
    n, r, v = len(ns), len(routes), len(variants)
    if len(set(routes)) != r or len(set(variants)) != v or variants.count('original') != 1:
        raise ValueError('Route/variant identities must be distinct and include original')
    if set(np.unique(ns)) != set(registered_lengths):
        raise ValueError('All and only registered lengths are required')
    if len(np.unique(np.asarray(arrays['anchor_indices']))) != n:
        raise ValueError('Anchor IDs must be distinct within a unit')
    for key in ('matched', 'truth_fixed_present', 'truth_constraint_changes'):
        if np.asarray(arrays[key]).shape != (n,):
            raise ValueError('Invalid scalar-anchor field: ' + key)
    if 'eligible' in arrays and np.asarray(arrays['eligible']).shape != (n,):
        raise ValueError('Invalid eligible support')
    for task in TASKS:
        truth = np.asarray(arrays['truth_' + task])
        pred = np.asarray(arrays['pred_' + task])
        trailing = (2,) if task != 'turns_before_fixed' else ()
        if truth.shape != (n,) + trailing or pred.shape != (r, v, n) + trailing:
            raise ValueError('Invalid answer dimensions for ' + task)
        if not np.issubdtype(truth.dtype, np.integer) or not np.issubdtype(pred.dtype, np.integer):
            raise ValueError('Answers must use integer/sentinel representation')
        for a in (truth, pred):
            if trailing:
                present = a[..., 0] != -1
                if np.any((a[..., 1] != -1) != present):
                    raise ValueError('Partial NONE interval')
                if task == 'longest' and not present.all():
                    raise ValueError('Unconstrained interval must always be present')
                if np.any(present & ((a[..., 0] < 1) | (a[..., 1] < a[..., 0]) | (a[..., 1] > ns))):
                    raise ValueError('Out-of-range interval')
            elif np.any((a < -1) | (a > np.maximum(ns - 2, 0))):
                raise ValueError('Out-of-range turn count')
    true_present = np.asarray(arrays['truth_fixed_present'], bool)
    if not np.array_equal(true_present, np.asarray(arrays['truth_fixed_longest'])[:, 0] >= 1):
        raise ValueError('Fixed-point presence and oracle interval disagree')
    if not np.array_equal(true_present, np.asarray(arrays['truth_turns_before_fixed']) >= 0):
        raise ValueError('Fixed-point presence and oracle count disagree')
    expected_change = true_present & np.any(np.asarray(arrays['truth_fixed_longest']) != np.asarray(arrays['truth_longest']), axis=-1)
    if not np.array_equal(expected_change, np.asarray(arrays['truth_constraint_changes'], bool)):
        raise ValueError('Constraint diagnostic must be truth-present with an actual answer change')

    original = variants.index('original')
    fp_variants = [i for i, name in enumerate(variants) if name.startswith('fixed_shuffle_')]
    turning_variants = [i for i, name in enumerate(variants) if name.startswith('turn_shuffle_')]
    if not fp_variants or not turning_variants:
        raise ValueError('Both prespecified mask-shuffle families are required')
    unconstrained = np.asarray(arrays['pred_longest'])
    for i in fp_variants:
        if not np.array_equal(unconstrained[:, i], unconstrained[:, original]):
            raise ValueError('Fixed-mask shuffling changed the task that ignores fixed points')
    no_interval = np.asarray(arrays['pred_fixed_longest'])[..., 0] < 0
    no_count = np.asarray(arrays['pred_turns_before_fixed']) < 0
    if not np.array_equal(no_interval, no_count):
        raise ValueError('The two fixed-point tasks must share predicted NONE decisions')
    if not np.all(no_interval == no_interval[:, original:original+1]):
        raise ValueError('Count-preserving shuffles changed predicted fixed-point absence')
    if not np.all(no_interval[:, original] == no_interval[0, original]):
        raise ValueError('All routes must use the same frozen fixed-point readout')
    true_none = ~true_present
    for task in ('fixed_longest', 'turns_before_fixed'):
        pred = np.asarray(arrays['pred_' + task])
        exact = np.all(pred == np.asarray(arrays['truth_' + task]), axis=-1) if task == 'fixed_longest' else pred == np.asarray(arrays['truth_' + task])
        if not np.all(exact[..., true_none] == exact[:, original:original+1, true_none]):
            raise ValueError('A shuffle changed exact correctness on true-NONE samples')
    return dict(routes=routes, variants=variants, anchors=n,
                fixed_shuffle_longest_sample_checks=r * len(fp_variants) * n,
                fixed_task_NONE_sample_checks=r * v * n,
                same_fixed_point_readout_across_routes=True)


def sample_metrics(task, truth, pred, ns):
    n = len(ns)
    if task == 'turns_before_fixed':
        tp, pp = truth >= 0, pred >= 0
        exact = truth == pred
    else:
        tp, pp = truth[:, 0] >= 1, pred[:, 0] >= 1
        exact = np.all(truth == pred, axis=-1)
    joint = tp & pp
    result = dict(exact=exact, truth_none=~tp, prediction_none=~pp,
                  both_none=~tp & ~pp, missed_presence=tp & ~pp,
                  false_presence=~tp & pp, joint_present=joint,
                  truth_present=tp)
    for name in METRICS:
        if name not in ('exact_accuracy', 'truth_none_rate', 'prediction_none_rate',
                        'joint_present_coverage', 'prediction_present_given_truth_present'):
            result[name] = np.full(n, np.nan)
    mismatch = tp != pp
    if task != 'turns_before_fixed':
        start = np.abs(pred[:, 0].astype(float) - truth[:, 0])
        end = np.abs(pred[:, 1].astype(float) - truth[:, 1])
        length = np.abs((pred[:, 1] - pred[:, 0]) - (truth[:, 1] - truth[:, 0])).astype(float)
        for name, value in [('start_mae', start), ('end_mae', end), ('endpoint_mae', (start+end)/2), ('length_mae', length)]:
            result[name][joint] = value[joint]
        result['normalized_endpoint_error'] = np.where(mismatch, 1., np.where(joint, (start+end)/(2*ns), 0.))
        result['normalized_length_error'] = np.where(mismatch, 1., np.where(joint, length/ns, 0.))
    else:
        error = np.abs(pred.astype(float) - truth)
        result['count_mae'][joint] = error[joint]
        result['normalized_count_error'] = np.where(mismatch, 1., np.where(joint, error/ns, 0.))
    return result


def selections(arrays, task):
    n = len(arrays['lengths'])
    eligible = np.asarray(arrays.get('eligible', np.ones(n, bool)), bool)
    supports = {'all': eligible, 'matched': eligible & np.asarray(arrays['matched'], bool)}
    strata = {'all': np.ones(n, bool)}
    if task != 'longest':
        strata.update(fixed_present=np.asarray(arrays['truth_fixed_present'], bool),
                      fixed_absent=~np.asarray(arrays['truth_fixed_present'], bool))
    if task == 'fixed_longest':
        strata['constraint_changes'] = np.asarray(arrays['truth_constraint_changes'], bool)
    return [(support, stratum, s & t) for support, s in supports.items() for stratum, t in strata.items()]


def one_length_row(identity, values, select):
    size = int(select.sum())
    row = dict(identity, n=size)
    for short, vector in [('exact_n', 'exact'), ('truth_none_n', 'truth_none'),
                          ('prediction_none_n', 'prediction_none'), ('both_none_n', 'both_none'),
                          ('missed_presence_n', 'missed_presence'), ('false_presence_n', 'false_presence'),
                          ('joint_present_n', 'joint_present')]:
        row[short] = int(values[vector][select].sum())
    row['truth_present_n'] = size - row['truth_none_n']
    if size:
        row.update(exact_accuracy=row['exact_n']/size, truth_none_rate=row['truth_none_n']/size,
                   prediction_none_rate=row['prediction_none_n']/size,
                   joint_present_coverage=row['joint_present_n']/size,
                   prediction_present_given_truth_present=(row['joint_present_n']/row['truth_present_n'] if row['truth_present_n'] else None))
    else:
        row.update({k: None for k in METRICS[:5]})
    for name in METRICS[5:]:
        selected = values[name][select]
        good = np.isfinite(selected)
        row[name] = float(selected[good].mean()) if good.any() else None
        row[name + '_finite_n'] = int(good.sum())
    row['undefined_reason'] = '' if size else 'empty_prespecified_support_at_length'
    return row


def aggregate_lengths(rows, registered_lengths):
    keys = ('unit', 'route', 'variant', 'task', 'support', 'stratum')
    grouped = defaultdict(list)
    for r in rows:
        grouped[tuple(r[k] for k in keys)].append(r)
    output = []
    for key, group in grouped.items():
        by_n = {int(r['length']): r for r in group}
        if len(by_n) != len(group) or set(by_n) != set(registered_lengths):
            raise ValueError('Missing or duplicate required length component')
        rr = [by_n[n] for n in registered_lengths]
        result = dict(zip(keys, key), length_count=len(rr), aggregation='fixed_equal_1/'+str(len(rr)))
        for name in COUNT_FIELDS + ('truth_present_n',):
            result[name] = sum(int(r[name]) for r in rr)
        for metric in METRICS:
            values = [r[metric] for r in rr]
            result[metric] = fixed_mean(values)
            result[metric + '_finite_lengths'] = sum(finite(v) is not None for v in values)
            result[metric + '_undefined_lengths'] = ';'.join(str(n) for n, v in zip(registered_lengths, values) if finite(v) is None)
        result['pooled_exact_accuracy'] = result['exact_n']/result['n'] if result['n'] else None
        for metric in METRICS[5:]:
            count_name = metric + '_finite_n'
            denominator = sum(int(r[count_name]) for r in rr)
            result['pooled_' + metric] = (math.fsum(float(r[metric])*int(r[count_name]) for r in rr if r[metric] is not None)/denominator if denominator else None)
        output.append(result)
    return output


def score_unit_arrays(unit, arrays, registered_lengths=tuple(range(10, 31))):
    """Return (per-length rows, fixed-length aggregate rows, invariant receipt)."""
    audit = validate_arrays(arrays, registered_lengths)
    ns = np.asarray(arrays['lengths'])
    length_rows = []
    selection = {task: selections(arrays, task) for task in TASKS}
    length_masks = {n: ns == n for n in registered_lengths}
    for ri, route in enumerate(audit['routes']):
        for vi, variant in enumerate(audit['variants']):
            for task in TASKS:
                values = sample_metrics(task, np.asarray(arrays['truth_'+task]), np.asarray(arrays['pred_'+task])[ri, vi], ns)
                for support, stratum, select in selection[task]:
                    for n in registered_lengths:
                        identity = dict(unit=unit, route=route, variant=variant, task=task,
                                        support=support, stratum=stratum, length=n)
                        length_rows.append(one_length_row(identity, values, select & length_masks[n]))
    return length_rows, aggregate_lengths(length_rows, registered_lengths), audit
