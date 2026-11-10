"""Fit source-only linear probes and evaluate held-out position readout."""

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits


ACTIONS = ('identity', 'complement', 'inverse')
ENCODINGS = ('canonical_cycle', 'one_line')
ALPHAS = (1., 10., 100., 1000.)


def orbit_keys(permutations):
    n = permutations.shape[1]
    reversed_positions = permutations[:, ::-1]
    orbit_half = [permutations, reversed_positions, n + 1 - permutations,
                  n + 1 - reversed_positions]
    orbit = [*orbit_half, *(np.argsort(x, axis=1) + 1 for x in orbit_half)]
    packed = np.stack(orbit, axis=1).astype(np.uint8)
    return [min(x.tobytes() for x in row) for row in packed]


def extrema(permutations):
    left, middle, right = permutations[:, :-2], permutations[:, 1:-1], permutations[:, 2:]
    return (middle > left) & (middle > right), (middle < left) & (middle < right)


def example_auc(scores, targets):
    ranks = rankdata(scores, axis=1, method='average')
    positives = targets.sum(axis=1)
    denominator = positives * (targets.shape[1] - positives)
    numerator = (ranks * targets).sum(axis=1) - positives * (positives + 1) / 2
    return np.divide(numerator, denominator, out=np.full(len(scores), np.nan), where=denominator > 0)


def average(values):
    finite = np.isfinite(values)
    return float(values[finite].mean()) if finite.any() else np.nan


def fit_probe(features, targets, fit, dev):
    scaling = StandardScaler().fit(features[fit])
    x_fit = scaling.transform(features[fit]).astype(np.float64)
    x_dev = scaling.transform(features[dev]).astype(np.float64)
    best_model, best_score, best_alpha = None, -np.inf, None
    for alpha in ALPHAS:
        model = Ridge(alpha=alpha, solver='cholesky').fit(x_fit, targets[fit])
        score = average(example_auc(model.predict(x_dev), targets[dev]))
        if score > best_score:
            best_model, best_score, best_alpha = model, score, alpha
    if best_model is None:
        raise ValueError('No defined probe validation scores')
    return lambda x: best_model.predict(scaling.transform(x).astype(np.float64)), best_alpha


def position_prior(fit_mask, score_mask):
    fit_count = fit_mask.sum(axis=1)
    prior = np.empty_like(score_mask, dtype=np.float64)
    score_count = score_mask.sum(axis=1)
    for k in np.unique(score_count):
        matching = fit_mask[fit_count == k]
        prior[score_count == k] = (matching if len(matching) else fit_mask).mean(axis=0)
    return prior


def load_feature_group(folder, encoding):
    arrays = {}
    for task in ('initial', 'peaks', 'valleys', 'descents', 'recoils'):
        with np.load(folder / f'{encoding}-{task}.npz', allow_pickle=False) as raw:
            arrays[task] = dict(raw)
    for split in ('validation', 'test'):
        idx = arrays['peaks'][f'{split}_indices']
        for raw in arrays.values():
            if not np.array_equal(raw[f'{split}_indices'], idx):
                raise ValueError(f'Anchor order mismatch for {encoding}/{split}')
    return arrays


def read_length(data, n):
    permutations = {s: data[f'{s}_permutations'][data[f'{s}_lengths'] == n, :n]
                    for s in ('train', 'validation', 'test')}
    orbits = {s: orbit_keys(p) for s, p in permutations.items()}
    train_orbits, test_orbits = set(orbits['train']), set(orbits['test'])
    valid = np.array([key not in train_orbits and key not in test_orbits
                      for key in orbits['validation']])
    dev = np.array([hashlib.sha256(b'peaks-probe-v1' + key).digest()[0] < 51
                    for key in orbits['validation']])
    score = np.array([key not in train_orbits for key in orbits['test']])
    fit, dev = valid & ~dev, valid & dev
    if min(fit.sum(), dev.sum(), score.sum()) == 0:
        raise ValueError(f'Missing orbit-disjoint split for length {n}')
    peak_fit, valley_fit = extrema(permutations['validation'])
    test = permutations['test']
    peak, valley = extrema(test)
    valley_masks = (valley, extrema(n + 1 - test)[1], extrema(np.argsort(test, axis=1) + 1)[1])
    if not np.array_equal(peak, valley_masks[1]):
        raise ValueError(f'Complement identity failed at length {n}')
    count = peak.sum(axis=1)
    common = score & (valley.sum(axis=1) == count) & (valley_masks[2].sum(axis=1) == count)
    return fit, dev, score, common, peak_fit, valley_fit, peak, valley_masks


def run(data, by_encoding, unit):
    coverage, strata, length_rows, learning = [], [], [], []
    first = next(iter(by_encoding.values()))['peaks']['test_indices']
    lengths = np.unique(data['test_lengths'][first])
    for n in lengths:
        fit, dev, test, common, peak_fit, valley_fit, peak, valleys = read_length(data, n)
        coverage.append(dict(unit=unit, length=n, fit_n=int(fit.sum()), dev_n=int(dev.sum()),
                             test_n=int(test.sum()), matched_n=int(common.sum()),
                             validation_excluded_n=int(len(fit) - fit.sum() - dev.sum()),
                             test_excluded_n=int(len(test) - test.sum())))
        for encoding, group in by_encoding.items():
            v = data['validation_lengths'][group['peaks']['validation_indices']] == n
            t = data['test_lengths'][group['peaks']['test_indices']] == n
            for phase in ('initial', 'trained'):
                peaks_feature = group['initial' if phase == 'initial' else 'peaks']
                valley_feature = group['initial' if phase == 'initial' else 'valleys']
                probe, alpha = fit_probe(peaks_feature['validation_0_hidden'][v], peak_fit, fit, dev)
                own_probe, own_alpha = fit_probe(valley_feature['validation_0_hidden'][v], valley_fit, fit, dev)
                # At step zero, both tasks have identical input-prefix states by causality
                # and the genuine shared architecture-level checkpoint.
                items = [('source_peaks', 'identity', 'peak', probe(peaks_feature['test_0_hidden'][t]), peak, alpha)]
                for action, name in enumerate(ACTIONS):
                    x = valley_feature[f'test_{action}_hidden'][t]
                    items.extend([
                        ('transfer', name, 'peak', probe(x), peak, alpha),
                        ('transfer', name, 'own_valley', probe(x), valleys[action], alpha),
                        ('target_own', name, 'own_valley', own_probe(x), valleys[action], own_alpha),
                    ])
                items.extend([
                    ('position_prior', 'identity', 'peak', position_prior(peak_fit[fit], peak), peak, np.nan),
                    ('position_prior', 'identity', 'own_valley', position_prior(valley_fit[fit], valleys[0]), valleys[0], np.nan),
                ])
                if phase == 'trained':
                    for task in ('descents', 'recoils'):
                        x = group[task]['test_1_hidden'][t]
                        items.append((task, 'complement', 'peak', probe(x), peak, alpha))
                    for task, feature, y_fit, y_test in (
                        ('peaks', peaks_feature, peak_fit, peak),
                        ('valleys', valley_feature, valley_fit, valleys[0]),
                    ):
                        train_perms = data['train_permutations'][data['train_lengths'] == n, :n]
                        train_count = extrema(train_perms)[task == 'valleys'].sum(axis=1)
                        mode = np.bincount(train_count).argmax()
                        count = y_test.sum(axis=1)
                        # In the original vocabulary, numeric answer k has token id k.
                        predicted = feature['test_0_prediction'][t]
                        learning.append(dict(unit=unit, encoding=encoding, length=n, task=task,
                                             n=int(test.sum()), mode=int(mode),
                                             accuracy=float(np.mean(predicted[test] == count[test])),
                                             baseline=float(np.mean(mode == count[test]))))
                for method, action, label, predicted, y, reg in items:
                    values = example_auc(predicted, y)
                    for name, selection in [('all', test), ('matched', common)]:
                        selected = values[selection]
                        row = dict(unit=unit, encoding=encoding, phase=phase, method=method, transform=action,
                                   label=label, length=n, support=name, alpha=reg,
                                   n=int(selection.sum()), defined_n=int(np.isfinite(selected).sum()),
                                   auc=average(selected))
                        length_rows.append(row)
                        for count in np.unique(peak.sum(axis=1)[selection]):
                            keep = selection & (peak.sum(axis=1) == count)
                            subset = values[keep]
                            strata.append({**row, 'source_peak_count': int(count), 'n': int(keep.sum()),
                                           'defined_n': int(np.isfinite(subset).sum()), 'auc': average(subset)})
    return map(pd.DataFrame, (coverage, length_rows, strata, learning))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--features', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--unit', default='archdev001')
    parser.add_argument('--encodings', nargs='+', choices=ENCODINGS, default=ENCODINGS)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    with np.load(args.data, allow_pickle=False) as raw:
        data = {k: raw[k] for k in raw.files if k.endswith(('_lengths', '_permutations'))}
    by_encoding = {enc: load_feature_group(args.features, enc) for enc in args.encodings}
    with threadpool_limits(limits=2):
        coverage, lengths, strata, learning = run(data, by_encoding, args.unit)
    for name, table in [('coverage', coverage), ('by_length', lengths),
                        ('by_length_and_count', strata), ('learning_by_length', learning)]:
        table.to_csv(args.out / f'{name}.csv', index=False)
    groups = ['unit', 'encoding', 'phase', 'method', 'transform', 'label', 'support']
    summary = lengths.groupby(groups, as_index=False).agg(
        auc=('auc', 'mean'), finite_lengths=('auc', 'count'), n=('n', 'sum'),
        defined_n=('defined_n', 'sum'))
    summary.to_csv(args.out / 'summary.csv', index=False)
    print(summary.to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
