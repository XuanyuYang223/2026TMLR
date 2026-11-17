"""Validation-only orthogonal alignment followed by an unchanged peaks probe."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import svd
from threadpoolctl import threadpool_limits

from analyze import ACTIONS, ENCODINGS, average, example_auc, fit_probe, position_prior, read_length

SHUFFLE_SEEDS = tuple(range(1026, 1034))


def rotation(cross):
    left, _, right = svd(cross, full_matrices=False)
    result = left @ right
    np.testing.assert_allclose(result.T @ result, np.eye(result.shape[0]), atol=2e-12)
    return result


def shuffled_indices(counts, rng):
    """Derange within count whenever a count stratum has multiple anchors."""
    result = np.arange(len(counts))
    for k in np.unique(counts):
        same = np.flatnonzero(counts == k)
        if len(same) > 1:
            order = rng.permutation(same)
            result[order] = np.roll(order, 1)
    return result


def load(path):
    with np.load(path, allow_pickle=False) as archive:
        return dict(archive)


def inputs_by_length(data, enc, phase, cached, extra, splits):
    initial = cached['initial']
    peaks = initial if phase == 'initial' else cached['peaks']
    valleys = initial if phase == 'initial' else cached['valleys']
    additions = load(extra / f'{enc}-{phase}.npz')
    assert additions['checkpoint_sha256'].item() == valleys['checkpoint_sha256'].item()
    for split in ('validation', 'test'):
        indices = np.arange(len(data[f'{split}_lengths']))
        assert np.array_equal(peaks[f'{split}_indices'], indices)
        assert np.array_equal(valleys[f'{split}_indices'], indices)
    assert np.array_equal(additions['validation_indices'], valleys['validation_indices'])
    rngs = [np.random.default_rng(seed) for seed in SHUFFLE_SEEDS]
    rows = []
    cross = np.zeros((3, 256, 256), dtype=np.float64)
    shuffled = np.zeros((len(rngs), 256, 256), dtype=np.float64)
    moved, pairs = np.zeros(len(rngs), dtype=int), 0

    for n, split in splits.items():
        fit, dev, eligible, matched, peak_fit, valley_fit, peak, valley_masks = split
        val = data['validation_lengths'] == n
        test = data['test_lengths'] == n
        x = peaks['validation_0_hidden'][val]
        y = [valleys['validation_0_hidden'][val]] + [
            additions[f'validation_{a}_hidden'][val] for a in (1, 2)
        ]
        probe, _ = fit_probe(x, peak_fit, fit, dev)
        own, _ = fit_probe(y[0], valley_fit, fit, dev)
        mu_x = x[fit].astype(np.float64).mean(axis=0)
        mu_y = [z[fit].astype(np.float64).mean(axis=0) for z in y]
        cx = x[fit] - mu_x
        cy = [z[fit] - mu for z, mu in zip(y, mu_y)]
        # Mean-centre by length and give each length equal weight in the map.
        for a, z in enumerate(cy):
            cross[a] += z.T @ cx / fit.sum()
        counts = peak_fit[fit].sum(axis=1)
        pairs += len(counts)
        for j, rng in enumerate(rngs):
            permutation = shuffled_indices(counts, rng)
            moved[j] += np.count_nonzero(permutation != np.arange(len(counts)))
            shuffled[j] += cy[1][permutation].T @ cx / fit.sum()
        rows.append(dict(length=n, eligible=eligible, matched=matched, peak=peak,
                         valley_masks=valley_masks, probe=probe, own=own, mu_x=mu_x,
                         mu_y=mu_y, peak_fit=peak_fit[fit],
                         xt=peaks['test_0_hidden'][test],
                         yt=[valleys[f'test_{a}_hidden'][test] for a in range(3)]))
    return rows, [rotation(c) for c in cross], [rotation(c) for c in shuffled], pairs, moved


def scores_for_length(row, maps, shuffles):
    peak, valleys = row['peak'], row['valley_masks']
    p, q = row['probe'], row['own']
    x, ys = row['xt'], row['yt']
    mx, my = row['mu_x'], row['mu_y']
    yc = ys[1] - my[1]
    # All alignment methods are primarily evaluated on the same complement features
    # with the same centering and labels, isolating the fitted rotation.
    results = {
        'direct': (p(ys[1]), peak),
        'means_only': (p(yc + mx), peak),
        'within_peaks': (p(x), peak),
        'within_valleys': (q(ys[1]), valleys[1]),
        'oracle_count_position_prior': (position_prior(row['peak_fit'], peak), peak),
    }
    for a, name in enumerate(ACTIONS):
        results[f'fit_{name}__test_complement'] = (p(yc @ maps[a] + mx), peak)
        results[f'fit_{name}__test_native'] = (p((ys[a] - my[a]) @ maps[a] + mx), peak)
        results[f'fit_{name}__test_native_own_valley'] = (
            p((ys[a] - my[a]) @ maps[a] + mx), valleys[a])
    for seed, transform in zip(SHUFFLE_SEEDS, shuffles):
        results[f'fit_shuffle_{seed}__test_complement'] = (p(yc @ transform + mx), peak)
    return results


def run(data, features, extra):
    splits = {n: read_length(data, n) for n in range(10, 31)}
    records, map_checks = [], []
    for enc in ENCODINGS:
        cached = {task: load(features / f'{enc}-{task}.npz')
                  for task in ('initial', 'peaks', 'valleys')}
        for phase in ('initial', 'trained'):
            rows, maps, shuffles, pairs, moved = inputs_by_length(
                data, enc, phase, cached, extra, splits)
            map_checks.append(dict(encoding=enc, phase=phase, fit_pairs=pairs,
                                   min_shuffled_fraction=float(moved.min() / pairs)))
            for row in rows:
                for name, (scores, labels) in scores_for_length(row, maps, shuffles).items():
                    auc = example_auc(scores, labels)
                    for support in ('eligible', 'matched'):
                        mask = row[support]
                        records.append(dict(encoding=enc, phase=phase, length=row['length'],
                                            support=support, method=name, n=int(mask.sum()),
                                            finite=int(np.isfinite(auc[mask]).sum()),
                                            auc=average(auc[mask])))
    return pd.DataFrame(records), pd.DataFrame(map_checks)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('data', 'features', 'validation', 'out'):
        parser.add_argument(f'--{flag}', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    data = load(args.data)
    rng = np.random.default_rng(21)
    x = rng.normal(size=(50, 8))
    q, _ = np.linalg.qr(rng.normal(size=(8, 8)))
    np.testing.assert_allclose(rotation((x @ q).T @ x), q.T, atol=2e-12)
    with threadpool_limits(limits=2):
        detail, checks = run(data, args.features, args.validation)
    groups = ['encoding', 'phase', 'support', 'method']
    summary = detail.groupby(groups, as_index=False).agg(auc=('auc', 'mean'),
        lengths=('auc', 'count'), anchors=('n', 'sum'), finite=('finite', 'sum'))
    if not (summary.lengths == 21).all():
        raise ValueError('Every method and condition requires 21 finite length means')
    detail.to_csv(args.out / 'by_length.csv', index=False)
    checks.to_csv(args.out / 'alignment_fit_audit.csv', index=False)
    summary.to_csv(args.out / 'summary.csv', index=False)


if __name__ == '__main__':
    main()
