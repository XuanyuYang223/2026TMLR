"""Run the identical one-line map fit on two other target specialists."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from alignment import inputs_by_length, load, rotation
from analyze import average, example_auc, read_length


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('data', 'features', 'validation', 'out'):
        parser.add_argument(f'--{flag}', required=True, type=Path)
    args = parser.parse_args()
    data = load(args.data)
    enc = 'one_line'
    cached = {t: load(args.features / f'{enc}-{t}.npz')
              for t in ('initial', 'peaks', 'valleys')}
    splits = {n: read_length(data, n) for n in range(10, 31)}
    validation_n, test_n = data['validation_lengths'], data['test_lengths']
    records = []
    with threadpool_limits(limits=2):
        rows, _, _, _, _ = inputs_by_length(data, enc, 'trained', cached, args.validation, splits)
        for target in ('descents', 'recoils'):
            train = load(args.validation / f'{enc}-{target}.npz')
            test = load(args.features / f'{enc}-{target}.npz')
            assert train['checkpoint_sha256'].item() == test['checkpoint_sha256'].item()
            assert np.array_equal(train['validation_indices'], cached['peaks']['validation_indices'])
            assert np.array_equal(test['test_indices'], cached['peaks']['test_indices'])
            cov = np.zeros((256, 256))
            means = {}
            for row in rows:
                n = row['length']
                fit = splits[n][0]
                val = validation_n == n
                x = cached['peaks']['validation_0_hidden'][val][fit] - row['mu_x']
                y = train['validation_1_hidden'][val][fit].astype(np.float64)
                means[n] = y.mean(axis=0)
                cov += (y - means[n]).T @ x / fit.sum()
            mapping = rotation(cov)
            for row in rows:
                n = row['length']
                y = test['test_1_hidden'][test_n == n]
                p = row['probe']
                scores = {'direct': p(y), 'procrustes': p((y - means[n]) @ mapping + row['mu_x'])}
                for method, score in scores.items():
                    auc = example_auc(score, row['peak'])
                    for support in ('eligible', 'matched'):
                        records.append(dict(target=target, length=n, support=support, method=method,
                                            auc=average(auc[row[support]])))
    detail = pd.DataFrame(records)
    detail.to_csv(args.out / 'other_specialists_by_length.csv', index=False)
    summary = detail.groupby(['target', 'support', 'method'], as_index=False).agg(
        auc=('auc', 'mean'), lengths=('auc', 'count'))
    assert (summary.lengths == 21).all()
    summary.to_csv(args.out / 'other_specialists.csv', index=False)
    print(summary.to_string(index=False))


if __name__ == '__main__':
    main()
