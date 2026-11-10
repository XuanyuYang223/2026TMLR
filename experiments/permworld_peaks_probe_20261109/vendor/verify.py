"""Compare released and reproduced count/CKA quantities; spot-check probe AUROC."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

from analyze import ENCODINGS, example_auc, extrema, load_feature_group


def cka(x, y):
    x, y = x.astype(np.float64), y.astype(np.float64)
    x -= x.mean(axis=0)
    y -= y.mean(axis=0)
    xx, yy, xy = x.T @ x, y.T @ y, x.T @ y
    return np.linalg.norm(xy)**2 / (np.linalg.norm(xx) * np.linalg.norm(yy))


def verify_features(data, folder, paper):
    details = []
    for enc in ENCODINGS:
        features = load_feature_group(folder, enc)
        test_idx = features['peaks']['test_indices']
        lengths = data['test_lengths'][test_idx]
        expected = paper.loc[(paper.relation == 'peak_complement') &
                             (paper.architecture == 'gru') & (paper.encoding == enc)].iloc[0]
        for phase in ('initial', 'trained'):
            left = features['initial' if phase == 'initial' else 'peaks']
            right = features['initial' if phase == 'initial' else 'valleys']
            for action, role in enumerate(('identity', 'correct', 'wrong')):
                score = np.mean([cka(left['test_0_hidden'][lengths == n],
                                     right[f'test_{action}_hidden'][lengths == n])
                                 for n in np.unique(lengths)])
                reference = float(expected[f'raw_within_length_linear_cka_{phase}_{role}'])
                if abs(score - reference) >= 2e-5:
                    raise ValueError(f'{enc}/{phase}/{role}: {score} vs. {reference}')
                details.append(dict(encoding=enc, phase=phase, role=role, actual=float(score),
                                    released=reference, absolute_error=abs(score - reference)))
    return details


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--features', type=Path, required=True)
    parser.add_argument('--published', type=Path, required=True)
    parser.add_argument('--learning', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    with np.load(args.data, allow_pickle=False) as data, threadpool_limits(limits=2):
        scores = verify_features(data, args.features, pd.read_csv(args.published))
        peak_matches = set(range(16))
        valley_matches = set(range(16))
        for n in range(10, 31):
            ids = data['test_lengths'] == n
            p = data['test_permutations'][ids, :n]
            labels = data['test_labels'][ids]
            peaks, valleys = extrema(p)
            inverse = np.argsort(p, axis=1) + 1
            peak_matches &= set(np.flatnonzero(np.all(
                labels[:, 0] == peaks.sum(axis=1)[:, None], axis=0)))
            expected_valley = np.column_stack((valleys.sum(axis=1), peaks.sum(axis=1),
                                               extrema(inverse)[1].sum(axis=1)))
            valley_matches &= set(np.flatnonzero(np.all(
                labels == expected_valley[:, :, None], axis=(0, 1))))
        assert len(peak_matches) == len(valley_matches) == 1 and peak_matches != valley_matches

    rng = np.random.default_rng(85116)
    truth = rng.integers(0, 2, size=(128, 12)).astype(bool)
    predicted = rng.integers(0, 5, size=(128, 12))
    actual = example_auc(predicted, truth)
    expected = np.array([roc_auc_score(y, x) if 0 < y.sum() < len(y) else np.nan
                         for y, x in zip(truth, predicted)])
    assert np.allclose(actual, expected, atol=1e-14, rtol=0, equal_nan=True)
    learn = pd.read_csv(args.learning)
    reproduced = learn[learn.encoding == 'canonical_cycle'].groupby('task')[['accuracy', 'baseline']].mean()
    references = {'peaks': (0.4069010416666667, 0.39815848214285715),
                  'valleys': (0.4109002976190476, 0.3931361607142857)}
    for task, reference in references.items():
        assert np.allclose(reproduced.loc[task], reference, atol=1e-14, rtol=0)
    result = dict(cka_checks=scores, cka_max_absolute_error=max(x['absolute_error'] for x in scores),
                  exact_oracle_lengths=21, peak_label_column=int(next(iter(peak_matches))),
                  valley_label_column=int(next(iter(valley_matches))), auc_independent_checks=128,
                  count_accuracy_and_length_baseline_checks=4, status='passed')
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
