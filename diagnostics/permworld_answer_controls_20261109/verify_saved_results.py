"""Independent fixed-case NumPy verification, run only after the full analysis.

This script never invokes a model or an optimizer. It deliberately does not
import the computational CKA module. Source reads retain the audited adapter.
"""
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.permworld_answer_controls_20261109.common import Sources, load_config

HERE = Path(__file__).resolve().parent
CASES = [
    ('archdev001', 'gru', 'canonical_cycle', 'peaks', 'valleys', 'complement'),
    ('unit02', 'transformer', 'one_line', 'descents', 'recoils', 'inverse'),
    ('unit05', 'gru', 'one_line', 'peaks', 'valleys', 'identity'),
]
LENGTHS = [10, 20, 30]


def read_csv(path):
    with Path(path).open(newline='') as stream:
        return list(csv.DictReader(stream))


def u_gram(features):
    kernel = features @ features.T
    np.fill_diagonal(kernel, 0)
    n = len(kernel)
    row_sum = kernel.sum(axis=1)
    kernel = (kernel - row_sum[:, None] / (n - 2)
              - row_sum[None, :] / (n - 2)
              + row_sum.sum() / ((n - 1) * (n - 2)))
    np.fill_diagonal(kernel, 0)
    return kernel


def u_cka(x, y):
    k, l = u_gram(x), u_gram(y)
    return np.sum(k * l) / np.sqrt(np.sum(k * k) * np.sum(l * l))


def one_hot(labels):
    _, inverse = np.unique(labels, return_inverse=True)
    return np.eye(inverse.max() + 1)[inverse]


def main():
    config = load_config()
    assert json.loads((HERE / 'reproduction.json').read_text())['status'] == 'passed'
    assert json.loads((HERE / 'cka_completion.json').read_text())['status'] == 'completed'
    sources = Sources(config)
    records = []
    for unit, architecture, encoding, left, right, transform in CASES:
        data = sources.data(unit)
        task_a, task_b = config['tasks'].index(left), config['tasks'].index(right)
        action = config['actions'].index(transform)
        for phase in config['phases']:
            x_all = sources.get(unit, architecture, encoding, left, phase, 'test', 'prefix_final')
            y_all = sources.get(unit, architecture, encoding, right, phase, 'test', 'prefix_final')
            base = Path(config['output']) / 'scores' / f'{unit}_{architecture}_{encoding}_{phase}'
            for n in LENGTHS:
                mask = data['test_lengths'] == n
                x = np.asarray(x_all[mask, 0], dtype=np.float64)
                y = np.asarray(y_all[mask, action], dtype=np.float64)
                a = data['test_labels'][mask, 0, task_a]
                b = data['test_labels'][mask, action, task_b]
                blocks = [np.ones((len(a), 1)), one_hot(a)]
                if not np.array_equal(a, b):
                    blocks.append(one_hot(b))
                design = np.column_stack(blocks)
                # An independent least-squares path, rather than saved Q bases.
                rx = x - design @ np.linalg.lstsq(design, x, rcond=None)[0]
                ry = y - design @ np.linalg.lstsq(design, y, rcond=None)[0]
                values = dict(within_length_debiased=u_cka(x, y),
                              within_length_dual_answer_debiased=u_cka(rx, ry))
                saved = {r['cka_version']: r for r in read_csv(base / f'n{n}.csv')
                         if r['source_task'] == left and r['candidate_task'] == right
                         and r['transform'] == transform}
                for version, observed in values.items():
                    expected = float(saved[version]['cka'])
                    error = abs(observed - expected)
                    assert np.isfinite(expected) and error <= 1e-8, (unit, n, phase, version, error)
                    records.append(dict(unit=unit, architecture=architecture, encoding=encoding,
                        source_task=left, candidate_task=right, transform=transform,
                        phase=phase, length=n, version=version, numpy_score=float(observed),
                        saved_score=expected, absolute_difference=float(error)))
    rows = read_csv(HERE / 'cka_results.csv')
    keys = ('unit', 'phase', 'architecture', 'encoding', 'source_task',
            'candidate_task', 'transform', 'cka_version')
    expected_rows = 5 * 2 * 2 * 2 * 16 * 15 * 3 * 3
    assert len(rows) == expected_rows
    assert len({tuple(row[k] for k in keys) for row in rows}) == expected_rows
    assert all(row['source_task'] != row['candidate_task'] for row in rows)
    for row in rows:
        if row['cka'] not in ('', 'None', 'nan'):
            assert -1 - 1e-8 <= float(row['cka']) <= 1 + 1e-8
        assert int(row['samples']) == 10752
    result = dict(status='passed', fixed_cases=CASES, fixed_lengths=LENGTHS,
        independent_numeric_checks=len(records), max_absolute_difference=max(r['absolute_difference'] for r in records),
        aggregate_unique_rows=len(rows), independent_path='NumPy Gram U-centering and direct least squares from saved features; no model forward',
        records=records, consumer_verified_file_count=len(sources.checked),
        consumer_verified_files=sources.checked, no_training=True)
    (HERE / 'independent_verification.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('records', 'consumer_verified_files')}))


if __name__ == '__main__':
    with threadpool_limits(limits=2):
        main()
