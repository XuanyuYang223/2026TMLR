"""Reports and provenance checks for the new PermWorld combination pilot."""
import argparse
import csv
import hashlib
from html import escape
from itertools import combinations
import json
from pathlib import Path
import sys

import numpy as np

from .analysis import linear_cka, write_csv


def read_csv(path):
    with Path(path).open() as handle:
        return list(csv.DictReader(handle))


def within_length_center(values, lengths):
    centered = values.copy().astype(np.float64)
    for n in np.unique(lengths):
        mask = lengths == n
        centered[mask] -= centered[mask].mean(axis=0)
    return centered


def analyze(output):
    output = Path(output)
    metadata = json.loads((output / 'metadata.json').read_text())
    config, groups = metadata['config'], metadata['groups']
    data = {key: value for key, value in np.load(output / 'data.npz').items()}
    sys.path.insert(0, str(Path(config['repository']) / 'src'))
    from neurips_permutations.math_ops import PROPERTY32_TASK_NAMES
    names = list(PROPERTY32_TASK_NAMES)
    lengths = data['representation_lengths']
    group_order = [g['id'] for g in groups]
    source_rows, curve_rows, cka_rows, consistency = [], [], [], []
    features = {}
    seeds = config['model_seeds']
    for seed in seeds:
        for group in groups:
            run_id = f"{group['id']}_s{seed}"
            run = json.loads((output / f'{run_id}.json').read_text())
            for split in ('source_train', 'source_validation'):
                source_rows.extend({'group': group['id'], 'seed': seed, 'split': split, **row} for row in run[split])
            curve_rows.extend({'group': group['id'], 'seed': seed, **row} for row in run['curve'])
            feature = np.load(output / f'{run_id}_features.npy')
            features[group['id'], seed, 'raw'] = feature
            features[group['id'], seed, 'within_length'] = within_length_center(feature, lengths)
        for mode in ('raw', 'within_length'):
            for left, right in combinations(group_order, 2):
                cka_rows.append({'seed': seed, 'left': left, 'right': right, 'centering': mode,
                                 'linear_cka': linear_cka(features[left, seed, mode], features[right, seed, mode])})
    for group in groups:
        for first, second in combinations(seeds, 2):
            for mode in ('raw', 'within_length'):
                a = np.load(output / f'initial_s{first}_features.npy')
                b = np.load(output / f'initial_s{second}_features.npy')
                if mode == 'within_length':
                    a, b = within_length_center(a, lengths), within_length_center(b, lengths)
                consistency.append({'group': group['id'], 'seed_a': first, 'seed_b': second, 'centering': mode,
                                    'trained_cka': linear_cka(features[group['id'], first, mode], features[group['id'], second, mode]),
                                    'random_cka': linear_cka(a, b)})
    source_summary = []
    prior_rows, label_statistics = [], []
    for group in groups:
        rows = [r for r in source_rows if r['group'] == group['id']]
        prior_accuracies = []
        for task in group['tasks']:
            predictions = np.zeros(len(data['validation_lengths']), dtype=np.int64)
            for n in np.unique(data['train_lengths']):
                training_labels = data['train_labels'][data['train_lengths'] == n, names.index(task)]
                predictions[data['validation_lengths'] == n] = np.bincount(training_labels).argmax()
            accuracy = float(np.mean(predictions == data['validation_labels'][:, names.index(task)]))
            prior_accuracies.append(accuracy)
            prior_rows.append({'group': group['id'], 'task': task, 'validation_accuracy': accuracy,
                               'method': 'train-label majority separately within each length'})
        entropies, joint_entropies, correlations = [], [], []
        for n in np.unique(data['train_lengths']):
            y = data['train_labels'][data['train_lengths'] == n][:, [names.index(t) for t in group['tasks']]]
            for column in y.T:
                _, counts = np.unique(column, return_counts=True)
                probabilities = counts / counts.sum()
                entropies.append(float(-np.sum(probabilities * np.log(probabilities))))
            _, counts = np.unique(y, axis=0, return_counts=True)
            probabilities = counts / counts.sum()
            joint_entropies.append(float(-np.sum(probabilities * np.log(probabilities))))
            sigma = y.std(axis=0)
            z = (y - y.mean(axis=0)) / np.where(sigma > 0, sigma, 1)
            correlation = z.T @ z / len(y)
            correlations.extend(abs(correlation[a, b]) for a, b in combinations(range(4), 2))
        label_statistics.append({'group': group['id'], 'mean_conditional_entropy': float(np.mean(entropies)),
                                 'empirical_conditional_joint_entropy': float(np.mean(joint_entropies)),
                                 'mean_abs_within_length_correlation': float(np.mean(correlations))})
        val_acc = float(np.mean([r['accuracy'] for r in rows if r['split'] == 'source_validation']))
        source_summary.append({'group': group['id'], 'family': group['family'], 'minimum_constraint_size': group['minimum_constraint_size'],
                               'train_accuracy': float(np.mean([r['accuracy'] for r in rows if r['split'] == 'source_train'])),
                               'validation_accuracy': val_acc, 'length_majority_validation_accuracy': float(np.mean(prior_accuracies)),
                               'validation_gain_over_length_majority': val_acc - float(np.mean(prior_accuracies)),
                               'validation_min_task_accuracy': float(min(r['accuracy'] for r in rows if r['split'] == 'source_validation')),
                               'null_match_distance': group.get('null_match_distance')})
    endpoints = read_csv(output / 'transfer_endpoints.csv')
    random = {(r['seed'], r['target'], r['budget'], r['mode']): float(r['accuracy']) for r in endpoints if r['group'] == 'random'}
    transfer_rows = []
    for row in endpoints:
        if row['group'] == 'random':
            continue
        baseline = random[row['seed'], row['target'], row['budget'], row['mode']]
        transfer_rows.append({**row, 'accuracy': float(row['accuracy']), 'random_accuracy': baseline,
                              'gain': float(row['accuracy']) - baseline})
    transfer_summary = []
    for group in groups:
        for budget in config['target_budgets']:
            for mode in ('probe', 'finetune'):
                selected = [r for r in transfer_rows if r['group'] == group['id'] and int(r['budget']) == budget and r['mode'] == mode]
                seed_gain = [np.mean([r['gain'] for r in selected if int(r['seed']) == seed]) for seed in seeds]
                transfer_summary.append({'group': group['id'], 'family': group['family'], 'minimum_constraint_size': group['minimum_constraint_size'],
                                         'budget': budget, 'mode': mode, 'accuracy': float(np.mean([r['accuracy'] for r in selected])),
                                         'random_accuracy': float(np.mean([r['random_accuracy'] for r in selected])),
                                         'gain': float(np.mean(seed_gain)), 'gain_seed_sd': float(np.std(seed_gain, ddof=1)),
                                         'positive_seed_mean_count': int(sum(g > 0 for g in seed_gain))})
    primary = [r for r in transfer_summary if r['budget'] == config['primary_budget'] and r['mode'] == config['primary_mode']]
    relation_contrasts = []
    for family in ('cycle', 'position', 'interior'):
        null = next(r for r in primary if r['group'] == f'{family}_none')
        for row in primary:
            if row['family'] == family and row['minimum_constraint_size']:
                relation_contrasts.append({'family': family, 'group': row['group'],
                                          'gain_minus_null': row['gain'] - null['gain'],
                                          'accuracy_minus_null': row['accuracy'] - null['accuracy']})
    summary = {'source_model_count': len(groups) * len(seeds), 'groups': source_summary,
               'pretrained_transfer_endpoints': len(transfer_rows), 'random_transfer_endpoints': len(random),
               'primary_budget': config['primary_budget'], 'primary_mode': config['primary_mode'],
               'transfer_summary': transfer_summary, 'primary_relation_contrasts': relation_contrasts,
               'scope': 'fresh disjoint permutations within the same length range; different targets but no out-of-length generalization',
               'limitations': config['limitations'], 'analysis_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    for filename, rows in [('source_accuracy.csv', source_rows), ('source_curves.csv', curve_rows), ('cka.csv', cka_rows),
                           ('cross_seed_cka.csv', consistency), ('source_summary.csv', source_summary),
                           ('source_length_majority.csv', prior_rows), ('source_label_statistics.csv', label_statistics),
                           ('transfer_gains.csv', transfer_rows), ('transfer_summary.csv', transfer_summary),
                           ('primary_relation_contrasts.csv', relation_contrasts)]:
        write_csv(output / filename, rows)
    (output / 'summary.json').write_text(json.dumps(summary, indent=2))
    figures(output, source_summary, transfer_summary, cka_rows, config, group_order)
    table = ''.join(f"<tr><td>{escape(g['id'])}</td><td>{g['minimum_constraint_size'] or 'No proven constraint'}</td><td>{escape(', '.join(g['tasks']))}</td><td>{float(g['selection_statistics']['mean_conditional_entropy']):.3f}</td><td>{float(g['selection_statistics']['mean_abs_within_length_correlation']):.3f}</td></tr>" for g in groups)
    outcomes = ''.join(f"<tr><td>{escape(r['group'])}</td><td>{100*r['accuracy']:.2f}%</td><td>{100*r['random_accuracy']:.2f}%</td><td>{100*r['gain']:+.2f} pp</td><td>{r['positive_seed_mean_count']}/3</td></tr>" for r in primary)
    effects = ''.join(f"<tr><td>{escape(r['group'])}</td><td>{100*r['gain_minus_null']:+.2f} pp</td></tr>" for r in relation_contrasts)
    low, high = min(r['validation_accuracy'] for r in source_summary), max(r['validation_accuracy'] for r in source_summary)
    html = f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>PermWorld New task-combination pilot</title>
<style>body{{max-width:1200px;margin:40px auto;padding:0 20px;font:16px/1.7 system-ui;color:#18202b}}img{{max-width:100%}}table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{padding:8px;border-bottom:1px solid #ddd;text-align:left}}aside{{background:#f2f4f7;padding:16px;margin:18px 0}}</style>
<h1>PermWorld:8  new four-task combinations, 24  source models</h1>
<p>All source-task counts are  4; three initialization seeds share all inputs/training schedules, with each task exposed  {config['pretrain_steps']*config['examples_per_task_per_step']:,}  times. Uses original property functions, Passage  numeric vocabulary, and causal  Transformer; width  {config['d_model']}, {config['layers']}  layers, trained for  {config['pretrain_steps']}  steps, predicting only answer/ token. This is small-scale screening of new combinations, not a full-scale replication of the original paper.</p>
<p>Three-task relationships include  fixed+exceed+def=n, fixed+nontrivial=cycles; four-task relationships eliminate intermediate properties, or  peaks+valleys+double_ascents+double_descents=n−2. Unconstrained means only no proven same-input rational linear identity, not no other mathematical relationship. Controls use independently generated pre-training label statistics, without viewing  CKA  or transfer.</p>
<table><tr><th>Combination</th><th>Minimum constraint size</th><th>Tasks</th><th>Mean conditional entropy</th><th>Mean within-length  |correlation|</th></tr>{table}</table>
<aside>The table gives selection statistics from the earlier independent design sample; <a href="source_label_statistics.csv">current source-training label statistics</a> are saved separately. Label statistics are only approximately matched, particularly poorly for the peaks/valleys controls. Source difficulty, complete output distributions, and joint entropy are not strictly controlled. Across-combination source-validation accuracy spans  {100*low:.1f}%–{100*high:.1f}%, so differences cannot be causally attributed to identity order. Single-task difficulty controls are not yet available.</aside>
<h2>Source learning and representations</h2><img src="source_learning.png"><img src="representation.png">
<p>Source validation is also compared against a permutation-length modal-class baseline derived from training labels only, avoiding equating label-frequency fit with full mathematical-task learning. Per-task baselines:  <a href="source_length_majority.csv">Length modal class</a>; per-combination differences:  <a href="source_summary.csv">Source-learning summary</a>. With insufficient source learning, CKA /transfer differences cannot establish use of higher-order structure.</p>
<p>CKA  measured on the same  420  task-independent prefixes, at ONE_END  positions. Raw  CKA  and within-length mean-centered  CKA are reported together to inspect length information. Combinations have paired initials; cross-seed results within combinations are separate. This does not redo correct/wrong input-transformation controls or change the original  24/24  final-layer  CKA  increase result.</p>
<h2>Shared unseen-task transfer</h2>
<p>Target tasks are  {escape(', '.join(config['target_tasks']))}, absent from every source combination. Target support/test, source training/validation, and representation inputs are five globally non-overlapping groups, all at lengths  10–30. Support budgets  64, 256 use uniform input sampling, are nested/shared across conditions, and do not enforce label balance. Random models use paired initials and identical support sets, updates, and learning rates.</p>
<img src="transfer.png"><p>The primary endpoint was fixed in advance as full-model finetuning with  256  labels; the table averages four targets and three initialization seeds. Error bars describe only the sample SD of the three initialization means, not independent-data replication or significance.</p>
<table><tr><th>Combination</th><th>Accuracy</th><th>Random-model accuracy</th><th>Transfer gain</th><th>Positive seed means</th></tr>{outcomes}</table>
<p>Identity-containing combinations minus the same-series no-proven-constraint control: </p><table><tr><th>Combination</th><th>Transfer-gain difference</th></tr>{effects}</table>
<aside>This tests new-task transfer on new inputs, not new lengths or mathematical domains. One data-generation seed is used; three initialization seeds are not three independent data worlds. All combinations, targets, budgets, and both readouts are retained; test accuracy did not select combinations or tune parameters. Differences generate future hypotheses, not confirm universal structural laws.</aside>
<p>Data: <a href="metadata.json">Fixed protocol and combinations</a>;<a href="summary.json">Summary</a>;<a href="source_accuracy.csv">Source-task accuracy</a>;<a href="cka.csv">CKA</a>;<a href="cross_seed_cka.csv">Cross-seed CKA</a>;<a href="transfer_gains.csv">Transfer endpoints</a>;<a href="support_indices.json">Support sets</a>;<a href="verification.json">Verification</a>.</p></html>'''
    (output / 'report.html').write_text(html)
    verify(output)
    print(json.dumps({key: summary[key] for key in ('source_model_count', 'pretrained_transfer_endpoints', 'random_transfer_endpoints', 'primary_relation_contrasts')}, indent=2))


def figures(output, source, transfer, cka, config, order):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    x = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(12, 4.5), constrained_layout=True)
    ax.bar(x - .25, [r['train_accuracy'] for r in source], .25, label='Source training')
    ax.bar(x, [r['validation_accuracy'] for r in source], .25, label='Unseen-input source validation')
    ax.bar(x + .25, [r['length_majority_validation_accuracy'] for r in source], .25, label='Length-conditioned label majority')
    ax.set_xticks(x, order, rotation=30, ha='right')
    ax.set(ylabel='Mean source answer accuracy', ylim=(0, 1), title='Equal exposure; source fitting is measured, not matched')
    ax.legend()
    fig.savefig(output / 'source_learning.png', dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    for ax, mode in zip(axes, ('raw', 'within_length')):
        matrix = np.eye(len(order))
        for i, left in enumerate(order):
            for j, right in enumerate(order):
                if i < j:
                    matrix[i, j] = matrix[j, i] = np.mean([r['linear_cka'] for r in cka if r['left'] == left and r['right'] == right and r['centering'] == mode])
        im = ax.imshow(matrix, vmin=0, vmax=1, cmap='viridis')
        ax.set_xticks(x, order, rotation=50, ha='right', fontsize=8)
        ax.set_yticks(x, order, fontsize=8)
        ax.set_title(f'Task-free final-layer CKA: {mode}')
        fig.colorbar(im, ax=ax, shrink=.65)
    fig.savefig(output / 'representation.png', dpi=180)
    fig.savefig(output / 'representation.pdf')
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)
    for ax, mode in zip(axes, ('probe', 'finetune')):
        for k, budget in enumerate(config['target_budgets']):
            rows = [next(r for r in transfer if r['group'] == group and r['budget'] == budget and r['mode'] == mode) for group in order]
            ax.bar(x + (k - .5) * .36, [100*r['gain'] for r in rows], .36,
                   yerr=[100*r['gain_seed_sd'] for r in rows], capsize=2, label=f'{budget} labels')
        ax.axhline(0, color='black', linewidth=.8)
        ax.set_xticks(x, order, rotation=35, ha='right', fontsize=8)
        ax.set(title=f'New-input target transfer: {mode}', ylabel='Gain over matched random model (pp)')
        ax.legend()
    fig.savefig(output / 'transfer.png', dpi=180)
    fig.savefig(output / 'transfer.pdf')
    plt.close(fig)


def verify(output):
    output = Path(output)
    metadata = json.loads((output / 'metadata.json').read_text())
    signature = {k: v for k, v in metadata.items() if k not in ('fingerprint', 'cka_landmark', 'source_objective')}
    assert hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest() == metadata['fingerprint']
    assert hashlib.sha256(Path('experiments/permworld_combinations.py').read_bytes()).hexdigest() == metadata['code_sha256']
    repo = Path(metadata['config']['repository'])
    for name, expected in metadata['upstream_files'].items():
        assert hashlib.sha256((repo / 'src/neurips_permutations' / name).read_bytes()).hexdigest() == expected
    assert hashlib.sha256(Path(metadata['config']['candidate_table']).read_bytes()).hexdigest() == metadata['candidate_sha256']
    from .permworld_combinations import select_groups
    assert select_groups(metadata['config']) == metadata['groups']
    source_count = 0
    for seed in metadata['config']['model_seeds']:
        for group in metadata['groups']:
            run_id = f"{group['id']}_s{seed}"
            record = json.loads((output / f'{run_id}.json').read_text())
            assert record['fingerprint'] == metadata['fingerprint']
            assert hashlib.sha256((output / 'checkpoints' / f'{run_id}.pt').read_bytes()).hexdigest() == record['checkpoint_sha256']
            assert record['exposures_per_task'] == metadata['config']['pretrain_steps'] * metadata['config']['examples_per_task_per_step']
            source_count += 1
    with np.load(output / 'data.npz') as data:
        arrays = [data[f'{split}_input'] for split in metadata['config']['examples_per_length']]
        tuples = [tuple(row) for array in arrays for row in array]
        assert len(tuples) == len(set(tuples)) == 6930
        support = json.loads((output / 'support_indices.json').read_text())
        order = np.random.default_rng(metadata['config']['data_seed'] + 50000).permutation(len(data['support_pool_input']))
        for budget in metadata['config']['target_budgets']:
            assert support[str(budget)] == order[:budget].tolist()
    endpoints = read_csv(output / 'transfer_endpoints.csv')
    keys = [(r['run_id'], r['target'], r['budget'], r['mode']) for r in endpoints]
    assert len(keys) == len(set(keys)) == 432
    assert source_count == 24
    assert len([r for r in endpoints if r['group'] == 'random']) == 48
    for row in endpoints:
        assert 0 <= float(row['accuracy']) <= 1
    for path in output.glob('*_transfer.json'):
        assert json.loads(path.read_text())['fingerprint'] == metadata['fingerprint']
    cka = read_csv(output / 'cka.csv')
    assert len(cka) == 168
    assert all(0 <= float(row['linear_cka']) <= 1 + 1e-10 for row in cka)
    from html.parser import HTMLParser
    class Links(HTMLParser):
        def __init__(self):
            super().__init__()
            self.values = []
        def handle_starttag(self, tag, attrs):
            self.values.extend(v for k, v in attrs if k in ('href', 'src'))
    links = Links()
    links.feed((output / 'report.html').read_text())
    result = {'status': 'passed', 'source_models': source_count, 'unique_transfer_endpoints': len(keys),
              'globally_disjoint_inputs': len(tuples), 'nested_shared_support': True,
              'training_and_checkpoint_fingerprints': True, 'group_selection_reproducible': True,
              'equal_source_exposure': True, 'cka_endpoints': len(cka),
              'analysis_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'data_sha256': hashlib.sha256((output / 'data.npz').read_bytes()).hexdigest()}
    (output / 'verification.json').write_text(json.dumps(result, indent=2))
    assert all((output / link).is_file() for link in links.values)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='results/permworld_combinations')
    args = parser.parse_args()
    analyze(args.output)
