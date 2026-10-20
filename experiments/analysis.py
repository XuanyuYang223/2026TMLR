"""Descriptive plots and predictions with entire source combinations held out."""
import csv
import html
import json
from pathlib import Path
import numpy as np


def linear_cka(left, right):
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    left = left - left.mean(0)
    right = right - right.mean(0)
    numerator = np.square(left.T @ right).sum()
    denominator = np.sqrt(np.square(left.T @ left).sum() * np.square(right.T @ right).sum())
    return float(numerator / denominator) if denominator > 0 else None


def grouped_predictions(rows, config, group_key='scenario'):
    target_ids = sorted({r['target_id'] for r in rows})
    world_ids = sorted({r['world_seed'] for r in rows})
    baseline = np.array([[r['source_auc'], r['source_accuracy'], r['mean_single_auc']]
                         + [float(r['target_id'] == t) for t in target_ids[1:]]
                         + [float(r['world_seed'] == w) for w in world_ids[1:]] for r in rows])
    structure = np.array([[r['minimum_circuit_size'], r['target_composition_size']] for r in rows])
    truth = np.array([r['transfer_gain'] for r in rows])
    labels = np.array([r[group_key] for r in rows])
    if len(set(labels)) < 2:
        return {'status': 'insufficient_holdout_groups'}
    results = {}
    design = baseline[:, 3:]
    for name, x in [('design_only', design), ('design_plus_structure', np.column_stack([design, structure])),
                    ('baseline', baseline), ('plus_circuit', np.column_stack([baseline, structure[:, 0]])),
                    ('plus_composition', np.column_stack([baseline, structure[:, 1]])),
                    ('plus_both', np.column_stack([baseline, structure]))]:
        predictions = np.zeros(len(truth))
        folds = []
        for label in sorted(set(labels)):
            test = labels == label
            train = ~test
            mean, std = x[train].mean(0), x[train].std(0)
            std = np.where(std > 1e-8, std, 1)
            xt = np.column_stack([np.ones(train.sum()), (x[train] - mean) / std])
            xv = np.column_stack([np.ones(test.sum()), (x[test] - mean) / std])
            penalty = np.eye(xt.shape[1]) * config['prediction_alpha']
            penalty[0, 0] = 0
            weights = np.linalg.solve(xt.T @ xt + penalty, xt.T @ truth[train])
            predictions[test] = xv @ weights
            folds.append({'held_out': str(label), 'mae': float(np.abs(predictions[test] - truth[test]).mean()),
                          'count': int(test.sum())})
        residual = np.square(predictions - truth).sum()
        total = np.square(truth - truth.mean()).sum()
        results[name] = {'r2': float(1 - residual / total) if total > 0 else None,
                         'mae': float(np.abs(predictions - truth).mean()),
                         'rmse': float(np.sqrt(np.mean(np.square(predictions - truth)))),
                         'folds': folds, 'predictions': predictions.tolist()}
    results['incremental'] = {'delta_r2': results['plus_both']['r2'] - results['baseline']['r2'],
                              'mae_reduction': results['baseline']['mae'] - results['plus_both']['mae'],
                              'folds_with_lower_mae': sum(a['mae'] > b['mae'] for a, b in zip(results['baseline']['folds'], results['plus_both']['folds'])),
                              'fold_count': len(set(labels))}
    results['design_incremental'] = {'delta_r2': results['design_plus_structure']['r2'] - results['design_only']['r2'],
                                     'mae_reduction': results['design_only']['mae'] - results['design_plus_structure']['mae']}
    return results


def write_csv(path, rows):
    with Path(path).open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(output, metrics, groups, config):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output = Path(output)
    figures = output / 'figures'
    figures.mkdir(exist_ok=True)
    write_csv(output / 'metrics.csv', metrics)
    primary = [r for r in metrics if r['mode'] == config['primary_mode'] and r['budget'] == config['primary_budget']]
    import hashlib
    analysis_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    predictions = {'analysis_sha256': analysis_hash,
                   'combination_holdout': grouped_predictions(primary, config),
                   'input_basis_holdout': grouped_predictions(primary, config, 'world_seed'),
                   'endpoint': 'target accuracy gain over matched random initialization',
                   'primary_budget': config['primary_budget'], 'primary_mode': config['primary_mode'],
                   'baseline_features': ['source learning-curve AUC', 'source final accuracy', 'mean single-source learning-curve AUC', 'target fixed effects', 'input-basis fixed effects'],
                   'structural_features': ['minimum circuit size', 'minimum number of source tasks whose linear span contains the target'],
                   'notes': ['Ridge penalty fixed before transfer evaluation; feature scaling fit inside each training fold.',
                             'Design-only predictions use no held-out source learning measurements; learning-controlled predictions condition on source fitting.',
                             'Combination holdout removes every seed and target row of the held-out source combination.',
                             'Both circuit families remain in training; this is not prediction of an unseen abstract relation family.']}
    (output / 'prediction_scores.json').write_text(json.dumps(predictions, indent=2))
    run_rows = []
    curves = []
    for g in groups:
        for w in config['world_seeds']:
            for m in config['model_seeds']:
                rid = f"{g['id']}_w{w}_m{m}"
                result = json.loads((output / f'{rid}.json').read_text())
                run_rows.append({'scenario': g['id'], 'family': g['family'], 'world_seed': w, 'model_seed': m,
                                 'source_accuracy': result['source_accuracy'], 'source_min_accuracy': result['source_min_accuracy']})
                curves.extend({'scenario': g['id'], 'world_seed': w, 'model_seed': m, 'step': c['step'],
                               'accuracy': float(np.mean(c['accuracy'])), 'loss': c['loss']} for c in result['source_curve'])
    write_csv(output / 'source_endpoints.csv', run_rows)
    write_csv(output / 'source_learning_curves.csv', curves)
    cka = np.zeros((len(groups), len(groups)))
    initial_cka, cka_rows = [], []
    for w in config['world_seeds']:
        for m in config['model_seeds']:
            representations = [np.load(output / f"{g['id']}_w{w}_m{m}_features.npy") for g in groups]
            initial = np.load(output / f'initial_w{w}_m{m}.npy')
            for i, rep in enumerate(representations):
                initial_cka.append(linear_cka(rep, initial))
                for j, other in enumerate(representations):
                    value = linear_cka(rep, other)
                    cka[i, j] += value / (len(config['world_seeds']) * len(config['model_seeds']))
                    cka_rows.append({'world_seed': w, 'model_seed': m, 'left': groups[i]['id'], 'right': groups[j]['id'], 'linear_cka': value})
    write_csv(output / 'cka.csv', cka_rows)
    summary_rows = []
    for mode in ['probe', 'finetune']:
        for budget in config['budgets']:
            for family in ['A', 'B']:
                rows = [r for r in metrics if r['mode'] == mode and r['budget'] == budget and r['family'] == family]
                if not rows:
                    continue
                world_accuracy = [np.mean([r['test_accuracy'] for r in rows if r['world_seed'] == w]) for w in config['world_seeds']]
                world_gain = [np.mean([r['transfer_gain'] for r in rows if r['world_seed'] == w]) for w in config['world_seeds']]
                summary_rows.append({'mode': mode, 'budget': budget, 'family': family,
                                     'accuracy_mean': float(np.mean(world_accuracy)),
                                     'accuracy_world_sd': float(np.std(world_accuracy, ddof=1)) if len(world_accuracy) > 1 else 0.,
                                     'gain_mean': float(np.mean(world_gain)),
                                     'gain_world_sd': float(np.std(world_gain, ddof=1)) if len(world_gain) > 1 else 0.})
    write_csv(output / 'transfer_summary.csv', summary_rows)
    gate = all(r['source_min_accuracy'] >= config['source_accuracy_gate'] for r in run_rows)
    summary = {'analysis_sha256': analysis_hash, 'pretrained_models': len(run_rows), 'source_combinations': len(groups),
               'input_bases': len(config['world_seeds']), 'initialization_seeds': len(config['model_seeds']),
               'target_count': config['target_count'], 'target_endpoints': len(metrics),
               'source_accuracy_min': min(r['source_min_accuracy'] for r in run_rows),
               'all_sources_pass_accuracy_gate': gate,
               'source_accuracy_partition': 'full source-training domain; this gate measures training fit, not unseen-input generalization',
               'source_examples_per_task': config['p'] ** config['dimension'] * config['pretrain_steps'],
               'mean_cka_to_initialization': float(np.mean(initial_cka)),
               'primary_prediction_incremental': predictions['combination_holdout'].get('incremental'),
               'transfer_summary': summary_rows,
               'limitations': ['This is transductive new-task transfer; every input appears in source training.',
                               'The source accuracy gate measures training fit, not generalization to new inputs.',
                               'Three random input bases give little statistical power; target rows are not independent replications.',
                               'One MLP architecture and one finite-field domain do not establish cross-domain generality.',
                               'No direction of the A-versus-B transfer effect was assumed.',
                               'Held-out combinations are coefficient variants of two known circuit templates.',
                               'Equal marginal output statistics do not guarantee equal computational difficulty.']}
    if not gate:
        summary['limitations'].append('Source-learning gate failed: transfer differences cannot be interpreted as matched learned tasks.')
    (output / 'summary.json').write_text(json.dumps(summary, indent=2))
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    colors = {'A': '#2166ac', 'B': '#d95f02'}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for ax, mode in zip(axes, ['probe', 'finetune']):
        for family in ['A', 'B']:
            rows = [r for r in summary_rows if r['mode'] == mode and r['family'] == family]
            ax.errorbar([r['budget'] for r in rows], [r['accuracy_mean'] for r in rows],
                        yerr=[r['accuracy_world_sd'] for r in rows], marker='o', color=colors[family], label=f'{family}: {3 if family == "A" else 4}-task circuit')
        random_rows = [r for r in metrics if r['mode'] == mode]
        ax.plot(config['budgets'], [np.mean([r['random_accuracy'] for r in random_rows if r['budget'] == b]) for b in config['budgets']], '--', color='gray', label='Matched random initialization')
        ax.axhline(1/config['p'], color='black', linewidth=.7, alpha=.4)
        ax.set(title='Frozen linear probe' if mode == 'probe' else 'Full encoder fine-tuning', xlabel='Labeled target examples', ylabel='Held-out label accuracy', ylim=(0, 1))
        ax.legend(fontsize=8)
    fig.savefig(figures / 'transfer.png', dpi=180)
    fig.savefig(figures / 'transfer.pdf')
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    for family in ['A', 'B']:
        ids = {g['id'] for g in groups if g['family'] == family}
        steps = sorted({r['step'] for r in curves})
        ax.plot(steps, [np.mean([r['accuracy'] for r in curves if r['scenario'] in ids and r['step'] == step]) for step in steps], label=family, color=colors[family])
    ax.set(xlabel='Source training updates', ylabel='Source training accuracy', ylim=(0, 1.02))
    ax.legend()
    fig.savefig(figures / 'source_learning.png', dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    im = ax.imshow(cka, vmin=0, vmax=1, cmap='viridis')
    ax.set_xticks(range(len(groups)), [g['id'] for g in groups], rotation=45)
    ax.set_yticks(range(len(groups)), [g['id'] for g in groups])
    ax.set_title('CKA on identical input rows, averaged over bases')
    fig.colorbar(im, ax=ax)
    fig.savefig(figures / 'cka.png', dpi=180)
    plt.close(fig)
    pred = predictions['combination_holdout']
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    labels = ['Baseline', '+ circuit', '+ composition', '+ both']
    scores = [pred[name]['r2'] for name in ['baseline', 'plus_circuit', 'plus_composition', 'plus_both']]
    ax.bar(labels, scores, color=['#777777', '#2166ac', '#d95f02', '#39824a'])
    ax.axhline(0, color='black', linewidth=.7)
    ax.set(ylabel='Grouped out-of-fold R squared', title=f"Transfer-gain prediction, {config['primary_budget']} labels")
    fig.savefig(figures / 'prediction.png', dpi=180)
    plt.close(fig)
    increment = pred['incremental']
    table = ''.join(f"<tr><td>{r['mode']}</td><td>{r['budget']}</td><td>{r['family']}</td><td>{100*r['accuracy_mean']:.2f}%</td><td>{100*r['gain_mean']:+.2f} pp</td></tr>" for r in summary_rows)
    conclusions = ('Structural features give only a small improvement; current evidence does not support stable incremental prediction.' if increment['delta_r2'] > 0 and increment['mae_reduction'] > 0 and increment['delta_r2'] < .02 else ('Structural features improve mean leave-combination-out prediction in this round; independent combination replication is still needed.' if increment['delta_r2'] > 0 and increment['mae_reduction'] > 0 else 'Structural features do not jointly improve  R²  and  MAE in this round; results do not support incremental predictive ability.'))
    oracle_text = ''
    if (output / 'symbolic_oracle.json').exists():
        oracle = json.loads((output / 'symbolic_oracle.json').read_text())
        oracle_text = f"<p>Post hoc diagnosis: trained  u/v/w  outputs combined using an externally supplied mathematical formula give target test accuracy  {100*oracle['mean_oracle_accuracy']:.2f}%. This shows target information is recoverable from representations of these previously seen inputs; poor ordinary readout does not imply missing information missing. This  oracle  uses an external formula, not spontaneously learned composition.<a href='symbolic_oracle.json'>Diagnosis and per-endpoint </a>.</p>"
    negative_text = ('Both groups have negative mean transfer gains across all label budgets and ordinary-readout settings.' if all(r['gain_mean'] < 0 for r in summary_rows) else 'Transfer gains in different settings are below.')
    document = f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>Algebraic relationships and transfer: first pilot</title>
<style>body{{max-width:1000px;margin:40px auto;padding:0 20px;font:16px/1.7 system-ui;color:#20242b}}img{{max-width:100%}}table{{border-collapse:collapse;width:100%}}td,th{{padding:8px;border-bottom:1px solid #ddd;text-align:left}}code{{background:#eee;padding:2px 4px}}</style>
<h1>Algebraic relationships and transfer: first controlled pilot</h1>
<p>Completed  {len(run_rows)}  source models, {len(groups)}  combinations, {len(config['world_seeds'])}  random input bases, and  {config['target_count']}  shared unseen targets. Minimum source-training accuracy  {100*summary['source_accuracy_min']:.2f}%;{100*config['source_accuracy_gate']:.0f}% Fit-quality threshold: {'passed' if gate else 'not passed'}. This threshold does not test source-task generalization to unseen inputs.</p>
<p>A  has a minimum three-task dependency, while B  requires four tasks. Both groups have four tasks, rank three, uniform marginals, pairwise-independent labels, and identical joint entropy. Each source task receives  {summary['source_examples_per_task']:,}  exposures.</p>
<p><strong>Scope: </strong>Source training covers every input; support/test separation applies only to target labels. This tests new-task transfer on the same entities, not new-entity or out-of-distribution generalization.</p>
<p>The primary result is full-finetuning gain with  {config['primary_budget']}  target labels. Hold out all seed/target rows for each complete source-task combination; baselines control source-learning curves, single-task learning curves, target, and input-basis effects. Adding structure gives  ΔR²={increment['delta_r2']:+.4f},MAE  reduction ={100*increment['mae_reduction']:+.3f}  percentage points, {increment['folds_with_lower_mae']}/{increment['fold_count']}  folds  MAE  improved.</p>
<p>{negative_text} {conclusions}</p>{oracle_text}<img src="figures/transfer.png"><img src="figures/prediction.png">
<table><tr><th>Evaluation</th><th>Label count</th><th>Group</th><th>Accuracy</th><th>Gain over random initialization</th></tr>{table}</table>
<img src="figures/source_learning.png"><img src="figures/cka.png">
<p>Error bars are sample standard deviations across random input bases.{len(metrics)}  target endpoints do not constitute  {len(metrics)}  independent replications. A single  MLP  and finite field do not establish cross-domain universality; coefficient combinations are held out, while training retains both dependency templates.</p>
<p>Raw data: <a href="metrics.csv">metrics.csv</a>;<a href="prediction_scores.json">Grouped predictions and folds</a>;<a href="algebra_audit.json">Exact algebraic audit</a>;<a href="metadata.json">Configuration and environment</a>.</p>
</html>'''
    (output / 'report.html').write_text(document)
    print(json.dumps({k: v for k, v in summary.items() if k != 'transfer_summary'}, indent=2), flush=True)


if __name__ == '__main__':
    import argparse
    from .algebra import scenarios
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='results/pilot')
    args = parser.parse_args()
    output = Path(args.output)
    metadata = json.loads((output / 'metadata.json').read_text())
    config = metadata['config']
    groups = scenarios()
    groups = [g for pair in zip(groups[:4], groups[4:]) for g in pair][:config['scenario_limit']]
    rows = []
    for g in groups:
        for w in config['world_seeds']:
            for m in config['model_seeds']:
                rows.extend(json.loads((output / f"{g['id']}_w{w}_m{m}.json").read_text())['metrics'])
    summarize(output, rows, groups, config)
