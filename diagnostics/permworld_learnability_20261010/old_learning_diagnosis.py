"""Read-only task-learning diagnosis of the completed sixteen-model development.

Uses supervised validation curves, cached prediction IDs and exact label counts.
Does not load geometry, CKA, Delta, model weights, hidden activations or output
probabilities. Test endpoints are explicitly old descriptive observations.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


SOURCE = Path('/mnt/yangx-projects-f/20261009/permworld_new_artifacts/encoding_archdev')
OUTPUT = Path(__file__).parent
TASKS = ('descents', 'recoils', 'peaks', 'valleys')
LABEL_TASKS = ('anti_fixed_points', 'components', 'deficiencies', 'descents',
               'exceedances', 'fixed_points', 'global_descents', 'lds_length',
               'left_to_right_maxima', 'left_to_right_minima', 'lis_length',
               'longest_decreasing_run', 'longest_increasing_run', 'peaks',
               'recoils', 'valleys')
ARCHS = ('transformer', 'gru')
ENCODINGS = ('one_line', 'canonical_cycle')
ACTIONS = ('identity', 'complement', 'inverse')
LENGTHS = tuple(range(10, 31))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(2 ** 20), b''):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(source, output):
    output.mkdir(parents=True, exist_ok=True)
    artifacts = []

    def record(path, purpose):
        item = dict(path=str(path), sha256=sha(path), purpose=purpose,
                    bytes=path.stat().st_size)
        artifacts.append(item)
        return item['sha256']

    protocol_path = source / 'protocol.json'
    protocol = json.loads(protocol_path.read_text())
    protocol_sha = record(protocol_path, 'training manifest and fixed lengths/tasks')
    assert tuple(protocol['tasks']) == TASKS
    assert tuple(protocol['lengths']) == LENGTHS
    assert len(protocol['training_manifest']) == 16
    data_path = source / 'worlds/archdev001/data.npz'
    data_sha = record(data_path, 'only lengths and exact label arrays are decoded')
    with np.load(data_path) as data:
        arrays = {f'{split}_{field}': data[f'{split}_{field}'].copy()
                  for split in ('train', 'validation', 'test')
                  for field in ('lengths', 'labels')}
    audit_path = source / 'worlds/archdev001/data_audit.json'
    data_audit = json.loads(audit_path.read_text())
    record(audit_path, 'frozen source label/data provenance')
    assert data_audit['data_sha256'] == data_sha
    reference = {}
    counts = []
    for task in TASKS:
        ti = LABEL_TASKS.index(task)
        for n in LENGTHS:
            labels = arrays['train_labels'][arrays['train_lengths'] == n, 0, ti]
            distribution = np.bincount(labels, minlength=31)
            assert distribution.size == 31 and len(labels) == 4096
            mode = int(np.flatnonzero(distribution == distribution.max())[0])
            reference[task, n] = mode
            counts.extend(dict(task=task, length=n, label=k, training_count=int(value),
                               training_majority_label=mode, tie_policy='smallest_label')
                          for k, value in enumerate(distribution))

    rows = []
    curves = []
    source_trends = []
    validation_final_verified = 0
    for job in protocol['training_manifest']:
        run_id = job['run_id']
        done_path = source / 'models' / run_id / 'completed.json'
        done = json.loads(done_path.read_text())
        record(done_path, 'real fixed-20k validation learning curve and run provenance')
        assert done['step'] == 20000
        assert done['protocol_sha256'] == protocol_sha
        assert done['data_sha256'] == data_sha
        assert len(done['curve']) == 20
        arch, enc, task = job['architecture'], job['encoding'], job['task']
        ti = LABEL_TASKS.index(task)
        for item in done['curve']:
            curves.append(dict(run_id=run_id, architecture=arch, encoding=enc, task=task,
                               step=item['step'], validation_answer_accuracy=item['answer_accuracy'],
                               last_update_training_loss=item['train_loss'],
                               validation_eos_accuracy=item['eos_accuracy'], samples=item['samples']))
        curve = {int(item['step']): item for item in done['curve']}
        source_trends.append(dict(run_id=run_id, architecture=arch, encoding=enc, task=task,
            validation_1k=curve[1000]['answer_accuracy'], validation_5k=curve[5000]['answer_accuracy'],
            validation_10k=curve[10000]['answer_accuracy'], validation_20k=curve[20000]['answer_accuracy'],
            validation_10k_to_20k_gain_pp=100 * (curve[20000]['answer_accuracy'] - curve[10000]['answer_accuracy']),
            validation_1k_to_20k_gain_pp=100 * (curve[20000]['answer_accuracy'] - curve[1000]['answer_accuracy']),
            early_1k_5k_median_last_update_loss=float(np.median([curve[k]['train_loss'] for k in range(1000, 5001, 1000)])),
            late_16k_20k_median_last_update_loss=float(np.median([curve[k]['train_loss'] for k in range(16000, 20001, 1000)])),
            fixed_windows_not_best_checkpoint=True))
        for split in ('validation', 'test'):
            cache_path = source / 'evaluation_cache' / f'{arch}_{enc}_{task}_trained_{split}.npz'
            metadata_path = cache_path.with_suffix('.json')
            metadata = json.loads(metadata_path.read_text())
            record(metadata_path, 'saved prediction cache provenance')
            cache_sha = record(cache_path, 'hash cache bytes; decode only prediction IDs, not representations')
            assert metadata['sha256'] == cache_sha
            assert metadata['signature']['protocol_sha256'] == protocol_sha
            assert metadata['signature']['data_sha256'] == data_sha
            assert metadata['signature']['checkpoint_sha256'] == done['checkpoint_sha256']
            with np.load(cache_path) as cache:
                predictions = cache['prediction'].copy()
            ns, truth_all = arrays[f'{split}_lengths'], arrays[f'{split}_labels'][:, :, ti]
            assert predictions.shape == truth_all.shape == (len(ns), 3)
            if split == 'validation':
                assert abs(float(np.mean(predictions[:, 0] == truth_all[:, 0])) - curve[20000]['answer_accuracy']) < 1e-12
                validation_final_verified += 1
            for action_id, action in enumerate(ACTIONS):
                for n in LENGTHS:
                    mask = ns == n
                    prediction, truth = predictions[mask, action_id], truth_all[mask, action_id]
                    baseline = reference[task, n]
                    baseline_correct = truth == baseline
                    model_correct = prediction == truth
                    differences = model_correct.astype(float) - baseline_correct.astype(float)
                    nonmajority = ~baseline_correct
                    row = dict(run_id=run_id, architecture=arch, encoding=enc, task=task,
                               split=split, usage='learning_diagnosis' if split == 'validation' else 'old_test_descriptive_only',
                               action=action, length=n, samples=len(truth), training_majority_label=baseline,
                               full_vocab_accuracy=float(model_correct.mean()),
                               training_length_majority_accuracy=float(baseline_correct.mean()),
                               gain_over_training_length_majority_pp=100 * float(differences.mean()),
                               paired_sampling_se_pp=100 * float(differences.std(ddof=1)) / np.sqrt(len(truth)),
                               observed_above_baseline=bool(differences.mean() > 0),
                               prediction_agreement_with_length_majority=float(np.mean(prediction == baseline)),
                               nonmajority_examples=int(nonmajority.sum()),
                               nonmajority_correct=int(np.sum(model_correct & nonmajority)),
                               nonmajority_accuracy=float(model_correct[nonmajority].mean()),
                               majority_examples=int(baseline_correct.sum()),
                               majority_accuracy=float(model_correct[baseline_correct].mean()),
                               gained_examples=int(np.sum(model_correct & ~baseline_correct)),
                               lost_examples=int(np.sum(~model_correct & baseline_correct)))
                    assert row['gained_examples'] - row['lost_examples'] == int(differences.sum())
                    rows.append(row)
    assert len(rows) == 2016 and validation_final_verified == 16
    summaries = []
    for job in protocol['training_manifest']:
        for split in ('validation', 'test'):
            for action in ACTIONS:
                selected = [row for row in rows if row['run_id'] == job['run_id'] and row['split'] == split and row['action'] == action]
                assert len(selected) == 21
                fields = ('full_vocab_accuracy', 'training_length_majority_accuracy',
                          'gain_over_training_length_majority_pp', 'prediction_agreement_with_length_majority',
                          'nonmajority_accuracy', 'majority_accuracy')
                summary = dict(run_id=job['run_id'], architecture=job['architecture'], encoding=job['encoding'],
                               task=job['task'], split=split, action=action,
                               usage='learning_diagnosis' if split == 'validation' else 'old_test_descriptive_only')
                summary.update({field: float(np.mean([row[field] for row in selected])) for field in fields})
                summary['observed_above_baseline_lengths'] = ','.join(str(row['length']) for row in selected if row['observed_above_baseline'])
                summary['observed_above_baseline_length_count'] = sum(row['observed_above_baseline'] for row in selected)
                summary['minimum_length_gain_pp'] = min(row['gain_over_training_length_majority_pp'] for row in selected)
                summary['maximum_length_gain_pp'] = max(row['gain_over_training_length_majority_pp'] for row in selected)
                summaries.append(summary)

    stem = 'old_learning_diagnosis'
    write_csv(output / f'{stem}_length.csv', rows)
    write_csv(output / f'{stem}_summary.csv', summaries)
    write_csv(output / f'{stem}_curves.csv', curves)
    write_csv(output / f'{stem}_trends.csv', source_trends)
    write_csv(output / f'{stem}_train_label_counts.csv', counts)

    plots = []
    colors = ('#355f9c', '#88aad0')
    for field in ('validation_answer_accuracy', 'last_update_training_loss'):
        fig, axes = plt.subplots(4, 2, figsize=(12, 13), sharex=True)
        for ti, task in enumerate(TASKS):
            for ai, arch in enumerate(ARCHS):
                axis = axes[ti, ai]
                for ei, enc in enumerate(ENCODINGS):
                    values = sorted([row for row in curves if row['task'] == task and row['architecture'] == arch and row['encoding'] == enc], key=lambda row: row['step'])
                    axis.plot([row['step'] for row in values], [row[field] for row in values], color=colors[ei], label=enc)
                if field == 'validation_answer_accuracy':
                    base = next(row for row in summaries if row['task'] == task and row['architecture'] == arch and row['encoding'] == 'one_line' and row['split'] == 'validation' and row['action'] == 'identity')['training_length_majority_accuracy']
                    axis.axhline(base, linestyle=':', color='black', label='training-defined length majority')
                    axis.set_ylim(0, 1.03)
                axis.set_title(f'{task} / {arch}')
                axis.grid(alpha=.2)
                if ai == 0: axis.set_ylabel('Validation accuracy' if field == 'validation_answer_accuracy' else 'Last-update loss')
                if ti == 3: axis.set_xlabel('fixed training update')
        axes[0, 0].legend(fontsize=8)
        fig.suptitle('Validation learning; fixed final20k, all sixteen models; minibatch loss is noisy')
        fig.tight_layout()
        path = output / f'{stem}_{field}.png'
        fig.savefig(path, dpi=150, bbox_inches='tight');plt.close(fig);plots.append(path.name)
    for split in ('validation', 'test'):
        fig, axes = plt.subplots(4, 2, figsize=(12, 13), sharex=True)
        for ti, task in enumerate(TASKS):
            for ai, arch in enumerate(ARCHS):
                axis = axes[ti, ai]
                for ei, enc in enumerate(ENCODINGS):
                    selected = [row for row in rows if row['task'] == task and row['architecture'] == arch and row['encoding'] == enc and row['split'] == split and row['action'] == 'identity']
                    axis.plot([row['length'] for row in selected], [row['gain_over_training_length_majority_pp'] for row in selected], color=colors[ei], marker='.', label=enc)
                axis.axhline(0, linestyle=':', color='black')
                axis.set_title(f'{task} / {arch}')
                axis.grid(alpha=.2)
                if ai == 0: axis.set_ylabel('Gain over length baseline (pp)')
                if ti == 3: axis.set_xlabel('permutation length')
        axes[0, 0].legend(fontsize=8)
        fig.suptitle('Validation diagnosis' if split == 'validation' else 'Old test: descriptive only, never choose lengths from this panel')
        fig.tight_layout()
        path = output / f'{stem}_{split}_length_gain.png'
        fig.savefig(path, dpi=150, bbox_inches='tight');plt.close(fig);plots.append(path.name)

    validation = [row for row in summaries if row['split'] == 'validation' and row['action'] == 'identity']
    txt = [
        'Historical16-model learnability diagnostic: primary analysis uses validation only; historical test results are separately descriptive.',
        'Read only fixed 20k completed.json curves, labels/lengths, and cached prediction IDs; no CKA, Delta, hidden states, output probabilities, or weights.',
        'This is one data-world/source-initialization development unit, without multiple-world/new-source confirmation. All 16 conditions and lengths10..30 are retained.',
        'Training length modes use4096 training labels/length with ties to the smallest answer; validation has256/length and historical test512/length.',
        'Curves save aggregated validation accuracy only every1000 steps; per-length predictions exist only at the endpoint. There are no per-length per-step validation trajectories or validation losses to claim as available.',
        'train_loss is microbatch loss from one update, not whole-training/validation loss; spikes alone cannot establish degradation. Fixed1k..5k and16k..20k medians describe early/late trends only, without best-checkpoint selection.',
        'Whether each length exceeds>the baseline describes sample proportions only, not significance or repeatability. paired_sampling_se_pp conditions on current models/input samples, not source/world uncertainty.',
        '',
        'Final identity validation (fixed1/21 mean):',
    ]
    for row in validation:
        txt.append(f"{row['architecture']} / {row['encoding']} / {row['task']}: acc={100*row['full_vocab_accuracy']:.2f}%, baseline={100*row['training_length_majority_accuracy']:.2f}%, gain={row['gain_over_training_length_majority_pp']:+.2f}pp; mode-agreement={100*row['prediction_agreement_with_length_majority']:.2f}%; non-majority acc={100*row['nonmajority_accuracy']:.2f}%; observed-above-baseline lengths=[{row['observed_above_baseline_lengths']}]")
    txt.extend(['', 'Fixed-window learning trends:'])
    for row in source_trends:
        txt.append(f"{row['architecture']} / {row['encoding']} / {row['task']}: val1k={100*row['validation_1k']:.2f}%, val10k={100*row['validation_10k']:.2f}%, val20k={100*row['validation_20k']:.2f}%, late accuracy gain={row['validation_10k_to_20k_gain_pp']:+.2f}pp; early/late minibatch-loss medians={row['early_1k_5k_median_last_update_loss']:.4f}/{row['late_16k_20k_median_last_update_loss']:.4f}")
    cycle_validation = [row for row in validation if row['encoding'] == 'canonical_cycle']
    cycle_trends = [row for row in source_trends if row['encoding'] == 'canonical_cycle']
    candidates = []
    for n in LENGTHS:
        selected = [row for row in rows if row['split'] == 'validation' and row['action'] == 'identity'
                    and row['encoding'] == 'canonical_cycle' and row['length'] == n]
        assert len(selected) == 8
        if all(row['gain_over_training_length_majority_pp'] > 0 for row in selected):
            candidates.append(n)
    txt.extend(['',
        f"Cycle-encoding validation summary: mean-gain range{min(row['gain_over_training_length_majority_pp'] for row in cycle_validation):+.2f}..{max(row['gain_over_training_length_majority_pp'] for row in cycle_validation):+.2f}pp;10k→20k validation-improvement range{min(row['validation_10k_to_20k_gain_pp'] for row in cycle_trends):+.2f}..{max(row['validation_10k_to_20k_gain_pp'] for row in cycle_trends):+.2f}pp. Relative to one-line encoding, learning stagnates near length-answer baselines; content tasks are not established as learned.",
        f"Length{list(candidates)}is a length where all eight cycle-encoding models show>0 gain; this is a descriptive sign intersection, not successful screening/significance or evidence of matched mastery.",
        'Interpretive boundary: near-mode-baseline scores or high mode agreement do not prove no permutation-content use; correct nonmode predictions and displaced errors are fully reported.',
        'Low agreement with one mode also does not establish content learning: choosing another common same-length answer yields some correct nonmode predictions while overall gains remain small.',
        'Length recommendation: retain original10..30 as the target, without dropping lengths based on old tests or geometry. All eight cycle models remain near length baselines at20k with small late gains, so historical validation identifies no successful length subset to recommend. Engineering debugging can first fix original minimum n=10, then a continuous10..12 window across four tasks and both architectures; this reduces input burden by a fixed rule, not selected successful historical lengths, and still requires independent development data without guaranteed learning.',
        '“Slightly above-mode means at some lengths” cannot confirm success after shortening. Prespecify meaningful training-mode gains and full task coverage on new development validation; failures remain failures rather than promoting successful lengths to primary conclusions.',
        'This script trains nothing and does not change historical reports, frozen protocols, or F17.',
    ])
    (output / f'{stem}.txt').write_text('\n'.join(txt) + '\n')
    record(Path(__file__).resolve(), 'new read-only diagnosis script')
    provenance = dict(status='passed',created_utc=datetime.now(timezone.utc).isoformat(),
                      source_root=str(source),protocol_sha256=protocol_sha,data_sha256=data_sha,
                      models=16,per_length_rows=2016,summary_rows=96,learning_curve_rows=320,
                      training_label_count_rows=len(counts),validation_final_matches_original_curves=16,
                      primary_split='validation',test_usage='old descriptive only; not for selecting lengths',
                      decoded_data_arrays=['train_lengths','train_labels','validation_lengths','validation_labels','test_lengths','test_labels'],
                      decoded_cache_array='prediction only',geometry_metrics_or_activations_or_probabilities_loaded=False,
                      weights_loaded=False,new_training_started=False,old_reports_or_f17_modified=False,
                      source_artifacts=artifacts,plots=plots,
                      missing_saved_quantities=['per-update per-length validation predictions','validation loss'],
                      descriptive_cycle_all_eight_observed_positive_gain_lengths=candidates,
                      length_recommendation='Old validation does not establish a learnable subset; keep complete10..30 target. Any debug reduction uses fixed smallest10 then contiguous10..12, all tasks/architectures; no old-test or geometry selection.')
    (output / f'{stem}_audit.json').write_text(json.dumps(provenance,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({k:v for k,v in provenance.items() if k != 'source_artifacts'},indent=2,ensure_ascii=False))


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,default=SOURCE)
    parser.add_argument('--output',type=Path,default=OUTPUT)
    arguments=parser.parse_args()
    main(arguments.source,arguments.output)
