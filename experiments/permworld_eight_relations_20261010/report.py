"""Complete exploratory eight-relation reader report, without new metrics."""
from __future__ import annotations

import os
os.environ['OPENBLAS_NUM_THREADS'] = '2'
os.environ['OMP_NUM_THREADS'] = '2'

from collections import defaultdict
import html
import json
import math
from pathlib import Path
import sys

import numpy as np

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.permworld_eight_relations_20261010.analysis_helpers import *

DISPLAY = {
    'descent_inverse': 'Descents—inverse descents', 'peak_complement': 'Peaks—valleys',
    'fixed_complement': 'Fixed points—anti-fixed points', 'exceedance_inverse': 'Exceedances—deficiencies',
    'subsequence_complement': 'LIS—LDS', 'run_complement': 'Increasing runs—decreasing runs',
    'record_complement': 'Left-to-right maximum—minimum records', 'decomposition_complement': 'Global descents—components (+1)',
}
FIELDS_DISPLAY = ['initial_correct', 'initial_wrong', 'initial_identity', 'trained_correct', 'trained_wrong',
                  'trained_identity', 'B0', 'Bt', 'dc', 'dw', 'Delta']


def formatted(value):
    if isinstance(value, bool):
        return 'yes' if value else 'no'
    try:
        x = number(value)
        return 'undefined' if not math.isfinite(x) else f'{x:.4f}'
    except (ValueError, TypeError):
        return str(value)


def table(records, fields):
    head = ''.join('<th>' + html.escape(k) + '</th>' for k in fields)
    body = ''.join('<tr>' + ''.join('<td>' + html.escape(formatted(row.get(k, ''))) + '</td>' for k in fields) + '</tr>'
                   for row in records)
    return '<div class="scroll"><table><thead><tr>' + head + '</tr></thead><tbody>' + body + '</tbody></table></div>'


def gold_summary(source, lengths):
    # Repeated configuration labels do not provide four independent baselines.
    selected = [r for r in source if r['architecture'] == 'transformer' and r['encoding'] == 'one_line']
    groups = defaultdict(list)
    fields = ('relation', 'control', 'baseline', 'role', 'metric')
    for row in selected:
        groups[tuple(row[k] for k in fields)].append(row)
    result = []
    for key, group in groups.items():
        require(sorted(int(r['length']) for r in group) == lengths, 'Static baseline lengths incomplete')
        result.append(dict(zip(fields, key), value=equal_length([r['value'] for r in group]),
                           error=equal_length([r['error'] for r in group]), length_count=21,
                           undefined_lengths=sum(not math.isfinite(number(r['value'])) for r in group),
                           static_no_training_delta=True,
                           repeated_configuration_rows_are_not_independent=True))
    require(len(result) == 384, 'Static baseline aggregate scope changed')
    return result


def plot_all(protocol, summary, cross, learning, length_rows):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    plt.rcParams.update({'font.size': 9})
    outdir = DIAG / 'figures'
    outdir.mkdir(exist_ok=True)
    conditions = [(a, e) for a in protocol['architectures'] for e in protocol['encodings']]
    names = [r['pair_id'] for r in protocol['relations']]
    lookup = {(r['architecture'], r['encoding'], r['relation'], r['control'], r['branch'], r['layer'], r['metric']): r
              for r in summary if r['contrast_type'] == 'condition_summary'}
    plots = []
    for control in protocol['controls']:
        fig, axes = plt.subplots(8, 4, figsize=(17, 22), sharex=True, sharey=True)
        for i, relation in enumerate(names):
            for j, (a, e) in enumerate(conditions):
                row = lookup[a, e, relation, control, 'hidden', 'final_norm', 'linear_cka']
                ax = axes[i, j]
                x = np.arange(3)
                ax.plot(x, [number(row['initial_' + r]) for r in ('correct', 'wrong', 'identity')], 'o-', label='initial')
                ax.plot(x, [number(row['trained_' + r]) for r in ('correct', 'wrong', 'identity')], 's-', label='20k')
                ax.set_xticks(x, ['correct', 'wrong', 'identity'])
                ax.set_ylim(-.04, 1.04)
                ax.grid(alpha=.25)
                if i == 0:
                    ax.set_title(a + ' / ' + e)
                if j == 0:
                    ax.set_ylabel(relation)
        axes[0, 0].legend()
        fig.suptitle('Raw similarity S: fixed21 equal-length final CKA / ' + control)
        fig.tight_layout(rect=(0, 0, 1, .985))
        path = outdir / f'similarity_{control}.png'
        fig.savefig(path, dpi=130)
        plt.close(fig)
        plots.append(path)
        for contrast_type in ('encoding_difference', 'configuration_difference'):
            selected = [r for r in summary if r['contrast_type'] == contrast_type and r['control'] == control
                        and r['branch'] == 'hidden' and r['layer'] == 'final_norm' and r['metric'] == 'linear_cka']
            require(len(selected) == 16, 'Every relation requires both architecture/encoding contrasts')
            fig, axes = plt.subplots(1, 5, figsize=(17, 7))
            labels = [r['relation'] + '/' + (r['architecture'] if contrast_type == 'encoding_difference' else r['encoding'])
                      for r in selected]
            for ax, field in zip(axes, ('B0', 'Bt', 'dc', 'dw', 'Delta')):
                ax.barh(np.arange(16), [number(r[field]) for r in selected])
                ax.axvline(0, color='black', lw=.7)
                ax.set_yticks(np.arange(16), labels if ax == axes[0] else [])
                ax.invert_yaxis()
                ax.set_title(('Gamma ' if contrast_type == 'encoding_difference' else 'GRU-TF ') + field)
                ax.grid(axis='x', alpha=.25)
            fig.suptitle(contrast_type + ' / ' + control + ' / all eight relation instances')
            fig.tight_layout(rect=(0, 0, 1, .96))
            path = outdir / f'{contrast_type}_{control}.png'
            fig.savefig(path, dpi=130)
            plt.close(fig)
            plots.append(path)
        for metric_name in ('linear_cka', 'knn_overlap', 'procrustes', 'debiased_cka'):
            fig, axes = plt.subplots(1, 5, figsize=(17, 10))
            labels = [r + '/' + a + '/' + e for r in names for a, e in conditions]
            for ax, field in zip(axes, ('B0', 'Bt', 'dc', 'dw', 'Delta')):
                values = [number(lookup[a, e, r, control, 'hidden', 'final_norm', metric_name][field])
                          for r in names for a, e in conditions]
                ax.barh(np.arange(32), values)
                ax.axvline(0, color='black', lw=.7)
                ax.set_yticks(np.arange(32), labels if ax == axes[0] else [])
                ax.invert_yaxis()
                ax.set_title(field)
                ax.grid(axis='x', alpha=.25)
            fig.suptitle(metric_name + ' / ' + control + '; axes are metric-specific, not comparable across metrics')
            fig.tight_layout(rect=(0, 0, 1, .96))
            path = outdir / f'decomposition_{control}_{metric_name}.png'
            fig.savefig(path, dpi=130)
            plt.close(fig)
            plots.append(path)
        for metric_name in LAYER_METRICS:
            fig, axes = plt.subplots(8, 5, figsize=(20, 23))
            for i, relation in enumerate(names):
                for j, field in enumerate(('B0', 'Bt', 'dc', 'dw', 'Delta')):
                    ax = axes[i, j]
                    for a, e in conditions:
                        layers = [f'block{k}' if a == 'transformer' else f'layer{k}' for k in range(1, 5)] + ['final_norm']
                        values = [number(lookup[a, e, relation, control, 'hidden', layer, metric_name][field]) for layer in layers]
                        ax.plot(np.arange(5), values, marker='.', label=a + '/' + e)
                    ax.axhline(0, color='black', lw=.6)
                    ax.set_xticks(np.arange(5), ['1', '2', '3', '4', 'norm'])
                    ax.grid(alpha=.25)
                    if i == 0:
                        ax.set_title(field)
                    if j == 0:
                        ax.set_ylabel(relation)
            axes[0, 0].legend(fontsize=6)
            fig.suptitle('All fixed layers, separate norm / ' + control + ' / ' + metric_name)
            fig.tight_layout(rect=(0, 0, 1, .985))
            path = outdir / f'layers_{control}_{metric_name}.png'
            fig.savefig(path, dpi=130)
            plt.close(fig)
            plots.append(path)
        # Do not choose one favorable metric or length heatmap.
        length_index = {(r['architecture'], r['encoding'], r['relation'], int(r['length'])): r for r in length_rows
                        if r['control'] == control and r['branch'] == 'hidden' and r['layer'] == 'final_norm' and r['metric'] == 'linear_cka'}
        fig, axes = plt.subplots(1, 5, figsize=(19, 10))
        label = [r + '/' + a + '/' + e for r in names for a, e in conditions]
        for ax, field in zip(axes, ('B0', 'Bt', 'dc', 'dw', 'Delta')):
            matrix = np.array([[number(length_index[a, e, r, n][field]) for n in protocol['lengths']]
                               for r in names for a, e in conditions])
            vmax = max(1e-9, float(np.nanmax(np.abs(matrix))))
            im = ax.imshow(matrix, cmap='coolwarm', aspect='auto', vmin=-vmax, vmax=vmax)
            ax.set_xticks([0, 5, 10, 15, 20], [10, 15, 20, 25, 30])
            ax.set_yticks(np.arange(32), label if ax == axes[0] else [])
            ax.set_title(field)
            fig.colorbar(im, ax=ax, fraction=.025)
        fig.suptitle('Every length retained / final linear CKA / ' + control + '; no pooled-length substitution')
        fig.tight_layout(rect=(0, 0, 1, .96))
        path = outdir / f'all_lengths_{control}.png'
        fig.savefig(path, dpi=130)
        plt.close(fig)
        plots.append(path)
    fig, axes = plt.subplots(1, 2, figsize=(14, 8))
    jobs = {(r['architecture'], r['encoding'], r['task']): r for r in learning
            if r['condition'] == 'trained' and r['split'] == 'test' and r['action'] == 'identity'}
    for ax, e in zip(axes, protocol['encodings']):
        for a, marker in [('transformer', 'o'), ('gru', 's')]:
            ax.plot([number(jobs[a, e, t]['answer_accuracy']) for t in protocol['label_task_names']],
                    np.arange(16), marker + '-', label=a)
        ax.plot([number(jobs['transformer', e, t]['length_majority_accuracy']) for t in protocol['label_task_names']],
                np.arange(16), 'k:', label='train-defined length mode')
        ax.set_yticks(np.arange(16), protocol['label_task_names'])
        ax.set_xlim(0, 1.02)
        ax.set_title(e + ' / all16 tasks / equal21 accuracy')
        ax.grid(alpha=.25)
        ax.legend()
    fig.tight_layout()
    path = outdir / 'source_learning.png'
    fig.savefig(path, dpi=130)
    plt.close(fig)
    plots.append(path)
    # One explicit A/B/C/D reader entry, while all detailed figures above stay
    # complete. It shows both trained preference and change, never Delta alone.
    control = 'answer_strata'
    fig, axes = plt.subplots(2, 2, figsize=(22, 19))
    row_ids = [(r, a, e) for r in names for a, e in conditions]
    row_labels = [r + '/' + a + '/' + e for r, a, e in row_ids]
    direction = np.array([[np.sign(number(lookup[a, e, r, control, 'hidden', 'final_norm', m][field]))
                           for field in ('Bt', 'Delta') for m in ('linear_cka', 'knn_overlap', 'procrustes')]
                          for r, a, e in row_ids])
    im = axes[0, 0].imshow(direction, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
    axes[0, 0].set_xticks(np.arange(6), ['Bt CKA', 'Bt k10', 'Bt Proc8', 'Delta CKA', 'Delta k10', 'Delta Proc8'], rotation=25)
    axes[0, 0].set_yticks(np.arange(32), row_labels, fontsize=6)
    axes[0, 0].set_title('A. Same activations, three fixed metrics: sign of Bt and Delta')
    fig.colorbar(im, ax=axes[0, 0], fraction=.03, ticks=[-1, 0, 1])
    configuration = [r for r in summary if r['contrast_type'] == 'configuration_difference'
        and r['control'] == control and r['branch'] == 'hidden' and r['layer'] == 'final_norm' and r['metric'] == 'linear_cka']
    encodings = [r for r in summary if r['contrast_type'] == 'encoding_difference'
        and r['control'] == control and r['branch'] == 'hidden' and r['layer'] == 'final_norm' and r['metric'] == 'linear_cka']
    for ax, selected, fields, labels, title in [
        (axes[0, 1], configuration, ('B0', 'Bt'), [r['relation'] + '/' + r['encoding'] for r in configuration],
         'B. Existing configuration contrast (GRU - Transformer): B0 and Bt'),
        (axes[1, 0], encodings, ('Gamma0', 'Gammat', 'GammaDelta'), [r['relation'] + '/' + r['architecture'] for r in encodings],
         'C. Encoding contrast (cycle - line): Gamma0, Gammat, GammaDelta')]:
        matrix = np.array([[number(r[k]) for k in fields] for r in selected])
        maximum = max(1e-9, float(np.nanmax(np.abs(matrix))))
        image = ax.imshow(matrix, cmap='coolwarm', vmin=-maximum, vmax=maximum, aspect='auto')
        ax.set_xticks(np.arange(len(fields)), fields)
        ax.set_yticks(np.arange(len(labels)), labels, fontsize=7)
        ax.set_title(title)
        for i in range(len(selected)):
            for j in range(len(fields)):
                if math.isfinite(matrix[i, j]):
                    ax.text(j, i, f'{matrix[i,j]:+.3f}', ha='center', va='center', fontsize=7)
        fig.colorbar(image, ax=ax, fraction=.03)
    matrix = []
    for r, a, e in row_ids:
        layers = [f'block{k}' if a == 'transformer' else f'layer{k}' for k in range(1, 5)] + ['final_norm']
        matrix.append([number(lookup[a, e, r, control, 'hidden', l, 'linear_cka'][field])
                       for field in ('Bt', 'Delta') for l in layers])
    matrix = np.array(matrix)
    maximum = max(1e-9, float(np.nanmax(np.abs(matrix))))
    image = axes[1, 1].imshow(matrix, cmap='coolwarm', vmin=-maximum, vmax=maximum, aspect='auto')
    axes[1, 1].set_xticks(np.arange(10), ['Bt1', 'Bt2', 'Bt3', 'Bt4', 'Bt norm', 'D1', 'D2', 'D3', 'D4', 'D norm'], rotation=25)
    axes[1, 1].set_yticks(np.arange(32), row_labels, fontsize=6)
    axes[1, 1].axvline(4.5, color='black', lw=1)
    axes[1, 1].set_title('D. All four layers and separate norm: Bt and Delta / CKA')
    fig.colorbar(image, ax=axes[1, 1], fraction=.03)
    fig.suptitle('Four controlled comparisons / all eight relations / answer strata / fixed equal21 lengths', fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, .975))
    path = outdir / 'four_controlled_comparisons.png'
    fig.savefig(path, dpi=140)
    plt.close(fig)
    plots.append(path)
    return plots


def main():
    protocol = verify_protocol()
    require(js(DIAG / 'metric_review/analysis_audit.json')['status'] == 'passed', 'Analysis not complete')
    source = rows(DIAG / 'per_length_results.csv')
    summary = rows(DIAG / 'contrast_summary.csv')
    cross = rows(DIAG / 'cross_arch_same_task_summary.csv')
    learning = rows(DIAG / 'source_learning_summary.csv')
    static = gold_summary(rows(DIAG / 'gold_baseline_length.csv'), protocol['lengths'])
    csv_write(DIAG / 'gold_baseline_summary.csv', static)
    plots = plot_all(protocol, summary, cross, learning, source)
    conditions = [r for r in summary if r['contrast_type'] == 'condition_summary']
    main = [dict(r, relation=DISPLAY.get(r['relation'], r['relation'])) for r in conditions
            if r['layer'] == 'final_norm' and r['branch'] == 'hidden' and r['metric'] == 'linear_cka' and r['control'] == 'answer_strata']
    require(len(main) == 32, 'Reader main table must display all32 fixed condition instances')
    source_jobs = [r for r in learning if r['condition'] == 'trained' and r['split'] == 'test' and r['action'] == 'identity']
    require(len(source_jobs) == 64, 'Every endpoint, including weak learners, must be visible')
    paired_analyses = ('encoding_difference', 'configuration_difference', 'interaction')
    contrasts = [r for r in summary if r['contrast_type'] in paired_analyses and r['branch'] == 'hidden'
                 and r['layer'] == 'final_norm' and r['control'] == 'answer_strata' and r['metric'] == 'linear_cka']
    directions = [r for r in summary if r['contrast_type'] == 'metric_direction']
    output = [r for r in conditions if r['layer'] == 'output' and r['control'] == 'answer_strata' and r['metric'] == 'linear_cka']
    alignment = rows(DIAG / 'relation_label_alignment.csv')
    require(len(alignment) == 1008, 'Exact and accidental label correspondences must cover all splits/lengths/roles')
    coverage = []
    for row in conditions:
        relation = next(r for r in protocol['relations'] if r['pair_id'] == row['relation'])
        undefined = int(row['undefined_lengths'])
        coverage.append({k: row[k] for k in GROUP_KEYS} | dict(status='undefined_component' if undefined else 'completed',
            fixed_lengths=21, undefined_Delta_lengths=undefined, relation_type=relation['input_transform'],
            right_label_offset=relation['right_label_offset'], dependency='One world; shared anchors/config init; same transform type',
            not_independent_replication=True, no_effect_selection=True))
    coverage += [dict(status='coupled', scope='32 relation configuration/encoding conditions',
        dependency='Eight relation instances in one world; only inverse and complement types, shared anchors and per-arch initial weights',
        not_independent_replication=True),
        dict(status='missing', scope='Original16 task-specific initial answer CE',
        dependency='Initial hidden sharing cannot recover task-specific logits; old scientific missing values preserved'),
        dict(status='not_extended', scope='Multiple independent initialization and data worlds',
        dependency='No confirmatory sample-size expansion authorized'),
        dict(status='not_extended', scope='Graphworld encoding-by-architecture grid',
        dependency='Only fixed reference Transformer and one graph encoding; separate bounded four-model pilot'),
        dict(status='not_extended', scope='Universal mathematical mechanism or functional inference',
        dependency='Representation similarity measurements alone do not identify these claims')]
    csv_write(DIAG / 'coverage_status.csv', coverage)
    parts = [
        '<!doctype html><html lang="en"><meta charset="utf-8"><title>Complete controlled comparison of eight PermWorld relations</title>',
        '<style>body{max-width:1500px;margin:32px auto;padding:0 20px;font:15px/1.6 sans-serif}table{border-collapse:collapse;white-space:nowrap}th,td{border:1px solid #ddd;padding:5px 9px}th{background:#eef2f7}.scroll{overflow:auto;margin:16px 0}img{max-width:100%}details{margin:12px 0}.notice{background:#fff5d9;padding:12px}</style>',
        '<h1>How mathematical correspondences change representational similarity: a controlled eight-relation PermWorld extension</h1>',
        '<p>This report includes all 64 single-task 20k endpoints from one existing data world, with one genuine initialization per architecture. Sixteen earlier endpoints and the verified results for two relations are reused, with at most 48 missing models added. No subset was selected by geometric direction, source-task accuracy, or undefined metrics. The eight relation instances cover only inversion and complement; the 32 conditions are not independent repetitions.</p>',
        '<p class="notice">The order is original similarity S → trained correct/wrong pairing preference Bt → change relative to genuine initialization Δ. A reversal of Δ does not guarantee a reversal of Bt, and positive Δ does not guarantee increased correct-pair similarity. This is an exploratory extension without confirmatory p-values, equivalence claims, or mathematical-mechanism inference. Source-model capacities and task mastery are unmatched, so configuration comparisons do not identify a pure architecture effect.</p>',
        '<h2>Original similarities and decompositions for all relations</h2>',
        '<p>The following is ordinary linear CKA with joint-answer stratification at the input-end position before the task token, after final normalization. Each length is computed separately and lengths 10…30 are aggregated with fixed weight 1/21. If any length is undefined, the aggregate for that component remains undefined. Lengths are not pooled, NaNs are not filled with zero, and missing components are not reweighted.</p>',
        table(main, ['relation', 'architecture', 'encoding'] + FIELDS_DISPLAY),
        '<p>B0=S_correct,initial−S_wrong,initial; Bt=S_correct,trained−S_wrong,trained; dc=correct_trained−correct_initial; dw=wrong_trained−wrong_initial; Δ=Bt−B0=dc−dw. Original identity-pair scores are retained separately, not treated as an upper bound that must be exceeded.</p>',
        '<p>The originally specified wrong transformation does not guarantee a wrong answer for every sample. Samples that happen to satisfy the formula are all retained; per-length matching rates are in relation_label_alignment.csv. Especially for near-constant tasks, wrong pairings can have high answer agreement; the negative control is not changed in response.</p>',
        '<h2>Encodings, model configurations, and layers</h2>',
        '<p><img src="figures/four_controlled_comparisons.png" alt="Overview of four controlled comparisons: metrics, model configurations, encodings, and layers"></p>',
        '<p>Γ is fixed as cycle notation minus one-line notation, including Γ0, Γt, Γc, Γw, and ΓΔ. Encoding effects on initial and trained preference must be distinguished. The configuration difference is GRU minus Transformer; parameter counts, compute, and source-task learning are unmatched. All four blocks/GRU layers and final normalization are shown. Normalization is listed separately to expose its effect; no favorable layer was selected. The end-token embedding is structurally constant within each length, so an undefined value is not negative evidence of learning.</p>',
        table(contrasts, ['contrast_type', 'relation', 'architecture', 'encoding', 'B0', 'Bt', 'dc', 'dw', 'Delta', 'Gamma0', 'Gammat', 'GammaDelta']),
        '<h2>Directions under fixed metrics and length-specific limits</h2>',
        '<p>Ordinary CKA, k=10 neighborhood overlap, and held-out Procrustes describe different geometries. Neighborhoods use squared Euclidean distance, stable global anchor IDs to break exact ties, and exclusion of self and duplicate anchors. Procrustes fits rank-8 PCA, RMS normalization, and the orthogonal map only on the validation fit split, with analysis inputs held out. S is negative NRMSE; all original errors are saved. U-centered CKA is the prespecified sensitivity analysis, and layer analyses use only these two CKA versions. Compare directions and patterns across metrics, not score magnitudes; metrics are not independent repetitions.</p>',
        '<details><summary>Three-metric directions for all 64 final-layer condition/control combinations</summary>' + table(directions,
            ['relation', 'architecture', 'encoding', 'control', 'B0_directions', 'Bt_directions', 'dc_directions', 'dw_directions', 'Delta_directions']) + '</details>',
        '<h2>Answers, outputs, and source-task learning</h2>',
        '<p>The joint-answer key is [n,left_identity,right_identity−offset,right_complement−offset,right_inverse−offset], with minimum group size 3 in each split. The global-descents/components relation explicitly satisfies components(Cπ)=global_descents(π)+1. Subtracting 1 from every right-side key is a bijective coordinate change; groups are identical to those from unshifted keys. Task labels, accuracies, and true-answer one-hot/scalar encodings retain their actual numeric values; distinct answers are not treated as equal.</p>',
        '<p>The original key dtype for the two earlier relations was int16; the new protocol uses int64 to handle +1 safely. Earlier CSV key hashes are preserved; current key hashes and value/partition/centering agreement are saved separately in reused_key_coordinate_audit.csv. This is a coordinate-representation difference, not a change of activations or recomputation of earlier scientific metrics.</p>',
        '<p>Ground-truth stratification removes within-group answer means, but does not remove confidence, prediction errors, or all nonlinear label information in output probabilities. Static answer baselines are not assigned initialization/training phases or fictitious Δ; they remain undefined when structurally constant after answer stratification. Output probabilities are conditional on numeric tokens 0…30, not the full 188-token vocabulary; source accuracy and answer CE use the full vocabulary. Hidden Δ minus output Δ cannot estimate an additional mathematical mechanism.</p>',
        '<details><summary>All ordinary-CKA output-probability controls</summary>' + table(output,
            ['relation', 'architecture', 'encoding'] + FIELDS_DISPLAY) + '</details>',
        table(source_jobs, ['task', 'architecture', 'encoding', 'answer_accuracy', 'minimum_length_accuracy',
                           'answer_ce', 'length_majority_accuracy', 'accuracy_minus_length_mode']),
        '<p>Length-specific modes use only original training identity-input labels, with ties resolved to the smallest number. Missing initial CE is not filled from another task readout; earlier EOS statistics are listed separately as the original teacher-forced validation results. Weakly learned tasks remain in the geometry tables, but their differences cannot establish encoding mechanisms at equal mastery.</p>',
        '<h2>32 same-task cross-configuration pairings</h2>',
        '<p>Here direct CKA compares Transformer and GRU on the same task and underlying inputs. Its answer key is [n,this_task_identity/complement/inverse], which differs from the joint key for relational task pairs and must not be conflated with that effect. It describes similarity between the existing configurations and cannot independently identify a shared algorithm.</p>',
        table(cross, ['task', 'encoding', 'control', 'metric', 'initial', 'trained', 'change']),
        '<h2>All prespecified figures</h2>',
    ]
    parts.extend('<details><summary>' + html.escape(path.name) + '</summary><img src="figures/' + html.escape(path.name) + '"></details>' for path in plots)
    files = ['per_length_results.csv', 'contrast_summary.csv', 'cross_arch_same_task.csv', 'cross_arch_same_task_summary.csv',
             'gold_baseline_length.csv', 'gold_baseline_summary.csv', 'source_learning.csv', 'source_learning_summary.csv',
             'source_validation_eos_original.csv', 'coverage_status.csv', 'protocol.json', 'training_freeze.json', 'analysis_freeze.json',
             'relation_label_alignment.csv', 'reused_key_coordinate_audit.csv',
             'model_gap_manifest.csv', 'training_manifest.csv', 'cost_estimate.csv', 'math_label_audit.json',
             'metric_review/analysis_audit.json', 'metric_review/independent_metric_actual.json']
    parts += ['<h2>Reproducible artifacts and interpretive scope</h2>',
        '<p>' + ' · '.join('<a href="' + html.escape(name) + '">' + html.escape(name) + '</a>' for name in files) + '</p>',
        '<p>Negative values, sparse strata, and undefined results are retained for every relation, condition, length, and metric. Genuine initialization identity, original data-orbit splits, source code, and activation-array hashes are recorded in the protocol and audits. The evidence only describes how similarity and pairing preference under mathematical correspondences depend on these measurement and modeling conditions; CKA alone cannot prove encoding mechanisms, causal representations of mathematical relations, or functional benefits.</p>',
        '<p>The four-task graph pilot has its own protocol and report. Its sizes, input structure, tasks, and sample distribution differ from PermWorld, so original CKA scores are not directly subtracted across domains. This report does not alter F17 or earlier reports and does not publish automatically. Work stops as authorized after the fixed graph pilot and combined report, without additional seeds or tuning.</p></html>']
    (DIAG / 'report.html').write_text('\n'.join(parts))
    text = ('Complete controlled eight-relation PermWorld extension (exploratory)\n'
            'Fixed original similarities S, trained correct−wrong preference Bt, and complete Δ decomposition relative to genuine initialization.\n'
            '64 single-task models, 32 relation conditions, 8 instances, 2 transformation types, and 21 fixed equally weighted lengths; these are not independent repetitions.\n'
            'The earlier two relations reuse 6048 per-length rows and 672 same-task rows value for value; six new relations complete the scope under the same definitions.\n'
            'Original similarities, answer/output controls, weak learning, all layers, three primary metrics, and U-centered sensitivity are all retained.\n'
            'Report: report.html; complete data: per_length_results.csv/contrast_summary.csv/source_learning.csv.\n'
            'A Delta reversal alone does not establish reversal of trained preference, and CKA alone does not establish a mechanism.\n')
    (DIAG / 'report.txt').write_text(text)
    write_json(DIAG / 'report_audit.json', dict(status='passed', per_length_rows=len(source), condition_summaries=len(conditions),
        displayed_primary_conditions=len(main), source_models_displayed=len(source_jobs), plots=len(plots),
        coverage_rows=len(coverage), all_relations_and_conditions_retained=True, no_metric_calculation=True,
        fixed21_no_NaN_reweighting=True, no_training=True, exploratory=True,
        report_sha256=sha(DIAG / 'report.html'), figures_sha256={str(p.relative_to(DIAG)): sha(p) for p in plots}))
    print(json.dumps(dict(status='passed', report=str(DIAG / 'report.html'), plots=len(plots))))


if __name__ == '__main__':
    main()
