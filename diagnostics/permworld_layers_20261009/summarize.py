"""Complete, fixed-rule summaries of the exploratory PermWorld layer diagnostic.

No training, adaptive metric selection, or modification of inherited artifacts.
Every source-level value requires every length 10..30.  Source dispersion is
descriptive initialization variation in a shared data world, not an iid world CI.
"""
import argparse
from collections import defaultdict
import csv
import hashlib
import html
import json
from pathlib import Path
import time

import numpy as np

ROOT = Path(__file__).resolve().parent
LENGTHS = tuple(range(10, 31))
PRIMARY = ('linear_cka', 'knn_overlap', 'procrustes')
METRICS = PRIMARY + ('debiased_cka',)
KEYS = ('domain', 'cohort', 'relation', 'family', 'kind', 'branch', 'control', 'metric')
UNIT_KEYS = KEYS + ('source_unit', 'seed', 'world')
DECOMPOSITION = ('B0', 'Bt', 'delta_correct', 'delta_wrong', 'delta')
FIELDS = ('initial_correct', 'initial_wrong', 'initial_identity',
          'trained_correct', 'trained_wrong', 'trained_identity') + DECOMPOSITION + (
          'initial_correct_raw_error', 'initial_wrong_raw_error', 'initial_identity_raw_error',
          'trained_correct_raw_error', 'trained_wrong_raw_error', 'trained_identity_raw_error')
MAIN_BRANCH = 'final_norm_end'
MAIN_CONTROL = 'answer_strata'
LAYERS = ('block_01', 'block_02', 'block_03', 'block_04', 'final_norm')
NEURAL = tuple(f'{layer}_{pool}' for layer in LAYERS for pool in ('end', 'value_mean'))
INPUTS = ('embedding_end', 'embedding_value_mean', 'position_numeric_onehot',
          'position_numeric_embedding')
CONTROLS = ('query_final_norm', 'output_prob', 'answer_onehot', 'answer_scalar')


def read(path):
    with Path(path).open(newline='') as handle:
        return list(csv.DictReader(handle))


def write(root, name, rows, fields=None):
    fields = fields or list(dict.fromkeys(key for row in rows for key in row))
    with (root / name).open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def number(value):
    if value is None or value == '':
        return None
    value = float(value)
    if not np.isfinite(value):
        raise ValueError('Non-finite metric must be explicitly undefined, never NaN.')
    return value


def mean_complete(values):
    return float(np.mean(values)) if values and all(v is not None for v in values) else None


def sign(value):
    return None if value is None else int(np.sign(value))


def by(rows, fields):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(key, '') for key in fields)].append(row)
    return groups


def stats(values):
    valid = [value for value in values if value is not None]
    return dict(defined=len(valid), mean=float(np.mean(valid)) if valid else None,
                median=float(np.median(valid)) if valid else None,
                min=min(valid) if valid else None, max=max(valid) if valid else None,
                source_sd=float(np.std(valid, ddof=1)) if len(valid) > 1 else None,
                positive=sum(value > 0 for value in valid),
                negative=sum(value < 0 for value in valid),
                zero=sum(value == 0 for value in valid))


def assert_decomposition(row):
    """Check both algebraic identities whenever their operands are defined."""
    terms = {field: number(row.get(field)) for field in DECOMPOSITION}
    if all(terms[field] is not None for field in ('B0', 'Bt', 'delta')):
        assert abs(terms['delta'] - (terms['Bt'] - terms['B0'])) < 2e-9, row
    if all(terms[field] is not None for field in ('delta_correct', 'delta_wrong', 'delta')):
        assert abs(terms['delta'] - (terms['delta_correct'] - terms['delta_wrong'])) < 2e-9, row


def source_summary(rows):
    output = []
    for key, group in sorted(by(rows, UNIT_KEYS).items()):
        lengths = [int(row['length']) for row in group]
        if len(lengths) != len(set(lengths)):
            raise ValueError(f'Duplicate length cell: {key}')
        if set(lengths) - set(LENGTHS):
            raise ValueError(f'Unexpected length cell: {key}')
        complete = set(lengths) == set(LENGTHS)
        row = dict(zip(UNIT_KEYS, key), planned_lengths=len(LENGTHS),
                   observed_lengths=len(lengths), all_lengths_present=complete,
                   initialization_grade='C', original_initial_identity_verified=False)
        for field in FIELDS:
            values = [number(item.get(field)) for item in group]
            defined = [value for value in values if value is not None]
            row[field] = mean_complete(values) if complete else None
            row[field + '_defined_lengths'] = len(defined)
            if field in DECOMPOSITION:
                row[field + '_positive_lengths'] = sum(value > 0 for value in defined)
                row[field + '_negative_lengths'] = sum(value < 0 for value in defined)
                row[field + '_zero_lengths'] = sum(value == 0 for value in defined)
                row[field + '_length_median'] = float(np.median(defined)) if complete and len(defined) == len(LENGTHS) else None
                row[field + '_length_min'] = min(defined) if defined else None
                row[field + '_length_max'] = max(defined) if defined else None
        assert_decomposition(row)
        output.append(row)
    return output


def relation_summary(units):
    output = []
    for key, group in sorted(by(units, KEYS).items()):
        row = dict(zip(KEYS, key), planned_sources=len(group),
                   source_units=';'.join(sorted(item['source_unit'] for item in group)),
                   uncertainty_scope='descriptive initialization variation; one fixed shared data world')
        for field in FIELDS:
            values = [item[field] for item in group]
            summary = stats(values)
            for name, value in summary.items():
                row[field + ('_sources' if name == 'defined' else '_' + name)] = value
            row[field + '_all_planned_sources_defined'] = summary['defined'] == len(group)
        for field in DECOMPOSITION:
            row[field + '_positive_all_lengths_sources'] = sum(
                item[field + '_positive_lengths'] == len(LENGTHS) for item in group)
            row[field + '_negative_all_lengths_sources'] = sum(
                item[field + '_negative_lengths'] == len(LENGTHS) for item in group)
        output.append(row)
    return output


def length_summary(rows):
    output = []
    for key, group in sorted(by(rows, KEYS + ('length',)).items()):
        row = dict(zip(KEYS + ('length',), key), planned_sources=len(group))
        for field in DECOMPOSITION:
            for name, value in stats([number(item.get(field)) for item in group]).items():
                row[field + ('_sources' if name == 'defined' else '_' + name)] = value
        output.append(row)
    return output


def direction_agreement(units):
    """Only common source/branch/control cells may contribute agreement counts."""
    groups = by(units, KEYS[:-1] + ('source_unit', 'seed', 'world'))
    output = []
    sensitivity = []
    for key, group in sorted(groups.items()):
        lookup = {row['metric']: row for row in group}
        row = dict(zip(KEYS[:-1] + ('source_unit', 'seed', 'world'), key))
        sens = row.copy()
        for field in DECOMPOSITION:
            values = [lookup.get(metric, {}).get(field) for metric in PRIMARY]
            complete = all(value is not None for value in values)
            row[field + '_all_three_defined'] = complete
            row[field + '_all_three_direction_agree'] = len(set(map(sign, values))) == 1 if complete else None
            row[field + '_all_three_positive'] = all(value > 0 for value in values) if complete else None
            for metric, value in zip(PRIMARY, values):
                row[field + '_' + metric] = value
            ordinary = lookup.get('linear_cka', {}).get(field)
            debiased = lookup.get('debiased_cka', {}).get(field)
            sens[field + '_ordinary'] = ordinary
            sens[field + '_debiased'] = debiased
            sens[field + '_both_defined'] = ordinary is not None and debiased is not None
            sens[field + '_direction_agree'] = sign(ordinary) == sign(debiased) if sens[field + '_both_defined'] else None
        output.append(row)
        sensitivity.append(sens)
    return output, sensitivity


def leave_one(relations):
    """Equal relation weights; family removals retain the dependence warning."""
    keys = ('domain', 'cohort', 'kind', 'branch', 'control', 'metric')
    output = []
    for key, group in sorted(by(relations, keys).items()):
        exclusions = [('none', '')]
        exclusions += [('relation', name) for name in sorted({row['relation'] for row in group})]
        exclusions += [('family', name) for name in sorted({row['family'] for row in group})]
        for level, excluded in exclusions:
            remaining = [row for row in group if level == 'none' or row[level] != excluded]
            row = dict(zip(keys, key), exclusion_level=level, excluded=excluded,
                       planned_relations=len(group), remaining_relations=len(remaining),
                       independent_relation_inference=False)
            for field in DECOMPOSITION:
                values = [item[field + '_mean'] for item in remaining]
                row[field] = mean_complete(values)
                row[field + '_defined_relations'] = sum(value is not None for value in values)
            output.append(row)
    return output


def layer_summary(relations):
    keys = ('domain', 'cohort', 'kind', 'branch', 'control', 'metric')
    output = []
    for key, group in sorted(by(relations, keys).items()):
        row = dict(zip(keys, key), planned_relations=len(group),
                   families=len({item['family'] for item in group}))
        for field in DECOMPOSITION:
            values = [item[field + '_mean'] for item in group]
            summary = stats(values)
            row[field + '_equal_relation_mean'] = mean_complete(values)
            row[field + '_median_relation_mean'] = summary['median']
            row[field + '_defined_relations'] = summary['defined']
            row[field + '_positive_relations'] = summary['positive']
            row[field + '_negative_relations'] = summary['negative']
        output.append(row)
    return output


def ordered_branches(branches):
    preferred = NEURAL + INPUTS + CONTROLS
    return [branch for branch in preferred if branch in branches] + sorted(set(branches) - set(preferred))


def plots(root, relations, lengths):
    """Every available layer, relation, metric and control is displayed."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figures = []
    for cohort in sorted({row['cohort'] for row in relations}):
        for control in sorted({row['control'] for row in relations if row['cohort'] == cohort}):
            for metric in METRICS:
                selected = [row for row in relations if (row['cohort'], row['control'], row['metric']) == (cohort, control, metric)]
                if not selected:
                    continue
                relation_ids = sorted({row['relation'] for row in selected})
                branches = ordered_branches({row['branch'] for row in selected})
                lookup = {(row['relation'], row['branch']): row for row in selected}
                fig, axes = plt.subplots(5, 1, figsize=(max(12, len(branches) * .77), max(12, len(relation_ids) * 2.05)), constrained_layout=True)
                for axis, field in zip(axes, DECOMPOSITION):
                    values = np.array([[lookup.get((rid, branch), {}).get(field + '_mean')
                                        if lookup.get((rid, branch), {}).get(field + '_mean') is not None else np.nan
                                        for branch in branches] for rid in relation_ids], dtype=float)
                    finite = np.abs(values[np.isfinite(values)])
                    span = max(float(finite.max()), .01) if len(finite) else 1.
                    color = plt.get_cmap('RdBu').copy(); color.set_bad('#d7dce0')
                    plotted = axis.imshow(np.ma.masked_invalid(values), cmap=color, vmin=-span, vmax=span, aspect='auto')
                    axis.set_yticks(range(len(relation_ids)), relation_ids, fontsize=8)
                    axis.set_xticks(range(len(branches)), branches, rotation=50, ha='right', fontsize=8)
                    axis.set_title(field + ' (equal length means; grey = undefined)')
                    for y, x in np.argwhere(np.isfinite(values)):
                        axis.text(x, y, f'{values[y,x]:.2f}', ha='center', va='center', fontsize=6,
                                  color='white' if abs(values[y,x]) > span * .65 else 'black')
                    fig.colorbar(plotted, ax=axis, fraction=.02, pad=.015)
                fig.suptitle(f'{cohort} / {control} / {metric}: all branches, no layer selection', fontsize=12)
                name = f'layers_{cohort}_{control}_{metric}.png'
                fig.savefig(root / name, dpi=135); plt.close(fig)
                figures.append(dict(cohort=cohort, control=control, metric=metric, file=name,
                                    scope='all branches; every decomposition component'))
        # Length sensitivity displays every contextual layer on separate lines.
        for control in sorted({row['control'] for row in lengths if row['cohort'] == cohort}):
            selected = [row for row in lengths if row['cohort'] == cohort and row['control'] == control and row['metric'] == 'linear_cka' and row['branch'] in NEURAL]
            ids = sorted({row['relation'] for row in selected})
            if not ids:
                continue
            fig, axes = plt.subplots(len(ids), 1, figsize=(12, max(3, 2.1 * len(ids))), squeeze=False, constrained_layout=True)
            for axis, rid in zip(axes[:, 0], ids):
                for branch in NEURAL:
                    cells = sorted([row for row in selected if row['relation'] == rid and row['branch'] == branch], key=lambda row: int(row['length']))
                    if cells:
                        axis.plot([int(row['length']) for row in cells],
                                  [np.nan if row['delta_mean'] is None else row['delta_mean'] for row in cells],
                                  label=branch, linewidth=1, linestyle='--' if branch.endswith('value_mean') else '-')
                axis.axhline(0, color='black', linewidth=.6); axis.set_title(rid, fontsize=9)
                axis.set_ylabel('Delta CKA'); axis.set_xticks([10, 15, 20, 25, 30])
            axes[0, 0].legend(ncol=5, fontsize=6, loc='upper center', bbox_to_anchor=(.5, 1.45))
            axes[-1, 0].set_xlabel('Length; all 21 lengths retained')
            fig.suptitle(f'{cohort} / {control}: length sensitivity, all contextual branches')
            name = f'lengths_{cohort}_{control}_linear_cka.png'
            fig.savefig(root / name, dpi=130); plt.close(fig)
            figures.append(dict(cohort=cohort, control=control, metric='linear_cka', file=name,
                                scope='length sensitivity; all contextual branches'))
    write(root, 'figure_manifest.csv', figures, ('cohort', 'control', 'metric', 'file', 'scope'))
    return figures


def fmt(value):
    if value is None or value == '':
        return 'undefined/missing'
    if isinstance(value, (float, np.floating)):
        return f'{value:.4f}'
    return html.escape(str(value))


def table(rows, fields):
    head = ''.join('<th>' + html.escape(field) + '</th>' for field in fields)
    cells = ''.join('<tr>' + ''.join('<td>' + fmt(row.get(field)) + '</td>' for field in fields) + '</tr>' for row in rows)
    return '<div class="scroll"><table><thead><tr>' + head + '</tr></thead><tbody>' + cells + '</tbody></table></div>'


def report(root, relations, units, agreements, sensitivity, leave, figures, audit):
    main = [row for row in relations if row['branch'] == MAIN_BRANCH and row['control'] == MAIN_CONTROL and row['metric'] in PRIMARY]
    compact = []
    for row in main:
        compact.append(dict(cohort=row['cohort'], relation=row['relation'], kind=row['kind'], metric=row['metric'],
                            sources=row['planned_sources'], defined=row['delta_sources'],
                            B0=row['B0_mean'], Bt=row['Bt_mean'], dc=row['delta_correct_mean'], dw=row['delta_wrong_mean'],
                            Delta=row['delta_mean'], median=row['delta_median'], sd=row['delta_source_sd'],
                            minimum=row['delta_min'], maximum=row['delta_max'],
                            positive_sources=row['delta_positive'], positive_all_lengths_sources=row['delta_positive_all_lengths_sources']))
    candidates = []
    grouped = by(main, ('cohort', 'relation', 'kind'))
    for key, group in sorted(grouped.items()):
        lookup = {row['metric']: row for row in group}
        complete = all(metric in lookup and lookup[metric]['delta_sources'] == lookup[metric]['planned_sources'] for metric in PRIMARY)
        if complete and all(lookup[metric]['delta_min'] > 0 for metric in PRIMARY):
            candidates.append(dict(cohort=key[0], relation=key[1], kind=key[2],
                                   sources=lookup['linear_cka']['planned_sources'],
                                   all_sources_three_metrics_positive=True,
                                   all_lengths_all_sources_three_metrics_positive=all(
                                       lookup[metric]['delta_positive_all_lengths_sources'] == lookup[metric]['planned_sources'] for metric in PRIMARY),
                                   CKA_delta=lookup['linear_cka']['delta_mean'],
                                   correct_change=lookup['linear_cka']['delta_correct_mean'],
                                   wrong_change=lookup['linear_cka']['delta_wrong_mean']))
    changed = [row for row in sensitivity if row['branch'] == MAIN_BRANCH and row['control'] == MAIN_CONTROL and
               (row['delta_direction_agree'] is False or row['Bt_direction_agree'] is False)]
    disagreement = [row for row in agreements if row['branch'] == MAIN_BRANCH and row['control'] == MAIN_CONTROL and
                    (row['delta_all_three_direction_agree'] is False or row['Bt_all_three_direction_agree'] is False)]
    candidate_names = '; '.join(f"{row['cohort']}/{row['relation']}({row['sources']} sources)" for row in candidates) or 'No relation satisfies positive increments for every available source on all three primary metrics'
    body = '<h1>PermWorld: multilayer and fixed-metric diagnostics of existing checkpoints</h1>'
    body += '<p class="lead">Training changed the relative preference for correct versus wrong input pairings. This round presents every layer and fixed pooling choice, separating changes in correct and wrong CKA; Δ is not automatically interpreted as correct-representation convergence or a mathematical mechanism.</p>'
    body += '<h2>Scope and evidential status</h2><p>Only historical completed checkpoints and the original fixed inputs are used. Added source training: 0; original reports and the F17 paper, results, and release artifacts remain unchanged. All existing PermWorld initialization references are grade C: reconstructed with the constructor, with original step-0 weight identity unverified. “Same seed” does not establish identical historical initialization. Below, “exceeding initialization” strictly means exceeding this reconstructed reference.</p>'
    body += '<p>The analysis contains' + str(audit['metric_rows']) + ' per-length metric cells and' + str(audit['source_summary_rows']) + ' source-level summaries. Each length is computed separately and aggregated using fixed 1/21 weights for 10–30; an undefined or missing length prevents an overall source value, without reweighting remaining lengths. Relation means and medians summarize only sources with complete defined lengths; defined explicitly reports available source counts.</p>'
    body += '<p>B0=initial correct−wrong; Bt=trained correct−wrong; dc/dw are the training changes in correct/wrong scores; Δ=Bt−B0=dc−dw. Positive Δ may result from declining wrong-pair scores; negative Δ does not mean no learning. Metrics have different scales; Procrustes score=−NRMSE, with raw errors retained.</p>'
    body += '<p><a href="protocol.json">Fixed analysis protocol</a> · <a href="freeze.json">Code/input versions</a> · <a href="extraction_audit.json">Checkpoint, parameter, and cache audit</a> · <a href="validation.json">Independent validation</a></p>'
    body += '<h2>1. Which relations have training increments exceeding reconstructed initialization references</h2>'
    body += '<p>The primary analysis remains final_norm_end (input end, before the task token), with answer-stratification control. Descriptive candidates with positive Δ on all three primary metrics for every available source:' + html.escape(candidate_names) + '. Single-source relations are observations for those sources; agreement across multiple sources is neither independent-new-world confirmation nor a significance conclusion.</p>'
    body += table(candidates, ('cohort', 'relation', 'kind', 'sources', 'all_lengths_all_sources_three_metrics_positive', 'CKA_delta', 'correct_change', 'wrong_change'))
    body += '<details open><summary>All relations, source coverage, and decompositions at the primary position (without selecting positive results)</summary>' + table(compact, ('cohort', 'relation', 'kind', 'metric', 'sources', 'defined', 'B0', 'Bt', 'dc', 'dw', 'Delta', 'median', 'sd', 'minimum', 'maximum', 'positive_sources', 'positive_all_lengths_sources')) + '</details>'
    body += '<h2>2. Dependence on layers, lengths, metrics, and task pairs</h2><p>The figures retain residual outputs from all four blocks and final_norm, each at the input-end position and the fixed contextual mean of numeric tokens (positions 4,6,…,2n+2). Block outputs precede final normalization and differ from the final_norm branch. The mean pools only numeric positions in one-line encoding, excluding task/query positions; it is a fixed sensitivity analysis, not a replacement for the primary position.</p>'
    body += '<p>The figures below jointly present B0, Bt, dc, dw, and Δ; each figure uses its own metric scale, and gray denotes undefined values. Both answer-stratified and ordinary per-length controls are shown, without choosing favorable layers based on results. Per-length plots retain all 21 lengths; pooling samples after subtracting length means does not replace per-length analysis.</p>'
    for cohort in sorted({row['cohort'] for row in figures}):
        body += '<details><summary>' + html.escape(cohort) + ': all fixed controls and metric figures</summary>'
        for fig in figures:
            if fig['cohort'] == cohort:
                body += '<p>' + html.escape(fig['control'] + ' / ' + fig['metric'] + ' / ' + fig['scope']) + '</p><img src="' + html.escape(fig['file'], quote=True) + '" alt="All predefined layer results">'
        body += '</details>'
    body += '<p>At the primary position,' + str(len(disagreement)) + ' source/relation cells have Bt or Δ sign disagreement across three primary metrics; ordinary/debiased CKA has' + str(len(changed)) + ' cells with Bt or Δ sign differences. Counts describe coverage with shared sources/relations, not independent replicates. All sign differences are retained in <a href="direction_agreement.csv">Common-source sign table</a>, <a href="debiased_sensitivity.csv">Debiased sensitivity table</a>.</p>'
    body += '<details><summary>Sign disagreements across the three metrics at the primary position</summary>' + table(disagreement, ('cohort', 'relation', 'source_unit', 'Bt_linear_cka', 'Bt_knn_overlap', 'Bt_procrustes', 'delta_linear_cka', 'delta_knn_overlap', 'delta_procrustes')) + '</details>'
    body += '<details><summary>Sign differences between ordinary/debiased CKA at the primary position</summary>' + table(changed, ('cohort', 'relation', 'source_unit', 'Bt_ordinary', 'Bt_debiased', 'delta_ordinary', 'delta_debiased')) + '</details>'
    primary_leave = [row for row in leave if row['branch'] == MAIN_BRANCH and row['control'] == MAIN_CONTROL and row['metric'] in PRIMARY]
    body += '<details><summary>Leave-one-relation and leave-one-shared-property-family analyses (primary position)</summary>' + table(primary_leave, ('cohort', 'kind', 'metric', 'exclusion_level', 'excluded', 'remaining_relations', 'B0', 'Bt', 'delta_correct', 'delta_wrong', 'delta')) + '</details>'
    body += '<p>Leaving out one relation does not remove other relations’ task/orbit dependence; leaving out a shared family removes all relations in that family together. Both are descriptive sensitivity analyses, not independent-relation bootstrap or new confirmation experiments. Complete <a href="leave_one_relation_family.csv">All-layer leave-one-out results</a> cover both self-invariances and cross-task relations, separated by cohort and kind.</p>'
    body += '<h2>3. Input embeddings, answers, and output associations</h2><p>Within a length, embeddings at the input-end token are identical; the unordered mean of numeric-token embeddings is also identical because every permutation contains exactly 1…n. Both geometries are mathematically degenerate and are explicitly recorded as undefined, without treating floating-point addition-order noise as input signal. This does not imply that downstream contextual means are degenerate.</p>'
    body += '<p>Flattened position-preserving numeric-token embedding/one-hot baselines were fixed before analysis. They preserve input position–value pairings; within-length positional and punctuation additive terms are sample constants that cancel under centering. Learned embedding baselines can change during training; static one-hot features have no interpretable training mechanism. These input baselines diagnose whether a signal already exists in input representations, rather than proving that the network discovers additional algebraic relations.</p>'
    controls = [row for row in relations if row['branch'] in INPUTS + CONTROLS and row['metric'] in PRIMARY]
    body += '<details><summary>All input, task-query, answer, and output controls</summary>' + table(controls, ('cohort', 'relation', 'branch', 'control', 'metric', 'planned_sources', 'B0_mean', 'Bt_mean', 'delta_correct_mean', 'delta_wrong_mean', 'delta_mean', 'delta_sources')) + '</details>'
    body += '<p>Answer stratification reuses the original protocol, separately subtracting the same answer-stratum means within fit/test; it cannot remove all confidence, error, or nonlinear answer information. The output-probability branch is retained; hidden Δ minus output Δ does not identify an “additional mechanism.” Ground-truth answers have no learnable step-0, so no training increment is invented for them. Query positions contain task prompts and are reported separately from pre-task states.</p>'
    body += '<h2>4. Connecting the task-count series and encoding controls</h2><p>This round does not repeat task-count drift analysis.<a href="../permworld_scope_inventory_20261009/report.html">Original inventory report</a>and<a href="../permworld_cka_20261009/report.html">Original CKA diagnostic</a>retain all original results. In the historical per-length initialization-retention curves, k16 exceeds k1 in all five replicates, but not every curve is monotone; R3/R4 are sensitivities to task subsets from the same sources, not new source seeds. High between-pool similarity together with high retention relative to each model’s initialization only suggests retained initial geometry; it cannot precisely decompose the sources. No new task-count conclusion is made here.</p>'
    body += '<p>The next engineering comparison of two encodings can test whether encoding changes B0 (initialization preference), Bt (trained pairing preference), and whether Δ comes from dc or dw. Genuine shared step-0, identical underlying permutations and orbit splits, output probabilities, ground-truth answers, and accuracy must be retained. If Δ changes sign across encodings while Bt is similar, this must not be called a reversal of the relation preference learned during training. Architectural controls must maintain shared input/supervision rules, with task learnability evaluated separately from geometric effects.</p>'
    body += '<h2>Conclusions and deliverables</h2><p>This round identifies layers/relations with consistent pairing-selectivity increments at fixed checkpoints and displays length/metric boundaries; CKA cannot establish encoding mechanisms, mathematical-relation representations, or causal function. The between-source standard deviation reflects only initialization differences within the shared world; no new input-sampling confidence intervals are constructed. Historical input-bootstrap conditional scope does not automatically extend to new layers and metrics.</p>'
    body += '<p><a href="layer_metric_results.csv">Complete raw relation/source/length table</a> · <a href="source_metric_summary.csv">Fixed-length source summaries</a> · <a href="relation_metric_summary.csv">Relation summaries</a> · <a href="length_metric_summary.csv">Per-length source differences</a> · <a href="layer_metric_summary.csv">Equal-relation descriptive summaries by layer</a> · <a href="summary_audit.json">Summary audit</a> · <a href="summarize.py">Summary script</a>. Completing these existing-model diagnostics does not automatically start formal batch training, GroupWorld, or F17 experiments.</p>'
    css = 'body{font:16px/1.65 system-ui,sans-serif;max-width:1400px;margin:34px auto;padding:0 20px;color:#25313d}h1,h2{line-height:1.3}h2{margin-top:35px}.lead{background:#e8f0f5;padding:16px;border-radius:7px}.scroll{overflow:auto}table{border-collapse:collapse;font-size:12px;margin:14px 0;width:100%}th,td{border:1px solid #d9e0e5;padding:6px;text-align:left;white-space:nowrap}th{background:#edf2f5}img{max-width:100%;margin:16px 0}details{margin:20px 0}a{color:#175aa1}'
    (root / 'report.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>PermWorld layer diagnostic</title><style>' + css + '</style>' + body + '</html>')
    text = f'''PermWorld exploratory multilayer diagnostics of existing checkpoints\nAdded source training: 0. All historical initialization references are C, with original step-0 identity unverified.\nPrimary-position final_norm_end, answer-stratified candidates with positive Δ across all available sources and three primary metrics:{candidate_names}.\nPrimary-position cells with Bt/Δ sign disagreement across three metrics:{len(disagreement)}; ordinary/debiased sign differences:{len(changed)}.\nAll layers, lengths, relations, and fixed pooling are retained; Δ is decomposed as correct-pair change minus wrong-pair change.\nThe end-position embedding and unordered numeric mean are mathematically constant; their metrics are undefined.\nThe task-count series reuses historical diagnostics without recomputation; similarity sources cannot be precisely decomposed.\nThe next encoding experiment asks whether B0 or Bt changes and requires genuine step-0 and shared underlying data.\nCKA, neighborhood overlap, and Procrustes alone do not identify mathematical mechanisms; source/relation dependence is retained, with no new confirmatory p-values.\n'''
    (root / 'brief_report.txt').write_text(text)


def run(root=ROOT, make_plots=True):
    root = Path(root)
    start = time.monotonic()
    path = root / 'layer_metric_results.csv'
    rows = read(path)
    if not rows:
        raise ValueError('No metric cells; never generate an empty success report.')
    for row in rows:
        assert row['domain'] == 'PermWorld', 'F17/F5 not in this extension.'
        assert row['metric'] in METRICS, 'Unexpected/unfrozen metric.'
        assert row.get('initialization_grade', 'C') == 'C'
        assert_decomposition(row)
    units = source_summary(rows)
    relations = relation_summary(units)
    lengths = length_summary(rows)
    agreements, sensitivity = direction_agreement(units)
    leave = leave_one(relations)
    layers = layer_summary(relations)
    outputs = [('source_metric_summary.csv', units), ('relation_metric_summary.csv', relations),
               ('length_metric_summary.csv', lengths), ('direction_agreement.csv', agreements),
               ('debiased_sensitivity.csv', sensitivity), ('leave_one_relation_family.csv', leave),
               ('layer_metric_summary.csv', layers)]
    for name, values in outputs:
        write(root, name, values)
    figures = plots(root, relations, lengths) if make_plots else []
    audit = dict(metric_rows=len(rows), source_summary_rows=len(units), relation_summary_rows=len(relations),
                 length_summary_rows=len(lengths), source_units=sorted({row['source_unit'] for row in rows}),
                 cohorts=sorted({row['cohort'] for row in rows}), planned_lengths=LENGTHS,
                 complete_length_rule=True, undefined_source_delta=sum(row['delta'] is None for row in units),
                 incomplete_length_groups=sum(not row['all_lengths_present'] for row in units),
                 decomposition_checks=len(rows) + len(units), new_source_training=0,
                 initialization_grade='C', original_initial_identity_verified=False,
                 seed_uncertainty='descriptive within shared worlds; no independent dataset/world CI',
                 relation_uncertainty='dependent tasks/actions retained; no independent relation CI',
                 input_uncertainty='fixed samples; no new input bootstrap manufactured',
                 metric_input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                 summary_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                 elapsed_seconds=time.monotonic() - start)
    (root / 'summary_audit.json').write_text(json.dumps(audit, indent=2, ensure_ascii=False) + '\n')
    report(root, relations, units, agreements, sensitivity, leave, figures, audit)
    print(json.dumps(dict(status='complete', metric_rows=len(rows), summary_rows=len(units),
                          elapsed_seconds=round(time.monotonic() - start, 2)), ensure_ascii=False), flush=True)
    return audit


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT, help='Independent diagnostic directory; fixtures may use a temporary path.')
    parser.add_argument('--no-plots', action='store_true', help='Only for CPU fixtures; full delivery includes all figures.')
    args = parser.parse_args()
    run(args.root, make_plots=not args.no_plots)
