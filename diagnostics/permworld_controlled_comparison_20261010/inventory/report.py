"""Controlled-comparison report for existing first-round20k models; reads only new analysis CSVs.

No model fitting or weight/raw-input loading; the three metrics retain separate coordinates, without a composite score.
Retain all missing/undefined values. Primary analysis is fixed at answer_strata, hidden, final_norm.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import html
import json
import math
from pathlib import Path
import tempfile

import numpy as np


DIRECTORY = Path(__file__).resolve().parent.parent
ARCHITECTURES = ('transformer', 'gru')
ENCODINGS = ('one_line', 'canonical_cycle')
RELATIONS = ('descent_inverse', 'peak_complement')
TASKS = ('descents', 'recoils', 'peaks', 'valleys')
METRICS = ('linear_cka', 'knn_overlap', 'procrustes')
FIELDS = ('initial_correct', 'initial_wrong', 'initial_identity',
          'trained_correct', 'trained_wrong', 'trained_identity',
          'B0', 'Bt', 'dc', 'dw', 'di', 'Delta')
NUMERIC = set(FIELDS) | {'length', 'samples', 'answer_accuracy', 'answer_ce',
                       'length_majority_accuracy', 'value', 'error',
                       'initial_correct_error', 'trained_correct_error'}
LABELS = dict(descent_inverse='Descents—Inverse descents / Inverse', peak_complement='Peaks—Valleys / Complement',
              transformer='Transformer', gru='GRU', one_line='one-line', canonical_cycle='canonical cycle',
              linear_cka='Linear CKA', knn_overlap='Neighborhood overlap k=10', procrustes='Held-out Procrustes (negative NRMSE)')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for value in iter(lambda: f.read(1048576), b''):
            h.update(value)
    return h.hexdigest()


def number(value):
    if value is None or str(value).strip().lower() in ('', 'none', 'nan', 'null'):
        return float('nan')
    return float(value)


def read_csv(path):
    with Path(path).open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        for name in set(row) & NUMERIC:
            row[name] = number(row[name])
    return rows


def key(row):
    return tuple(row.get(k, '') for k in ('architecture', 'encoding', 'relation', 'control', 'branch', 'layer', 'metric'))


def finite(value):
    return isinstance(value, (float, int)) and math.isfinite(value)


def display(value, percentage=False):
    if isinstance(value, (float, int)):
        if not math.isfinite(value):
            return 'undefined/missing'
        return f'{100*value:.2f}%' if percentage else f'{value:.6f}'
    return str(value)


def direction(values):
    if any(not finite(v) for v in values):
        return 'undefined'
    signs = [1 if v > 0 else -1 if v < 0 else 0 for v in values]
    return 'same direction' if len(set(signs)) == 1 else 'different directions'


def collect(directory):
    directory = Path(directory)
    required = ('protocol.json', 'per_length_results.csv', 'contrast_summary.csv',
                'source_learning.csv', 'cross_arch_same_task.csv', 'cost_estimate.json')
    for name in required:
        assert (directory / name).is_file(), ('Missing required report input', name)
    protocol = json.loads((directory / 'protocol.json').read_text())
    rows = read_csv(directory / 'contrast_summary.csv')
    lengths = read_csv(directory / 'per_length_results.csv')
    conditions = [r for r in rows if r['contrast_type'] == 'condition_summary']
    lookup = {key(row): row for row in conditions}
    assert len(lookup) == len(conditions), 'Duplicate condition summaries'
    main_keys = [(a, e, r, 'answer_strata', 'hidden', 'final_norm', m)
                 for a in ARCHITECTURES for e in ENCODINGS for r in RELATIONS for m in METRICS]
    assert all(k in lookup for k in main_keys), 'The fixed primary analysis lacks conditions; do not select a subset to generate a report'
    # These equations are identities, not significance or mechanism tests.
    for row in conditions:
        if all(finite(row.get(f)) for f in ('B0', 'Bt', 'dc', 'dw', 'Delta')):
            assert abs(row['Delta'] - (row['Bt'] - row['B0'])) < 1e-8
            assert abs(row['Delta'] - (row['dc'] - row['dw'])) < 1e-8
    main = [lookup[k] for k in main_keys]
    directions = []
    for a in ARCHITECTURES:
        for e in ENCODINGS:
            for relation in RELATIONS:
                metrics = [lookup[a, e, relation, 'answer_strata', 'hidden', 'final_norm', m] for m in METRICS]
                directions.append(dict(architecture=a, encoding=e, relation=relation,
                    B0_direction=direction([r['B0'] for r in metrics]),
                    Bt_direction=direction([r['Bt'] for r in metrics]),
                    Delta_direction=direction([r['Delta'] for r in metrics])))
    source = read_csv(directory / 'source_learning.csv')
    source_groups = {}
    for row in source:
        group = tuple(row[k] for k in ('architecture', 'encoding', 'task', 'condition', 'split', 'action'))
        source_groups.setdefault(group, []).append(row)
    source_summary = []
    for group, group_rows in sorted(source_groups.items()):
        ns = [int(r['length']) for r in group_rows if finite(r['length'])]
        complete = sorted(ns) == list(range(10, 31)) and len(group_rows) == 21
        summary = dict(zip(('architecture', 'encoding', 'task', 'condition', 'split', 'action'), group))
        for field in ('answer_accuracy', 'answer_ce', 'length_majority_accuracy'):
            values = [row.get(field, float('nan')) for row in group_rows]
            summary[field] = math.fsum(values) / 21 if complete and all(finite(v) for v in values) else float('nan')
        summary['lengths_complete'] = complete
        summary['minimum_length_answer_accuracy'] = min(r['answer_accuracy'] for r in group_rows) if complete and all(finite(r.get('answer_accuracy')) for r in group_rows) else float('nan')
        summary['undefined_reason'] = '' if complete else 'Incomplete fixed21-length support; remaining lengths are not reweighted'
        source_summary.append(summary)
    encoding_changes = []
    for a in ARCHITECTURES:
        for relation in RELATIONS:
            line = lookup[a, 'one_line', relation, 'answer_strata', 'hidden', 'final_norm', 'linear_cka']
            cycle = lookup[a, 'canonical_cycle', relation, 'answer_strata', 'hidden', 'final_norm', 'linear_cka']
            change = dict(architecture=a, relation=relation)
            for name, field in (('Gamma0', 'B0'), ('Gammat', 'Bt'), ('Gammac', 'dc'), ('Gammaw', 'dw'), ('GammaDelta', 'Delta')):
                change[name] = cycle[field] - line[field] if finite(cycle[field]) and finite(line[field]) else float('nan')
            change['Delta reverses while Bt does not'] = all(finite(r[f]) for r in (line, cycle) for f in ('Bt', 'Delta')) and line['Delta'] * cycle['Delta'] < 0 and line['Bt'] * cycle['Bt'] > 0
            encoding_changes.append(change)
    metadata = {name: json.loads((directory / name).read_text()) for name in ('cost_estimate.json', 'validation_audit.json', 'completion.json') if (directory / name).exists()}
    gold = read_csv(directory / 'gold_baseline_length.csv') if (directory / 'gold_baseline_length.csv').exists() else []
    inputs = {str(directory / name): sha(directory / name) for name in required}
    for name in ('validation_audit.json', 'completion.json', 'gold_baseline_length.csv', 'comparison_manifest.csv', 'scope_status.csv'):
        if (directory / name).exists():
            inputs[str(directory / name)] = sha(directory / name)
    return dict(directory=directory, protocol=protocol, conditions=conditions, lookup=lookup,
                contrast_rows=rows, length_rows=lengths, main=main, directions=directions,
                source_rows=source, source_summary=source_summary, encoding_changes=encoding_changes,
                cross_arch=read_csv(directory / 'cross_arch_same_task.csv'), gold=gold,
                metadata=metadata, input_sha256=inputs)


def main_figure(result, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    figure = plt.figure(figsize=(18, 15))
    grid = figure.add_gridspec(2, 2, hspace=.34, wspace=.25)
    groups = [(a, e, r) for a in ARCHITECTURES for e in ENCODINGS for r in RELATIONS]
    names = [f'{"TF" if a=="transformer" else "GRU"}/{"line" if e=="one_line" else "cycle"}/{"inv" if r=="descent_inverse" else "comp"}' for a, e, r in groups]
    lookup = result['lookup']; colors = dict(one_line='#2871ad', canonical_cycle='#d67330')
    undefined_labels = []
    sub = grid[0, 0].subgridspec(1, 3, wspace=.36)
    for col, metric in enumerate(METRICS):
        axis = figure.add_subplot(sub[0, col])
        for i, (a, e, r) in enumerate(groups):
            row = lookup[a, e, r, 'answer_strata', 'hidden', 'final_norm', metric]
            for offset, field, marker, color in ((-.12, 'B0', 'o', '#7f7f7f'), (.12, 'Bt', 's', '#235a97')):
                value = row[field]
                if finite(value):
                    axis.plot(value, i + offset, marker, color=color, markersize=5)
                else:
                    axis.annotate('undefined', xy=(0, i + offset), fontsize=6, color='#a22')
                    undefined_labels.append(f'A/{a}/{e}/{r}/{metric}/{field}')
        axis.axvline(0, color='#bbb', linewidth=.6)
        axis.set_yticks(range(8), names if col == 0 else [''] * 8, fontsize=7)
        axis.invert_yaxis(); axis.grid(axis='x', alpha=.2)
        axis.set_title(('A  ' if col == 0 else '') + ('Linear CKA' if metric=='linear_cka' else 'kNN overlap' if metric=='knn_overlap' else '-NRMSE Procrustes'), fontsize=10)
        axis.set_xlabel('Correct - wrong; separate metric scale', fontsize=8)
    axis = figure.add_subplot(grid[0, 1])
    for i, (a, e, r) in enumerate(groups):
        row = lookup[a, e, r, 'answer_strata', 'hidden', 'final_norm', 'linear_cka']
        for offset, field, marker in ((-.12, 'B0', 'o'), (.12, 'Bt', 's')):
            if finite(row[field]):
                axis.plot(row[field], i + offset, marker, color=colors[e], markerfacecolor='none' if field=='B0' else colors[e])
            else:
                axis.annotate('undefined', (0, i + offset), fontsize=7)
                undefined_labels.append(f'B/{a}/{e}/{r}/{field}')
    axis.axvline(0, color='#bbb', linewidth=.6); axis.set_yticks(range(8), names, fontsize=8)
    axis.invert_yaxis(); axis.grid(axis='x', alpha=.2)
    axis.set_title('B  Fixed CKA: model configuration comparison')
    axis.set_xlabel('B0 open circles / Bt filled squares; capacity is unmatched')
    axis = figure.add_subplot(grid[1, 0])
    for i, row in enumerate(result['encoding_changes']):
        for offset, field, marker, color in ((-.12, 'Gamma0', 'o', '#777'), (.12, 'Gammat', 's', '#235a97')):
            if finite(row[field]):
                axis.plot(row[field], i + offset, marker, color=color)
            else:
                axis.annotate('undefined', (0, i + offset), fontsize=7)
                undefined_labels.append('C/' + row['architecture'] + '/' + row['relation'] + '/' + field)
    axis.set_yticks(range(4), [f'{r["architecture"]}/{r["relation"]}' for r in result['encoding_changes']], fontsize=9)
    axis.axvline(0, color='#bbb', linewidth=.6); axis.invert_yaxis(); axis.grid(axis='x', alpha=.2)
    axis.set_title('C  Fixed CKA: encoding difference cycle - line')
    axis.set_xlabel('Gamma0 circles / Gammat squares; initialization and trained preference separated')
    sub = grid[1, 1].subgridspec(2, 1, hspace=.35)
    for j, relation in enumerate(RELATIONS):
        axis = figure.add_subplot(sub[j, 0])
        for a in ARCHITECTURES:
            for e in ENCODINGS:
                layers = [f'{"block" if a=="transformer" else "layer"}{i}' for i in range(1, 5)] + ['final_norm']
                x = [1, 2, 3, 4, 4.65]
                for field in ('B0', 'Bt'):
                    values = []
                    for layer in layers:
                        row = lookup.get((a, e, relation, 'answer_strata', 'hidden', layer, 'linear_cka'))
                        value = row.get(field, float('nan')) if row else float('nan')
                        values.append(value)
                        if not finite(value):
                            undefined_labels.append(f'D/{a}/{e}/{relation}/{layer}/{field}')
                    axis.plot(x, values, color=colors[e], linestyle='--' if field=='B0' else '-',
                              marker='o' if a=='transformer' else '^', linewidth=1,
                              alpha=.55 if field=='B0' else 1., label=f'{a}/{e}/{field}')
        axis.axhline(0, color='#bbb', linewidth=.6); axis.grid(alpha=.15)
        axis.set_xticks([1, 2, 3, 4, 4.65], ['1', '2', '3', '4', 'final norm'])
        axis.set_title(('D  Fixed CKA: complete layers — ' if j==0 else '') + relation, fontsize=10)
        axis.set_ylabel('Correct - wrong')
        axis.legend(fontsize=6, ncol=2, loc='best')
    figure.suptitle('First16 fixed20k PermWorld models; answer-strata control; one world\nInitialization B0 and trained Bt shown, no metric averaging or significance claim', fontsize=14)
    figure.text(.02, .01, 'Embedding at the input-end token is constant within length: undefined values are retained, never drawn as noise points. Layer4 and final norm are distinct.\nDashed layer lines=B0; solid=Bt. TF block/GRU layer numbering is not a matched computational stage. Missing plotted cells: ' + str(len(undefined_labels)), fontsize=9)
    figure.subplots_adjust(top=.92, bottom=.08)
    figure.savefig(Path(output) / 'four_part_main.png', dpi=140)
    plt.close(figure)
    return undefined_labels


def table(rows, columns=None):
    if not rows:
        return '<p>Not provided; retain missing status without fabricating results.</p>'
    columns = columns or list(dict.fromkeys(k for row in rows for k in row))
    return '<div class="scroll"><table><thead><tr>' + ''.join('<th>' + html.escape(k) + '</th>' for k in columns) + '</tr></thead><tbody>' + ''.join('<tr>' + ''.join('<td>' + html.escape(display(row.get(k, ''))) + '</td>' for k in columns) + '</tr>' for row in rows) + '</tbody></table></div>'


def render(result, output, fixture=False):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    undefined_plot = main_figure(result, output)
    directions = result['directions']
    bt_agree = sum(r['Bt_direction']=='same direction' for r in directions)
    delta_agree = sum(r['Delta_direction']=='same direction' for r in directions)
    bt_missing = sum(r['Bt_direction']=='undefined' for r in directions)
    delta_missing = sum(r['Delta_direction']=='undefined' for r in directions)
    flip_without_bt = sum(r['Delta reverses while Bt does not'] for r in result['encoding_changes'])
    configuration = [r for r in result['contrast_rows'] if r['contrast_type']=='configuration_difference' and r.get('metric')=='linear_cka' and r.get('branch')=='hidden' and r.get('layer')=='final_norm' and r.get('control')=='answer_strata']
    layer_changes = [r for r in result['contrast_rows'] if r['contrast_type']=='layer_difference' and r.get('metric')=='linear_cka' and r.get('branch')=='hidden' and r.get('control')=='answer_strata']
    interactions = [r for r in result['contrast_rows'] if r['contrast_type']=='interaction']
    embedding = [r for r in result['conditions'] if r.get('layer')=='embedding']
    layer_sign_change = 0; layer_incomplete = 0
    for a in ARCHITECTURES:
        for e in ENCODINGS:
            for relation in RELATIONS:
                names = [f'{"block" if a=="transformer" else "layer"}{i}' for i in range(1, 5)] + ['final_norm']
                values = [result['lookup'].get((a,e,relation,'answer_strata','hidden',layer,'linear_cka'),{}).get('Bt',float('nan')) for layer in names]
                status = direction(values)
                layer_sign_change += status == 'different directions'; layer_incomplete += status == 'undefined'
    text = [
        'PermWorld first-round 16 models: controlled representation comparison',
        'Generated UTC: ' + datetime.now(timezone.utc).isoformat(),
        'Synthetic CPU fixture, not actual results.' if fixture else 'Only primary queue: first-round four tasks×two encodings×two configurations, fixed20k, n10..30, genuine saved step-0.',
        'First answer four controlled comparisons (all exploratory descriptions, not confirmatory tests):',
        f'A Metrics: across8 configuration×encoding×relation primary conditions, the three primary metrics agree on trained Bt direction in{bt_agree}/8 and on Delta direction in{delta_agree}/8; undefined counts are respectively{bt_missing}, {delta_missing}. Do not compare absolute scores/Delta magnitudes across metrics or average them into a composite.',
        f'B Model configuration: holding CKA, encoding, task pair, underlying inputs, and20k budget fixed, retain{len(configuration)} primary configuration contrasts (GRU−Transformer), with full B0/Bt/dc/dw/Delta in the table;3.24 million and1.66 million parameter counts are unmatched, precluding a pure architectural causal interpretation.',
        f'C Encoding: all four within-configuration relation contrasts retain Gamma0/Gammat/Gammac/Gammaw/GammaDelta; the count reversing Delta without reversing Bt across encodings is{flip_without_bt}/4. Such changes are not reversals of trained relation preference.',
        f'D Extraction layer: show four layers and separate final_norm for both configurations, encodings, and relations; among8 conditions, Bt changes direction across layers in{layer_sign_change}, and conditions with undefined layers number{layer_incomplete}. Constant input-end-token embeddings remain undefined, without plotting floating-point noise; distinguish normalization changes from layer changes.',
        '',
        'Shared fixed items: the same data world/orbit split/anchor order; compute per length before fixed1/21 aggregation. Joint-answer stratification is primary, with raw unadjusted, output-probability, and ground-truth baselines alongside. An undefined length makes the fixed-weight aggregate undefined; do not reweight remaining lengths.',
        'Retain correct, wrong, and identity raw similarities in each condition: B0=S_correct,initial−S_wrong,initial; Bt=S_correct,trained−S_wrong,trained; dc/dw are correct/wrong changes; Delta=Bt−B0=dc−dw. Positive Delta does not guarantee increased correct similarity; initial pairing preferences require separate inspection.',
        'The two encodings share genuine initial parameters and vocabulary within each configuration, but encoded-input geometry may differ. GRU uses explicit interlayer dropout with exact migration of21 original initialization tensors, rather than assuming equal seeds imply equal weights. Initial weights differ across configurations.',
        'Procrustes uses negative held-out NRMSE and retains raw errors: fit8-dimensional PCA/orthogonal maps on independent validation data and evaluate only on test data. kNN uses k=10, Euclidean distance, and stable anchor-ID tie breaking. Ordinary CKA and U-centered sensitivity retain previous definitions. See protocol for metric-specific preprocessing.',
        'Answer stratification does not establish removal of output probabilities, confidence, or error structure. Do not subtract output Delta from hidden Delta to estimate extra mechanisms. No initialization/training Delta is invented for static ground-truth baselines.',
        'Descents–inverse descents and peaks–valleys are two relation instances with different task pairs and output statistics; comparing them is not a causal experiment changing only the mathematical relation. Correct/wrong pairing selectivity also does not establish an algebraic computation mechanism.',
        'Retain all low-performing task conditions. These compare encoding/configuration under a fixed budget, not mechanisms at matched mastery. No90% pass threshold, additional training, or best-checkpoint selection is used. Previous teacher-forced validation EOS accuracy cannot replace answer accuracy.',
        'There is one data world and one initialization per configuration. The16 models,21 lengths, and three metrics are not independent replications. No confirmatory p-values, equivalence tests, or population-generalization confidence claims are reported. Individual contrasts do not rule out interactions; architecture×encoding interactions are exploratory only.',
        'Historical82-model multilayer results remain a separate reference at diagnostics/permworld_layers_20261009/; initialization was reconstructed by the original procedure but exact identity is unverified. They are excluded from the primary queue, as are all second-round8k learnability models.',
        'This round performs forward passes/metrics/reporting only: no optimizer, backward pass, added samples/updates, or tuning; no GroupWorld,768-model confirmation, finite-field, GELU, null-space, or F17 work.',
    ]
    metadata_ready = 'validation_audit.json' in result['metadata'] and 'completion.json' in result['metadata']
    text.append('Final verification/stopping status: ' + ('see validation_audit.json and completion.json; this report does not modify execution state.' if metadata_ready else 'Final audit or completion is missing; this report remains pending verification and does not claim full completion.'))
    text += ['', 'Three-primary-metric direction table (numerical signs, not significance):', json.dumps(directions, ensure_ascii=False, indent=2),
             '', 'Encoding decomposition:', json.dumps([{k:(v if not isinstance(v,float) or math.isfinite(v) else None) for k,v in r.items()} for r in result['encoding_changes']],ensure_ascii=False,indent=2),
             '', 'All primary-condition B0/Bt/dc/dw/Delta and raw correct/wrong/identity scores:']
    for row in result['main']:
        text.append('/'.join(str(v) for v in key(row)) + ' | ' + ', '.join(f'{f}={display(row.get(f,float("nan")))}' for f in FIELDS) + ' | ' + row.get('undefined_reason',''))
    text += ['', 'Source-performance summaries over fixed21 lengths (all condition/split/action values retained):']
    for row in result['source_summary']:
        text.append('/'.join(str(row[k]) for k in ('architecture','encoding','task','condition','split','action')) +
                    f' | answer acc={display(row["answer_accuracy"],True)}, answerCE={display(row["answer_ce"])}, train-defined per-length mode={display(row["length_majority_accuracy"],True)}, min-length acc={display(row["minimum_length_answer_accuracy"],True)} | {row["undefined_reason"]}')
    text += ['', 'See per_length_results.csv/contrast_summary.csv for raw per-length results, all differences, and identity pairings; retain all NaNs and reasons.',
             'Direct same-task cross-configuration similarity is separate in cross_arch_same_task.csv; it describes model similarity and is not combined with relation selectivity as one effect.',
             'Compute budget and cost sources:', json.dumps(result['metadata'].get('cost_estimate.json',{}), ensure_ascii=False,indent=2),
             'Stop after feasible comparisons, independent checks, and reporting. Identify missing items explicitly; do not automatically expand training or start another system.']
    (output / 'reader_report.txt').write_text('\n'.join(text) + '\n')
    links = ['comparison_manifest.csv','per_length_results.csv','contrast_summary.csv','source_learning.csv',
             'cross_arch_same_task.csv','gold_baseline_length.csv','scope_status.csv','validation_audit.json','completion.json']
    main_html = '<h1>PermWorld controlled representation comparison</h1><pre>' + html.escape('\n'.join(text[:9])) + '</pre><p>' + ' · '.join(f'<a href="{name}">{name}</a>' for name in links) + '</p>'
    main_html += '<img class="main" src="four_part_main.png" alt="Fixed four-part figure: metrics, configuration, encoding, and layers">'
    main_html += '<p>All primary plotted conditions use fixed answer strata. Undefined plotted-cell count:'+str(len(undefined_plot))+'; see report_export_audit.json for cells. Undefined embeddings are retained separately, not interpreted as zero effects. Metrics have separate axes.</p>'
    main_html += '<details open><summary>What can and cannot be concluded</summary><pre>' + html.escape('\n'.join(text[9:20])) + '</pre></details>'
    main_html += '<h2>A Primary metric directions and complete primary conditions</h2>' + table(directions) + table(result['main'], ['architecture','encoding','relation','metric',*FIELDS,'undefined_reason','undefined_lengths'])
    main_html += '<h2>B Configuration difference: GRU−Transformer</h2>' + table(configuration)
    main_html += '<h2>C Encoding difference: canonical cycle−one-line</h2>' + table(result['encoding_changes'])
    main_html += '<h2>D All layer differences and undefined embeddings</h2>' + table(layer_changes) + table(embedding)
    main_html += '<h2>Architecture×encoding interactions: exploratory, separating initial and trained states</h2>' + table(interactions)
    main_html += '<h2>Source performance: fixed21 lengths, all conditions retained</h2>' + table(result['source_summary'])
    for title, subset in (
        ('Raw unadjusted readings alongside controls', [r for r in result['conditions'] if r.get('control')=='raw_within_length']),
        ('Output-probability baseline: conditional probabilities over valid numbers0..30', [r for r in result['conditions'] if r.get('branch')=='output_prob']),
        ('Prespecified U-centered CKA sensitivity', [r for r in result['conditions'] if r.get('metric')=='debiased_cka']),
        ('Direct same-task cross-configuration similarity, with genuine initialization references', result['cross_arch']),
    ):
        main_html += '<details><summary>'+title+'</summary>' + table(subset) + '</details>'
    main_html += '<details><summary>Complete per-length static ground-truth baseline: no invented Delta</summary>' + table(result['gold']) + '</details>'
    main_html += '<details><summary>All contrast rows / no condition selection</summary>' + table(result['contrast_rows']) + '</details>'
    main_html += '<details><summary>All per-length tables / no missing-length reweighting</summary>' + table(result['length_rows']) + '</details>'
    main_html += '<pre>' + html.escape(text[18]+'\n'+text[19]) + '</pre>'
    main_html += '<details><summary>Costs, final checks, and stopping records</summary><pre>' + html.escape(json.dumps(result['metadata'],ensure_ascii=False,indent=2)) + '</pre></details>'
    (output / 'reader_report.html').write_text('<!doctype html><meta charset="utf-8"><title>PermWorld controlled comparison</title><style>body{max-width:1350px;margin:25px auto;padding:0 20px;font:15px/1.5 system-ui}pre{white-space:pre-wrap}.main{width:100%}.scroll{overflow:auto}table{border-collapse:collapse;white-space:nowrap;font-size:12px}td,th{border:1px solid #ddd;padding:5px}th{background:#edf1f5}details{margin:18px 0}summary{font-weight:600;cursor:pointer}</style>'+main_html)
    audit = dict(status='passed', generated_utc=datetime.now(timezone.utc).isoformat(), fixture=fixture,
        reporter_sha256=sha(__file__), input_sha256=result['input_sha256'],
        main_conditions=24, metric_direction_conditions=8,
        main_metric_scales_not_averaged=True, fixed_lengths=list(range(10,31)),
        source_group_count=len(result['source_summary']), source_input_rows=len(result['source_rows']),
        source_incomplete_length_groups=sum(not r['lengths_complete'] for r in result['source_summary']),
        undefined_plot_cells=undefined_plot, embedding_summary_rows=len(embedding),
        no_nan_zero_imputation=True, no_incomplete_length_reweighting=True,
        final_audit_and_completion_available=metadata_ready,
        scope=dict(training_started=False, optimizer_constructed=False, weights_loaded=False,
                   raw_inputs_read=False, raw_activations_read=False, test_labels_read=False,
                   second_round8k_used=False, F17_modified=False, old_results_modified=False),
        output_sha256={name:sha(output/name) for name in ('reader_report.html','reader_report.txt','four_part_main.png')})
    (output / 'report_export_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    return audit


def report(directory=DIRECTORY):
    return render(collect(directory), directory)


def fixture():
    checks = {}
    def check(name, value):
        assert value, name
        checks[name] = True
    with tempfile.TemporaryDirectory(prefix='controlled-comparison-report-') as tmp:
        directory=Path(tmp)
        def csv_write(name, rows):
            with (directory/name).open('w',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
                writer.writeheader();writer.writerows(rows)
        (directory/'protocol.json').write_text(json.dumps(dict(lengths=list(range(10,31)),purpose='synthetic_only')))
        (directory/'cost_estimate.json').write_text(json.dumps(dict(fixture_only=True,GPU_seconds=0)))
        rows=[];lengths=[]
        for a in ARCHITECTURES:
            for e in ENCODINGS:
                for relation in RELATIONS:
                    layers=['embedding']+[f'{"block" if a=="transformer" else "layer"}{n}' for n in range(1,5)]+['final_norm']
                    for layer in layers:
                        for metric in (('linear_cka','knn_overlap','procrustes','debiased_cka') if layer=='final_norm' else ('linear_cka',)):
                            b0=.1 if e=='one_line' else .3;bt=.2
                            fields=dict(initial_correct=.4+b0,initial_wrong=.4,initial_identity=.7,
                                trained_correct=.5+bt,trained_wrong=.5,trained_identity=.8,
                                B0=b0,Bt=bt,dc=.1+bt-b0,dw=.1,di=.1,Delta=bt-b0)
                            if layer=='embedding':fields={f:float('nan') for f in fields}
                            row=dict(contrast_type='condition_summary',architecture=a,encoding=e,relation=relation,
                                control='answer_strata',branch='hidden',layer=layer,metric=metric,**fields,
                                undefined_reason='constant input-end embedding' if layer=='embedding' else '',
                                undefined_lengths='10..30' if layer=='embedding' else '')
                            rows.append(row)
                            lengths.extend(dict(row,length=n) for n in range(10,31))
        csv_write('contrast_summary.csv',rows);csv_write('per_length_results.csv',lengths)
        source=[dict(architecture=a,encoding=e,task=t,condition='trained',split=s,action='identity',length=n,
            samples=256 if s=='validation' else 512,answer_accuracy=.31 if e=='canonical_cycle' else .91,
            answer_ce=1.3 if e=='canonical_cycle' else .3,length_majority_accuracy=.30 if s=='test' else '')
            for a in ARCHITECTURES for e in ENCODINGS for t in TASKS for s in ('validation','test') for n in range(10,31)]
        csv_write('source_learning.csv',source)
        cross=[dict(task=t,encoding=e,condition='initial',value=.3) for t in TASKS for e in ENCODINGS]
        csv_write('cross_arch_same_task.csv',cross)
        result=collect(directory);audit=render(result,directory,fixture=True)
        check('all24_main_metric_conditions',audit['main_conditions']==24 and len(result['main'])==24)
        check('all8_direction_conditions',len(result['directions'])==8)
        check('no_metric_averaged_score',audit['main_metric_scales_not_averaged'])
        check('undefined_embedding_retained',all(not finite(r['B0']) for r in result['conditions'] if r['layer']=='embedding'))
        check('Delta_flip_not_Bt_flip_identified',all(r['Delta reverses while Bt does not'] for r in result['encoding_changes']))
        check('source_validation_missing_baseline_not_zero',all(not finite(r['length_majority_accuracy']) for r in result['source_summary'] if r['split']=='validation'))
        check('four_part_main_generated',(directory/'four_part_main.png').is_file())
        check('missing_final_audit_not_claimed_complete',not audit['final_audit_and_completion_available'])
        check('all16_source_models_both_splits',len(result['source_summary'])==32)
        check('21_fixed_lengths',audit['fixed_lengths']==list(range(10,31)))
    answer=dict(status='passed',reporter_sha256=sha(__file__),checks=checks,synthetic_CPU_only=True,
                real_new_results_read=False,GPU_started=False,training_started=False,fixtures_removed=True)
    (Path(__file__).parent/'report_fixture.json').write_text(json.dumps(answer,ensure_ascii=False,indent=2)+'\n')
    return answer


if __name__=='__main__':
    parser=argparse.ArgumentParser();mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--fixture',action='store_true');mode.add_argument('--directory',type=Path)
    args=parser.parse_args()
    print(json.dumps(fixture() if args.fixture else report(args.directory),ensure_ascii=False,indent=2))
