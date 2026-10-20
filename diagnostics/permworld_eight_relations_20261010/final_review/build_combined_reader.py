"""English combined reader of completed, audited CSV/JSON artifacts only.

Run explicitly with --run after the fixed scientific scope and provenance
audit finish. No model, activation, raw data, scientific metric, fitting,
statistical test, checkpoint selection, or training is performed here.
All writes are confined to this final_review directory. This file does not
change the frozen scientific readers or their reports.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import html
import json
import math
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
PERM = HERE.parent
GRAPH = PERM.parent / 'graphworld_pilot_20261010'
FIELDS = tuple(f'{phase}_{role}' for phase in ('initial', 'trained')
               for role in ('correct', 'wrong', 'identity')) + ('B0', 'Bt', 'dc', 'dw', 'di', 'Delta')
METRICS = ('linear_cka', 'knn_overlap', 'procrustes')
CKAS = ('linear_cka', 'debiased_cka')
P_CONTROLS = ('raw_within_length', 'answer_strata')
G_CONTROLS = ('raw_within_n', 'answer_strata', 'answer_density_strata')
ARCHS = ('transformer', 'gru')
ENCODINGS = ('one_line', 'canonical_cycle')
RELATIONS = ('descent_inverse', 'peak_complement', 'fixed_complement',
             'exceedance_inverse', 'subsequence_complement', 'run_complement',
             'record_complement', 'decomposition_complement')
G_RELATIONS = ('reverse_max', 'reverse_min', 'complement_out', 'complement_in')
DISPLAY = {
    'descent_inverse': 'Descents—inverse descents(inversion)',
    'peak_complement': 'Peaks—valleys(complement)',
    'fixed_complement': 'Fixed points—anti-fixed points(complement)',
    'exceedance_inverse': 'Exceedances—deficiencies(inversion)',
    'subsequence_complement': 'LIS—LDS(complement)',
    'run_complement': 'Increasing runs—decreasing runs(complement)',
    'record_complement': 'Left-to-right maximum—minimum records(complement)',
    'decomposition_complement': 'Global descents—components(complement; right answer +1)',
    'reverse_max': 'Maximum out-degree—maximum in-degree (transpose)',
    'reverse_min': 'Minimum out-degree—minimum in-degree (transpose)',
    'complement_out': 'Maximum out-degree—minimum out-degree (graph complement)',
    'complement_in': 'Maximum in-degree—minimum in-degree (graph complement)',
}


def require(value, message):
    if not value:
        raise RuntimeError(message)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def number(value):
    return float(value) if value not in ('', None, 'None', 'nan', 'NaN') else float('nan')


def direction(value):
    value = number(value)
    return 'undefined' if not math.isfinite(value) else 'positive' if value > 0 else 'negative' if value < 0 else 'zero'


def same(a, b):
    a, b = number(a), number(b)
    return (math.isnan(a) and math.isnan(b)) or a == b


def yes(value):
    return str(value).lower() == 'true'


def display(value):
    if isinstance(value, bool):
        return 'yes' if value else 'no'
    if value in ('', None, 'None', 'nan', 'NaN'):
        return 'undefined'
    try:
        value = float(value)
        return f'{value:.7g}' if math.isfinite(value) else 'undefined'
    except (ValueError, TypeError):
        return str(value)


def field_row(row, prefix='', **identity):
    return identity | {field: row.get(prefix + field, '') for field in FIELDS}


def indexed(rows, keys, expected):
    result = {tuple(row[key] for key in keys): row for row in rows}
    require(len(result) == len(rows) == expected, 'Missing or duplicated cells: ' + '/'.join(keys))
    return result


class Inputs:
    """SHA-check small artifacts; never open a checkpoint or numeric array."""
    def __init__(self):
        self.records = {}
        self.expected = {}

    def record(self, path, reason, expected=None):
        path = Path(path).resolve()
        require(path.is_file(), 'Missing required artifact: ' + str(path))
        require(path.suffix.lower() in ('.json', '.csv', '.html', '.txt', '.png'),
                'Reader refuses model/data/cache artifacts: ' + str(path))
        key = str(path)
        if key not in self.records:
            before = path.stat()
            digest = sha(path)
            after = path.stat()
            require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns),
                    'Input changed while hashing: ' + key)
            self.records[key] = dict(path=key, sha256=digest, bytes=after.st_size,
                                     mtime_ns=after.st_mtime_ns, reasons=[])
        item = self.records[key]
        if reason not in item['reasons']:
            item['reasons'].append(reason)
        expected = expected or self.expected.get(key)
        if expected is not None:
            require(item['sha256'] == expected, 'Audited SHA differs: ' + key)
            item['verified_against_existing_audit'] = True
        return path

    def js(self, path, reason, expected=None):
        return json.loads(self.record(path, reason, expected).read_text())

    def csv(self, path, reason, expected=None):
        with self.record(path, reason, expected).open(newline='') as stream:
            return list(csv.DictReader(stream))

    def stable(self):
        for item in self.records.values():
            stat = Path(item['path']).stat()
            require((stat.st_size, stat.st_mtime_ns) == (item['bytes'], item['mtime_ns']),
                    'Input changed during report assembly: ' + item['path'])


def load_completed():
    inputs = Inputs()
    audit = inputs.js(HERE / 'final_audit.json', 'completed provenance audit')
    require(audit.get('status') == 'passed' and audit.get('all_fixed_stages_complete') is True,
            'Complete provenance audit must pass before combined reporting')
    require(audit.get('old_artifacts_modified') is False and audit.get('GPU_started') is False,
            'Unexpected final provenance audit scope')
    ledger = inputs.js(HERE / 'physical_file_manifest.json', 'completed physical SHA ledger')
    require(ledger.get('status') == 'passed', 'Physical provenance ledger did not pass')
    for item in ledger['records']:
        key = str(Path(item['path']).resolve())
        require(key not in inputs.expected or inputs.expected[key] == item['sha256'],
                'Conflicting physical ledger SHA')
        inputs.expected[key] = item['sha256']
    # Recheck already-read ledger-bound files without scanning giant weights or caches.
    for item in list(inputs.records.values()):
        inputs.record(item['path'], 'provenance metadata identity')
    binding = inputs.js(HERE / 'combined_reader_freeze.json', 'combined writer source/protocol binding')
    require(binding['source_sha256'] == sha(__file__), 'Combined writer source differs from its binding')
    p = inputs.js(PERM / 'protocol.json', 'fixed PermWorld protocol', binding['perm_protocol_sha256'])
    g = inputs.js(GRAPH / 'protocol.json', 'fixed GraphWorld protocol', binding['graph_protocol_sha256'])
    require(p['lengths'] == list(range(10, 31)) and g['lengths'] == [6, 7, 8], 'Fixed lengths changed')
    require(tuple(r['pair_id'] for r in p['relations']) == RELATIONS, 'Eight-relation order changed')
    require(tuple(r['relation'] for r in g['relations']) == G_RELATIONS, 'Four graph relations changed')
    require(tuple(p['architectures']) == ARCHS and tuple(p['encodings']) == ENCODINGS,
            'Fixed architecture/encoding grid changed')
    require(len(p['checkpoint_inventory']) == 64 and len(p['training_manifest']) == 48 and
            len(g['training_manifest']) == 4, 'Fixed model inventory changed')
    timing = inputs.js(PERM / 'phase_timing.json', 'finalized fixed phases')
    orchestration = inputs.js(PERM / 'orchestration_freeze.json', 'fixed stage list')
    expected = {row[0] for row in orchestration['perm_stages'] + orchestration['graph_stages']}
    expected |= {'perm_result_readout', 'graph_result_readout'}
    phases = timing['phases']
    require(len(phases) == len(expected) == 14 and {row['phase'] for row in phases} == expected and
            all(row.get('exit_code') == 0 and row.get('finish_utc') for row in phases),
            'All fourteen fixed phases must finish successfully')
    pr = inputs.js(PERM / 'metric_review/final_reader_audit.json', 'independent PermWorld table audit')
    pa = inputs.js(PERM / 'report_audit.json', 'complete frozen PermWorld reader audit')
    gr = inputs.js(GRAPH / 'report_reader_audit.json', 'independent graph reader audit')
    gc = inputs.js(GRAPH / 'completion.json', 'fixed graph completion')
    require(pr['status'] == pa['status'] == gr['status'] == 'passed', 'Reader audit did not pass')
    require(pr.get('fixture_only', False) is False and pr['per_length_rows'] == 24192 and
            pr['source_learning_rows'] == 16128 and pa['all_relations_and_conditions_retained'] is True,
            'Partial or synthetic PermWorld artifacts cannot be reported')
    require(gr['source_count'] == 4 and gr['problems'] == [] and gc['status'] == 'completed_and_stopped',
            'Partial graph pilot cannot be reported')
    data = dict(inputs=inputs, audit=audit, p=p, g=g, timing=timing, pr=pr, pa=pa, gr=gr, gc=gc)
    for domain, diag in (('permworld', PERM), ('graphworld', GRAPH)):
        status = inputs.js(HERE / domain / 'readout_status.json', 'audited supplementary reader status')
        require(status['status'] == 'completed_readout' and not status.get('fixture_only', False),
                'Missing/pending/synthetic supplementary results: ' + domain)
        require(status['scientific_metrics_recomputed'] == status['training_updates'] == 0,
                'Supplementary stage unexpectedly changed scientific scope')
        for name, digest in status['input_sha256'].items():
            inputs.record(diag / name, 'supplementary source identity', digest)
        data[domain + '_status'] = status
    reader_binding = inputs.js(HERE / 'readout_freeze.json', 'frozen supplementary reader source binding')
    for domain in ('permworld', 'graphworld'):
        require(data[domain + '_status']['reader_sha256'] == reader_binding['source_sha256'] and
                data[domain + '_status']['readout_freeze_sha256'] == sha(HERE / 'readout_freeze.json'),
                'Supplementary output source differs from its frozen binding')
    for name, digest in pr['input_sha256'].items():
        inputs.record(PERM / name, 'independently audited scientific table', digest)
    data['length'] = inputs.csv(PERM / 'per_length_results.csv', 'all original per-length scientific cells')
    data['summary'] = inputs.csv(PERM / 'contrast_summary.csv', 'all original fixed-weight decompositions')
    data['cross'] = inputs.csv(PERM / 'cross_arch_same_task.csv', 'all direct same-task per-length comparisons')
    data['cross_summary'] = inputs.csv(PERM / 'cross_arch_same_task_summary.csv', 'all direct same-task aggregates')
    data['gold'] = inputs.csv(PERM / 'gold_baseline_length.csv', 'all original static answer geometry')
    data['source'] = inputs.csv(PERM / 'source_learning.csv', 'all source per-length learning')
    data['source_summary'] = inputs.csv(PERM / 'source_learning_summary.csv', 'all source fixed-weight learning')
    for key, name, count in (
        ('main', 'main32_final_cka_readout.csv', 32),
        ('encoding', 'encoding16_decomposition.csv', 16),
        ('directions', 'three_metric_directions32.csv', 32),
        ('layers', 'all_layer_directions.csv', 768),
        ('source64', 'source64_learning_and_baseline.csv', 64),
        ('coverage', 'supplementary_component_coverage.csv', 13824),
    ):
        data[key] = inputs.csv(HERE / 'permworld' / name, 'complete supplementary ' + key)
        require(len(data[key]) == count, 'Incomplete supplementary table: ' + name)
    data['curves'] = inputs.csv(HERE / 'permworld/source_learning_curves.csv', 'all archived validation curves')
    require(len(data['curves']) == 1280, 'All sixty-four fixed twenty-point learning curves are required')
    for key, name, count in (
        ('g_main', 'graph_main4_readout.csv', 4),
        ('g_directions', 'graph_three_metric_directions4.csv', 4),
        ('g_source4', 'graph_source4_learning.csv', 4),
        ('g_coverage', 'graph_component_coverage.csv', 2880),
    ):
        data[key] = inputs.csv(HERE / 'graphworld' / name, 'complete graph supplementary ' + key)
        require(len(data[key]) == count, 'Incomplete graph supplementary table: ' + name)
    for key, name, count in (
        ('g_length', 'geometry_per_n.csv', 720),
        ('g_summary', 'geometry_summary.csv', 240),
        ('g_source', 'source_learning.csv', 240),
    ):
        data[key] = inputs.csv(GRAPH / name, 'audited complete graph table')
        require(len(data[key]) == count, 'Incomplete graph table: ' + name)
    for key, count in (('length', 24192), ('summary', 2912), ('cross', 2688),
                       ('cross_summary', 128), ('gold', 32256), ('source', 16128), ('source_summary', 768)):
        require(len(data[key]) == count, 'Incomplete original table: ' + key)
    inputs.record(PERM / 'report.html', 'original PermWorld report')
    inputs.record(GRAPH / 'report.html', 'original graph report')
    inputs.record(PERM / 'figures/four_controlled_comparisons.png', 'original four-part figure',
                  pa['figures_sha256']['figures/four_controlled_comparisons.png'])
    inputs.record(GRAPH / 'answer_control_decomposition.png', 'original graph pilot figure')
    # Every original Perm plot stays linked; no favorable plot is selected.
    for name, digest in pa['figures_sha256'].items():
        inputs.record(PERM / name, 'complete original fixed figure inventory', digest)
    for relative in ('extraction_validation.json', 'metric_review/independent_metric_actual.json',
                     'coverage_status.csv', 'relation_label_alignment.csv'):
        inputs.record(PERM / relative, 'original validation/coverage counterpart')
    for name in ('independent_actual_small_batch.json', 'coverage_status.csv'):
        inputs.record(GRAPH / name, 'original graph validation/coverage counterpart')
    verify_derivative_tables(data)
    return data


def verify_derivative_tables(data):
    """Identity checks against audited cells; never recompute a metric."""
    keys = ('architecture', 'encoding', 'relation', 'control', 'branch', 'layer', 'metric')
    conditions = indexed([row for row in data['summary'] if row['contrast_type'] == 'condition_summary'], keys, 1152)
    data['conditions'] = conditions
    main = indexed(data['main'], ('relation', 'architecture', 'encoding'), 32)
    require(set(main) == {(r, a, e) for r in RELATIONS for a in ARCHS for e in ENCODINGS},
            'Main table does not preserve every one of the thirty-two coupled cells')
    for row in main.values():
        a, e, r = (row[key] for key in ('architecture', 'encoding', 'relation'))
        for control in P_CONTROLS:
            for metric in CKAS:
                for branch, layer, prefix in (('hidden', 'final_norm', f'{control}_{metric}_'),
                                             ('output_prob', 'output', f'output_{control}_{metric}_')):
                    original = conditions[a, e, r, control, branch, layer, metric]
                    for field in FIELDS:
                        require(same(row[prefix + field], original[field]), 'Main supplementary value differs from audit')
        for field in FIELDS:
            require(same(row[field], conditions[a, e, r, 'answer_strata', 'hidden', 'final_norm', 'linear_cka'][field]),
                    'Main unprefixed values are not the fixed primary endpoint')
    for row in data['layers']:
        key = tuple(row[key] for key in ('architecture', 'encoding', 'relation', 'control'))
        original = conditions[*key, 'hidden', row['layer'], row['metric']]
        for field in FIELDS:
            require(same(row[field], original[field]), 'Layer supplementary value differs from audited aggregate')
    indexed(data['layers'], ('architecture', 'encoding', 'relation', 'control', 'metric', 'layer'), 768)
    for row in data['directions']:
        a, e, r = (row[key] for key in ('architecture', 'encoding', 'relation'))
        for control in P_CONTROLS:
            for metric in METRICS:
                original = conditions[a, e, r, control, 'hidden', 'final_norm', metric]
                for field in ('B0', 'Bt', 'dc', 'dw', 'Delta'):
                    require(same(row[f'{control}_{metric}_{field}'], original[field]),
                            'Metric-direction readout value differs from audit')
    enc = indexed(data['encoding'], ('relation', 'architecture'), 16)
    for row in enc.values():
        a, r = row['architecture'], row['relation']
        for control in P_CONTROLS:
            for metric in CKAS:
                prefix = f'{control}_{metric}_'
                for e, label in (('one_line', 'one_line'), ('canonical_cycle', 'cycle')):
                    original = conditions[a, e, r, control, 'hidden', 'final_norm', metric]
                    for field in ('B0', 'Bt', 'dc', 'dw', 'Delta'):
                        require(same(row[prefix + label + '_' + field], original[field]),
                                'Encoding endpoint readout differs from audited aggregate')
                original = next(value for value in data['summary'] if value['contrast_type'] == 'encoding_difference'
                    and value['relation'] == r and value['architecture'] == a and value['control'] == control
                    and value['branch'] == 'hidden' and value['layer'] == 'final_norm' and value['metric'] == metric)
                for field in ('Gamma0', 'Gammat', 'Gammac', 'Gammaw', 'GammaDelta'):
                    require(same(row[prefix + field], original[field]), 'Encoding Gamma differs from audit')
                bt = (direction(row[prefix + 'one_line_Bt']), direction(row[prefix + 'cycle_Bt']))
                dt = (direction(row[prefix + 'one_line_Delta']), direction(row[prefix + 'cycle_Delta']))
                for pair, name in ((bt, 'Bt'), (dt, 'Delta')):
                    actual = 'undefined' if 'undefined' in pair else 'yes' if set(pair) == {'positive', 'negative'} else 'no'
                    require(row[prefix + name + '_flip'] == actual, 'Encoding flip label contradicts audited signs')
                require(yes(row[prefix + 'Delta_flip_without_Bt_flip']) ==
                        (row[prefix + 'Delta_flip'] == 'yes' and row[prefix + 'Bt_flip'] == 'no'),
                        'Delta-only flip label contradicts its two component labels')
    source = indexed(data['source64'], ('architecture', 'encoding', 'task'), 64)
    learning = indexed(data['source_summary'], ('architecture', 'encoding', 'task', 'condition', 'split', 'action'), 768)
    for key, row in source.items():
        original = learning[*key, 'trained', 'test', 'identity']
        for field in ('answer_accuracy', 'answer_ce', 'minimum_length_accuracy', 'length_majority_accuracy', 'accuracy_minus_length_mode'):
            require(same(row[field], original[field]), 'Source learning readout differs from audit')
    jobs = {job['run_id']: job for job in data['p']['checkpoint_inventory']}
    curve_groups = defaultdict(list)
    for row in data['curves']:
        curve_groups[row['run_id']].append(row)
    require(set(curve_groups) == set(jobs), 'Learning curves must preserve all sixty-four jobs')
    for row in data['source64']:
        job = jobs[row['run_id']]
        require(Path(row['curve_completion_path']).resolve() == Path(job['completed_path']).resolve(),
                'Curve completion identity differs from frozen inventory')
        completion = data['inputs'].js(row['curve_completion_path'], 'audited source curve sidecar', row['curve_completion_sha256'])
        require(completion['step'] == 20000 and len(completion['curve']) == 20, 'Curve sidecar is not final twenty thousand updates')
        original_curve = {int(value['step']): value for value in completion['curve']}
        shown_curve = {int(value['step']): value for value in curve_groups[row['run_id']]}
        require(set(original_curve) == set(shown_curve) == set(range(1000, 20001, 1000)),
                'Fixed source validation curve steps are missing or duplicated')
        for step, value in original_curve.items():
            for field, original_value in value.items():
                require(str(original_value) == shown_curve[step].get(field), 'Saved source curve value differs from frozen readout')
    graphs = indexed(data['g_summary'], ('relation', 'control', 'branch', 'metric'), 240)
    for row in data['g_main']:
        for control in G_CONTROLS:
            for metric in CKAS:
                for branch, prefix in (('prefix_final', f'{control}_{metric}_'),
                                       ('output_prob', f'output_{control}_{metric}_')):
                    original = graphs[row['relation'], control, branch, metric]
                    for field in FIELDS:
                        require(same(row[prefix + field], original[field]), 'Graph readout differs from audited geometry')
        for field in FIELDS:
            require(same(row[field], graphs[row['relation'], 'answer_strata', 'prefix_final', 'linear_cka'][field]),
                    'Graph primary endpoint differs from audit')
    for row in data['g_directions']:
        for control in G_CONTROLS:
            for metric in METRICS:
                original = graphs[row['relation'], control, 'prefix_final', metric]
                for field in ('B0', 'Bt', 'dc', 'dw', 'Delta'):
                    require(same(row[f'{control}_{metric}_{field}'], original[field]), 'Graph metric-direction values differ from audit')
    indexed(data['g_main'], ('relation',), 4)
    indexed(data['g_directions'], ('relation',), 4)
    for row in data['g_source4']:
        records = [value for value in data['g_source'] if value['task'] == row['task'] and
                   value['condition'] == 'trained' and value['split'] == 'analysis' and value['action'] == 'identity']
        require(sorted(int(value['n']) for value in records) == [6, 7, 8], 'Graph source fixed sizes are incomplete')
        for field in ('answer_accuracy', 'answer_ce', 'mode_accuracy', 'edge_mean', 'density_mean'):
            # Only audit arithmetic of already-stored scalar summaries, never a metric from arrays.
            expected = math.fsum(number(value[field]) for value in records) / 3
            require(same(row[field], expected), 'Graph source summary differs from stored per-size scalar values')
    for coverage_key, length_key, aggregate_index, components, length_name in (
            ('coverage', 'length', conditions, data['length'], 'length'),
            ('g_coverage', 'n', graphs, data['g_length'], 'n')):
        group_keys = keys if length_key == 'length' else ('relation', 'control', 'branch', 'metric')
        groups = defaultdict(list)
        for value in components:
            groups[tuple(value[key] for key in group_keys)].append(value)
        indexed(data[coverage_key], group_keys + ('component',), len(aggregate_index) * len(FIELDS))
        for row in data[coverage_key]:
            key = tuple(row[name] for name in group_keys)
            field = row['component']
            require(field in FIELDS and key in aggregate_index, 'Coverage component is outside fixed scientific grid')
            original = aggregate_index[key][field]
            missing = sorted(int(value[length_name]) for value in groups[key] if not math.isfinite(number(value[field])))
            require(same(row['aggregate'], original) and json.loads(row['undefined_lengths']) == missing and
                    int(row['undefined_length_count']) == len(missing) and
                    yes(row['defined_aggregate']) == math.isfinite(number(original)) and
                    yes(row['missing_weights_reallocated']) is False,
                    'Coverage readout is inconsistent with audited components')


class Document:
    def __init__(self):
        self.html = ['<!doctype html><html lang="en"><head><meta charset="utf-8">',
                     '<title>PermWorld and GraphWorld: fixed-scope combined report</title>',
                     '<style>body{max-width:1650px;margin:30px auto;padding:0 20px;font:15px/1.65 sans-serif;color:#17212b}'
                     'table{border-collapse:collapse;white-space:nowrap}th,td{border:1px solid #d8dce2;padding:4px 7px}'
                     'th{background:#edf1f7;position:sticky;top:0}.scroll{overflow:auto;max-height:850px;margin:12px 0}'
                     'img{max-width:100%}details{margin:14px 0}.notice{padding:14px;background:#fff4d5}'
                     'code{word-break:break-all}a{color:#1756a9}</style></head><body>']
        self.text = []

    def title(self, text, level=2):
        self.html.append(f'<h{level}>{html.escape(text)}</h{level}>')
        self.text.extend(['', text, ''])

    def paragraph(self, text, notice=False):
        self.html.append('<p' + (' class="notice"' if notice else '') + '>' + html.escape(text) + '</p>')
        self.text.extend([text, ''])

    def table(self, rows, columns, caption=None, collapsed=False):
        require(rows, 'Reader refuses an empty displayed table')
        if collapsed:
            self.html.append('<details><summary>' + html.escape(caption or 'Complete table') + f' ({len(rows)} rows)</summary>')
        elif caption:
            self.html.append('<h4>' + html.escape(caption) + '</h4>')
        if caption:
            self.text.extend([caption + f' ({len(rows)} rows)', ''])
        self.html.append('<div class="scroll"><table><thead><tr>' +
                         ''.join('<th>' + html.escape(name) + '</th>' for name in columns) + '</tr></thead><tbody>')
        self.text.append('\t'.join(columns))
        for row in rows:
            self.html.append('<tr>' + ''.join('<td title="' + html.escape(str(row.get(name, '')), quote=True) + '">' +
                html.escape(display(row.get(name, ''))) + '</td>' for name in columns) + '</tr>')
            self.text.append('\t'.join(str(row.get(name, '')) if row.get(name, '') not in ('', None) else 'undefined' for name in columns))
        self.html.append('</tbody></table></div>')
        if collapsed:
            self.html.append('</details>')
        self.text.append('')

    def link(self, path, label=None):
        path = Path(path).resolve()
        require(path.is_file(), 'Broken local report link: ' + str(path))
        relative = os.path.relpath(path, HERE)
        self.html.append('<p><a href="' + html.escape(relative, quote=True) + '">' + html.escape(label or path.name) + '</a></p>')
        self.text.append((label or path.name) + ': ' + str(path))

    def image(self, path, caption):
        path = Path(path).resolve()
        require(path.is_file(), 'Missing original figure')
        relative = os.path.relpath(path, HERE)
        self.html.append('<figure><img src="' + html.escape(relative, quote=True) + '" alt="' +
                         html.escape(caption, quote=True) + '"><figcaption>' + html.escape(caption) + '</figcaption></figure>')
        self.text.append(caption + ': ' + str(path))


def report(data):
    doc = Document()
    doc.title('Representational similarity under mathematical correspondences: combined report on eight PermWorld and four GraphWorld relations', 1)
    doc.paragraph('This report reads only existing completed tables that passed audit. PermWorld includes 64 fixed 20k single-task endpoints (16 earlier models reused and 48 missing models completed), yielding 32 cells from eight relations × two encodings × two configurations. GraphWorld adds a reference-Transformer pilot with four tasks and four relations. Each system has only one data world, with initialization shared within the specified configuration. The 32 cells, 64 models, eight relations, and four graph relations are not independent repetitions.', True)
    doc.paragraph('Read the six original similarity scores S → trained correct−wrong preference Bt → initial preference B0 and correct/wrong changes dc,dw → Δ. Here “original S” means the metric before taking differences, not the raw_within_length control version. Answer-stratified S is still an original score. All values are exploratory descriptions, without confirmatory p-values, equivalence claims, or mechanism conclusions.')
    doc.paragraph('B0=S_initial,correct−S_initial,wrong; Bt=S_trained,correct−S_trained,wrong; dc=S_trained,correct−S_initial,correct; dw=S_trained,wrong−S_initial,wrong; di=training change for identity pairings; Δ=Bt−B0=dc−dw. Positive Δ does not require increased correct S, and negative Δ does not mean no learning. Identity pairings are retained separately and are not an upper bound that must be exceeded.')
    doc.paragraph('The primary readout uses final-normalization states at input end before the task token. Scores are computed separately by length, then aggregated with fixed weights: 1/21 for PermWorld lengths 10…30 and 1/3 for GraphWorld sizes 6,7,8. If a component is undefined at any length, its aggregate remains undefined; lengths are not pooled, zeros are not imputed, and missing weights are not redistributed.')
    doc.image(PERM / 'figures/four_controlled_comparisons.png', 'Original four-part controlled-comparison figure: fixed metrics, model configurations, encodings, and all layers; Bt and Δ retained separately')
    main = indexed(data['main'], ('relation', 'architecture', 'encoding'), 32)
    doc.title('1. All 32 cells across eight relations: original scores and training increments')
    for relation in RELATIONS:
        doc.title(DISPLAY[relation], 3)
        rows = [main[relation, a, e] for a in ARCHS for e in ENCODINGS]
        doc.table(rows, ['architecture', 'encoding'] + list(FIELDS), 'Primary answer stratification, final normalization, ordinary CKA: all four cells')
        for row in rows:
            doc.paragraph(f"{row['architecture']} / {row['encoding']}:Bt={display(row['Bt'])} ({direction(row['Bt'])}),"
                f"Δ={display(row['Delta'])} ({direction(row['Delta'])}); correct S change dc={display(row['dc'])}, wrong S change dw={display(row['dw'])}."
                'These changes jointly determine Δ; it cannot simply be called convergence of correct representations.')
        control_rows, output_rows = [], []
        for row in rows:
            for control in P_CONTROLS:
                for metric in CKAS:
                    identity = dict(architecture=row['architecture'], encoding=row['encoding'], control=control, metric=metric)
                    control_rows.append(field_row(row, f'{control}_{metric}_', **identity))
                    output_rows.append(field_row(row, f'output_{control}_{metric}_', **identity))
        columns = ['architecture', 'encoding', 'control', 'metric'] + list(FIELDS)
        doc.table(control_rows, columns, 'Hidden representations: raw/answer-stratified and ordinary/U-centered CKA', True)
        doc.table(output_rows, columns, 'Output probabilities: parallel controls retaining original scores and decompositions', True)
    doc.paragraph('Joint-answer stratification retains the full prespecified relation key and minimum group size 3; samples, lengths, and negative controls are not changed in response to results. components(Cπ)=global_descents(π)+1; subtracting 1 from the right-side stratum key changes only integer coordinates, while actual labels and answer one-hot/scalar encodings retain their original numbers. Wrong transformations can accidentally preserve the answer formula on some inputs; those samples were not removed.')
    doc.link(PERM / 'relation_label_alignment.csv', 'True-answer formula agreement for all correct, wrong, and identity pairings')

    doc.title('2. Does encoding change initial or trained preference?')
    doc.paragraph('Γ is fixed as canonical cycle notation minus one-line notation: Γ0 is the B0 difference, Γt the Bt difference, Γc and Γw the differences in correct/wrong changes, and ΓΔ the Δ difference. All 16 groups from eight relations × two configurations, both controls, and both CKA versions are retained below. Δ reversal and Bt reversal are distinct criteria; if Δ reverses but Bt does not, this is not a reversal of learned relation preference.')
    encoding_rows = []
    for relation in RELATIONS:
        for architecture in ARCHS:
            row = next(value for value in data['encoding'] if value['relation'] == relation and value['architecture'] == architecture)
            for control in P_CONTROLS:
                for metric in CKAS:
                    prefix = f'{control}_{metric}_'
                    record = dict(relation=relation, architecture=architecture, control=control, metric=metric)
                    names = [f'{encoding}_{field}' for encoding in ('one_line', 'cycle') for field in ('B0', 'Bt', 'dc', 'dw', 'Delta')]
                    names += ['Gamma0', 'Gammat', 'Gammac', 'Gammaw', 'GammaDelta', 'Bt_flip', 'Delta_flip', 'Delta_flip_without_Bt_flip']
                    record.update({name: row[prefix + name] for name in names})
                    encoding_rows.append(record)
    doc.table(encoding_rows, ['relation', 'architecture', 'control', 'metric'] + names, '16 encoding decompositions and all fixed control/sensitivity versions')
    original_encoding = [row for row in data['summary'] if row['contrast_type'] == 'encoding_difference'
                         and row['branch'] == 'hidden' and row['layer'] == 'final_norm']
    require(len(original_encoding) == 128, 'Full final encoding contrast grid changed')
    doc.table(original_encoding, ['relation', 'architecture', 'control', 'metric', 'Gamma0', 'Gammat', 'Gammac', 'Gammaw', 'GammaDelta'],
              'Original three primary metrics and U-centered CKA: all final-layer encoding differences; magnitudes are not compared across metrics', True)
    interaction = [row for row in data['summary'] if row['contrast_type'] == 'interaction'
                   and row['branch'] == 'hidden' and row['layer'] == 'final_norm']
    require(len(interaction) == 64, 'Full final configuration-by-encoding contrast grid changed')
    doc.table(interaction, ['relation', 'control', 'metric', 'B0', 'Bt', 'dc', 'dw', 'Delta', 'I0', 'It', 'I'],
              'All final-layer configuration × encoding interactions; exploratory descriptions, not pure architecture causes', True)
    doc.paragraph('The two encodings copy genuine step-0 parameters within each architecture, pairing underlying permutations, orbit splits, and update budgets; initialization is not claimed identical across architectures. Encoding also changes sequence organization, initial input geometry, and learning difficulty. Cells with different source accuracies cannot identify a pure encoding mechanism at equal task mastery.')

    doc.title('3. Metrics, configurations, layers, and all lengths')
    doc.paragraph('The three primary metrics are ordinary linear CKA, k=10 neighborhood overlap, and held-out Procrustes with PCA rank≤8 fitted on validation data. Procrustes similarity is negative normalized prediction error, so higher is more similar. U-centered CKA is the fixed sensitivity version. Compare directions and patterns across metrics, without subtracting their magnitudes or treating them as independent repetitions.')
    metric_rows = []
    for row in data['directions']:
        for control in P_CONTROLS:
            for metric in METRICS:
                record = {key: row[key] for key in ('relation', 'architecture', 'encoding')}
                record.update(control=control, metric=metric)
                for field in ('B0', 'Bt', 'dc', 'dw', 'Delta'):
                    record[field] = row[f'{control}_{metric}_{field}']
                    record[field + '_direction'] = direction(record[field])
                    record[field + '_three_metric_agreement'] = row[f'{control}_{field}_agreement']
                metric_rows.append(record)
    doc.table(metric_rows, ['relation', 'architecture', 'encoding', 'control', 'metric'] +
              [name for field in ('B0', 'Bt', 'dc', 'dw', 'Delta') for name in (field, field + '_direction', field + '_three_metric_agreement')],
              '32 cells × two controls × three metrics: all directions and cross-metric agreement', True)
    configuration = [row for row in data['summary'] if row['contrast_type'] == 'configuration_difference'
                     and row['branch'] == 'hidden' and row['layer'] == 'final_norm']
    require(len(configuration) == 128, 'Complete configuration contrast grid changed')
    doc.table(configuration, ['relation', 'encoding', 'control', 'metric', 'B0', 'Bt', 'dc', 'dw', 'di', 'Delta'],
              'All final-layer configuration differences (GRU minus Transformer)', True)
    doc.paragraph('Configuration comparisons retain parameter counts and source-task performance. Transformer and GRU capacities, computation, and mastery are unmatched; these results describe the existing configurations and do not identify a pure architecture effect.')
    doc.table(data['layers'], ['relation', 'architecture', 'encoding', 'control', 'metric', 'layer'] + list(FIELDS) +
              ['Bt_direction', 'Delta_direction', 'Bt_flip_vs_layer4', 'Delta_flip_vs_layer4', 'Bt_encoding_flip', 'Delta_encoding_flip'],
              'All four blocks/GRU layers, with final normalization and embedding listed separately: 768 rows', True)
    doc.paragraph('Layer order is fixed; layer 4 is a direction reference, not a selected best layer. Final normalization simply lists the normalized state separately. The pre-task input-end token embedding is structurally constant within each length, so undefined similarity is not evidence of “no learning.” Layerwise Δ changes still require B0 and Bt: initial preference, trained pairing preference, or both may change.')
    # Every scientific per-length cell remains visible, not just the main endpoint.
    for relation in RELATIONS:
        rows = [row for row in data['length'] if row['relation'] == relation]
        require(len(rows) == 3024, 'A relation lost scientific per-length cells')
        doc.table(rows, ['architecture', 'encoding', 'control', 'branch', 'layer', 'metric', 'length', 'fit_samples', 'test_samples'] +
                  list(FIELDS) + ['error_initial_correct', 'error_initial_wrong', 'error_initial_identity',
                                  'error_trained_correct', 'error_trained_wrong', 'error_trained_identity', 'nan_reason'],
                  DISPLAY[relation] + ': all 3024 per-length cells, including outputs, four layers, and both controls', True)
    doc.link(PERM / 'per_length_results.csv', 'Complete original per-length CSV and activation/index/preprocessing provenance hashes')

    doc.title('4. Answer geometry, outputs, and learning of all source models')
    doc.paragraph('Ground-truth stratification removes within-group answer means, but not all confidence, prediction-error, or nonlinear answer information. Output probabilities are conditional on numeric tokens 0…30; source accuracy and CE use the 188-token vocabulary. Hidden Δ and output Δ cannot be subtracted to estimate an additional mechanism. True-answer geometry is a static reference without fabricated initialization, training, or Δ.')
    # The frozen report's summary is checked against the original complete length table.
    gold_groups = defaultdict(list)
    for row in data['gold']:
        if row['architecture'] == 'transformer' and row['encoding'] == 'one_line':
            gold_groups[tuple(row[key] for key in ('relation', 'control', 'baseline', 'role', 'metric'))].append(row)
    static = []
    for key, rows in gold_groups.items():
        require(sorted(int(row['length']) for row in rows) == list(range(10, 31)), 'Static answer reference lost a fixed length')
        # Display all original per-length values: no new static metric or fitted summary.
        static.extend({name: value for name, value in row.items() if name not in ('architecture', 'encoding')} for row in rows)
    require(len(static) == 8064, 'Static baseline scope is incomplete')
    doc.table(static, ['relation', 'control', 'baseline', 'role', 'metric', 'length', 'value', 'error', 'fit_samples', 'test_samples'],
              'Complete static answer geometry: 8064 unique relation/control/role/metric/length cells; duplicate copies across four configurations add no evidence', True)
    doc.link(PERM / 'gold_baseline_length.csv', 'Original 32256-row gold-label baseline (including repeated indices across four configurations)')
    doc.table(data['source64'], ['task', 'architecture', 'encoding', 'parameters', 'answer_accuracy', 'minimum_length_accuracy',
              'answer_ce', 'length_majority_accuracy', 'accuracy_minus_length_mode', 'gain_over_train_defined_length_mode_pp',
              'baseline_comparison', 'length_gain_directions'], '64 source models: fixed 20k, identity inputs, all 21 lengths equally weighted')
    doc.paragraph('Length-specific modes use original training identity-input labels, break ties to the smallest number, and are then fixed for analysis. Exceeding the mode is descriptive, not a prespecified “sufficient learning” threshold; falling below it does not prove no input use. All weakly learned models are retained without accuracy-based deletion of geometry cells. Missing task-specific step-0 answer CE for the original 16 models remains missing; shared hidden states or another task’s CE cannot fill it.')
    source_details = [row for row in data['source'] if row['condition'] == 'trained' and row['action'] == 'identity']
    require(len(source_details) == 2688, 'Source identity learning details are incomplete')
    doc.table(source_details, ['architecture', 'encoding', 'task', 'split', 'length', 'answer_accuracy', 'answer_ce',
              'length_majority_accuracy', 'accuracy_minus_length_mode'], '64 sources: per-length identity-task performance on analysis and validation', True)
    doc.table(data['curves'], ['run_id', 'architecture', 'encoding', 'task', 'step', 'train_loss', 'answer_accuracy',
              'numeric_accuracy', 'eos_accuracy', 'samples'], '20 originally saved validation records per model; not used for checkpoint selection', True)
    doc.paragraph('The curves’ train_loss is the originally recorded training loss, not full-training-set answer CE; the curves did not save answer CE at every record. The final answer CE above comes from source-learning evaluation at fixed endpoints; CE is not invented for the curves. numeric_accuracy is an originally saved auxiliary score restricted to numeric readout, not a replacement for full 188-token answer accuracy. Curve EOS uses teacher-forced answer inputs, unlike prompt-only representation evaluation.')
    doc.link(PERM / 'source_learning.csv', 'Source-learning tables for all input transformations and initial/trained states; missing CE retained')

    doc.title('5. Direct same-task cross-configuration similarity')
    doc.paragraph('The 32 pairings directly compare Transformer and GRU on the same task and encoding, retaining both controls and CKA versions. This is a same-task identity--identity comparison without a mathematically correct/wrong relation; its joint-answer key [n,y_identity,y_C,y_I] differs from relational task-pair stratification. Shared inputs and capacity/mastery differences still constrain interpretation.')
    doc.table(data['cross_summary'], ['task', 'encoding', 'control', 'metric', 'length_count', 'initial', 'trained', 'change'],
              'All 128 fixed aggregates for 32 same-task pairings')
    doc.table(data['cross'], ['task', 'encoding', 'control', 'metric', 'length', 'fit_samples', 'test_samples',
              'initial_similarity', 'trained_similarity', 'similarity_change', 'nan_reason'], 'All 2688 same-task per-length cells', True)

    doc.title('6. GraphWorld: four-relation pilot using the same definitions')
    doc.paragraph('The graph pilot uses only the original 256-dimensional, four-layer, eight-head Transformer reference configuration and one binary-edge encoding, copying its genuine initialization and saved starting RNG. Four tasks share one graph-data world; the four relations cover only transpose and graph complement. This is not a cross-domain replication of the full eight-relation × encoding × architecture grid. Changing systems also changes sizes, input encoding, tasks, and distributions; PermWorld and GraphWorld CKA scores are not directly subtracted.')
    doc.image(GRAPH / 'answer_control_decomposition.png', 'Original GraphWorld three-metric answer-control decomposition, fixed equal weights for n=6,7,8')
    g_main = indexed(data['g_main'], ('relation',), 4)
    for relation in G_RELATIONS:
        doc.title(DISPLAY[relation], 3)
        row = g_main[relation,]
        doc.table([row], ['relation', 'left', 'right', 'correct_transform', 'wrong_transform', 'mapped_right'] + list(FIELDS),
                  'Primary answer control, final normalization, ordinary CKA')
        doc.paragraph(f"Bt={display(row['Bt'])} ({direction(row['Bt'])}),Δ={display(row['Delta'])} ({direction(row['Delta'])}),"
                      f"dc={display(row['dc'])}, dw={display(row['dw'])}; positive Δ and increased correct S remain separate judgments.")
        hidden, output = [], []
        for control in G_CONTROLS:
            for metric in CKAS:
                hidden.append(field_row(row, f'{control}_{metric}_', control=control, metric=metric))
                output.append(field_row(row, f'output_{control}_{metric}_', control=control, metric=metric))
        doc.table(hidden, ['control', 'metric'] + list(FIELDS), 'raw, answer, answer+density controls, and U-centered sensitivity')
        doc.table(output, ['control', 'metric'] + list(FIELDS), 'Parallel graph output-probability controls', True)
    graph_directions = []
    for row in data['g_directions']:
        for control in G_CONTROLS:
            for metric in METRICS:
                record = dict(relation=row['relation'], control=control, metric=metric)
                for field in ('B0', 'Bt', 'dc', 'dw', 'Delta'):
                    record[field] = row[f'{control}_{metric}_{field}']
                    record[field + '_direction'] = direction(record[field])
                    record[field + '_three_metric_agreement'] = row[f'{control}_{field}_agreement']
                graph_directions.append(record)
    doc.table(graph_directions, ['relation', 'control', 'metric'] +
              [name for field in ('B0', 'Bt', 'dc', 'dw', 'Delta') for name in (field, field + '_direction', field + '_three_metric_agreement')],
              'All four graph relations, three controls, and three primary-metric directions', True)
    doc.table(data['g_summary'], ['relation', 'control', 'branch', 'metric', 'static_reference'] + list(FIELDS),
              'All 240 graph aggregates; static input/answer branches retain undefined training increments', True)
    doc.table(data['g_length'], ['relation', 'n', 'control', 'branch', 'metric', 'fit_count', 'analysis_count'] + list(FIELDS) +
              ['initial_correct_error', 'initial_wrong_error', 'initial_identity_error',
               'trained_correct_error', 'trained_wrong_error', 'trained_identity_error'], 'All 720 per-size graph cells', True)
    doc.paragraph('Density control adds the true anchor edge count m to the joint-answer key rather than selecting a density range after seeing results. The group threshold remains 3; reduced samples or insufficient variance/rank after control leave undefined values. Thresholds cannot be relaxed, n weights redistributed, or NaN interpreted as “no relation.” The positional binary one-hot input kernel is insensitive to the prespecified slot/category isometries; its high similarity is an input reference, not training evidence.')
    doc.paragraph('Full <R,C,W> orbit exclusion only ensures that the specified transformations do not cross splits; isomorphic graphs under arbitrary vertex relabeling may still occur in different splits. The original wrong transformation W preserves edge count and may accidentally preserve task answers; the negative-control definition is unchanged. Graph density, maximum/minimum degree, and accidental answer agreement under wrong pairings must be read alongside source performance and static branches.')
    doc.table(data['g_source4'], ['task', 'architecture', 'encoding', 'answer_accuracy', 'minimum_size_accuracy', 'answer_ce',
              'mode_accuracy', 'gain_over_train_defined_mode_pp', 'edge_mean', 'density_mean', 'EOS_evaluated'], 'All four graph sources: learning and training-defined modes')
    doc.table(data['g_source'], ['task', 'condition', 'split', 'n', 'action', 'samples', 'answer_accuracy', 'answer_ce',
              'training_mode', 'mode_accuracy', 'edge_mean', 'density_mean', 'EOS_evaluated'], 'All 240 graph source-learning cells', True)
    doc.paragraph('Graph source answer accuracy/CE also use the full 188-token vocabulary. Prompt-only representation extraction did not evaluate EOS; original teacher-forced source-validation EOS is a different measure and cannot be filled in here as prompt-only capability. Weak learning is retained without automatic retraining, additional models, or protocol changes.')

    doc.title('7. Undefined results, reuse, and dependent evidence')
    for key, label in (('coverage', 'PermWorld missing lengths and aggregate identity per component'), ('g_coverage', 'GraphWorld missing sizes and aggregate identity per component')):
        undefined = [row for row in data[key] if not yes(row['defined_aggregate'])]
        doc.paragraph(label + f':{len(undefined)} undefined-component records. This counts records from the same dependent cells, not independent effects.')
        if undefined:
            columns = [name for name in ('relation', 'architecture', 'encoding', 'control', 'branch', 'layer', 'metric', 'component',
                       'undefined_lengths', 'undefined_length_count', 'fixed_length_count', 'aggregate', 'defined_aggregate',
                       'missing_weights_reallocated', 'static_reference') if any(name in row for row in undefined)]
            doc.table(undefined, columns, label + ': all undefined records', True)
    doc.paragraph('Initialization identity is verified from actual saved step-0 parameters and original RNG states, not inferred from equal seeds. The 16 earlier endpoints and 6048 scientific rows for the two original relations are reused value for value. Earlier stratum-key integer dtypes may differ from current safe int64 coordinates; value, grouping, and centering agreement are audited separately. Multiple metric readouts share preprocessed inputs and anchors and do not create additional samples.')
    doc.table([{'field': key, 'value': value} for key, value in data['pr'].items()
               if key in ('original_scientific_rows_reused', 'exact_binary64_comparisons', 'primary_real_independent_fixture_checks',
                          'actual_independent_metric_max_error', 'full21_groups', 'no_NaN_reweight')],
              ['field', 'value'], 'Actual evidence from independent review of the original scientific tables')
    doc.paragraph('The 32 current cells depend on one world, shared anchors, genuine initial weights within architecture, and two transformation types; the eight instances are not eight independent algebraic systems. The four graph relations share four tasks and two transformations. There is no multi-initialization/multi-world confirmation, statistical significance claim, equivalence test, or mechanism identification. Protection of F17, earlier training/reports, and published artifacts is recorded by the final provenance audit; this combined writing does not alter them.')
    doc.table(data['timing']['phases'], ['phase', 'start_utc', 'finish_utc', 'exit_code', 'wall_seconds',
              'user_cpu_seconds', 'system_cpu_seconds', 'max_rss_kib'], 'Actual records for 14 fixed phases; overlapping wait times are not added', True)
    doc.table([dict(field=key, value=json.dumps(data['audit'][key], ensure_ascii=False, sort_keys=True))
               for key in ('formal_sources', 'source_versions', 'actual_saved_initializations', 'old_artifact_protection',
                           'tracked_git', 'timing', 'physical_storage')],
              ['field', 'value'], 'Complete costs, genuine initialization, and protection records from the final provenance audit', True)
    doc.paragraph('Training-process time, phase wall time, CPU time, and GPU active time are different measures. Without independent GPU-kernel timing, GPU active time remains unknown; a CPU-only read audit does not imply zero training GPU time, and costs of the 16 historical models are not counted as newly completed training. Physical storage is deduplicated by real path; bytes for shared aliases and overlapping groups cannot simply be added.')

    doc.title('8. Answers to the original research questions and stopping boundary')
    positive = [row for row in data['main'] if math.isfinite(number(row['Delta'])) and number(row['Delta']) > 0]
    doc.paragraph(f'1. Which relation/configuration training increments exceed their genuine initialization reference? Under primary answer-stratified final-layer ordinary CKA, there are {len(positive)} cells with positive Δ (a descriptive count among 32 dependent cells). They are listed below; all negative and undefined values receive equal presentation in Section 1. We do not call this “stable” or “significant” excess, and positive Δ does not imply increased correct S.')
    if positive:
        doc.table(positive, ['relation', 'architecture', 'encoding', 'B0', 'Bt', 'dc', 'dw', 'Delta'], 'Cells with positive primary Δ; not independent repetitions')
    doc.paragraph('2. Do the observations exceed input structure and simple answer/output explanations? Read the original scores under matched controls, static answers, outputs, and graph input branches in Sections 1,4,6. Genuine initialization is an input/model-structure reference, not an exact decomposition of similarity sources. Answer controls retain probability and prediction-error information; these data cannot identify an additional encoding mechanism through CKA alone.')
    doc.paragraph('3. Which phase is changed by encoding? Section 2 gives Γ0,Γt,Γc,Γw,ΓΔ and both reversal criteria for every group. Only a Bt reversal means trained preference reverses under the current measurement. A Δ reversal without Bt reversal is limited to a reversal of the increment relative to initialization; further interpretation requires Γ0 and source learning.')
    doc.paragraph('4. Does the observation depend on metric, configuration, layer, or length? Section 3 retains every prespecified cell. The three metrics may differ in direction, and layerwise B0 and Bt may change separately. No claim selects a best layer, metric, or length. Configuration capacity and source-task mastery differ, so configuration differences cannot be equated with pure architecture causes.')
    doc.paragraph('5. What does same-task cross-configuration similarity provide? Section 5 gives direct pairings at genuine initialization and after training, together with changes. It describes the geometry of these two configurations; its stratum key differs from cross-task relations. High CKA alone cannot prove a shared algorithm.')
    doc.paragraph('6. Does GraphWorld support cross-system generalization? Section 6 gives actual readouts for all four relations and three controls. It is a reference-configuration pilot in which system changes also change inputs, sizes, and tasks. At most it provides conditional cross-system observations, not a universal mathematical mechanism or cross-system architecture × encoding conclusion.')
    doc.paragraph('7. What can a subsequent encoding comparison resolve? This round separates initial and trained preference using genuine shared step-0 weights. It remains unknown whether directions and magnitudes reproduce at comparable task mastery with independent initializations and data worlds, and what information extends beyond answers/confidence. These open questions are recorded without automatically starting new experiments.')
    doc.paragraph('8. Is this round complete? The prerequisite audit confirms that all fixed 64 PermWorld endpoints, four GraphWorld endpoints, and 14 phases finished. This report adds only read-only synthesis. Work stops after the fixed scope, without additional seeds, tasks, training budgets, larger models, GroupWorld, cycle-encoding extensions, null-space, finite-field, or GELU composition experiments, and without automatic upload or publication.')

    doc.title('Original reports, all figures/tables, and input identity')
    doc.link(PERM / 'report.html', 'Original frozen complete PermWorld report')
    doc.link(GRAPH / 'report.html', 'Original GraphWorld pilot report')
    for name in data['pa']['figures_sha256']:
        doc.link(PERM / name, 'Original prespecified figure: ' + name)
    links = [PERM / name for name in ('per_length_results.csv', 'contrast_summary.csv', 'cross_arch_same_task.csv',
             'cross_arch_same_task_summary.csv', 'gold_baseline_length.csv', 'source_learning.csv', 'source_learning_summary.csv',
             'relation_label_alignment.csv', 'coverage_status.csv', 'protocol.json', 'training_freeze.json', 'analysis_freeze.json',
             'extraction_validation.json', 'metric_review/independent_metric_actual.json', 'metric_review/final_reader_audit.json')]
    links += [GRAPH / name for name in ('geometry_per_n.csv', 'geometry_summary.csv', 'source_learning.csv', 'coverage_status.csv',
             'protocol.json', 'freeze_manifest.json', 'verification_freeze.json', 'independent_actual_small_batch.json',
             'report_reader_audit.json', 'completion.json')]
    links += [HERE / name for name in ('final_audit.json', 'physical_file_manifest.json', 'combined_reader_freeze.json')]
    links += [Path(item['path']) for item in data['inputs'].records.values()
              if '/final_review/permworld/' in item['path'] or '/final_review/graphworld/' in item['path']]
    seen = set()
    for path in links:
        if str(path.resolve()) not in seen:
            data['inputs'].record(path, 'linked original artifact')
            doc.link(path)
            seen.add(str(path.resolve()))
    doc.paragraph('combined_input_hashes.json lists each small file actually read or referenced in this synthesis, its SHA, original audit binding, and report SHA. Large weights, activations, and data arrays were not reread. HTML display values retain original-value tooltips, text tables retain input numeric strings, and CSV contains complete values and provenance metadata. Root finalizes the archive and root completion after review of the combined report to avoid cyclic hash dependencies.')
    doc.html.append('</body></html>')
    return '\n'.join(doc.html) + '\n', '\n'.join(doc.text) + '\n'


def atomic_text(name, text):
    path = HERE / name
    require(path.parent.resolve() == HERE, 'Output must stay inside current final_review')
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(text)
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true', help='Generate only after every required audited input is complete')
    args = parser.parse_args()
    require(args.run, 'Explicit --run is required; no automatic or partial report generation')
    data = load_completed()
    markup, prose = report(data)
    data['inputs'].stable()
    atomic_text('combined_report.html', markup)
    atomic_text('combined_report.txt', prose)
    result = dict(status='completed_reader_generation_pending_root_read_review',
        generated_utc=datetime.now(timezone.utc).isoformat(), source_path=str(Path(__file__).resolve()),
        source_sha256=sha(__file__), input_hashes=sorted(data['inputs'].records.values(), key=lambda item: item['path']),
        outputs={name: dict(path=str(HERE / name), sha256=sha(HERE / name))
                 for name in ('combined_report.html', 'combined_report.txt')},
        original_scientific_artifacts_unchanged=True, all32_coupled_conditions_displayed=True,
        all16_encoding_pairs_displayed=True, all4_graph_relations_displayed=True,
        scientific_metrics_recomputed=0, statistical_tests=0, models_constructed=0,
        activations_loaded=0, raw_data_loaded=0, training_updates=0, GPU_started=False,
        root_must_review_actual_values_and_interpretation=True, no_automatic_followup=True)
    atomic_text('combined_input_hashes.json', json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(status=result['status'], output=str(HERE / 'combined_report.html'),
                          input_hashes=str(HERE / 'combined_input_hashes.json')), ensure_ascii=False))


if __name__ == '__main__':
    main()
