"""Chinese combined reader of completed, audited CSV/JSON artifacts only.

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
    'descent_inverse': '下降数—逆下降数（取逆）',
    'peak_complement': '峰—谷（补排列）',
    'fixed_complement': '不动点—反不动点（补排列）',
    'exceedance_inverse': '超越数—亏缺数（取逆）',
    'subsequence_complement': 'LIS—LDS（补排列）',
    'run_complement': '上升连续段—下降连续段（补排列）',
    'record_complement': '左至右极大—极小记录（补排列）',
    'decomposition_complement': '全局下降数—组件数（补排列，右答案+1）',
    'reverse_max': '最大出度—最大入度（转置）',
    'reverse_min': '最小出度—最小入度（转置）',
    'complement_out': '最大出度—最小出度（补图）',
    'complement_in': '最大入度—最小入度（补图）',
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
    return '未定义' if not math.isfinite(value) else '正' if value > 0 else '负' if value < 0 else '零'


def same(a, b):
    a, b = number(a), number(b)
    return (math.isnan(a) and math.isnan(b)) or a == b


def yes(value):
    return str(value).lower() == 'true'


def display(value):
    if isinstance(value, bool):
        return '是' if value else '否'
    if value in ('', None, 'None', 'nan', 'NaN'):
        return '未定义'
    try:
        value = float(value)
        return f'{value:.7g}' if math.isfinite(value) else '未定义'
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
                    actual = 'undefined' if '未定义' in pair else 'yes' if set(pair) == {'正', '负'} else 'no'
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
        self.html = ['<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">',
                     '<title>PermWorld 与 GraphWorld：固定范围综合报告</title>',
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
            self.html.append('<details><summary>' + html.escape(caption or '完整表格') + f'（{len(rows)}行）</summary>')
        elif caption:
            self.html.append('<h4>' + html.escape(caption) + '</h4>')
        if caption:
            self.text.extend([caption + f'（{len(rows)}行）', ''])
        self.html.append('<div class="scroll"><table><thead><tr>' +
                         ''.join('<th>' + html.escape(name) + '</th>' for name in columns) + '</tr></thead><tbody>')
        self.text.append('\t'.join(columns))
        for row in rows:
            self.html.append('<tr>' + ''.join('<td title="' + html.escape(str(row.get(name, '')), quote=True) + '">' +
                html.escape(display(row.get(name, ''))) + '</td>' for name in columns) + '</tr>')
            self.text.append('\t'.join(str(row.get(name, '')) if row.get(name, '') not in ('', None) else '未定义' for name in columns))
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
    doc.title('数学对应下的表征相似性：PermWorld 八关系与 GraphWorld 四关系综合报告', 1)
    doc.paragraph('本报告只读取完成并审计通过的既有表格。PermWorld纳入64个固定20k单任务终点（16个旧模型复用、48个缺失模型补齐），八关系×两编码×两配置为32格；GraphWorld另有四任务、四关系的参考Transformer试跑。两系统分别只有一个数据世界，初始化在指定配置内共享。32格、64模型、八关系和图论四关系均不能当成独立重复。', True)
    doc.paragraph('阅读顺序为六个相似度原值S → 训练后正确−错误偏好Bt → 初始化偏好B0及正确/错误变化dc、dw → Δ。这里“S原值”表示尚未作差分的指标，不是raw_within_length这一控制版本。答案分层后的S仍是原值。所有数值属于探索性描述，不给确认性p值、等价性或机制结论。')
    doc.paragraph('B0=S初始正确−S初始错误；Bt=S训练正确−S训练错误；dc=S训练正确−S初始正确；dw=S训练错误−S初始错误；di=恒等配对的训练变化；Δ=Bt−B0=dc−dw。正Δ不要求正确S上升，负Δ也不等于没有学习。恒等配对单独保存，不构成必须超过的上限。')
    doc.paragraph('主指标是在任务token出现前、输入结束位置的final norm状态，逐长度分别计算后按固定权重汇总。PermWorld长度10…30各1/21，GraphWorld大小6、7、8各1/3。任一长度中某分量未定义，其汇总仍未定义；不合并长度、不填零、不重新分配缺失权重。')
    doc.image(PERM / 'figures/four_controlled_comparisons.png', '原四部分受控比较图：固定指标、模型配置、编码、全部层；Bt与Δ分别保留')
    main = indexed(data['main'], ('relation', 'architecture', 'encoding'), 32)
    doc.title('一、八关系的全部32格：原值与训练增量')
    for relation in RELATIONS:
        doc.title(DISPLAY[relation], 3)
        rows = [main[relation, a, e] for a in ARCHS for e in ENCODINGS]
        doc.table(rows, ['architecture', 'encoding'] + list(FIELDS), '主答案分层、final norm、普通CKA：四格全部列出')
        for row in rows:
            doc.paragraph(f"{row['architecture']} / {row['encoding']}：Bt={display(row['Bt'])}（{direction(row['Bt'])}），"
                f"Δ={display(row['Delta'])}（{direction(row['Delta'])}）；正确S变化dc={display(row['dc'])}，错误S变化dw={display(row['dw'])}。"
                '这些变化共同决定Δ，不能只把它称为正确表征趋同。')
        control_rows, output_rows = [], []
        for row in rows:
            for control in P_CONTROLS:
                for metric in CKAS:
                    identity = dict(architecture=row['architecture'], encoding=row['encoding'], control=control, metric=metric)
                    control_rows.append(field_row(row, f'{control}_{metric}_', **identity))
                    output_rows.append(field_row(row, f'output_{control}_{metric}_', **identity))
        columns = ['architecture', 'encoding', 'control', 'metric'] + list(FIELDS)
        doc.table(control_rows, columns, '隐藏表征：raw/答案分层与普通/U-centered CKA', True)
        doc.table(output_rows, columns, '输出概率：平行对照，保留各自原值与分解', True)
    doc.paragraph('共同答案分层保留完整的预定关系key和最小组数3；不会按结果改变样本、长度或负对照。components(Cπ)=global_descents(π)+1，右侧分层key减1只改变整数坐标，实际标签及答案onehot/scalar仍使用原数字。错误变换在部分输入上可能偶然保持答案公式，这些样本没有删去。')
    doc.link(PERM / 'relation_label_alignment.csv', '全部正确、错误、恒等配对的真实答案公式匹配率')

    doc.title('二、编码改变初始化偏好，还是训练后偏好？')
    doc.paragraph('Γ方向固定为规范化轮换记号减一行记号：Γ0为B0差，Γt为Bt差，Γc、Γw为正确/错误变化差，ΓΔ为Δ差。下面保留八关系×两配置共16组、两控制及两CKA版本。Δ翻转和Bt翻转是不同判别；若Δ翻转而Bt未翻转，不能称学习后的关系偏好翻转。')
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
    doc.table(encoding_rows, ['relation', 'architecture', 'control', 'metric'] + names, '16组编码分解及全部固定控制/敏感性版本')
    original_encoding = [row for row in data['summary'] if row['contrast_type'] == 'encoding_difference'
                         and row['branch'] == 'hidden' and row['layer'] == 'final_norm']
    require(len(original_encoding) == 128, 'Full final encoding contrast grid changed')
    doc.table(original_encoding, ['relation', 'architecture', 'control', 'metric', 'Gamma0', 'Gammat', 'Gammac', 'Gammaw', 'GammaDelta'],
              '原三主指标及U-centered：全部final编码差，指标幅度不跨指标比较', True)
    interaction = [row for row in data['summary'] if row['contrast_type'] == 'interaction'
                   and row['branch'] == 'hidden' and row['layer'] == 'final_norm']
    require(len(interaction) == 64, 'Full final configuration-by-encoding contrast grid changed')
    doc.table(interaction, ['relation', 'control', 'metric', 'B0', 'Bt', 'dc', 'dw', 'Delta', 'I0', 'It', 'I'],
              '全部final配置×编码交互；探索性描述，不是纯架构原因', True)
    doc.paragraph('两编码复制同架构内的真实step-0参数，底层排列、轨道划分和更新预算配对；不同架构不声称同一初始化。编码也改变序列组织、初始化输入几何和学习难度。源准确率不同的格不能被解释成同等任务掌握下的纯编码机制。')

    doc.title('三、指标、配置、层与全部长度')
    doc.paragraph('三主指标是普通线性CKA、k=10邻域重叠和验证fit上PCA rank≤8的留出Procrustes；Procrustes相似度为负归一化预测误差，越大越相似。U-centered CKA是固定敏感性版本。不同指标比较方向及模式，不把数值大小相减或当成独立重复。')
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
              '32格×两控制×三指标，全部方向及是否共同同向', True)
    configuration = [row for row in data['summary'] if row['contrast_type'] == 'configuration_difference'
                     and row['branch'] == 'hidden' and row['layer'] == 'final_norm']
    require(len(configuration) == 128, 'Complete configuration contrast grid changed')
    doc.table(configuration, ['relation', 'encoding', 'control', 'metric', 'B0', 'Bt', 'dc', 'dw', 'di', 'Delta'],
              '所有final配置差（GRU减Transformer）', True)
    doc.paragraph('配置比较保留参数量和源任务成绩。Transformer与GRU的容量、计算形式及掌握程度并未匹配，因此这里只能描述现有配置差异，不能识别纯架构效应。')
    doc.table(data['layers'], ['relation', 'architecture', 'encoding', 'control', 'metric', 'layer'] + list(FIELDS) +
              ['Bt_direction', 'Delta_direction', 'Bt_flip_vs_layer4', 'Delta_flip_vs_layer4', 'Bt_encoding_flip', 'Delta_encoding_flip'],
              '全部四个block/GRU层、单列final norm与embedding：768行', True)
    doc.paragraph('层顺序固定，第四层是方向对照而非选出的最佳层；final norm只单列归一化后的状态。任务前输入结束token的embedding在每个固定长度内是结构常量，相似度未定义不是“没有学习”的证据。层间Δ变化仍须看B0和Bt：可能是初始化偏好变化，也可能是训练后的配对偏好变化。')
    # Every scientific per-length cell remains visible, not just the main endpoint.
    for relation in RELATIONS:
        rows = [row for row in data['length'] if row['relation'] == relation]
        require(len(rows) == 3024, 'A relation lost scientific per-length cells')
        doc.table(rows, ['architecture', 'encoding', 'control', 'branch', 'layer', 'metric', 'length', 'fit_samples', 'test_samples'] +
                  list(FIELDS) + ['error_initial_correct', 'error_initial_wrong', 'error_initial_identity',
                                  'error_trained_correct', 'error_trained_wrong', 'error_trained_identity', 'nan_reason'],
                  DISPLAY[relation] + '：全部3024逐长度单元，含输出、四层和两控制', True)
    doc.link(PERM / 'per_length_results.csv', '完整逐长度原始CSV及激活/索引/预处理来源哈希')

    doc.title('四、答案几何、输出与全部源模型学习')
    doc.paragraph('真值分层移除组内答案均值，但不移除所有置信度、预测误差或非线性答案信息。输出概率是数字0…30上的条件概率；源任务准确率与CE则使用188词表。隐藏Δ与输出Δ不能相减来估计额外机制。真值答案几何是静态参照，不构造初始化、训练或Δ。')
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
              '完整静态答案几何：8064唯一关系/控制/角色/指标/长度单元；四配置重复副本不增加证据', True)
    doc.link(PERM / 'gold_baseline_length.csv', '原32256行金标基线（含四配置的重复索引）')
    doc.table(data['source64'], ['task', 'architecture', 'encoding', 'parameters', 'answer_accuracy', 'minimum_length_accuracy',
              'answer_ce', 'length_majority_accuracy', 'accuracy_minus_length_mode', 'gain_over_train_defined_length_mode_pp',
              'baseline_comparison', 'length_gain_directions'], '64个源模型：固定20k、恒等输入、完整21长度等权')
    doc.paragraph('长度众数从原训练集恒等标签定义，平局取最小数字，随后固定用于分析集。高于众数只是描述，不是预设“充分学习”门槛；低于众数也不证明模型没有使用输入。弱学习模型全部保留，不根据准确率删掉几何格。原16个模型的任务特定step-0答案CE缺失保持缺失，不能用共享hidden或另一任务的CE填入。')
    source_details = [row for row in data['source'] if row['condition'] == 'trained' and row['action'] == 'identity']
    require(len(source_details) == 2688, 'Source identity learning details are incomplete')
    doc.table(source_details, ['architecture', 'encoding', 'task', 'split', 'length', 'answer_accuracy', 'answer_ce',
              'length_majority_accuracy', 'accuracy_minus_length_mode'], '64源：分析和验证的每长度恒等任务成绩', True)
    doc.table(data['curves'], ['run_id', 'architecture', 'encoding', 'task', 'step', 'train_loss', 'answer_accuracy',
              'numeric_accuracy', 'eos_accuracy', 'samples'], '原保存的20个验证记录/模型；不用于选择检查点', True)
    doc.paragraph('曲线中的train_loss是原记录的训练损失，不是完整训练集答案CE；曲线本身未保存逐次答案CE。上面的最终答案CE来自固定终点的源学习评估，不替曲线编造CE。numeric_accuracy是原保存的数字内读出辅助分数，不替代188词表答案准确率；曲线EOS采用teacher-forced答案输入，口径与prompt-only表征评估不同。')
    doc.link(PERM / 'source_learning.csv', '所有输入变换与初始/训练状态的源学习表；缺失CE保留')

    doc.title('五、同任务跨配置的直接相似度')
    doc.paragraph('同一任务、相同编码下直接比较Transformer与GRU的32个配对；两控制及两CKA版本完整保留。这是同任务identity--identity比较，没有数学正确/错误关系，其共同答案key为[n,y_identity,y_C,y_I]，与关系任务对的分层定义不同。共享输入和容量/掌握差异仍然影响解释。')
    doc.table(data['cross_summary'], ['task', 'encoding', 'control', 'metric', 'length_count', 'initial', 'trained', 'change'],
              '32个同任务配对的128个完整固定汇总')
    doc.table(data['cross'], ['task', 'encoding', 'control', 'metric', 'length', 'fit_samples', 'test_samples',
              'initial_similarity', 'trained_similarity', 'similarity_change', 'nan_reason'], '同任务全部2688逐长度单元', True)

    doc.title('六、GraphWorld：四关系的同定义试跑')
    doc.paragraph('图论试跑只采用原256维、四层、八头Transformer参考配置和一套二进制边编码，复制其真实初始化与保存的起始RNG。四任务共享一个图数据世界、四关系只有转置和补图两类变换；它不是八关系×编码×架构全网格的跨领域复现。系统变化同时改变大小、输入编码、任务和分布，不直接相减PermWorld与GraphWorld的CKA。')
    doc.image(GRAPH / 'answer_control_decomposition.png', '原GraphWorld三指标答案控制分解图，固定n=6、7、8等权')
    g_main = indexed(data['g_main'], ('relation',), 4)
    for relation in G_RELATIONS:
        doc.title(DISPLAY[relation], 3)
        row = g_main[relation,]
        doc.table([row], ['relation', 'left', 'right', 'correct_transform', 'wrong_transform', 'mapped_right'] + list(FIELDS),
                  '主答案控制、final norm、普通CKA')
        doc.paragraph(f"Bt={display(row['Bt'])}（{direction(row['Bt'])}），Δ={display(row['Delta'])}（{direction(row['Delta'])}），"
                      f"dc={display(row['dc'])}、dw={display(row['dw'])}；正Δ与正确S上升仍分别判断。")
        hidden, output = [], []
        for control in G_CONTROLS:
            for metric in CKAS:
                hidden.append(field_row(row, f'{control}_{metric}_', control=control, metric=metric))
                output.append(field_row(row, f'output_{control}_{metric}_', control=control, metric=metric))
        doc.table(hidden, ['control', 'metric'] + list(FIELDS), 'raw、答案、答案+密度控制与U-centered敏感性')
        doc.table(output, ['control', 'metric'] + list(FIELDS), '图论输出概率平行对照', True)
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
              '图论全部四关系、三控制、三主指标方向', True)
    doc.table(data['g_summary'], ['relation', 'control', 'branch', 'metric', 'static_reference'] + list(FIELDS),
              '完整240个图论汇总，静态输入/答案分支保留未定义训练增量', True)
    doc.table(data['g_length'], ['relation', 'n', 'control', 'branch', 'metric', 'fit_count', 'analysis_count'] + list(FIELDS) +
              ['initial_correct_error', 'initial_wrong_error', 'initial_identity_error',
               'trained_correct_error', 'trained_wrong_error', 'trained_identity_error'], '完整720个图论逐大小单元', True)
    doc.paragraph('密度控制在共同答案key上增加真实anchor边数m，而不是事后选择密度范围。组阈值仍为3，控制后样本减少或方差/秩不足会留下未定义值。不能放宽阈值、重分配n权重，或把NaN解读成“没有关系”。位置二进制onehot输入核对预定槽位/类别等距变换不敏感，它的高相似度是输入参照而不是训练证据。')
    doc.paragraph('完整<R,C,W>轨道排除只保证指定变换不跨数据划分；任意顶点重标记后的同构图仍可能分布在不同split。原错误变换W保留边数且可能偶然保留任务答案，这不改负对照定义。图密度、最大/最小度及错配的偶然答案一致都需与源成绩和静态分支一起读。')
    doc.table(data['g_source4'], ['task', 'architecture', 'encoding', 'answer_accuracy', 'minimum_size_accuracy', 'answer_ce',
              'mode_accuracy', 'gain_over_train_defined_mode_pp', 'edge_mean', 'density_mean', 'EOS_evaluated'], '图论全部四源学习与训练众数')
    doc.table(data['g_source'], ['task', 'condition', 'split', 'n', 'action', 'samples', 'answer_accuracy', 'answer_ce',
              'training_mode', 'mode_accuracy', 'edge_mean', 'density_mean', 'EOS_evaluated'], '图论全240源学习单元', True)
    doc.paragraph('图论源答案准确率/CE同样按完整188词表计算。prompt-only表征提取没有评估EOS；原teacher-forced源验证EOS另有不同口径，不能在这里补成prompt-only能力。弱学习仍保留，不据此自动重训、增加模型或改变协议。')

    doc.title('七、未定义、复用与证据依赖')
    for key, label in (('coverage', 'PermWorld每分量的缺失长度与汇总身份'), ('g_coverage', 'GraphWorld每分量的缺失大小与汇总身份')):
        undefined = [row for row in data[key] if not yes(row['defined_aggregate'])]
        doc.paragraph(label + f'：{len(undefined)}个未定义分量记录。该数量是同一批耦合格的记录数量，不是独立效应数量。')
        if undefined:
            columns = [name for name in ('relation', 'architecture', 'encoding', 'control', 'branch', 'layer', 'metric', 'component',
                       'undefined_lengths', 'undefined_length_count', 'fixed_length_count', 'aggregate', 'defined_aggregate',
                       'missing_weights_reallocated', 'static_reference') if any(name in row for row in undefined)]
            doc.table(undefined, columns, label + '：全部未定义记录', True)
    doc.paragraph('初始化身份使用实际保存的step-0参数和原随机状态核验，而不是把相同种子当作同一权重。旧16终点及原两条关系的6048科学行逐值复用；旧分层key整数dtype与当前安全int64坐标可不同，逐值、分组和中心化一致性另有审计。多指标读数共用预处理后输入与anchor，不能把多指标当作更多样本。')
    doc.table([{'field': key, 'value': value} for key, value in data['pr'].items()
               if key in ('original_scientific_rows_reused', 'exact_binary64_comparisons', 'primary_real_independent_fixture_checks',
                          'actual_independent_metric_max_error', 'full21_groups', 'no_NaN_reweight')],
              ['field', 'value'], '原科学表格独立审查中的实际证据')
    doc.paragraph('当前32格依赖于同一世界、共享anchor、同架构真实初始权重及两类变换，八实例也并非八个独立代数系统。图论四关系共享四任务和两变换。没有多初始化/多世界确认、统计显著性、等价性检验或机制鉴定。F17、旧训练/报告和发布产物的保护由最终来源审计记录；本综合写作不会更改它们。')
    doc.table(data['timing']['phases'], ['phase', 'start_utc', 'finish_utc', 'exit_code', 'wall_seconds',
              'user_cpu_seconds', 'system_cpu_seconds', 'max_rss_kib'], '14个固定阶段的实际记录；不与重叠等待时间相加', True)
    doc.table([dict(field=key, value=json.dumps(data['audit'][key], ensure_ascii=False, sort_keys=True))
               for key in ('formal_sources', 'source_versions', 'actual_saved_initializations', 'old_artifact_protection',
                           'tracked_git', 'timing', 'physical_storage')],
              ['field', 'value'], '最终来源审计中的完整成本、真实初始化和保护记录', True)
    doc.paragraph('训练进程时间、阶段壁钟、CPU时间和GPU活跃时间是不同口径。没有独立GPU核计时就保持未知，不以CPU只读审计推断训练GPU时间为0，也不把历史16模型成本计成此次补齐的训练成本。物理存储按真实路径去重；共享别名与重叠组的字节数不能简单相加。')

    doc.title('八、对原研究问题的回答与停止边界')
    positive = [row for row in data['main'] if math.isfinite(number(row['Delta'])) and number(row['Delta']) > 0]
    doc.paragraph(f'1. 哪些关系配置的训练增量超过自身真实初始化参照？主答案分层final普通CKA中有{len(positive)}个格的Δ方向为正（32个耦合格中的描述计数）。它们如下；完整负值与未定义值已在第一部分同等展示。这里不使用“稳定超过”或“显著超过”，也不从正Δ推出正确S上升。')
    if positive:
        doc.table(positive, ['relation', 'architecture', 'encoding', 'B0', 'Bt', 'dc', 'dw', 'Delta'], '主读数Δ为正的格，非独立重复')
    doc.paragraph('2. 是否超过输入结构和简单答案/输出解释？应读第一、四、六部分的同控制原值、静态答案、输出及图论输入分支。真实初始化是输入/模型结构参照，不能被当作对相似性来源的精确分解。答案控制仍保留概率与预测误差信息，当前数据不能仅靠CKA鉴定额外编码机制。')
    doc.paragraph('3. 编码改变的是哪个阶段？第二部分逐组同时给Γ0、Γt、Γc、Γw、ΓΔ和两种翻转判别。只有Bt翻转才是现有测量下训练后偏好方向翻转；Δ翻转而Bt未翻转主要限定为相对初始化的增量方向变化，进一步解释需结合Γ0及源学习。')
    doc.paragraph('4. 现象是否依赖指标、配置、层或长度？第三部分保留所有预定格。三指标方向可不同，各层的B0与Bt可分别改变；不存在按最佳层、指标或长度筛选出的主张。配置容量与源任务掌握不同，因此不能把配置差异等同为纯架构原因。')
    doc.paragraph('5. 同任务跨配置相似度提供什么？第五部分给真实初始化与训练后的直接配对及变化。它描述当前两个模型配置的几何关系，分层key与跨任务关系不同；高CKA不能单独证明共享算法。')
    doc.paragraph('6. GraphWorld是否支持跨系统推广？第六部分给四个完整关系和三套控制的真实读数。它是一个参考配置试跑，系统间输入、大小和任务同时变化，最多提供有条件的跨系统观察，不能建立普适数学机制或架构×编码跨系统结论。')
    doc.paragraph('7. 下一步编码对照能解决哪个问题？本轮已经以真实共享step-0把初始化偏好与训练后偏好分开。仍未解决的是在相近任务掌握、独立初始化和数据世界下，这些方向及幅度是否复现，以及答案/置信度之外的信息究竟是什么。当前记录这些未决问题，不自动开展新实验。')
    doc.paragraph('8. 本轮是否完成？前置审计确认固定64个PermWorld终点、4个GraphWorld终点及14阶段全部结束；本报告只增加只读综合写作。完成固定范围后停止，不增种子、任务、训练预算、大模型、GroupWorld、轮换扩展、零空间、有限域或GELU组合实验，不自动上传或发布。')

    doc.title('原报告、全部图表及输入身份')
    doc.link(PERM / 'report.html', '原冻结PermWorld完整报告')
    doc.link(GRAPH / 'report.html', '原GraphWorld试跑报告')
    for name in data['pa']['figures_sha256']:
        doc.link(PERM / name, '原预定图：' + name)
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
    doc.paragraph('combined_input_hashes.json列出本综合写作实际读取或引用的每个小文件SHA、原审计绑定以及报告SHA。大型权重、激活、数据数组均未再次读取。HTML显示值保留原值提示，txt表格保留输入数值字符串，CSV是完整数值与来源元数据。归档与根completion由Root在综合报告审查后写定，避免循环哈希依赖。')
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
