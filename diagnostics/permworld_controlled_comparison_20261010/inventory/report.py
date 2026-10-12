"""首轮20k既有模型的中文受控比较报表，仅读新分析CSV。

不拟合模型、不加载权重或原始输入；三个指标保持各自坐标，不制造总分。
缺失/未定义值完整保留，主分析固定answer_strata、hidden、final_norm。
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
LABELS = dict(descent_inverse='下降—逆下降／取逆', peak_complement='峰—谷／补排列',
              transformer='Transformer', gru='GRU', one_line='一行', canonical_cycle='轮换',
              linear_cka='线性 CKA', knn_overlap='邻域重叠 k=10', procrustes='留出 Procrustes（负 NRMSE）')


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
            return '未定义/缺失'
        return f'{100*value:.2f}%' if percentage else f'{value:.6f}'
    return str(value)


def direction(values):
    if any(not finite(v) for v in values):
        return '未定义'
    signs = [1 if v > 0 else -1 if v < 0 else 0 for v in values]
    return '同向' if len(set(signs)) == 1 else '不同向'


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
    assert all(k in lookup for k in main_keys), '固定主分析缺少条件，不可挑选子集生成报告'
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
        summary['undefined_reason'] = '' if complete else '不是完整固定21长度；未按剩余长度重加权'
        source_summary.append(summary)
    encoding_changes = []
    for a in ARCHITECTURES:
        for relation in RELATIONS:
            line = lookup[a, 'one_line', relation, 'answer_strata', 'hidden', 'final_norm', 'linear_cka']
            cycle = lookup[a, 'canonical_cycle', relation, 'answer_strata', 'hidden', 'final_norm', 'linear_cka']
            change = dict(architecture=a, relation=relation)
            for name, field in (('Gamma0', 'B0'), ('Gammat', 'Bt'), ('Gammac', 'dc'), ('Gammaw', 'dw'), ('GammaDelta', 'Delta')):
                change[name] = cycle[field] - line[field] if finite(cycle[field]) and finite(line[field]) else float('nan')
            change['Delta翻转而Bt未翻转'] = all(finite(r[f]) for r in (line, cycle) for f in ('Bt', 'Delta')) and line['Delta'] * cycle['Delta'] < 0 and line['Bt'] * cycle['Bt'] > 0
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
        return '<p>未提供；保留缺失，不补造结果。</p>'
    columns = columns or list(dict.fromkeys(k for row in rows for k in row))
    return '<div class="scroll"><table><thead><tr>' + ''.join('<th>' + html.escape(k) + '</th>' for k in columns) + '</tr></thead><tbody>' + ''.join('<tr>' + ''.join('<td>' + html.escape(display(row.get(k, ''))) + '</td>' for k in columns) + '</tr>' for row in rows) + '</tbody></table></div>'


def render(result, output, fixture=False):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    undefined_plot = main_figure(result, output)
    directions = result['directions']
    bt_agree = sum(r['Bt_direction']=='同向' for r in directions)
    delta_agree = sum(r['Delta_direction']=='同向' for r in directions)
    bt_missing = sum(r['Bt_direction']=='未定义' for r in directions)
    delta_missing = sum(r['Delta_direction']=='未定义' for r in directions)
    flip_without_bt = sum(r['Delta翻转而Bt未翻转'] for r in result['encoding_changes'])
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
                layer_sign_change += status == '不同向'; layer_incomplete += status == '未定义'
    text = [
        'PermWorld 首轮16模型：受控表征比较',
        '生成 UTC：' + datetime.now(timezone.utc).isoformat(),
        '合成CPU fixture；不是真实结果。' if fixture else '唯一主队列：首轮四任务×两编码×两配置，固定20k、n10..30、真实保存step-0。',
        '先回答四类控制比较（均为探索性描述，不是确认性检验）：',
        f'A 指标：在8个配置×编码×关系主条件中，三主指标对训练后Bt方向同向{bt_agree}/8、对Delta方向同向{delta_agree}/8；未定义分别{bt_missing}、{delta_missing}。不同指标不比较绝对分值或Delta大小，不平均成总分。',
        f'B 模型配置：固定CKA、编码、任务对、底层输入和20k预算，共保留{len(configuration)}项主配置差（GRU−Transformer），完整B0/Bt/dc/dw/Delta列在表中；324万与166万参数未匹配，不能解释为纯架构因果效应。',
        f'C 编码：四个配置内的关系对照均保留Gamma0/Gammat/Gammac/Gammaw/GammaDelta；其中Delta随编码翻转而Bt不翻转的单元为{flip_without_bt}/4。这类变化不能写成训练后的关系偏好翻转。',
        f'D 提取层：两配置两编码两关系均展示四层及独立final_norm；8条件中Bt方向跨层变化{layer_sign_change}，存在未定义层{layer_incomplete}。输入结束token的embedding常量读数保留未定义，不画浮点噪声；归一化变化与层变化分开。',
        '',
        '共同固定项：同一数据world/轨道划分/anchor顺序；每长度先计算，再固定1/21汇总；共同答案分层控制为主，raw未调整、输出概率及真值基线并列保留。未定义长度导致整项固定权重汇总未定义，不按剩余长度重加权。',
        '每条件保留正确、错误、恒等原始相似度：B0=S正确,初始−S错误,初始；Bt=S正确,训练−S错误,训练；dc/dw分别为正确/错误变化；Delta=Bt−B0=dc−dw。正Delta不保证正确相似度提高，初始化配对起点必须单独看。',
        '两编码在各配置内共享真实初始参数和词表；编码输入几何仍可不同。GRU为显式层间dropout版，保存的原初始化21tensor逐值迁移，非用相同seed冒充相同权重。跨配置初始权重不相同。',
        'Procrustes使用负留出NRMSE并保留原误差：独立validation拟合8维PCA/正交映射，test只评价。kNN k=10、欧氏距离、稳定anchor ID并列排序；普通CKA与U-centered敏感性沿旧定义。不同指标专属预处理详见protocol。',
        '答案分层并不证明排除了输出概率、置信度及错误结构；不能用隐藏Delta减输出Delta估算额外机制。静态真值基线没有伪造初始化/训练Delta。',
        '下降—逆下降和峰—谷是不同任务对与输出统计的两个关系实例；其差异不是只改变数学关系的因果实验。正确/错误配对选择性也不证明神经网络实现了代数计算机制。',
        '所有低任务成绩条件保留：比较是固定预算下的编码/配置差异，不是同等任务掌握程度下的机制差异。没有90%通过门槛、补训或最佳checkpoint选择。EOS旧validation teacher-forced准确率不能替代答案准确率。',
        '一个数据world、每配置一个初始权重；16模型、21长度与三指标不是独立重复。不报告确认p值、等价性检验或总体推广置信度。逐项比较仍不能排除因素交互；架构×编码交互仅探索性列出。',
        '历史82模型多层结果单独留作引用 diagnostics/permworld_layers_20261009/，初始化按原程序重建、真实身份未验证；未并入主队列。第二轮8k可学习性模型完全排除。',
        '本轮只前向/指标与报告；不创建优化器、不反传、不增加样本/更新、不调参，不开展GroupWorld、768确认、有限域、GELU、零空间或F17工作。',
    ]
    metadata_ready = 'validation_audit.json' in result['metadata'] and 'completion.json' in result['metadata']
    text.append('最终核查/停止状态：' + ('见validation_audit.json及completion.json；本report不自行修改执行状态。' if metadata_ready else '最终审计或completion尚缺，当前报告保留“待核查”，不能冒称本轮全完成。'))
    text += ['', '三主指标方向表（数值符号，非显著性）：', json.dumps(directions, ensure_ascii=False, indent=2),
             '', '编码分解：', json.dumps([{k:(v if not isinstance(v,float) or math.isfinite(v) else None) for k,v in r.items()} for r in result['encoding_changes']],ensure_ascii=False,indent=2),
             '', '全部主条件B0/Bt/dc/dw/Delta及原始正确/错误/恒等：']
    for row in result['main']:
        text.append('/'.join(str(v) for v in key(row)) + ' | ' + ', '.join(f'{f}={display(row.get(f,float("nan")))}' for f in FIELDS) + ' | ' + row.get('undefined_reason',''))
    text += ['', '源任务固定21长度成绩汇总（全部condition/split/action保留）：']
    for row in result['source_summary']:
        text.append('/'.join(str(row[k]) for k in ('architecture','encoding','task','condition','split','action')) +
                    f' | answer acc={display(row["answer_accuracy"],True)}, answerCE={display(row["answer_ce"])}, train-defined长度众数={display(row["length_majority_accuracy"],True)}, min-length acc={display(row["minimum_length_answer_accuracy"],True)} | {row["undefined_reason"]}')
    text += ['', '原始逐长度结果、全部差值与恒等配对见per_length_results.csv/contrast_summary.csv；所有NaN及原因保留。',
             '同任务跨配置直接相似度单列cross_arch_same_task.csv，回答模型本身相似程度，不与关系选择性混成同一效应。',
             '计算预算及成本来源：', json.dumps(result['metadata'].get('cost_estimate.json',{}), ensure_ascii=False,indent=2),
             '完成可行比较、独立核对、报告后停止。缺失项明确报告，不自动扩展训练或下一系统。']
    (output / 'reader_report.txt').write_text('\n'.join(text) + '\n')
    links = ['comparison_manifest.csv','per_length_results.csv','contrast_summary.csv','source_learning.csv',
             'cross_arch_same_task.csv','gold_baseline_length.csv','scope_status.csv','validation_audit.json','completion.json']
    main_html = '<h1>PermWorld 受控表征比较</h1><pre>' + html.escape('\n'.join(text[:9])) + '</pre><p>' + ' · '.join(f'<a href="{name}">{name}</a>' for name in links) + '</p>'
    main_html += '<img class="main" src="four_part_main.png" alt="指标、配置、编码、层四部分固定图">'
    main_html += '<p>图的所有主条件均固定答案分层。未定义绘图单元数：'+str(len(undefined_plot))+'；具体单元见report_export_audit.json。embedding未定义单独保留，不代表零效果。不同指标轴独立。</p>'
    main_html += '<details open><summary>能判断与仍不能判断</summary><pre>' + html.escape('\n'.join(text[9:20])) + '</pre></details>'
    main_html += '<h2>A 主指标方向与完整主条件</h2>' + table(directions) + table(result['main'], ['architecture','encoding','relation','metric',*FIELDS,'undefined_reason','undefined_lengths'])
    main_html += '<h2>B 配置差：GRU−Transformer</h2>' + table(configuration)
    main_html += '<h2>C 编码差：轮换−一行</h2>' + table(result['encoding_changes'])
    main_html += '<h2>D 全部层差与embedding未定义</h2>' + table(layer_changes) + table(embedding)
    main_html += '<h2>架构×编码交互：探索性、初始化与训练后分开</h2>' + table(interactions)
    main_html += '<h2>源任务成绩：固定21长度，全条件保留</h2>' + table(result['source_summary'])
    for title, subset in (
        ('raw未调整并列读数', [r for r in result['conditions'] if r.get('control')=='raw_within_length']),
        ('输出概率基线：合法数字0..30条件概率', [r for r in result['conditions'] if r.get('branch')=='output_prob']),
        ('U-centered CKA既定敏感性', [r for r in result['conditions'] if r.get('metric')=='debiased_cka']),
        ('同任务跨配置直接相似度，含真实初始化参照', result['cross_arch']),
    ):
        main_html += '<details><summary>'+title+'</summary>' + table(subset) + '</details>'
    main_html += '<details><summary>完整静态真值基线逐长度：无伪造Delta</summary>' + table(result['gold']) + '</details>'
    main_html += '<details><summary>全部contrast行／不挑选条件</summary>' + table(result['contrast_rows']) + '</details>'
    main_html += '<details><summary>全部逐长度表／未重加权缺失</summary>' + table(result['length_rows']) + '</details>'
    main_html += '<pre>' + html.escape(text[18]+'\n'+text[19]) + '</pre>'
    main_html += '<details><summary>成本、最终核查与停止记录</summary><pre>' + html.escape(json.dumps(result['metadata'],ensure_ascii=False,indent=2)) + '</pre></details>'
    (output / 'reader_report.html').write_text('<!doctype html><meta charset="utf-8"><title>PermWorld 受控比较</title><style>body{max-width:1350px;margin:25px auto;padding:0 20px;font:15px/1.5 system-ui}pre{white-space:pre-wrap}.main{width:100%}.scroll{overflow:auto}table{border-collapse:collapse;white-space:nowrap;font-size:12px}td,th{border:1px solid #ddd;padding:5px}th{background:#edf1f5}details{margin:18px 0}summary{font-weight:600;cursor:pointer}</style>'+main_html)
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
        check('Delta_flip_not_Bt_flip_identified',all(r['Delta翻转而Bt未翻转'] for r in result['encoding_changes']))
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
