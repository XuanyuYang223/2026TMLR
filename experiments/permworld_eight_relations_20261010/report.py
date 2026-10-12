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
    'descent_inverse': '下降数—逆下降数', 'peak_complement': '峰—谷',
    'fixed_complement': '不动点—反不动点', 'exceedance_inverse': '超越数—亏缺数',
    'subsequence_complement': 'LIS—LDS', 'run_complement': '上升连续段—下降连续段',
    'record_complement': '左至右极大—极小记录', 'decomposition_complement': '全局下降数—组件数(+1)',
}
FIELDS_DISPLAY = ['initial_correct', 'initial_wrong', 'initial_identity', 'trained_correct', 'trained_wrong',
                  'trained_identity', 'B0', 'Bt', 'dc', 'dw', 'Delta']


def formatted(value):
    if isinstance(value, bool):
        return '是' if value else '否'
    try:
        x = number(value)
        return '未定义' if not math.isfinite(x) else f'{x:.4f}'
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
        '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>PermWorld 八关系完整受控比较</title>',
        '<style>body{max-width:1500px;margin:32px auto;padding:0 20px;font:15px/1.6 sans-serif}table{border-collapse:collapse;white-space:nowrap}th,td{border:1px solid #ddd;padding:5px 9px}th{background:#eef2f7}.scroll{overflow:auto;margin:16px 0}img{max-width:100%}details{margin:12px 0}.notice{background:#fff5d9;padding:12px}</style>',
        '<h1>数学对应如何改变表征相似性：PermWorld 八关系受控扩展</h1>',
        '<p>本报告完整纳入一个既有数据世界、每架构一个真实初始化、64个单任务20k终点。16个旧终点及两条关系的已核验结果复用，最多补48个缺失模型；没有因为几何正负、源任务准确率或未定义指标选择子集。八个关系实例只覆盖取逆与补排列两种变换类型，32个条件不是独立重复。</p>',
        '<p class="notice">顺序是原始相似度 S → 训练后正确/错误偏好 Bt → 相对真实初始化变化 Δ。Δ翻转不保证Bt翻转，正Δ不保证正确相似度上升；本轮属于探索性扩展，不作确认性p值、等价性或数学机制的推断。源模型容量和任务掌握程度不匹配，配置比较不等于纯架构效应。</p>',
        '<h2>全部关系的原始相似度与分解</h2>',
        '<p>下面是答案共同分层、任务token前输入结束位置、final norm的普通线性CKA。每一长度独立计算，再按10…30固定1/21汇总。任一长度未定义，该分量汇总仍未定义；没有长度合并、NaN填0或缺失重加权。</p>',
        table(main, ['relation', 'architecture', 'encoding'] + FIELDS_DISPLAY),
        '<p>B0=S正确,初始−S错误,初始；Bt=S正确,训练−S错误,训练；dc=正确训练−正确初始，dw=错误训练−错误初始；Δ=Bt−B0=dc−dw。恒等配对原值独立保留，不当作要超过的上限。</p>',
        '<p>原定错误变换不保证每个样本的答案都错误；偶然仍满足公式的样本完整保留，逐长度匹配比例见relation_label_alignment.csv。尤其近乎常量的任务，错误配对也可能有很高答案匹配率，不据此修改负对照。</p>',
        '<h2>编码、模型配置与层</h2>',
        '<p><img src="figures/four_controlled_comparisons.png" alt="四部分统一受控比较总览：指标、模型配置、编码、层"></p>',
        '<p>Γ固定为轮换减一行，包含Γ0、Γt、Γc、Γw、ΓΔ。需要分别判断编码改变初始化偏好还是训练后偏好。配置差为GRU减Transformer；模型参数量、计算和源学习程度都没有匹配。四个block/GRU层和final norm完整显示，norm只用于明确最后层归一化的影响，未选择最有利层。end-token embedding在逐长度内为结构常量，因此未定义不是没有学习的负证据。</p>',
        table(contrasts, ['contrast_type', 'relation', 'architecture', 'encoding', 'B0', 'Bt', 'dc', 'dw', 'Delta', 'Gamma0', 'Gammat', 'GammaDelta']),
        '<h2>固定指标的方向与长度边界</h2>',
        '<p>普通CKA、k=10邻域重叠、留出Procrustes分别描述不同几何。邻域使用平方欧氏距离、稳定全局anchor ID打破精确距离相等、排除自身及重复anchor；Procrustes仅在验证fit上拟合PCA rank8、RMS及正交映射，analysis输入留出，S为负NRMSE，原始误差全部保存。U-centered CKA是预定敏感性，层分析仅用这两种CKA。不同指标比较方向和模式，不比较数值幅度，也不当独立重复。</p>',
        '<details><summary>全部64个final条件/控制的三指标方向</summary>' + table(directions,
            ['relation', 'architecture', 'encoding', 'control', 'B0_directions', 'Bt_directions', 'dc_directions', 'dw_directions', 'Delta_directions']) + '</details>',
        '<h2>答案、输出与源任务学习</h2>',
        '<p>共同答案key为[n,left恒等,right恒等−offset,right补排列−offset,right取逆−offset]，每split最小组3。全局下降—组件关系明确满足components(Cπ)=global_descents(π)+1；对所有右侧key减1是双射坐标改变，分组与未平移key完全相同。任务标签、准确率和真实答案onehot/scalar仍使用实际数字，没有把不同答案误当相等。</p>',
        '<p>旧两关系的key原dtype为int16，新协议为安全处理+1使用int64。旧CSV key哈希保持原值，current key哈希与逐值/分区/中心化一致性另存reused_key_coordinate_audit.csv；这是坐标表示差异，不是更换激活或重算旧科学指标。</p>',
        '<p>真值分层消去组内答案均值，但不消去输出概率中的置信度、预测误差和所有非线性标签信息。静态答案基线不设初始化/训练或虚构Δ；答案分层下它们结构常量时保留未定义。输出概率是数字0…30的条件概率，不是188词表完整概率；源准确率和答案CE则使用188词表。不能用隐藏Δ减输出Δ估计额外数学机制。</p>',
        '<details><summary>全部输出概率普通CKA对照</summary>' + table(output,
            ['relation', 'architecture', 'encoding'] + FIELDS_DISPLAY) + '</details>',
        table(source_jobs, ['task', 'architecture', 'encoding', 'answer_accuracy', 'minimum_length_accuracy',
                           'answer_ce', 'length_majority_accuracy', 'accuracy_minus_length_mode']),
        '<p>长度众数只用原train恒等标签定义，平局取最小数字。初始CE缺失不从别的任务读出填入，旧EOS只作为原teacher-forced验证统计另列。弱学习任务仍纳入几何表，但不能据其差异推断同等掌握程度下的编码机制。</p>',
        '<h2>32个同任务跨配置配对</h2>',
        '<p>这里比较Transformer与GRU在同任务、同底层输入的直接CKA。它的答案key为[n,该任务恒等/补排列/取逆]，与关系任务对的共同key不同，不能混成一种效应。它描述现有配置间相似性，不能独立识别共享算法。</p>',
        table(cross, ['task', 'encoding', 'control', 'metric', 'initial', 'trained', 'change']),
        '<h2>全部预定图</h2>',
    ]
    parts.extend('<details><summary>' + html.escape(path.name) + '</summary><img src="figures/' + html.escape(path.name) + '"></details>' for path in plots)
    files = ['per_length_results.csv', 'contrast_summary.csv', 'cross_arch_same_task.csv', 'cross_arch_same_task_summary.csv',
             'gold_baseline_length.csv', 'gold_baseline_summary.csv', 'source_learning.csv', 'source_learning_summary.csv',
             'source_validation_eos_original.csv', 'coverage_status.csv', 'protocol.json', 'training_freeze.json', 'analysis_freeze.json',
             'relation_label_alignment.csv', 'reused_key_coordinate_audit.csv',
             'model_gap_manifest.csv', 'training_manifest.csv', 'cost_estimate.csv', 'math_label_audit.json',
             'metric_review/analysis_audit.json', 'metric_review/independent_metric_actual.json']
    parts += ['<h2>可复现产物与本轮解释范围</h2>',
        '<p>' + ' · '.join('<a href="' + html.escape(name) + '">' + html.escape(name) + '</a>' for name in files) + '</p>',
        '<p>逐关系、逐条件、逐长度和指标的负值、稀疏分层及未定义结果均保留。初始化真实身份、原数据轨道划分、源代码和激活数组哈希在协议及审计中记录。当前证据只能描述数学对应下的相似性与配对偏好如何依赖这些测量和建模条件；CKA不能单独证明编码机制、数学关系的因果表示或功能收益。</p>',
        '<p>图论四任务小试由独立协议和报告管理；它与PermWorld大小、输入结构、任务和样本分布不同，不直接相减跨域原始CKA。本报告不改F17、不改旧报告、不自动发布；完成固定图论试跑和总报告后按授权停止，不扩充种子或继续调参。</p></html>']
    (DIAG / 'report.html').write_text('\n'.join(parts))
    text = ('PermWorld八关系完整受控扩展（探索性）\n'
            '固定原始相似度S、训练后正确−错误偏好Bt、相对真实初始化Δ的完整分解。\n'
            '64单任务模型，32关系条件，8实例，2变换类型，21固定等权长度；它们不是独立重复。\n'
            '旧两关系6048逐长度行及同任务672行逐值复用；新六关系同定义补齐。\n'
            '原始相似度、答案/输出控制、弱学习、各层、三主指标及Ucenter敏感性完整保留。\n'
            '报告路径：report.html；全数据：per_length_results.csv/contrast_summary.csv/source_learning.csv。\n'
            '不单凭Delta翻转认定训练偏好翻转，不单凭CKA认定机制。\n')
    (DIAG / 'report.txt').write_text(text)
    write_json(DIAG / 'report_audit.json', dict(status='passed', per_length_rows=len(source), condition_summaries=len(conditions),
        displayed_primary_conditions=len(main), source_models_displayed=len(source_jobs), plots=len(plots),
        coverage_rows=len(coverage), all_relations_and_conditions_retained=True, no_metric_calculation=True,
        fixed21_no_NaN_reweighting=True, no_training=True, exploratory=True,
        report_sha256=sha(DIAG / 'report.html'), figures_sha256={str(p.relative_to(DIAG)): sha(p) for p in plots}))
    print(json.dumps(dict(status='passed', report=str(DIAG / 'report.html'), plots=len(plots))))


if __name__ == '__main__':
    main()
