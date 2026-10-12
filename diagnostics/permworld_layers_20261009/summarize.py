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
        return '未定义／缺失'
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
    candidate_names = '；'.join(f"{row['cohort']}/{row['relation']}（{row['sources']}源）" for row in candidates) or '无关系同时满足全部可用源、三个主指标的正增量条件'
    body = '<h1>PermWorld：已有检查点的多层与固定多指标诊断</h1>'
    body += '<p class="lead">训练改变了正确与错误输入配对的相对偏好。此轮完整展示各层和固定 pooling，拆开正确 CKA 与错误 CKA 的变化；不将 Δ 自动解释为正确表征趋同或数学机制。</p>'
    body += '<h2>范围与证据身份</h2><p>仅使用历史完成检查点与原固定输入。新增源训练为0；原报告、F17论文、结果和发布产物均保持不变。所有现有 PermWorld 初始化参照为C：按构造器重建，原始 step-0 权重身份未验证。“同种子”不等于已经证明相同历史初始化。下文的“超过初始化”严格指超过该重建参照。</p>'
    body += '<p>分析共' + str(audit['metric_rows']) + '个逐长度指标单元、' + str(audit['source_summary_rows']) + '个源级汇总。每个长度独立计算，以10–30的固定1/21权重汇总；任一长度未定义或缺失即不产生该源整体值，不按剩余长度重加权。关系均值与中位数仅汇总完整长度的源，defined列明确给出可用源数。</p>'
    body += '<p>B0=初始化正确−错误；Bt=训练正确−错误；dc/dw分别是正确/错误指标的训练变化；Δ=Bt−B0=dc−dw。正Δ可能来自错误配对下降，负Δ也不等于没有学习。各指标值处于不同尺度；Procrustes score为−NRMSE，原始误差保留。</p>'
    body += '<p><a href="protocol.json">固定分析协议</a> · <a href="freeze.json">代码/输入版本</a> · <a href="extraction_audit.json">检查点、参数与缓存审计</a> · <a href="validation.json">独立验证</a></p>'
    body += '<h2>1. 哪些关系的训练增量超过重建初始化参照</h2>'
    body += '<p>主分析始终是 final_norm_end（输入结束、任务 token 之前），答案分层控制。下列是所有可用源在三个主指标中均正Δ的描述性候选：' + html.escape(candidate_names) + '。单源关系只能算该源观察；多源同向不是独立新数据世界确认，也不等同显著性结论。</p>'
    body += table(candidates, ('cohort', 'relation', 'kind', 'sources', 'all_lengths_all_sources_three_metrics_positive', 'CKA_delta', 'correct_change', 'wrong_change'))
    body += '<details open><summary>主位置所有关系、源覆盖与分解（不挑选正结果）</summary>' + table(compact, ('cohort', 'relation', 'kind', 'metric', 'sources', 'defined', 'B0', 'Bt', 'dc', 'dw', 'Delta', 'median', 'sd', 'minimum', 'maximum', 'positive_sources', 'positive_all_lengths_sources')) + '</details>'
    body += '<h2>2. 层、长度、指标与任务对依赖</h2><p>图中完整保留四个 block 的残差输出和 final_norm，分别取输入结束位置及固定数值 token 的 contextual mean（位置4,6,…,2n+2）。block输出在最终归一化之前，与 final_norm 分支不同。mean仅对一行编码中的数值位置汇总，不包含任务/查询位置；它是固定敏感性分析，不替换主位置。</p>'
    body += '<p>以下图同时呈现B0、Bt、dc、dw、Δ；各图按自身指标尺度着色，灰色是未定义。答案分层与普通逐长度控制都展示，不按结果选择最有利层。逐长度图保留21个长度，不用去长度均值后合并样本替代逐长度分析。</p>'
    for cohort in sorted({row['cohort'] for row in figures}):
        body += '<details><summary>' + html.escape(cohort) + '：全部固定控制与指标图</summary>'
        for fig in figures:
            if fig['cohort'] == cohort:
                body += '<p>' + html.escape(fig['control'] + ' / ' + fig['metric'] + ' / ' + fig['scope']) + '</p><img src="' + html.escape(fig['file'], quote=True) + '" alt="All predefined layer results">'
        body += '</details>'
    body += '<p>主位置有' + str(len(disagreement)) + '个源/关系单元的Bt或Δ在三个主指标中方向不一致；普通/去偏CKA有' + str(len(changed)) + '个单元发生Bt或Δ方向差异。计数描述的是共享源/关系的覆盖，不是独立重复数。全部方向差异保留于 <a href="direction_agreement.csv">共同源方向表</a>、<a href="debiased_sensitivity.csv">debiased敏感性表</a>。</p>'
    body += '<details><summary>主位置三个指标的方向分歧</summary>' + table(disagreement, ('cohort', 'relation', 'source_unit', 'Bt_linear_cka', 'Bt_knn_overlap', 'Bt_procrustes', 'delta_linear_cka', 'delta_knn_overlap', 'delta_procrustes')) + '</details>'
    body += '<details><summary>主位置普通／去偏CKA的方向差异</summary>' + table(changed, ('cohort', 'relation', 'source_unit', 'Bt_ordinary', 'Bt_debiased', 'delta_ordinary', 'delta_debiased')) + '</details>'
    primary_leave = [row for row in leave if row['branch'] == MAIN_BRANCH and row['control'] == MAIN_CONTROL and row['metric'] in PRIMARY]
    body += '<details><summary>留一关系与留一共享性质族（主位置）</summary>' + table(primary_leave, ('cohort', 'kind', 'metric', 'exclusion_level', 'excluded', 'remaining_relations', 'B0', 'Bt', 'delta_correct', 'delta_wrong', 'delta')) + '</details>'
    body += '<p>留一关系不切断其他关系的任务/轨道依赖；留一共享族将该族所有关系同时移除。两者是敏感性描述，不能当作独立关系bootstrap或新确认实验。完整 <a href="leave_one_relation_family.csv">所有层留一结果</a> 同时覆盖自不变性与跨任务关系，按cohort和kind分开。</p>'
    body += '<h2>3. 输入 embedding、答案与输出关联</h2><p>同一长度下，输入结束 token 的 embedding 完全相同；数值 token 的无序 embedding均值也完全相同，因为每个排列恰含1…n。二者几何在数学上退化，本报告明确记为未定义，不将浮点加法次序噪声当作输入信号。这不表示下游 contextual mean 退化。</p>'
    body += '<p>位置保留的数值 token embedding／onehot展平基线固定在分析前。它们保留输入位置和值的配对；同长度的位置、标点加法项为样本常量，在中心化后抵消。学习后的 embedding 基线可随训练改变；静态onehot没有可解释的训练机制。这些输入基线诊断信号是否已存在于输入表示，不证明网络发现额外代数关系。</p>'
    controls = [row for row in relations if row['branch'] in INPUTS + CONTROLS and row['metric'] in PRIMARY]
    body += '<details><summary>全部输入、任务查询、答案与输出对照</summary>' + table(controls, ('cohort', 'relation', 'branch', 'control', 'metric', 'planned_sources', 'B0_mean', 'Bt_mean', 'delta_correct_mean', 'delta_wrong_mean', 'delta_mean', 'delta_sources')) + '</details>'
    body += '<p>答案分层复用原协议，在fit/test内按同一答案分层分别去均值；不能移除全部置信度、错误或非线性答案信息。输出概率分支仍保留，隐藏Δ减输出Δ不能识别“额外机制”。真值答案没有可学习的step-0，不为它虚构训练增量。查询位置含任务提示，与任务前状态分开报告。</p>'
    body += '<h2>4. 任务数系列与编码对照的衔接</h2><p>此轮不重复任务数漂移分析。<a href="../permworld_scope_inventory_20261009/report.html">原库存报告</a>与<a href="../permworld_cka_20261009/report.html">原CKA诊断</a>保留全部原始结果。旧逐长度初始化保留曲线五个重复的k16均高于k1，但并非每条都单调；R3/R4是同源任务子集敏感性，不是新源种子。高池间相似度与高自身初始化保留同时出现，只支持保留初始几何的解释线索，不能精确拆分来源。这里没有新增任务数结论。</p>'
    body += '<p>下一步两编码的工程对照可检验：编码改变的是B0（初始化偏好），还是Bt（训练后配对偏好），以及Δ来自dc还是dw。需保存真实共享step-0、使用相同底层排列与轨道划分，并保留输出概率、真值答案和准确率。若Δ随编码翻转而Bt接近，不应写为训练学到的关系偏好翻转。架构对照须维持共同输入/监督规则，任务可学习性与几何效果分开评价。</p>'
    body += '<h2>结论与交付</h2><p>本轮可判断固定检查点下哪些层/关系具有一致的配对选择性增量，并展示其长度和指标边界；不能凭CKA认定编码机制、数学关系表征或因果功能。源间标准差只反映共享世界中的初始化差异；本轮不制造新的输入抽样置信区间。历史输入bootstrap的条件范围不能自动扩到新层和新指标。</p>'
    body += '<p><a href="layer_metric_results.csv">完整逐关系／逐源／逐长度原始表</a> · <a href="source_metric_summary.csv">固定长度源汇总</a> · <a href="relation_metric_summary.csv">关系汇总</a> · <a href="length_metric_summary.csv">逐长度源差异</a> · <a href="layer_metric_summary.csv">各层等权关系描述</a> · <a href="summary_audit.json">汇总审计</a> · <a href="summarize.py">汇总脚本</a>。已有模型诊断完成后不自动启动正式批量训练、GroupWorld或F17实验。</p>'
    css = 'body{font:16px/1.65 system-ui,sans-serif;max-width:1400px;margin:34px auto;padding:0 20px;color:#25313d}h1,h2{line-height:1.3}h2{margin-top:35px}.lead{background:#e8f0f5;padding:16px;border-radius:7px}.scroll{overflow:auto}table{border-collapse:collapse;font-size:12px;margin:14px 0;width:100%}th,td{border:1px solid #d9e0e5;padding:6px;text-align:left;white-space:nowrap}th{background:#edf2f5}img{max-width:100%;margin:16px 0}details{margin:20px 0}a{color:#175aa1}'
    (root / 'report.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>PermWorld layer diagnostic</title><style>' + css + '</style>' + body + '</html>')
    text = f'''PermWorld 已有检查点多层探索诊断\n新增源训练：0。旧初始化参照均C，无法验证原step-0身份。\n主位置final_norm_end、答案分层、全部可用源三主指标均正Δ的候选：{candidate_names}。\n主位置三指标Bt/Δ方向分歧单元：{len(disagreement)}；普通/debiased方向差异：{len(changed)}。\n所有层、长度、关系和固定pooling均保留；Δ拆分为正确变化减错误变化。\nEmbedding结束位置和无序数值均值数学上恒定，相关指标未定义。\n任务数系列沿用旧诊断，未重复计算；不能精确分解相似度来源。\n下一步编码实验回答B0还是Bt改变，需真实step-0与共同底层数据。\nCKA、邻域与Procrustes都不单独识别数学机制；源与关系依赖保留，无新确认p值。\n'''
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
