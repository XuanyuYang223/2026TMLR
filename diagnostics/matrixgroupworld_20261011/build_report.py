"""Read saved MatrixGroupWorld artifacts; no model, labels, cache or metric calls.

The root may invoke this after an engineering/budget stop or after complete
analysis. It is a reader, not a stopping rule, science implementation or selector.
"""
from __future__ import annotations
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import html
import json
import math
from pathlib import Path

DEFAULT_DIAG = Path('/home/yangx/ICML/diagnostics/matrixgroupworld_20261011')
DEFAULT_ROOT = Path('/mnt/yangx-projects-f/20261011/matrixgroupworld')
OLD = Path('/home/yangx/ICML/diagnostics/permworld_eight_relations_20261010/final_review')
TABLE_COUNTS = {'geometry_per_n.csv': 432, 'geometry_summary.csv': 144,
                'selection_masks.csv': 96, 'source_learning.csv': 816,
                'pair_per_n.csv': 144, 'pair_summary.csv': 48,
                'category_summary.csv': 96, 'designated_category_comparisons.csv': 64,
                'common_support_training_comparisons.csv': 48, 'designated_rank.csv': 32,
                'W_actual_per_n.csv': 48, 'W_actual_per_anchor.csv': 12288,
                'W_actual_summary.csv': 16}
METRICS = ('linear_cka', 'knn_overlap', 'procrustes')
CONTROLS = ('raw_within_n', 'answer_strata')
COMPONENTS = ('initial_correct', 'initial_wrong', 'initial_identity',
              'trained_correct', 'trained_wrong', 'trained_identity',
              'B0', 'Bt', 'dc', 'du', 'di', 'Delta')
NUMERIC = set(COMPONENTS) | {'n', 'step', 'samples', 'answer_ce', 'eos_accuracy', 'eos_ce',
    'full_vocab_answer_accuracy', 'combined_two_token_ce', 'training_mode_accuracy',
    'fixed_transform_score_initial', 'fixed_transform_score_trained', 'training_score_change',
    'designated_score', 'designated_rank', 'candidate_mean', 'candidate_median',
    'nondesignated_mean', 'nondesignated_median', 'designated_minus_all_mean',
    'designated_minus_nondesignated_mean', 'exact_score_tie_count', 'candidate_finite',
    'candidate_total', 'fit_count', 'analysis_count', 'selected_count', 'stratum_count',
    'residual_df', 'fit_residual_df', 'analysis_residual_df', 'fit_strata', 'analysis_strata',
    'wrong_formula_satisfied_count', 'correct_formula_satisfied_count', 'retained_count',
    'wrong_formula_fraction', 'equal_n_fraction', 'pooled_numerator', 'pooled_denominator',
    'pooled_fraction', 'Gcat', 'mean', 'median', 'total_count', 'finite_count', 'undefined_count',
    'common_initial_mean', 'common_trained_mean', 'designated_minus_common_initial',
    'designated_minus_common_trained', 'designated_minus_category_mean', 'designated_minus_category_median'}


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def number(value):
    if value is None or str(value).strip() in ('', 'None', 'null', 'nan', 'NaN'):
        return None
    value = float(value)
    if not math.isfinite(value):
        raise ValueError('Unexpected infinite saved value, rather than a structural undefined cell')
    return value


def json_read(path, inputs):
    path = Path(path)
    if not path.exists():
        return None
    inputs[str(path)] = sha(path)
    return json.loads(path.read_text())


def csv_read(path, inputs):
    path = Path(path)
    if not path.exists():
        return []
    inputs[str(path)] = sha(path)
    with path.open(newline='') as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        for key in row:
            if key in NUMERIC or key.endswith('_error'):
                row[key] = number(row[key])
    return rows


def fixed_mean(values):
    return math.fsum(values) / len(values) if len(values) == 3 and all(v is not None for v in values) else None


def sign(value):
    return None if value is None else 1 if value > 0 else -1 if value < 0 else 0


def counts(rows, field):
    c = Counter(sign(r[field]) for r in rows)
    return {'positive': c[1], 'negative': c[-1], 'zero': c[0], 'undefined': c[None], 'total': len(rows)}


def formatted(value, percent=False):
    if value is None:
        return 'NaN / 未定义或不适用'
    if isinstance(value, bool):
        return '是' if value else '否'
    if isinstance(value, (float, int)):
        return f'{100 * value:.2f}%' if percent else f'{value:.6g}'
    return str(value)


def source_summary(rows):
    """Presentation-only fixed 1/3 aggregation of existing identity score rows."""
    groups = {}
    for row in rows:
        if row['action'] != 'identity':
            continue
        key = row['unit'], row['task'], row['condition'], row['split']
        groups.setdefault(key, []).append(row)
    result = []
    for (unit, task, condition, split), rr in sorted(groups.items()):
        rr.sort(key=lambda r: r['n'])
        if len({r['n'] for r in rr}) != len(rr):
            raise ValueError('Repeated saved source score for one scale')
        complete = [r['n'] for r in rr] == [6, 8, 10]
        s = dict(unit=unit, task=task, condition=condition, split=split, n_count=len(rr),
                 complete_fixed_three_scales=complete, weighting='1/3; missing scales retain undefined')
        for field in ('full_vocab_answer_accuracy', 'answer_ce', 'eos_accuracy', 'eos_ce',
                      'combined_two_token_ce', 'training_mode_accuracy'):
            s[field] = fixed_mean([r[field] for r in rr]) if complete else None
        s['minimum_n_answer_accuracy'] = min(r['full_vocab_answer_accuracy'] for r in rr) if complete and all(r['full_vocab_answer_accuracy'] is not None for r in rr) else None
        s['gain_over_train_mode_pp'] = 100 * (s['full_vocab_answer_accuracy'] - s['training_mode_accuracy']) if s['full_vocab_answer_accuracy'] is not None and s['training_mode_accuracy'] is not None else None
        result.append(s)
    return result


def initial_candidate_support(rows):
    """Describe saved raw-init scores; never infer unobserved equality.

    A common actual initialization produces task-independent pre-query states
    and query probabilities, checked by the evaluator. With fixed f/T and raw
    support, candidate initial scores should therefore coincide. Task-specific
    answer strata change both support and centering even with identical states.
    """
    groups = {}
    for row in rows:
        if row['control'] == 'raw_within_n':
            groups.setdefault((row['unit'], row['relation']), []).append(row)
    summaries = []
    for (unit, relation), rr in sorted(groups.items()):
        values = [r['fixed_transform_score_initial'] for r in rr]
        finite = [v for v in values if v is not None]
        lo, hi = (min(finite), max(finite)) if finite else (None, None)
        summaries.append(dict(unit=unit, relation=relation, control='raw_within_n',
                              expected_candidate_count=3, saved_candidate_count=len(rr), finite_count=len(finite),
                              initial_score_min=lo, initial_score_max=hi,
                              initial_score_range=hi-lo if finite else None,
                              exact_equal_complete_three=len(rr) == len(finite) == 3 and lo == hi,
                              evidence_scope='Saved three-size equal-weight initial scores only; no cache/metric recomputation.'))
    return summaries


def execution_mode(tables_complete, numerical_ok, engineering, cost_estimate, statuses):
    if tables_complete:
        return 'complete_analysis' if numerical_ok else 'completed_with_verification_caveat'
    if engineering.get('status') == 'failed':
        return 'engineering_failed'
    if cost_estimate.get('estimated_remaining_within_budget') is False:
        return 'estimated_cost_exceeds_bound'
    if any('interrupted' in status or 'budget' in status or 'stop' in status or 'failed' in status for status in statuses):
        return 'stopped_incomplete'
    return 'pending'


def model_inventory(config, root, inputs):
    rows = []
    for job in config.get('training_manifest', []):
        directory = Path(job['checkpoint_path']).parent if job.get('checkpoint_path') else root / 'models' / job['run_id']
        completed_path, partial_path = directory / 'completed.json', directory / 'partial.json'
        receipt = json_read(completed_path, inputs)
        completed_file = receipt is not None
        if receipt is None:
            receipt = json_read(partial_path, inputs)
        receipt = receipt or {}
        valid = (completed_file and receipt.get('step') == 20000 and receipt.get('formal_completed') is True
                 and receipt.get('finite_gradient_updates') == 20000 and receipt.get('run_id') == job['run_id']
                 and receipt.get('unit') == job['unit'] and receipt.get('task') == job['task'])
        rows.append(dict(run_id=job['run_id'], unit=job['unit'], task=job['task'],
                         actual_updates=receipt.get('step', 0), formal20k_completed=valid,
                         status=receipt.get('status', 'not_started'),
                         elapsed_process_wall_seconds=receipt.get('elapsed_seconds'),
                         sample_exposures=receipt.get('sample_exposures'),
                         supervised_tokens=receipt.get('supervised_tokens'),
                         initial_parameter_sha256=receipt.get('initial_parameter_sha256', job.get('initial_parameter_sha256')),
                         checkpoint_sha256=receipt.get('checkpoint_sha256'),
                         completed_receipt=str(completed_path) if completed_file else '',
                         receipt=receipt))
    return rows


def collect(diag, root):
    inputs = {}; diag, root = Path(diag), Path(root)
    protocol_path = diag / 'protocol.json'
    formal_protocol = protocol_path.exists()
    if not formal_protocol:
        protocol_path = diag / 'protocol_draft.json'
    config = json_read(protocol_path, inputs)
    if config is None:
        raise ValueError('No saved protocol or protocol draft')
    objects = {}
    for name in ('completion.json', 'execution_state.json', 'training_complete.json', 'gpu_engineering.json', 'freeze_manifest.json',
                 'gpu_budget.json', 'cost_snapshot.json', 'cost_estimate.json', 'analysis_completion.json',
                 'evaluation_completion.json', 'actual_numerical_verification.json', 'protection_start.json',
                 'protection_end.json', 'math_audit.json', 'math_audit_input_overlap.json', 'task_start.json',
                 'knn_failure_diagnostic.json', 'table_arithmetic_audit.json'):
        objects[name] = json_read(diag / name, inputs)
    diagnostic_text_path = diag / 'knn_failure_diagnostic.txt'
    objects['knn_failure_diagnostic.txt'] = diagnostic_text_path.read_text() if diagnostic_text_path.exists() else None
    if diagnostic_text_path.exists():
        inputs[str(diagnostic_text_path)] = sha(diagnostic_text_path)
    for name in ('training_complete.json',):
        if objects[name] is None:
            objects[name] = json_read(root / name, inputs)
    objects['engineering_audit.json'] = json_read(root / 'engineering/engineering_audit.json', inputs)
    objects['data_ready.json'] = json_read(root / 'data_ready.json', inputs)
    objects['cache_index.json'] = json_read(root / 'evaluation_cache/index.json', inputs)
    tables = {name: csv_read(diag / name, inputs) for name in TABLE_COUNTS}
    inventory = model_inventory(config, root, inputs)
    problems = []
    frozen = objects['freeze_manifest.json'] or {}
    if formal_protocol:
        if frozen.get('protocol_sha256') != sha(protocol_path):
            problems.append('Formal protocol does not match its saved freeze')
        for objname in ('analysis_completion.json', 'evaluation_completion.json', 'actual_numerical_verification.json'):
            obj = objects[objname]
            if obj is not None and (obj.get('protocol_sha256') != sha(protocol_path) or obj.get('source_version_sha256') != frozen.get('source_version_sha256')):
                problems.append('Protocol/source binding mismatch: ' + objname)
        for row in inventory:
            if row['formal20k_completed']:
                receipt = row['receipt']; job = next(j for j in config['training_manifest'] if j['run_id'] == row['run_id'])
                checks = dict(protocol_sha256=sha(protocol_path), source_version_sha256=frozen.get('source_version_sha256'),
                              **{k: job[k] for k in ('data_sha256', 'batch_schedule_sha256', 'initial_parameter_sha256', 'initial_checkpoint_sha256', 'starting_rng_hash')})
                if any(receipt.get(k) != expected for k, expected in checks.items()):
                    problems.append('Saved20k receipt identity mismatch: ' + row['run_id'])
    diagnostic = objects['knn_failure_diagnostic.json']
    if diagnostic is not None and (diagnostic.get('protocol_sha256') != sha(protocol_path) or
                                   diagnostic.get('original_verification_sha256') != inputs.get(str(diag / 'actual_numerical_verification.json'))):
        problems.append('kNN localization diagnostic does not bind the current original verification/protocol')
    for name, expected in TABLE_COUNTS.items():
        if tables[name] and len(tables[name]) != expected:
            problems.append(f'{name}: {len(tables[name])} rows instead of {expected}')
    analysis = objects['analysis_completion.json'] or {}
    if analysis.get('status') == 'completed':
        for filename, expected in analysis.get('output_sha256', {}).items():
            if not Path(filename).exists() or sha(filename) != expected:
                problems.append(f'Analysis table digest mismatch: {filename}')
        for filename, expected in TABLE_COUNTS.items():
            if len(tables[filename]) != expected:
                problems.append(f'Completed analysis missing {filename}')
    numerical = objects['actual_numerical_verification.json'] or {}
    num_ok = numerical.get('status') in ('passed', 'passed_with_unavailable_checks') and numerical.get('failure_count') == 0
    eval_ok = (objects['evaluation_completion.json'] or {}).get('status') == 'completed'
    all_models = len(inventory) == 8 and all(r['formal20k_completed'] for r in inventory)
    tables_complete = formal_protocol and all_models and eval_ok and analysis.get('status') == 'completed' and not problems
    full = tables_complete and num_ok
    verification_caveat = tables_complete and not num_ok
    eng = objects['engineering_audit.json'] or {}
    cost_estimate = eng.get('cost_estimate') or objects['cost_estimate.json'] or {}
    statuses = [str((objects[n] or {}).get('status', '')) for n in ('completion.json', 'execution_state.json')]
    mode = execution_mode(tables_complete, num_ok, eng, cost_estimate, statuses)
    old = {}
    for name, path in (
        ('PermWorld32', OLD / 'permworld/main32_final_cka_readout.csv'),
        ('PermWorldEncoding16', OLD / 'permworld/encoding16_decomposition.csv'),
        ('PermWorldSource64', OLD / 'permworld/source64_learning_and_baseline.csv'),
        ('GraphWorld4', OLD / 'graphworld/graph_main4_readout.csv'),
        ('GraphWorldSource4', OLD / 'graphworld/graph_source4_learning.csv')):
        old[name] = csv_read(path, inputs)
    return dict(diag=diag, root=root, config=config, formal_protocol=formal_protocol, inputs=inputs,
                objects=objects, tables=tables, inventory=inventory, problems=problems, mode=mode,
                full_analysis=full, tables_complete=tables_complete, verification_caveat=verification_caveat,
                historical=old, source_summary=source_summary(tables['source_learning.csv']),
                initial_candidate_support=initial_candidate_support(tables['pair_summary.csv']))


def results_text(data):
    mode = data['mode']; models = data['inventory']; done = sum(r['formal20k_completed'] for r in models)
    zh = [f'当前状态：{mode}；实际完整20k终点 {done}/8。']
    en = [f'Run state: {mode}; {done}/8 fixed 20,000-update endpoints are complete.']
    if not data.get('tables_complete', data['full_analysis']):
        engineering = data['objects']['engineering_audit.json'] or {}
        for problem in engineering.get('problems', []):
            zh.append('工程停止原因：' + problem); en.append('Engineering stop reason: ' + problem)
        if mode == 'estimated_cost_exceeds_bound':
            zh.append('固定方案的成本估计超过授权上限，正式跨领域检验未完成；这不是负科学结果。')
            en.append('The fixed plan exceeds the authorized cost estimate; the third-domain scientific comparison is incomplete, not a negative scientific result.')
        else:
            zh.append('没有完整正式分析时，不对关系几何、任务对排名或跨领域复现作结论。工程小样本学会仅检验实现，不证明源任务泛化。')
            en.append('Without complete formal analysis, no conclusion is drawn about relation geometry, task-pair rankings or cross-domain reproduction. Engineering overfit checks implementation rather than generalization.')
        return zh, en
    if data.get('verification_caveat'):
        numerical = data['objects']['actual_numerical_verification.json'] or {}
        failed = [r for r in numerical.get('records', []) if r['status'] == 'failed']
        zh.append('全部正式模型、缓存和固定表已生成，但独立数值核验未全部通过；本报告完整展示保存结果，不能称fully verified或scientific_analysis_complete。核验只覆盖预定第一单元R1/n6的固定64样本子集，不是全部432生产格。')
        en.append('All fixed endpoints, caches and result tables are available, but numerical verification is not fully passed. The saved results are reported with a verification caveat; scientific_analysis_complete remains false. Verification covers the prescribed first-unit R1/n6 subset of 64 inputs, not every production cell.')
        zh.append(f'保存的核验状态：{numerical.get("status_counts")}；288个分层选择核验保留原记录。预定普通CKA、Procrustes及隐藏空间kNN未记录失败；一致的未定义值仍未定义。')
        en.append(f'Saved verification counts: {numerical.get("status_counts")}; the 288 selector checks retain their original records. No failures were recorded for the planned ordinary CKA, Procrustes or hidden-state kNN checks; consistently undefined checks remain undefined.')
        for r in failed:
            label = '/'.join(str(r[k]) for k in ('unit', 'relation', 'n', 'control', 'branch', 'phase', 'role', 'metric'))
            zh.append(f'核验失败：{label}；生产值={r["production"]:.9g}，独立值={r["independent"]:.9g}，差={r["absolute_error"]:.9g}，冻结容差={r["tolerance"]:.9g}。不修改方法、数据、容差或原核验结果。')
            en.append(f'Failed verification: {label}; production={r["production"]:.9g}, independent={r["independent"]:.9g}, absolute difference={r["absolute_error"]:.9g}, frozen tolerance={r["tolerance"]:.9g}. Methods, data, tolerance and the original verification result are unchanged.')
        diagnostic = data['objects'].get('knn_failure_diagnostic.json') or {}
        if diagnostic.get('status') == 'cause_localized_original_verification_remains_failed':
            zh.append('只读诊断已重现原两条计算路径：近onehot而非完全饱和的概率向量，在约10⁻¹³量级的平方距离上，Gram相减的约2.9×10⁻¹⁵舍入误差改变邻居边界；错误分支的极小中心化舍入也影响Gram并列边界。两项分数差各对应64×10计数中的1个邻居。诊断定位了这批保存数组的数值差异，不替代冻结方法，也不把原failed改为通过。输出kNN保留限制，主CKA按预定已核验范围解读。')
            en.append('The read-only diagnosis reproduced both original arithmetic paths. For near-onehot, not exactly saturated vectors, Gram-subtraction roundoff of about 2.9e-15 at squared distances near 1e-13 reverses neighborhood boundaries; tiny preceding centering roundoff also affects a Gram boundary tie in the wrong branch. Each score discrepancy is one neighbor count out of 64×10. This localization does not replace the frozen method or turn the original failed verification into a pass.')
        else:
            zh.append('两项失败都在训练后raw输出概率kNN；未见绑定的定位诊断时，不预判原因。输出kNN相关读数保留并带限制，主CKA按其已完成预定核验范围限定解读。')
            en.append('Both discrepancies affect trained raw output-probability kNN. Without a bound saved diagnostic, their cause is not inferred. Output-kNN results retain this limitation; primary CKA claims are restricted to the completed validation scope.')
    source = [r for r in data['source_summary'] if r['condition'] == 'trained' and r['split'] == 'analysis']
    finite = [r for r in source if r['full_vocab_answer_accuracy'] is not None]
    if finite:
        lo = min(r['full_vocab_answer_accuracy'] for r in finite); hi = max(r['full_vocab_answer_accuracy'] for r in finite)
        gains = [r['gain_over_train_mode_pp'] for r in finite]
        zh.append(f'源任务：8个最终模型的恒等分析输入、全188词表答案准确率范围为 {lo:.2%}–{hi:.2%}；相对训练集固定长度众数的差为 {min(gains):+.2f}–{max(gains):+.2f} 个百分点。逐n成绩与CE、EOS全部保留；没有额外设定学习成功阈值。')
        en.append(f'On identity analysis inputs, full-vocabulary answer accuracy spans {lo:.2%}–{hi:.2%}, with gains of {min(gains):+.2f}–{max(gains):+.2f} percentage points over the train-defined per-size mode. Per-size accuracy, CE and teacher-conditioned EOS are retained without adding a mastery threshold.')
    summary = data['tables']['geometry_summary.csv']
    for control, label in (('raw_within_n', 'raw'), ('answer_strata', '答案控制')):
        all_relations = [r for r in summary if r['branch'] == 'prefix_final' and r['metric'] == 'linear_cka' and r['control'] == control]
        b, d = counts(all_relations, 'Bt'), counts(all_relations, 'Delta')
        zh.append(f'普通prefix CKA总览（{label}，全部两单元×四关系）：Bt正 {b["positive"]}/{b["total"]}、Delta正 {d["positive"]}/{d["total"]}；负/零/未定义分别见全表。正确配对相对错误的优势与相对初始化的增量分开。')
        en.append(f'Across all two-unit/four-relation prefix-state ordinary-CKA cells ({control}), Bt is positive in {b["positive"]}/{b["total"]} and Delta in {d["positive"]}/{d["total"]}. Pairing preference and increment over initialization are separate quantities.')
    main = [r for r in summary if r['branch'] == 'prefix_final' and r['metric'] == 'linear_cka' and r['control'] == 'answer_strata']
    for names, label, enlabel in ((('R1', 'R3'), '主要R1/R3', 'Primary R1/R3'), (('R2', 'R4'), '补充R2/R4', 'Supplementary R2/R4')):
        rr = [r for r in main if r['relation'] in names]
        b, d = counts(rr, 'Bt'), counts(rr, 'Delta')
        zh.append(f'{label}，答案控制后普通CKA的{len(rr)}格：Bt 正/负/零/未定义={b["positive"]}/{b["negative"]}/{b["zero"]}/{b["undefined"]}；Delta={d["positive"]}/{d["negative"]}/{d["zero"]}/{d["undefined"]}。这些是共享任务与关系的描述格数，不是独立重复。')
        en.append(f'{enlabel}: controlled ordinary-CKA Bt positive/negative/zero/undefined counts are {b["positive"]}/{b["negative"]}/{b["zero"]}/{b["undefined"]}; Delta counts are {d["positive"]}/{d["negative"]}/{d["zero"]}/{d["undefined"]}. These coupled cells are descriptive counts, not independent replications.')
    for relation in ('R1', 'R3'):
        rr = sorted([r for r in main if r['relation'] == relation], key=lambda r: r['unit'])
        components = '; '.join(f'{r["unit"]}: Bt={r["Bt"]:+.6f}, Delta={r["Delta"]:+.6f}, dc={r["dc"]:+.6f}, du={r["du"]:+.6f}' if all(r[k] is not None for k in ('Bt', 'Delta', 'dc', 'du')) else r['unit'] + ': undefined components' for r in rr)
        zh.append(f'主要{relation}的答案控制分解：{components}。')
        en.append(f'Controlled decomposition for primary {relation}: {components}.')
    f3_n10 = sorted([r for r in data['tables']['source_learning.csv'] if r['condition'] == 'trained' and r['action'] == 'identity' and r['split'] == 'analysis' and r['task'] == 'F3' and r['n'] == 10], key=lambda r: r['unit'])
    if f3_n10:
        values = '; '.join(f'{r["unit"]}: accuracy={r["full_vocab_answer_accuracy"]:.2%}, train-mode baseline={r["training_mode_accuracy"]:.2%}, CE={r["answer_ce"]:.6f}' for r in f3_n10)
        zh.append('R3的F3源在n10未充分学会，限制跨规模机制解释；其恒等分析成绩为 ' + values + '。R3正Delta仍必须与这些源学习差异一起报告，不能称正确几何普遍趋同。')
        en.append('F3 remains weak at n10, limiting the interpretation of R3 across scales: ' + values + '. Positive R3 Delta must be read alongside these source-learning differences, rather than as universal convergence of the correct geometry.')
    full_n10 = [r for r in data['tables']['source_learning.csv'] if r['condition'] == 'trained' and r['action'] == 'identity'
                and r['task'] in ('F3', 'F4') and r['n'] == 10]
    if full_n10:
        for unit in (u['unit'] for u in data['config']['units']):
            for task in ('F3', 'F4'):
                rr = {r['split']: r for r in full_n10 if r['unit'] == unit and r['task'] == task}
                values = ', '.join(f'{split}={rr[split]["full_vocab_answer_accuracy"]:.2%} (CE={rr[split]["answer_ce"]:.6f})' for split in ('train', 'validation', 'analysis') if split in rr)
                zh.append(f'n10完整源评价 {unit}/{task}：{values}。')
                en.append(f'Complete n10 source evaluation {unit}/{task}: {values}.')
        zh.append('F3的两单元和F4的第二单元在n10的完整训练集成绩也低，不能仅用验证/分析集过拟合解释；具体原因尚未识别，固定预算不追加训练。')
        en.append('At n10, both F3 units and the second F4 unit also have low full-training-set accuracy. The weakness cannot be attributed solely to validation/analysis generalization; its cause is not identified and the fixed budget is not extended.')
    rankings = [r for r in data['tables']['designated_rank.csv'] if r['phase'] == 'trained' and r['control'] == 'answer_strata']
    available = [r for r in rankings if r['designated_score'] is not None]
    first = sum(r['designated_rank'] == 1 for r in available)
    unique = sum(r['designated_rank'] == 1 and r['exact_score_tie_count'] == 1 and r['candidate_finite'] == 3 for r in available)
    above = sum(r['designated_minus_nondesignated_mean'] is not None and r['designated_minus_nondesignated_mean'] > 0 for r in available)
    zh.append(f'固定正确T、遍历其余任务的单独比较：训练后答案控制下，指定对应排名第一 {first}/{len(available)} 格，其中三候选完整且无并列的第一为{unique}格；高于另两候选均值为{above}格。平均优势与超过每个候选分开，排名不称作Bt或Delta。')
    en.append(f'For the separate fixed-T task comparison, the designated task ranks first in {first}/{len(available)} finite controlled trained cells ({unique} complete, uniquely first), and exceeds the mean of the other candidates in {above}. Candidate rankings are distinct from Bt and Delta.')
    raw_rankings = [r for r in data['tables']['designated_rank.csv'] if r['phase'] == 'trained' and r['control'] == 'raw_within_n']
    raw_available = [r for r in raw_rankings if r['designated_score'] is not None]
    raw_first = sum(r['designated_rank'] == 1 for r in raw_available)
    zh.append(f'同一固定T任务对统计的raw版本：指定对应排名第一 {raw_first}/{len(raw_available)}；答案控制下第一 {first}/{len(available)}、高于候选均值 {above}/{len(available)}。这与前面的正确-错误Bt/Delta是两类问题，不合并计数。')
    en.append(f'The raw fixed-T comparison ranks the designated task first in {raw_first}/{len(raw_available)}, versus {first}/{len(available)} after answer control and {above}/{len(available)} above the alternative mean. These are not counts of correct-versus-wrong Bt or Delta.')
    for field in ('Bt', 'Delta'):
        aligned = 0; total = 0
        for row in main:
            others = [r for r in summary if r['unit'] == row['unit'] and r['relation'] == row['relation'] and r['branch'] == 'prefix_final' and r['control'] == 'answer_strata']
            ss = [sign(r[field]) for r in others]
            if len(ss) == 3 and None not in ss:
                total += 1; aligned += len(set(ss)) == 1
        by_relation = {}
        for r in main:
            by_relation.setdefault(r['relation'], []).append(sign(r[field]))
        ufinite = [v for v in by_relation.values() if len(v) == 2 and None not in v]
        uagree = sum(v[0] == v[1] for v in ufinite)
        zh.append(f'{field}方向：三指标一致 {aligned}/{total} 个完整格；普通CKA两独立单元同向 {uagree}/{len(ufinite)} 条有限关系。近零符号没有不确定性或显著性解释，缺值不计作一致。')
        en.append(f'{field} directions agree across the three metrics in {aligned}/{total} complete cells; ordinary CKA agrees across the two units for {uagree}/{len(ufinite)} finite relations. Near-zero signs have no significance interpretation; undefined values are not agreement.')
    zh.append('两单元仅独立重复第三领域。GL输入、转置与固定左右乘法不证明学习了群乘法；这些weight公式也成立于一般二值方阵。本轮不做编码交互或层搜索，不能据此取代PermWorld的编码解释或独立复现旧领域。')
    en.append('The two units independently repeat only this third domain. GL inputs and transpose/fixed multiplication actions do not establish learned group multiplication; the weight identities also hold for general binary square matrices. This study does not test encoding interactions, search layers or independently reproduce the previous domains.')
    historical = data.get('historical', {})
    comparisons = []
    for name in ('PermWorld32', 'GraphWorld4'):
        rr = historical.get(name, [])
        if rr:
            b, d = counts(rr, 'Bt'), counts(rr, 'Delta')
            comparisons.append(f'{name}: Bt+ {b["positive"]}/{b["total"]}, Delta+ {d["positive"]}/{d["total"]}')
    b, d = counts(main, 'Bt'), counts(main, 'Delta')
    comparisons.append(f'MatrixGroupWorld8: Bt+ {b["positive"]}/{b["total"]}, Delta+ {d["positive"]}/{d["total"]}')
    zh.append('保存的各领域答案控制指定主表计数：' + '; '.join(comparisons) + '。Graph这里仅原四格答案控制，未混入答案+精确边数敏感性。共同点是Bt、Delta与控制必须分开；不同是正负分布及本轮R1的指标方向冲突。条件、关系、控制和学习程度均不同，不能把这些比例比较成跨领域效应，也不能合并为独立样本。')
    en.append('Saved designated answer-controlled endpoint counts: ' + '; '.join(comparisons) + '. Graph counts use only its original four answer-controlled cells, excluding the answer-plus-exact-edge-count sensitivity. Bt, Delta and control effects require separate interpretation in every domain; sign distributions and the present R1 metric disagreement differ. Conditions, relations, controls and source learning differ, so these proportions are neither cross-domain effect estimates nor pooled independent samples.')
    return zh, en


def table(rows, fields=None):
    if not rows:
        return '<p>尚无该项保存结果。</p>'
    fields = fields or list(dict.fromkeys(k for r in rows for k in r))
    body = '<tr>' + ''.join(f'<th>{html.escape(k)}</th>' for k in fields) + '</tr>'
    for row in rows:
        body += '<tr>' + ''.join(f'<td>{html.escape(formatted(row.get(k)))}</td>' for k in fields) + '</tr>'
    return '<div class="scroll"><table>' + body + '</table></div>'


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def saved_validation_curves(data):
    """Copy the actual twenty validations per source; never choose an endpoint.

    CE train fields are the final training batch at that step, not complete
    training-set evaluations. No Torch or feature archive is read.
    """
    rows, means, endpoint = [], [], []
    expected_steps = list(range(1000, 20001, 1000))
    validation_source = {(r['run_id'], int(r['n'])): r for r in data['tables']['source_learning.csv']
                         if r['condition'] == 'trained' and r['split'] == 'validation' and r['action'] == 'identity'}
    if not data['tables_complete']:
        return rows, means, endpoint
    for model in data['inventory']:
        receipt = model['receipt']; path = model['completed_receipt']; curve = receipt['curve']
        if [r['step'] for r in curve] != expected_steps:
            raise ValueError('Saved source curve lacks one of its twenty fixed validation points')
        binding = dict(run_id=model['run_id'], unit=model['unit'], task=model['task'],
                       completed_receipt_path=path, completed_receipt_sha256=data['inputs'][path],
                       final_checkpoint_sha256=model['checkpoint_sha256'],
                       final_parameter_sha256=receipt['parameter_sha256'],
                       initial_parameter_sha256=model['initial_parameter_sha256'],
                       protocol_sha256=receipt['protocol_sha256'], source_version_sha256=receipt['source_version_sha256'])
        for item in curve:
            val = item['validation']; per = val.get('per_n', val.get('per_length'))
            if val.get('validation_complete') is not True or sorted(r['n'] for r in per) != [6, 8, 10] or len(per) != 3:
                raise ValueError('Incomplete saved three-scale validation point')
            training = {f'last_training_batch_{k}': item.get(k) for k in ('train_combined_ce', 'train_answer_ce', 'train_eos_ce')}
            for original in per:
                row = dict(**binding, step=item['step'], n=original['n'], **training,
                           last_training_batch_scheduled_n=item.get('length'),
                           training_CE_scope='The single scheduled training batch at this update; not a full training-set evaluation.',
                           validation_source='Copied original completed.json curve.validation.per_n',
                           **{f'validation_{k}': v for k, v in original.items()})
                rows.append(row)
                if item['step'] == 20000:
                    source = validation_source[model['run_id'], original['n']]
                    for curve_field, source_field in (('answer_accuracy', 'full_vocab_answer_accuracy'),
                                                       ('eos_accuracy', 'eos_accuracy'),
                                                       ('length_majority_accuracy', 'training_mode_accuracy')):
                        difference = original[curve_field] - source[source_field]
                        endpoint.append(dict(run_id=model['run_id'], unit=model['unit'], task=model['task'], step=20000,
                                             n=original['n'], metric=curve_field, original_curve_value=original[curve_field],
                                             source_learning_value=source[source_field], signed_difference=difference,
                                             exact_equal=difference == 0., completed_receipt_sha256=binding['completed_receipt_sha256'],
                                             source_learning_sha256=data['inputs'][str(data['diag'] / 'source_learning.csv')]))
                        if difference != 0.:
                            raise ValueError('Saved20k source accuracy does not exactly reproduce the identity validation CSV')
            means.append(dict(**binding, step=item['step'], n_count=3, fixed_n_weights='1/3 each', **training,
                              training_CE_scope='Last single training batch only',
                              **{f'validation_{k}': val[k] for k in ('answer_accuracy', 'eos_accuracy', 'answer_ce', 'eos_ce', 'combined_ce', 'length_majority_accuracy')},
                              mean_source='Copied original saved equal-n validation summary, not a selected checkpoint'))
    if len(rows) != 480 or len(means) != 160 or len(endpoint) != 72:
        raise ValueError('Incomplete full source curve/endpoint comparison grid')
    write_csv(data['diag'] / 'validation_curves_per_n.csv', rows)
    write_csv(data['diag'] / 'validation_curves_equal_n.csv', means)
    write_csv(data['diag'] / 'validation_curve_endpoint_checks.csv', endpoint)
    return rows, means, endpoint


def plot_data(data):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figures = {}; diag = data['diag']
    if not data['tables_complete']:
        records = (data['objects']['engineering_audit.json'] or {}).get('overfit_records', [])
        if not records:
            return figures
        fig, axs = plt.subplots(1, 2, figsize=(12, 4))
        for record in records:
            curve = record['receipt'].get('curve', [])
            points = [(r['step'], r.get('validation')) for r in curve if r.get('validation')]
            if points:
                axs[0].plot([s for s, v in points], [v['answer_accuracy'] for s, v in points], 'o-', label=record['task'])
                axs[1].plot([s for s, v in points], [v['answer_ce'] for s, v in points], 'o-', label=record['task'])
        axs[0].set(ylabel='Full188 answer accuracy', ylim=(0, 1.02), xlabel='Engineering updates')
        axs[1].set(ylabel='Answer CE (EOS separate)', xlabel='Engineering updates')
        for ax in axs: ax.grid(alpha=.2); ax.legend()
        fig.suptitle('Engineering fixed32 n6 samples; not formal generalization')
        fig.tight_layout(); path = diag / 'engineering_curves.png'; fig.savefig(path, dpi=150); plt.close(fig)
        figures['engineering'] = str(path); return figures
    labels = [f'{u["unit"]}/{task}' for u in data['config']['units'] for task in data['config']['tasks']]
    ss = {(r['unit'] + '/' + r['task'], r['split']): r for r in data['source_summary'] if r['condition'] == 'trained'}
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    for split, offset in (('train', -.22), ('validation', 0), ('analysis', .22)):
        use = [ss.get((label, split), {}) for label in labels]
        xs = [i + offset for i in range(len(labels))]
        axs[0].bar(xs, [r['full_vocab_answer_accuracy'] if r.get('full_vocab_answer_accuracy') is not None else float('nan') for r in use], width=.2, label=split)
        axs[1].plot(range(len(labels)), [r.get('answer_ce') if r.get('answer_ce') is not None else float('nan') for r in use], 'o-', label=split)
    mode = [ss.get((label, 'analysis'), {}).get('training_mode_accuracy') for label in labels]
    axs[0].plot(range(len(labels)), mode, 'kx', label='Train-defined size mode on analysis')
    axs[0].set(ylabel='Answer accuracy, equal n weights', ylim=(0, 1.04)); axs[1].set(ylabel='Answer CE, equal n weights')
    for ax in axs: ax.set_xticks(range(len(labels)), labels, rotation=35, ha='right'); ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.suptitle('All eight fixed20k models; identity inputs; EOS reported separately')
    fig.tight_layout(); path = diag / 'source_learning.png'; fig.savefig(path, dpi=150); plt.close(fig); figures['source'] = str(path)
    curves = data.get('validation_curves', [])
    if curves:
        fig, axs = plt.subplots(2, 4, figsize=(18, 8), sharex=True, sharey=True)
        colors = {6: '#3973a6', 8: '#d17c22', 10: '#399660'}
        for ax, label in zip(axs.flat, labels):
            unit, task = label.split('/')
            for n in (6, 8, 10):
                rr = sorted([r for r in curves if r['unit'] == unit and r['task'] == task and r['n'] == n], key=lambda r: r['step'])
                ax.plot([r['step'] / 1000 for r in rr], [r['validation_answer_accuracy'] for r in rr], 'o-', ms=2,
                        lw=1.1, color=colors[n], label=f'n{n}')
                if rr:
                    ax.axhline(rr[0]['validation_length_majority_accuracy'], ls='--', lw=.7, color=colors[n], alpha=.6,
                               label=f'n{n} train-mode baseline')
            ax.set_title(label); ax.grid(alpha=.2); ax.set(ylim=(0, 1.03), xlabel='Saved update / 1000', ylabel='Full188 validation answer accuracy')
        axs[0, 0].legend(fontsize=7, loc='lower right')
        fig.suptitle('All eight sources / all three sizes / all 20 fixed checkpoints; validation only, no checkpoint selection')
        fig.tight_layout(); path = diag / 'validation_accuracy_curves.png'; fig.savefig(path, dpi=150); plt.close(fig); figures['validation_curves'] = str(path)
    summary = data['tables']['geometry_summary.csv']; cells = [(u['unit'], r['relation']) for u in data['config']['units'] for r in data['config']['relations']]
    clabel = [u + '/' + r for u, r in cells]
    fig, axs = plt.subplots(2, 3, figsize=(16, 8), sharex=True)
    for ci, control in enumerate(CONTROLS):
        rr = {(r['unit'], r['relation']): r for r in summary if r['metric'] == 'linear_cka' and r['branch'] == 'prefix_final' and r['control'] == control}
        for ax, fields, title in ((axs[ci, 0], ('B0', 'Bt'), 'Pairing preference'),
                                 (axs[ci, 1], ('dc', 'du'), 'Correct/wrong score changes'),
                                 (axs[ci, 2], ('Delta',), 'Training increment')):
            for field in fields:
                ax.plot(range(8), [rr[k][field] if rr[k][field] is not None else float('nan') for k in cells], 'o-', label=field)
            ax.axhline(0, color='k', lw=.6); ax.set_title(control + ': ' + title, fontsize=10); ax.legend(); ax.grid(alpha=.2)
            ax.set_xticks(range(8), clabel, rotation=35, ha='right')
    fig.suptitle('Ordinary CKA: B0, Bt, dc, du and Delta kept separate; all four relations / two units')
    fig.tight_layout(); path = diag / 'geometry_decomposition.png'; fig.savefig(path, dpi=150); plt.close(fig); figures['decomposition'] = str(path)
    fig, axs = plt.subplots(2, 3, figsize=(16, 8), sharex=True)
    for ci, control in enumerate(CONTROLS):
        for mi, metric in enumerate(METRICS):
            ax = axs[ci, mi]
            rr = {(r['unit'], r['relation']): r for r in summary if r['metric'] == metric and r['branch'] == 'prefix_final' and r['control'] == control}
            for field in ('Bt', 'Delta'):
                ax.plot(range(8), [rr[k][field] if rr[k][field] is not None else float('nan') for k in cells], 'o-', label=field)
            ax.axhline(0, color='k', lw=.6); ax.set_title(control + ': ' + metric, fontsize=10); ax.grid(alpha=.2); ax.legend()
            ax.set_xticks(range(8), clabel, rotation=35, ha='right')
    fig.suptitle('Fixed sensitivity metrics; independent axes; gaps are undefined, never zero-filled')
    fig.tight_layout(); path = diag / 'metric_sensitivity.png'; fig.savefig(path, dpi=150); plt.close(fig); figures['sensitivity'] = str(path)
    pairs = data['tables']['pair_summary.csv']; fig, axs = plt.subplots(2, 2, figsize=(15, 8), sharex=True)
    for ci, control in enumerate(CONTROLS):
        for pi, phase in enumerate(('initial', 'trained')):
            ax = axs[ci, pi]
            for ix, key in enumerate(cells):
                rr = [r for r in pairs if (r['unit'], r['relation']) == key and r['control'] == control]
                for j, row in enumerate(rr):
                    value = row['fixed_transform_score_' + phase]
                    if value is not None:
                        ax.scatter(ix + (j - 1) * .13, value, marker='*' if row['category'] == 'designated' else 'o',
                                   s=100 if row['category'] == 'designated' else 22, color='#bd3434' if row['category'] == 'designated' else '#3973a6')
            ax.set_title(control + ': ' + phase); ax.set_ylabel('Fixed-T ordinary CKA'); ax.grid(alpha=.2)
            ax.set_xticks(range(8), clabel, rotation=35, ha='right')
    fig.suptitle('All 24 candidate task pairs: red star designated, blue circles other tasks; not Bt/Delta')
    fig.tight_layout(); path = diag / 'fixed_transform_candidates.png'; fig.savefig(path, dpi=150); plt.close(fig); figures['candidates'] = str(path)
    return figures


def build(diag=DEFAULT_DIAG, root=DEFAULT_ROOT, plots=True):
    data = collect(diag, root)
    validation_curves, validation_means, validation_checks = saved_validation_curves(data)
    data['validation_curves'] = validation_curves
    zh, en = results_text(data); figpaths = plot_data(data) if plots else {}
    diag = data['diag']; objects = data['objects']; tables = data['tables']; config = data['config']
    sections = []
    def section(title, content): sections.append('<h2>' + html.escape(title) + '</h2>' + content)
    section('当前结论', ''.join('<p>' + html.escape(p) + '</p>' for p in zh))
    section('English results', ''.join('<p>' + html.escape(p) + '</p>' for p in en))
    section('执行范围与固定解释', '<p>GL(n,F2)，n=6/8/10，四个单任务、两独立数据与初始化单元；8模型各20k，Transformer256维4层8头，共用188词表、row-major完整二值编码、统一查询token。主位置是输入结束、查询之前的final norm。R1/R3为主要，R2/R4为预定补充。转置是反自同构，左右乘是群作用；不称为群同态。只覆盖转置与固定作用两类关系。</p><p>正式160,000更新与工程更新分别记录：四任务小样本工程至多500/任务，恢复工程0→32加0→16→32共64更新，不是额外正式模型。旧Perm32与Graph4表、原保护产物不重写。</p><p>B0=S正确初始−S错误初始；Bt=S正确训练−S错误训练；dc和du分别是正确、错误分支变化；Delta=Bt−B0=dc−du。正Delta不证明正Bt，也不证明正确分支上升。相同初始权重不保证不同变换输入的初始几何相同。未定义值原样保留，不填0、不重分配三个规模的1/3权重。</p><p>答案控制使用各任务对自己的 joint[n,left(identity),right(identity),right(correct),right(wrong)]≥3 分层并组内减均值。不同候选的有效样本可能不同，不能将残差几何差异当成数学关系纯因果效应。输出概率为完整188词表softmax；EOS在给定真答案的条件下单独计分。隐藏Delta不能减输出Delta来估计额外机制。真值onehot是静态基线，不存在初始化或训练增量。</p>')
    shown_models = [{k: v for k, v in r.items() if k != 'receipt'} for r in data['inventory']]
    section('实际模型终点与缺失', table(shown_models))
    if validation_curves:
        section('源准确率曲线与终点核对', '<p>全部8模型×20保存点×3规模，共480原始验证行；另保留160行原始等权均值。未选择最佳点、未重复前向。20k的答案准确率、EOS准确率和训练众数基线共72项与source_learning.csv的validation identity行完全一致。CSV绑定每模型receipt、终点权重与参数SHA。训练CE列是当时单个训练batch的记录，不能当完整训练集拟合程度；完整训练集终点评价仍见source_learning.csv。</p><p><a href="validation_curves_per_n.csv">480逐step/逐n验证行</a> · <a href="validation_curves_equal_n.csv">160等权验证摘要</a> · <a href="validation_curve_endpoint_checks.csv">72项终点准确率精确核对</a></p>')
    initial_cache = objects['cache_index.json'] or {}
    clone_evidence = dict(cache_index_status=initial_cache.get('status'),
                         initial_hidden_and_answer_probabilities_task_independent_bitwise=initial_cache.get('initial_hidden_and_answer_probabilities_task_independent_bitwise'),
                         evidence_source='Saved evaluator index metadata, not a new array comparison',
                         raw_initial_candidate_scores=data['initial_candidate_support'])
    section('克隆初始化与候选样本支持', '<p>同一单元四个specialist的真实初始化相同，评估器会逐值核对任务前prefix与答案概率缓存一致。固定左f、T和raw输入样本时，初始化候选CKA应相等；下表只展示已保存分数的实际min/max/range，尚无CSV时不假定核验完成。答案分层的初始化候选可以不同，因为right-specific joint groups、保留样本与组内中心化不同；这属于控制和样本支持差异，不能解释为初始模型之间有任务几何差异。</p><p>Within each unit, exact cloned initial models have task-independent pre-query states and query probabilities, with bitwise checks recorded by the evaluator. Fixed-f/T raw initial candidate scores should coincide. Answer-controlled scores can differ through right-specific sample support and centering, even for identical initial states; this does not establish task-specific initial model geometry.</p><pre>' + html.escape(json.dumps(clone_evidence, ensure_ascii=False, indent=2)) + '</pre>')
    engineering = objects['engineering_audit.json'] or {}
    eng_rows = []
    for task in config['tasks']:
        record = next((r for r in engineering.get('overfit_records', []) if r['task'] == task), None)
        receipt = (record or {}).get('receipt', {}); val = receipt.get('final_validation') or {}
        eng_rows.append(dict(task=task, attempted=record is not None, passed=(record or {}).get('passed'),
                             actual_updates=receipt.get('step'), full188_answer_accuracy=val.get('answer_accuracy'),
                             answer_ce=val.get('answer_ce'), teacher_conditioned_eos_accuracy=val.get('eos_accuracy'),
                             EOS_ce=val.get('eos_ce'), combined_two_token_ce=val.get('combined_ce')))
    section('工程门槛与恢复', table(eng_rows) + '<p>固定32个n6工程样本，各任务至多500真实学习率更新；未执行项保持缺失。失败后停止不是正式负结果。连续0→32与独立进程0→16→32恢复检查及预算估算：</p><pre>' + html.escape(json.dumps({k: engineering.get(k) for k in ('status', 'problems', 'resume', 'cost_estimate', 'elapsed_seconds')}, ensure_ascii=False, indent=2)) + '</pre>')
    if objects.get('knn_failure_diagnostic.json') is not None:
        section('输出kNN核验差异的只读定位', '<p>定位诊断并不替代冻结生产指标，也不改变原failed状态。原始64行的Gram与直接距离路径及边界舍入证据见 <a href="knn_failure_diagnostic.json">JSON</a>、<a href="knn_failure_diagnostic.txt">诊断正文</a>；两文件SHA已绑定在读者报告审计中。</p><pre>' + html.escape(objects.get('knn_failure_diagnostic.txt') or '尚无诊断正文') + '</pre>')
    if objects.get('table_arithmetic_audit.json') is not None:
        audit = objects['table_arithmetic_audit.json']
        section('独立保存表算术审查', '<p>这项审查只核对已有CSV的算术、分类、覆盖与哈希，不重新计算激活指标，也不覆盖原输出kNN两项失败。</p><pre>' + html.escape(json.dumps({k: audit.get(k) for k in ('status', 'checks_total', 'failure_count', 'maximum_absolute_differences', 'candidate_categories', 'scope', 'existing_independent_numerical_verification_status', 'existing_independent_numerical_verification_failure_count')}, ensure_ascii=False, indent=2)) + '</pre>')
    section('实际成本、上限与未知口径', '<pre>' + html.escape(json.dumps({k: objects[k] for k in ('gpu_budget.json', 'cost_snapshot.json', 'gpu_engineering.json')}, ensure_ascii=False, indent=2)) + '</pre><p>授权GPU阶段总墙钟4小时、GPU阶段之外CPU外部墙钟2小时、新产物表观20GiB。各模型elapsed_seconds是包含验证/保存的进程墙钟；不能当作GPU活跃时间，也不能与总体GPU阶段墙钟再相加。CPU与GPU阶段时长不相加称GPU小时。实际活跃kernel时长未知，不能填0；尺寸以根快照去重口径为准。</p>')
    section('数学、轨道、身份与保护', '<pre>' + html.escape(json.dumps({'math': objects['math_audit.json'],
      'audit_input_actual_overlap': objects['math_audit_input_overlap.json'],
      'data_summary': {k: (objects['data_ready.json'] or {}).get(k) for k in ('status', 'cross_unit_unique_orbit_overlap', 'historical_exclusions', 'elapsed_process_wall_seconds')},
      'protection_start_status': (objects['protection_start.json'] or {}).get('status'),
      'protection_end_status': (objects['protection_end.json'] or {}).get('status'),
      'formal_protocol_saved': data['formal_protocol'], 'actual_numerical': objects['actual_numerical_verification.json']}, ensure_ascii=False, indent=2)) + '</pre><p>H为前两坐标嵌入GL2，全部LAR / LAᵀR闭合轨道最多72成员；正式划分按packed canonical轨道去重。接受canonical轨道不宣称等概率。旧矩阵检索仅覆盖已定位元数据与明确列出的档案，不宣称全盘历史排除。四模型共享本单元真实step0和起始RNG；单元间初始化独立，具体权重身份由协议、保存哈希与根审计绑定。</p>')
    if data['problems']:
        section('报告输入问题', '<pre>' + html.escape('\n'.join(data['problems'])) + '</pre>')
    for name, path in figpaths.items():
        section('固定图：' + name, '<img src="' + html.escape(Path(path).name) + '" alt="' + name + '">')
    if data['tables_complete']:
        geometry = tables['geometry_summary.csv']
        for relation_names, title in ((('R1', 'R3'), '主要关系R1/R3：所有分支原值与拆解'), (('R2', 'R4'), '補充关系R2/R4：所有分支原值与拆解')):
            section(title, table([r for r in geometry if r['relation'] in relation_names]))
        section('完整逐规模结果', table(tables['geometry_per_n.csv']))
        section('源学习：固定1/3 identity摘要', table(data['source_summary']))
        section('源学习：全部816行，保留所有action和EOS', table(tables['source_learning.csv']))
        section('固定T的24任务对：两控制、初始化和训练后', table(tables['pair_summary.csv']) + table(tables['pair_per_n.csv']))
        section('指定排名、所有候选分布与共同支持', table(tables['designated_rank.csv']) + table(tables['category_summary.csv']) + table(tables['designated_category_comparisons.csv']) + table(tables['common_support_training_comparisons.csv']) + '<p>other_known_exact固定类别为空时保持空；not_designated不证明不存在其他数学关系。Gcat是固定候选共同支持上的相对变化，不能称作原Delta；指定高于均值不等于高于每一候选。</p>')
        section('错误公式偶然一致与控制有效样本', table(tables['W_actual_per_n.csv']) + table(tables['W_actual_summary.csv']) + table(tables['selection_masks.csv']) + '<p>原始逐anchor记录保存在W_actual_per_anchor.csv；全部偶然一致样本保留，raw与答案控制的分母不同，等权与pooled比例分开。</p>')
    historical = data['historical']; historical_counts = {}
    for name in ('PermWorld32', 'GraphWorld4'):
        rr = historical[name]
        if rr:
            historical_counts[name] = dict(rows=len(rr), Bt=counts(rr, 'Bt'), Delta=counts(rr, 'Delta'))
    section('与已保存PermWorld / GraphWorld比较', '<p>这里仅复用旧结果，未重新计算旧指标、前向或标签。领域、规模、输入编码、任务、样本分布、错误变换和控制不同；不计算跨领域相减的因果效应。新两单元仅属于MatrixGroupWorld，旧Perm/Graph仍保留其原独立性和学习限制。旧Graph weight类任务与新矩阵行列weight存在联系，GL输入与GF2混合尚不能单独证明群特有效应。</p><pre>' + html.escape(json.dumps(historical_counts, ensure_ascii=False, indent=2)) + '</pre>' +
      '<details><summary>旧Perm32主格</summary>' + table(historical['PermWorld32']) + '</details>' +
      '<details><summary>旧编码16格：Gamma0 / Gammat / GammaDelta分开</summary>' + table(historical['PermWorldEncoding16']) + '</details>' +
      '<details><summary>旧Graph4主格及源学习</summary>' + table(historical['GraphWorld4']) + table(historical['GraphWorldSource4']) + '</details>' +
      '<details><summary>旧Perm64源成绩与众数</summary>' + table(historical['PermWorldSource64']) + '</details>' +
      '<p>本轮只有一种编码，因此无新的编码Gamma；历史Delta翻转而Bt未翻转的情形不能称作学习关系偏好翻转。两新独立单元、四关系实例、三规模、三指标、24任务对不是可相加的独立样本；无显著性、等价性或功效保证。</p>')
    section('产物链接与停止边界', '<p>只读报告，不训练、不重提取、不调用任何科学指标，也不生成或读取最终分析标签。工程或预算停止时仅交付已完成范围；正式固定范围完成后停止，不扩模型、不增world、不自动上传，不改F17或旧产物。</p>' +
      '<ul>' + ''.join('<li><a href="' + html.escape(name) + '">' + html.escape(name) + '</a></li>' for name in TABLE_COUNTS if (diag / name).exists()) + '</ul>' +
      '<p>另见 <a href="table_arithmetic_audit.json">独立表算术审查</a>（与本报告分开执行，只有实际保存且读入的文件才出现在输入SHA清单中）。</p>' +
      '<pre>' + html.escape(json.dumps({'input_sha256': data['inputs']}, ensure_ascii=False, indent=2)) + '</pre>')
    document = '<!doctype html><html lang="zh"><meta charset="utf-8"><title>MatrixGroupWorld 本轮报告</title><style>body{font:15px/1.6 system-ui,sans-serif;max-width:1450px;margin:30px auto;padding:0 18px}h1,h2{color:#263b55}.scroll{overflow:auto;max-height:650px}table{border-collapse:collapse;font-size:12px;white-space:nowrap}td,th{border:1px solid #ddd;padding:5px 8px}th{background:#edf1f5;position:sticky;top:0}pre{white-space:pre-wrap;font-size:12px;overflow-wrap:anywhere;background:#f4f5f7;padding:12px}img{max-width:100%}details{margin:12px 0}</style><h1>MatrixGroupWorld：固定第三领域检验</h1><p>' + html.escape(datetime.now(timezone.utc).isoformat()) + '</p>' + ''.join(sections) + '</html>\n'
    (diag / 'report.html').write_text(document)
    txt = '\n\n'.join(zh + ['English results'] + en) + '\n\n' + '\n'.join(('范围：两世界+真实初始化，8×20k；R1/R3主要，R2/R4补充。',
      'B0、Bt、dc、du、Delta及raw/答案控制分别报告；NaN不填0、不换权重。',
      '固定T任务对排名与正确-错误Bt/Delta分开。',
      '同unit克隆初始状态下，固定f/T的raw初始化候选应相等；答案分层初始差异可来自right-specific样本支持/中心化，不能当初始模型任务几何不同。实际min/max与缓存逐值核对元信息见HTML。',
      '输出概率完整188类；EOS给定真答案。hiddenDelta-outputDelta不是机制估计。',
      '只有两独立第三领域单元，不把4关系/3n/3指标/24候选算独立重复。',
      '工程/预算未完成不包装成负科学结果。GPU阶段墙钟、CPU外部墙钟与源进程墙钟分别计账；kernel活跃时间未知。',
      '只读全部保存结果，停止后不扩训练、不发GitHub、不改旧结果或F17。',
      '完整表、工程、审计和历史范围见report.html。')) + '\n'
    (diag / 'report.txt').write_text(txt)
    result = {'status': 'report_written', 'run_mode': data['mode'], 'scientific_analysis_complete': data['full_analysis'],
              'tables_complete': data['tables_complete'], 'verification_caveat': data['verification_caveat'],
              'completed20k_models': sum(r['formal20k_completed'] for r in data['inventory']),
              'formal_models_planned': 8, 'units': 2, 'input_table_row_counts': {k: len(v) for k, v in tables.items()},
              'reader_input_problems': data['problems'], 'input_sha256': data['inputs'], 'reporter_sha256': sha(__file__),
              'figures': figpaths, 'no_model_calls': True, 'no_metric_calls': True, 'no_analysis_labels_read': True,
              'initial_clone_reader_evidence': clone_evidence,
              'validation_curve_export': {'per_n_rows': len(validation_curves), 'equal_n_rows': len(validation_means),
                                          'endpoint_checks': len(validation_checks), 'endpoint_all_exact_equal': all(r['exact_equal'] for r in validation_checks) if validation_checks else None,
                                          'source_receipt_sha256': {r['completed_receipt']: data['inputs'][r['completed_receipt']] for r in data['inventory'] if r['completed_receipt']},
                                          'no_model_or_forward_calls': True, 'last_training_batch_CE_is_not_full_train_CE': True},
              'outputs': {str(p): sha(p) for p in [diag / 'report.html', diag / 'report.txt', *map(Path, figpaths.values()),
                            *[diag / name for name in ('validation_curves_per_n.csv', 'validation_curves_equal_n.csv', 'validation_curve_endpoint_checks.csv') if (diag / name).exists()]]},
              'zh_results': zh, 'en_results': en, 'next_training_started': False}
    (diag / 'report_audit.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    return result


def fixture(diag):
    tests = []
    def check(name, value):
        assert value, name; tests.append({'check': name, 'passed': True})
    check('undefined aggregation does not reweight', fixed_mean([1., None, 2.]) is None)
    check('fixed_three_equal_weights', fixed_mean([1., 2., 3.]) == 2.)
    check('zero is a defined value', number('0') == 0 and sign(0.) == 0)
    check('structural NaN retained', number('nan') is None)
    try: number('inf')
    except ValueError: check('unexpected infinity rejected', True)
    else: check('unexpected infinity rejected', False)
    fake = [dict(unit='unit001', task='F1', condition='trained', split='analysis', action='identity', n=n,
                 full_vocab_answer_accuracy=.2, answer_ce=3., eos_accuracy=1., eos_ce=0.,
                 combined_two_token_ce=1.5, training_mode_accuracy=.2) for n in (6, 8, 10)]
    summary = source_summary(fake)[0]
    check('equal model and mode does not become learning gain', summary['gain_over_train_mode_pp'] == 0.)
    check('missing source scale stays undefined', source_summary(fake[:2])[0]['full_vocab_answer_accuracy'] is None)
    candidates = [dict(unit='unit001', relation='R1', control='raw_within_n', fixed_transform_score_initial=.25) for _ in range(3)]
    candidates.append(dict(unit='unit001', relation='R1', control='answer_strata', fixed_transform_score_initial=.9))
    support = initial_candidate_support(candidates)
    check('raw cloned-score agreement uses only actual three saved candidates', len(support) == 1 and support[0]['exact_equal_complete_three'] and support[0]['initial_score_range'] == 0.)
    check('missing raw candidate is not exact equality evidence', not initial_candidate_support(candidates[:2])[0]['exact_equal_complete_three'])
    check('absent candidate results remain absent', initial_candidate_support([]) == [])
    check('complete tables with numerical failure are caveated rather than pending', execution_mode(True, False, {}, {}, ['failed']) == 'completed_with_verification_caveat')
    check('complete table presentation does not imply verification pass', execution_mode(True, True, {}, {}, ['completed']) == 'complete_analysis')
    check('engineering failure without formal tables stays engineering stop', execution_mode(False, False, {'status': 'failed'}, {}, []) == 'engineering_failed')
    data = {'mode': 'engineering_failed', 'inventory': [dict(formal20k_completed=False)] * 8,
            'full_analysis': False, 'objects': {'engineering_audit.json': {'problems': ['synthetic failure']}}}
    zh, en = results_text(data)
    check('failed engineering yields no scientific conclusion', '没有完整正式分析' in '\n'.join(zh))
    check('report emits both languages', bool(en) and 'Run state' in en[0])
    result = {'status': 'passed', 'count': len(tests), 'checks': tests, 'synthetic_only': True,
              'source_sha256': sha(__file__), 'no_model_or_metric_or_analysis_label_calls': True}
    (Path(diag) / 'report_fixture.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--diag', type=Path, default=DEFAULT_DIAG)
    p.add_argument('--root', type=Path, default=DEFAULT_ROOT); p.add_argument('--fixture', action='store_true')
    p.add_argument('--no-plots', action='store_true'); args = p.parse_args()
    result = fixture(args.diag) if args.fixture else build(args.diag, args.root, not args.no_plots)
    print(json.dumps({k: result[k] for k in ('status',) + (('count',) if args.fixture else ('run_mode', 'completed20k_models', 'scientific_analysis_complete'))}, ensure_ascii=False))
