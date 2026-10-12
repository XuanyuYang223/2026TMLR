"""Delivery metadata and a reader annotation; does not recompute any metric."""
from common import *
from datetime import datetime, timezone
from collections import Counter
import argparse
import html
import os
import shutil


def active_scientific_jobs():
    jobs = []
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            args = (entry / 'cmdline').read_bytes().split(b'\0')
            args = [a.decode(errors='replace') for a in args if a]
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        # Inspect actual executable/script arguments, never search our shell's text.
        scripts = [a for a in args[1:] if a.endswith('.py')]
        for script in scripts:
            if ('permworld_controlled_comparison_20261010' in script and
                    Path(script).name in ['analyze.py', 'extract_layers.py', 'source_learning.py']):
                jobs.append(dict(pid=int(entry.name), argv=args))
            if ('permworld_encoding_archdev_20261009' in script and
                    Path(script).name in ['training.py', 'train.py', 'run.py']):
                jobs.append(dict(pid=int(entry.name), argv=args))
    return jobs


def prepare():
    validation = js(DIAG / 'validation_audit.json')
    assert validation['status'] == 'passed'
    assert not active_scientific_jobs(), 'Scientific calculation still running'
    cost_path = DIAG / 'cost_estimate.json'
    original = DIAG / 'cost_estimate_before_execution.json'
    if not original.exists():
        shutil.copyfile(cost_path, original)
    cost = js(original)
    index = js(FROOT / 'layers/index.json')
    metric = js(DIAG / 'metric_review/analysis_audit.json')
    cost.update(status='completed_actual_stage_costs_with_original_estimate',
        original_estimate_path=str(original), original_estimate_sha256=sha(original),
        actual_inventory_wall_seconds=js(DIAG / 'inventory/source_index.json')['inventory_elapsed_seconds'],
        actual_setup_wall_seconds=js(DIAG / 'setup_audit.json')['elapsed_seconds'],
        actual_gpu_extraction_pipeline_wall_seconds=index['actual_extraction_wall_seconds'],
        actual_forward_and_store_seconds_sum=sum(r['forward_and_store_seconds'] for r in index['records']),
        actual_cpu_metric_pipeline_wall_seconds=metric['elapsed_cpu_wall_seconds'],
        actual_extraction_readonly_audit_wall_seconds=js(DIAG / 'extraction_review/index_reader_audit.json')['reader_audit_wall_seconds'],
        actual_validation_wall_seconds=validation['elapsed_seconds'],
        actual_new_cache_bytes=sum(Path(r[k + '_path']).stat().st_size for r in index['records']
            for k in ['features', 'lengths', 'answer_ce', 'answer_argmax'] if k + '_path' in r),
        timing_interpretation='GPU pipeline wall includes encoding, transfer and IO; CPU pipeline wall includes hashes, cache IO, preprocessing and metrics. Neither is device-kernel time or CPU-core hours. Parallel stage times must not be added as an elapsed-turn duration.',
        cuda_kernel_seconds_measured=False,
        real_costs_exclude_agent_reasoning_and_report_generation=True)
    json_write(cost_path, cost)
    manifest = rows(DIAG / 'comparison_manifest.csv')
    completion = dict(status='scientific_analysis_complete_report_pending',
        generated_utc=datetime.now(timezone.utc).isoformat(),
        primary_cohort='First16, fixed20k, n10..30, one saved initial per architecture',
        models=16, lengths=21, relation_instances=2,
        comparisons=len(manifest), comparison_status_counts=dict(Counter(r['match_status'] for r in manifest)),
        available_comparisons_computed=416,
        structurally_undefined_embedding_comparisons=48,
        missing_checkpoints=0, missing_original_inputs=0,
        per_length_rows=validation['per_length_rows'],
        condition_summary_rows=validation['condition_summary_rows'],
        cross_arch_same_task_rows=validation['cross_arch_same_task_rows'],
        source_learning_rows=4032,
        mathematically_undefined_Delta_rows=validation['undefined_Delta_rows'],
        auxiliary_missing=[dict(item='initial_answer_CE', rows=2016,
            reason='Not originally cached. Exact shared initial hidden states do not establish shared task-specific CE; left missing as declared before extraction.')],
        actual_costs=cost,
        all_scientific_calculations_ended=True, active_scientific_jobs=[],
        report_complete=False, stopped=False,
        new_training=False, optimizer_constructed=False, backward=False,
        new_samples=False, updates_added=0, GroupWorld_started=False, F17_changed=False,
        original768_confirmation_started=False, second8k_cohort_used=False,
        old_protected_files_unchanged=validation['protected_files'],
        automatic_followup_planned=False)
    json_write(DIAG / 'completion.json', completion)


def complete():
    for name in ['reader_report.html', 'four_part_main.png', 'report_export_audit.json',
                 'inventory/report_reader_review.txt', 'metric_review/final_reader_audit.json']:
        assert (DIAG / name).is_file(), name
    assert js(DIAG / 'report_export_audit.json')['status'] == 'passed'
    assert js(DIAG / 'metric_review/final_reader_audit.json')['status'] == 'passed'
    assert not active_scientific_jobs()
    result = js(DIAG / 'completion.json')
    result.update(status='completed_and_stopped', report_complete=True, stopped=True,
        all_scientific_calculations_ended=True, active_scientific_jobs=[],
        completed_utc=datetime.now(timezone.utc).isoformat(),
        stop_boundary='No new training or downstream experiment. Final HTML refresh/annotation and file-hash packaging only.')
    json_write(DIAG / 'completion.json', result)


def annotate():
    report = DIAG / 'reader_report.html'
    exported = js(DIAG / 'report_export_audit.json')
    assert sha(report) == exported['output_sha256']['reader_report.html']
    assert js(DIAG / 'completion.json')['status'] == 'completed_and_stopped'
    raw = report.read_text()
    summaries = rows(DIAG / 'contrast_summary.csv')
    lookup = {tuple(r[k] for k in ['architecture', 'encoding', 'relation', 'control', 'branch', 'layer', 'metric']):r
        for r in summaries if r['contrast_type'] == 'condition_summary'}
    def cell(a,e,r,c='answer_strata',m='linear_cka'):
        return lookup[a,e,r,c,'hidden','final_norm',m]
    tf_inv = cell('transformer','one_line','descent_inverse')
    gru_inv = cell('gru','one_line','descent_inverse')
    tf_comp = cell('transformer','one_line','peak_complement')
    raw_tf_comp = cell('transformer','one_line','peak_complement','raw_within_length')
    knn_gru_comp = cell('gru','one_line','peak_complement',m='knn_overlap')
    def f(row,key): return f'{number(row[key]):+.4f}'
    direction_rows = [r for r in summaries if r['contrast_type'] == 'metric_direction' and r['control'] == 'answer_strata']
    assert len(direction_rows) == 8
    assert sum(r['Bt_direction_consistent'] == 'True' for r in direction_rows) == 8
    assert sum(r['Delta_direction_consistent'] == 'True' for r in direction_rows) == 7
    assert number(knn_gru_comp['Delta']) < 0 < number(cell('gru','one_line','peak_complement')['Delta'])
    assert all(number(r['dw']) < 0 < number(r['dc']) for r in [tf_inv,gru_inv])
    findings = f'''<aside style="padding:14px;border:1px solid #b7cbd9;background:#f6faff"><b>这轮结果的具体读法</b>
    <p>三主指标对八个终点条件的 Bt 方向全部一致，对 Δ 方向七个条件一致。
    唯一 Δ 分歧是 GRU／一行／峰—谷：邻域重叠 Δ={f(knn_gru_comp,'Delta')}，CKA 和留出 Procrustes 为正。
    这描述方向一致性，不是显著性，也不保证21长度逐一同向。</p>
    <p>编码使两配置的初始化偏好反向：一行在初始化时偏好补排列、较不偏好取逆；轮换则相反。
    四个 CKA 编码对比中，两项 Δ 翻转但 Bt 不翻转（Transformer／峰—谷、GRU／下降—逆下降）。
    Transformer／下降—逆下降的 Bt 确实随编码由正转负。完整 Γ 分解见下表，不能只用 Δ 叙述学习偏好。</p>
    <p>一行／取逆的正 Δ 主要来自错误配对下降：Transformer 的 dc={f(tf_inv,'dc')}、dw={f(tf_inv,'dw')}，
    GRU 的 dc={f(gru_inv,'dc')}、dw={f(gru_inv,'dw')}。它们支持配对选择性被重塑，不能描述为正确几何大幅趋同。</p>
    <p>层也影响结论：GRU／一行／取逆、GRU／轮换／补排列在低层的 Bt 为负、后层为正；其余六条件的逐长度汇总 Bt 跨层同号。
    答案控制影响增量解释：Transformer／一行／峰—谷的 raw CKA Δ={f(raw_tf_comp,'Delta')}，
    答案分层后 Δ={f(tf_comp,'Delta')}。U-centered CKA 在八个最终主条件中保留普通 CKA 的 Bt、Δ 方向；这不排除输出关联。</p>
    <p>轮换源任务仍接近训练标签固定的长度基线，一行任务掌握程度也随配置和任务不同。
    因此这里支持固定预算下编码、配置和层的条件依赖；一个世界、每配置一个初始化、两个关系实例，仍不能认定普适数学机制。</p></aside>'''
    annotation = '''<aside style="padding:14px;border:1px solid #b7cbd9;background:#f6faff"><b>交付口径与审计附注</b>
    <p>本轮已完成并停止。没有新增训练、更新数或样本；F17 和原结果保持不变。</p>
    <p>初始答案 CE 的 2,016 项为明确缺失，原因是原缓存未保存该指标；共享隐藏初始化不代表共享任务 logits。最终模型答案 CE 已按全词表计算。
    原 <a href="source_validation_eos_original.csv">16 模型验证集答案／EOS teacher-forced 成绩</a>单独保留；新 prompt-only 答案准确率与原答案成绩逐项一致，EOS 不作为任务学会的证据。</p>
    <p>输出概率几何复用原 0..30 数字条件分布，不等同全词表概率；答案 CE 使用完整词表，两者口径分开。
    嵌入层在固定长度的逻辑结束 token 上为常量，因此相似度未定义，不能解读为零效应。</p>
    <p>独立核查：<a href="metric_review/final_reader_audit.txt">完整表读者核查</a>、
    <a href="extraction_review/index_reader_audit.txt">新层缓存逐值／哈希核查</a>、
    <a href="inventory/report_reader_review.txt">报告核查</a>。
    <a href="inventory/model_inventory.csv">实际16模型与初始化清单</a>、
    <a href="inventory/match_audit.txt">固定项审计</a>及
    <a href="cross_arch_same_task_summary.csv">四任务跨配置相似度汇总</a>均单独保留。
    小批修复仅将独立中层验证改为相同 full-prompt/batch 形状；原 2e−5 容差、提取公式和主结果未改变。
    <a href="extraction_review/engineering_repair.json">修复记录</a>保留先前失败；prefix-only 中层数值差异也完整保留。
    <a href="REPRODUCE.txt">复现说明</a>列出真实数据、初始化、冻结版本、大缓存及命令。</p></aside>'''
    needle = '<h1>PermWorld 受控表征比较</h1>'
    assert raw.count(needle) == 1
    shutil.copyfile(report, DIAG / 'reader_report_frozen_writer.html')
    report.write_text(raw.replace(needle, needle + findings + annotation))
    json_write(DIAG / 'reader_annotation_audit.json', dict(status='passed',
        frozen_report_writer_sha256=exported['reporter_sha256'],
        original_writer_report_sha256=exported['output_sha256']['reader_report.html'],
        preserved_writer_report_sha256=sha(DIAG / 'reader_report_frozen_writer.html'),
        annotated_report_sha256=sha(report),
        annotation_helper_sha256=sha(__file__), metric_values_changed=False,
        actual_main_Bt_direction_agreement=8, actual_main_Delta_direction_agreement=7,
        source_CE_missing_rows=2016, teacher_forced_EOS_separate=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['prepare', 'complete', 'annotate'])
    args = parser.parse_args()
    {'prepare': prepare, 'complete': complete, 'annotate': annotate}[args.stage]()
    print(args.stage, 'passed', flush=True)
