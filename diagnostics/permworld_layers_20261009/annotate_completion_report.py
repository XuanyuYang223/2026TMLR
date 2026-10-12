"""Add measurement-scope notes to an independently preserved finished report.

No evaluation, training, scientific selection, or modification of source reports.
The original generated report and its audit remain byte-identical.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    audit = json.loads((ROOT / 'completion_report_audit.json').read_text())
    stop = json.loads((ROOT / 'final_stop_audit.json').read_text())
    assert audit['status'] == stop['status'] == 'passed'
    assert audit['fixture_only'] is False
    assert stop['completed_development_models'] == 16
    assert stop['active_owned_training_commands'] == []
    assert stop['formal_training_started'] is False
    sources = ['completion_report.html', 'completion_report.txt']
    for name in sources:
        assert sha(ROOT / name) == audit['output_sha256'][name]
    notes = [
        '本开发单元最清楚的编码结果来自初始化起点：答案分层的GRU取逆'
        'CKA中，一行/规范轮换的B0为−0.6169/+0.8010，Bt为+0.0675/+0.0793，'
        'Δ为+0.6844/−0.7217。Δ翻转没有对应训练后偏好翻转；Transformer'
        '该关系的Bt则为+0.0931/−0.0320。单单元结果不能概括为普遍翻转。',
        '规范轮换的八个模型在identity测试上的答案准确率约30%–41%，'
        '仅高于训练标签定义的长度众数0.27–1.78个百分点。编码间任务掌握'
        '程度不匹配，架构参数量也不匹配；工程流程通过不能把这些差异'
        '隔离为同等学习程度下的编码或架构机制。长度基线是协议后明确'
        '标记的描述性补充，不改变工程门槛或训练预算。',
        '取逆的编码ΓΔ在本开发单元的两架构、两个固定控制、三个主指标'
        '及debiased CKA中方向一致，但每个条件的Δ并非都跨控制和指标'
        '同号。输出概率也有类似编码交互，尚未排除输出相关的解释；'
        '不把隐藏Δ减输出Δ当成独立机制的估计。完整复核见'
        'development_results_review.txt及encoding_development/reader_review.json。',
        'output_prob是在合法数字答案0–30的logits内重新归一化的条件概率，'
        '没有保留数字答案的总概率质量，也不是完整词表概率。完整词表准确率'
        '与数字条件准确率分别报告，不能用这个基线排除全部输出相关信息。',
        '测量状态位于任务token之前；完整评价前向仍提供任务和等号suffix。'
        '因果掩码或单向GRU以及prefix-only独立检查支持suffix不影响该状态。'
        '各模型的权重已受对应单任务训练影响，任务前状态不等于模型不知道任务。',
        '新开发的独立激活检查覆盖2304个固定向量；独立CKA复算限于预定第一'
        '关系、n=10、全部架构/编码/控制/分支/配对。CSV覆盖、哈希和算术审计'
        '不等于每个CSV单元的独立指标复算。',
        '冻结协议中“all32”是继承的文字残留，实际清单为16模型。并行只改变'
        '执行顺序，每个模型保留一次20k更新；各模型elapsed_seconds之和不能'
        '当作实际壁钟时间或GPU小时。',
    ]
    marker = '<h1>PermWorld 本轮完成报告</h1>'
    raw = (ROOT / sources[0]).read_text()
    assert raw.count(marker) == 1
    addition = ('\n<section><h2>测量口径与完成状态</h2>' +
                ''.join('<p>' + note + '</p>' for note in notes) +
                '<p><a href="development_code_review.txt">开发代码只读审查</a>；'
                '<a href="development_results_review.txt">开发结果独立复核</a>；'
                '<a href="encoding_development/reader_review.json">导出与基线复核</a>；'
                '<a href="final_stop_audit.json">完成与停止审计</a>。</p></section>\n')
    reader = raw.replace(marker, marker + addition, 1)
    assert reader.replace(addition, '', 1) == raw
    (ROOT / 'completion_reader_report.html').write_text(reader)
    raw_text = (ROOT / sources[1]).read_text()
    (ROOT / 'completion_reader_report.txt').write_text(
        '测量口径补充\n\n' + '\n\n'.join(notes) + '\n\n' + raw_text)
    for name in sources:
        assert sha(ROOT / name) == audit['output_sha256'][name]
    result = dict(status='passed', created_utc=datetime.now(timezone.utc).isoformat(),
                  presentation_only=True, original_report_unchanged=True,
                  original_tables_byte_identical=True, training_started=False,
                  source_report_sha256={n: sha(ROOT / n) for n in sources},
                  reader_report_sha256={n: sha(ROOT / n) for n in
                                        ['completion_reader_report.html', 'completion_reader_report.txt']},
                  stop_audit_sha256=sha(ROOT / 'final_stop_audit.json'))
    (ROOT / 'completion_annotation_audit.json').write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
