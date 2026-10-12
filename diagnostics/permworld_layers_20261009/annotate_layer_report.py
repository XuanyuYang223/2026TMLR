"""Correct a reporting attribution without changing frozen analysis outputs."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    raw = ROOT / 'report.html'
    text = raw.read_text()
    original = '源间标准差只反映共享世界中的初始化差异'
    corrected = '源间标准差描述固定任务／数据世界内的训练种子变异，包含初始化、dropout、batch／优化随机性'
    assert text.count(original) == 1
    annotated = text.replace(original, corrected)
    anchor = '<h2>范围与证据身份</h2>'
    assert annotated.count(anchor) == 1
    banner = ('<p class="lead">报告文字审查：源种子不只改变初始化。此阅读版本修正该归因，'
              '完整保留冻结计算、图、表和结论。<a href="review_notes.txt">审查说明</a> · '
              '<a href="real_results_review.txt">真实结果独立复核</a> · '
              '<a href="statistical_review.json">汇总独立复算</a> · '
              '<a href="existing_final_audit.json">覆盖与F17保护审计</a>。</p>')
    annotated = annotated.replace(anchor, banner + anchor)
    target = ROOT / 'reader_report.html'
    target.write_text(annotated)
    audit = dict(created_utc=datetime.now(timezone.utc).isoformat(),
                 original_sha256=hashlib.sha256(raw.read_bytes()).hexdigest(),
                 annotated_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                 correction=dict(original=original, corrected=corrected),
                 numeric_outputs_changed=False, frozen_analysis_code_changed=False,
                 original_report_preserved=True)
    (ROOT / 'report_annotation_audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(audit, ensure_ascii=False))


if __name__ == '__main__':
    main()
