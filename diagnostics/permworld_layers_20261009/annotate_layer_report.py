"""Correct a reporting attribution without changing frozen analysis outputs."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    raw = ROOT / 'report.html'
    text = raw.read_text()
    original = 'The between-source standard deviation reflects only initialization differences within the shared world'
    corrected = 'The between-source standard deviation describes training-seed variability within fixed tasks and the shared data world, including initialization, dropout, and batch/optimization randomness'
    assert text.count(original) == 1
    annotated = text.replace(original, corrected)
    anchor = '<h2>Scope and evidential status</h2>'
    assert annotated.count(anchor) == 1
    banner = ('<p class="lead">Wording review: source seeds change more than initialization. This reader version corrects that attribution,'
              'while preserving all frozen calculations, figures, tables, and conclusions.<a href="review_notes.txt">Review note</a> · '
              '<a href="real_results_review.txt">Independent verification of actual results</a> · '
              '<a href="statistical_review.json">Independent recomputation of summaries</a> · '
              '<a href="existing_final_audit.json">Coverage and F17 protection audit</a>.</p>')
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
