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
        'The clearest encoding result in this development unit concerns the initialization starting point: answer-stratified GRU inverse'
        'CKA has one-line/canonical-cycle B0 of −0.6169/+0.8010 and Bt of +0.0675/+0.0793,'
        'with Δ of +0.6844/−0.7217. The Δ sign change does not correspond to a reversal of trained preference; the Transformer'
        'has Bt of +0.0931/−0.0320 for this relation. A single-unit result cannot establish a universal reversal.',
        'The eight canonical-cycle models have identity-test answer accuracies of approximately 30%–41%,'
        'only 0.27–1.78 percentage points above the length mode defined from training labels. Task mastery'
        'is unmatched across encodings, as are architectural parameter counts; passing the engineering pipeline cannot isolate these differences'
        'as encoding or architectural mechanisms at equal learning levels. The length baseline is an explicitly'
        'labeled post-protocol descriptive supplement that changes neither engineering gates nor training budgets.',
        'The inverse relation’s encoding ΓΔ has the same direction across both architectures, both fixed controls, all three primary metrics,'
        'and debiased CKA in this development unit, but individual conditions’ Δ signs do not all agree across controls and metrics.'
        'Output probabilities show similar encoding interactions, so output-related explanations remain possible;'
        'hidden Δ minus output Δ is not an estimate of an independent mechanism. Complete verification is in'
        'development_results_review.txt and encoding_development/reader_review.json.',
        'output_prob is a conditional probability renormalized over logits for legal numeric answers 0–30;'
        'it preserves neither total numeric-answer probability mass nor the full-vocabulary probability distribution. Full-vocabulary accuracy'
        'and numeric-conditional accuracy are reported separately; this baseline cannot exclude all output-related information.',
        'The measured state precedes the task token; the full evaluation forward pass still includes task and equals-sign suffixes.'
        'The causal mask or unidirectional GRU and an independent prefix-only check support that the suffix does not affect this state.'
        'Each model’s weights reflect the corresponding single-task training; a pre-task state does not mean the model has no task knowledge.',
        'The new development’s independent activation check covers 2304 fixed vectors; independent CKA recomputation is limited to the prespecified first'
        'relation, n=10, across all architectures/encodings/controls/branches/pairings. CSV coverage, hash, and arithmetic audits'
        'are not independent metric recomputation for every CSV cell.',
        'The frozen protocol’s “all32” is inherited wording; the actual manifest contains 16 models. Parallelism changes only'
        'execution order; each model retains one 20k update budget, and summed per-model elapsed_seconds cannot'
        'be treated as actual wall-clock time or GPU-hours.',
    ]
    marker = '<h1>PermWorld completion report for this round</h1>'
    raw = (ROOT / sources[0]).read_text()
    assert raw.count(marker) == 1
    addition = ('\n<section><h2>Measurement definitions and completion status</h2>' +
                ''.join('<p>' + note + '</p>' for note in notes) +
                '<p><a href="development_code_review.txt">Read-only review of development code</a>; '
                '<a href="development_results_review.txt">Independent verification of development results</a>; '
                '<a href="encoding_development/reader_review.json">Export and baseline verification</a>; '
                '<a href="final_stop_audit.json">Completion and stopping audit</a>.</p></section>\n')
    reader = raw.replace(marker, marker + addition, 1)
    assert reader.replace(addition, '', 1) == raw
    (ROOT / 'completion_reader_report.html').write_text(reader)
    raw_text = (ROOT / sources[1]).read_text()
    (ROOT / 'completion_reader_report.txt').write_text(
        'Supplementary measurement definitions\n\n' + '\n\n'.join(notes) + '\n\n' + raw_text)
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
