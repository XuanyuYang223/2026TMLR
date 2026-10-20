"""Independent English full-training/validation report; no training, test reading, or geometry.

First run CPU synthetic verification with --fixture. Actual results are read only with explicit --root.
All 16 fixed models, both splits, and three lengths are shown; descriptive categories do not change historical joint qualification.
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


ROOT = Path('/mnt/yangx-projects-f/20261010/permworld_fullset_eval')
OUTPUT = Path(__file__).resolve().parent
TASKS = ('descents', 'recoils', 'peaks', 'valleys')
ARCHITECTURES = ('transformer', 'gru')
ENCODINGS = ('one_line', 'canonical_cycle')
SPLITS = ('train', 'validation')
LENGTHS = (10, 11, 12)
FIELDS = ('answer_accuracy', 'answer_ce', 'eos_accuracy', 'eos_ce', 'combined_ce')
NAMES = dict(descents='Descents', recoils='Recoils', peaks='Peaks', valleys='Valleys',
             transformer='Transformer', gru='GRU', one_line='One-line notation',
             canonical_cycle='Canonical-cycle notation', train='Full training set', validation='Full validation set')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                    allow_nan=False) + '\n')


def read_csv(path, by_length=False):
    with Path(path).open(newline='') as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row['samples'] = int(row['samples'])
        if by_length:
            row['length'] = int(row['length'])
        for field in FIELDS:
            row[field] = float(row[field])
            assert math.isfinite(row[field]), ('Nonfinite evaluation metric', field, row['run_id'])
        assert row['samples'] > 0
        assert 0 <= row['answer_accuracy'] <= 1 and 0 <= row['eos_accuracy'] <= 1
        assert all(row[field] >= 0 for field in ('answer_ce', 'eos_ce', 'combined_ce'))
        assert abs(row['combined_ce'] - (row['answer_ce'] + row['eos_ce']) / 2) <= 1e-6
    return rows


def condition(row):
    return row['architecture'], row['encoding'], row['task'], row['split']


def description(train, validation, protocol):
    thresholds = protocol['diagnostic_learning_thresholds']
    mean_min = float(thresholds['mean_answer_accuracy_min'])
    length_min = float(thresholds['per_length_answer_accuracy_min'])
    train_ok = train['answer_accuracy'] + 1e-10 >= mean_min and train['minimum_length_accuracy'] + 1e-10 >= length_min
    val_ok = validation['answer_accuracy'] + 1e-10 >= mean_min and validation['minimum_length_accuracy'] + 1e-10 >= length_min
    if train_ok and val_ok:
        category = 'Both sufficient: mastery in this condition'
    elif train_ok:
        category = 'Training sufficient, validation insufficient: candidate generalization deficit'
    elif not val_ok:
        category = 'Both insufficient: candidate fitting/optimization deficit'
    else:
        category = 'Training insufficient, validation sufficient: reversed discrepancy requiring cautious interpretation'
    return dict(training_sufficient=train_ok, validation_sufficient=val_ok,
                category=category, descriptive_only=True)


def collect(root, output):
    root, output = Path(root), Path(output)
    protocol_path = root / 'protocol.json'
    if not protocol_path.exists():
        protocol_path = output / 'protocol.json'
    protocol = json.loads(protocol_path.read_text())
    assert protocol['diagnostic_learning_thresholds']['mean_answer_accuracy_min'] == .90
    assert protocol['diagnostic_learning_thresholds']['per_length_answer_accuracy_min'] == .85
    summary_path, length_path = root / 'metrics_summary.csv', root / 'metrics_by_length.csv'
    summary = read_csv(summary_path)
    lengths = read_csv(length_path, True)
    keys = {(a, e, t, s) for a in ARCHITECTURES for e in ENCODINGS for t in TASKS for s in SPLITS}
    assert len(summary) == 32 and {condition(row) for row in summary} == keys, 'All 16 models×2 splits must be retained'
    assert len(lengths) == 96 and {(condition(row), row['length']) for row in lengths} == {
        (key, n) for key in keys for n in LENGTHS}, 'All 16 models×2 splits×3 lengths must be retained'
    lookup = {condition(row): row for row in summary}
    assert len(lookup) == 32
    lookup_length = {(condition(row), row['length']): row for row in lengths}
    assert len(lookup_length) == 96
    manifest = protocol['training_manifest']
    assert len(manifest) == 16 and len({j['run_id'] for j in manifest}) == 16
    manifest_ids = {(j['architecture'], j['encoding'], j['task']): j['run_id'] for j in manifest}
    assert len(manifest_ids) == 16
    for row in summary:
        assert row['run_id'] == manifest_ids[row['architecture'], row['encoding'], row['task']], 'Evaluated model identity is not in the frozen manifest'
    for key, row in lookup.items():
        rows = [lookup_length[key, n] for n in LENGTHS]
        assert len({r['samples'] for r in rows}) == 1, 'Three lengths within each split must have equal sample counts and equal weights'
        assert row['samples'] == sum(r['samples'] for r in rows)
        assert all(r['run_id'] == row['run_id'] for r in rows)
        for field in FIELDS:
            equal = math.fsum(r[field] for r in rows) / 3
            assert abs(equal - row[field]) <= 1e-6, ('Summary is not the fixed equal-three-length aggregate', key, field)
        row['minimum_length_accuracy'] = min(r['answer_accuracy'] for r in rows)
    models = []
    for architecture in ARCHITECTURES:
        for task in TASKS:
            for encoding in ENCODINGS:
                train = lookup[architecture, encoding, task, 'train']
                validation = lookup[architecture, encoding, task, 'validation']
                assert train['run_id'] == validation['run_id']
                models.append(dict(architecture=architecture, encoding=encoding, task=task,
                    run_id=train['run_id'], train=train, validation=validation,
                    **description(train, validation, protocol)))
    inputs = {str(p): sha(p) for p in (protocol_path, summary_path, length_path)}
    reproduction_path = root / 'validation_reproduction.csv'
    reproduction = []
    if reproduction_path.exists():
        inputs[str(reproduction_path)] = sha(reproduction_path)
        with reproduction_path.open(newline='') as f:
            reproduction = list(csv.DictReader(f))
        assert len(reproduction) == 320, 'All 16 models×4 length summaries×5 reproduction metrics must be retained'
        assert {(r['run_id'], r['length'], r['metric']) for r in reproduction} == {
            (j['run_id'], n, metric) for j in manifest for n in ('10', '11', '12', 'equal_length') for metric in FIELDS}
        for row in reproduction:
            assert abs(float(row['signed_difference']) - (float(row['recomputed']) - float(row['original']))) < 1e-12
    metadata = {}
    for name in ('evaluation_execution.json', 'prediction_manifest.json'):
        path = root / name
        if path.exists():
            inputs[str(path)] = sha(path)
            metadata[name] = json.loads(path.read_text())
    return dict(protocol=protocol, summary=summary, lengths=lengths, models=models,
                input_sha256=inputs, reproduction=reproduction,
                reproduction_path=str(reproduction_path) if reproduction_path.exists() else None,
                metadata=metadata)


def plots(result, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = dict(one_line='#2866a1', canonical_cycle='#da7532')
    output = Path(output)
    lookup = {(m['architecture'], m['task'], m['encoding']): m for m in result['models']}
    names = []
    for field, name, title in (
        ('answer_accuracy', 'answer_accuracy_all_conditions.png', 'Full-set answer accuracy: all fixed models'),
        ('answer_ce', 'answer_ce_all_conditions.png', 'Full-set answer cross-entropy: all fixed models'),
    ):
        fig, axes = plt.subplots(2, 4, figsize=(15, 7), squeeze=False)
        for i, architecture in enumerate(ARCHITECTURES):
            for j, task in enumerate(TASKS):
                axis = axes[i, j]
                for k, encoding in enumerate(ENCODINGS):
                    model = lookup[architecture, task, encoding]
                    values = [model[s][field] for s in SPLITS]
                    for x, v, split in zip((k * 3, k * 3 + 1), values, SPLITS):
                        axis.bar(x, v, color=colors[encoding], alpha=.25 if split == 'train' else 1., width=.8,
                                 hatch='//' if split == 'train' else None,
                                 label=f'{encoding}/{split}')
                        axis.text(x, v + (.025 if field == 'answer_accuracy' else max(.01, max(values) * .015)),
                                  f'{100*v:.1f}%' if field == 'answer_accuracy' else f'{v:.3f}',
                                  ha='center', fontsize=7)
                if field == 'answer_accuracy':
                    axis.axhline(.90, color='#777', linestyle='--', linewidth=.7)
                    axis.set_ylim(0, 1.13)
                else:
                    highest = max(lookup[architecture, task, e][s][field] for e in ENCODINGS for s in SPLITS)
                    axis.set_ylim(0, max(.05, highest * 1.18))
                axis.set_title(f'{architecture}/{task}', fontsize=10)
                axis.set_xticks([0, 1, 3, 4], ['line/train', 'line/val', 'cycle/train', 'cycle/val'], rotation=30, fontsize=7)
                axis.grid(axis='y', alpha=.18)
                if j == 0:
                    axis.set_ylabel('Accuracy' if field == 'answer_accuracy' else 'Cross-entropy')
        fig.suptitle(title + '\nFixed equal weights for n10/n11/n12; no test evaluation')
        fig.tight_layout(rect=(0, 0, 1, .92))
        fig.savefig(output / name, dpi=140)
        plt.close(fig)
        names.append(name)
    return names


def display(value, percentage=False):
    if isinstance(value, (int, float)):
        return f'{value*100:.3f}%' if percentage else f'{value:.6f}'
    return str(value)


def table(rows, columns):
    return '<table><thead><tr>' + ''.join('<th>' + html.escape(label) + '</th>' for _, label, _ in columns) + '</tr></thead><tbody>' + ''.join(
        '<tr>' + ''.join('<td>' + html.escape(display(row.get(field, ''), percentage)) + '</td>' for field, _, percentage in columns) + '</tr>' for row in rows) + '</tbody></table>'


def render(result, output, fixture=False):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    images = plots(result, output)
    counts = {}
    for model in result['models']:
        counts[model['category']] = counts.get(model['category'], 0) + 1
    text = [
        'Existing 16 PermWorld models: full training-set and validation-set diagnostics',
        'Generated UTC:' + datetime.now(timezone.utc).isoformat(),
        'Synthetic CPU fixture, not genuine experimental results.' if fixture else 'All evaluations use existing fixed 8k models only, without any parameter updates.',
        'Core question: did the previously low final-batch training loss correspond to sufficient full-training-set fitting?',
        'Four tasks×two architectures×two encodings=16 models; train/validation each have lengths10/11/12, totaling96 per-length cells. All lengths have fixed1/3 weights, without task/length/checkpoint selection.',
        'Answer accuracy uses full-vocabulary argmax and answer CE full-vocabulary cross entropy; EOS is teacher-forced on the true answer token, not a replacement for answer evaluation. combinedCE=(answer CE+EOS CE)/2 is separate.',
        'Final-batch training loss is retained only as a historical diagnostic clue; this round’s training performance is established by full-training-set reevaluation, not estimated from single-batch loss.',
        'Retain historical mean answer accuracy ≥90% and each length≥85% for descriptive classification; not a new qualification test, changing neither historical two-encoding joint0/8 results nor passing gates.',
        'Sufficient training/insufficient validation→candidate generalization deficit; both insufficient→candidate fitting/optimization deficit; both sufficient→mastery in this condition. None proves failure causes, mathematical-encoding mechanisms, or cross-world generalization.',
        'Sufficient training/validation describes one condition in this data world/fixed model only; it does not establish matched learning across encodings or automatically start representation evaluation/new training.',
        '', 'Descriptive category counts:' + json.dumps(counts, ensure_ascii=False, sort_keys=True),
        '', 'Complete 16-model primary table (accuracy in %, CE listed separately):',
    ]
    rows = []
    for m in result['models']:
        t, v = m['train'], m['validation']
        row = dict(run_id=m['run_id'], architecture=m['architecture'], encoding=m['encoding'], task=m['task'],
            training_answer_accuracy=t['answer_accuracy'], validation_answer_accuracy=v['answer_accuracy'],
            training_min_length_accuracy=t['minimum_length_accuracy'], validation_min_length_accuracy=v['minimum_length_accuracy'],
            training_answer_ce=t['answer_ce'], validation_answer_ce=v['answer_ce'],
            training_eos_accuracy=t['eos_accuracy'], validation_eos_accuracy=v['eos_accuracy'],
            training_eos_ce=t['eos_ce'], validation_eos_ce=v['eos_ce'],
            training_combined_ce=t['combined_ce'], validation_combined_ce=v['combined_ce'],
            category=m['category'])
        rows.append(row)
        text.append(f'{m["run_id"]}: train answer acc={t["answer_accuracy"]*100:.3f}%, min length={t["minimum_length_accuracy"]*100:.3f}%, answer CE={t["answer_ce"]:.6f}, EOS acc={t["eos_accuracy"]*100:.3f}%, EOS CE={t["eos_ce"]:.6f}, combinedCE={t["combined_ce"]:.6f}; '
                    f'validation answer acc={v["answer_accuracy"]*100:.3f}%, min length={v["minimum_length_accuracy"]*100:.3f}%, answer CE={v["answer_ce"]:.6f}, EOS acc={v["eos_accuracy"]*100:.3f}%, EOS CE={v["eos_ce"]:.6f}, combinedCE={v["combined_ce"]:.6f}; {m["category"]}.')
    text += ['', 'Complete per-length results:']
    for r in sorted(result['lengths'], key=lambda r: (ARCHITECTURES.index(r['architecture']), TASKS.index(r['task']), ENCODINGS.index(r['encoding']), SPLITS.index(r['split']), r['length'])):
        text.append(f'{r["run_id"]}/{r["split"]}/n{r["length"]},samples={r["samples"]}: answer acc={r["answer_accuracy"]*100:.3f}%,answer CE={r["answer_ce"]:.6f},EOS acc={r["eos_accuracy"]*100:.3f}%,EOS CE={r["eos_ce"]:.6f},combinedCE={r["combined_ce"]:.6f}.')
    text += ['', 'Validation reproduction table: validation_reproduction.csv retains original/new values and signed_diff; positive means new minus old, without replacing by absolute differences.',
             'Validation-reproduction CSV row count:' + str(len(result['reproduction'])) + '. Independent audits check specific values and numerical tolerances.',
             'Data/initialization/checkpoint/code-version provenance and independent audits are separate artifacts; this report loads no weights or raw predictions.',
             'This round is diagnostic, not a new confirmation experiment. One data world and one source initialization per architecture; models/lengths are not independent-world replicates, and Transformer/GRU capacities are unmatched.',
             'Stop after completion. No new training, test-label/prediction evaluation, CKA/Delta, GroupWorld, null-space, or F17 operations.']
    if 'evaluation_execution.json' in result['metadata']:
        text += ['Evaluation-execution records (time is inference, not training or GPU-utilization hours):',
                 json.dumps(result['metadata']['evaluation_execution.json'], ensure_ascii=False, indent=2)]
    if 'prediction_manifest.json' in result['metadata']:
        text += ['Raw predictions remain at F-drive paths in prediction_manifest.json; this report reads only the manifest, not prediction arrays.']
    (output / 'report.txt').write_text('\n'.join(text) + '\n')
    columns = [
        ('architecture', 'Architecture', False), ('task', 'Task', False), ('encoding', 'Encoding', False),
        ('training_answer_accuracy', 'Training answer acc', True), ('validation_answer_accuracy', 'Validation answer acc', True),
        ('training_min_length_accuracy', 'Training min-length acc', True), ('validation_min_length_accuracy', 'Validation min-length acc', True),
        ('training_answer_ce', 'Training answer CE', False), ('validation_answer_ce', 'Validation answer CE', False),
        ('training_eos_accuracy', 'Training EOS acc', True), ('validation_eos_accuracy', 'Validation EOS acc', True),
        ('training_eos_ce', 'Training EOS CE', False), ('validation_eos_ce', 'Validation EOS CE', False),
        ('training_combined_ce', 'Training combinedCE', False), ('validation_combined_ce', 'Validation combinedCE', False),
        ('category', 'Descriptive category', False)]
    length_columns = [('architecture', 'Architecture', False), ('task', 'Task', False), ('encoding', 'Encoding', False),
                      ('split', 'split', False), ('length', 'Length', False), ('samples', 'Samples', False),
                      ('answer_accuracy', 'answer acc', True), ('answer_ce', 'answer CE', False),
                      ('eos_accuracy', 'EOS acc', True), ('eos_ce', 'EOS CE', False), ('combined_ce', 'combinedCE', False)]
    main = '<h1>PermWorld full-training-set and validation-set diagnostics</h1><pre>' + html.escape('\n'.join(text[:13])) + '</pre>'
    main += '<p><a href="report.txt">Complete text and every per-length value</a> · <a href="report_audit.json">Report audit</a></p>'
    main += '<div class="wide">' + table(rows, columns) + '</div>'
    main += ''.join('<figure><img src="' + name + '" alt="Fixed figures for all conditions"></figure>' for name in images)
    main += '<h2>All96 per-length results</h2><div class="wide">' + table(result['lengths'], length_columns) + '</div>'
    if result['reproduction']:
        main += '<h2>Validation reproduction: retain signed differences</h2><div class="wide">' + table(result['reproduction'], [(k, k, False) for k in result['reproduction'][0]]) + '</div>'
    main += '<pre>' + html.escape('\n'.join(text[-7:])) + '</pre>'
    (output / 'report.html').write_text('<!doctype html><meta charset="utf-8"><title>PermWorld fullset learning diagnostic</title>'
        '<style>body{max-width:1350px;margin:25px auto;padding:0 20px;font:15px/1.55 system-ui}pre{white-space:pre-wrap}img{width:100%}.wide{overflow:auto}table{border-collapse:collapse;white-space:nowrap;font-size:12px}th,td{border:1px solid #ddd;padding:6px}th{background:#eef2f5}figure{margin:20px 0}</style>' + main)
    audit = dict(status='passed', synthetic_fixture=fixture, reporter_sha256=sha(__file__),
        input_sha256=result['input_sha256'], model_conditions=16, split_conditions=32,
        length_conditions=96, fixed_lengths=list(LENGTHS), length_weights=[1/3] * 3,
        category_counts=counts, descriptive_thresholds=result['protocol']['diagnostic_learning_thresholds'],
        original_encoding_joint_qualification_unchanged='0/8',
        validation_reproduction_rows=len(result['reproduction']), signed_differences_preserved=True,
        scope=dict(training_started=False, checkpoints_loaded=False, raw_predictions_read=False,
                   test_read=False, geometry_read=False, old_artifacts_modified=False, F17_modified=False),
        metadata_reference_files=list(result['metadata']),
        output_sha256={name: sha(output / name) for name in ('report.html', 'report.txt', *images)})
    write_json(output / 'report_audit.json', audit)
    return audit


def report(root=ROOT, output=OUTPUT):
    return render(collect(Path(root), Path(output)), Path(output))


def fixture():
    protocol = dict(diagnostic_learning_thresholds=dict(mean_answer_accuracy_min=.90,
                                                        per_length_answer_accuracy_min=.85))
    protocol['training_manifest'] = [dict(run_id=f'fixture_{a}_{e}_{t}', architecture=a,
                                         encoding=e, task=t)
                                     for a in ARCHITECTURES for e in ENCODINGS for t in TASKS]
    checks = {}
    def check(name, value):
        assert value, name
        checks[name] = True
    def measurement(acc, minimum=None):
        return dict(answer_accuracy=acc, minimum_length_accuracy=acc if minimum is None else minimum)
    check('both_sufficient', description(measurement(.95), measurement(.94), protocol)['category'].startswith('Both sufficient'))
    check('full_train_fit_validation_weak', description(measurement(.99), measurement(.4), protocol)['category'].startswith('Training sufficient'))
    check('both_weak', description(measurement(.4), measurement(.4), protocol)['category'].startswith('Both insufficient'))
    check('mean_does_not_replace_min_length_gate', not description(measurement(.99, .84), measurement(.95), protocol)['training_sufficient'])
    with tempfile.TemporaryDirectory(prefix='permworld-fullset-report-fixture-') as tmp:
        root = Path(tmp) / 'root'; out = Path(tmp) / 'out'; root.mkdir(); out.mkdir()
        write_json(root / 'protocol.json', protocol)
        summary, by_length = [], []
        for a in ARCHITECTURES:
            for e in ENCODINGS:
                for t in TASKS:
                    for s in SPLITS:
                        accuracy = .99 if s == 'train' else .4
                        row = dict(run_id=f'fixture_{a}_{e}_{t}', architecture=a, encoding=e, task=t,
                            split=s, samples=300, answer_accuracy=accuracy, answer_ce=.2 if s == 'train' else 3.,
                            eos_accuracy=1., eos_ce=.01, combined_ce=.105 if s == 'train' else 1.505)
                        summary.append(row)
                        by_length.extend(dict(row, samples=100, length=n) for n in LENGTHS)
        def csv_write(path, rows):
            with path.open('w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        csv_write(root / 'metrics_summary.csv', summary); csv_write(root / 'metrics_by_length.csv', by_length)
        reproduction = []
        for job in protocol['training_manifest']:
            for n in ('10', '11', '12', 'equal_length'):
                for metric in FIELDS:
                    before = .4 if metric == 'answer_accuracy' else 1. if metric == 'eos_accuracy' else 1.
                    signed = 1e-7 if metric == 'answer_ce' else -1e-7 if metric == 'eos_ce' else 0.
                    reproduction.append(dict(**job, length=n, metric=metric, original=before,
                        recomputed=before + signed, signed_difference=signed,
                        tolerance=1e-5, passed=True, reference_path='synthetic_not_real'))
        csv_write(root / 'validation_reproduction.csv', reproduction)
        result = collect(root, out); audit = render(result, out, fixture=True)
        check('all16_models_32_splits_96_lengths', audit['model_conditions'] == 16 and audit['split_conditions'] == 32 and audit['length_conditions'] == 96)
        check('all_conditions_are_generalization_candidates', audit['category_counts'].get('Training sufficient, validation insufficient: candidate generalization deficit') == 16)
        check('two_fixed_complete_plots', len([p for p in out.glob('*.png')]) == 2)
        check('answer_eos_combined_are_separate', all(k in (out / 'report.html').read_text() for k in ('answer CE', 'EOS CE', 'combinedCE')))
        check('old_joint_qualification_unchanged', audit['original_encoding_joint_qualification_unchanged'] == '0/8')
        check('all320_signed_reproduction_rows_retained', audit['validation_reproduction_rows'] == 320)
        check('negative_reproduction_difference_not_made_absolute', any(float(r['signed_difference']) < 0 for r in result['reproduction']))
        csv_write(root / 'metrics_by_length.csv', by_length[:-1])
        try:
            collect(root, out)
        except AssertionError:
            check('missing_length_cannot_be_silently_dropped', True)
        else:
            raise AssertionError('Incomplete fixture was accepted')
    result = dict(status='passed', reporter_sha256=sha(__file__), checks=checks,
        synthetic_cpu_only=True, real_new_results_read=False, test_read=False,
        geometry_read=False, training_started=False, fixtures_removed=True)
    write_json(OUTPUT / 'report_fixture.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--fixture', action='store_true')
    mode.add_argument('--root', type=Path)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    print(json.dumps(fixture() if args.fixture else report(args.root, args.output),
                     indent=2, ensure_ascii=False, sort_keys=True))
