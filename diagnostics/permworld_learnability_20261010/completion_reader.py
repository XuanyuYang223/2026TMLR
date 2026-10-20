"""CPU-only Chinese reading companion to the frozen validation report."""
import csv
import html
import json
from pathlib import Path

D = Path(__file__).resolve().parent
ROOT = Path('/mnt/yangx-projects-f/20261010/permworld_learnability')


def main():
    models = list(csv.DictReader((D / 'model_learning_summary.csv').open()))
    pairs = list(csv.DictReader((D / 'pair_learning_gates.csv').open()))
    protocol = json.loads((D / 'protocol.json').read_text())
    provenance = json.loads((D / 'completion_audit.json').read_text())
    independent = json.loads((D / 'independent_learning_audit.json').read_text())
    assert len(models) == 16 and len(pairs) == 8
    assert all(int(r['actual_step']) == 8000 and r['validation_complete'] == 'True' for r in models)
    assert provenance['status'] == independent['status'] == 'passed'
    model_pass = sum(r['gate_pass'] == 'True' for r in models)
    pair_pass = sum(r['gate_pass'] == 'True' for r in pairs)
    lookup = {(r['architecture'], r['encoding'], r['task']): r for r in models}
    columns = [('transformer', 'one_line'), ('transformer', 'canonical_cycle'),
               ('gru', 'one_line'), ('gru', 'canonical_cycle')]
    tasks = dict(descents='Descents', recoils='Recoils', peaks='Peaks', valleys='Valleys')
    title = 'PermWorld learnability development: completed, joint learning qualification failed'
    paragraphs = [
        f'After existing-model diagnostics, this round ran one fixed development scheme: lengths10,11,12, four tasks, two encodings, two architectures,16 models; every model completed8000 updates, with1024 validation permutations/length. Actual GPU-stage wall clock was{provenance["execution_wall_seconds"]/60:.2f} minutes, including restoration engineering, validation, and saving, below120 minutes; not GPU-kernel utilization timing.',
        f'Learning qualification:{model_pass}/16; joint encoding/task-pair qualification:{pair_pass}/8. Completed training and failed qualification are different: no missing artifacts or budget interruption, but no setting with sufficient, similar learning across encodings. No independent geometry-confirmation manifest, new CKA, GroupWorld, or further training was started.',
        'All table accuracies use full-vocabulary answer argmax on identity-validation inputs, calculated per length then averaged with 1/3 weights. Length modes are fixed from training labels, not validation modes. EOS is teacher-forced given the true answer, at 99.97%—100% this round, and does not count as answer-task mastery.',
    ]
    header = ['Task', 'Length baseline', 'Transformer one-line', 'Transformer cycle', 'GRU one-line', 'GRU cycle']
    table = []
    for task in protocol['tasks']:
        base = lookup[('transformer', 'one_line', task)]
        table.append([tasks[task], f'{100*float(base["equal_length_baseline"]):.2f}%'] +
            [f'{100*float(lookup[(a,e,task)]["equal_length_accuracy"]):.2f}%' for a,e in columns])
    paragraphs_after = [
        'Single-model qualifiers are Transformer one-line recoils/peaks and all four GRU one-line tasks. No cycle model qualifies. Required mean accuracy ≥90%, each length ≥85%, length-mode gain ≥20pp, EOS ≥99%; same-task/same-architecture encoding gaps must be ≤5pp in mean and maximum per-length difference. These development gates are not TOST or statistical-equivalence evidence.',
        'Curves suggest two failure patterns: Transformer cycle validation accuracy and answer CE mostly remain near baseline; GRU cycle final-batch combined training CE is low while validation answer CE rises to 4.09—5.22. Separation between consistently defined final-batch training combined CE and validation combined CE is compatible with fitting without generalization, but single-batch loss is not whole-training accuracy and does not establish a particular memorization mechanism.',
        'Some one-line conditions also fail to learn sufficiently: Transformer descents 56.67%, valleys 80.99%. Thus all learning deficits cannot be attributed to cycle encoding, nor current geometry differences explained solely as encoding-induced changes in mathematical-relation representations.',
        'Within architecture, encodings share permutations, mathematical labels, batches, genuinely saved step-0, and update budgets. New inputs exclude historical/previous-development full eight-element orbits; training, validation, and reserved test orbits are disjoint. Test permutations are used only for isolation auditing; training, selection, and reporting read no test labels or test scores.',
        'This round uses one development scheme but jointly changes length window, training size, learning rate, dropout, microbatch, and updates, so improvements cannot be attributed to one factor. Transformer has approximately 3.24 million parameters and GRU 1.66 million; architectures are not parameter-matched. One new data world and one initialization per architecture are not 16 independent worlds or a new confirmation experiment.',
        f'Provenance/engineering audits passed: frozen{provenance["frozen_source_files"]} source files; genuine CUDA continuous32 versus16→32 restoration across four architecture×encoding conditions gives parameter/optimizer differences0 and identical RNG;{provenance["protected_legacy_files"]} protected historical hashes match; F17/original multilayer reports remain unchanged. Loss-numeric/report-schema CPU repairs were recorded and refrozen before new training; data, models, objectives, and genuine initial weights did not change.',
        'Independent arithmetic review recomputes baselines, learning gates, and encoding gaps from saved per-length statistics; it reviews aggregation rather than rerunning model forwards. All training subprocesses ended. Current results support “joint learnability was not attained with this development configuration,” not universal unlearnability of cycle encoding.',
    ]
    links = [
        ('Complete frozen report', 'learnability_report.html'),
        ('16-model endpoints and all gates', 'model_learning_summary.csv'),
        ('Per-length, per-stage data', 'validation_length.csv'),
        ('Answer and loss curve data', 'validation_curves.csv'),
        ('8 paired qualifications', 'pair_learning_gates.csv'),
        ('All answer-accuracy curves', 'learnability_plots/equal_length_accuracy.png'),
        ('All answer-CE curves', 'learnability_plots/equal_length_answer_ce.png'),
        ('Existing-model learning diagnosis', 'old_learning_diagnosis.txt'),
        ('Complete final curve review', 'learning_curve_review.txt'),
        ('Data and initialization identity review', 'data_identity_review.txt'),
        ('Independent results review', 'independent_learning_audit.json'),
        ('Protection, checkpoint, budget, and stopping audits', 'completion_audit.json'),
        ('Frozen protocol', 'protocol.json'),
        ('Source freeze and engineering-repair records', 'freeze.json'),
    ]
    text = title + '\n\n' + '\n\n'.join(paragraphs) + '\n\n'
    text += '\t'.join(header) + '\n' + '\n'.join('\t'.join(r) for r in table)
    text += '\n\n' + '\n\n'.join(paragraphs_after)
    text += '\n\nArtifact entry points:\n' + '\n'.join(f'{name}: {path}' for name,path in links)
    text += f'\n\nLarge artifacts (weights, data, genuine initializations, logs):{ROOT}\n'
    (D / 'completion_reader_report.txt').write_text(text)
    body = ''.join('<p>'+html.escape(p)+'</p>' for p in paragraphs)
    body += '<table><thead><tr>'+''.join('<th>'+html.escape(h)+'</th>' for h in header)+'</tr></thead><tbody>'
    body += ''.join('<tr>'+''.join('<td>'+html.escape(v)+'</td>' for v in r)+'</tr>' for r in table)+'</tbody></table>'
    body += ''.join('<p>'+html.escape(p)+'</p>' for p in paragraphs_after)
    body += '<p>Artifact entry points:</p><ul>'+''.join('<li><a href="'+html.escape(path)+'">'+html.escape(name)+'</a></li>' for name,path in links)+'</ul>'
    body += '<p>Large artifacts:'+html.escape(str(ROOT))+'</p>'
    document = '<!doctype html><html lang="en"><meta charset="utf-8"><title>'+html.escape(title)+'</title>'
    document += '<style>body{max-width:1100px;margin:32px auto;font:16px/1.7 system-ui,sans-serif;padding:0 20px;color:#202632}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}td,th{padding:8px 10px;border:1px solid #cdd3dd;text-align:right}td:first-child,th:first-child{text-align:left}th{background:#eef1f6}a{color:#1f5caa}</style>'
    document += '<h1>'+html.escape(title)+'</h1>'+body+'</html>'
    (D / 'completion_reader_report.html').write_text(document)
    print(json.dumps(dict(status='written',models=16,learning_passed=model_pass,pair_passed=pair_pass),ensure_ascii=False))


if __name__ == '__main__':
    main()
