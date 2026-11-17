"""Render completed closing-audit evidence and new manuscript fragments only."""
from pathlib import Path
from datetime import datetime, timezone
import csv
import hashlib
import json
import statistics as stats

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
WRITING = HERE / 'writing'


def read(name):
    return json.loads((HERE / name).read_text())


def csv_rows(path):
    with Path(path).open(newline='') as handle:
        return list(csv.DictReader(handle))


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def write_csv(path, values):
    with Path(path).open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(values[0]))
        writer.writeheader(); writer.writerows(values)


def table(header, values):
    return '\n'.join(['|' + '|'.join(header) + '|', '|' + '|'.join(['---'] * len(header)) + '|',
                      *('|' + '|'.join(map(str, row)) + '|' for row in values)])


def f(value, digits=6):
    return f'{float(value):.{digits}f}' if value != '' else 'undefined'


def sci(value):
    return f'{float(value):.4g}' if value != '' else 'undefined'


def ms(row, scientific=False):
    fmt = sci if scientific else f
    return fmt(row['mean']) + ' ± ' + fmt(row['sample_sd'])


def main():
    WRITING.mkdir(exist_ok=True)
    assert read('alignment/completion.json')['status'] == read('ed/completion.json')['status'] == 'completed'
    assert read('gradient/summary_completion.json')['status'] == 'completed'
    source_paths = [HERE / name for name in [
        'protocol.json', 'freeze.json', 'alignment/spec.json', 'alignment/by_unit.csv',
        'alignment/contrasts_by_unit.csv', 'alignment/four_unit_summary.csv', 'alignment/final_audit.json',
        'ed/spec.json', 'ed/cka_four_unit_summary.csv', 'ed/encoding_contrasts_summary.csv',
        'ed/probe_four_unit_summary.csv', 'ed/cka_per_unit.csv', 'ed/independent_verification.json',
        'gradient/spec.json', 'gradient/all_units.csv', 'gradient/four_unit_summary.csv',
        'gradient/summary_completion.json']]
    source_paths += [ROOT / 'diagnostics' / name for name in [
        'permworld_answer_controls_20261109/READER_REPORT.md',
        'permworld_local_timing_20261110/headlines/main_by_unit.csv',
        'permworld_local_timing_20261110/headlines/auxiliary_eight_cells.csv',
        'permworld_lemma_composition_20261110/main_comparison_by_unit.csv',
        'permworld_lemma_composition_20261110/README.md']]
    bindings = {str(path): sha(path) for path in source_paths}
    alignment = csv_rows(HERE / 'alignment/four_unit_summary.csv')
    routes = csv_rows(HERE / 'alignment/by_unit.csv')
    contrasts = csv_rows(HERE / 'alignment/contrasts_by_unit.csv')
    aa = {(row['support'], row['condition']): row for row in alignment}
    av = aa['full_test', 'trained_valleys']; a0 = aa['full_test', 'actual_saved_step0']
    af = aa['full_test', 'trained_fixed_points']
    d0 = aa['full_test', 'valleys_minus_step0']; df = aa['full_test', 'valleys_minus_fixed_points']
    dm = aa['full_test', 'valleys_minus_best_common_mapping_control']
    cka = csv_rows(HERE / 'ed/cka_four_unit_summary.csv')
    cc = {(r['architecture'], r['encoding'], r['cka_version'], r['effect']): r for r in cka}
    enc = csv_rows(HERE / 'ed/encoding_contrasts_summary.csv')
    ee = {(r['architecture'], r['cka_version'], r['effect']): r for r in enc}
    grad = csv_rows(HERE / 'gradient/four_unit_summary.csv')
    gg = {(r['task'], r['group'], r['sign'], r['metric']): r for r in grad}
    cells = [(task, group, str(sign)) for task in ['peaks', 'valleys']
             for group in ['main', 'matched'] for sign in [-1, 1]]
    source = ROOT / 'diagnostics/permworld_lemma_composition_20261110/main_comparison_by_unit.csv'
    composition = [r for r in csv_rows(source) if r['unit'] != 'archdev001' and r['support'] == 'all']
    comp = []
    question_names = {'longest': 'Longest monotone interval',
                      'fixed_longest': 'Longest interval containing a fixed point',
                      'turns_before_fixed': 'Turns before the first fixed point'}
    for task in sorted({r['task'] for r in composition}):
        group = [r for r in composition if r['task'] == task]
        assert len(group) == 4
        differences = [float(r['aligned_minus_same_model_twice']) for r in group]
        comp.append(dict(task=task, question=question_names.get(task, task),
                         aligned_mean=stats.mean(float(r['aligned_exact_accuracy']) for r in group),
                         baseline_mean=stats.mean(float(r['same_model_twice_exact_accuracy']) for r in group),
                         difference_mean=stats.mean(differences), difference_sample_sd=stats.stdev(differences),
                         difference_minimum=min(differences), difference_maximum=max(differences),
                         negative_units=sum(x < 0 for x in differences)))
    write_csv(HERE / 'composition_evidence_reproduced.csv', comp)
    late_rows = csv_rows(ROOT / 'diagnostics/permworld_local_timing_20261110/headlines/main_by_unit.csv')
    assert len(late_rows) == 32
    assert all(float(r['early_relative_step']) == float(r['late_relative_step']) == 0 and
               float(r['early_any_numeric_change']) == float(r['late_any_numeric_change']) == 0 for r in late_rows)
    unit_values = []
    for unit in ['unit02', 'unit03', 'unit04', 'unit05', 'archdev001']:
        values = {r['target']: r for r in routes if r['unit'] == unit and r['support'] == 'full_test'}
        differences = {r['contrast']: r for r in contrasts if r['unit'] == unit and r['support'] == 'full_test'}
        unit_values.append([unit, *[f(values[name]['auroc']) for name in
                                   ['trained_valleys', 'actual_saved_step0', 'trained_fixed_points']],
                            f(differences['valleys_minus_step0']['difference']),
                            f(differences['valleys_minus_fixed_points']['difference'])])
    alignment_table = table(['Unit', 'Trained valleys', 'Saved step-0 target', 'Trained fixed_points',
                             'Valleys − step-0', 'Valleys − fixed_points'], unit_values)
    geometry_table = table(['Expert', 'Construction', 'Sign', 'Donor-difference norm', 'Actual edit norm',
                            'Projection energy fraction'],
                           [[*key, *[ms(gg[*key, metric], True) for metric in
                                     ['donor_difference_norm', 'actual_displacement_norm', 'projection_energy_fraction']]]
                            for key in cells])
    gradient_table = table(['Expert', 'Construction', 'Sign', 'Gradient norm', 'g·actual edit',
                            'Angle (degrees)', 'Baseline margin', 'Cached actual margin change', 'Equal-norm random signed mean'],
                           [[*key, *[ms(gg[*key, metric], True) for metric in
                                     ['gradient_norm', 'g_dot_delta', 'angle_degrees', 'baseline_logit_margin', 'actual_logit_margin_change',
                                      'random_equal_norm_mean_g_dot_delta']]] for key in cells])
    residual = 'within_length_dual_answer_debiased'
    ed_table = table(['Architecture', 'Encoding', 'Initial advantage', 'Trained advantage', 'Trained range'],
                     [[arch, coding, ms(cc[arch, coding, residual, 'initial_transform_advantage']),
                       ms(cc[arch, coding, residual, 'trained_transform_advantage']),
                       f"[{f(cc[arch, coding, residual, 'trained_transform_advantage']['minimum'])}, "
                       f"{f(cc[arch, coding, residual, 'trained_transform_advantage']['maximum'])}]"]
                      for arch in ['gru', 'transformer'] for coding in ['one_line', 'canonical_cycle']])
    ed_contrasts = table(['Architecture', 'Version', 'Cycle − one-line', 'Positive units /4'],
                         [[arch, version, ms(ee[arch, version, 'trained_transform_advantage_cycle_minus_one_line']),
                           ee[arch, version, 'trained_transform_advantage_cycle_minus_one_line']['positive_units']]
                          for arch in ['gru', 'transformer'] for version in
                          ['pooled_raw', 'within_length_debiased', residual]])
    gradient_units = csv_rows(HERE / 'gradient/all_units.csv')
    formal_gradient = [r for r in gradient_units if r['unit'] != 'archdev001']
    undefined_angles = sum(int(r['undefined_samples']) for r in formal_gradient if r['metric'] == 'angle_degrees')
    undefined_energy = sum(int(r['undefined_samples']) for r in formal_gradient if r['metric'] == 'projection_energy_fraction')
    status = [dict(check='PV trained-source specificity', status='completed',
                   evidence='alignment/README.md; alignment/final_audit.json', missing='none'),
              dict(check='Local-edit geometry and early-path gradients', status='completed',
                   evidence='gradient/README.md; gradient/independent_verification.json', missing='none'),
              dict(check='E/D encoding and initialization recheck', status='completed',
                   evidence='ed/report.md; ed/independent_verification.json', missing='none')]
    write_csv(HERE / 'status_and_missing.csv', status)
    claims = [
        ('G1', 'Geometric selectivity', 'Correct correspondences have positive residual selectivity on average, with substantial condition dependence.',
         'Existing answer-control report; ed/cka_per_unit.csv', 'Prior eight-pair residual average0.125783; E/D new audit retains all versions and initial scores.',
         'Selected task catalog, saved pre-task final norm, specified controls and estimators.',
         'Not all-answer removal, a unique mathematical mechanism, or uniform encoding/configuration effects.'),
        ('G2', 'E/D encoding', 'For GRU E/D, the specified answer control reverses the encoding ordering; CKA and readout margins rank encodings differently.',
         'ed/encoding_contrasts_summary.csv; ed/probe_per_unit.csv',
         'Residual cycle−one0.07727083; raw−0.03156094; length-only−0.10874317. Probe margins one0.01008156/cycle0.00014951.',
         'Four new units; this pair and tested model configurations. Transformer residual contrast3/4negative, not uniform.',
         'No causal encoding mechanism; the original double-dissociation prediction remains failed.'),
        ('R1', 'Readout transfer', 'Trained valleys supports additional transferable peak-position readout under the fixed alignment procedure.',
         'alignment/by_unit.csv; alignment/contrasts_by_unit.csv',
         f"AUROC{f(av['mean'])}; valleys−step0{ms(d0)}; valleys−fixed_points{ms(df)}; bothpositive4/4.",
         'One-line GRU, peaks→valleys, complement pairing, fixed trained source probe/scaler, original fit/test support.',
         'No automatically discovered correspondence, naturally shared decoder, identical computation, or causal use.'),
        ('R2', 'Mapping controls', 'Correct complement alignment exceeds the eleven common mapping controls.',
         'alignment/common_mapping_controls_by_unit.csv', f"Full-test margin{ms(dm)}, positive4/4.",
         'Maximum taken after fixed-length aggregation, all mappings score the same target complement input.',
         'Model controls and mapping controls answer different questions and are not interchangeable.'),
        ('C1', 'Composition', 'The aligned two-specialist composition pipeline does not outperform the same-model-twice baseline.',
         '../permworld_lemma_composition_20261110/main_comparison_by_unit.csv',
         'All four units lower on all three questions; full and matched supports retained.',
         'Specified analyst-supplied algorithms and frozen supervised position readouts.',
         'Not a test that specialists autonomously invoke lemmas or that no composition method can succeed.'),
        ('C2', 'Composition controls', 'Count-preserving position shuffles reduce pipeline performance on questions using those positions.',
         '../permworld_lemma_composition_20261110/README.md',
         'Previously reported turn/fixed-point shuffle effects; NONE cases and true-present subsets separated.',
         'The explicit decoder/composer pipeline, with shared frozen thresholds/readouts.',
         'Does not establish causal use of those positions in the original counting network.'),
        ('I1', 'Model-internal intervention', 'The tested one-dimensional edit flips the local checker but does not change the selected output count, including with current-token upper replay.',
         '../permworld_local_timing_20261110/headlines/main_by_unit.csv',
         '80640 formal cases, both timings zero changes; checker crossings98.8159–100% across weighted unit cells.',
         'Frozen L1/right-value time/dose1/axis and registered counterfactual constructions, one-line GRU.',
         'Not evidence that local extrema are unused, that all axes fail, or that information must be distributed.'),
        ('I2', 'Interface control', 'Replacing the complete first layer can transport the desired count change at the same interface.',
         '../permworld_local_timing_20261110/headlines/auxiliary_eight_cells.csv',
         'Early whole-L1 exact-step cell means99.743–100%; mixed upper swaps can yield other changes.',
         'These mixed-state donor interventions, including matched-prefix-count groups.',
         'Does not locate a necessary counter layer or equate a whole layer with a local abstract predicate.'),
        ('I3', 'Geometry/gradient diagnostic', 'Projection and derivative records characterize this fixed edit and local behavioral sensitivity.',
         'gradient/four_unit_summary.csv; gradient/results/',
         f'All4units/2experts/2constructions/2signs; undefined formal angle/energy samples={undefined_angles}/{undefined_energy}.',
         'Raw recurrent-state Euclidean metric; baseline gradient of z[c+s]−z[c] through early replay and the original suffix.',
         'No mechanism from small projection, small derivative or mean angle; no linearity inference from cached FP32 residuals.'),
        ('S1', 'Statistical scope', 'Independent stochastic units, not positions or lengths, define repetitions.',
         'protocol.json; all stage unit tables',
         'Four new data/initialization/training bundles; historicalunitseparate; fixed length/center weights.',
         'Within-unit models intentionally share initial weights; stochastic factors vary jointly between units.',
         'Does not independently identify each random factor, random-task population, or unrelated-specialist-initialization regime.')]
    claim_rows = [dict(claim_id=x[0], question=x[1], allowed_statement=x[2], evidence=x[3],
                       quantitative_support=x[4], scope=x[5], unsupported_inference=x[6]) for x in claims]
    write_csv(WRITING / 'claim_evidence_scope.csv', claim_rows)
    (WRITING / 'claim_evidence_scope.md').write_text('# Claim–evidence–scope map\n\n' + table(
        ['ID', 'Allowed statement', 'Evidence', 'Scope and boundary'],
        [[r['claim_id'], r['allowed_statement'] + ' ' + r['quantitative_support'], r['evidence'],
          r['scope'] + ' ' + r['unsupported_inference']] for r in claim_rows]) + '\n')
    structure = '''# Proposed TMLR main-text structure

The paper asks four progressively stronger **questions**, with different measurements and controls. Their values are not one effect shrinking through a sequence of tests. The later within-model intervention is not a direct test of computation shared between models.

1. **Introduction.** State the exact-correspondence question and distinguish geometric preference, recoverability after a fitted map, usefulness in an explicit composition procedure, and use in the original network. Present a conditional result and an evidential boundary, not a shared-lemma mechanism claim.
2. **Task system and paired design.** Define PermWorld's16tasks/eight directed relations, two encodings and two model configurations. Describe within-unit actual shared initial weights and four independently resampled data/initialization/training bundles; historical exploration remains separate. State saved step0 identities, all lengths, supports, exclusions, controls, task-learning limitations and code/data provenance.
3. **Measurements.** Start with the questions before formulas. Keep ComparisonA (fix tasks, vary transformation), ComparisonB (fix transformation, vary task), and stricter inverse-minus-best-eligible-transform advantage distinct. Define B0/Bt/Delta and correct/wrong changes. Raw ComparisonB has a shared-initialization zero reference; candidate-specific answer residualization need not. Define the additive answer-mean control and its limits. Explain validation-only isometry fitting, source probes and count-matched sensitivity separately.
4. **Geometric selectivity.** Give the complete original grid and independent repetitions without mixing definitions/counts. Show length and answer controls, initialization decomposition and negative conditions. Use E/D as an illustrative control-dependent ordering, including Transformer unit02's contrary direction; do not present it as a general encoding law. Layer/metric/cross-domain historical cases belong in prescribed supplementary coverage if space is limited.
5. **Readout transfer.** Report full-test peaks→valleys transfer and matched sensitivity. Explicitly reuse the same trained source probe/scaler for trainedvalleys, saved step0target and trainedfixed_points. Present both model contrasts and all11mappingcontrols. Explain that this demonstrates fitted transferable readout under the provided correspondence, not an automatically shared decoder or naturally reused computation.
6. **Composition with frozen readouts.** Give the three held-out questions and the same-model-twice baseline. Keep the negative paired comparison central; show direct-transfer improvement and count-preserving positional shuffles as pipeline diagnostics. Separate true-NONE and nontrivial fixed-point cases. State that the algorithm, map and position supervision are supplied by the experimenter.
7. **Within-model intervention and bounded diagnostics.** Describe local-only and cumulative-count-matched donors, independent checker, exact±1 vs oracle-goal scoring, late vs early paths and whole-layer wiring controls. Present the zero selected-count changes and whole-layer positive response together. Add raw-coordinate projection and gradient checks, fixed finite differences, numerical limitations and all undefined cases. Do not claim a necessary counter layer, distributed causal code, or cross-model shared computation.
8. **Discussion and limitations.** Explain what each question adds and leaves unresolved. Recommend separating trained preference from initialization increments, varying tasks at fixed transform, reporting answer controls and layer/metric dependence, using task-level and model baselines, and distinguishing decodability from natural causal use. Restrict evidence to tested tasks/configurations and paired initialization scheme. No mechanism explanation is required to report the bounded findings honestly.

Suggested main figures: two-comparison schematic; complete controlled geometry grid with initialization decomposition; trained-source/model-control alignment; explicit-composition paired differences; one-dimensional vs whole-layer interventions. Put the three closing audit tables and numerical checks in appendices with main-text pointers. Avoid a funnel graphic implying comparable numerical effect sizes across CKA, AUROC, exact-answer accuracy and derivatives.

This is an integration plan and new draft fragment set. The existing manuscript, references, F17 materials and DMLR direction are unchanged. No further experimental stage is scheduled by this document.
'''
    (WRITING / 'tmlr_structure.md').write_text(structure)
    gradient_sentence = '; '.join(
        f"{task}/{group}/{sign}: projection energy {sci(gg[task, group, sign, 'projection_energy_fraction']['mean'])}, "
        f"g·δ {sci(gg[task, group, sign, 'g_dot_delta']['mean'])}, "
        f"cached Δm {sci(gg[task, group, sign, 'actual_logit_margin_change']['mean'])}"
        for task, group, sign in cells)
    energy_range = [float(gg[*key, 'projection_energy_fraction']['mean']) for key in cells]
    gradient_range = [float(gg[*key, 'gradient_norm']['mean']) for key in cells]
    margin_range = [float(gg[*key, 'baseline_logit_margin']['mean']) for key in cells]
    response_range = [float(gg[*key, 'actual_logit_margin_change']['mean']) for key in cells]
    result_text = f'''# Results paragraphs for a revised TMLR draft

These paragraphs integrate the completed analyses; they are exploratory post-result diagnostics, not a retrospectively preregistered confirmation. Numerical support is in `claim_evidence_scope.csv` and the linked unit tables. The old manuscript has not been edited.

**Geometric selectivity depends on its control and reference.** In the existing four-unit analysis, the designated-transform advantage under within-length dual-answer residual U-CKA averaged0.125783 across the selected32conditions, with22conditions positive in every new unit. This stricter advantage subtracts the strongest eligible alternative transformation and differs from the original one-wrong-transformation Bt. A training increment relative to shared initialization need not reflect increasing correct-pair similarity: the original analysis retained both score components and found that much of the increment reflected declining identity-pair similarity. The current audit exactly reproduced the E/D table. For GRU, the mean cycle-minus-one-line advantage was−0.031561 for pooled ordinary CKA and−0.108743 for within-length U-CKA, but+0.077271 after the specified additive dual-answer-mean control, positive in all four new units. Transformer showed a negative residual contrast on average (−0.059307), with three negative units and one positive unit. Residualization removes the stated additive means, not all answer-related information; these comparisons do not identify a causal encoding mechanism.

**Aligned position readout has training-task specificity in the tested pair.** Keeping the exact trained peaks probe and scaler fixed, complement-aligned trained valleys reached a full-test mean per-example position AUROC of {ms(av)} across the four new one-line GRU units. The corresponding saved step-0 target and trained fixed_points target scored {ms(a0)} and {ms(af)}, respectively. Each target used its own fit-only means and the same-budget orthogonal mapping into trained peaks coordinates. The paired valleys-minus-step-0 difference was {ms(d0)}, range [{f(d0['minimum'])}, {f(d0['maximum'])}], and valleys-minus-fixed_points was {ms(df)}, range [{f(df['minimum'])}, {f(df['maximum'])}]; both were positive in all four units. The mean margin over the best of the eleven common mapping controls was {ms(dm)}. The original count-matched valleys mean0.637309 remains a sensitivity result; full-test mean0.633139 is primary here. This supports additional transferable position readout associated with the trained target task under the fitted correspondence. It does not establish that the networks naturally share a decoder or counting computation.

**Geometric and readout contrasts need not rank encodings alike.** The residual E/D CKA advantage was0.019517 for one-line GRU and0.096787 for cycle GRU. In contrast, the final-state position-readout margin over the best common map control was0.010082±0.001878 for one-line and0.000150±0.000971 for cycle, with the latter positive in only two of four units. Cycle's higher raw aligned AUROC did not produce the larger mapping-specific margin. CKA and AUROC have different definitions, fits and nuisance controls, so these numerical effects are not subtracted. The original predicted cross-task encoding double dissociation failed and remains a failed prediction.

**Frozen readouts can be composed without yielding a cross-specialist advantage.** In the previously completed composition test, the aligned two-specialist route scored lower than the same-model-twice baseline in every new unit on all three questions, under both full and matched supports. Full-test mean exact accuracy was9.43% versus14.84% for the longest monotone interval,49.39% versus52.29% for the longest interval containing a fixed point, and53.50% versus54.15% for turns before the first fixed point. Alignment improved the interval questions over unaligned transfer, and count-preserving position shuffles reduced performance where the supplied algorithms used those positions. These are results for analyst-supplied composition rules applied to supervised readouts, not evidence that the networks themselves invoke a reusable lemma.

**The selected local edit does not change the original count.** Across80,640 existing formal directed cases, the registered one-dimensional L1 edit flipped the independent local checker at weighted rates of98.8159–100% across unit cells, but changed the selected count in none of the cases under either late injection or current-token upper replay. This includes cumulative-count-matched donors. Whole-L1 replacement at the same interface produced requested relative±1 changes at99.743–100% across the eight early-condition cell means, so the interface can convey changes that this specific axis/dose does not reproduce. Complete-layer swaps mix state histories and do not locate a uniquely necessary counter layer.

**Projection and derivative measurements bound the interpretation.** The edit is a raw-recurrent-state Euclidean donor projection after converting the saved scaler/ridge coefficients back into raw coordinates; the ideal projection and actual rounded FP32 displacement are retained separately. At the baseline recipient state we differentiated m=z[c+s]−z[c] through the upper current-token replay and the original suffix, retaining the true recipient c+s goal for count-matched donors. Across the eight registered task/construction/sign cell means, projection energy was {100*min(energy_range):.4f}–{100*max(energy_range):.4f}% of the donor-difference energy, while the gradient norm was {min(gradient_range):.4f}–{max(gradient_range):.4f}. Thus the modal-count null is not a zero-gradient result. Mean actual margin changes ranged from {min(response_range):.6f} to {max(response_range):.6f} logit units, against baseline goal margins of {min(margin_range):.4f} to {max(margin_range):.4f}; responses were not uniformly goal-positive. The condition-level tables retain both signs and donor constructions, every unit and all eight random directions. Fixed finite differences on a FP64 copy of the same smooth graph check derivative wiring, while FP32 no-edit and edited logits remain within the original tolerance of the saved paths. Actual margin changes use cached FP32 logits; individual sufficiently small changes can approach backend roundoff, so a discrepancy from g·δ does not by itself identify curvature or a mechanistic residual. Small projection energy, a small directional derivative or an angle statistic alone establishes neither absence of local information use nor a necessarily distributed causal code.

**Evidence levels remain distinct.** Geometric preference, recoverability after an explicitly fitted map, performance of an explicit composition pipeline and the response to a specific model-internal intervention answer different questions. Their numerical values are not a single effect shrinking across stages. In particular, the within-model intervention does not directly test computation shared across specialists. The repeated units jointly resample data, architecture-level initialization and training randomness; they do not separately identify those sources or cover unrelated specialist initializations, other task families or arbitrary model configurations.
'''
    # Keep the draft easy to paste; minor spacing here is editorial, not metric processing.
    for a, b in [('averaged0.', 'averaged 0.'), ('selected32', 'selected 32'), ('with22', 'with 22'),
                 ('32conditions', '32 conditions'), ('22conditions', '22 conditions'),
                 ('0.010082±0.001878', '0.010082 ± 0.001878'), ('0.000150±0.000971', '0.000150 ± 0.000971'),
                 ('relative±1', 'relative ±1'),
                 ('was−', 'was −'), ('and−', 'and −'), ('but+', 'but +'), ('by0.', 'by 0.'),
                 ('mean0.', 'mean 0.'), ('was0.', 'was 0.'), ('and0.', 'and 0.'),
                 ('at99.', 'at 99.'), ('Across80,', 'Across 80,'), ('of98.', 'of 98.'),
                 ('was9.', 'was 9.'), ('versus14.', 'versus 14.'), ('interval,49.', 'interval, 49.'),
                 ('versus52.', 'versus 52.'), ('and53.', 'and 53.'), ('versus54.', 'versus 54.')]:
        result_text = result_text.replace(a, b)
    (WRITING / 'results_draft.md').write_text(result_text)
    # Paragraph-only TeX; no template, author, or existing manuscript files touched.
    import re
    passages = [part for part in result_text.split('\n\n') if part.startswith('**')]
    tex = '% New standalone result fragments; not inserted into the existing manuscript.\n'
    for part in passages:
        heading, body = part[2:].split('** ', 1)
        text = heading + ' ' + body
        text = text.replace('%', r'\%').replace('_', r'\_').replace('&', r'\&')
        text = text.replace('±', r' $\pm$ ').replace('→', r' $\rightarrow$ ').replace('−', '-').replace('–', '--')
        text = text.replace('δ', r'$\delta$').replace('Δ', r'$\Delta$').replace('·', r'$\cdot$')
        tex += r'\paragraph{' + heading.replace('_', r'\_') + '} ' + text[len(heading) + 1:] + '\n\n'
    (WRITING / 'results_draft.tex').write_text(tex)
    report = f'''# PermWorld closing checks: evidence for writing

All three checks are complete. This is an **exploratory post-result audit and supplementary diagnosis**, with rules frozen before the newly requested calculations. It is not a retrospectively preregistered confirmation experiment. Four new units define every primary mean/SD/range; the historical unit is separate. No base model or probe was fitted, no direction/layer/time/dose was selected, and no new task/world was evaluated. Only ten missing target-control orthogonal maps were fitted by the original procedure.

## 1. Same trained-source probe, three target representations

{alignment_table}

The last row is historical reference only. Full-test four-unit AUROCs are trained valleys **{ms(av)}**, saved step-0 target **{ms(a0)}**, and trained fixed_points **{ms(af)}**. Paired differences are **{ms(d0)}** and **{ms(df)}**, both positive in4/4 units. The common mapping-control margin is **{ms(dm)}**. Each target uses its own fit-only mean and map, but the exact trained peaks probe/scaler and original fit/test IDs are identical throughout. The old initial-phase result used an initial source probe and has been retained as a different comparison, not substituted for this control.

This supports a training-task-related extra transferable position readout for this setting. It does not establish a naturally shared decoder, discovered relation, shared computation or causal use. [All unit routes](alignment/by_unit.csv), [paired contrasts](alignment/contrasts_by_unit.csv), [full/matched summary](alignment/four_unit_summary.csv), [all eleven common mapping controls](alignment/common_mapping_controls_by_unit.csv), [source audit](alignment/README.md). Undefined per-example AUROCs and all denominators remain in per-length/per-sample outputs.

## 2. Raw-coordinate edit and local behavior gradient

Here h is the actual unnormalized L1 recurrent state after the right-neighbor value token. The normalized axis is the saved ridge coefficient divided by its saved scaler scale, then normalized in the **raw Euclidean metric**. The ideal edit is `(d·v)v`; the actual FP32 displacement is saved separately, including rounding differences. The upper current-token update is recomputed from the recipient's preceding upper states, and the original suffix remains on the differentiable path. Model parameters remain frozen and all dropout remains in eval mode.

The target scalar is `m=z[c+s]−z[c]`, using the true recipient count c and the original local sign s, including count-matched pairs. It is already goal-oriented and is not multiplied by s again. Values below are four-unit mean ± sample SD; `main` means local-only, `matched` means cumulative-count-matched.

{geometry_table}

{gradient_table}

Formal undefined angle observations: **{undefined_angles}**; undefined donor projection-energy observations: **{undefined_energy}**. The complete per-pair vectors, both signed directions, every random direction and finite/undefined support remain saved. The displayed random derivative averages eight signed values within pair and can cancel; it is not a statement that every random direction is weak or strong. A mean angle also does not establish per-example orthogonality.

The zero selected-count response is **not zero local sensitivity**: the eight cell-mean gradient norms span {min(gradient_range):.4f}–{max(gradient_range):.4f}. Actual cached margin responses span {min(response_range):.6f} to {max(response_range):.6f} in cell means, while the corresponding baseline goal margins are {min(margin_range):.4f} to {max(margin_range):.4f}. These are small, direction-dependent logit responses without a selected-count change. They are not uniformly in the requested direction. Projection energy spans {100*min(energy_range):.4f}–{100*max(energy_range):.4f}% in cell means; this alone does not explain the count null or establish distributed causal information.

The original early/late one-dimensional modal-count change remains zero in every formal condition; whole-L1 replacement is the positive interface control. The derivative audit answers only local sensitivity of this edit near the recipient state. Small projection/derivative values cannot show that local information is unused or must be distributed. Primary gradients use the native FP32 recurrent graph while endpoint changes reuse cached FP32 logits; the original1e−4 cross-backend logit guard can exceed these small changes. A cached-endpoint-minus-linearized discrepancy must not be called curvature, nonlinear mechanism or unexplained causal effect. FP64 finite differences validate the nominal smooth graph and fixed-weight wiring, not FP32 rounding.

[Per-unit gradient table](gradient/all_units.csv), [mean/SD/range](gradient/four_unit_summary.csv), [detailed diagnostic report](gradient/README.md), [independent arithmetic and wiring audit](gradient/independent_verification.json). The fixed old centers, all21lengths and original weights are retained; undefined entries are not filled with zero or silently reweighted.

## 3. E/D encoding comparison remains control- and metric-specific

{ed_table}

{ed_contrasts}

The reference means are reproduced at their displayed rounding. For GRU, residual CKA favors cycle in4/4 new units, while the raw and length-only advantages favor one-line in4/4. Transformer has a negative residual cycle-minus-one-line mean but only3/4 negative units; unit02 is positive. The already completed E/D probe comparison instead has mean common-control margins0.01008156(one-line) and0.00014951(cycle); the cycle margin changes sign across units. These are different metrics and controls, not subtractable effect sizes. The original cross-pair encoding double-dissociation prediction failed and is **not rewritten as a success**.

Answer residualization removes the specified additive answer means; it does not erase covariance, confidence, joint-answer interactions or all answer information. These configuration/coding comparisons do not isolate a causal encoding effect. [All three CKA versions and initial scores](ed/report.md), [unit contrasts](ed/encoding_contrasts_by_unit.csv), [probe comparison](ed/probe_per_unit.csv), [Transformer clarification](ed/reader_clarification.md).

![E/D: geometry and readout margins use separate scales](ed/ed_encoding_comparison.png)

## Writing deliverables and scope

* [Claim–evidence–scope table](writing/claim_evidence_scope.md), with a [machine-readable CSV](writing/claim_evidence_scope.csv).
* [TMLR main-text structure](writing/tmlr_structure.md).
* [Ready-to-edit English results paragraphs](writing/results_draft.md), with [paragraph-only TeX](writing/results_draft.tex).
* [Three-check status/source/missing-material table](status_and_missing.csv).
* [Frozen protocol](protocol.json), [source/spec bindings](freeze.json), [reproduction commands](reproduce.sh), and final delivery/source-protection receipts.

Geometry, readout transfer, explicit composition and model-internal intervention represent increasingly demanding questions with different measurements. They are not one numerical effect shrinking at each stage. Within-model intervention does not directly test computation shared across models. No mechanism explanation is claimed to be complete.

All stages preserve the original splits, anchors,21fixedlengthweights and undefined-value handling. The four data/initialization/training bundles are the repeat units; shared tasks/models, lengths, positions, controls and paired edits are dependent. This round introduces no inferential significance or equivalence test. All input/output manifests state their exact scope rather than claiming continuous monitoring of every disk file. No existing manuscript, F17 artifact, DMLR direction, old report or dataset was rewritten; no upload/publication or further experiment is launched.
'''
    for a, b in [('in4/4', 'in 4/4'), ('only3/4', 'only 3/4'), ('all21lengths', 'all 21 lengths'),
                 ('all21', 'all 21'), ('original1e', 'original 1e'), ('margins0.', 'margins 0.'),
                 ('(one-line) and0.', ' (one-line) and 0.'), ('0.01008156(one-line)', '0.01008156 (one-line)'),
                 ('0.00014951(cycle)', '0.00014951 (cycle)'), ('anchors,21fixedlengthweights', 'anchors, 21 fixed length weights')]:
        report = report.replace(a, b)
    (HERE / 'README.md').write_text(report)
    (WRITING / 'sources.json').write_text(json.dumps(dict(created_utc=datetime.now(timezone.utc).isoformat(),
        code_sha256=sha(__file__), source_sha256=bindings,
        source_tables_only=True, no_fits_or_forwards=True,
        note='Post-result editorial integration of completed evidence, not a new experimental protocol.'), indent=2) + '\n')
    print('Wrote closing report, claim map, structure and new English/TeX fragments.')


if __name__ == '__main__':
    main()
