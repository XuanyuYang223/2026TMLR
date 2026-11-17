"""Render the completed, frozen timing tables; no fitting or new evaluations."""
from pathlib import Path
import csv
import json

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent


def read(name):
    return json.loads((HERE / name).read_text())


def rows(name):
    with (HERE / 'headlines' / name).open(newline='') as handle:
        return list(csv.DictReader(handle))


def pct(value, digits=3):
    return f'{100 * float(value):.{digits}f}'


def table(header, values):
    return '\n'.join(['|' + '|'.join(header) + '|',
                      '|' + '|'.join(['---'] * len(header)) + '|',
                      *('|' + '|'.join(map(str, row)) + '|' for row in values)])


def identity(row):
    return [row['task'], 'count-matched' if row['group'] == 'matched' else 'local-only',
            f"{int(row['sign']):+d}"]


def mean_sd(row, prefix):
    return f"{pct(row[prefix + '_mean'])} ± {pct(row[prefix + '_sample_sd'])}"


def main():
    protocol = read('protocol.json')
    summary = read('summary/completion.json')
    headline = read('headlines/completion.json')
    audit = read('saved_results_audit.json')
    protection = read('protection_end.json')
    assert summary['status'] == 'completed_saved_timing_summary'
    assert headline['status'].startswith('completed')
    assert audit['status'] == protection['status'] == 'passed'
    main_rows = rows('main_eight_cells.csv')
    aux_rows = rows('auxiliary_eight_cells.csv')
    positive_rows = rows('positive_controls_eight_cells.csv')
    main_units = rows('main_by_unit.csv')
    aux_units = rows('auxiliary_by_unit.csv')
    assert len(main_rows) == len(aux_rows) == len(positive_rows) == 8
    for row in main_units:
        for field in ['late_relative_step', 'early_relative_step', 'late_any_numeric_change',
                      'early_any_numeric_change', 'late_any_token_change', 'early_any_token_change',
                      'early_minus_late_relative_step', 'did_relative_step_vs_random',
                      'did_relative_step_vs_same_label']:
            assert float(row[field]) == 0, (row['unit'], row['task'], field)

    receipts = [read(f"results/{unit['unit']}/completion.json") for unit in protocol['units']]
    production_maxima = {
        key: max(task['maxima'][key] for receipt in receipts for task in receipt['task_reports'])
        for key in receipts[0]['task_reports'][0]['maxima']
    }
    checker_min = min(float(row['early_checker_crossing']) for row in main_units)
    checker_max = max(float(row['early_checker_crossing']) for row in main_units)
    expected_max = max(abs(float(row['early_signed_expectation_change'])) for row in main_units)
    main_table = table(
        ['Expert', 'Pair construction', 'Direction', 'Late exact step (%)', 'Early exact step (%)',
         'Paired early−late (pp)', 'DiD vs same label (pp)', 'DiD vs random (pp)',
         'Checker crossing, mean ± SD (%)'],
        [identity(row) + [pct(row[key + '_mean']) for key in
                         ['late_relative_step', 'early_relative_step', 'early_minus_late_relative_step',
                          'did_relative_step_vs_same_label', 'did_relative_step_vs_random']]
         + [mean_sd(row, 'early_checker_crossing')] for row in main_rows])
    aux_table = table(
        ['Expert', 'Pair construction', 'Direction', 'Whole L1 late (%)', 'Whole L1 early (%)',
         'Paired early−late (pp)', 'Late L2 (%)', 'Late L3 (%)', 'Late L4 (%)'],
        [identity(row) + [mean_sd(row, 'late_whole_layer1_relative_step_hit'),
                         mean_sd(row, 'early_whole_layer1_relative_step_hit'),
                         mean_sd(row, 'early_minus_late_whole_layer1_relative_step_hit')]
         + [pct(row[f'late_whole_layer{layer}_relative_step_hit_mean']) for layer in [2, 3, 4]]
         for row in aux_rows])
    other_table = table(
        ['Expert', 'Pair construction', 'Direction', 'L2 other change (%)',
         'L3 other change (%)', 'L4 other change (%)'],
        [identity(row) + [pct(row[f'late_whole_layer{layer}_other_numeric_count_change_mean'])
                          for layer in [2, 3, 4]] for row in aux_rows])
    unit_table = table(
        ['Unit', 'Expert', 'Pair construction', 'Direction', 'Whole L1 late (%)', 'Whole L1 early (%)',
         'Paired early−late (pp)'],
        [[row['unit']] + identity(row) + [pct(row[key]) for key in
          ['late_whole_layer1_relative_step_hit', 'early_whole_layer1_relative_step_hit',
           'early_minus_late_whole_layer1_relative_step_hit']] for row in aux_units])
    positive_table = table(
        ['Expert', 'Pair construction', 'Direction', 'Full-four-layer donor oracle (%)',
         'Full-four-layer local goal (%)', 'Real local counterfactual oracle (%)'],
        [identity(row) + [pct(row[key + '_mean'], 4) for key in
                         ['full4_donor_oracle_accuracy', 'full4_local_desired_answer',
                          'actual_cf_oracle_accuracy']] for row in positive_rows])
    coverage = table(
        ['Unit', 'Role', 'Directed cases', 'Saved intervention rows', 'Production seconds'],
        [[receipt['unit'], 'reference' if receipt['historical_reference'] else 'formal',
          receipt['rows'] // 29, receipt['rows'], f"{receipt['elapsed_seconds']:.3f}"]
         for receipt in receipts])
    text = fr'''# Does current-token upper-layer replay rescue the local edit?

Completed: **no rescue of the chosen one-dimensional edit**. In each of `unit02`–`unit05`, both experts, both directions and both donor constructions had **zero changes to the selected output count**, under both timings. The independent local checker nevertheless crossed toward the donor on {pct(checker_min, 4)}–{pct(checker_max, 4)}% of cases after the registered within-unit weighting. Whole-L1 replacement produced large expected count changes at the same interface. This separates a successful local readout manipulation from a successful count intervention.

This is an exploratory follow-up on the **same existing models and pairs**. Prior late-edit results were known. The new protocol and executable/summary sources were hash-frozen before the new formal timing results. The historical `archdev001` is shown separately and never included in four-unit statistics. This is not a fresh independent scientific replication.

## What changed between the two timings?

At the right-neighbor value token, let the recipient states before that token be `h_l(t−1)`, and let the registered donor edit produce `h1_edit(t)`.

* **Late:** carry `(h1_edit(t), h2(t), h3(t), h4(t))` into the original suffix.
* **Early:** keep exactly the same FP32 `h1_edit(t)`; compute layers 2–4 sequentially at the current token, each from its own recipient `h_l(t−1)` and the edited current output of the layer below. Carry these recomputed states into the same suffix.

The value token is **not rerun through layer 1**. Early and late inject byte-identical L1 states, so their checker readings and injected L1 norms are equal. The total four-layer carried-state norm can change under upper replay; it is reported separately and was not rescaled to erase the intended upper-layer effect.

The original dose 1, edit axis, independent checker, eight random directions (seeds 1026–1033), same-label donors, pairs, signs and output scoring were loaded from the previous stage. No probe, direction, threshold or model was fitted. Production model parameters were checked before and after every expert. All 21 lengths, n=10–30, remain included. Registered target positions are i=2…n−2 for local-only pairs and i=6…n−2 for count-matched pairs; this does not cover every possible interior position.

## Primary paired comparison

An **exact relative step** means the numeric output argmax changes from the recipient model's original prediction by exactly the requested +1 or −1. An **oracle goal hit** instead means the edited prediction equals the recipient's true count plus that sign. Their full-precision columns are separate; an initially incorrect model can already predict the oracle goal before any edit.

{main_table}

All displayed relative hit rates, paired timing effects and differences-in-differences are exactly zero, including every raw formal unit. Any other numeric count change and any full-vocabulary argmax-token change are also zero for the main one-dimensional edit. Numeric argmax coverage is 100%. This is a bounded negative observation, not an equivalence test or proof that this information was never used elsewhere.

Small probability shifts remain: the largest absolute formal **unit × expert × construction × sign** mean signed change in conditional numeric expectation under the early edit is {expected_max:.9g} count units. Directions are not uniformly positive. These shifts do not satisfy the protocol's criterion of a count-changing rescue. Full-vocabulary numeric mass, expectation contrasts and all raw control conditions remain in the detailed summary.

For valleys −1, oracle goal hits have tiny nonzero four-unit means (local-only 0.00551146%, count-matched 0.00646998%); these are already present before editing in unit04 and are unchanged across timing. They must not be described as successful count changes. See [the full primary table](headlines/main_eight_cells.csv), [all four units](headlines/main_by_unit.csv), and [paired timing/control contrasts](summary/timing_contrasts_by_unit.csv).

## Whole-layer diagnostics

Entries below are exact relative-step hit rates. Whole-L1 columns and their paired difference show four-unit mean ± **sample SD**; late L2–L4 entries show the four-unit mean. A whole recurrent layer has 256 coordinates and can move more than the selected local feature.

{aux_table}

Complete first-layer replacement is effective under both timings, including the count-matched construction. Early replay raises its eight cell means to 99.743–100%, versus 84.667–99.620% late. The largest late variability is valleys decrease: its local-only four-unit SD is about 17.01 percentage points, so the raw unit results below matter. These findings show that the tested interface can transport the expected local count change. They do not isolate a unique counter layer or show that the one-dimensional probe axis is a naturally used mathematical predicate.

Low expected-step success for upper-layer swaps must not be confused with no output effect. **Count-matched L3 and L4 swaps frequently produce other changes**, often in the direction opposite the registered local goal. Their donor histories include the compensating earlier change; this diagnostic mixes state histories rather than intervening on an abstract local fact alone.

{other_table}

“Other” means a nonzero numeric count change different from the requested signed step. The signed conditional expectation, every nonnumeric outcome, oracle goal hit and count change remain available in [the auxiliary table](headlines/auxiliary_eight_cells.csv) and [all auxiliary unit results](headlines/auxiliary_by_unit.csv). Single-layer swaps establish sufficiency for a change in these mixed states; they do not locate when or where natural accumulation occurs.

### Whole-L1 results in each formal unit

{unit_table}

The early−late whole-L1 contrast is nonnegative in all 32 listed unit cells. Historical values are kept in `headlines/auxiliary_historical_reference.csv`; full mean/SD/minimum/maximum and raw values are preserved in the eight-cell CSV.

![All four units: local one-dimensional edit and whole-L1 replacement](headlines/timing_payload_comparison.png)

The [PDF figure](headlines/timing_payload_comparison.pdf) and [plot script](plot_headlines.py) use these same saved per-unit tables, with all eight cells shown.

## Positive references have different targets

The real local-only counterfactual is scored against recipient true count ±1. The full-four-layer swap is a wiring check against the donor's actual continuation and true full count. In count-matched pairs the donor's count equals the recipient's count, so it is **not** the local-only ±1 goal. These references were kept separate; previously verified native counterfactual outputs were reused, while the full-state swap continuation was rerun and checked against the saved native donor output.

{positive_table}

These are oracle accuracies or local-goal hit rates as explicitly labeled, not inferred causal success. No-edit replay causes no count-category drift. See [reference results](headlines/positive_controls_eight_cells.csv) and [per-unit references](headlines/positive_controls_by_unit.csv).

## Coverage, weights and interpretation

{coverage}

There are 80,640 formal directed cases and 20,160 historical cases. Each saves 29 rows: 11 common conditions at each timing, four late whole-layer swaps, early whole-L1 replay, and the two distinct positive references. The total is 2,923,200 production rows. These are dependent comparisons, **not independent statistical replicates**.

Every primary metric first averages registered cases within a center, then equally weights the registered centers within a length, then uses the original fixed 1/21 length weights. Tasks, constructions and signs remain separate. Timing differences are computed within each case before aggregation. The eight random directions are averaged within case, never selected by maximum. Four-unit mean, sample SD, range and all raw unit values are retained. Original-baseline-correct, checker-baseline-correct, common-position, available-cell and pooled summaries are explicitly labeled sensitivities; none replaces the primary weights.

The protocol's second outcome applies: **the whole-layer interface can convey changes that the tested one-dimensional edit does not reproduce**. The early-vs-late hypothesis did not rescue the local edit, including cumulative-count-matched donors. This does not establish that a natural counting computation ignores local extrema, that all one-dimensional interventions would fail, or that the exact accumulation time has been located. Probe readability, token rank/adjacent-order confounds, mixed-state distribution shifts and potential distributed encodings remain distinct issues.

## Validation and provenance

* Protocol SHA256: `{read('freeze.json')['protocol_sha256']}`; original Git HEAD: `{protocol['git_head']}`. This follow-up adds separate untracked directories; the frozen production source, old studies and tracked repository contents were not rewritten.
* Input audit: 249,547 checks, no failures; 4,498 named physical input files freshly hashed at start, including all ten actually used old test-state payloads. A fixed old-cache subset also passed 90 byte comparisons. [Input audit](input_audit.json), [initial protection scope](protection_start.json).
* Replay tests: seven focused synthetic tests. The historical no-edit preflight had maximum logit error 2.861023e−6 under the original FP32/TF32-off flags, below the original 1e−4 tolerance. An initial default-TF32 preflight failed and was corrected by restoring the original numerical flags **before formal runs**, without loosening tolerance. [Preflight record](replay_preflight.json).
* All-run guard maxima: no-edit current-state error {production_maxima['noedit_state']:.9g}; early no-edit vs native query error {production_maxima['early_noedit_native_logits']:.9g}; rerun late vs original late error {production_maxima['late_old_logits']:.9g}; full-state swap vs native donor error {production_maxima['fullswap_native_logits']:.9g}. Logit tolerance remains 1e−4; state tolerance 2e−5. L1 paired state/readout differences are zero. All ten expert parameter hashes are unchanged; no no-op or late-rerun argmax mismatch occurred.
* The independent saved-result audit passed {audit['checks']:,} checks: 180 fixed cases, 5,220 CSV rows and 2,160 manual NumPy GRU gate-equation comparisons. Maximum independent gate-equation error was 5.677393e−7; metric/probe arithmetic error 1.776357e−14. This is a bounded implementation audit, not an independent scientific experiment. [Saved-result audit](saved_results_audit.json).
* The end audit rehashed 4,507 named physical dependencies, with matching endpoints and unchanged HEAD/clean tracked state. This covers the explicitly listed files; it is **not** continuous monitoring or a claim that every file on the disks was checked. Unused fit/dev payloads and the old 40-GB token cache were not reread. [End protection audit](protection_end.json).
* Frozen aggregation and derived headline receipts are in [summary/completion.json](summary/completion.json) and [headlines/completion.json](headlines/completion.json); 2,048 independent headline scalar checks exactly matched the frozen summary.

## Reproduction and files

Run `bash diagnostics/permworld_local_timing_20261110/reproduce.sh` from this repository. It uses the existing local environment/vendor path and old frozen caches; it never calls the parent stage's calibration or fitting commands. Completed production units are hash-checked and reused. See [the command script](reproduce.sh) for individual stages. Missing old inputs are errors, not authorization to regenerate or refit them.

* Executable timing/replay sources: [`experiments/permworld_local_timing_20261110/`](../../experiments/permworld_local_timing_20261110/).
* Raw production rows and cell states: `/mnt/yangx-projects-f/20261110/permworld_local_timing/<unit>/`. Each `results/<unit>/pair_results.csv` is a symlink to that unit's physical CSV. Cell NPZs store all condition logits, the recipient previous states, injected L1 states, early recomputed upper states, IDs and old-source/checkpoint/data/protocol bindings.
* [Main eight-cell table](headlines/main_eight_cells.csv), [auxiliary table](headlines/auxiliary_eight_cells.csv), [positive references](headlines/positive_controls_eight_cells.csv); each includes mean, sample SD, range and raw unit values. Their `*_by_unit.csv` and `*_historical_reference.csv` files preserve the four formal units and historical reference separately.
* Detailed [summary tables](summary/) include `by_length.csv`, `by_position.csv.gz`, `scored_pairs.csv.gz`, complete control conditions and paired contrasts. The summary has {summary['scored_rows']:,} scored rows because it additionally materializes within-case random means and contrasts; these are **not extra interventions**.
* The delivery audit additionally verifies every new physical production CSV/cell hash and writes a checkpoint/seed manifest. The final delivery manifest inventories only this new experiment and links the protected-input audits.

This completed round ends here. DAS, cross-model swaps, new-task generalization, additional fitting/training and publication are outside this delivery.
'''
    (HERE / 'README.md').write_text(text)
    print('Wrote', HERE / 'README.md')


if __name__ == '__main__':
    main()
