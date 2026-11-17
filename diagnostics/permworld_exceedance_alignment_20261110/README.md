# Exceedance/inverse-deficiency transfer and the encoding comparison

**The predicted encoding crossover did not appear.** Peaks/valleys has a larger mapping advantage in one-line notation in all four new units. Exceedances/deficiencies also has a larger advantage in one-line, so the two predicted directions hold simultaneously in **0/4 units**. The original count-matched sensitivity gives the same 0/4 result.

The table below reports the primary **full-test** comparison. Each unit uses all 21 lengths, n=10–30. A margin is correct-map AUROC minus the largest of the eleven common mapping controls, after each method has first been averaged with fixed equal length weights. Direct transfer and own-model readout are separate references, not members of that maximum.

| Task pair | Encoding | unit02 | unit03 | unit04 | unit05 | Mean aligned AUROC | Mean margin ± sample SD | Margin range |
|---|---|---|---|---|---|---|---|---|
| peaks_valleys | one_line | +0.124881 | +0.085815 | +0.115320 | +0.129235 | 0.633139 | +0.113813 ± 0.019549 | +0.085815 to +0.129235 |
| peaks_valleys | canonical_cycle | +0.006787 | +0.013805 | +0.008931 | +0.014775 | 0.532766 | +0.011074 ± 0.003835 | +0.006787 to +0.014775 |
| exceedances_deficiencies | one_line | +0.011404 | +0.010213 | +0.007386 | +0.011324 | 0.782949 | +0.010082 ± 0.001878 | +0.007386 to +0.011404 |
| exceedances_deficiencies | canonical_cycle | +0.001518 | -0.000495 | -0.000585 | +0.000160 | 0.830156 | +0.000150 ± 0.000971 | -0.000585 to +0.001518 |

[four_unit_main_table.csv](four_unit_main_table.csv) retains full-precision correct/direct/control/within-model values for every unit × task pair × encoding. [All units, supports and controls](summary/per_unit_comparison.csv) also include the historical engineering reference, matched sensitivity and every individual shuffled map. [Cross-pair contrasts](summary/per_unit_cross_pair.csv) and [sample SD/range summaries](summary/cross_pair_summary.csv) preserve the paired unit structure.

## Why raw AUROC is not enough here

For exceedances/deficiencies, cycle has a higher aligned raw AUROC (0.830156 versus 0.782949), but its correct-minus-best-common margin is only 0.000150, compared with 0.010082 for one-line. Cycle's margin is positive in two of the four units and negative in two. Unconditional positional priors and correctly aligned **step-0 targets evaluated with the trained source probe** score higher than the trained-target correct map in both encodings in every new unit. The source and target counting tasks are nevertheless learned strongly: cycle native count accuracy is at least 99.9814%, while one-line source and target minima are 99.8884% and 99.2560%, respectively, across the new units' equal-length means.

Thus high recoverability/transfer AUROC is not, by itself, evidence of relation-specific learned position coding. This comparison does not identify an internal counting algorithm, functional task benefit, or a general causal effect of encoding. It covers two task pairs, one GRU configuration and four independent data/initialization/training bundles. Shared models, lengths, positions and shuffle maps are not independent repetitions.

The peaks/valleys one-line mean here is 0.633139 because **full-test support is now primary**. The earlier 0.637309 is the original count-matched value; both reproduce from the saved results. No earlier output was overwritten.

## Indexing, fitting and controls

For j=pi(i), the exact identity is E_i(pi)=[pi_i>i]=D_j(inverse(pi)). Fixed points are negative. A transferred vector-valued source probe already returns source-index i outputs and is scored directly. A target-owned deficiency probe or prior is naturally indexed by j and is gathered by pi_i-1 **exactly once**. This equality and the actual token layout were independently checked exhaustively on small permutations.

Source probes use validation-fit identity activations and their own position labels, with the unchanged scaler/ridge/source-dev alpha selection. Each orthogonal map uses fit-only per-length means and equal-length cross-products. The identity, complement and eight count-preserving shuffled fits all score the same held-out inverse target activations, using the inverse fit mean and the same frozen source probe. Step-0 and fixed_points controls use their own inverse activations/means mapped into the trained source coordinates. Fixed_points receives no position labels for this control.

The eleven common controls are means-only, identity, the pair-specific wrong transform (inverse for peaks/valleys; complement for exceedances/deficiencies), and each of the eight shuffled mappings individually. The maximum is taken **after**, not before, equal-length aggregation. Source/target position priors, explicitly oracle-count-conditioned prior references, direct transfer and model controls remain separate. No position-score threshold is fitted or used.

Each new unit has 10,752 test anchors (512 per length), shared across both encodings. ED's matched sets contain 507, 539, 513 and 525 anchors; they require source E counts to agree with target D counts under identity and complement (inverse agreement is exact). PV retains its old matched mask. These pair-specific sensitivities are not described as identical samples across task pairs. Every AUROC-undefined sample and finite per-length denominator is recorded; missing lengths would stay undefined with no reweighting.

## Separate local-token diagnostic

The separate diagnostic has completed for all five units. Four-new-unit means are:

| Encoding | Source own BEFORE → AFTER | Correct map AFTER | Best common AFTER | Source n/i prior | Target n/j prior |
|---|---|---|---|---|---|
| one_line | 0.718585 → 0.956123 | 0.773969 | 0.654500 | 0.838467 | 0.839435 |
| canonical_cycle | 0.479870 → 0.980939 | 0.740063 | 0.662802 | 0.838467 | 0.839435 |

Source positions are strongly readable with the shared AFTER-fitted scalar probe at the edge reveal. Correct inverse maps exceed the eleven common mapping controls in 4/4 units under each encoding, but remain below both positional priors in 4/4; matched support and the cycle both-internal sensitivity retain that limitation. This is recoverability under the specified probe, not evidence that BEFORE contains no decodable information. For a scalar affine probe, means-only adds a position-constant offset, so its per-anchor AUROC equals direct transfer by construction.

See [token-stage report](token_report.md), [four-unit raw table](token_four_unit_table.csv), [SD/range statistics](token_four_unit_stats.csv) and [coverage](token_coverage_by_length.csv). These scalar local-state results do not replace or change the final-state vector-probe comparison.

One-line extraction uses the value token in the corresponding slot. Canonical-cycle extraction uses the second endpoint for an internal edge, and the closing delimiter for a closing edge or singleton. Before/after denotes local explicit serialization reveal, **not earliest logical entailment** from all possible permutation completions. The before state can contain useful canonical-order/position priors. Closure/singleton strata can be structurally single-class; their undefined AUROC must remain visible. A shared scalar source probe fitted on after-states is applied to both sides of the edge-reveal point; differences also reflect that fixed readout and distribution change. Token-kind annotations use the complete input, so kind priors at BEFORE are contextual diagnostic references, not guaranteed online baselines. Inverse-edge pairing is provided by the analysis.

## Artifacts and reproducibility

- [Main report and figures](summary/report.md), [count accuracy diagnostics](summary/model_diagnostics_equal_length.csv), [all per-length scores](summary/all_pair_by_length.csv), and [finite-sample coverage](summary/coverage.csv).
- [Frozen protocol](protocol.json), [frozen source hashes](freeze.json), [cache/actual checkpoint audit](cache_audit.json), [math audit](math_design_audit.md), and [independent saved-main reconstruction](main_saved_score_audit.json).
- [Reproduction commands](reproduce.sh). Large final-state score files are under `/mnt/yangx-projects-f/20261110/permworld_exceedance_alignment/main/<unit>/` for ED and `pv_scores/<unit>/` for rebuilt PV. Each contains unthresholded per-position `scores`, `per_sample_auroc`, method names, labels, original anchor indices, stable sample IDs, lengths and support masks. Padding is NaN and valid position lengths are explicit. Readout/scaler/map parameters and checkpoint/data bindings are saved alongside.
- [Source-only and map fit records](results/): all five unit receipts, alphas, shuffle permutations, actual rotations and output hashes. PV raw score reconstruction checks every original trained cell within 1e-12 before export. ED raw score/ridge/mean/gather reconstruction passes at the same fixed tolerance without refitting.

Only the secondary token diagnostic needs new frozen checkpoint forwards. The main experiment uses existing prefix_final caches; no counting specialist is retrained, no new world is generated, and no old report, paper or F17 artifact is edited. Results remain local unless separately uploaded.
