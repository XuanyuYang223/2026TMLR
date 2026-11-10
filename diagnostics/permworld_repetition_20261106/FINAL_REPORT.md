# Completed five-unit PermWorld repetition

Completed **9 November 2026 (America/Los_Angeles)**. All **256 additional specialists** reached their fixed 20k endpoints; all four new main analyses and the five-unit summary completed. Including the original 64 specialists, this is **320 trained endpoints in five units**. Training has stopped, and no follow-up experiment was launched.

The supported finding is repeated prevalence of positive pairing selectivity under the declared shared-initialization design, with reproducible increment directions and important condition-specific instability. The observations do not identify a uniquely mathematical mechanism or a functional benefit.

## Primary results

These counts use joint-answer strata, the final normalized hidden state before the task token, ordinary linear CKA, and separate n=10–30 calculations with fixed 1/21 weights. Each denominator is **32 dependent conditions within a unit**.

| Unit | A: Bt > 0 | A: Delta > 0 | B: above other-candidate mean | B: uniquely first |
|---|---:|---:|---:|---:|
| Historical exploratory unit | 26/32 | 18/32 | 29/32 | 19/32 |
| unit02 | 28/32 | 18/32 | 30/32 | 18/32 |
| unit03 | 29/32 | 18/32 | 30/32 | 17/32 |
| unit04 | 28/32 | 18/32 | 30/32 | 20/32 |
| unit05 | 26/32 | 18/32 | 29/32 | 18/32 |

## Four new units, excluding the exploratory unit

| Count statistic | Units | Mean | Sample SD | Range |
|---|---:|---:|---:|---:|
| A: positive Bt | 4 | 27.75 | 1.258306 | 26–29 |
| A: positive Delta | 4 | 18.00 | 0.000000 | 18–18 |
| B: designated above other mean | 4 | 29.75 | 0.500000 | 29–30 |
| B: designated uniquely first | 4 | 18.25 | 1.258306 | 17–20 |

## All five units, including the exploratory unit

| Count statistic | Units | Mean | Sample SD | Range |
|---|---:|---:|---:|---:|
| A: positive Bt | 5 | 27.40 | 1.341641 | 26–29 |
| A: positive Delta | 5 | 18.00 | 0.000000 | 18–18 |
| B: designated above other mean | 5 | 29.60 | 0.547723 | 29–30 |
| B: designated uniquely first | 5 | 18.40 | 1.140175 | 17–20 |

These are unit-level descriptives, not pooled binomial trials. The two cohort views overlap; they are not independent comparison groups.

## Continuous effects and direction stability

Across the four new units, Bt directions agree in **26/32 conditions**, Delta directions in **32/32**, and fixed-transform designated-minus-other-mean directions in **28/32**. The same agreement counts hold with the historical unit included. The same 18 conditions have positive Delta throughout; constant counts do not mean constant effect magnitudes.

| Mean over the 32 dependent conditions within each new unit | Across-unit mean | Across-unit sample SD |
|---|---:|---:|
| A: Bt | 0.163883 | 0.013023 |
| A: Delta | 0.211308 | 0.008777 |
| B: trained designated-minus-other-mean | 0.167206 | 0.013159 |

Each observation here is a complete unit's fixed-grid mean. Averaging heterogeneous conditions does not establish that every condition is positive or stable. Full-precision per-unit and per-condition values remain in the CSVs.

A counterexample to uniform stability is canonical-cycle GRU global-descents/components. Bt is **−0.03976771085546295, −0.13231985497482063, +0.13054923656741776, −0.0051350403094189275** across the four new units: a range of **0.26286909154223836**. All disagreements are retained; they cannot all be described as tiny signs near zero. Stable Delta directions also do not prove that correct-pair CKA itself increased: Delta is the change of a correct-minus-wrong contrast.

The scope remains the 16 properties, two declared encodings/configurations, this sampler and fixed historical exclusion set, final-state measurement, and prescribed metric/control versions. Other metrics need not reproduce the ordinary-CKA direction grid. Data, architecture-level starts, batches and training streams change jointly, so their individual causal contributions are not isolated. Shared initialization across specialists within each unit is preserved; independently initialized specialists are not tested. No practical-equivalence threshold, new p value, or functional claim was introduced.

## Execution and verification

- Four fixed new seed rows, 64 models per unit, unchanged sampler/model/training/endpoint protocol; no replacement seeds or learning-based exclusions.
- The original four-task versus twelve-task training-RNG sharing policy is retained. Evaluation has no separate random sampler; its anchors are sequential data-seed draws.
- All 256 new checkpoint byte hashes match their receipts. The audit verified eight genuine initialization state dictionaries, 32 saved start-RNG states, all seed/data/batch/init bindings, 60 frozen source files, and 151 specifically named original files. The intentionally updated entry point and working manuscript are excluded from the old-file protection claim.
- One interrupted worker resumed its saved 18k optimizer/RNG state without replacing its seed or endpoint. Engineering copies and their 40-update data overlap are documented in the manuscript and excluded from formal endpoint counts.
- The finite-scope original-activation implementation comparison passed 2,096 checks; it does not independently recompute every CKA. Final saved-table arithmetic and all candidate rankings were audited separately.

## Deliverables

- [Separate new-four and all-five reader](cohort_summary/report.md).
- [All per-unit counts](summary/unit_counts.csv), [unrounded effects](summary/condition_effects_by_unit.csv), [condition-level summaries](summary/condition_effects_summary.csv), and [all-unit figure](summary/condition_effects_by_unit.png).
- [Execution audit](final_execution_audit.json) and [independent saved-table audit](final_saved_results_audit.json).
- [Protocol](protocol.json), [fixed 256-job seed manifest](training_manifest.csv), [exact execution-source snapshot](execution_source_archive.json), and [reproduction commands](README.md).
- [English manuscript PDF](../../manuscripts/representation_selectivity_20261012/manuscript.pdf), [LaTeX source](../../manuscripts/representation_selectivity_20261012/manuscript.tex), [self-contained source ZIP](../../manuscripts/representation_selectivity_20261012/manuscript_source.zip), and [build receipt](paper_finalization.json).

The limited independent-specialist-initialization proposal and historical cost basis are in [ORIGINAL_AUDIT.md](ORIGINAL_AUDIT.md). That proposal remains unexecuted. No extra worlds, larger models, GroupWorld, matrix/F17 experiments, automatic Git upload or release publication were started.
