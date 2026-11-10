# Exploratory length-and-answer-control diagnostic

This report summarizes saved scores only. It does not establish a mathematical mechanism or functional benefit. The new transform advantage is designated-transform CKA minus the maximum over every eligible alternative transform, including identity; mathematically also-correct alternatives are excluded by the frozen formula catalog. This is not the old one-wrong-transform Bt, and its trained-minus-initial change is not the old Delta.

The pooled ordinary-CKA comparator is newly requested, rather than the frozen paper primary analysis (which was already within length and joint-answer stratified). Moving from pooled ordinary CKA to within-length U-CKA changes both length handling and estimator. Moving from within-length U-CKA to dual-answer residual U-CKA adds the specified additive one-hot regression.

## Prespecified task-pair bootstrap

| Cohort | Version | Effect | Mean | 95% percentile interval | Pairs |
|---|---|---|---:|---:|---:|
| new_units | pooled_raw | trained_transform_advantage | 0.317309 | 0.184564 to 0.449421 | 8 |
| new_units | pooled_raw | advantage_training_increment | 0.454961 | 0.313572 to 0.590450 | 8 |
| new_units | within_length_debiased | trained_transform_advantage | 0.343678 | 0.250223 to 0.446950 | 8 |
| new_units | within_length_debiased | advantage_training_increment | 0.971469 | 0.875258 to 1.079875 | 8 |
| new_units | within_length_dual_answer_debiased | trained_transform_advantage | 0.125783 | 0.078508 to 0.175252 | 8 |
| new_units | within_length_dual_answer_debiased | advantage_training_increment | 0.754125 | 0.689672 to 0.819765 | 8 |
| all_units | pooled_raw | trained_transform_advantage | 0.321306 | 0.186564 to 0.455308 | 8 |
| all_units | pooled_raw | advantage_training_increment | 0.453160 | 0.312305 to 0.590617 | 8 |
| all_units | within_length_debiased | trained_transform_advantage | 0.343775 | 0.249179 to 0.448467 | 8 |
| all_units | within_length_debiased | advantage_training_increment | 0.973570 | 0.876072 to 1.083045 | 8 |
| all_units | within_length_dual_answer_debiased | trained_transform_advantage | 0.126393 | 0.075537 to 0.179627 | 8 |
| all_units | within_length_dual_answer_debiased | advantage_training_increment | 0.756816 | 0.688148 to 0.826265 | 8 |

Bootstrap: 10,000 draws of the 8 prescribed task-pair blocks, seed 20261109. All architectures, encodings and stochastic units stay inside their selected task-pair block. This small, exploratory interval is conditional on the finite task catalog and its two transform families, not a population guarantee over mathematical relationships. Unit repetitions do not increase the number of task pairs. Undefined required lengths, alternatives or grid components are not filled with zero or silently reweighted.

## Cells and sensitivity

| Cohort | Version | Cells positive in all registered units | Fixed cells |
|---|---|---:|---:|
| new_units | pooled_raw | 26 | 32 |
| new_units | within_length_debiased | 25 | 32 |
| new_units | within_length_dual_answer_debiased | 22 | 32 |
| all_units | pooled_raw | 25 | 32 |
| all_units | within_length_debiased | 24 | 32 |
| all_units | within_length_dual_answer_debiased | 20 | 32 |

Full per-unit scores, ranks, initial references, contrast changes, cell means/ranges and every exclusion remain in CSV. The largest absolute deletions are marked within task-pair, architecture and encoding exclusion types; this identifies influence rather than selecting a favorable retained subset.

| Cohort | Residual effect | Exclusion type | Largest absolute influence | Retained mean | Change from full |
|---|---|---|---|---:|---:|
| new_units | trained_transform_advantage | task_pair | peak_complement | 0.108430 | -0.017353 |
| new_units | trained_transform_advantage | architecture | transformer | 0.195874 | 0.070092 |
| new_units | trained_transform_advantage | encoding | canonical_cycle | 0.221953 | 0.096170 |
| new_units | advantage_training_increment | task_pair | peak_complement | 0.733397 | -0.020728 |
| new_units | advantage_training_increment | architecture | gru | 0.694798 | -0.059327 |
| new_units | advantage_training_increment | encoding | one_line | 0.816202 | 0.062077 |
| new_units | advantage_training_increment | encoding | canonical_cycle | 0.692048 | -0.062077 |
| all_units | trained_transform_advantage | task_pair | peak_complement | 0.107451 | -0.018941 |
| all_units | trained_transform_advantage | architecture | transformer | 0.194285 | 0.067892 |
| all_units | trained_transform_advantage | architecture | gru | 0.058501 | -0.067892 |
| all_units | trained_transform_advantage | encoding | canonical_cycle | 0.222759 | 0.096367 |
| all_units | advantage_training_increment | task_pair | peak_complement | 0.734583 | -0.022233 |
| all_units | advantage_training_increment | architecture | transformer | 0.815505 | 0.058689 |
| all_units | advantage_training_increment | encoding | one_line | 0.818263 | 0.061447 |

## Coding-simplicity hypothesis

| Cohort | Version | Effect | Simple minus scrambling | Pair-bootstrap interval |
|---|---|---|---:|---:|
| new_units | pooled_raw | trained_transform_advantage | 0.184650 | -0.019391 to 0.418152 |
| new_units | pooled_raw | advantage_training_increment | 0.064404 | -0.113924 to 0.285823 |
| new_units | within_length_debiased | trained_transform_advantage | 0.238847 | -0.019908 to 0.507931 |
| new_units | within_length_debiased | advantage_training_increment | -0.470606 | -0.757563 to -0.180800 |
| new_units | within_length_dual_answer_debiased | trained_transform_advantage | 0.184336 | 0.076233 to 0.304712 |
| new_units | within_length_dual_answer_debiased | advantage_training_increment | -0.523980 | -0.659520 to -0.384751 |
| all_units | pooled_raw | trained_transform_advantage | 0.182252 | -0.018999 to 0.418825 |
| all_units | pooled_raw | advantage_training_increment | 0.071161 | -0.106682 to 0.296780 |
| all_units | within_length_debiased | trained_transform_advantage | 0.227349 | -0.037609 to 0.502786 |
| all_units | within_length_debiased | advantage_training_increment | -0.478344 | -0.771732 to -0.182578 |
| all_units | within_length_dual_answer_debiased | trained_transform_advantage | 0.181768 | 0.072528 to 0.302472 |
| all_units | within_length_dual_answer_debiased | advantage_training_increment | -0.522601 | -0.658708 to -0.384592 |

The simple/scrambling labels were specified from the input encoding, not chosen from CKA: complement in one-line is token relabeling; inverse in canonical cycles reverses each cycle orientation about its fixed minimum. Complement in canonical cycles and inverse in one-line are the prespecified scrambling cases. Each simplicity contrast pairs the two encodings within a task pair, architecture and stochastic unit before the task-pair bootstrap; it is not an unpaired comparison between the six complement and two inverse pairs. This is a coding-description hypothesis, not proof of matched computational difficulty or an independently identified cause.

## Learning and peaks/valleys

The source-learning CSVs must distinguish full-vocabulary NLL from numeric-conditional NLL. A model is not declared at the length baseline solely from exact-match accuracy; prior-relative NLL and expectation–truth rank correlation are retained continuously. No arbitrary near-baseline tolerance was selected after observing results. Cell-level learning associations are descriptive because cells share tasks and models.

For GRU / canonical cycles, `peaks_valleys_rankings.csv` retains valleys' rank and exact-tie count at identity, complement and inverse, for every version and available initial/trained unit. Rank one with fourteen exact ties at shared initialization is not unique task identification. Comparisons across transformations distinguish task identity from transformation-specific selectivity.

| Cohort | Phase | CKA version | Transform | Finite units | Rank range | Uniquely first | Maximum ties |
|---|---|---|---|---:|---:|---:|---:|
| new_units | initial | pooled_raw | identity | 4 | 1.000000–1.000000 | 0/4 | 14 |
| new_units | initial | pooled_raw | complement | 4 | 1.000000–1.000000 | 0/4 | 14 |
| new_units | initial | pooled_raw | inverse | 4 | 1.000000–1.000000 | 0/4 | 14 |
| new_units | initial | within_length_debiased | identity | 4 | 1.000000–1.000000 | 0/4 | 14 |
| new_units | initial | within_length_debiased | complement | 4 | 1.000000–1.000000 | 0/4 | 14 |
| new_units | initial | within_length_debiased | inverse | 4 | 1.000000–1.000000 | 0/4 | 14 |
| new_units | initial | within_length_dual_answer_debiased | identity | 4 | 1.000000–1.000000 | 0/4 | 14 |
| new_units | initial | within_length_dual_answer_debiased | complement | 4 | 10.000000–14.000000 | 0/4 | 0 |
| new_units | initial | within_length_dual_answer_debiased | inverse | 4 | 4.000000–10.000000 | 0/4 | 0 |
| new_units | trained | pooled_raw | identity | 4 | 1.000000–3.000000 | 2/4 | 0 |
| new_units | trained | pooled_raw | complement | 4 | 1.000000–2.000000 | 3/4 | 0 |
| new_units | trained | pooled_raw | inverse | 4 | 1.000000–3.000000 | 3/4 | 0 |
| new_units | trained | within_length_debiased | identity | 4 | 1.000000–3.000000 | 1/4 | 0 |
| new_units | trained | within_length_debiased | complement | 4 | 1.000000–2.000000 | 2/4 | 0 |
| new_units | trained | within_length_debiased | inverse | 4 | 10.000000–11.000000 | 0/4 | 0 |
| new_units | trained | within_length_dual_answer_debiased | identity | 4 | 1.000000–4.000000 | 1/4 | 0 |
| new_units | trained | within_length_dual_answer_debiased | complement | 4 | 1.000000–2.000000 | 2/4 | 0 |
| new_units | trained | within_length_dual_answer_debiased | inverse | 4 | 8.000000–13.000000 | 0/4 | 0 |
| all_units | initial | pooled_raw | identity | 5 | 1.000000–1.000000 | 0/5 | 14 |
| all_units | initial | pooled_raw | complement | 5 | 1.000000–1.000000 | 0/5 | 14 |
| all_units | initial | pooled_raw | inverse | 5 | 1.000000–1.000000 | 0/5 | 14 |
| all_units | initial | within_length_debiased | identity | 5 | 1.000000–1.000000 | 0/5 | 14 |
| all_units | initial | within_length_debiased | complement | 5 | 1.000000–1.000000 | 0/5 | 14 |
| all_units | initial | within_length_debiased | inverse | 5 | 1.000000–1.000000 | 0/5 | 14 |
| all_units | initial | within_length_dual_answer_debiased | identity | 5 | 1.000000–1.000000 | 0/5 | 14 |
| all_units | initial | within_length_dual_answer_debiased | complement | 5 | 10.000000–14.000000 | 0/5 | 0 |
| all_units | initial | within_length_dual_answer_debiased | inverse | 5 | 4.000000–10.000000 | 0/5 | 0 |
| all_units | trained | pooled_raw | identity | 5 | 1.000000–3.000000 | 2/5 | 0 |
| all_units | trained | pooled_raw | complement | 5 | 1.000000–2.000000 | 4/5 | 0 |
| all_units | trained | pooled_raw | inverse | 5 | 1.000000–3.000000 | 3/5 | 0 |
| all_units | trained | within_length_debiased | identity | 5 | 1.000000–3.000000 | 2/5 | 0 |
| all_units | trained | within_length_debiased | complement | 5 | 1.000000–2.000000 | 3/5 | 0 |
| all_units | trained | within_length_debiased | inverse | 5 | 10.000000–12.000000 | 0/5 | 0 |
| all_units | trained | within_length_dual_answer_debiased | identity | 5 | 1.000000–4.000000 | 2/5 | 0 |
| all_units | trained | within_length_dual_answer_debiased | complement | 5 | 1.000000–2.000000 | 3/5 | 0 |
| all_units | trained | within_length_dual_answer_debiased | inverse | 5 | 8.000000–13.000000 | 0/5 | 0 |

Raw shared-initialization candidate ties need not survive candidate-dependent answer regression; the initial residual ranks are measured rather than assumed tied.

| Unit | Task | Accuracy minus length mode | Full NLL minus length-prior NLL | Conditional-numeric NLL minus prior | Expected-answer Spearman |
|---|---|---:|---:|---:|---:|
| archdev001 | peaks | 0.008743 | -0.042960 | -0.043093 | 0.271547 |
| archdev001 | valleys | 0.017764 | -0.043558 | -0.043657 | 0.277039 |
| unit02 | peaks | 0.003162 | -0.011143 | -0.011215 | 0.127914 |
| unit02 | valleys | 0.017113 | -0.043046 | -0.043141 | 0.275279 |
| unit03 | peaks | 0.000930 | -0.015520 | -0.015604 | 0.157747 |
| unit03 | valleys | 0.012835 | -0.064664 | -0.064832 | 0.322559 |
| unit04 | peaks | 0.011812 | -0.031920 | -0.032032 | 0.234755 |
| unit04 | valleys | 0.014509 | -0.036309 | -0.036429 | 0.251119 |
| unit05 | peaks | 0.012649 | -0.036764 | -0.036878 | 0.247612 |
| unit05 | valleys | 0.027530 | -0.062086 | -0.062225 | 0.329735 |

Learning entries above are identity-input, fixed-length summaries. Lower model-minus-prior NLL is better; numeric-conditional NLL is a separate quantity, never a replacement for unavailable full-vocabulary NLL. Undefined Spearman can result from a constant answer at a required length and does not mean zero ability. No near-baseline equivalence margin was registered, so these values are not converted into a binary claim that a specialist uses no input information.

## Interpretation limits

No practical disappearance/equivalence bound was specified. A confidence interval crossing zero is not evidence that the effect is absent. A residual advantage, if present, supports structure beyond the prescribed additive label regressors; it does not remove joint-answer interactions, confidence, errors, input geometry or every nonlinear output feature. Using unbiased HSIC components does not by itself turn in-sample fitted residuals into an unbiased conditional-independence test. Uniform positivity, magnitude, initialization-relative changes and deletion sensitivity must be read together before advancing to probes. No probe or new-model training is launched here.
