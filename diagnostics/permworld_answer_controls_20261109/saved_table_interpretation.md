# Saved-table interpretation: length and additive-answer controls

This is a secondary reading of the completed frozen pipeline output. It reads saved scalar tables only: no model or activation processing, new CKA/regression/bootstrap/ranking calculation, training, or probes. The scalar decompositions below use descriptive arithmetic on saved columns. They do not alter the protocol, frozen code, or original reports.

The new advantage is designated-transform CKA minus the maximum CKA over the other eligible transforms, including identity. It differs from the paper's original correct-minus-one-wrong `Bt`; its trained-minus-initial increment also differs from the original `Delta`. The finite catalog comprises eight prescribed task pairs, two architectures, two encodings, and five stochastic units. The original unit remains exploratory and is reported separately from the four new units.

## Does a residual advantage remain?

Yes, within the precise scope of fixed length and the prescribed additive one-hot answer regressors. It is heterogeneous and is not a mechanism claim.

| Saved quantity | Four new units | All five units |
|---|---:|---:|
| Mean trained residual strongest-alternative advantage | 0.125783 | 0.126393 |
| Exploratory 95% task-pair bootstrap interval | [0.078508, 0.175252] | [0.075537, 0.179627] |
| Cells positive in every included unit | 22/32 | 20/32 |
| Mean initialization-relative increment | 0.754125 | 0.756816 |
| Cells with a positive increment in every included unit | 32/32 | 32/32 |

These percentile intervals resample eight task-pair blocks, retain all conditions and units within a block, and are conditional on this finite catalog and its two transformation families. They do not turn 32 conditions or four/five seeds into independent mathematical relationships. There is no registered practical-null margin, so neither a small effect nor a sign flip is relabeled as zero.

Adding dual-answer regression to within-length U-CKA changes the four-new-unit mean from 0.343678 to 0.125783. The saved paired change is −0.217895, with task-pair interval [−0.324702, −0.111301]. This substantial attenuation and the retained positive mean should both be reported. Pooled ordinary CKA is a newly requested comparator, not the old paper primary analysis; pooled-to-within also changes the estimator, so it cannot isolate a length contribution by itself.

## Broad across pairs, concentrated across configurations

Every task-pair average is positive after averaging its four architecture/encoding conditions and four new units:

| Pair | New-four residual advantage |
|---|---:|
| descent_inverse | 0.074981 |
| peak_complement | 0.247254 |
| fixed_complement | 0.062527 |
| exceedance_inverse | 0.068703 |
| subsequence_complement | 0.207351 |
| run_complement | 0.143325 |
| record_complement | 0.157397 |
| decomposition_complement | 0.044724 |

Deleting one task pair leaves a mean from 0.108430 to 0.137363; deleting peaks/valleys gives the lowest retained mean. Thus no single pair is necessary for the aggregate sign. This is weaker than positivity of every configuration. One-line conditions average 0.221953 versus canonical-cycle conditions 0.029613; GRU averages 0.195874 versus Transformer 0.055691. Removing either architecture or encoding retains a positive descriptive mean, with strongly different magnitudes.

The complete new-four cell table follows; its ranges are across stochastic units, not confidence intervals. Negative and mixed cells remain visible.

| Pair | Architecture | Encoding | Mean | Unit minimum | Unit maximum | Positive units |
|---|---|---|---:|---:|---:|---:|
| descent_inverse | transformer | one_line | 0.142968 | 0.070914 | 0.183512 | 4/4 |
| descent_inverse | transformer | canonical_cycle | -0.020964 | -0.041745 | 0.008016 | 1/4 |
| descent_inverse | gru | one_line | 0.047992 | 0.034963 | 0.058132 | 4/4 |
| descent_inverse | gru | canonical_cycle | 0.129927 | 0.069620 | 0.171391 | 4/4 |
| peak_complement | transformer | one_line | 0.094333 | 0.050722 | 0.159613 | 4/4 |
| peak_complement | transformer | canonical_cycle | 0.070931 | 0.025085 | 0.102934 | 4/4 |
| peak_complement | gru | one_line | 0.773215 | 0.582109 | 0.854218 | 4/4 |
| peak_complement | gru | canonical_cycle | 0.050535 | -0.070032 | 0.187756 | 2/4 |
| fixed_complement | transformer | one_line | 0.053167 | 0.027584 | 0.075763 | 4/4 |
| fixed_complement | transformer | canonical_cycle | -0.002147 | -0.024802 | 0.017235 | 2/4 |
| fixed_complement | gru | one_line | 0.194169 | 0.184052 | 0.207322 | 4/4 |
| fixed_complement | gru | canonical_cycle | 0.004919 | -0.020411 | 0.017439 | 3/4 |
| exceedance_inverse | transformer | one_line | 0.108908 | 0.084486 | 0.124298 | 4/4 |
| exceedance_inverse | transformer | canonical_cycle | 0.049600 | 0.005937 | 0.111837 | 4/4 |
| exceedance_inverse | gru | one_line | 0.019517 | 0.010580 | 0.037189 | 4/4 |
| exceedance_inverse | gru | canonical_cycle | 0.096787 | 0.039588 | 0.171300 | 4/4 |
| subsequence_complement | transformer | one_line | 0.125377 | 0.079544 | 0.167074 | 4/4 |
| subsequence_complement | transformer | canonical_cycle | -0.007705 | -0.055719 | 0.102856 | 1/4 |
| subsequence_complement | gru | one_line | 0.476988 | 0.460481 | 0.502124 | 4/4 |
| subsequence_complement | gru | canonical_cycle | 0.234745 | 0.167977 | 0.274688 | 4/4 |
| run_complement | transformer | one_line | 0.149189 | 0.056670 | 0.198537 | 4/4 |
| run_complement | transformer | canonical_cycle | -0.007453 | -0.048843 | 0.019143 | 2/4 |
| run_complement | gru | one_line | 0.643877 | 0.554674 | 0.695322 | 4/4 |
| run_complement | gru | canonical_cycle | -0.212312 | -0.241309 | -0.170428 | 0/4 |
| record_complement | transformer | one_line | 0.108971 | 0.091010 | 0.120636 | 4/4 |
| record_complement | transformer | canonical_cycle | 0.012647 | -0.057890 | 0.057133 | 3/4 |
| record_complement | gru | one_line | 0.366871 | 0.297810 | 0.443991 | 4/4 |
| record_complement | gru | canonical_cycle | 0.141101 | 0.130028 | 0.157315 | 4/4 |
| decomposition_complement | transformer | one_line | 0.049080 | 0.011146 | 0.116853 | 4/4 |
| decomposition_complement | transformer | canonical_cycle | -0.035842 | -0.067738 | -0.015815 | 0/4 |
| decomposition_complement | gru | one_line | 0.196623 | 0.107445 | 0.340891 | 4/4 |
| decomposition_complement | gru | canonical_cycle | -0.030963 | -0.156213 | 0.105967 | 1/4 |

All 16 one-line cells are positive in every new unit. In canonical cycles, six of sixteen are positive throughout. Two cells are negative in all four new units: GRU run complement (mean −0.212312, range [−0.241309, −0.170428]) and Transformer decomposition complement (mean −0.035842, range [−0.067738, −0.015815]). Other cycle effects include small or mixed values; they cannot all be dismissed as numerical noise. Adding the historical unit breaks uniform positivity for GRU/cycle descent–inverse (historical −0.071994) and GRU/one-line exceedance–inverse (historical −0.004754). Their substantially different magnitudes should remain explicit.

## Coding simplicity and initialization answer different questions

The frozen simplicity comparison is paired within task pair, architecture, and unit: complement/one-line and inverse/canonical-cycle are labeled simple; the opposite encodings are labeled scrambling. It is a coding-description hypothesis, not matched computational cost or a causal encoding experiment.

| Residual simple minus scrambling | Four new units | All five units |
|---|---:|---:|
| Trained advantage | 0.184336 [0.076233, 0.304712] | 0.181768 [0.072528, 0.302472] |
| Training increment | −0.523980 [−0.659520, −0.384751] | −0.522601 [−0.658708, −0.384592] |

The trained-state observation supports higher selectivity in the prespecified simple encoding on average. The opposite sign for the increment means that the same statement does not describe change from initialization: scrambling conditions start much further below their identity comparator. A large increment cannot be read as a larger final effect or a larger increase of correct CKA.

The following arithmetic decomposes the mean residual advantage from saved designated/strongest-alternative columns:

| Component | Four-new initial | Four-new trained | Change | All-five initial | All-five trained | Change |
|---|---:|---:|---:|---:|---:|
| Designated-transform CKA | 0.371658 | 0.219127 | -0.152531 | 0.369577 | 0.221484 | -0.148093 |
| Strongest-alternative CKA | 1.000000 | 0.093344 | -0.906656 | 1.000000 | 0.095091 | -0.904909 |
| Advantage | -0.628342 | 0.125783 | 0.754125 | -0.630423 | 0.126393 | 0.756816 |

Identity is the strongest initial alternative in every saved condition. The large positive increment is predominantly a fall in this shared-initialization identity similarity. Mean designated CKA itself decreases. Training changes relative preference; it does not generally make the designated states absolutely more similar. The maximizing alternative can change after training, so this is a strongest-alternative contrast decomposition, not a fixed-control causal decomposition.

## GRU/canonical-cycle peaks–valleys

Rows below give valleys' rank among the 15 candidates for source peaks; columns use fixed unit order. There are no exact trained ties in these rows.

| Version / transform | Historical | unit02 | unit03 | unit04 | unit05 |
|---|---:|---:|---:|---:|---:|
| pooled_raw / identity | 2 | 3 | 1 | 2 | 1 |
| pooled_raw / complement | 1 | 2 | 1 | 1 | 1 |
| pooled_raw / inverse | 2 | 3 | 1 | 1 | 1 |
| within_length_debiased / identity | 1 | 2 | 3 | 1 | 2 |
| within_length_debiased / complement | 1 | 2 | 2 | 1 | 1 |
| within_length_debiased / inverse | 12 | 11 | 10 | 11 | 10 |
| within_length_dual_answer_debiased / identity | 1 | 3 | 4 | 1 | 3 |
| within_length_dual_answer_debiased / complement | 1 | 2 | 2 | 1 | 1 |
| within_length_dual_answer_debiased / inverse | 12 | 11 | 11 | 13 | 8 |

With fixed length and additive-answer regression, complement ranks 2, 2, 1, 1 in the four new units; identity ranks 3, 4, 1, 3 and inverse ranks 11, 11, 13, 8. Valleys therefore remains near the top under complement but is not uniformly first, and its occasional first rank under identity is not transformation-specific. More decisively, the complement-minus-best-alternative advantage is negative in unit02 (−0.070032) and unit03 (−0.064361), positive in unit04 (0.148778) and unit05 (0.187756), despite all four having large positive increments from initialization. A favorable candidate rank and a favorable transform advantage answer different questions.

The identity-input learning records show negative full-NLL gaps for both specialists in all five units: peaks ranges from −0.042960 to −0.011143 nats and valleys from −0.064664 to −0.036309 nats. Expected-answer Spearman is positive throughout: peaks 0.127914–0.271547, valleys 0.251119–0.329735. Their exact-match gaps are small, but these probability/correlation diagnostics do not support treating them as identical to a no-input length baseline. They also do not establish task mastery or performance equivalence, and they describe original identity inputs rather than every transformed input.

## How initial ranks should be read

For a fixed transform, every initial right specialist has the same parameters within a unit/architecture/encoding. In the two uncorrected versions, all 15 candidate ranks are therefore analytically tied: the candidate state is identical across tasks. Rank one is not unique identification, at identity or either other transform. At identity, left and right states are also equal, so every defined self-CKA is mathematically 1.

Dual-answer regression changes its design with the candidate. At nonidentity transforms it can consequently induce initial candidate scores/ranks without learned specialist information. At identity, however, each pair still residualizes two equal states using the same design, so all candidates remain analytically tied at self-CKA 1. The separate bounded numerical audit found that packed ordinary U-CKA can assign strict numerical ranks from differences of a few floating-point ulps. These exact stored ranks must not be interpreted as initial task preference. This note clarifies the frozen report's general sentence about candidate-dependent initial residual ranks: it applies to nonidentity comparisons, not mathematical self-CKA identity ties.

## What remains unresolved

The residual result excludes the specified additive one-hot answer means as a complete explanation of the observed average advantage. It does not eliminate joint-answer interactions, confidence, prediction errors, input geometry, or arbitrary nonlinear encodings of answers. U-centered HSIC components after in-sample residual fitting are not automatically an unbiased conditional-independence test. Across the 32 new-unit cell means, the saved descriptive correlation of residual advantage with pair-average full-NLL gap is −0.484971 and with pair-average learning Spearman is 0.499633; cells share tasks and models, so these are not independent replications or causal learning effects.

The appropriate narrow result is: **within the tested shared-initialization PermWorld design, a positive transform advantage survives fixed-length additive-answer control on average and across many cells; its magnitude and reliability depend strongly on encoding and configuration. The large change from initialization mostly reflects reduced identity similarity rather than an increase of correct-transform CKA.** No universal relation mechanism, complete removal of output information, or functional benefit follows. No further experiment is launched by this interpretation.
