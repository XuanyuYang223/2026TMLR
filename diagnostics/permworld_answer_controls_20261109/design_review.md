# Mathematical and statistical design review

This is a pre-computation mathematical audit for the checkpoint-only answer-control diagnostic. It uses the task registry and exact property functions, not neural checkpoints, activations, new CKA values, or optimization. The eight original relations and their reciprocal directions were classified before the new similarity analysis. See [math_classification.json](math_classification.json) and [formula_validation.json](formula_validation.json).

## Corrections to the starting description

The existing primary analysis is already **separate ordinary CKA within each n, equal weights 1/21 over n=10–30**, with a separate joint-answer-stratified version. It is not a pooled-across-lengths CKA. The requested pooled ordinary CKA should therefore be named a new descriptive baseline rather than a reproduction of the original primary endpoint.

The old joint-answer control forms keys from n, the left task's identity answer, and all three right-task answers (identity, complement, inverse). It keeps strata with at least three samples and subtracts their means. The new control instead keeps the common full anchor set and regresses on the two answers actually used in a candidate/transform comparison. These controls answer related but different questions; neither dominates the other in every respect.

Seven registered pairs have exact correct-role answer equality. The eighth satisfies `components(C(p)) = global_descents(p) + 1`. Its category labels have the same one-hot column span after a bijective shift, despite not being numerically identical. Population distribution equality under a bijection does not imply identical finite empirical histograms of A(p) and B(p) on a shared random identity-anchor sample. Correct-role per-sample equality/offset is exact.

The original repeated-study result also does not mean every cell is stable: six Bt directions change across its five units. Four data/initialization/training repetitions support a within-design repeated observation, not eight or sixteen independent task families.

## Exact formulas and eligible alternative transforms

The authoritative registry supplies eight directed relationships covering sixteen distinct tasks. Their correct transform is either complement or inverse. The scalar formula is

```text
B(T(p)) = A(p) + right_label_offset
```

All eight transforms are involutions. The reverse-directed relationship has the same transform and the opposite offset. The sixteen-source candidate grid therefore has sixteen natural designated directions, but only **eight unordered task-pair identities**: eight original primary directions and eight reciprocal supplemental directions. Reciprocal rows must not double the bootstrap unit count.

Full enumeration of every permutation in S1 through S8 checked **46,233 permutations**, all **720 off-diagonal task/transform triples**, and the canonical-cycle inverse construction. All sixteen proved equality/offset directions passed. No other off-diagonal tested equality, or per-n constant-offset equality, survived this finite audit. A positive finite audit is not itself a universal proof; the sixteen catalog entries also contain analytical proof sketches. Counterexamples refute only a particular tested equality, not every possible mathematical relationship.

For every original pair, only its registered transform satisfies its declared equality/offset among identity, complement, and inverse. Thus both remaining transforms are eligible alternatives for this specific formula. No transform is excluded based on its observed CKA. The classification JSON retains the first explicit counterexample for each failed formula.

The identity peaks/valleys comparison has an important additional relationship: `abs(peaks(p) - valleys(p)) <= 1`. Interior extrema are alternating switches in adjacent up/down comparisons. This bound passed the same full enumeration and follows directly from that alternation. Identity remains an eligible alternative to the **specific exact-equality** complement formula, but is not a mathematically unrelated control. This strong answer association is particularly relevant when valleys ranks highly among peaks' candidates without complementing the input.

Seven selected tasks are invariant under inverse in the finite audit: fixed and anti-fixed points, LIS and LDS lengths, global descents, components, and left-to-right minima. They are self-task relations, so they are not extra off-diagonal candidates. Their mathematical mechanisms are point transposition, direct/skew decomposition, subsequence transposition, and empty lower-left plot rectangles. They must not be used to claim that every non-designated transformation destroys all structure.

## Prespecified encoding-operation classes

| Encoding | Operation | Class | Exact interpretation |
|---|---|---|---|
| one-line | complement | simple | Replace each body-value token v by n+1−v; keep positions, punctuation, size token and boundaries fixed. This relabeling depends on n. |
| canonical cycle | inverse | simple | Each cycle `(m,a2,...,ak)` becomes `(m,ak,...,a2)`. The minimum m stays first, cycle order remains sorted by minima, lengths and delimiter positions stay fixed. |
| canonical cycle | complement | scrambling | Left composition with value reversal generally changes cycle membership and requires cycle decomposition and canonicalization. It is not conjugation by a simple relabeling. |
| one-line | inverse | scrambling | Positions and values exchange roles, requiring content-dependent redistribution. |
| either | identity | identity/reference | No change; it is not an extra observation in the simple-versus-scrambling contrast. |

Canonical-cycle inverse is a structured within-cycle reordering, not a pointwise token relabeling. Its exact construction passed 46,233 checks. The descriptor “simple” concerns serialization, not a proved ranking of Transformer/GRU circuit complexity or learning difficulty. Do not alter the annotation after viewing CKA.

Every task pair has one simple encoding and one scrambling encoding. Test the conjecture with the **paired within-task-pair difference**:

```text
d_pair = mean_architecture,unit[
    advantage(simple encoding) - advantage(scrambling encoding)
]
```

Then average the eight pair-level differences and resample whole pairs. This prevents the two inverse pairs versus six complement pairs from becoming an unpaired transformation-family confound. Report learning associations separately; proficiency and geometry associations do not isolate a causal effect of serialization simplicity.

## Similarity and residualization definitions

The requested advantage is a **new statistic**:

```text
transform_advantage = S(designated transform)
                    - max S(eligible alternative transforms)
```

The original Bt subtracts one original wrong transform and does not maximize over identity. The original Delta is a training increment. Keep both meanings separate in field names. Use `transform_advantage_training_increment` for the trained-minus-initial value of the new advantage; do not overwrite or rename the old scientific Bt/Delta tables.

Version (a) versus (b) changes both length handling and the CKA estimator. A decline cannot be attributed only to length. Retain the already-defined within-length ordinary-CKA bridge as a prespecified comparison. Ordinary CKA and debiased CKA can also disagree in sign around zero.

For version (c), within each n and candidate/transform, form one shared nuisance design

```text
Z = [intercept, onehot(A(p)), onehot(B(T(p)))]
X_resid = X - Z @ lstsq(Z, X)
Y_resid = Y - Z @ lstsq(Z, Y)
```

Both sides must use the same design. Identical answers or bijective shifts can generate duplicate spans. Use a prespecified rank-revealing float64 solver, not a direct inverse, and record its tolerance, columns, numerical rank, residual degrees of freedom, residual energy, and undefined cases. Equal counts across lengths already hold for the original geometry test anchors, so the equal-count rule should not trigger arbitrary extra sample selection.

This removes arbitrary additive functions of the individual answer categories. It does not remove all joint answer interactions, confidence, prediction errors, or every nonlinear answer association. The original joint-stratum control can remove interactions, but uses different variables and may remove rows. The nuisance projection can differ across candidate transforms; report those differences rather than claim identical residual supports or a pure causal effect of mathematical correspondence.

U-centering implements the unbiased-HSIC component estimator for fixed kernels. Its normalized CKA ratio is not itself an unbiased estimator in general. After fitting nuisance regressions on the same samples, it is especially inappropriate to call the result an unbiased conditional-independence test. Negative values should remain negative; near-zero/degenerate self-HSIC terms must not be replaced by zero scores or by a more favorable kernel.

Use saved genuine step-0 weights wherever available. A matching seed alone does not establish numerical identity of an older reconstructed start. Mirror the exact new statistic and nuisance designs on initial states, keeping initial geometry separate from trained preference.

## Dependence-preserving summaries and interpretation

- For the primary 32-condition grid, average seed/unit values within each fixed task-pair/architecture/encoding cell, report every seed value and range, then resample **eight entire unordered task-pair blocks**. Keep their architectures, encodings, reciprocal displays and seed/unit values together.
- Report the four new units separately from all five including the original exploratory unit. These are overlapping views, not independent groups. Data, initialization and training streams vary jointly within a unit.
- The eight pairs were selected rather than sampled independently from all mathematical tasks; they also share input worlds, initial states and two transformation families. A percentile task-pair bootstrap is an **exploratory conditional sensitivity interval**. It does not establish nominal coverage for a population of arbitrary mathematical relationships.
- A leave-one-pair analysis can identify domination by one pair. Leave-one-architecture and leave-one-encoding analyses leave one declared configuration/encoding; they are robustness descriptions, not independent architecture populations. Optional transformation-family leaveout is descriptive: only two families are available.
- If “most task pairs” is used as a decision criterion, define it before results (for example, at least five of eight pair-level effects) and distinguish it from “most of 32 dependent cells.” Do not count reciprocal directions as sixteen independent pairs.
- No scientifically justified negligible-effect threshold is supplied by the request. Do not infer “effect disappeared” merely from a bootstrap interval crossing zero. Show attenuation, interval and actual magnitudes, and refrain from equivalence claims without a declared meaningful bound.
- The peaks/valleys special case should report all three transforms' ranks under each version, not just whether complement wins. Its label difference-at-most-one relation and source proficiency remain relevant even when exact equality fails under identity.
- Residual superiority over initialization would support selective geometry beyond the specified additive answer/length model. It would not on its own identify a neural encoding mechanism or warrant automatically starting probes or new training.

The subsequent result interpretation should retain unsupported, negative and undefined observations. The mathematical catalog, alternative-transform policy and encoding annotations are fixed independently of the new CKA values.
