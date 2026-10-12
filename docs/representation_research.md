# Research question, comparisons and evidence

The study asks how exact mathematical correspondences accompany **representation pairing selectivity**, and how the measurement depends on configuration, encoding, layer, metric, initialization reference and answer controls. It preserves the logic of holding a comparison's declared factors fixed. Differences between a Transformer and the existing GRU configuration are not a pure architecture intervention: capacity and learning proficiency are not matched.

## Source studies

PermWorld contains 16 specialists organized into eight directed relations, with one-line and normalized cycle encodings in each of two original model configurations. Its primary analysis uses final normalization at the input end before the task query, ordinary linear CKA within each of the 21 lengths 10–30, then weight 1/21 each. The 64 true 20k endpoints comprise 16 reused endpoints and 48 previously missing specialists completed in the fixed run. They are not 64 independently sampled worlds. All eight relations and four configuration/encoding combinations remain reported. Six negative primary `Bt` cells occur in cycle encoding, but this descriptive concentration does not establish encoding as their sole cause. Historical initialization and layer investigations have separate reconstructed-reference limitations.

GraphWorld uses four specialists and four directed reversal/complement relations on fixed graph anchors n=6/7/8. Each n has weight 1/3. Raw, joint-answer, and joint-answer-plus-exact-edge-count controls stay separate. The wrong transformation can accidentally satisfy the designated answer formula; all such rows remain. An answer-control result is not silently replaced by the stronger density sensitivity.

MatrixGroupWorld uses two independently sampled matrix datasets and two true initializations, each copied across four single-task specialists. With n=6/8/10, `F1=wt(Ae1)`, `F2=wt(A^T e1)`, `F3=wt(A(e1+e2))`, and `F4=wt(A^T(e1+e2))`. All arithmetic except the ordinary integer weight is over F2. Let `Q=I+E21`, `P` exchange the first two coordinates. The fixed correspondences `g(T(A))=f(A)` are:

| Relation | f → g | Correct T(A) | Original wrong U(A) | Wrong answer |
|---|---|---|---|---|
| R1, primary | F1 → F2 | A^T | P A^T | second-column weight |
| R2, supplementary | F3 → F4 | A^T | Q^T A^T | first-column weight |
| R3, primary | F1 → F3 | A Q | A Q^T | second-column weight |
| R4, supplementary | F2 → F4 | Q^T A | Q A | second-row weight |

The full embedded GL(2,F2) orbit `{LAR, LA^T R}` has at most 72 members and separates train, validation and analysis anchors. Full-rank rejection uses exact finite-field rank. Deduplicating canonical orbits changes the sampling distribution; it is not uniform orbit sampling. These formulas also hold for general binary square matrices, and row/column weights are related to graph degrees. Only transpose and fixed multiplication actions are covered; four relation instances do not give four independent relation types or demonstrate a group-specific computation.

## Two different contrasts

The original contrast fixes the task pair and changes input pairing:

\[
B_0=S_c^0-S_u^0,\quad B_t=S_c^t-S_u^t,\quad d_c=S_c^t-S_c^0,\quad d_u=S_u^t-S_u^0,
\]
\[
\Delta=B_t-B_0=d_c-d_u.
\]

Initialization-reference changes affect `Delta`, not an already fixed trained `Bt`. Positive `Delta` can be driven by a decrease in the wrong score; it does not require an increase in the correct score. The original individual CKA values and identity pairing are retained. Shared seed is not evidence of identical initial weights. The real saved initials of current models and constructor-rebuilt/unverified historical references are explicitly separated.

The later contrast fixes the correct input transformation and left model, then varies the right task among all other existing specialists. It reports `fixed_transform_score_initial/trained`, predefined categories, ranks, means and medians. Perm480, Graph12 and Matrix24 candidates share tasks/models/anchors/transformations; none is an independent replication count. Non-designated means only that this formula was not designated, not that the tasks have no mathematical relation. Designated above the candidate mean does not imply higher than every individual candidate.

## Controls and limitations

Joint-answer grouping, minimum group size, retained counts, residual degrees of freedom and missing components follow each original protocol. Different candidate tasks have different strata and support. Centering on gold labels does not remove confidence, prediction-error structure or all nonlinear answer information. Hidden `Delta` minus output `Delta` is not an estimate of an extra mechanism. Raw and controlled geometry have different interpretive scope; all negative and undefined cells remain.

The frozen kNN and held-out Procrustes measurements are sensitivities, not independent replications or functional tests. Model accuracy is independently reported over the full vocabulary with answer CE, teacher-conditioned EOS and train-defined per-size modes. Exceeding the mode is not full task mastery. Some cycle-encoding or XOR specialists remain weak; that limitation neither erases a geometry observation nor explains every encoding effect.

Matrix's raw ordinary-CKA `Bt` is positive in 8/8 cells, but answer-controlled `Bt` and `Delta` are positive in 5/8. R1's controlled ordinary-CKA `Bt` and `Delta` are negative in both units, despite its designated right task ranking first in the **different fixed-transform comparison**. R3 has positive controlled `Delta` in both units, with both correct and wrong CKA decreasing; its F3 n=10 learners remain weak. After answer control, the three hidden metrics have the same `Bt` direction in 5/8 cells and the same `Delta` direction in 6/8. This is not a universally consistent metric pattern.

The original Matrix numerical audit records **64 passed, 2 failed, 6 undefined_consistent**, and `scientific_analysis_complete=false`. The failures are trained raw output-probability kNN on the first-unit R1/n6 fixed 64-sample subset. The read-only distance diagnosis localizes roundoff at the nearest-neighbor boundary without changing scores, tolerances or the failed status. All actual tables and eight endpoints are present; a complete table set is distinct from a fully passed numerical check. The audit does not independently verify all 432 production cells.

No fresh training, model forward, probe, composition mechanism or functional benefit is introduced by this publication task. Saved-table replay, source SHA, archive reconstruction and GitHub byte retrieval are engineering checks. Current evidence supports exploratory mathematical-correspondence-associated pairing geometry, with conditions and alternatives still unresolved; it does not isolate a mathematics-specific mechanism or universal law.
