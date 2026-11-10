# Interpretation of the fixed answer-residual analysis

Let the two hidden-state matrices have rows corresponding to the same held-out anchors. At one length, let `Z` contain an intercept and the two observed answer one-hot designs, and let `Q` span its columns. The requested residuals are

\[
X_r=(I-QQ^\top)X,\qquad Y_r=(I-QQ^\top)Y.
\]

This removes any additive feature mean expressible as a lookup of either answer. A nonlinear function of a single categorical answer is still a one-hot lookup and is removed if it enters additively. What is not generally removed includes answer-dependent covariance, joint-answer interactions, confidence, prediction-error structure, and other input information correlated with the answers. The phrase “beyond the specified additive answer regressors” is therefore narrower and more accurate than “independent of the answers.”

The residual matrices are fitted on the same evaluation anchors, as requested. U-centering the Gram matrices does not restore the independent-row assumptions after this fitting. To see the issue, take independent feature-noise matrices with independent standard-normal entries, with `d_X` and `d_Y` columns, and a fixed residual-maker `M=I-QQ^T`. Conditional on `M`,

\[
\mathbb E[(MX)(MX)^\top]=d_XM,
\qquad
\mathbb E[(MY)(MY)^\top]=d_YM.
\]

Since U-centering is linear, the expected **unnormalized** inner product of these two U-centered Grams contains

\[
d_Xd_Y\lVert U(M)\rVert_F^2.
\]

This is a statement about the Gram inner product, not a formula for the expectation of the normalized CKA ratio. It illustrates why the residual score is not automatically an unbiased conditional-independence statistic. Candidate-specific design rank and leverage can also change this shared-projection geometry. All requested scores are retained; no metric is selected or changed after observing them.

At genuine shared initialization and under identity inputs, the two specialists' pre-task hidden states are identical. Applying the same residual projection preserves that equality for every candidate. Every defined identity CKA is consequently one in exact arithmetic. Floating-point differences of one unit in the last place can produce artificial ordering under a strict numerical rank rule. Such initial identity ranks do not identify a task partner, even if the unmodified CSV contains a strict rank other than a complete tie. Other initial transforms need not remain tied after candidate-specific answer residualization.

Finally, the bootstrap resamples the eight prescribed task-pair blocks. It retains dependence across encodings, architectures and stochastic units within each pair. It is an exploratory finite-catalog sensitivity analysis. The two transform families and any shared algebraic structure still limit generalization beyond this catalog. Removing a family, architecture, encoding, or pair is reported descriptively rather than treated as a fresh independent experiment.
