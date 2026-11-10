# Independent-unit repetition results

Completed analyses: 5/5 listed units.

A unit jointly resamples the original data mechanism, shared initialization within an architecture, and training randomness. Its 32 conditions, shared tasks, candidate pairs, lengths, and metrics are dependent measurements.

Comparison A reports trained pairing selectivity Bt separately from B0, correct/wrong changes, and Delta. Comparison B fixes the input transform and changes the right specialist; its raw shared-initialization reference is zero, whereas candidate-specific answer strata need not give a zero reference.

The table below describes unit-level counts. The denominator remains 32 conditions within each unit; counts are not pooled into a binomial trial. Sample standard deviation is undefined with fewer than two completed units.

| Comparison | Control | Metric | Count | Finite units | Mean | Sample SD | Min–max |
|---|---|---|---|---:|---:|---:|---:|
| A | raw_within_length | linear_cka | Bt_positive | 5 | 28.4 | 0.894427 | 27–29 |
| A | raw_within_length | linear_cka | Delta_positive | 5 | 23.8 | 1.30384 | 22–25 |
| A | raw_within_length | debiased_cka | Bt_positive | 5 | 28.4 | 0.894427 | 27–29 |
| A | raw_within_length | debiased_cka | Delta_positive | 5 | 23.8 | 1.30384 | 22–25 |
| A | raw_within_length | knn_overlap | Bt_positive | 5 | 25.4 | 0.894427 | 24–26 |
| A | raw_within_length | knn_overlap | Delta_positive | 5 | 18.6 | 0.547723 | 18–19 |
| A | raw_within_length | procrustes | Bt_positive | 5 | 28.2 | 1.09545 | 27–29 |
| A | raw_within_length | procrustes | Delta_positive | 5 | 25 | 1.87083 | 22–27 |
| B | raw_within_length | linear_cka | above_other_mean | 5 | 29.8 | 1.09545 | 28–31 |
| B | raw_within_length | linear_cka | unique_first | 5 | 24.6 | 1.14018 | 23–26 |
| A | answer_strata | linear_cka | Bt_positive | 5 | 27.4 | 1.34164 | 26–29 |
| A | answer_strata | linear_cka | Delta_positive | 5 | 18 | 0 | 18–18 |
| A | answer_strata | debiased_cka | Bt_positive | 5 | 27.4 | 1.34164 | 26–29 |
| A | answer_strata | debiased_cka | Delta_positive | 5 | 18 | 0 | 18–18 |
| A | answer_strata | knn_overlap | Bt_positive | 5 | 25.6 | 1.14018 | 24–27 |
| A | answer_strata | knn_overlap | Delta_positive | 5 | 16.6 | 0.547723 | 16–17 |
| A | answer_strata | procrustes | Bt_positive | 5 | 26.2 | 1.09545 | 25–28 |
| A | answer_strata | procrustes | Delta_positive | 5 | 18 | 0 | 18–18 |
| B | answer_strata | linear_cka | above_other_mean | 5 | 29.6 | 0.547723 | 29–30 |
| B | answer_strata | linear_cka | unique_first | 5 | 18.4 | 1.14018 | 17–20 |

Continuous effects and all per-unit values are in the companion CSV files. Exact signs use the original strict zero boundary; no new near-zero threshold is introduced. Interpret any sign reversal alongside the unrounded magnitude, range, and within-unit decomposition. Shared-initialization repetitions do not test specialists initialized independently within a unit.
