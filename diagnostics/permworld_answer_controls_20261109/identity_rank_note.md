# Bounded initial-identity numerical audit

The old-result reproduction gate passed before this audit. This note checks three fixed single-length cases from saved genuine step-0 activations; it performs no forward pass, optimization, or source-code change. It does not replace pooled or 21-length aggregate results.

For source **peaks**, all 15 right-task initial identity activation arrays are bit-identical in every audited case (maximum absolute difference 0). Mathematically, each defined self-CKA is 1, including pair-specific shared-design dual-answer residual CKA. Consequently the analytical rank is a 15-way tie. Candidate-specific regression designs can change residual energy, but cannot identify a preferred task from equal source/target states.

| Unit / configuration / n | Readout | Numeric score range | valleys exact rank / ties | Max absolute deviation from 1 |
|---|---|---|---|---|
| unit02 / gru / canonical_cycle / 10 | single_length_biased_linear | 0.99999999999999967 to 0.99999999999999967 | 1 / 14 | 3.33e-16 |
| unit02 / gru / canonical_cycle / 10 | single_length_debiased | 0.99999999999999944 to 0.99999999999999967 | 13 / 2 | 5.55e-16 |
| unit02 / gru / canonical_cycle / 10 | single_length_dual_answer_debiased | 1 to 1 | 1 / 14 | 0 |
| archdev001 / transformer / one_line / 20 | single_length_biased_linear | 1 to 1 | 1 / 14 | 0 |
| archdev001 / transformer / one_line / 20 | single_length_debiased | 0.99999999999999956 to 0.99999999999999956 | 1 / 14 | 4.44e-16 |
| archdev001 / transformer / one_line / 20 | single_length_dual_answer_debiased | 1 to 1 | 1 / 14 | 0 |
| unit05 / gru / canonical_cycle / 30 | single_length_biased_linear | 1.0000000000000002 to 1.0000000000000002 | 1 / 14 | 2.22e-16 |
| unit05 / gru / canonical_cycle / 30 | single_length_debiased | 0.99999999999999967 to 0.99999999999999967 | 1 / 14 | 3.33e-16 |
| unit05 / gru / canonical_cycle / 30 | single_length_dual_answer_debiased | 1 to 1 | 1 / 14 | 0 |

The table retains exact numerical ties/ranks from the frozen CPU metric functions; no tolerance was added. Any ulp-scale ordering is numerical sensitivity and must not be interpreted as initial task specificity. An independent NumPy Gram/feature calculation is included in the JSON for every candidate. CPU and later GPU arithmetic may differ at roundoff scale, so these three cases do not certify exhaustive equality of every stored numeric rank.

Each case uses the original 512 test anchors at its stated length. JSON records all candidate scores, answer-design ranks, residual energy fractions, anchor-row IDs and hashes, activation hashes, checkpoint hashes, verified file SHA-256 values, and independent NumPy differences. No missing score was replaced by zero.

The main trained-task ranking comparison and strongest-alternative transform advantage retain their frozen definitions. This identity caveat concerns the interpretation of strict numerical initial-identity ranks only.
