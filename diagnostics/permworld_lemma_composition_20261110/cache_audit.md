# Downstream-composition cache audit

All five units have the required one-line GRU payloads: **30 raw validation/test feature records**, **15 trained checkpoints**, and **55 saved trained one-line maps**. No missing file or binding mismatch was found.

Current SHA verification covers **127 physical files**, 1,605,985,284 bytes. Raw feature shapes are validation `[5376,3,256]` and test `[10752,3,256]`, FP32, in original anchor order. All 21 lengths have 256 validation and 512 test anchors.

Fixed-point counts from position labels match the registered task column `5` in 420 per-length/action checks. Existing peak/valley adapted identity/test arrays match raw source slices in 50 array/order checks.

| Unit | Peak raw C available | Fixed-point validation I/test C | Saved trained R keys |
|---|---|---|---|
| archdev001 | yes | yes, including endpoints | R_034 identity, R_035 complement, R_036 inverse, R_037–R_044 shuffles |
| unit02 | yes | yes, including endpoints | R_034 identity, R_035 complement, R_036 inverse, R_037–R_044 shuffles |
| unit03 | yes | yes, including endpoints | R_034 identity, R_035 complement, R_036 inverse, R_037–R_044 shuffles |
| unit04 | yes | yes, including endpoints | R_034 identity, R_035 complement, R_036 inverse, R_037–R_044 shuffles |
| unit05 | yes | yes, including endpoints | R_034 identity, R_035 complement, R_036 inverse, R_037–R_044 shuffles |

## Reuse and interpretation

The main input is `pi=C(x)`. The shared first activation is `peaks(C(x))`. The two-call same-model baseline uses `peaks(x)` for the valley mask; the two-model route maps `valleys(C(x))` to source peaks coordinates and uses the same source probe. All mapping controls use that same valleys input.

The fixed-point specialist fits on identity validation inputs `x` with its own fixed-point labels, and is evaluated on cached action 1 for `C(x)`. True fixed counts are not needed as input to prediction or composition.

Only rotations and shuffle permutations were saved previously. Per-length centering and ridge scalers/coefficients were not serialized. Rebuild probes with the frozen source-only fitter and reconstruct means from the original probe-fit rows; use complement target means for every map applied to the common complement target input.

The former “standardized” files are schema-adapted raw FP32 states. No new global normalization should be applied. Activation provenance and SHA were verified; no neural forward, probe, alignment fit, threshold selection or downstream scoring was run by this audit.
