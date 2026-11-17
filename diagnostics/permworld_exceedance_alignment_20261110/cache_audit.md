# Exceedance/deficiency source-cache audit

Status: **passed**. All five units, both GRU encodings, all three specialists and both actual-initial/trained phases were audited without fitting or neural forward.

Found **120 logical validation/test cache records**, **30 trained checkpoints**, and **5 actual saved shared initializations**. Initial prefix equality across the three specialists was checked in 40 whole-array comparisons.

Validation arrays are `[5376,3,256]`; test arrays are `[10752,3,256]`, FP32, actions identity/complement/inverse. Each n=10–30 contains256validation and512test anchors in original order. Hash verification covers 322 named physical files (3,757,551,423 bytes).

| Unit | Raw records | Trained endpoints | Actual shared initialization |
|---|---:|---:|---|
| archdev001 | 24 | 6 | verified actual saved state |
| unit02 | 24 | 6 | verified actual saved state |
| unit03 | 24 | 6 | verified actual saved state |
| unit04 | 24 | 6 | verified actual saved state |
| unit05 | 24 | 6 | verified actual saved state |

## Mathematical and extraction binding

`E_i(x)=[x_i>i]` corresponds to `D_{x_i}(inv(x))=[inv(x)_{x_i}<x_i]`. Target own-coordinate scores must be indexed by `x_i-1` once. The audit checked registered counts for all source actions; it did not fit a probe or score this relation.

There is no token time axis in the selected saved schemas. Their `layers` field denotes per-layer prefix states. The exact model and extraction definitions, hashes, line positions and causal token interpretation are recorded in JSON. Necessary token diagnostics can hook `final_norm` on frozen eval/inference forwards; before a value token is at t-1 and after is at t. Cycle edge closure requires explicit delimiter-aware indexing.

Historical initialization is a saved value-preserving key migration of the actual pre-training state, not a same-seed reconstruction. Stale historical missing-authorized receipt statuses are retained as caveats after validating the actual hash-bound20k payloads.

Missing named files: 0. Binding/check failures: 0. All prior files were read only; the endpoint hash scope is the named inputs and sources, not every file on disk.
