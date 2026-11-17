# Orthogonal-alignment source and design audit

Read-only source review of the supplied pilot and prior probe. No map or probe was fitted, no checkpoint was constructed, and no new result was inspected.

The target-to-source Procrustes orientation and equal-length objective are mathematically correct. For row-vector features, `C = sum_n Yc_n.T @ Xc_n / m_n`; if `C = U S Vh`, the code uses `R = U @ Vh`. This is an O(256) map, allowing reflections. It is one map across all lengths within a unit/encoding/phase/pairing. The omitted common factor1/21 does not affect R.

The source ridge probe and StandardScaler are fitted/tuned independently at each length using only the original source fit/dev split. Alignment uses raw centered final_norm features. Main maps all score the same target complement test input with the same complement mean, source mean, peak labels and source probe. Native-transform rows remain supplementary.

The eight shuffled controls preserve length and true source peakcount, cyclically derange each nonsingleton stratum, and retain singletons. Fixed seeds are paired across encodings/phases; shuffled maps do not provide eight independent study replications. Paired-map fitting itself uses no position labels; count strata and the oracle-count position prior do use true counts.

## Control interpretation

Freeze the exact best-control definition before results. A map-control maximum should include means-only, identity-map, inverse-map and each of the eight shuffled-map values, rather than only the shuffle average. An expanded negative-control maximum can additionally include the oracle-count prior, step-0 aligned complement, and one-line descents/recoils direct and aligned results. Within-model peaks/valleys probes are positive readout references and should not silently enter a negative-control maximum. Step-0 uses separately refitted probes; it is not a fixed-readout intervention.

## Required wrapper checks

Check original caches, metadata, data SHA and checkpoint receipts, plus full row ordering. The adapted NPZ format is standardized only in schema: do not z-score validation features globally. Preserve raw FP32 features and add actions1/2. Both encodings need initial/trained target additions; supplied descents/recoils controls are one-line only. Require all21 finite length means and preserve per-example undefined counts. Report each unit, mean/sampleSD/range and exact control maxima without treating lengths or shuffles as independent units.

## Limits

Restored linear readout is evidence about held-out decodability after a fitted coordinate map. It does not prove causal use of positions or that an unfitted shared basis emerged. Other task controls are not matched for difficulty/learning. SVD nullspaces can be nonunique, so historical CPU-forward versus saved GPU features need not yield bit-identical results. Built-in orthogonality assertions retain NumPy default relative tolerance; claim only the checks actually performed.

Full source hashes and itemized checks are in `math_design_audit.json`.
