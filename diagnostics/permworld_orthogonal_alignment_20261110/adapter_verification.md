# Validation-format adapter verification

All **30 adapted validation files** (six per unit, including the historical calibration) passed. Every one of the **50 exported action arrays**, comprising **68,812,800 FP32 values**, is bit-identical to its audited source `prefix_final[:, action, :]`. No extra normalization, fitting, subsampling, or reordering was introduced.

Every output preserves `validation_indices = arange(5376)` and the correct scalar checkpoint SHA. Output SHA, source/data/checkpoint bindings, action list, row count, and original-order flags match both `adaptation.json` and the unit completion receipt. All21 lengths retain256 validation rows.

All five completion receipts report feature reuse. All **50 original standardized files** retain their prior SHA bindings, sizes and modification times. No importer rebuild is recorded. This is endpoint/receipt evidence rather than continuous system-call monitoring; the separate global protection audit checks current original-file hashes.

This verification reads saved arrays and scalar metadata only. It performs no neural forward, model/probe/alignment fit, or old-file write. Exact paths and per-array content hashes are in `adapter_verification.json`.
