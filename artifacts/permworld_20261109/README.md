# Completed PermWorld result tables

These archives publish saved scalar result tables from the completed five-unit repetition and existing-checkpoint answer-control diagnostic. They contain no model weights or activation arrays and do not perform new analysis or training. Existing releases and their manifests are unchanged.

| Archive | Contents |
|---|---|
| [answer_control_tables.zip](answer_control_tables.zip) | All 840 length-specific score tables, 40 aggregate blocks, per-unit learning/prior/summary CSVs, and an internal member-SHA manifest. |
| [independent_repetition_tables.zip](independent_repetition_tables.zip) | All four newer units' saved main-analysis CSV/JSON tables, with an internal member-SHA manifest. The original unit's tables are in the repository diagnostic directory. |

[manifest.json](manifest.json) records each archive's SHA-256, byte size, and member count. Members retain paths relative to their original external experiment root. The internal manifests preserve source paths as provenance; extraction does not require those paths. Extract into a new directory, rather than overwriting original experiment inputs.

- [Answer-control reader report](../../diagnostics/permworld_answer_controls_20261109/READER_REPORT.md), [commands](../../diagnostics/permworld_answer_controls_20261109/README.md), and [checkpoint/seed inventory](../../diagnostics/permworld_answer_controls_20261109/checkpoint_inventory.csv).
- [Completed repetition report](../../diagnostics/permworld_repetition_20261106/FINAL_REPORT.md) and [original-unit analysis tables](../../diagnostics/permworld_repetition_20261106/historical_analysis/).
- [English working paper](../../manuscripts/representation_selectivity_20261012/manuscript.pdf) includes the completed repetition update. The newer answer-control diagnostic is reported separately and has not been silently incorporated as a new manuscript claim.

The historical `no_upload` fields describe the state when the corresponding analysis stopped. Upload was authorized subsequently. Frozen protocols, scientific score tables, checkpoint identities, and existing protection receipts are retained byte-for-byte.
