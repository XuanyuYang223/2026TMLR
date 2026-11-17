# Exceedance/inverse-deficiency position alignment

Existing GRU specialist checkpoints are examined under both encodings, with
source-only position probes, validation-only orthogonal maps and fixed controls.
The final-state cross-pair comparison and the local-token diagnostic are kept
separate. No counting specialist is retrained.

[Protocol, full results and reproduction](../../diagnostics/permworld_exceedance_alignment_20261110/README.md)

`oracles.py` supplies independent labels, exactly-once target pullback and token
edge indices. `shared.py` binds input identities and orbit splits. `run.py` fits
the fixed final-state inventory. `rebuild_pv_scores.py` reconstructs earlier
position scores with saved maps and verifies them against the original tables.
The token-stage code records before/after states in a separate external cache.
