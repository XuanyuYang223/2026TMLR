# Frozen position-readout composition

This package composes peaks/valleys and fixed-point position masks into three
new questions using cached one-line GRU activations. It does not train the
specialists or refit alignment maps. Formal combiners receive masks and length
only; source labels calibrate readouts, downstream truths only score answers.

The fixed protocol, complete results, source hashes, original-unit reference,
four-unit comparisons, all controls and reproduction commands are in
[`diagnostics/permworld_lemma_composition_20261110`](../../diagnostics/permworld_lemma_composition_20261110/README.md).

`oracles.py` is an independent pure-Python permutation oracle; `combiners.py`
contains the scalar and batched mask-only algorithms; `calibration.py` selects
source-only thresholds; `run.py` freezes inputs and runs the fixed inventory;
`metrics.py` scores predictions with explicit missing-answer handling;
`export_predictions.py` converts a saved route to per-anchor CSV without fitting.
