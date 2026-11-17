#!/usr/bin/env bash
set -euo pipefail
cd /home/yangx/ICML
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export PYTHONPATH=/home/yangx/ICML/experiments/permworld_peaks_probe_20261109/vendor:/mnt/yangx-projects-f/20261109/permworld_peaks_probe/dependencies

# Existing completed units are hash/configuration checked and skipped.
# These commands only fit source position probes, read saved activations/maps,
# compose predicted masks and score; they do not train counting specialists.
.venv/bin/python -m pytest -q tests/test_permworld_lemma_oracles.py tests/test_permworld_lemma_calibration.py
.venv/bin/python -m experiments.permworld_lemma_composition_20261110.run freeze
.venv/bin/python -m experiments.permworld_lemma_composition_20261110.run all
.venv/bin/python diagnostics/permworld_lemma_composition_20261110/summarize_composition.py
.venv/bin/python diagnostics/permworld_lemma_composition_20261110/finalize_composition_audit.py
.venv/bin/python diagnostics/permworld_lemma_composition_20261110/build_reader_delivery.py

# Per-anchor CSVs for both primary routes; all other routes/variants remain in NPZ.
.venv/bin/python diagnostics/permworld_lemma_composition_20261110/export_saved_predictions.py --all-primary
.venv/bin/python diagnostics/permworld_lemma_composition_20261110/finalize_delivery.py
