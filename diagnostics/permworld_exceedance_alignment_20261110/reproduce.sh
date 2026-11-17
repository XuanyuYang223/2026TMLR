#!/usr/bin/env bash
set -euo pipefail
cd /home/yangx/ICML
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export PYTHONPATH=/home/yangx/ICML/experiments/permworld_peaks_probe_20261109/vendor:/mnt/yangx-projects-f/20261109/permworld_peaks_probe/dependencies

# Math and extraction-index checks use independent permutation oracles.
.venv/bin/python -m pytest -q tests/test_permworld_exceedance_oracles.py tests/test_permworld_exceedance_token_diagnostic.py

# Resuming a completed unit verifies its protocol/artifact identities and skips
# fitting it again. Nothing below trains a counting specialist or edits old data.
.venv/bin/python -m experiments.permworld_exceedance_alignment_20261110.run freeze
.venv/bin/python -m experiments.permworld_exceedance_alignment_20261110.run all
.venv/bin/python diagnostics/permworld_exceedance_alignment_20261110/audit_saved_main.py
.venv/bin/python diagnostics/permworld_exceedance_alignment_20261110/summarize_results.py

# Separate local-token endpoint: actual frozen checkpoint forwards and scalar
# probes; AFTER-only source fit/scaler/means are unchanged at BEFORE evaluation.
.venv/bin/python -m experiments.permworld_exceedance_alignment_20261110.token_diagnostic freeze
.venv/bin/python -m experiments.permworld_exceedance_alignment_20261110.token_diagnostic preflight
.venv/bin/python -m experiments.permworld_exceedance_alignment_20261110.token_diagnostic all
.venv/bin/python -m experiments.permworld_exceedance_alignment_20261110.token_diagnostic summarize
.venv/bin/python diagnostics/permworld_exceedance_alignment_20261110/summarize_tokens_reader.py

.venv/bin/python diagnostics/permworld_exceedance_alignment_20261110/finalize_audit.py
.venv/bin/python diagnostics/permworld_exceedance_alignment_20261110/build_reader.py
.venv/bin/python diagnostics/permworld_exceedance_alignment_20261110/finalize_delivery.py
