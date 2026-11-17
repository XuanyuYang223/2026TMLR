#!/usr/bin/env bash
# Existing checkpoints only; fits scalar probes/maps, never base networks.
set -euo pipefail
cd /home/yangx/ICML
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export MKL_NUM_THREADS=2
export PYTHONPATH=/home/yangx/ICML/experiments/permworld_peaks_probe_20261109/vendor:/mnt/yangx-projects-f/20261109/permworld_peaks_probe/dependencies
.venv/bin/python -m pytest -q tests/test_permworld_exceedance_oracles.py tests/test_permworld_exceedance_token_diagnostic.py
# Validates the existing freeze and skips complete, hash-bound units.
# It refuses to run until the separate prefix-final experiment is complete.
.venv/bin/python -u -m experiments.permworld_exceedance_alignment_20261110.token_diagnostic all
