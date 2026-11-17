#!/usr/bin/env bash
set -euo pipefail
cd /home/yangx/ICML
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export PYTHONPATH=/home/yangx/ICML/experiments/permworld_peaks_probe_20261109/vendor:/mnt/yangx-projects-f/20261109/permworld_peaks_probe/dependencies

.venv/bin/python -m pytest -q tests/test_permworld_local_causal_pairs.py tests/test_permworld_local_causal_recurrent.py
.venv/bin/python diagnostics/permworld_local_causal_20261110/source_audit.py
.venv/bin/python diagnostics/permworld_local_causal_20261110/prepare_pairs.py
.venv/bin/python -m experiments.permworld_local_causal_20261110.run engineering-freeze
.venv/bin/python -m experiments.permworld_local_causal_20261110.run calibrate
.venv/bin/python -m experiments.permworld_local_causal_20261110.run freeze
.venv/bin/python diagnostics/permworld_local_causal_20261110/summarize_results.py --freeze
.venv/bin/python -m experiments.permworld_local_causal_20261110.run all
.venv/bin/python diagnostics/permworld_local_causal_20261110/summarize_results.py
.venv/bin/python diagnostics/permworld_local_causal_20261110/exact_delta_check.py
.venv/bin/python diagnostics/permworld_local_causal_20261110/finalize_audit.py
.venv/bin/python diagnostics/permworld_local_causal_20261110/build_reader.py
.venv/bin/python diagnostics/permworld_local_causal_20261110/finalize_delivery.py

# Completed, hash/configuration-consistent units are skipped. The counting GRUs
# are always frozen; this script fits readouts and runs prescribed patches only.
# No cross-model stage, novel combination task or publication is invoked.
