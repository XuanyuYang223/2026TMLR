#!/usr/bin/env bash
# Existing-input timing follow-up only; do not rerun any parent fitting stage.
set -euo pipefail
timing_repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$timing_repo_root"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export PYTHONPATH="$timing_repo_root/experiments/permworld_peaks_probe_20261109/vendor:/mnt/yangx-projects-f/20261109/permworld_peaks_probe/dependencies${PYTHONPATH:+:$PYTHONPATH}"
timing_python="$timing_repo_root/.venv/bin/python"
timing_diag=diagnostics/permworld_local_timing_20261110

"$timing_python" -m pytest -q tests/test_permworld_local_timing_replay.py
# The starting audit is immutable once it is bound by the protocol.
if [[ ! -f "$timing_diag/input_audit.json" ]]; then
  "$timing_python" "$timing_diag/audit_inputs.py"
fi
"$timing_python" -m experiments.permworld_local_timing_20261110.run freeze
"$timing_python" "$timing_diag/summarize_results.py" --freeze
"$timing_python" "$timing_diag/build_headline_tables.py" --freeze
"$timing_python" -u -m experiments.permworld_local_timing_20261110.run all
"$timing_python" "$timing_diag/summarize_results.py"
"$timing_python" "$timing_diag/build_headline_tables.py"
"$timing_python" "$timing_diag/finalize_audit.py"
"$timing_python" "$timing_diag/build_reader_report.py"
"$timing_python" "$timing_diag/plot_headlines.py"
"$timing_python" "$timing_diag/verify_delivery.py"
"$timing_python" "$timing_diag/finalize_delivery.py"
