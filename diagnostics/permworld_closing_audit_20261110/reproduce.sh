#!/usr/bin/env bash
# Reuse the frozen inputs. Never invoke any upstream training/calibration stage.
set -euo pipefail
closing_repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$closing_repo_root"
closing_python="$closing_repo_root/.venv/bin/python"
closing_diag=diagnostics/permworld_closing_audit_20261110
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export PYTHONPATH="$closing_repo_root/experiments/permworld_peaks_probe_20261109/vendor:/mnt/yangx-projects-f/20261109/permworld_peaks_probe/dependencies${PYTHONPATH:+:$PYTHONPATH}"

"$closing_python" "$closing_diag/freeze_protocol.py"
"$closing_python" -m experiments.permworld_closing_audit_20261110.alignment freeze
"$closing_python" -m experiments.permworld_closing_audit_20261110.alignment run
"$closing_python" "$closing_diag/alignment/finalize.py"
bash "$closing_diag/ed/reproduce.sh"
"$closing_python" -m pytest -q tests/test_permworld_closing_gradient.py
"$closing_python" -m experiments.permworld_closing_audit_20261110.gradient_run freeze
"$closing_python" -m experiments.permworld_closing_audit_20261110.gradient_run all
"$closing_python" "$closing_diag/gradient/audit_saved_gradients.py"
"$closing_python" "$closing_diag/gradient/independent_check.py"
"$closing_python" "$closing_diag/gradient/build_report.py"
"$closing_python" "$closing_diag/build_writing.py"
"$closing_python" "$closing_diag/finalize_delivery.py"
