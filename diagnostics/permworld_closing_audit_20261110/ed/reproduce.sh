#!/usr/bin/env bash
set -euo pipefail
closing_ed_repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$closing_ed_repo"
closing_ed_python="$closing_ed_repo/.venv/bin/python"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
"$closing_ed_python" diagnostics/permworld_closing_audit_20261110/ed/analyze.py \
  --protocol-sha256 762dcc6e6e7f5ee1b2ea1358dbd97df56a74f2d615a59cf444a9fce3125ffde6
"$closing_ed_python" diagnostics/permworld_closing_audit_20261110/ed/independent_check.py
"$closing_ed_python" diagnostics/permworld_closing_audit_20261110/ed/plot_ed.py
