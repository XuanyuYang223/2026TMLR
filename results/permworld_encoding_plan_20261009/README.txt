 from  /home/yangx/ICML execution; PYTHONDONTWRITEBYTECODE=1 prevents modification of legacy-module caches.

Completed preparation checks (no formal training):
  PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.permworld_encoding.validate --output results/permworld_encoding_plan_20261009/encoding_validation.json
  PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.permworld_encoding.checks --config results/permworld_encoding_plan_20261009/planning_candidate.json --output results/permworld_encoding_plan_20261009/integration_checks.json

Future formal execution (commands below were not executed in this round; microbenchmarks are not formal training):
  PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.permworld_encoding.data --config results/permworld_encoding_plan_20261009/protocol.json
  PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.permworld_encoding.train --config results/permworld_encoding_plan_20261009/protocol.json --run-id u001_one_line_descents
  Following training_manifest.csv in run_id, execute every model serially in recorded alphabetical order, without performance-based reordering/deletion.
  PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.permworld_encoding.evaluate --config results/permworld_encoding_plan_20261009/protocol.json --unit u001
  After all prespecified models complete, explicitly evaluate each unit once, then:
  PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.permworld_encoding.evaluate --config results/permworld_encoding_plan_20261009/protocol.json --summarize

Formal data/step0/training checkpoints are written only to results/permworld_encoding_confirmation_v1, not F17 or old diagnostic directories. Resume interruptions using the identical command; mismatched protocol/data/code hashes reject resumption. Never overwrite the formal protocol; changes require an explicit new version and refreeze.
Primary results: unit_components.csv, encoding_interactions.csv, primary_statistics.csv; accompanied by baseline_unit_components.csv, baseline_encoding_interactions.csv, task_accuracy_summary.csv and all per-length tables.
missing/undefined units are not silently dropped or automatically replaced/expanded. Representation-metric components are not causal contribution percentages; hidden Delta minus probability Delta does not define a mechanism.
