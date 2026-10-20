PermWorld CKA: exploratory existing-checkpoint diagnosis (2026-10-09)

Scope: eight single-property relationships, three existing seeds and their nine ordered pairings; Property32 five task-pool splits and50 existing models at k=1/2/4/8/16. No new training. This directory did not modify the F17 paper/results/release artifacts or original CKA analyses in the diagnosis round.

Read report.html first. Full data are in CSVs; figures are available as PNG and PDF.

Environment and data:
  /home/yangx/ICML @ 1073dcb3be4db90ec23476bab8f503c9e00f364b
  /home/yangx/neurips @ 74f0de2017115f06f11b7285366b66e7e8431d30
  external/neurips/src is auditable upstream code; existing checkpoints and Property32 caches are in /home/yangx/neurips.
  Existing specialist data/caches: results/specialist_cka_controls.
  Existing per-length ordinary/input-adjusted values: results/specialist_input_adjustment.
  SHAs for models, configurations, caches, data, and this directory's code are in the respective audit JSONs and deliverable_manifest.json.
  Complete reproduction requires original checkpoints/activation caches; missing artifacts must not be replaced by retraining.

Run sequentially from the workspace root (initialization construction and forwards only; no optimizer or training):
  PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python diagnostics/permworld_cka_20261009/inventory.py
  PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python diagnostics/permworld_cka_20261009/audit_history.py
  PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python diagnostics/permworld_cka_20261009/analyze_relations.py
  PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python diagnostics/permworld_cka_20261009/analyze_k.py
  PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python diagnostics/permworld_cka_20261009/verify.py
  PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python diagnostics/permworld_cka_20261009/report.py
  Stop after completion. No model training is needed.

Main deliverables:
  protocol.json: length weights, input pairings, and scope fixed before computing new metrics.
  model_inventory.csv / initialization_audit.json:98 logical models and94 unique checkpoints; initial identities, configurations, weight hashes, and missing artifacts.
  initialization_history.json / vocabulary_audit.json: historical analysis revisions, vocabulary, and RNG initialization paths; not substitutes for original-weight evidence.
  artifact_inventory.json / cache_validation.json: data/version provenance and146 cache-forward extraction checks.
  relation_length_cka.csv:9,072 rows of raw CKA and Delta for eight relationships x nine seed pairings x21 lengths x two controls x three metrics.
  relation_seed_summary.csv / relation_summary.csv: fixed-length-weight aggregates, means, medians, seed ranges, and counts of positive lengths.
  macro_summary.csv / transform_family_summary.csv: descriptive summaries over the fixed eight relationships, without increasing the number of independent relationships.
  leave_one_relation.csv / leave_one_length.csv: full relationship/length sensitivities.
  answer_output_length_baselines.csv / input_length_baselines.csv: answer, probability-output, and prescribed input-kernel baselines.
  permutation_length_calibration.csv / permutation_summary.csv / permutation_delta.npz: per-length version of the original32 conditional permutations, for descriptive calibration only.
  input_bootstrap_delta.npz / resampling_diagnostics.csv: shared orbit-anchor resampling distributions. Finite-sample bias is substantial; percentiles are not unbiased population confidence intervals.
  relation_dependencies.json: dependence through shared data/orbits/transformation families.
  k_length_cka.csv / k_model_self_initial_length.csv / k_summary.csv: per-length between-pool and self-initialization tables for the task-count series.
  k_pooled_control.csv / k_shared_initialization_audit.csv / k_length_input_audit.csv: pooled-length, shared-initialization, and duplicate-input audits.
  k_initial_reference_seed*.npy / k_probe_inputs.npz: reused/additional-forward initialization references and original fixed probe inputs.
  k_drift.png/pdf: the requested two parallel curves.
  verification.json / protected_before.json: independent numerical checks and protected-file SHA comparisons.
  completion.json: completed and stopped, with no new training.

Statistical interpretation:
  All results are exploratory, not confirmatory. The3 seeds or9 seed pairings do not constitute new independent relationships.
  None of the original initial-weight identities is directly verifiable; current results use references reconstructed with the original program.
  R0/R1/R2 are three existing source seeds; R3/R4 are fixed seed17 task-subset sensitivities sharing corpus and probes.
  Main analysis computes CKA separately for each length and averages with fixed weights; pooled CKA after removing length means is not a substitute.
  CKA is undefined for constant gold-answer residuals; preserve missing values instead of filling0.
  U-centered CKA may be negative; Delta is a difference of contrasts and can exceed1.
  Confidence controls reuse existing exploratory results only: no new regression models, kernels, or task filtering.
  Nonsignificance does not imply no learning; CKA alone cannot identify answer-independent information, relational encoding, or a mathematical computation mechanism.

Files in this directory can be regenerated; original experiments and F17 artifacts were protected in the diagnosis round. No automatic publication or GitHub release modification occurred. The repository-wide English edition is documented separately in docs/english_localization/README.txt; historical hashes refer to the original-language revision.
