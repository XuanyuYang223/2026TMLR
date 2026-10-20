PermWorld: existing-model multilayer diagnostics and an encoding × architecture engineering development unit
Start date: 2026-10-09; execution continued into 2026-10-10 (America/Los_Angeles).

Completion status: existing-model multilayer analysis and the 16-model development pilot are complete; all development models
use fixed 20k updates. Engineering, independent activation/CKA verification, completeness, and stopping audits passed; this round's
execution processes have ended, and the formal confirmation manifest was not started.

Reading entry points
- completion_reader_report.html / completion_reader_report.txt: overall report for this round,
  including complete decompositions, task mastery, and measurement definitions.
- reader_report.html: existing-model multilayer results and 30 figures.
- encoding_development/report.html: 16-model development results and 40 figures.
- final_audit.json, final_stop_audit.json: completeness, F17 protection, and stopping evidence.
- development_results_review.txt, encoding_development/reader_review.json:
  Independent arithmetic/baseline recomputation, missing-value checks, and scientific-interpretation review.

The most direct new finding in this round: encoding changes initialization pairing preferences, thereby substantially changing Delta.
For answer-controlled GRU inverse CKA, for example, one-line/cycle B0 is -0.6169/+0.8010,
Bt is +0.0675/+0.0793, and Delta is +0.6844/-0.7217; the Delta sign change does not correspond to
a reversal of trained preference. The eight cycle-encoded models exceed the training-defined length mode by only
0.27--1.78 percentage points. Task mastery is unmatched across encodings, so the results cannot be interpreted as
encoding mechanisms at equal learning levels. Limitations from one development world and different architectural parameter counts remain.

This directory is independent of the existing F17 paper, experiments, and release files. Large new caches, development data,
and weights are stored on drive F; paths are listed in storage.json and the development root below.

Scope and stopping boundary
1. Reuse 82 existing checkpoints and fixed inputs, adding all layers and one fixed pooling choice.
2. Complete the engineering development unit: four properties × two encodings × two architectures = 16 models, with fixed 20k updates.
3. Stop after reporting. Development results are excluded from formal confirmation; do not automatically run the old formal manifest, add
   seeds, or launch GroupWorld, finite-field/matrix, null-space, GELU-composition, or larger-model experiments.

Existing-model analysis
- run.py, input_geometry.py, summarize.py, and all input/source dependencies are constrained by freeze.json.
  Compatible historical extraction versions and necessary numerical repairs are retained separately in compatible_extraction_
  freezes.json, engineering_repairs.json, and *_before_* snapshots.
- Inputs, orbits, lengths 10..30, per-length 1/21 weights, wrong transforms, and three primary metrics
  (linear CKA, k10 neighborhood overlap, held-out Procrustes) retain the historical protocol. Debiased
  CKA is a fixed sensitivity analysis. Original final-layer, query, output-probability, and ground-truth-answer values are directly reused.
- All historical initial weights are grade C: reconstructed with the original construction procedure; original step-0 identity
  is unverified. New checks do not upgrade them to exact matches.
- Original block residuals and final_norm are separate; the input end before the task is the primary position.
  The contextual mean of numeric tokens is the only pooling sensitivity. At fixed length,
  embedding_end and embedding_value_mean are constant, and results are explicitly undefined.
- position_numeric_onehot is a static isometric input reference; no initialization or Delta is constructed.
  position_numeric_embedding uses each checkpoint's genuine token embeddings.
- 62 same-seed model pairs yield 187488 per-length cells, including static/undefined baselines; these cannot
  be treated as independent replicates. Training-seed variability includes initialization and optimization randomness; it is not a new-world CI.
  review_notes.txt corrects the narrower “initialization differences” wording in the frozen summary template.

Read-only recomputation of existing layer metrics (no training):
  OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 .venv/bin/python diagnostics/permworld_layers_20261009/run.py validate
  OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 .venv/bin/python diagnostics/permworld_layers_20261009/run.py evaluate
  OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 .venv/bin/python diagnostics/permworld_layers_20261009/summarize.py
Extraction/completed pairs are skipped after checking existing caches. Complete deliverables include layer_metric_results.csv,
layer/length/relation summaries, leave-one sensitivity tables, figures, report.html, and brief_report.txt.

Encoding × architecture development (one world, one source unit)
Code: experiments/permworld_encoding_archdev_20261009/
Large artifacts: /mnt/yangx-projects-f/20261009/permworld_new_artifacts/encoding_archdev/
Small artifacts and report: encoding_development/ in this directory (exported by frozen encoding_summary.py).
Properties: descents, recoils, peaks, valleys. Encodings: one_line, canonical_cycle.
Transformer=width 256, 4 layers, 8 heads; GRU=width 256, 4 layers. Parameter counts differ; this is not a capacity-matched experiment.
Genuine step-0 is shared within each architecture; identical parameters are not claimed across architectures. GRU parameters retain a value-by-value
equal key migration; explicit dropout repairs backend RNG restoration. Audits and original weights are retained.

Each step is an explicit command; the following is not an automatically executed formal training plan:
  .venv/bin/python -m experiments.permworld_encoding_archdev_20261009 --stage cpu-check
  .venv/bin/python -m experiments.permworld_encoding_archdev_20261009 --stage overfit
  .venv/bin/python -m experiments.permworld_encoding_archdev_20261009 --stage resume
  .venv/bin/python -m experiments.permworld_encoding_archdev_20261009 --stage train
  .venv/bin/python -m experiments.permworld_encoding_archdev_20261009 --stage evaluate
  .venv/bin/python -m experiments.permworld_encoding_archdev_20261009 --stage report
  .venv/bin/python diagnostics/permworld_layers_20261009/encoding_summary.py
The train command accesses only the independent development manifest and requires prerequisite engineering gates to pass; it does not invoke the formal manifest.
Development uses only the fixed 20k endpoint, does not pass models based on geometric-effect direction, and does not extend weak models' budgets.
This execution also used parallel_tail_driver.py for up to 8 independent models at the end of the fixed manifest,
reducing waiting; the serial main driver verifies and skips completed models. Each model still receives one 20k budget only.
Scheduling uses only manifest position and completion status, never effects to decide conditions. Tail and middle receive
disjoint tasks, with JSON/log records of execution timing; the same worker cannot be started twice.
parallel_orchestration_manifest.json and v1 script snapshots record when a second worker was added;
the frozen training function, hyperparameters, budget, and 16-model manifest were not changed.
parallel_reserved_driver.py also ran fixed manifest index 8 (GRU one-line descents) early,
while the main driver was still at index 3 (zero-based indices). Middle later skipped the same completed model, avoiding a serial tail wait;
this added an execution process, not a model or scientific condition. Reserved audits, historical orchestration snapshots, and manifest
revision timing are retained; every completed model still has only one fixed 20k update budget.
After middle stopped at its fixed-position gate, parallel_recoils_driver.py continued original manifest index 9
(GRU one-line recoils); before starting, it verified that middle had ended and this model had never started, using
the vacated execution slot. Only execution order in the original manifest changed; complete audits and orchestration snapshots are retained.

Shared readouts
B0=initial(correct-wrong), Bt=trained(correct-wrong), dc=correct CKA change,
dw=wrong CKA change, Delta=Bt-B0=dc-dw. Encoding interactions are canonical_cycle
minus one_line, with Gamma0/Gammat/Gamma_correct/Gamma_wrong/GammaDelta reported separately.
Positive Delta alone cannot be interpreted as correct-representation convergence; hidden Delta minus output Delta cannot decompose mechanisms.
Source-task accuracy is reported separately; engineering gates do not replace evidence of task-content learning.

Audits
final_audit.py checks completed artifacts only, without training. final_audit.json contains unique-cell
coverage, decomposition arithmetic, itemwise consistency of inherited results, frozen versions, and F17 protected-file hashes.
review_notes.txt and engineering-repair logs retain problems and repair order; development is not presented as confirmation.

See development_code_review.txt for the read-only development-code review: “all32” in the protocol is inherited
wording; the actual fixed manifest contains 16 models. output_prob is renormalized only over legal numeric answers 0..30,
not the full-vocabulary distribution. The measured state precedes the task token and is causally constrained;
the actual full forward pass still includes the task/equals suffix, with an independent prefix-only numerical check.
Answer stratification uses separate fit/test ground-truth group means; it is a measurement control, not a deployable prediction step.

Read-only summary of completed artifacts (no training or changes to original evaluation data):
  .venv/bin/python diagnostics/permworld_layers_20261009/overview.py
  .venv/bin/python diagnostics/permworld_layers_20261009/annotate_completion_report.py
overview.py and encoding_summary.py were frozen before complete results appeared, constrained respectively by
overview_freeze.json and encoding_exporter_freeze.json. Reader reports add measurement notes only;
the original report and tables remain byte-for-byte identical, as recorded in completion_annotation_audit.json.
Before summarization, two display-format edits in overview.py were found not to match the frozen hash; only
those edits were reverted, exact frozen bytes restored, and reports regenerated. The original freeze record was not redirected; erroneously
generated reports and code are saved in before_freeze_restore/, with the complete sequence in issue.json.
Training data, genuine step-0, complete checkpoints, and large activation caches are retained on drive F; the local diagnostic directory
retains code, protocols, small artifacts, tables, and figures without copying large files onto the SSD.
