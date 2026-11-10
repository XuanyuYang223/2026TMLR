# Existing-checkpoint length and answer controls

This exploratory diagnostic uses the five completed PermWorld units: the original `archdev001` and the four subsequent independent units. It neither trains models nor fits probes. The original manuscript, results, input sets, and checkpoints are protected inputs.

Start with the [reader report](READER_REPORT.md), or open [the offline HTML version](reader_report.html) in a browser. The complete score/rank tables are linked there.

## Scope and provenance

`protocol.json` fixes the scientific analysis; `freeze.json` binds its SHA-256 and computational dependencies. Engineering revisions record a cache-axis adapter correction and an auxiliary probability-mass precision correction, both caught before any new scientific CKA score was computed. The scientific protocol was unchanged. Earlier freezes, failure logs, and the original FP32 probability-mass arrays are retained.

`source_inventory.json` records the 320 trained specialists, ten genuine saved initializations, data files, and activation records. Initializations are shared within each unit and architecture, as in the original experiment. No initialization is reconstructed from a seed. `protection_start.json` explicitly names the old files protected by the start/end audit; this audit is not a claim about every file in the repository or continuous monitoring.

All conditions retain the original held-out anchors, pre-task final-normalized state, and three input actions: identity, complement, and inverse. Each unit has 512 anchors at every length from 10 through 30: 10,752 anchors in total. No balancing subsample is necessary because the original per-length counts are already equal.

## Reproduction comes first

The original primary analysis was per-length ordinary CKA with the original joint-answer-strata control. It was not pooled CKA. The reproduction gate recomputes its raw and controlled scores, original designated/wrong/identity comparisons, candidate comparisons, aggregate counts, and ranks. The fixed absolute tolerance is `1e-8`; undefined values must match. A scientific mismatch stops the new analysis.

The three requested diagnostic versions are distinct:

1. `pooled_raw`: ordinary linear CKA over all original lengths together.
2. `within_length_debiased`: U-centered linear-kernel CKA separately at each length, followed by the fixed equal `1/21` average.
3. `within_length_dual_answer_debiased`: within each length, regress both feature matrices on the same intercept and one-hot encodings of both true answers, then compute U-centered CKA on the residuals and average with the same weights.

Identical answer columns are deduplicated. SVD also handles linearly dependent, nonidentical encodings, such as a deterministic answer offset. Rank, residual degrees of freedom, residual energy, and undefined components are retained. Undefined lengths are not replaced by zero or silently reweighted. Per-length ordinary CKA is saved as a bridge: versions 1 and 2 change both length handling and the estimator, so their difference cannot be attributed to length alone.

## Contrasts and limits

For a prescribed task pair and input action, the new transform advantage is the designated-action score minus the maximum score among the other eligible actions, including identity. This differs from the old `Bt`, which used one fixed wrong action, and from the old training `Delta`. Additional exact actions are excluded only according to the frozen mathematical catalog; none exists for the eight designated cross-task relations among the three tested actions.

The trained-minus-initial difference of the new advantage is also reported. At shared initialization, identity states are identical across specialists within an architecture, encoding, and unit. Their defined identity CKA is one, including after a shared residual projection. Consequently, an improvement over this initial advantage need not mean that the designated-action CKA increased. Scores for every action are retained to inspect this distinction.

Removing additive answer means does not remove every possible form of answer-related information. In particular, residual covariance, confidence, prediction errors, and interactions between answers can remain. U-HSIC is unbiased under its usual independent-observation assumptions; same-sample residual fitting introduces dependence between rows, so this descriptive residual U-CKA is not automatically an unbiased conditional-independence test.

The eight prescribed task pairs are the bootstrap blocks. Architectures, encodings, and units remain coupled within a resampled pair. Results for the four newer units are shown separately from results for all five. These intervals describe sensitivity within this selected task catalog; they do not turn the pairs into an independent random sample of mathematical relationships. No new practical-zero or equivalence threshold is introduced.

## Learning diagnostics

Training-only per-length answer frequencies define a fixed length prior with `alpha=0.5` smoothing over the 31 valid answer IDs. Reports retain exact accuracy, full 188-token model NLL, conditional numeric NLL, and Spearman correlation of the conditional numeric expectation with the answer. Full-vocabulary predictions and probability mass outside the valid answer support are also audited. Full NLL is not inferred from renormalized numeric probabilities.

The original 64 specialists have saved full cross-entropy. For the other 256, additional forward computation is FP32 evaluation of the original identity test inputs. It checks saved predictions, conditional probabilities, and unchanged parameter/checkpoint hashes. It performs no optimizer or backward operation. The first attempt's FP32 difference of two log-sum-exp values occasionally produced an impossible support probability slightly above one (maximum `1.0000019073486328`); validation stopped the run. The corrected evaluation sums full-softmax probabilities in FP64 and bounds only final FP64 reduction roundoff. The original arrays and failure record are preserved, and rerun full cross-entropy is required to be bit-identical to the already valid result.

## Reproduce

From `/home/yangx/ICML`:

```bash
.venv/bin/python -m pytest -q tests/test_permworld_answer_cka.py tests/test_permworld_answer_learning.py tests/test_permworld_answer_summary.py
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python -u -m experiments.permworld_answer_controls_20261109.run all
```

The frozen run stages can also be invoked separately: `reproduce`, `full-nll`, `learning`, `calculate`, and `summary`. `freeze` creates the protocol only when it does not already exist. Completed score blocks and forward passes are reused after their configured provenance checks; do not edit their files in place.

Large per-length score tables, answer-design bases, forward arrays, and logs live under `/mnt/yangx-projects-f/20261109/permworld_answer_controls`. The repository diagnostic directory contains provenance, aggregate score tables, summaries, figures, verification, and the reader report. No files are automatically uploaded.

After the analysis stages finish, the bounded independent checks and delivery audits can be repeated:

```bash
.venv/bin/python diagnostics/permworld_answer_controls_20261109/verify_saved_results.py
.venv/bin/python diagnostics/permworld_answer_controls_20261109/build_ordinary_bridge.py
.venv/bin/python diagnostics/permworld_answer_controls_20261109/review_learning.py
.venv/bin/python diagnostics/permworld_answer_controls_20261109/export_inventory.py
.venv/bin/python diagnostics/permworld_answer_controls_20261109/finalize_protection.py --workers 2
.venv/bin/python diagnostics/permworld_answer_controls_20261109/finalize_delivery.py
```

The endpoint protection audit rehashes approximately 32 GB of named source files. The finalizer hashes all 840 per-length score tables and 40 aggregate blocks, renders the offline report, and saves the delivery manifest and completion receipt. Neither audit launches a forward pass or training. The human interpretation and mathematical notes are also delivered; they should be reviewed alongside the automated summary.
