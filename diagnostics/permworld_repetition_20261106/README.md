# PermWorld independent-unit repetition

**Completed and stopped: all five units now have complete main analyses.** All 256 new fixed-20k specialists and their analyses are finished, alongside the 64 original specialists. Engineering models are excluded from these totals. Read the [final report](FINAL_REPORT.md) first, then the [separate new-four/all-five summaries](cohort_summary/report.md). The [execution status](execution_status.json) distinguishes completed endpoints from completed analyses. The commands below support audit, recovery and reproduction; no additional experiment is scheduled.

[Final completion receipt](completion.json), [checkpoint/init/protection audit](final_execution_audit.json), [independent table audit](final_saved_results_audit.json), and [standalone paper-source build validation](final_paper_validation.json) all pass. Recorded new-source training process time totals **72.55 hours**, with **0.94 hours** for the four new main analyses; these are not active GPU kernel hours and exclude separately recorded preparation, extraction, engineering, audit and paper work. See [actual cost records](final_cost_summary.json).

## Evidence and entry points

- [Original-unit audit](ORIGINAL_AUDIT.md): actual sharing, seeds, checkpoint identities, independent arithmetic reproduction, and the limited independent-specialist-initialization proposal.
- [Frozen protocol](protocol.json), [source freeze](freeze.json), and [all 256 prescribed jobs and seeds](training_manifest.csv).
- [Engineering verification](engineering_audit.json) and [measured preflight costs/initial identities](preflight_cost_and_identity.json).
- [Completed-unit results](summary/results.md), [per-unit counts](summary/unit_counts.csv), [completion/pending entries](summary/unit_status.csv), [count descriptives](summary/unit_count_summary.csv), [continuous effects](summary/condition_effects_by_unit.csv), and [condition-level descriptives](summary/condition_effects_summary.csv). With only one complete unit, across-unit sample SD is undefined. Pending units are not zeros or negative findings.
- [Separate repetition-cohort report](cohort_summary/report.md): the four new units are summarized first, followed by all five including the historical exploratory unit. These are overlapping views, not independent comparison groups. [New-four unit counts](cohort_summary/new_four/unit_counts.csv) and [all-five unit counts](cohort_summary/all_five/unit_counts.csv) retain the individual observations; empty or incomplete cohorts do not receive invented zero estimates.
- [Unitwise effects figure](summary/condition_effects_by_unit.png). It displays only completed analyses; it is not evidence that pending units have replicated.
- [Runner](../../experiments/permworld_repetition_20261106/runner.py), [cache-only analysis](../../experiments/permworld_repetition_20261106/analysis.py), and [analysis tests](../../tests/test_permworld_repetition_analysis.py).
- [English manuscript methods](../../manuscripts/representation_selectivity_20261012/sections/independent_repetition_methods.tex) and [generated results-status section](../../manuscripts/representation_selectivity_20261012/sections/independent_repetition_status.tex). The latter refreshes after a whole unit completes, without claiming completion in advance.

The original unit was independently recovered as follows. All denominators remain the original 32 dependent conditions **within a unit**.

| Original control | A: positive Bt | A: positive Delta | B: above other-candidate mean | B: uniquely first |
|---|---:|---:|---:|---:|
| Raw, within length | 27/32 | 25/32 | 30/32 | 25/32 |
| Joint-answer strata | 26/32 | 18/32 | 29/32 | 19/32 |

No new practical-effect threshold or rank tolerance is introduced. Read exact signs together with unrounded effects and their ranges. Shared tasks, initials, inputs, transformations, lengths, and metrics are dependent observations. Five units cannot be pooled into independent binomial trials over 160 conditions.

## Registered design

The original model sizes, 16 exact tasks, two encodings, two configurations, data scale, optimizer, learning-rate schedule, microbatching, 20k endpoint, and final-state main analyses are retained. All four new seed rows were fixed before their sampling or fitting. The original exploratory unit is kept separately identifiable.

Within each unit, all 64 specialists share underlying data, splits, and saved batch indices. Within an architecture, all tasks and encodings copy its **actual numerical step-0**, whose state-dictionary hash is checked before optimization and recorded in each job. Transformer and GRU do not share parameters. The inherited training-description string mentions 32 models; that is not the manifest size: there are 32 specialists per architecture and 64 per unit, all using the same schedule file.

The four original tasks retain separate training streams, paired across encodings. The other twelve retain the original sharing of their architecture's descents stream. Across units, data, batches, initial parameters, and training-stream bases are changed. Evaluation has **no separate sampling seed**: train/validation/test are sequential draws from the same data-seeded generator, followed by deterministic evaluation-mode extraction. The exact fixed historical exclusion index is reused, and accidental cross-unit overlap is allowed by the original mechanism. No new domain or task distribution is sampled.

Comparison A keeps initial/trained correct, wrong, and identity similarities and their B0/Bt/dc/dw/Delta decomposition. Comparison B fixes the left specialist and transform and evaluates all 15 right candidates. Shared raw initial states give its raw zero reference. Candidate-specific answer residualization can change grouping/support, so its controlled initial contrast is measured rather than assumed zero. A unique first rank requires no strictly higher candidate and no exact tie.

The repeated scope is the original **final-normalized main analysis**: hidden states and conditional output probabilities, ordinary/debiased CKA, kNN, and held-out Procrustes for A; ordinary CKA for B; raw and joint-answer controls. Separate all-layer and input-pooling historical sensitivity studies are not relabeled as repeated. Every length is calculated separately and weighted 1/21; undefined components propagate without reweighting.

## Validation actually run

1. Original 64 endpoint/completion bindings: 320 checks passed; both genuine step-0 tensor hashes independently recomputed. Saved-table counts above recovered.
2. Five analysis tests pass, covering all four metrics, both comparisons, exact ties, missing-length behavior, cache-analysis recovery and missing units.
3. Sixteen engineering cases (four tasks × two encodings × two architectures) used 40 real updates. Continuous versus on-disk step-20 recovery had **zero maximum parameter and optimizer discrepancy** and identical final RNG in every case. Independent CKA formulas differed by at most `1.11e-16`.
4. Real engineering states were saved, reloaded, extracted and analyzed using the unchanged main metric code on a bounded n=10 adapter (128 fit/test anchors, four tasks, three right candidates). This checks execution, not task mastery, the formal 32-condition counts, or the sign of the hypothesis. No accuracy or geometry sign was an engineering pass criterion.
5. A separate [saved-original activation check](original_metric_verification/summary.json) covers four fixed relation/configuration/length cases, including the offset-one decomposition formula: 64 A rows and 120 B candidate rows, **2,096 checks with zero failures**, maximum discrepancy `9.992e-16` against the unchanged `1e-8` tolerance. Sixteen consistently undefined values are retained. This is a finite-scope analysis implementation check using saved activations, not a rerun of original training or all metrics in every cell. The [script](validate_saved_original.py) and [check rows](original_metric_verification/checks.csv) are retained.
6. The [original-artifact protection check](protected_originals_check.json) verifies 151 specifically named original files, excluding the deliberately updated manuscript and entry point. It also verifies both new real step-0 hashes and all 32 physical engineering checkpoints. It is a selected-file check at the recorded time, not proof of all-repository or continuous immutability.

Engineering took about 135 seconds, excluding approximately 20 seconds for first-world generation. Median 40-update process times were about 2.39 seconds for Transformer and 2.75 seconds for GRU, including validation; short-run extrapolation is not a full-budget timing guarantee. The historical full-unit anchor is 18.496 process hours, or about 73.98 hours for four additional units, **not active GPU time**. Current full-run observations can refine costs without changing the manifest. Large artifacts use the separate F drive; available storage was about 3.3 TB and the current device was an RTX 5070 with 12 GB.

## Run, inspect, and recover

Run these commands from the repository root. The environment uses `.venv/bin/python` and the existing local source artifacts. The registered config path is fixed; `--config` does not silently select a different scientific protocol.

```bash
# Read-only current endpoint/analysis status; refreshes status snapshots.
.venv/bin/python -m experiments.permworld_repetition_20261106.runner status

# Tests: no model fitting.
.venv/bin/python -m pytest -q tests/test_permworld_repetition_analysis.py

# Resume the registered whole pipeline ONLY if its worker is no longer active.
.venv/bin/python -u -m experiments.permworld_repetition_20261106.runner run

# Explicit single-job recovery; do not run concurrently with the whole pipeline.
.venv/bin/python -u -m experiments.permworld_repetition_20261106.runner job \
  --run-id unit02_transformer_one_line_descents

# Cache-only re-analysis of one completed unit, with matching frozen inputs.
.venv/bin/python -m experiments.permworld_repetition_20261106.runner analyze --unit unit02

# Rebuild summaries and manuscript status using completed units only.
.venv/bin/python -m experiments.permworld_repetition_20261106.runner summary

# Separate new-four and all-five views of the completed saved analyses.
.venv/bin/python tools/summarize_permworld_repetition_cohorts.py
```

`freeze`, `prepare`, and `engineering` are explicit stages and have already been executed for the first new unit. Re-running them validates/reuses matching artifacts. Formal training checkpoints every 1,000 updates with model, AdamW state, all RNG states, hashes, progress and validation curves. A process interrupted after a checkpoint resumes that exact state; updates after the last saved checkpoint are repeated from that state. Completed jobs are skipped only after identity/hash checks. The single pipeline takes an exclusive lock; do not start a standalone job/extraction against the active worker.

Large artifacts and live logs:

```text
/mnt/yangx-projects-f/20261106/permworld_repetition/
  protocol.json, launch_record.json, active_process.json, pipeline.log
  unit02/ ... unit05/
    data.npz, data_audit.json, batch_indices.npy, batch_lengths.npy
    unit_config.json, events.jsonl
    initializations/{architecture}/step0.pt, step0.json, start_rng_*.pt
    models/{run_id}/checkpoint.pt, progress.json, completed.json
    features/*.npz, *.json
    analysis/{per-length tables, aggregate tables, shards, verification, completion}
```

The pipeline is detached from the chat terminal. Inspect its PID in `launch_record.json`, and read the log directly:

```bash
tail -n 20 /mnt/yangx-projects-f/20261106/permworld_repetition/pipeline.log
```

A technical exception records its traceback and stops execution for diagnosis. Resumption uses the same seeds/configuration; no substitute seed is generated. Nonfinite training is separately classified. Weak validation accuracy is retained and does not cause replacement, early stopping, or a larger budget. Incomplete cache/data artifacts are preserved as technical orphans and regenerated with the same inputs. A failed or incomplete unit is explicitly distinct from a completed unit with unfavorable results.

The original fixed-budget protocol has no optimization-convergence threshold. Consequently a saved 20k endpoint is **not labeled converged merely because it completed**. Validation curves, final loss/gradient records, per-length source accuracy and the training-derived length-mode baseline remain available to diagnose insufficient fitting or weak generalization. No post hoc accuracy threshold is invented to turn weak endpoints into technical failures, and these outcomes do not exclude a unit from its prescribed analysis.

## Paper completion

A separate [document-only finalizer](../../tools/finalize_permworld_repetition.py) completed after the formal worker ended. It does not train or recompute geometry. The final build verifies all four new 32-condition analyses, the five-complete-unit summary, and the successful manifest-exit receipt. The pending/failed guard remains part of the recovery implementation.

The completed five-unit build contains data-dependent English count summaries (mean/sample SD/range), direction agreement with finite-unit counts, all 96 primary continuous-effect summaries, and the all-unit figure. The PDF and self-contained 47-file source archive are built from these actual records. Earlier interim builds had 45 files and reported their then-current completed count. Original tracked PDF/ZIP bytes are backed up externally; historical provenance and public release objects are not overwritten.

```bash
# Document-only rebuild from the current recorded status, including an interim report.
.venv/bin/python tools/finalize_permworld_repetition.py --build-current
```

See [paper build receipt](paper_finalization.json), [document waiter launch](paper_finalizer_launch.json), and `paper_finalizer.log` under the external root. The current interim archive was extracted and compiled independently. Synthetic final-only rendering was tested outside the real paper without creating fake completion records.

## Provenance and boundaries

Earlier English-only edits changed some bytes referenced by legacy training freezes. The new execution therefore binds the current dependencies in its own 60-file SHA manifest, preserving the old freezes rather than falsely claiming byte identity with them. Unchanged original optimizer/update/model/math/metric functions are reused. The initially attempted source enumeration encountered a non-file runtime module label and was corrected **before** registration, new data, or training; no scientific run was restarted or selected on outcomes.

The registration records its parent Git HEAD and then-uncommitted changes; that HEAD alone does not identify the newly added implementation. An [exact 62-member execution snapshot](execution_source_archive.json) on the external drive contains all 60 hash-verified source files plus the protocol and freeze. This preserves the actual registered bytes without requiring a Git upload. [All four worlds and eight genuine new starts](all_units_provisioning_audit.json) have also been provisioned and audited; their distinct hashes establish identities, not a mathematical proof of random independence or a claim that all queued models have already run.

The original PermWorld/GraphWorld/Matrix reports, tables, checkpoints and release objects remain historical sources. The separate F17 project is not modified. This task does not launch the proposed independent-specialist-initialization supplement, extra seeds, new worlds beyond the four registered units, larger models, GroupWorld, or other mechanism studies. No automatic Git upload or release publication is part of the pipeline. Final completion is the four new full-unit analyses and their unit-level summary, not merely successful software execution or 256 prescribed filenames.
