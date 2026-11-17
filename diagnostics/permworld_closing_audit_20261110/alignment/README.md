# Peaks → valleys: trained-source specificity audit

## Observation

With the **same saved trained peaks probe and scaler** applied to all three targets, aligned trained valleys achieved full-test AUROC **0.633139313 ± 0.017599993** (sample SD across four new units). Actual saved step-0 targets gave **0.502418782 ± 0.002164594** and trained fixed_points gave **0.502715680 ± 0.002018380**. Valleys exceeded both model controls in every formal unit, with paired differences of about 0.10–0.14 AUROC. These are substantive differences in the tested setting, rather than tiny sign fluctuations.

The old score labeled `phase=initial` used an **initial-source probe** (`alignment.py: inputs_by_length`). It did not answer the present question. This audit instead keeps the source trained peaks probe/scaler unchanged and fits only the two missing control maps. The old initial-source result remains unchanged in its original files.

## Full-test primary result

|Unit|Trained valleys|Step-0 target|Trained fixed_points|Valleys − step0|Valleys − fixed_points|Valleys − best map control|
|---|---|---|---|---|---|---|
|unit02|0.644224449|0.502007849|0.500754244|+0.142216600|+0.143470206|+0.124881221|
|unit03|0.607052553|0.505578950|0.501368209|+0.101473604|+0.105684345|+0.085814635|
|unit04|0.638055973|0.500797466|0.505106418|+0.137258508|+0.132949556|+0.115319952|
|unit05|0.643224276|0.501290862|0.503633850|+0.141933414|+0.139590426|+0.129234965|

The last column retains the original common mapping comparison: means-only, identity-fit, inverse-fit and each of eight count-preserving shuffled-fit maps are separately evaluated on the same trained valleys complement test input. Each control is first aggregated across lengths, then the maximum is taken. Their individual results and selected maximum identity are in `common_mapping_controls_by_unit.csv` and `contrasts_by_unit.csv`.

## Original count-matched sensitivity support

|Unit|Trained valleys|Step-0 target|Trained fixed_points|Valleys − step0|Valleys − fixed_points|Valleys − best map control|
|---|---|---|---|---|---|---|
|unit02|0.651792017|0.503461822|0.505704079|+0.148330195|+0.146087938|+0.125409109|
|unit03|0.610031581|0.498523165|0.505217990|+0.111508416|+0.104813590|+0.097302860|
|unit04|0.642335720|0.509176063|0.504882390|+0.133159657|+0.137453329|+0.125680080|
|unit05|0.645075125|0.489310586|0.492445525|+0.155764540|+0.152629600|+0.132739309|

The earlier headline near 0.637 corresponds to this count-matched support (mean 0.637308611); it is not inconsistent with the full-test primary mean near 0.633. Maps and source readouts are identical across supports; only the original test mask differs.

## Four-unit summaries

|Support|Route or contrast|Mean|Sample SD|Range|
|---|---|---|---|---|
|full_test|actual_saved_step0|0.502418782|0.002164594|[0.500797466, 0.505578950]|
|full_test|trained_fixed_points|0.502715680|0.002018380|[0.500754244, 0.505106418]|
|full_test|trained_valleys|0.633139313|0.017599993|[0.607052553, 0.644224449]|
|count_matched|actual_saved_step0|0.500117909|0.008417707|[0.489310586, 0.509176063]|
|count_matched|trained_fixed_points|0.502062496|0.006420182|[0.492445525, 0.505704079]|
|count_matched|trained_valleys|0.637308611|0.018613579|[0.610031581, 0.651792017]|
|full_test|valleys_minus_best_common_mapping_control|0.113812693|0.019549270|[0.085814635, 0.129234965]|
|full_test|valleys_minus_fixed_points|0.130423633|0.017055359|[0.105684345, 0.143470206]|
|full_test|valleys_minus_step0|0.130720531|0.019630047|[0.101473604, 0.142216600]|
|count_matched|valleys_minus_best_common_mapping_control|0.120282840|0.015691314|[0.097302860, 0.132739309]|
|count_matched|valleys_minus_fixed_points|0.135246114|0.021219025|[0.104813590, 0.152629600]|
|count_matched|valleys_minus_step0|0.137190702|0.019535480|[0.111508416, 0.155764540]|

These four unit values, not positions, anchors, lengths or methods, are the repeated units. No new significance or equivalence inference is attached. Raw values retain their original precision in CSV files.

## Historical engineering reference, excluded from summaries

|Support|Trained valleys|Step-0 target|Trained fixed_points|
|---|---|---|---|
|full_test|0.638165851|0.501916369|0.502314229|
|count_matched|0.643920445|0.502082904|0.503266617|

## Source, fit budget and support

All routes pair trained peaks at π with the relevant target at complement(π). Targets have their own per-length fit-only centering and one global target-to-trained-source O(256) map, obtained by the original equally weighted per-length cross-products and SVD. The trained valleys map, means, probe/scaler and per-example scores were already saved and were reused after exact reproduction. Only the missing step0-target and fixed_points-target maps were fitted (two per unit). No probes/scalers were fitted, no hyperparameters were selected, and no neural forward or model training was run.

The step-0 target cache is tied to the actual saved shared architecture checkpoint, not reconstructed from a seed. Its original cache task name can be peaks/descents because the pre-task prefix has no task token and those initial specialist states are identical under the saved shared weights; it does **not** imply use of an initial-source probe. Source probe/scaler bindings identify the trained peaks checkpoint for every route.

|Unit|Support|Fit anchors across21lengths|Test support|Finite AUROC|Undefined AUROC|
|---|---|---|---|---|---|
|archdev001|full_test|4295|10752|10752|0|
|archdev001|count_matched|4295|1757|1757|0|
|unit02|full_test|4272|10752|10752|0|
|unit02|count_matched|4272|1707|1707|0|
|unit03|full_test|4320|10752|10751|1|
|unit03|count_matched|4320|1675|1675|0|
|unit04|full_test|4274|10752|10752|0|
|unit04|count_matched|4274|1753|1753|0|
|unit05|full_test|4283|10752|10752|0|
|unit05|count_matched|4283|1755|1755|0|

The original orbit-disjoint split, test exclusions, source fit/tuning indices and all 21 lengths n=10–30 are unchanged. AUROC is computed within each example over the n−2 interior positions, then finite example means within length, then fixed 1/21 length weights. A length with no finite scores remains undefined rather than having its weight redistributed. The single undefined full-test example in unit03 has no positive/negative comparison and is retained in the sample table; its coverage is reported for every route.

## Allowed and unsupported interpretation

**Allowed:** in these one-line GRU specialists and four existing repetitions, mathematically paired trained valleys supports additional transferable position readout relative to both a true step-0 target and a different trained task, under the same frozen trained source probe, target-specific fit-only centering and orthogonal map budget. Correct pairing also remains superior to the original common mapping controls.

**Unsupported:** these readouts alone do not prove common computation, a causal use of the transferred feature, an encoding-caused effect, or universality beyond the reported models/tasks. The comparison changes the target task/learned representation; it is not a mechanism isolation experiment. No output or causal conclusion is inferred from AUROC alone.

This is a post-result exploratory audit. The protocol was fixed before these added control scores, not a retrospectively registered confirmation experiment.

## Validation and reproduction

All five trained valleys routes reproduced saved per-position scores exactly (maximum error 0; original tolerance 1e−12), so no original score was overwritten. The same trained-source archive hash is recorded for all three targets. The end audit passed **2960** checks, including 180 independent pairwise-AUROC calculations, all new map orthogonality checks and fresh endpoint SHA verification of **90** explicitly used input files. Endpoint protection covers these named files, not whole disks or continuous monitoring. See `final_audit.json`, `source_plan.json`, `freeze.json`, `synthetic_verification.json` and `engineering_log.json`.

Run from repository root with the existing environment:

```bash
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export PYTHONPATH=/home/yangx/ICML/experiments/permworld_peaks_probe_20261109/vendor:/mnt/yangx-projects-f/20261109/permworld_peaks_probe/dependencies
.venv/bin/python experiments/permworld_closing_audit_20261110/alignment.py run
.venv/bin/python diagnostics/permworld_closing_audit_20261110/alignment/finalize.py
```

Completed units are SHA-checked and reused. The existing `source_plan.json` and stage freeze are immutable; `prepare` is an initial setup command and must not rewrite a frozen source plan during reproduction. New raw scores and maps are stored at `/mnt/yangx-projects-f/20261110/permworld_closing_audit/alignment/<unit>/`; per-length/sample tables and receipts are under this diagnostic directory. Missing inputs should be reported, not regenerated by training or probe fitting.
