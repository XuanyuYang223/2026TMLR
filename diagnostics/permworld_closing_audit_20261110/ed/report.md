# E/D encoding audit: separate geometric and readout comparisons

Completed saved-table arithmetic audit. Four new units (unit02–unit05) define each summary; archdev001 remains a separate historical row. No fresh CKA, activation forward, probe, mapping or base model was fitted. This is a post-results audit, not a confirmation experiment.

## Three CKA versions and their initialization references

The trained transform advantage is inverse CKA minus max(identity CKA, complement CKA). This is not the paper’s old one-wrong-transform Bt, and its trained-minus-initial difference is not the old Delta. The maximum is taken after each transform’s registered length aggregation.

|Architecture|Encoding|Version|Initial advantage, mean ± sample SD|Trained advantage, mean ± sample SD|Trained range|
|---|---|---|---|---|---|
|gru|one_line|pooled_raw|-0.14786331 ± 0.02721130|0.73107224 ± 0.02181914|[0.70686589, 0.75950131]|
|gru|one_line|within_length_debiased|-0.94484837 ± 0.00677722|0.59186474 ± 0.00230226|[0.58974587, 0.59459515]|
|gru|one_line|within_length_dual_answer_debiased|-0.94300399 ± 0.00707345|0.01951662 ± 0.01204955|[0.01058010, 0.03718882]|
|gru|canonical_cycle|pooled_raw|-0.06346995 ± 0.00986181|0.69951130 ± 0.01759346|[0.67483163, 0.71523653]|
|gru|canonical_cycle|within_length_debiased|-0.13516415 ± 0.00823293|0.48312157 ± 0.01395934|[0.46685351, 0.49823545]|
|gru|canonical_cycle|within_length_dual_answer_debiased|-0.15557328 ± 0.00934206|0.09678745 ± 0.05537906|[0.03958765, 0.17130045]|
|transformer|one_line|pooled_raw|-0.02308214 ± 0.00512792|0.62999500 ± 0.13779913|[0.42847155, 0.72683670]|
|transformer|one_line|within_length_debiased|-0.95640433 ± 0.00950832|0.71842420 ± 0.03667919|[0.66836727, 0.75641038]|
|transformer|one_line|within_length_dual_answer_debiased|-0.95523151 ± 0.00990095|0.10890765 ± 0.01758175|[0.08448610, 0.12429754]|
|transformer|canonical_cycle|pooled_raw|-0.01180504 ± 0.00133182|0.09655091 ± 0.04747997|[0.04144280, 0.14754710]|
|transformer|canonical_cycle|within_length_debiased|-0.18353613 ± 0.09067928|0.18596061 ± 0.01372317|[0.17047425, 0.20110428]|
|transformer|canonical_cycle|within_length_dual_answer_debiased|-0.19451732 ± 0.09391523|0.04960034 ± 0.05186168|[0.00593731, 0.11183709]|

Pooled_raw is ordinary CKA mixing the original lengths. Within_length_debiased computes U-centered CKA at each length; within_length_dual_answer_debiased adds the specified intercept and additive one-hot regressors for both true answers, with shared same-evaluation-sample residualization. Per-length versions use all original 512 anchors per n, and each n=10–30 has fixed1/21 weight. Undefined required components propagate rather than being reweighted. Moving from pooled to within-length changes both length handling and estimator.

## Paired encoding contrasts

|Architecture|Version|Trained cycle − one-line, mean ± sample SD|Range|Positive units /4|
|---|---|---|---|---|
|gru|pooled_raw|-0.03156094 ± 0.02346804|[-0.05767535, -0.00688544]|0|
|gru|within_length_debiased|-0.10874317 ± 0.01588998|[-0.12607830, -0.09151042]|0|
|gru|within_length_dual_answer_debiased|0.07727083 ± 0.05391891|[0.02900756, 0.15453127]|4|
|transformer|pooled_raw|-0.53344410 ± 0.15587111|[-0.65186323, -0.30623130]|0|
|transformer|within_length_debiased|-0.53246360 ± 0.04396829|[-0.56335892, -0.46726299]|0|
|transformer|within_length_dual_answer_debiased|-0.05930731 ± 0.06551332|[-0.11836023, 0.02735098]|1|

## GRU readout migration is a different measurement

|Encoding|Correct mapping AUROC, mean ± sample SD|Margin over best common mapping control, mean ± sample SD|Margin range|
|---|---|---|---|
|one_line|0.78294850 ± 0.00431998|0.01008156 ± 0.00187764|[0.00738567, 0.01140395]|
|canonical_cycle|0.83015591 ± 0.00193651|0.00014951 ± 0.00097097|[-0.00058529, 0.00151823]|

These are the existing final-state, full-eligible-support probe results. The common comparison takes max over means-only, identity, the wrong complement mapping and each of eight shuffled mappings, after equal-length averaging. Direct transfer and model-internal probes remain separate. Matched support is a labeled sensitivity; its results and undefined/finite denominators are retained in the CSVs.

After the specified additive answer control, GRU CKA favors cycle over one-line, whereas the E/D position-probe margin favors one-line. Transformer CKA has the opposite encoding ordering to GRU. CKA selectivity and an aligned probe’s readout transfer are different questions with different nuisance controls; their numerical values must not be subtracted or treated as one effect. The original cross-task-pair encoding double-dissociation prediction failed and remains failed. This audit does not rehabilitate it.

The residual removes the prescribed additive answer-mean terms; it does not erase all answer information. Answer-dependent covariance, joint-answer interactions, confidence and prediction errors can remain. Same-evaluation residual fitting also couples row geometry; U-centered HSIC does not turn this into an unbiased conditional-independence test. Shared initialization means initial identity states are identical before task-specific training; candidate-specific residual designs can still affect other initial-transform scores.

Both encoding and model-family rankings here are observational across the tested tasks, trained configurations, datasets and saved initialization scheme. They do not alone identify an encoding-caused mathematical mechanism. Four stochastic units quantify within-setting variation; lengths, positions and derived comparisons are not additional independent units.

## Files and scope

* cka_per_unit.csv and cka_by_phase.csv: all five raw unit results, historical flagged.
* cka_four_unit_summary.csv: each architecture/encoding/version, initial/trained/increment; raw four-unit values, mean, sample SD, minimum and maximum.
* encoding_contrasts_by_unit.csv and encoding_contrasts_summary.csv: paired cycle−one-line contrasts including initialization and the training increment.
* cka_per_length_components.csv: all5,040 existing E/D length components, sample counts and residual-rank/DF fields preserved. No new per-length CKA was calculated.
* probe_per_unit.csv, probe_four_unit_summary.csv and probe_per_length_source.csv: the saved E/D probe comparison with support and undefined/finite counts.
* reference_checks.csv: reference numbers compared at their displayed rounding precision; references were not computational targets.
* spec.json and completion.json: named source SHA bindings, current code hash, global protocol binding and validation scope.

No paper or old result was overwritten. The primary deliverable is bounded evidence for writing, not a new mechanism claim or a new experiment.
