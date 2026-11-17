# Explicit-edge token diagnostic: readout and control comparison

This secondary diagnostic is separate from the completed prefix-final, vector-valued alignment experiment. It uses one shared scalar position classifier per length, fitted on source AFTER states and source labels only. All base models remain frozen. The historical unit is an engineering reference; statistics below use four new units.

The primary score is per-anchor position AUROC, averaged within length and then across all 21 registered lengths with weight 1/21 each. Full eligible support is primary; matched support is secondary. Pooled-edge AUROC is supplementary. Every map scores the same paired inverse-target states; initial-target and fixed-point-target controls use their own states. Controls and priors are reported individually.

The same AFTER-fitted probe, scaler and centering are applied to BEFORE states. Before/after differences describe a distribution shift and local serialization stage; they do not identify the earliest point where the mathematical relation was logically inferable.

## Full-support per-unit comparison, all edges

Each common-control maximum includes means-only, identity, complement and each of eight shuffled-pair maps, after all-length aggregation. Priors, own readouts, direct transfer, initial-target and fixed-target controls remain separate.

|Unit|Encoding|Point|Source own|Correct map|Best common|Margin|Source n/i prior|Target n/j prior|Source kind prior|Target kind prior|Initial target|Fixed target|Target own|
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|archdev001|canonical_cycle|before|0.5181|0.5137|0.5905|-0.0768|0.8391|0.8385|0.6806|0.6813|0.5825|0.6244|0.4670|
|archdev001|canonical_cycle|after|0.9919|0.6965|0.6572|0.0393|0.8391|0.8385|0.6806|0.6813|0.6012|0.6536|0.9178|
|archdev001|one_line|before|0.7182|0.7012|0.6510|0.0501|0.8391|0.8385|0.5000|0.5000|0.7867|0.7717|0.7293|
|archdev001|one_line|after|0.9534|0.7886|0.7108|0.0778|0.8391|0.8385|0.5000|0.5000|0.7936|0.7884|0.9547|
|unit02|canonical_cycle|before|0.4168|0.5053|0.6235|-0.1182|0.8388|0.8389|0.6807|0.6811|0.6163|0.6478|0.4362|
|unit02|canonical_cycle|after|0.9820|0.7409|0.6512|0.0898|0.8388|0.8389|0.6807|0.6811|0.6294|0.6516|0.9099|
|unit02|one_line|before|0.7350|0.7094|0.5642|0.1452|0.8388|0.8389|0.5000|0.5000|0.7867|0.7948|0.7048|
|unit02|one_line|after|0.9468|0.7541|0.6360|0.1182|0.8388|0.8389|0.5000|0.5000|0.7947|0.8195|0.9476|
|unit03|canonical_cycle|before|0.5806|0.4867|0.5653|-0.0786|0.8389|0.8393|0.6806|0.6814|0.5962|0.5857|0.5021|
|unit03|canonical_cycle|after|0.9974|0.7772|0.7183|0.0589|0.8389|0.8393|0.6806|0.6814|0.6137|0.6287|0.9153|
|unit03|one_line|before|0.6892|0.6840|0.5609|0.1231|0.8389|0.8393|0.5000|0.5000|0.7895|0.7324|0.7269|
|unit03|one_line|after|0.9588|0.7605|0.6323|0.1283|0.8389|0.8393|0.5000|0.5000|0.7998|0.7876|0.9605|
|unit04|canonical_cycle|before|0.4705|0.5313|0.6302|-0.0989|0.8389|0.8405|0.6798|0.6805|0.6052|0.6353|0.5162|
|unit04|canonical_cycle|after|0.9729|0.7246|0.6178|0.1068|0.8389|0.8405|0.6798|0.6805|0.6206|0.6713|0.9192|
|unit04|one_line|before|0.7068|0.7167|0.6582|0.0585|0.8389|0.8405|0.5000|0.5000|0.7787|0.7704|0.6946|
|unit04|one_line|after|0.9670|0.7957|0.7258|0.0700|0.8389|0.8405|0.5000|0.5000|0.7905|0.7885|0.9795|
|unit05|canonical_cycle|before|0.4516|0.4540|0.6404|-0.1864|0.8373|0.8391|0.6818|0.6823|0.6053|0.6084|0.5390|
|unit05|canonical_cycle|after|0.9715|0.7175|0.6639|0.0536|0.8373|0.8391|0.6818|0.6823|0.6194|0.6310|0.9431|
|unit05|one_line|before|0.7433|0.7161|0.5677|0.1484|0.8373|0.8391|0.5000|0.5000|0.7965|0.7940|0.6996|
|unit05|one_line|after|0.9518|0.7855|0.6240|0.1615|0.8373|0.8391|0.5000|0.5000|0.8088|0.8187|0.9522|

## Four-new-unit means and variability

|Encoding|Point|Metric|Mean|Sample SD|Min|Max|
|---|---|---|---:|---:|---:|---:|
|canonical_cycle|after|source_own|0.9809|0.0119|0.9715|0.9974|
|canonical_cycle|after|target_own|0.9218|0.0147|0.9099|0.9431|
|canonical_cycle|after|fit_target_inverse|0.7401|0.0266|0.7175|0.7772|
|canonical_cycle|after|fit_initial_inverse|0.6208|0.0065|0.6137|0.6294|
|canonical_cycle|after|fit_fixed_inverse|0.6456|0.0200|0.6287|0.6713|
|canonical_cycle|after|source_n_i_prior|0.8385|0.0008|0.8373|0.8389|
|canonical_cycle|after|target_n_j_prior|0.8394|0.0007|0.8389|0.8405|
|canonical_cycle|after|source_token_kind_prior|0.6807|0.0008|0.6798|0.6818|
|canonical_cycle|after|target_token_kind_prior|0.6813|0.0008|0.6805|0.6823|
|canonical_cycle|after|best_common_control|0.6628|0.0418|0.6178|0.7183|
|canonical_cycle|after|correct_minus_best_common|0.0773|0.0253|0.0536|0.1068|
|canonical_cycle|before|source_own|0.4799|0.0707|0.4168|0.5806|
|canonical_cycle|before|target_own|0.4984|0.0441|0.4362|0.5390|
|canonical_cycle|before|fit_target_inverse|0.4943|0.0325|0.4540|0.5313|
|canonical_cycle|before|fit_initial_inverse|0.6057|0.0082|0.5962|0.6163|
|canonical_cycle|before|fit_fixed_inverse|0.6193|0.0278|0.5857|0.6478|
|canonical_cycle|before|source_n_i_prior|0.8385|0.0008|0.8373|0.8389|
|canonical_cycle|before|target_n_j_prior|0.8394|0.0007|0.8389|0.8405|
|canonical_cycle|before|source_token_kind_prior|0.6807|0.0008|0.6798|0.6818|
|canonical_cycle|before|target_token_kind_prior|0.6813|0.0008|0.6805|0.6823|
|canonical_cycle|before|best_common_control|0.6149|0.0338|0.5653|0.6404|
|canonical_cycle|before|correct_minus_best_common|-0.1205|0.0468|-0.1864|-0.0786|
|one_line|after|source_own|0.9561|0.0088|0.9468|0.9670|
|one_line|after|target_own|0.9600|0.0141|0.9476|0.9795|
|one_line|after|fit_target_inverse|0.7740|0.0198|0.7541|0.7957|
|one_line|after|fit_initial_inverse|0.7985|0.0079|0.7905|0.8088|
|one_line|after|fit_fixed_inverse|0.8036|0.0179|0.7876|0.8195|
|one_line|after|source_n_i_prior|0.8385|0.0008|0.8373|0.8389|
|one_line|after|target_n_j_prior|0.8394|0.0007|0.8389|0.8405|
|one_line|after|source_token_kind_prior|0.5000|0.0000|0.5000|0.5000|
|one_line|after|target_token_kind_prior|0.5000|0.0000|0.5000|0.5000|
|one_line|after|best_common_control|0.6545|0.0478|0.6240|0.7258|
|one_line|after|correct_minus_best_common|0.1195|0.0378|0.0700|0.1615|
|one_line|before|source_own|0.7186|0.0251|0.6892|0.7433|
|one_line|before|target_own|0.7064|0.0143|0.6946|0.7269|
|one_line|before|fit_target_inverse|0.7065|0.0154|0.6840|0.7167|
|one_line|before|fit_initial_inverse|0.7879|0.0074|0.7787|0.7965|
|one_line|before|fit_fixed_inverse|0.7729|0.0293|0.7324|0.7948|
|one_line|before|source_n_i_prior|0.8385|0.0008|0.8373|0.8389|
|one_line|before|target_n_j_prior|0.8394|0.0007|0.8389|0.8405|
|one_line|before|source_token_kind_prior|0.5000|0.0000|0.5000|0.5000|
|one_line|before|target_token_kind_prior|0.5000|0.0000|0.5000|0.5000|
|one_line|before|best_common_control|0.5877|0.0470|0.5609|0.6582|
|one_line|before|correct_minus_best_common|0.1188|0.0417|0.0585|0.1484|

## Coverage and interpretation

All source/target internal, closure, singleton and both-internal strata are retained without selecting convenient lengths or dropping undefined cells. One-line and cycle-only kinds naturally have no edges in the other encoding. In canonical cycle notation, source closures are all negative, inverse-target closures are all paired positives, and singletons are negative. Their within-stratum AUROC can therefore be structurally undefined. Such entries are not zero accuracy or evidence of absent information.

Positional and token-kind priors can rank labels without recovering an input-specific neural computation. A high correct-map AUROC is therefore read together with these controls, own-model readouts and initial/fixed-target controls. Differences between scores are comparisons under this frozen diagnostic, not an algebraic separation of independent information sources or proof that the counting network naturally uses this computation.

Raw per-edge scores preserve anchor IDs, source i, paired target j, before/after token positions, edge kinds, labels and support masks in the external stage-two score archives listed by the per-unit completion receipts. This reader uses saved AUROCs only; it does not fit, forward, select methods, select lengths, or change the main cross-pair result.

Tables: `token_four_unit_table.csv` (new-unit all-edge primary/sensitivity rows), `token_historical_reference.csv`, `token_per_unit_all_strata.csv`, `token_four_unit_stats.csv`, `token_before_after_by_unit.csv`, `token_before_after_stats.csv`, `token_coverage_by_length.csv`, `token_aggregation_checks.csv`.
