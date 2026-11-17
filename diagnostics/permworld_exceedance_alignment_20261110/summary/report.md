# Existing-cache position-alignment comparison

This is an exploratory cross-pair comparison using saved activations. The historical unit is an engineering reference; the four new units are reported individually and summarized with sample SD and range. This script fits no probe or map and performs no neural forward.

Full eligible support is primary; count-matched support is secondary. Within each length, AUROC is averaged over defined per-sample position rankings. Every length n=10–30 then has fixed weight 1/21. Undefined lengths and units remain undefined. The common maximum includes means-only, identity, the pair-specific incorrect-transform map, and each of eight shuffled-pair maps separately. Maximization occurs after length aggregation. Direct transfer, model-internal readouts, step-0 and count-prior diagnostics remain separate.

## Primary per-unit results

|Pair|Unit|Encoding|Direct|Correct map|Best common|Margin|Own source|Own target|
|---|---|---|---:|---:|---:|---:|---:|---:|
|peaks_valleys|archdev001|canonical_cycle|0.522982|0.542899|0.532888|0.010010|0.553548|0.558491|
|peaks_valleys|archdev001|one_line|0.503694|0.638166|0.516921|0.121245|0.728030|0.716697|
|peaks_valleys|unit02|canonical_cycle|0.512299|0.525944|0.519157|0.006787|0.515477|0.550276|
|peaks_valleys|unit02|one_line|0.501325|0.644224|0.519343|0.124881|0.748582|0.734723|
|peaks_valleys|unit03|canonical_cycle|0.498470|0.526845|0.513040|0.013805|0.523653|0.583480|
|peaks_valleys|unit03|one_line|0.504586|0.607053|0.521238|0.085815|0.728658|0.702372|
|peaks_valleys|unit04|canonical_cycle|0.518829|0.537633|0.528702|0.008931|0.544205|0.548664|
|peaks_valleys|unit04|one_line|0.503880|0.638056|0.522736|0.115320|0.723888|0.725546|
|peaks_valleys|unit05|canonical_cycle|0.516024|0.540644|0.525869|0.014775|0.543139|0.589669|
|peaks_valleys|unit05|one_line|0.502895|0.643224|0.513989|0.129235|0.754016|0.768417|
|exceedances_deficiencies|archdev001|canonical_cycle|0.820612|0.830079|0.830460|-0.000380|0.830988|0.829991|
|exceedances_deficiencies|archdev001|one_line|0.725662|0.786145|0.778156|0.007989|0.854565|0.864894|
|exceedances_deficiencies|unit02|canonical_cycle|0.800464|0.827699|0.826181|0.001518|0.830624|0.828865|
|exceedances_deficiencies|unit02|one_line|0.724880|0.778952|0.767548|0.011404|0.857444|0.863164|
|exceedances_deficiencies|unit03|canonical_cycle|0.829524|0.831831|0.832326|-0.000495|0.830722|0.831201|
|exceedances_deficiencies|unit03|one_line|0.715514|0.788133|0.777920|0.010213|0.856690|0.859514|
|exceedances_deficiencies|unit04|canonical_cycle|0.825719|0.831572|0.832157|-0.000585|0.830865|0.832126|
|exceedances_deficiencies|unit04|one_line|0.731104|0.779861|0.772475|0.007386|0.858745|0.872118|
|exceedances_deficiencies|unit05|canonical_cycle|0.822425|0.829522|0.829362|0.000160|0.828717|0.831088|
|exceedances_deficiencies|unit05|one_line|0.722235|0.784848|0.773524|0.011324|0.856049|0.856373|

## Step-0 and position-prior diagnostics

The old peaks→valleys initial phase refits a probe with an initial source and target. The new exceedances→deficiencies step-0 target control instead preserves the trained source and probe. These are different references, not interchangeable initialization effects. Prior and fixed-point controls are excluded from the common maximum and kept individually.

|Pair|Unit|Encoding|PV initial-source map|ED initial target|ED fixed-point target|Source prior|Reindexed target prior|
|---|---|---|---:|---:|---:|---:|---:|
|peaks_valleys|archdev001|canonical_cycle|0.503242|undefined|undefined|undefined|undefined|
|peaks_valleys|archdev001|one_line|0.500772|undefined|undefined|undefined|undefined|
|peaks_valleys|unit02|canonical_cycle|0.505464|undefined|undefined|undefined|undefined|
|peaks_valleys|unit02|one_line|0.502186|undefined|undefined|undefined|undefined|
|peaks_valleys|unit03|canonical_cycle|0.503650|undefined|undefined|undefined|undefined|
|peaks_valleys|unit03|one_line|0.501908|undefined|undefined|undefined|undefined|
|peaks_valleys|unit04|canonical_cycle|0.500787|undefined|undefined|undefined|undefined|
|peaks_valleys|unit04|one_line|0.502201|undefined|undefined|undefined|undefined|
|peaks_valleys|unit05|canonical_cycle|0.506022|undefined|undefined|undefined|undefined|
|peaks_valleys|unit05|one_line|0.502437|undefined|undefined|undefined|undefined|
|exceedances_deficiencies|archdev001|canonical_cycle|undefined|0.839045|0.822880|0.839056|0.838518|
|exceedances_deficiencies|archdev001|one_line|undefined|0.840719|0.748330|0.839056|0.838518|
|exceedances_deficiencies|unit02|canonical_cycle|undefined|0.838734|0.818695|0.838754|0.838878|
|exceedances_deficiencies|unit02|one_line|undefined|0.840438|0.732198|0.838754|0.838878|
|exceedances_deficiencies|unit03|canonical_cycle|undefined|0.838894|0.824167|0.838900|0.839321|
|exceedances_deficiencies|unit03|one_line|undefined|0.840404|0.748358|0.838900|0.839321|
|exceedances_deficiencies|unit04|canonical_cycle|undefined|0.838913|0.827037|0.838898|0.840454|
|exceedances_deficiencies|unit04|one_line|undefined|0.841907|0.745212|0.838898|0.840454|
|exceedances_deficiencies|unit05|canonical_cycle|undefined|0.837298|0.824275|0.837316|0.839085|
|exceedances_deficiencies|unit05|one_line|undefined|0.838731|0.745483|0.837316|0.839085|

## Paired encoding contrasts

For peaks→valleys the contrast is one-line minus cycle; for exceedances→deficiencies it is cycle minus one-line. Both contrasts compare correct-minus-best-common margins, not raw aligned AUROC. Their signs are reported together within each unit; task pairs, encodings, lengths and shuffle seeds are not additional independent units.

|Unit|Support|PV one-line−cycle|ED cycle−one-line|Both positive|
|---|---|---:|---:|---|
|archdev001|eligible|0.111234|-0.008370|False|
|archdev001|matched|0.120512|-0.007645|False|
|unit02|eligible|0.118094|-0.009886|False|
|unit02|matched|0.127271|-0.009674|False|
|unit03|eligible|0.072010|-0.010708|False|
|unit03|matched|0.090066|-0.013674|False|
|unit04|eligible|0.106389|-0.007971|False|
|unit04|matched|0.127880|-0.005920|False|
|unit05|eligible|0.114460|-0.011164|False|
|unit05|matched|0.120468|-0.010211|False|

## Four-new-unit summaries

|Pair|Encoding|Support|Metric|Mean|Sample SD|Min|Max|Positive units|
|---|---|---|---|---:|---:|---:|---:|---:|
|exceedances_deficiencies|canonical_cycle|eligible|correct_map_auc|0.830156|0.001937|0.827699|0.831831|4|
|exceedances_deficiencies|canonical_cycle|eligible|margin_vs_best_common|0.000150|0.000971|-0.000585|0.001518|2|
|exceedances_deficiencies|canonical_cycle|eligible|gain_vs_direct|0.010623|0.011259|0.002307|0.027234|4|
|exceedances_deficiencies|canonical_cycle|matched|correct_map_auc|0.834998|0.001550|0.832840|0.836517|4|
|exceedances_deficiencies|canonical_cycle|matched|margin_vs_best_common|-0.000414|0.001082|-0.001241|0.001153|1|
|exceedances_deficiencies|canonical_cycle|matched|gain_vs_direct|0.011191|0.012208|0.003696|0.029423|4|
|exceedances_deficiencies|one_line|eligible|correct_map_auc|0.782949|0.004320|0.778952|0.788133|4|
|exceedances_deficiencies|one_line|eligible|margin_vs_best_common|0.010082|0.001878|0.007386|0.011404|4|
|exceedances_deficiencies|one_line|eligible|gain_vs_direct|0.059515|0.010435|0.048757|0.072619|4|
|exceedances_deficiencies|one_line|matched|correct_map_auc|0.791325|0.005928|0.785377|0.799204|4|
|exceedances_deficiencies|one_line|matched|margin_vs_best_common|0.009456|0.003470|0.004914|0.013114|4|
|exceedances_deficiencies|one_line|matched|gain_vs_direct|0.068212|0.015634|0.050666|0.087880|4|
|peaks_valleys|canonical_cycle|eligible|correct_map_auc|0.532766|0.007469|0.525944|0.540644|4|
|peaks_valleys|canonical_cycle|eligible|margin_vs_best_common|0.011074|0.003835|0.006787|0.014775|4|
|peaks_valleys|canonical_cycle|eligible|gain_vs_direct|0.021361|0.006478|0.013645|0.028374|4|
|peaks_valleys|canonical_cycle|matched|correct_map_auc|0.528330|0.008903|0.520471|0.538346|4|
|peaks_valleys|canonical_cycle|matched|margin_vs_best_common|0.003862|0.007109|-0.002200|0.012272|2|
|peaks_valleys|canonical_cycle|matched|gain_vs_direct|0.018468|0.008336|0.009200|0.027615|4|
|peaks_valleys|one_line|eligible|correct_map_auc|0.633139|0.017600|0.607053|0.644224|4|
|peaks_valleys|one_line|eligible|margin_vs_best_common|0.113813|0.019549|0.085815|0.129235|4|
|peaks_valleys|one_line|eligible|gain_vs_direct|0.129968|0.018696|0.102467|0.142899|4|
|peaks_valleys|one_line|matched|correct_map_auc|0.637309|0.018614|0.610032|0.651792|4|
|peaks_valleys|one_line|matched|margin_vs_best_common|0.120283|0.015691|0.097303|0.132739|4|
|peaks_valleys|one_line|matched|gain_vs_direct|0.127135|0.019732|0.097680|0.139691|4|

## Interpretation limits

Position AUROC measures recoverable and transferable position information under the frozen linear-probe and orthogonal-map protocol. It does not establish the neural counting mechanism, natural computational reuse, task accuracy gains or universal encoding effects. Per-sample true counts are only scoring/stratification information (and explicitly labeled oracle-prior diagnostics), never new fitting targets or test-time threshold selection. Shared checkpoints and inputs induce dependence. Undefined entries, support counts and every individual control are retained in the CSVs.
