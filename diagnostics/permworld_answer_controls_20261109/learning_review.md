# Independent learning-diagnostic review

Status: **passed**; 320 specialist summaries, 6720 prescribed model/length rows, 75782 checks, 0 failures.

This reads saved outputs, full answer-position cross-entropies, training labels and diagnostic CSVs only. It does not load or fit model weights, run a forward pass, generate inputs, or compute new CKA.

All full-vocabulary NLLs are checked separately from the conditional numeric NLL. Negative `full_nll_minus_length_prior_nll` means better observed logloss than the train-only smoothed prior. Spearman uses the conditional numeric expectation. Neither a small NLL gap nor a nonsignificant association establishes practical equivalence. Undefined correlations are retained.

## GRU + canonical-cycle peaks and valleys

| Unit | Task | Accuracy minus mode | Full NLL minus prior | Conditional NLL minus prior | Expectation Spearman | Undefined rho lengths |
|---|---|---:|---:|---:|---:|---:|
| archdev001 | peaks | 0.008742559523809524 | -0.04295975818302501 | -0.043093284085744005 | 0.2715473671988999 | 0 |
| archdev001 | valleys | 0.017764136904761904 | -0.043558399187552894 | -0.04365675513326931 | 0.2770390060660751 | 0 |
| unit02 | peaks | 0.003162202380952381 | -0.011143190484773816 | -0.011214621430507691 | 0.12791439301119428 | 0 |
| unit02 | valleys | 0.01711309523809524 | -0.04304572407763953 | -0.04314078060469988 | 0.2752789686479324 | 0 |
| unit03 | peaks | 0.0009300595238095238 | -0.015519567569713405 | -0.01560441024986825 | 0.15774678454362473 | 0 |
| unit03 | valleys | 0.012834821428571428 | -0.06466445535696735 | -0.06483221627870327 | 0.32255886408163903 | 0 |
| unit04 | peaks | 0.011811755952380952 | -0.031919842241985484 | -0.0320318788496015 | 0.2347553264997067 | 0 |
| unit04 | valleys | 0.014508928571428572 | -0.03630868230577247 | -0.03642893152616965 | 0.25111876678001066 | 0 |
| unit05 | peaks | 0.012648809523809524 | -0.03676444268758651 | -0.0368781744815229 | 0.24761181339873262 | 0 |
| unit05 | valleys | 0.027529761904761904 | -0.0620860507045985 | -0.06222507104618105 | 0.3297353914893018 | 0 |

These observations concern identity inputs only. They do not assert that the same learning statistics were measured on transformed partner inputs. The geometry association join records this identity fallback explicitly.

Positive expectation correlation can coexist with worse-than-prior logloss: ordering, calibration and probability mass outside the answer support are different properties. This audit therefore does not turn exact-match ties, near-zero gaps, or undefined correlations into a declaration that a model is exactly at the length baseline.

The JSON retains every specialist's summary, all 210 peaks/valleys length rows, undefined reasons, numerical identity checks and source guards. Counted specialists/lengths are an inventory, not independent repetitions for a statistical test.
