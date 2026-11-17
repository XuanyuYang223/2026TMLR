# Local-edit geometry and early-path gradient audit

This is an exploratory audit of already observed null count interventions, not a retrospective preregistered confirmation. It retains the exact prior dose 1, L1, right-neighbor update, donors, signs, eight fixed random directions, and independent checker. Formal summaries use units02–05; `archdev001` is a historical engineering reference. No model, probe, direction, threshold, layer or dose was fitted or selected.

## Coordinate and projection audit

All geometry uses the **raw, unnormalized 256-coordinate recurrent L1 state** and its Euclidean metric. The stored probe was originally fitted after StandardScaler, but its saved raw-space weight is `ridge_coef / scaler_scale`. The intervention axis is the normalized raw weight. All ten saved probes pass this conversion check. Thus the ideal existing edit is the raw-space orthogonal projection `(d·v)v`, with `d = donor L1 − recipient L1`.

The actual displacement is taken from the previous experiment's saved injected FP32 state minus the recipient state, not reconstructed and silently substituted. The maximum actual-vs-ideal L2 discrepancy is 1.88381973e-07; the original coordinate dot-product discrepancy is 0. The discrepancy is the recorded FP32 cast. Both ideal projection and actual displacement vectors are preserved. Projection energy means `(d·v)^2 / ||d||^2`, not an angle or standardized-space quantity.

## Registered four-unit results

Every entry is mean ± **sample SD across four units**. Registered cases are averaged within center, centers equally within length, and all 21 lengths n10–30 use fixed1/21 weights. Empty quantities remain undefined; no length is reweighted to improve coverage. “main” is the local-only donor construction; “matched” is the cumulative-count-matched donor construction. All signs remain separate.

|Task|Donors|Sign|Projection energy %|Actual displacement L2|Gradient L2|Angle degrees|Undefined angle cases|
|---|---|---|---|---|---|---|---|
|peaks|main|-1|0.630496 ± 0.114309|0.237969 ± 0.0329813|0.24608 ± 0.0404387|88.3878 ± 1.58302|0|
|peaks|main|1|0.630496 ± 0.114309|0.237969 ± 0.0329813|0.379455 ± 0.0460377|91.6308 ± 0.747175|0|
|peaks|matched|-1|0.641649 ± 0.115913|0.23984 ± 0.0331802|0.257691 ± 0.0463071|88.4793 ± 1.86727|0|
|peaks|matched|1|0.641649 ± 0.115913|0.23984 ± 0.0331802|0.37937 ± 0.0448954|91.1947 ± 0.904203|0|
|valleys|main|-1|0.705571 ± 0.306688|0.261756 ± 0.0608|0.239601 ± 0.00854303|89.863 ± 1.44523|0|
|valleys|main|1|0.705571 ± 0.306688|0.261756 ± 0.0608|0.345563 ± 0.0557908|89.8219 ± 1.21666|0|
|valleys|matched|-1|0.719486 ± 0.308754|0.263993 ± 0.0604733|0.263927 ± 0.0132042|89.7352 ± 1.42228|0|
|valleys|matched|1|0.719486 ± 0.308754|0.263993 ± 0.0604733|0.394963 ± 0.0960385|89.6626 ± 0.956817|0|

The angle uses the gradient and the actual signed edit vector. Norms ≤1e−12 have undefined angles; exact zero and numerical-near-zero flags are recorded separately in the raw and unit tables. Undefined counts above are raw cases over four formal units, not additional independent replications.

## Behavioral margin and directional derivatives

The fixed scalar is `m = z[c+s] − z[c]`, with **c the recipient's true original count**, even when the donor's total count is matched. It is already oriented toward the local goal and is not multiplied by the sign again.

Autograd differentiates through current-token upper replay from the recipient previous upper states, and then through the original suffix. The leaf is raw recipient L1 immediately after the current token. There is no inference-mode helper, cached activation substitution or detach on that graph. Frozen model parameters have `requires_grad=False`, while the state leaf has `requires_grad=True`. All modules remain in eval mode; CuDNN is disabled only to support eval-mode recurrent backward.

|Task|Donors|Sign|g·actual delta|Random equal-norm mean derivative|Local − random mean|Cached actual margin change|
|---|---|---|---|---|---|---|
|peaks|main|-1|0.00201543 ± 0.00185604|0.000176019 ± 0.000580681|0.00183941 ± 0.00129213|0.00206501 ± 0.00189706|
|peaks|main|1|-0.00179168 ± 0.000744918|0.000222534 ± 0.000848926|-0.00201421 ± 0.000694562|-0.00174617 ± 0.000776738|
|peaks|matched|-1|0.00205331 ± 0.00201798|7.73995e-05 ± 0.000630401|0.00197591 ± 0.0014284|0.00210651 ± 0.00206869|
|peaks|matched|1|-0.00127382 ± 0.00133212|0.000332935 ± 0.000675087|-0.00160676 ± 0.0011913|-0.00121355 ± 0.00138918|
|valleys|main|-1|7.90353e-05 ± 0.00146926|-0.000383232 ± 0.000611039|0.000462267 ± 0.000885719|9.80506e-05 ± 0.00148366|
|valleys|main|1|0.000565994 ± 0.00123593|-0.00040958 ± 0.00105455|0.000975574 ± 0.00186333|0.000639989 ± 0.00123502|
|valleys|matched|-1|0.000177337 ± 0.00166585|-0.000356544 ± 0.000765438|0.000533881 ± 0.00100607|0.000198293 ± 0.00168012|
|valleys|matched|1|0.0012526 ± 0.00083086|-0.000310719 ± 0.00140669|0.00156332 ± 0.00216455|0.00133063 ± 0.000811138|

The eight condition means of gradient norm are about0.240–0.395, with no zero/near-zero gradient cases under the frozen threshold. Thus this is **not a zero-gradient result**. Mean angles range88.388–91.631 degrees; mean projection energy fractions are0.6305–0.7195%. Actual margin changes are signed and not uniformly goal-positive: their condition means range−0.001746 to+0.002107, while mean original margins range−9.954 to−9.188. These are weighted means of per-case quantities, not a ratio computed from separately averaged norms. They do not change the prior zero modal-count-change result.

All eight original random displacements are retained. Their primary comparison is rescaled algebraically to exactly the local edit's raw Euclidean norm, without any new intervention or forward. The maximum original cached norm mismatch is 5.4327008e-08, within the original1e−5+1e−5×norm rule. The displayed random mean is **signed** and can cancel; it is not a mean absolute sensitivity. Each direction, cosine and actual cached endpoint change is available in `results/<unit>/random_directions.csv.gz`.

Actual margin changes come from the original saved FP32 early-replay logits, whereas gradients use the CuDNN-off implementation of the same graph. Maximum baseline and edited logit discrepancies are 2.28881836e-05 and 1.62124634e-05, both below the unchanged1e−4 guard. This guard can exceed tiny endpoint changes. Consequently `linear_prediction_error` is an arithmetic comparison between these quantities, **not evidence about curvature or a failed linear approximation**. FP32 autograd is the derivative of the nominal smooth operations, not of floating-point rounding.

## Independent fixed-subset verification

Four synthetic tests passed: independent gate unroll, derivative connectivity with frozen parameters, directional finite differences and both local-goal signs. The preset actual-model subset includes n10/20/30, center6, first saved pair per construction/sign, both tasks/all five units: **120 cases**, two directions each. Symmetric finite differences use epsilon1e−4 in an FP64 copy of the exact frozen weight values, with the fixed tolerance1e−7+1e−3×|analytic|. This shadow check validates the nominal graph, not a new primary model or FP32 rounding response.

All 240 shadow directional checks passed; maximum absolute error 1.2894854e-10. The read-only link from saved primary FP32 gradients to that shadow also passed, with maximum directional discrepancy 3.8759997e-08. Fixed-subset independent projection and dot-product arithmetic error was 1.11022302e-16. See [the audit receipt](saved_gradient_audit.json) and [primary-to-shadow directional table](primary_gradient_fd_link.csv).

Every unit has an explicit input binding file, with fresh start/end hashes of all consumed checkpoint, probe, old intervention cell, new timing cell, selected-family and relevant receipt inputs. All ten actual model parameter hashes are unchanged. This is a named-input endpoint audit, not whole-disk or continuous monitoring.

## Scope of interpretation

These results describe local geometry and sensitivity at the **tested early-replay interface**. Projection energy, a small derivative or a near-right angle does not prove that the network ignores extrema, that causal information must be highly distributed, or that a probe axis is a natural lemma. Whole-layer replacement and the existing null one-dimensional edit differ in payload. The prior absence of modal count changes is retained; this audit does not turn small numerical shifts into functional success.

Files: [all units](all_units.csv), [four-unit mean/SD/range/raw values](four_unit_summary.csv), per-unit `by_length.csv`, all per-pair and eight-direction CSVs, and the full raw d/g/delta/ideal/random vectors in `/mnt/yangx-projects-f/20261110/permworld_closing_audit/gradient/<unit>/cells/`. Unit receipts bind every new output to its source cells, checkpoint, data identifier and [specification](spec.json). The physical data file itself is not reread by this stage; actual permutations and oracle counts are taken from the exact saved selected-family input.

Reproduce from the repository root with the existing vendor/dependency PYTHONPATH:

```bash
.venv/bin/python -m experiments.permworld_closing_audit_20261110.gradient_run freeze
.venv/bin/python -m experiments.permworld_closing_audit_20261110.gradient_run all
.venv/bin/python diagnostics/permworld_closing_audit_20261110/gradient/audit_saved_gradients.py
.venv/bin/python diagnostics/permworld_closing_audit_20261110/gradient/build_report.py
```

Completed units are hash-checked and reused. The stage never invokes the parent's training or probe calibration workflow. No DAS, direction search, dose sweep, cross-model swap or new-task experiment follows this audit.
