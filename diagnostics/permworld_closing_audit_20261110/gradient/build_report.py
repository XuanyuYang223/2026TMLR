"""Render only the registered gradient summary; no new neural evaluation."""
from pathlib import Path
import csv
import json

HERE = Path(__file__).resolve().parent


def table(headers, rows):
    return '\n'.join(['|' + '|'.join(headers) + '|', '|' + '|'.join(['---'] * len(headers)) + '|',
                      *['|' + '|'.join(map(str, row)) + '|' for row in rows]])


def main():
    with (HERE / 'four_unit_summary.csv').open() as handle:
        summary = list(csv.DictReader(handle))
    lookup = {(row['task'], row['group'], row['sign'], row['metric']): row for row in summary}
    def value(task, group, sign, metric, scale=1):
        row = lookup[task, group, sign, metric]
        if row['mean'] == '': return 'undefined'
        return f"{scale*float(row['mean']):.6g} ± {scale*float(row['sample_sd']):.6g}"
    cells = [(task, group, sign) for task in ['peaks','valleys'] for group in ['main','matched'] for sign in ['-1','1']]
    geometry = table(['Task','Donors','Sign','Projection energy %','Actual displacement L2','Gradient L2','Angle degrees','Undefined angle cases'],
        [[task,group,sign,value(task,group,sign,'projection_energy_fraction',100),
          value(task,group,sign,'actual_displacement_norm'),value(task,group,sign,'gradient_norm'),
          value(task,group,sign,'angle_degrees'),lookup[task,group,sign,'angle_degrees']['undefined_samples']]
         for task,group,sign in cells])
    derivatives = table(['Task','Donors','Sign','g·actual delta','Random equal-norm mean derivative','Local − random mean','Cached actual margin change'],
        [[task,group,sign,*[value(task,group,sign,metric) for metric in
          ['g_dot_delta','random_equal_norm_mean_g_dot_delta','local_minus_random_equal_norm_g_dot_delta','actual_logit_margin_change']]]
         for task,group,sign in cells])
    reports = [json.loads((HERE/'results'/unit/'completion.json').read_text())
               for unit in ['archdev001','unit02','unit03','unit04','unit05']]
    maximum = lambda field: max(task['maxima'][field] for receipt in reports for task in receipt['task_reports'])
    audit = json.loads((HERE/'saved_gradient_audit.json').read_text())
    spec = json.loads((HERE/'spec.json').read_text())
    fd_max = 0.
    for unit in ['archdev001','unit02','unit03','unit04','unit05']:
        with (HERE/'results'/unit/'finite_difference.csv').open() as handle:
            fd_max=max(fd_max,max(float(row['absolute_error']) for row in csv.DictReader(handle)))
    text = f'''# Local-edit geometry and early-path gradient audit

This is an exploratory audit of already observed null count interventions, not a retrospective preregistered confirmation. It retains the exact prior dose 1, L1, right-neighbor update, donors, signs, eight fixed random directions, and independent checker. Formal summaries use units02–05; `archdev001` is a historical engineering reference. No model, probe, direction, threshold, layer or dose was fitted or selected.

## Coordinate and projection audit

All geometry uses the **raw, unnormalized 256-coordinate recurrent L1 state** and its Euclidean metric. The stored probe was originally fitted after StandardScaler, but its saved raw-space weight is `ridge_coef / scaler_scale`. The intervention axis is the normalized raw weight. All ten saved probes pass this conversion check. Thus the ideal existing edit is the raw-space orthogonal projection `(d·v)v`, with `d = donor L1 − recipient L1`.

The actual displacement is taken from the previous experiment's saved injected FP32 state minus the recipient state, not reconstructed and silently substituted. The maximum actual-vs-ideal L2 discrepancy is {maximum('ideal_vs_actual_l2'):.9g}; the original coordinate dot-product discrepancy is {maximum('projection_coordinate_error'):.9g}. The discrepancy is the recorded FP32 cast. Both ideal projection and actual displacement vectors are preserved. Projection energy means `(d·v)^2 / ||d||^2`, not an angle or standardized-space quantity.

## Registered four-unit results

Every entry is mean ± **sample SD across four units**. Registered cases are averaged within center, centers equally within length, and all 21 lengths n10–30 use fixed1/21 weights. Empty quantities remain undefined; no length is reweighted to improve coverage. “main” is the local-only donor construction; “matched” is the cumulative-count-matched donor construction. All signs remain separate.

{geometry}

The angle uses the gradient and the actual signed edit vector. Norms ≤1e−12 have undefined angles; exact zero and numerical-near-zero flags are recorded separately in the raw and unit tables. Undefined counts above are raw cases over four formal units, not additional independent replications.

## Behavioral margin and directional derivatives

The fixed scalar is `m = z[c+s] − z[c]`, with **c the recipient's true original count**, even when the donor's total count is matched. It is already oriented toward the local goal and is not multiplied by the sign again.

Autograd differentiates through current-token upper replay from the recipient previous upper states, and then through the original suffix. The leaf is raw recipient L1 immediately after the current token. There is no inference-mode helper, cached activation substitution or detach on that graph. Frozen model parameters have `requires_grad=False`, while the state leaf has `requires_grad=True`. All modules remain in eval mode; CuDNN is disabled only to support eval-mode recurrent backward.

{derivatives}

The eight condition means of gradient norm are about0.240–0.395, with no zero/near-zero gradient cases under the frozen threshold. Thus this is **not a zero-gradient result**. Mean angles range88.388–91.631 degrees; mean projection energy fractions are0.6305–0.7195%. Actual margin changes are signed and not uniformly goal-positive: their condition means range−0.001746 to+0.002107, while mean original margins range−9.954 to−9.188. These are weighted means of per-case quantities, not a ratio computed from separately averaged norms. They do not change the prior zero modal-count-change result.

All eight original random displacements are retained. Their primary comparison is rescaled algebraically to exactly the local edit's raw Euclidean norm, without any new intervention or forward. The maximum original cached norm mismatch is {maximum('random_norm_error'):.9g}, within the original1e−5+1e−5×norm rule. The displayed random mean is **signed** and can cancel; it is not a mean absolute sensitivity. Each direction, cosine and actual cached endpoint change is available in `results/<unit>/random_directions.csv.gz`.

Actual margin changes come from the original saved FP32 early-replay logits, whereas gradients use the CuDNN-off implementation of the same graph. Maximum baseline and edited logit discrepancies are {maximum('baseline_logit_error'):.9g} and {maximum('edited_logit_error'):.9g}, both below the unchanged1e−4 guard. This guard can exceed tiny endpoint changes. Consequently `linear_prediction_error` is an arithmetic comparison between these quantities, **not evidence about curvature or a failed linear approximation**. FP32 autograd is the derivative of the nominal smooth operations, not of floating-point rounding.

## Independent fixed-subset verification

Four synthetic tests passed: independent gate unroll, derivative connectivity with frozen parameters, directional finite differences and both local-goal signs. The preset actual-model subset includes n10/20/30, center6, first saved pair per construction/sign, both tasks/all five units: **120 cases**, two directions each. Symmetric finite differences use epsilon1e−4 in an FP64 copy of the exact frozen weight values, with the fixed tolerance1e−7+1e−3×|analytic|. This shadow check validates the nominal graph, not a new primary model or FP32 rounding response.

All 240 shadow directional checks passed; maximum absolute error {fd_max:.9g}. The read-only link from saved primary FP32 gradients to that shadow also {audit['status']}, with maximum directional discrepancy {audit['maximum_primary_fp32_vs_shadow_fp64_error']:.9g}. Fixed-subset independent projection and dot-product arithmetic error was {audit['maximum_independent_arithmetic_error']:.9g}. See [the audit receipt](saved_gradient_audit.json) and [primary-to-shadow directional table](primary_gradient_fd_link.csv).

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
'''
    (HERE/'README.md').write_text(text)
    print('Wrote',HERE/'README.md')


if __name__=='__main__': main()
