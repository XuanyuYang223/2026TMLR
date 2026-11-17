# Local recurrent-state intervention: design review before new outcomes

This stage asks whether a low-dimensional change in a **real recurrent state**, directed by a local extrema readout, changes the frozen model's final count. It is separate from prefix-final orthogonal alignment, representational similarity, downstream mask composition, and the matrix/F17 research line. Historical data can select a geometric manipulation; unit02–unit05 supply the four paired formal units. No additional stage is authorized by a favorable result.

## Endpoints and counterfactual meaning

For an interior position i, let b be its true local peak/valley indicator and let s be the desired change, +1 or −1. Every treatment and control for that recipient must retain the same desired local endpoint b+s and the same desired final count `recipient_true_count + s`.

In the count-matched donor group, an earlier opposite local flip compensates for the flip at i, so the donor's resolved-prefix total can equal the recipient's. This is exactly why **the donor's whole count is not the intended endpoint of the local-only intervention**. Preserve both counts in the raw table and verify the local-only counterfactual separately. The actual local-only counterfactual is a positive control, not a fitted target.

Full-state replacement is a different intervention. Its oracle final answer is the donor's complete answer only when the transplanted donor prefix is followed by exactly the donor's future tokens. Otherwise it corresponds to a prefix/suffix splice, whose validity and answer must be established separately. Do not call an arbitrary splice a successful donor-state positive control.

## Manipulation check and selection

For a frozen linear probe `a(h)=w·h+b`, the dose-one projection

`h_edit = h_recipient + ((a(h_donor)-a(h_recipient))/||w||²) w`

copies the donor's probe score exactly. This algebraic fact is not an empirical causal result. The useful manipulation check requires a **true boundary crossing**: the recipient starts on its own-label side of the source-validation boundary and the edit ends on the desired opposite side. Also report post-edit desired-side accuracy and signed score movement separately, so pre-existing target-side classifications cannot masquerade as crossings.

A crossing-rate target of 0.80 is fixed as an engineering manipulation criterion, not a theorem or power calculation. The finalized design uses AFTER the right-value token only for the primary intervention, dose one, and one common layer across the two historical tasks. Layer selection maximizes the minimum independent-checker crossing rate across task × group × sign; ties choose the higher layer. If no candidate qualifies, freeze the best geometric candidate and explicitly label the manipulation weak; do not search using final-count effects. BEFORE is a readout diagnostic only.

The edit direction is a fixed-regularization Ridge classifier (lambda 10, boundary 0.5) fitted only on count-matched FIT_A states. An independent same-family checker fitted on component-disjoint FIT_B states supplies the primary manipulation check. A separate FIT_B prefix-count decoder is a diagnostic, not a new intervention condition. Each formal model fits this same declared pipeline; the common layer/timing/dose cannot adapt to formal outputs. Final count labels, changed final probabilities and counterfactual count accuracy cannot choose the candidate.

## Required controls

- **Same-label real donor:** match task, length, queried position/cut, resolved-prefix count, valid future-token suffix, and every feasible declared nuisance constraint. Its local label stays the recipient's. Save donor IDs, labels, selection failures and perturbation norms. Natural same-label score projections can be much smaller than opposite-label projections, so this comparison alone does not isolate semantic content at matched perturbation magnitude.
- **Random direction:** use the same recipient state, layer, timing and dose, with L2 magnitude equal to that pair's treatment perturbation. Pair random signs and use fixed seeds. Average the multiple random controls within recipient and unit; they are not additional experimental units.
- **Unedited and wiring controls:** exact continuation without editing must reproduce the ordinary output. Full-state replacement and the true local-only counterfactual remain separate positive controls with their own oracle endpoints; they are not part of a convenient best-control maximum.
- **Both flip directions:** keep +1 and −1 effects separate before any equally weighted sign summary. A one-sided result must not be hidden by unequal pair counts.

## Final-output reporting

Report signed conditional numeric-expectation change, `s*(E_edited−E_recipient)`, together with the probability mass assigned to numeric tokens 0–30. Conditioning on numeric outputs can otherwise conceal a shift into nonnumeric outputs. Preserve the full-vocabulary argmax and mark nonnumeric predictions explicitly.

The exact desired-count endpoint is whether the full-vocabulary argmax is the true recipient count plus s. It is not merely whether the edited argmax differs by one from an already incorrect baseline. Report both the exact endpoint and the signed argmax displacement when both predictions are numeric. The baseline-correct subset is a diagnostic, not the primary sample definition.

The three descriptive questions are: did the targeted local signal change; did the final count respond in the intended direction relative to controls; and did that response remain in the count-matched group? Use unit-level paired treatment-minus-control effects, means, sample SD, range and raw unit values. Do not treat recipients, positions, random directions or the two tasks as independent replications of four units.

## Limits on interpretation

Count matching removes one particularly direct explanation: a donor already differing in the resolved-prefix count. It does not make a local-label probe direction orthogonal to every general counting direction. The intervention can still change a generic count axis, alter confidence, or create a state outside the natural recurrent-state distribution. A strong manipulation followed by a count effect supports a functional effect of this specified direction in this model and setting. It does not, by itself, establish that the network naturally uses the same local lemma, that the representation uniquely encodes mathematical relations, or that it composes reusable lemmas in ordinary inference.

The real recurrent-state implementation must state whether the cut is before or after the token revealing pi(i+1), patch the selected layer's carried hidden state, and continue the exact remaining token stream. Replacing one layer after a completed time step without recomputing the other layers is a well-defined intervention, but differs from recomputing a natural counterfactual prefix. Preserve this distinction and verify unedited continuation numerically.
