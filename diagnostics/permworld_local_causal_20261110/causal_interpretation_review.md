# Causal intervention design and interpretation review

The implemented design matches the frozen within-model protocol. No blocking source discrepancy was found. This review used source, historical local-geometry selection arithmetic and saved raw pair metadata; it did not inspect formal model outcomes or run fits or forwards.

Formal protocol SHA256: `016ef16f07924f8bf55a232d21754612afa15dd2655cd4ac9d0425ff3657b176`. Engineering and formal frozen source hashes match.

## What is verified

- The selector uses the historical unit only and chooses by independent-checker local geometry. Independent recalculation gives a common layer1, AFTER the right value, dose1. It never selects on the model’s final count changes.
- Editing and checking readers use disjoint FIT_A/FIT_B components, only from count-matched fitting examples. Dev/test do not enter these fits. No identical AFTER prefix overlaps either fitted half with the other half, dev or test.
- A few distinct dev/test inputs have identical observable AFTER prefixes (10,4,10,2,8 across units). This is documented without changing the frozen full-permutation/component split. Formal-unit dev is not used to choose the layer.
- The actual GRU recurrent states are patched and resumed at correct absolute positions. LayerNorm outputs are not fed back. The other three layers retain their current recipient states.
- The matched donor keeps the true total unchanged; the local-only counterfactual changes it by the intended sign. The code keeps these endpoints separate. Native/no-op and full-state donor checks are specified, but were not rerun in this review.

## What the manipulation check means

At dose1, replacing the projection onto the editing probe’s normalized weight forces that same probe to return the donor score. This equality is algebraic. The separately fitted FIT_B checker on held-out components is the substantive manipulation check; it is still a probe-based semantic check, not proof that the network naturally uses an isolated mathematical variable.
Matched true counts do not force the editing vector to avoid count-sensitive directions. A linear count reader with weight c changes by (c·v)((h_d−h_r)·v). The diagnostic count reader cannot exclude all nonlinear or distributed count information.
The eight random-direction edits are norm matched. Same-label edits use naturally differing projection magnitudes and are not norm matched. Compare these controls with that distinction and retain the earlier-only versus full-read-prefix annotation.

## Causal and timing limits

A null effect after the current update cannot show that the local fact was unused earlier. The selected layer1 has already passed its current-token output to upper layers, whose recipient states are retained. Information may also be redundant, distributed or represented in other directions. A positive change establishes sensitivity to the defined operational patch; it does not by itself identify a unique natural computation.
The ascending-quartet construction also changes local token/value ranks. In particular, the last value read at the cut changes from b to a when a peak is created. Count matching removes the direct true-count difference but does not eliminate those proxies or off-manifold effects. Same-label full-prefix controls preserve the prescribed label and counts, while other extrema locations may change.

## Scope and stop rule

Directions are separately fitted within each peaks/valleys checkpoint. This phase does not test a shared cross-model lemma or a new composed task. Keep conclusions conditional on constructed motifs, one-line GRUs, frozen20k checkpoints and the single historically selected AFTER projection. Complete this stage and stop regardless of its sign; cross-model work requires separate authorization.

The review JSON contains source hashes, split metadata and a fixed raw-permutation endpoint example. No formal outcome claim is made here.
