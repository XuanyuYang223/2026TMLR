# Data and mathematical review of the downloaded position probe

No blocking mathematical or split error was found for the full saved-cache layout. This audit reads existing data and source code only; it does not fit probes or score activations.

All **588** data-label checks passed. The actual fit, tuning and test orbit sets are disjoint from the specialist training orbits and from one another. Complement swaps peaks and valleys at exactly the same interior one-line indices.

| Unit | Probe fit | Probe tuning | Test | Count-matched | Matched range per length |
|---|---:|---:|---:|---:|---:|
| unit02 | 4272 | 1104 | 10752 | 1707 | 55–116 |
| unit03 | 4320 | 1056 | 10752 | 1675 | 55–120 |
| unit04 | 4274 | 1102 | 10752 | 1753 | 65–113 |
| unit05 | 4283 | 1093 | 10752 | 1755 | 63–107 |

Every unit has all 21 lengths, with 256 validation and 512 test rows per length; no orbit exclusions were necessary on these inputs. Every count-matched row has both positive and negative target positions, so its within-example AUROC is defined. Data-array shapes, per-length counts and source SHA-256 are in the companion JSON. Feature payload or checkpoint provenance is outside this audit.

## Implementation review

The scaler is fitted on source probe-fit rows. Ridge alpha is selected from the fixed four values on source tuning rows only. The resulting source scaling and weights are reused without a target-space map. The AUROC formula is the standard average-tie rank statistic, and ordinary randomized ranking has expectation 0.5.

## Required interpretation boundaries

- The summary uses pandas mean, which silently skips undefined length AUROCs. Require all 21 finite lengths before accepting each equal-length primary summary; do not reweight silently.
- The length/count/position prior conditions on the true test answer count. It is an oracle-count diagnostic, not a deployable length-only baseline.
- Matched support is selected using source peak count and target valley counts under all three actions. It defines a restricted, outcome-conditioned estimand and does not remove all answer or position-frequency effects.
- Matched support is unit-specific: historical 1757 anchors must not be reused as IDs for independent worlds. Keep the same unit-specific anchors across both encodings and all actions.
- The same source probe transfers across targets/actions within each phase/length/encoding. Initial and trained phases refit and retune separate probes; their difference is not a fixed-readout intervention.
- Outputs are positions in the original one-line permutation, excluding endpoints, not token positions in the cycle string. Complement preserves these indices; inverse does not generally align them.
- Direct transfer is coordinate-basis dependent; weak transfer does not imply information is absent or CKA is wrong. Positive transfer is not evidence that the counting mechanism uses these position features.
- The 21 lengths are dependent conditions, not 21 independent repetitions. The four new units are the new replication units; report their raw effects and ranges.
- Compute primary unit C-minus-best-alternative as mean_n(C_n) minus max(mean_n(I_n),mean_n(V_n)). This differs from mean_n(C_n-max(I_n,V_n)); retain the separately requested per-length dominance counts.
- Non-designated descents/recoils controls receive complement-transformed inputs in code (test_1_hidden). Do not describe them as identity-input controls.
- The vendor verify.py hardcodes historical count-accuracy references, so it cannot be applied unchanged as a new-unit acceptance test.

## Recommendation

Run the vendor protocol unchanged after the separate feature/checkpoint provenance check. Report each new unit’s complement minus identity, complement minus inverse, and complement minus initial-complement; retain original and own-valley labels, both supports, both encodings, all lengths and the selected alphas. Require all 21 finite lengths, preserve all negative or mixed results, and keep the historical pilot separate from the four newer units.
