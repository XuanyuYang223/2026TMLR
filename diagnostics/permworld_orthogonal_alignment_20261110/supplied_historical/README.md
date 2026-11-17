# Historical peaks-to-valleys orthogonal alignment pilot

## Result

The historical PermWorld unit `archdev001` was tested across all 21 registered lengths. The primary columns below are equal-length averages of within-sample position AUROC on the 1,757 common count-matched test anchors.

| Readout or map fit | Trained one-line | Trained canonical cycle |
| --- | ---: | ---: |
| Direct frozen peaks probe | 0.50553 | 0.52527 |
| Means only | 0.50725 | 0.53182 |
| Complement Procrustes | 0.64392 | 0.54536 |
| Identity Procrustes | 0.42412 | 0.45703 |
| Inverse Procrustes | 0.50488 | 0.50161 |
| Shuffled complement Procrustes, mean of eight | 0.48714 | 0.52630 |
| Within peaks | 0.74232 | 0.55334 |
| Within valleys | 0.72408 | 0.56423 |

Each of the first six rows scores the same held-out complement input in the valleys model against the same source peak labels. The three paired maps differ in the validation transform used to fit the rotation. At step zero, complement alignment gives 0.50134 for one-line and 0.50898 for canonical cycle. On the full 10,752 eligible test anchors, trained one-line complement alignment gives 0.63816, compared with 0.50369 for the direct probe and 0.52121 for the oracle-count position prior. One-line matched gains over the direct probe are positive at 21/21 lengths, with range +0.04466 to +0.24532. Lengths are dependent conditions within this historical unit.

The one-line shuffle range is 0.46971 to 0.51181 on matched test anchors. Independently mapping the descents and recoils specialists, both with complement inputs, gives 0.52271 and 0.50180. These are two same-input task controls. The largest observed improvement is the recovery of out-of-sample linear position readout from the trained valleys representation in a learned orthogonal coordinate system. This is a historical-unit result about decodability. Generalization across units and causal use of the decoded positions require separate evidence.

## Method

- Use the original fixed split, source-only ridge fitting and alpha selection, same scaler, same scoring, and the same common matched support as the published direct-probe experiment. The direct scores reproduce the historical pilot exactly.
- Fit one target-to-source orthogonal map per phase, encoding and validation pairing. Within each length, centre both sides on their probe-fit validation means. Average the cross-products equally across lengths and take the orthogonal polar factor by SVD. The map uses 4,295 validation feature pairs, with the original probe-fit anchor set. The fit has no position labels.
- Apply the frozen peak probe to held-out target features mapped into its source coordinates. For the common complement test, all maps use the same target complement mean and source mean from validation. The means-only control isolates the intercept shift.
- The eight shuffled maps independently scramble the complement pairing within length and true source peak count on the fit set. At least 99.72% of pairs move. Keep results under the map's native evaluation transform and the corresponding own-valley labels as auxiliary rows.
- Reuse the historical source/test activations. For alignment fitting, extract the missing validation transforms from the released, hash-verified fixed checkpoints. The supplemental controls use two additional validation forwards for one-line descents and recoils. Only alignment maps and probes are fitted.
- Verify map orientation on a known synthetic orthogonal change of basis, orthogonality for every fitted map, anchor/index and checkpoint identities, exact recovery of all original direct baselines, and 21 finite length means for each comparison.

## Files and reproduction

The scripts use `analyze.py`, `infer.py`, and the pinned source definitions from the earlier historical probe package. Set `BASE` to that package's root, containing `code/`, the restored `probe_inputs/`, and the inferred `probe_features/`. Run from this package's root with the same Python dependencies as the historical probe.

```bash
export PYTHONPATH="$BASE/code"
for encoding in one_line canonical_cycle; do
  python code/extract_validation.py --inputs "$BASE/probe_inputs" --original-code "$BASE/code" --out validation --encoding "$encoding"
done
python code/extract_validation.py --inputs "$BASE/probe_inputs" --original-code "$BASE/code" --out validation --encoding one_line --controls
python code/alignment.py --data "$BASE/probe_inputs/data.npz" --features "$BASE/probe_features" --validation validation --out reproduced
python code/other_specialists.py --data "$BASE/probe_inputs/data.npz" --features "$BASE/probe_features" --validation validation --out reproduced
```

The newer four units are outside this pilot. Their validation transforms are already present in the original repetition caches under `prefix_final`, so the same alignment can be fitted from cached arrays on the machine holding those payloads.

- [Historical direct-probe protocol and repeated direct scores](https://github.com/XuanyuYang223/2026TMLR/tree/6d745815c0a1a2a41ad5fc8c0aa6e018f9bab249/diagnostics/permworld_peaks_probe_20261109)
- [Historical model release](https://github.com/XuanyuYang223/2026TMLR/releases/tag/representation-selectivity-2026-10-11-caveat)
