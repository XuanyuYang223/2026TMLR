# Timing experiment: main edit and whole-layer diagnostics

This is an exploratory follow-up on the same five existing model/data units; their prior late-edit outcomes had been viewed. Historical results are engineering reference only. Four-unit summaries below retain both tasks, donor constructions and signs separately. Timings, positions and control directions are dependent comparisons within a unit.

## One-dimensional local edit: relative modal changes and oracle answers

Relative success means the edited numeric argmax differs from the recipient numeric argmax by exactly the requested sign. Oracle-answer success means the edited numeric answer equals the true recipient count plus that sign. These differ when the baseline is wrong. Both are reported; nonnumeric and other-count changes remain explicit in the CSVs.

|Task|Group|Sign|Late relative step|Early relative step|Late oracle answer|Early oracle answer|Early−late relative|DiD vs random|DiD vs same label|
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
|peaks|main|-1|0.00%|0.00%|0.00%|0.00%|0.00%|0.00%|0.00%|
|peaks|main|1|0.00%|0.00%|0.00%|0.00%|0.00%|0.00%|0.00%|
|peaks|matched|-1|0.00%|0.00%|0.00%|0.00%|0.00%|0.00%|0.00%|
|peaks|matched|1|0.00%|0.00%|0.00%|0.00%|0.00%|0.00%|0.00%|
|valleys|main|-1|0.00%|0.00%|0.01%|0.01%|0.00%|0.00%|0.00%|
|valleys|main|1|0.00%|0.00%|0.00%|0.00%|0.00%|0.00%|0.00%|
|valleys|matched|-1|0.00%|0.00%|0.01%|0.01%|0.00%|0.00%|0.00%|
|valleys|matched|1|0.00%|0.00%|0.00%|0.00%|0.00%|0.00%|0.00%|

The injected first-layer state, local checker and one-dimensional dose are identical across early and late. Early replays upper current-token states; it can therefore change total carried-state norm even with identical injected L1 norm. Timing and DiD contrasts are formed within pair before equal-center and fixed1/21-length aggregation; random controls are averaged within pair, never maximized.

## Diagnostic 1: late replacement of each complete layer state

Each payload below is a complete 256-coordinate recurrent state. These replacements are distinct from the one-dimensional local edit and can transport history as well as local information. Columns show relative-step / oracle-answer success. They cannot identify a uniquely necessary counter layer.

|Task|Group|Sign|Whole L1 late|Whole L2 late|Whole L3 late|Whole L4 late|
|---|---|---:|---:|---:|---:|---:|
|peaks|main|-1|99.62% / 99.62%|1.40% / 1.40%|0.00% / 0.00%|0.00% / 0.00%|
|peaks|main|1|98.27% / 98.27%|0.22% / 0.22%|0.00% / 0.00%|0.00% / 0.00%|
|peaks|matched|-1|99.55% / 99.55%|0.64% / 0.64%|0.00% / 0.00%|0.00% / 0.00%|
|peaks|matched|1|98.49% / 98.49%|0.18% / 0.18%|0.00% / 0.00%|0.00% / 0.00%|
|valleys|main|-1|84.67% / 84.67%|2.04% / 2.05%|0.00% / 0.01%|0.01% / 0.01%|
|valleys|main|1|95.70% / 95.69%|4.35% / 4.34%|0.01% / 0.00%|0.01% / 0.00%|
|valleys|matched|-1|85.07% / 85.07%|1.74% / 1.75%|0.00% / 0.00%|0.00% / 0.00%|
|valleys|matched|1|96.01% / 96.00%|3.56% / 3.55%|0.00% / 0.00%|0.01% / 0.00%|

## Diagnostic 2: complete L1 replacement before upper replay

Early complete-L1 replacement and late complete-L1 replacement inject byte-identical L1 states; their upper-state handling differs. The paired contrast below uses the same interface and pairs. A whole-state effect does not show that the chosen one-dimensional feature reproduces it.

|Task|Group|Sign|Whole L1 late relative|Whole L1 early relative|Paired early−late relative|Whole L1 early oracle answer|
|---|---|---:|---:|---:|---:|---:|
|peaks|main|-1|99.62%|100.00%|0.38%|100.00%|
|peaks|main|1|98.27%|99.74%|1.47%|99.74%|
|peaks|matched|-1|99.55%|100.00%|0.45%|100.00%|
|peaks|matched|1|98.49%|99.85%|1.36%|99.85%|
|valleys|main|-1|84.67%|99.99%|15.32%|99.99%|
|valleys|main|1|95.70%|99.98%|4.28%|99.99%|
|valleys|matched|-1|85.07%|99.99%|14.92%|99.99%|
|valleys|matched|1|96.01%|99.99%|3.98%|99.97%|

## Physical positive controls and local desired answer

The full-four-layer transplant has the donor whole-input oracle; the real local-only counterfactual has recipient+sign as its oracle. In the matched group those donor and local-only endpoints differ. Mixed single-layer states have no natural full-input oracle and are evaluated against the stated local target separately.

|Task|Group|Sign|Full-four-state donor oracle|Full-four-state local goal|Real counterfactual oracle|
|---|---|---:|---:|---:|---:|
|peaks|main|-1|100.00%|100.00%|100.00%|
|peaks|main|1|100.00%|100.00%|100.00%|
|peaks|matched|-1|99.99%|0.01%|100.00%|
|peaks|matched|1|100.00%|0.00%|99.99%|
|valleys|main|-1|99.99%|99.99%|99.99%|
|valleys|main|1|99.99%|99.99%|99.99%|
|valleys|matched|-1|99.99%|0.01%|99.99%|
|valleys|matched|1|99.99%|0.00%|99.98%|

All means, sample SD, minima, maxima and four raw unit values are retained in `main_eight_cells.csv`, `auxiliary_eight_cells.csv`, and `positive_controls_eight_cells.csv`. The corresponding `*_by_unit.csv` files show individual formal units; historical files are separate. The original completed frozen summary retains other-change rates, numeric mass/coverage, signed count expectations, checker values, common-center/baseline-correct diagnostics, available-cell sensitivities and pooled sensitivities.

The input construction couples incoming token rank to the local label, and these are mixed-state interventions. Results can support the tested current interlayer route, not precise natural accumulation timing, an abstract predicate-specific axis, or a naturally reused one-dimensional lemma. No new units, models, readouts, doses, conditions, fits or forwards are introduced by this reader.
