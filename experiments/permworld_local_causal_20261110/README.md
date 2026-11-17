# Local recurrent-state causal diagnostic

This first-stage experiment uses existing one-line GRU peaks/valleys count
specialists. It validates exact counterfactual pairs, learns independent local
readouts on actual recurrent states, freezes a historical geometry-based
layer/time/dose, and evaluates thirteen within-model intervention conditions.
Counting specialists remain frozen throughout.

[Protocol, outcomes and reproduction](../../diagnostics/permworld_local_causal_20261110/README.md)

`pairs.py` is an independent mathematical constructor/oracle. `recurrent.py`
implements actual GRU state carry and chunked continuation. `features.py`
captures raw states and native/no-op logits. `probes.py` fits component-separated
edit/checker/count readouts. `calibrate.py` selects geometry from historical
development only. `formal.py` runs the fixed conditions and `run.py` guards both
freezes. No cross-model exchange or new combination task is part of this run.
