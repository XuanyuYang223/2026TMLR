PermWorld encoding × architecture development only

Scope: descents/recoils/peaks/valleys × one_line/canonical_cycle × Transformer/GRU.
This is one exploratory engineering world, not the frozen formal24-unit plan.
256-wide four-layer architectures use common tokens and causal answer+EOS loss.
Transformer has8heads; GRU does not match the Transformer's parameter count.
GRU implementation revision: four single-layer GRUs with native dropout0 and
three explicit torch.Dropout(.1) modules between layers. This makes ordinary
checkpoint RNG sufficient for CUDA resume; the native multilayer cuDNN dropout
implementation failed the separate toy CUDA restart check. The exact evidence
and unchanged parameter-coordinate mapping appear in explicit_dropout_repair_log.
The original real GRU step0 is preserved as step0_before_explicit_dropout_fix.pt;
the current step0 is a bitwise value-preserving key migration, not a claimed new
constructor snapshot. Parameter count1660416, embedding dropout, final norm and
tied head are unchanged. Transformer initialization and the corrected data stay.
True initialization snapshots are shared only within an architecture.
Each paired encoding has identical underlying examples, batches and starting RNG.
Input prefixes are exactly2n+4 tokens in either encoding, including singleton cycles.

Output root (F drive):
  /mnt/yangx-projects-f/20261009/permworld_new_artifacts/encoding_archdev

No command automatically invokes the next stage. Run from /home/yangx/ICML with:
  PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.permworld_encoding_archdev_20261009 --stage setup
  PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.permworld_encoding_archdev_20261009 --stage cpu-check
  PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.permworld_encoding_archdev_20261009 --stage prepare

Explicit GPU engineering/training stages after review:
  PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.permworld_encoding_archdev_20261009 --stage overfit
  PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.permworld_encoding_archdev_20261009 --stage resume
  PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.permworld_encoding_archdev_20261009 --stage train
  PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.permworld_encoding_archdev_20261009 --stage evaluate
  PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.permworld_encoding_archdev_20261009 --stage report

Resume validation compares a genuinely uninterrupted single-process0->2000
reference against a separate0->1000 run restarted in another process to2000.
Interrupted continuous attempts are retained and a fresh attempt is used;
continuous references are never resumed. Its restart1000 checkpoint and split
directories are separate from full source outputs. Source-version hashes prevent
checkpoint/completion reuse across modified execution code.
Source fitting validates protocol, data, batch schedule, actual step0 and RNG.
Every source uses fixed20k updates, LR3e-4,64 examples/update and2 supervised tokens.
Weak learning is retained and reported rather than silently dropped or extended.

Geometry: final_norm before task token; per length10..30 with fixed1/21 weights.
CKA, k10-neighborhood overlap and heldout rank-cap8 PCA+Procrustes are fixed.
De-biased CKA is a sensitivity analysis; constants remain undefined.
Procrustes fits only validation inputs and evaluates orbit-disjoint test inputs.
Raw and exact joint-answer-strata controls use identical masks across conditions.
Report B0/Bt, correct/wrong changes, Delta, encoding Gamma components, conditional
numeric output-probability geometry, static true-answer geometry and task accuracy.
Static true-answer baseline has no fictional training delta.
Actual saved activations are checked by an explicit prefix-only hidden forward
for both architectures/encodings, initial/trained, n10/20/30 first8test objects.
The first relationship/length CKA is independently recomputed from covariance
matrices and independent strata centering, with all controls and output branches.
No confirmation p-values, learned-mechanism claim, formal plan launch, extra seeds,
GroupWorld, F17 intervention, or larger-model experiment follows this report.
