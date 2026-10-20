PermWorld: validation-driven learnability development, 2026-10-10

This round is completed and stopped.16/16 models completed fixed 8k updates and final validation; GPU-stage
actual wall clock1703.47 seconds (28.39 minutes including engineering/validation/saving), below the2h limit.
6/16 models passed prespecified learning gates and0/8 encoding/task pairs passed joint gates; thus no
joint sufficiently learned setting was found, and no new geometry confirmation, independent confirmation training, or GroupWorld occurred.
Brief English entry: completion_reader_report.html / .txt.
Complete report: learnability_report.html / .txt; per-length, curve, and gate CSVs and
8 fixed figures are saved.407 protected historical hashes agree; F17 was not modified.
completion_audit.json records source/genuine-initialization/checkpoint/budget/stopping audits;
independent_learning_audit.json independently recomputes reports, without new model forwards.

Data generation, CPU checks, CUDA restoration, development training, and report commands have all run;
do not restart develop to read results. Read-only report rebuilding may explicitly use --stage report.

This round follows completed multilayer analysis and the16-model development pilot. New work first inspects existing
training curves, per-length answer accuracy, and training length modes, then runs one prespecified
shorter-window development scheme. Historical CKA/Delta does not select configuration, length, or checkpoint.

Historical diagnostic entry: old_learning_diagnosis.txt and corresponding CSV/plots/audit artifacts.
Code reviews: code_readiness_review.txt, independent_code_review.txt.
Independent cost recommendations: protocol_review.txt; recommendations are not the final protocol; protocol.json
is the frozen scheme (e.g.,1024 validation inputs/length and5pp mean encoding-gap bound).

Single development scheme: n10..12, descents/recoils/peaks/valleys, both encodings, Transformer/GRU.
Small models remain width256/four layers (Transformer8 heads). Fresh data, genuinely saved step0; tasks
and encodings share initial parameters within architecture.8k updates, batch64, microbatch64, LR1e-3,
warmup400, dropout0, with remaining AdamW settings unchanged. Multiple training factors change together,
so no improvement can be attributed to one factor alone.

Learning qualification frozen before training: mean answer accuracy ≥90%, each length≥85%, gain over
the training-label-defined length mode≥20pp, EOS≥99%, and within-task/architecture encoding
gaps≤5pp for both mean and maximum per-length accuracy. Partial2k/4k does not replace fixed 8k final qualification.
These are development gates, not statistical-equivalence tests or evidence of independent mathematical mechanisms.

First cover all16 conditions at2k, then fixed4k/8k; never drop tasks or select metrics based on effects.
GPU-stage total wall-clock limit7200 seconds, with final30 seconds reserved for cleanup. All provenance, initialization,
protocol, source version, actual steps, sample exposure, and time are retained. Engineering checks include CPU known
logit/loss/causal independent validation and genuine CUDA continuous32 versus16+32 restoration in independent processes.
No updates after cutoff; states missing final validation are not described as complete models.

Training/validation use identity-task samples only; the learning driver does not load reserved test arrays,
evaluate tests, or compute CKA. Generation/split audits inspect all-split orbits for
data identity/isolation, not test-based selection. Exclude eight-element orbits containing any candidate or previous-round train/validation/
test inputs; both encodings share underlying objects, labels, and batches.

Large artifacts: /mnt/yangx-projects-f/20261010/permworld_learnability/
Code: experiments/permworld_learnability_20261010/
This directory stores frozen protocol, source hashes, reports, CSVs, and figures. F17 and previous-round PermWorld
reports/data/code are protected read-only; see protected_legacy_before.json.

Stages are explicit commands; do not run develop just to read reports:
  .venv/bin/python -m experiments.permworld_learnability_20261010 --stage setup
  .venv/bin/python -m experiments.permworld_learnability_20261010 --stage prepare
  .venv/bin/python -m experiments.permworld_learnability_20261010 --stage cpu-check
  .venv/bin/python -m experiments.permworld_learnability_20261010 --stage develop
  .venv/bin/python -m experiments.permworld_learnability_20261010 --stage report

Independent read-only validation:
  .venv/bin/python diagnostics/permworld_learnability_20261010/completion_audit.py
  .venv/bin/python diagnostics/permworld_learnability_20261010/independent_learning_audit.py --run-real
  .venv/bin/python diagnostics/permworld_learnability_20261010/completion_reader.py

If joint learning gates fail, preserve every negative result and stop; do not replace with new geometry results.
If gates pass, first freeze an independent fresh-world/fresh-initialization validation manifest; development is not confirmation.
No GroupWorld, F17 edits, extra candidates, formal geometry training, or automatic scaling this round.
