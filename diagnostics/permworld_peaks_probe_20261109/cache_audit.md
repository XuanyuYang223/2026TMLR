# Probe cache and checkpoint audit

All 64 feature records needed by the unchanged cache importer exist for units02--05. Each unit has 5,376 validation and 10,752 test anchors, three actions, 256-dimensional FP32 states, and the original input-end final-normalized point. Every metadata binding matches the frozen source protocol, dataset, actual source checkpoint, and parameter-before/after receipts.

The current streaming audit checked 36 physical checkpoint payloads (32 trained specialists and four genuine step-0 snapshots), totalling 665,798,484 bytes. All SHA-256 values matched. Every trained receipt records the same actual loaded initialization within its unit across the two encodings and four relevant tasks. This verification uses current file SHA and original execution receipts; it does not newly reconstruct or reload tensor initializations. No initialization is inferred merely from a seed.

Feature payload hashing is deliberately left to the original importer on actual use; the audit checks their metadata without duplicating those reads. The importer alone does not bind the physical checkpoints or frozen protocol SHA, so this audit supplies those checks.

The historical pilot can also be reproduced from saved arrays without a GRU forward. Its data is `/mnt/yangx-projects-f/20261009/permworld_new_artifacts/encoding_archdev/worlds/archdev001/data.npz`; its validation/test array index is `/mnt/yangx-projects-f/20261010/permworld_eight_relations/evaluation/index.json`. All 32 relevant index records bind to the downloaded pilot's exact checkpoint/data identities. The JSON records the precise prefix/prediction paths and their expected hashes. Downloaded `infer.py` defaults to CPU FP32, so new extraction and the original saved FP32 arrays are not assumed bit-identical; measure any numerical reproduction discrepancy.

No model construction, forward pass, probe fitting, optimizer, source edit, or upload was performed by this audit.

The historical source verification additionally streamed all nine physical GRU checkpoints (eight trained, one genuine initialization) and the old dataset against the downloaded source inventory. All matched. Current-task checkpoint coverage is therefore **45/45**, with no forward pass.
