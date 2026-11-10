# PermWorld independent repetition: original-unit audit

Audit date: 6 November 2026. Source commit: `4e3558ac13a635d780d1801978986fd195555177`.

The existing main PermWorld experiment contains **one complete exploratory unit**, `archdev001`. Its 64 real 20,000-update endpoints cover 16 tasks, two encodings, and two architectures. The current repetition target is **five units in total**, including that existing unit. Four additional units and their fixed seeds are implemented in the new runner; this document does **not** report those four units as complete.

The current run state must be read from [execution_status.json](execution_status.json) when available. The audited counts below describe the original unit, not a five-unit summary.

## What the audit verified

- All 64 recorded endpoint files and their completion records exist, and their bytes match the checkpoint inventory SHA256 values. Step counts, finite gradient-update counts, and saved initial-parameter bindings also match: **320 checks passed**.
- The two saved real step-0 state dictionaries were loaded without forward passes. Their parameter hashes were independently recomputed and match the declared hashes. The GRU's recorded value-preserving key conversion remains identified separately.
- The original comparison A counts were independently recomputed from final-normalized hidden-state ordinary-CKA **per-length** rows and the fixed `1/21` weights.
- The comparison B counts were independently recomputed from all fifteen candidate-task scores in the saved summary, rather than copied from the manuscript's count table.
- The audit wrote only its external audit directory. It performed no training, optimizer calls, new sampling, or model forward pass. Checking saved bytes and arithmetic does not reproduce the historical training trajectories or independently recompute every CKA from activations.

Machine-readable evidence: [original_audit.json](original_audit.json), [comparison A recomputation](original_comparison_A_recomputed.csv), and [comparison B recomputation](original_comparison_B_recomputed.csv). Each CSV contains 64 rows: 32 conditions under each of two controls. These are dependent measurements from one unit.

## Recovered original-unit results

| Main control | A: positive trained preference `Bt` | A: positive training increment `Delta` | B: designated above other-task mean | B: designated uniquely first |
|---|---:|---:|---:|---:|
| Raw, within length | 27/32 | 25/32 | 30/32 | 25/32 |
| Joint-answer strata | 26/32 | 18/32 | 29/32 | 19/32 |

All entries use the original final normalized hidden state, ordinary linear CKA, and equal weights over lengths 10 through 30. Comparison A fixes the two tasks and changes the input pairing. Comparison B fixes the left task and input transformation and varies the right specialist. Its mean contains fourteen non-designated candidates; its ranking includes the designated candidate. Rank is `1 + count(other_score > designated_score)`, and unique first additionally requires zero exact ties. No new tie tolerance or practical-effect threshold is introduced.

### Why comparison B has a raw zero reference

Within each architecture, all tasks and both encodings copied the same saved numerical initial parameters. Features are measured in evaluation mode at the input end **before the task token**. With the same encoding and transformed anchors, all right specialists therefore have identical raw initial features and identical raw comparison B scores, regardless of the task they will later learn.

The saved raw scores are exactly identical across all fifteen candidates in each of the 32 conditions. Subtracting a floating-point mean of fourteen identical scores can leave roundoff: the independently recomputed raw contrasts range from `0` to `1.3877787807814457e-17`. This is consistent with the mathematical zero reference.

Candidate-specific answer stratification changes the residualization and sometimes the retained support. Its initial contrasts need not be zero; the recovered range is `[-0.24772915346919538, 0.09350747919383118]`. The raw zero statement must not be applied to this controlled readout or to task-conditioned outputs. Comparison A likewise need not start at zero, because its correct and wrong inputs differ even when weights are shared.

## Randomness and sharing recovered from implementation

| Factor | Original unit | Intentional sharing |
|---|---|---|
| Data generation | `95200001` | One sequential generator for train, validation, and test; all tasks, encodings, and architectures use the same underlying objects and splits. |
| Batch generation | `95300001` | Same saved length schedule and batch indices across all specialists. |
| Transformer initialization | `95100001` | One real step-0 state copied across its 16 tasks and both encodings. |
| GRU initialization | `95100002` | One real step-0 state copied across its 16 tasks and both encodings; no cross-architecture parameter identity. |
| Training streams for four original tasks | `95400000 + 1000 * architecture_index + task_index` | Architecture order is Transformer, GRU; task order is descents, recoils, peaks, valleys. Each task's two encodings share the saved start RNG. |
| Training streams for twelve added tasks | Same architecture's saved descents stream | The twelve added tasks and descents are coupled, rather than having sixteen independent task streams. |
| Evaluation | No separate sampling seed | Evaluation anchors come from the sequential data generator. Feature extraction is deterministic evaluation, with dropout disabled. |

The shared Transformer initial checkpoint has parameter hash `03dee240f607ea45d5ec6d09ebe2121452ba7378dea4f262ec9925d91f2c8dab`. The shared GRU initial checkpoint has parameter hash `2568285183400a3266e1338d3a169255ebe10df68c3ca1fb9b725663e7870ca0`. Full checkpoint, start-RNG, data, batch, code, and table SHA256 values are recorded in the JSON audit.

### Original data mechanism retained for repetitions

Each length from 10 through 30 has 4,096 train, 256 validation, and 512 test anchors: 86,016 / 5,376 / 10,752 in total. The same original rejection sampler accepts full eight-member transformation orbits, excludes the fixed historical input index, and prevents within-unit orbit overlap across all splits. Acceptance does not balance answers or choose examples by learning outcomes.

The historical index contains **18,260,223 hashes** and has SHA256 `fc657f8b3ffec8d0a6decab752eac49688e968d947ac4002d03138e0b290e8be`. It is fixed across the original and additional units. The old unit and earlier new units are not progressively added to that index. Accidental cross-unit overlaps are permitted by the original mechanism, and cross-unit model pairs are not main endpoints. This tests independent sampling conditional on a fixed mathematical universe and exclusion rule, not new task definitions or an independent mathematical domain.

## Fixed additional seed grid

The seed grid is read from the new [runner](../../experiments/permworld_repetition_20261106/runner.py). Its protocol copies the original model, optimizer, data scale, training budget, and evaluation definitions. Each row requires 64 fixed-20k endpoints and their analyses; engineering checks do not count as endpoints.

| Unit | Data seed | Batch seed | Transformer initial seed | GRU initial seed | Training-stream base |
|---|---:|---:|---:|---:|---:|
| `unit02` | 96200002 | 96300002 | 96100004 | 96100005 | 96420000 |
| `unit03` | 96200003 | 96300003 | 96100006 | 96100007 | 96430000 |
| `unit04` | 96200004 | 96300004 | 96100008 | 96100009 | 96440000 |
| `unit05` | 96200005 | 96300005 | 96100010 | 96100011 | 96450000 |

Training-stream offsets retain the original four-task / twelve-task sharing pattern above. Within each architecture and unit, parameter hashes must establish actual equality of the starts. Between units, these seeds jointly change data, initial weights, batch order, and training randomness; they do not isolate the contribution of each factor. Evaluation records explicitly mark the absence of a separate random evaluation sampler.

All fixed endpoints and failures must remain represented. A technical interruption can resume its matching checkpoint, optimizer, and RNG. Weak learning is a result and does not authorize a replacement seed, budget change, or exclusion. The original unit remains exploratory; freezing the new protocol after viewing the original results does not retroactively make it a prospective confirmation experiment.

## Reproducing this audit

The copied [audit script](original_audit_run.py) reads repository tables and the existing large-artifact paths recorded in the original inventory. It requires those original files to be available locally; it does not download substitutes or recreate missing models. The table counts can be inspected in the two delivered CSVs without those large files.

From the repository root:

```bash
.venv/bin/python diagnostics/permworld_repetition_20261106/original_audit_run.py \
  --repo-root . \
  --output /mnt/yangx-projects-f/20261106/permworld_repetition/reaudit
```

The command writes a new audit and recomputed CSVs to the specified output directory. It does not mutate the frozen original diagnostics. The machine-readable audit is a record of the original audit time and its exact input hashes, not a claim that every repository file remained unchanged throughout the session.

## Finite independent-specialist initialization supplement: proposal only

**Not implemented or executed by this repetition task.** The five-unit design changes the shared initial weights between units while retaining common weights among specialists inside each unit. It therefore cannot establish robustness when the specialists being compared begin from different numerical parameters.

A minimal additional study could use one preselected architecture, two to four of the existing directed relationships, and both encodings. Two disjoint task pairs require four tasks and **eight new specialists per data world**; four pairs require eight tasks and **sixteen specialists per data world**. Pair choices should be fixed before fitting and include an inversion relation, a complement relation with a previously negative cycle-encoding result, and, if resources permit, the well-learned GRU exceedance/deficiency example. These choices are motivated by observed exploratory results and must be described that way.

Each task would receive an independently drawn initial state, with that task's two encodings still copying the same real step-0 parameters. Use data worlds already present in the shared-initialization repetition, retain their labels, batches, training-stream coupling and budgets, and compare against their matched shared-initialization models. This changes the dependence of initial parameters across tasks while retaining the marginal constructor distribution. Independent worlds, rather than task pairs or encodings, remain the replication units.

The smallest study can retain comparison A on the selected relationships, with each model pair's actual initial scores. A restricted comparison B must list its smaller, fixed candidate set and denominator. It cannot reuse the original fifteen-candidate interpretation or claim to reproduce `19/32`. Its raw initial task contrast must be measured rather than assumed zero. Recovering the full original comparison B would require all sixteen specialists per encoding and architecture, outside this minimal proposal.

| One-architecture option, per world | New specialists | Historical Transformer process-wall anchor | Historical GRU process-wall anchor |
|---|---:|---:|---:|
| Two disjoint relationships, both encodings | 8 | Approximately 1.70 hours | Approximately 2.93 hours |
| Four disjoint relationships, both encodings | 16 | Approximately 3.39 hours | Approximately 5.86 hours |

These are proportional estimates from the recorded original 32-model per-architecture totals, not new timing measurements or power guarantees. They exclude data preparation, feature extraction, analysis, verification, storage overhead, interruptions, and hardware differences. Two new worlds would double them. If both architectures are included, the model counts double and the corresponding architecture costs add.

For context, the original full 64-model unit recorded **18.495823733745272 process-wall hours** across historical runs. Four more full units would scale to about **73.98 process-wall hours** under that crude historical average. This is neither active GPU time nor an estimate validated on current hardware; current engineering measurements must be reported separately. The old sixteen models were historically slower than the later forty-eight, so a single mean is an imprecise planning anchor.

The supplement would test whether the selected trained pairing preferences and initialization-relative changes survive removal of cross-task parameter identity. It would not by itself identify a uniquely mathematical mechanism, demonstrate function, or establish universal robustness across all task pairs and architectures.
