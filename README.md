# Mathematical correspondences and representation pairing selectivity

Do exact mathematical correspondences accompany geometric pairing preferences in trained neural networks, and how do those observations depend on input encoding, model configuration, layer, metric, initialization reference, and answer controls?

This branch contains the **PermWorld / GraphWorld / MatrixGroupWorld representation study**. It retains exploratory, negative, weak-learning, and undefined results. Geometry alone does not establish a mathematics-specific mechanism or functional benefit.

## Scope and current results

| Study | Fixed scope | Saved primary ordinary-CKA readout |
|---|---|---|
| PermWorld | 16 specialists, eight directed relations, one-line/canonical-cycle encoding × Transformer/GRU; 64 true 20k endpoints in one paired data world | Answer-stratified final-layer, per-length equal weighting: positive trained preference `Bt` in **26/32**, positive increment `Delta` in **18/32** |
| GraphWorld | Four specialists and four directed reversal/complement relations; one Transformer/configuration, n=6/7/8 equally weighted | Answer control: positive `Bt` in **2/4**, positive `Delta` in **0/4**; raw and answer-plus-exact-edge-count sensitivity results remain available |
| MatrixGroupWorld | GL(n,F2), n=6/8/10; four specialists; two independent data/true-initialization units; eight true 20k endpoints | Raw `Bt` positive **8/8**; joint-answer-controlled `Bt` and `Delta` positive **5/8** each |

The current configurations use a 188-token vocabulary, maximum sequence length 128, hidden width 256, four layers and dropout 0.1. The Transformer has eight heads and feed-forward ratio four; the GRU uses four recurrent cells with explicit inter-layer dropout. Their parameter capacities are not matched. GraphWorld and MatrixGroupWorld use the same declared Transformer configuration; each published primary endpoint has 20k updates. Exact configs and task learning scores remain in the linked protocols and tables.

These counts use different tasks, inputs, controls and dependencies. They are **not** pooled independent replications or estimates of a cross-domain causal effect. The new matrix units independently repeat only MatrixGroupWorld.

The primary matrix relationships are **R1 (transpose, column/row weight)** and **R3 (right multiplication, column/XOR-column weight)**; R2/R4 are supplementary. Under answer control, ordinary-CKA `Delta` is negative for R1 in both units, while R3 is +0.048498/+0.121332. R3's positive increment comes from a larger decline in the wrong-pairing score; the correct score also declines. F3 at n=10 remains weak in both units. The complete per-size train, validation and held-out source scores are retained.

MatrixGroupWorld is **`completed_with_verification_caveat`**, with `scientific_analysis_complete=false` in its original saved state. The prescribed independent saved-feature check records **64 passed, 2 failed, 6 undefined_consistent**. Both failures concern trained raw **output-probability kNN** on the fixed first-unit R1/n=6 subset of 64 inputs. Each difference is one neighbor count out of 640; a read-only diagnostic localizes near-onehot Gram-distance roundoff. The original failed values, fixed tolerance, numerical methods and failure status remain unchanged. This is not independent validation of all 432 production cells.

## Comparisons and definitions

For a fixed source task pair and relation, compare the correct transformed input with the original wrong transformed input:

```text
B0    = S_correct(initial) - S_wrong(initial)
Bt    = S_correct(trained) - S_wrong(trained)
dc    = S_correct(trained) - S_correct(initial)
du    = S_wrong(trained)   - S_wrong(initial)
Delta = Bt - B0 = dc - du
```

Keep all four original scores. Positive `Delta` need not mean positive `Bt`, or an increase in the correct score. Real step-0, reconstructed historical references and unavailable initial identities are explicitly distinguished.

The **fixed-transform task comparison** instead holds the left model, transform and anchor inputs fixed and varies the right specialist. It contains Perm480 / Graph12 / Matrix24 logical candidates, classified before their new scores as designated, other-known-exact, or not-designated. The last label does not mean mathematically unrelated. Its `fixed_transform_score_initial/trained`, candidate ranks and category contrasts are separate from `Bt/Delta`. Shared models, inputs, tasks and transforms create dependencies.

The main state is taken at the input end **before the task query**, with final normalization. The original protocols specify within-size/length computation and fixed equal weights; undefined components are not set to zero and weights are not redistributed. Ordinary linear CKA is primary. Fixed kNN overlap and held-out Procrustes, other layers, pooling, output-probability and answer-geometry baselines are reported only in their original prescribed scopes. Joint-answer controls use pair-specific retained strata and support, so their between-task contrast is not a pure causal effect of mathematical correspondence.

GL inputs add finite-field linear mixing. Transpose is an anti-automorphism; left/right multiplication by a fixed invertible element is an action, not a group homomorphism. These weight identities also hold for general binary matrices and are related to graph-degree statistics. They do not demonstrate learned group multiplication or a group-specific mechanism.

## Reports, tables and provenance

- [Current research index](CURRENT_RESEARCH.txt) and [scope/formula notes](docs/representation_research.md).
- [PermWorld eight-relation report](diagnostics/permworld_eight_relations_20261010/report.html), [combined Perm/Graph reader](diagnostics/permworld_eight_relations_20261010/final_review/combined_report.html), and [32 primary cells](diagnostics/permworld_eight_relations_20261010/final_review/permworld/main32_final_cka_readout.csv).
- [GraphWorld report](diagnostics/graphworld_pilot_20261010/report.html), [frozen protocol](diagnostics/graphworld_pilot_20261010/protocol.json), and [full geometry](diagnostics/graphworld_pilot_20261010/geometry_summary.csv).
- [Fixed-transform review](diagnostics/representation_pairing_review_20261011/reader_report.html), [protocol](diagnostics/representation_pairing_review_20261011/protocol.json), candidate tables and wrong-formula preservation diagnostics.
- [MatrixGroupWorld report](diagnostics/matrixgroupworld_20261011/report.html), [short bilingual results](diagnostics/matrixgroupworld_20261011/report.txt), [protocol](diagnostics/matrixgroupworld_20261011/protocol.json), [full geometry](diagnostics/matrixgroupworld_20261011/geometry_summary.csv), [source learning](diagnostics/matrixgroupworld_20261011/source_learning.csv), [original numerical verification](diagnostics/matrixgroupworld_20261011/actual_numerical_verification.json), and [failure diagnosis](diagnostics/matrixgroupworld_20261011/knn_failure_diagnostic.txt).
- Directly supporting older initialization, layer, encoding and learnability diagnostics remain under `diagnostics/`; the curation inventory records why each retained source/record belongs here.

Frozen scientific sources, tables, protocols, reports and failure records retain their original bytes. Publication metadata and portable wrappers are separate; original absolute paths remain provenance and are mapped explicitly rather than silently edited.

## Minimal reproduction: no training or model forward

Use Python 3.11 or newer for the standard-library table replay. A fresh clone needs no local F-drive paths, upstream checkout, CUDA, weights or hidden caches to check the saved aggregates:

```bash
git clone https://github.com/XuanyuYang223/2026TMLR.git
cd 2026TMLR
python -m venv .venv
source .venv/bin/activate
python -B tools/recompute_tables.py --self-test
python -B tools/recompute_tables.py --repo-root . --output /tmp/representation-table-check.json
```

The exact replay command, download profiles and published verification record are specified in [reproduction instructions](docs/representation_reproduction.md). The replay independently aggregates Perm/Graph fixed-transform tables and Matrix R1/R3 decompositions and checks the preserved 64/2/6 caveat. This checks saved-table arithmetic and integrity, **not** all model forwards, all activation metrics, training reproducibility, or independent statistical replication.

The science environment records Python 3.13.13, NumPy 2.4.4, PyTorch 2.11.0+cu128, Matplotlib 3.11.2 and threadpoolctl 3.6.0. Full-source dependencies and the pinned upstream snapshot are retained separately from the minimal table checker. CI performs only lightweight saved-result checks; it never starts training.

## Data, weights and caches

Related original data, genuine step-0/RNG, final weights, arrays and essential logs are published through the [independent representation-selectivity artifact prerelease](https://github.com/XuanyuYang223/2026TMLR/releases/tag/representation-selectivity-2026-10-11-caveat) and [artifacts manifest](artifacts_manifest.json). Every logical file maps to a content SHA, a release asset, download URL and extraction path. Packages contain real file bytes, not links to a local disk. SHA-identical contents are stored once; all logical aliases remain recorded. No missing historical initial identity is replaced with a newly trained or unverified model.

[Download/restore instructions](docs/representation_reproduction.md) distinguish full publication, the smaller clean-clone download profile, per-file local checks and the exact remote bytes actually downloaded again. A successful upload or checksum does not resolve a scientific verification failure.

## Previous independent work and licensing

The independent F17 composition/null-space/GELU manuscript and its dedicated experiments have exited this branch's current tree. Their local originals, original history, tags and releases remain intact; the [pre-curation source tree](https://github.com/XuanyuYang223/2026TMLR/tree/e4e9ded9648aa2dc92a1cceef929cc9521546418) is still accessible. The preservation ref and explicit Git-only removals are recorded in the curation audit. Shared historical modules are retained only where this study depends on their exact bytes.

The upstream PermWorld snapshot comes from `XuanyuYang223/neurips` at `74f0de2017115f06f11b7285366b66e7e8431d30`. Its original attribution and any provided license files are retained. The inspected upstream snapshot did not supply a LICENSE/COPYING/NOTICE or package license declaration; this publication does not invent or relicense upstream terms. See [upstream provenance](docs/upstream_provenance.json).

Repository curation, release publication and byte integrity are engineering checks. They do not prove mathematical representation mechanisms. This task ends after publication and clean-clone verification; it does not launch more training or another world.
