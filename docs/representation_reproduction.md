# Restoring artifacts and checking saved results

## Small saved-table replay

The checker uses only Python's standard library (Python 3.10 or newer). Run from the repository root:

```bash
python -B tools/recompute_tables.py --self-test
python -B tools/recompute_tables.py --repo-root . --output /tmp/representation-table-check.json
```

It opens 17 exact repository-relative published inputs, rejects absolute paths and external symlink fallbacks, and independently checks saved Perm480/Graph12 fixed-transform aggregation, designated matches, classifications, and Matrix R1/R3 decomposition. Matrix's original 64 passed / 2 failed / 6 undefined status and `scientific_analysis_complete=false` remain required. The output audit names the exact checked inputs and scope. A pass is table arithmetic/integrity, not CKA from every hidden array, model forward, training replay or new replication.

## Published original bytes

`artifacts_manifest.json` is the machine entry for the independent representation-selectivity release. Each logical original points to a unique content SHA, package, download URL and safe extraction path. Every package contains actual regular `objects/<sha>` files and `PACKAGE_INDEX.json`. SHA-identical payloads are stored once while all original aliases remain recorded. Package sources were verified against the before-publication ledger; source snapshots, protocols and numerical results were not rewritten.

Install the separate small decompression dependency:

```bash
python -m pip install -r requirements-artifacts.txt
python -B tools/download_artifacts.py --manifest artifacts_manifest.json --profile minimal --directory artifacts/downloads --restore-dir artifacts/restored --output /tmp/representation-download-check.json
```

The manifest specifies which packages the minimal profile contains and its actual bytes. For all published related originals use `--profile full`. A repeated download verifies existing files before reuse; different existing bytes are refused rather than overwritten. Every asset, concatenated compressed stream and restored object is checked for length and SHA. Archive members must be expected regular files with safe names; links and path traversal are rejected. These packages currently use single-part streams; the tool also tests and supports explicitly ordered multi-part reconstruction.

By default restored payloads are regular CAS files plus a plain `path_map.json`, not links to local disks. `--materialize` additionally copies the originals to the manifest's `restore_path` tree. This needs extra disk space for logical aliases; the tool reports the estimated space. Original absolute paths in frozen scientific files remain provenance. Resolve one source or data reference explicitly:

```bash
python -B tools/resolve_legacy_path.py /home/yangx/ICML/diagnostics/matrixgroupworld_20261011/protocol.json --repo-root . --restore-dir artifacts/restored
```

The resolver looks only inside the clone or the explicitly restored CAS; it does not fall back to the historical host. It maps paths and verifies data bytes, without changing frozen numerical algorithms, samples, tolerance or initial identity. Original training/analysis entry points retain historical path constants: they are scientific source snapshots, not the portable no-model replay entry. This publication task does not run their training or model-forward branches. The exact sources and inputs are available for explicit future, separately configured replay.

## Scope, identity and missing historical revisions

Current Perm64 / Graph4 / Matrix8 checkpoints and their actual saved step-0/RNG identities are included. Historical CKA/layer checkpoints are available, but their old original step-0 and starting RNG identities were never verified: a documented reconstruction is not a saved genuine initial. These limitations are listed in `docs/repository_curation/initialization_identity_scope.json`; no training is used to repair them.

Seven unique before-repair historical development source SHAs could not be located in the inspected source/history/artifact scope. Their original expected hashes and repair records remain; they are not silently replaced with current bytes. One missing common-source revision was recovered and is provided as an exact historical copy. See `docs/repository_curation/historic_source_byte_coverage.json` and `publication_provenance/historical_sources/`. These are distinct from the retained current main experiment source bindings. No claim of complete exact replay of every historical engineering state is made.

## Publication checks and previous work

`docs/repository_curation/` gives file-level decisions, explicit Git-only exits, dependency edges and byte checks from the original curation round. Local originals were not removed or moved in that round. Old independent F17/LIS/composition work remains accessible through prior Git history and releases, but does not belong to this current tree's scientific evidence. For source-exact historical replay, use the recorded original revisions and archives. The current English working-tree derivatives are documented in `docs/english_localization/README.txt`; translation is not a new scientific run.

`docs/representation_publication_audit.json` records the source/upload commits, public assets, old-release preservation and actual clean-clone/readback checks. It distinguishes provider-reported asset digests from bytes actually retrieved again. The smaller downloaded scope must not be described as full independent remote readback of all packages. Likewise, local complete stream checks are not new model or statistical checks.

GitHub's documented ordinary-Git single-file limit and per-release-asset limit guided this separation: [large files](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github), [releases](https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases). All publication assets stay below 1.5 GiB; individual payload files preserve their original bytes.

## Optional fresh-public-clone qualification

[tools/verify_publication.py](../tools/verify_publication.py) is the separate engineering wrapper used for the recorded public check. By default it performs no network or Git operation. Its explicit run requires the expected main commit, scientific source commit, manifest SHA, a fresh outside directory and a base Python interpreter with venv/ensurepip. See [its usage](../tools/CLEAN_CLONE_VERIFICATION.txt). The recorded run used a Miniforge base Python 3.13.13 only to create a fresh venv without inherited scientific packages; the decoder was installed separately. This is distinct from the original research virtual environment.

The current report-entry links were checked separately from [legacy HTML navigation gaps](repository_curation/legacy_link_coverage.json). Two historical HTML files retain 24 broken snapshot-relative or exited-independent-project links. Their original bytes are preserved in history; their current prose has an English derivative. These navigation limitations do not turn source fingerprints or tables into complete historical runtime reconstruction.

Final actual checks are recorded in [the publication report](repository_curation/FINAL_REPORT.txt) and [public/readback audit](representation_publication_audit.json). The independent clone qualified commit 29ec0802987fda8d1c5b9986426233d4ceb4b4c8; later metadata-only commits publish these receipts. 13 actual package downloads cover 3,771,371,797 compressed bytes and 1974 SHA-verified CAS objects; 206 required identities were independently reconciled, with no second current-CAS content hash requested. 76 other packages were not independently downloaded again.
