# Completed PermWorld follow-ups: reports, result files and writing evidence

This publication collects six previously local November follow-ups. Start with the [three-check closing report](../../diagnostics/permworld_closing_audit_20261110/README.md), then the [claim–evidence–scope table](../../diagnostics/permworld_closing_audit_20261110/writing/claim_evidence_scope.md). All primary means use **unit02–unit05**; the historical unit is a separate reference. These are exploratory analyses following earlier results, not retrospectively preregistered confirmation experiments.

**Published and verified.** The [completion receipt](publication_completion.json) records 20 release attachments (18 result packages, the reader ZIP and the manifest). [Public verification](public_verification.json) passed 2,574 checks: every asset's provider size/digest/state, all 623 selected files in a clean public clone, complete reader-ZIP download/member hashes, and the smallest package's public download/restore. The other large packages were fully verified locally, not downloaded again in full. Four older releases and their 117 assets retained their recorded metadata. This upload is complete; no new experiment is running.

## Read the results

| Package | Reader entry | Scope |
|---|---|---|
| Peaks → valleys orthogonal alignment | [Report](../../diagnostics/permworld_orthogonal_alignment_20261110/README.md) | Both GRU encodings, all 21 lengths, each unit and mapping control |
| Exceedances → deficiencies | [Report](../../diagnostics/permworld_exceedance_alignment_20261110/README.md) | Final-state alignment plus separately reported token-state diagnostics; failed predicted encoding crossover retained |
| Explicit downstream composition | [Report](../../diagnostics/permworld_lemma_composition_20261110/README.md) | Three questions using frozen position readers; same-specialist-twice baseline retained |
| Within-model local edits | [Report](../../diagnostics/permworld_local_causal_20261110/README.md) | Verified counterfactual pairs, main and cumulative-count-matched groups |
| Early versus late edits | [Reader report](../../diagnostics/permworld_local_timing_20261110/headlines/reader_report.md) | Paired timing comparison, whole-layer controls and four-unit raw values |
| Three closing checks and writing | [Report](../../diagnostics/permworld_closing_audit_20261110/README.md) | Same trained-source model controls, local gradients and E/D recheck |

The existing E/D report has one historical relative-link error: its token coverage table belongs at `diagnostics/permworld_exceedance_alignment_20261110/token_reader/token_coverage_by_length.csv`, rather than at that report's root. This large table is supplied through the release manifest and restored under `repo/`, rather than hosted as an ordinary Git file. Frozen reports were not edited to repair their provenance-bound bytes.

## Writing materials

- [Claim–evidence–scope table](../../diagnostics/permworld_closing_audit_20261110/writing/claim_evidence_scope.md), also available as [CSV](../../diagnostics/permworld_closing_audit_20261110/writing/claim_evidence_scope.csv).
- [TMLR main-text structure](../../diagnostics/permworld_closing_audit_20261110/writing/tmlr_structure.md).
- [English result paragraphs](../../diagnostics/permworld_closing_audit_20261110/writing/results_draft.md), with [TeX paragraphs](../../diagnostics/permworld_closing_audit_20261110/writing/results_draft.tex).

The old working-paper PDF predates these additions. Its source and build products have not been overwritten. Geometry, readout transfer, analyst-defined composition and within-model intervention answer different questions; they are not one effect progressively shrinking. Within-model editing is not a direct test of cross-model shared computation.

## Download real result files

For reports, code and smaller tables only, download the [18 MB reader/code ZIP](https://github.com/XuanyuYang223/2026TMLR/releases/download/permworld-followups-2026-11-16/PermWorld_Followups_Reports_Code.zip); its [checksum and exact scope](reader_archive.json) are separate from the full-result manifest below. It does not contain the large numerical output arrays.

Read [the release](https://github.com/XuanyuYang223/2026TMLR/releases/tag/permworld-followups-2026-11-16) and the [download/restore manifest](manifest.json). Repository code, readable reports, figures and smaller tables are directly browsable. Larger tables and binary outputs are in content-addressed release attachments. Local absolute symlinks were **not** committed as substitutes for data: their targets' actual bytes and logical aliases are in the manifest.

The attachments cover the recorded output bundles: fitted probes/scalers/maps, unthresholded position scores, per-example predictions, intervention outputs, gradients, counterfactual families and supporting receipts. They also include originals of the smaller Git-hosted bundle files, so a full download preserves one bounded result snapshot. Some large-file links in unchanged historical reports require restoring the release files first; use `repo/` for repository-relative files and `external/20261110/` for previously F-drive outputs.

**Not included:** the separately inventoried raw recurrent and token-state intermediate caches, prior base checkpoints/input activation caches, and third-party Python dependencies. See [the explicit excluded-cache inventory](excluded_intermediate_caches.json). Full model replay still requires the source dependencies stated by each original protocol. This is not a claim to publish every input or every file on the F drive.

From a clone, restore the complete result release to a new directory:

```bash
python -m pip install -r requirements-artifacts.txt
python tools/download_artifacts.py \
  --manifest artifacts/permworld_followups_20261116/manifest.json \
  --profile full \
  --directory artifacts/downloads/permworld_followups_20261116 \
  --restore-dir artifacts/restored/permworld_followups_20261116 \
  --materialize \
  --output artifacts/downloads/permworld_followups_20261116/restore_audit.json
```

`--profile minimal` retrieves only the smallest package for a bounded download/restore check; it does **not** restore all six studies. Restoration verifies asset and object SHA-256 and creates regular files in the chosen restoration directory. Existing differing files are never overwritten. Original absolute paths remain provenance keys in `path_map.json`; no automatic relocation of an old experiment or training is performed.

## Publication versus scientific validation

[File inventory](file_inventory.json), [source snapshot](source_snapshot.json), [packing audit](packaging_audit.json) and [complete local archive verification](local_package_verification.json) document this upload's scope. The [publisher](../../tools/publish_permworld_followups.py) reads existing files and reuses the established byte-preserving packer; it never trains, fits a probe, evaluates a network or computes a scientific metric. Re-running `prepare` after modifying Git status intentionally rejects a changed snapshot; use the saved inventory for packaging and restoration.

The analysis-time `no_upload` flags in old completion receipts describe those earlier rounds. The user separately authorized this publication on 16 November. Successful packaging, upload and byte verification do not resolve scientific limitations, count samples as independent repetitions, establish a mechanism, or turn a failed prediction into a success. No additional experiment was started.
