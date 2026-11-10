# Peaks-to-valleys probe replication download

- [Completed replication package](PermWorld_Peaks_Valleys_Probe_Replications.zip): all unit02–unit05 result directories, the cache-reproduced historical pilot, pooled reports/figures, protocol/audits, and unchanged supplied probe code. It preserves repository-relative paths under `diagnostics/` and `experiments/`; see the diagnostic README after extraction.
- [Original user-supplied package](PermWorld_Peaks_Valleys_Probe.zip), preserved byte-for-byte (SHA-256 `abc932d855e8f32b808cc1b665693ab66ed7b1f4f8cdd8f3d46cc97f800befdd`).
- [Archive checksums and sizes](manifest.json). The replication archive also includes a per-member content manifest and was read back to verify every member hash.

Read the [result overview](../../diagnostics/permworld_peaks_probe_20261109/README.md) and [full result tables/figures](../../diagnostics/permworld_peaks_probe_20261109/summary/report.md). The archive contains scalar tables and code, not weights or activation arrays. Re-fitting requires the original data and cached features described by the frozen source inventory. No new neural forward or base training was performed.

The cycle complement-minus-best-alternative contrast is positive in all four newer units, averaging **0.010535**, but complement AUROC averages **0.509862** and is below 0.5 in one unit. The historical 0.525270 is higher than all four new values. This is a small relative readout preference for one task pair; it is not a uniform above-chance localizer or evidence of a general mathematical mechanism.
