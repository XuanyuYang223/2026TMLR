Repository-wide English edition
Date: 2026-10-15 (America/Los_Angeles)

Scope
The current tracked working tree of XuanyuYang223/2026TMLR and the existing local manuscript draft are converted to English: reader reports, historical requests/reviews, source comments/docstrings/output templates, JSON/CSV display labels, entry documents, and HTML language tags. The manuscript reader guide is renamed reader_guide.txt. Git history, ignored independent F17 originals, and previously published binary release archives are not rewritten.

Scientific and historical integrity
This is a language edition, not a new experiment. No new training, checkpoint loading, model forward, activation-metric computation, parameter change, task selection, or scientific failure repair occurs. Original numerical tables, formulas, input/task/model identifiers, tolerances, and failed/undefined outcomes are retained. Matching human-readable status labels and their generating code are translated consistently.

Recorded historical SHAs continue to identify their original source/report/protocol bytes at the stated revision. A translated source/report is a derivative and must not be claimed to pass an old byte-identity check. For exact replay of a frozen historical source, check out its recorded revision or restore its original release object. Do not weaken historical tolerances or replace a saved genuine initial state with a reconstruction.

file_changes.json records the before/after identities of this language edition. validation.json describes actual syntax, data-shape/value, source-structure, language-coverage, and lightweight saved-table checks. These are language/engineering checks, not independent scientific replications. Original-language text is recoverable from Git; the local untracked manuscript and any additional original texts were also backed up outside the repository before conversion.

Current no-model table replay
The ordinary no-model replay remains tools/recompute_tables.py. Its checksum inventory explicitly identifies the English working-tree edition; original publication checksum records are retained separately as historical_table_checksums.json. Changed language-only metadata is bound to its current SHA without relabeling it as the original published bytes. Published download payload hashes/URLs remain their original identities.

Publication boundary
This conversion does not upload a new release, repack old scientific payloads, rewrite Git history, or submit a manuscript. Already published archives may contain original-language historical logs; they remain byte-identical artifacts. The English manuscript source archive is a separate local working-draft package, with its own receipt.

Subsequent Git publication
The user authorized committing and pushing the English edition and working draft on 2026-10-19. The original 2026-10-15 validation receipts above record the earlier local conversion, before that authorization. The later manuscript preparation receipt records its publication-status wording, refreshed source archive and successful compilation. Git commit history and the remote branch identify the actual publication outcome. This Git publication does not create a release or constitute journal submission.
