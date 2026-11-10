#!/usr/bin/env python3
"""Describe the four new repetitions separately from the full five-unit set.

Only completed saved tables are summarized through the frozen unit-summary
implementation. No training, forward pass, CKA, selection, or ranking is run.
The original exploratory unit is never counted as one of the four new units.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
DIAG = ROOT / "diagnostics/permworld_repetition_20261106"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments.permworld_repetition_20261106 import analysis


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _records(config, diag):
    root = Path(config["output"])
    historical = Path(diag) / "historical_analysis"
    records = []
    for unit, directory in [(config["historical_unit"], historical)] + [
            (u["unit"], root / u["unit"] / "analysis") for u in config["units"]]:
        completion = directory / "completion.json"
        value = read(completion) if completion.exists() else {}
        completed = value.get("status") == "completed"
        if unit != config["historical_unit"]:
            completed = completed and value.get("expected_conditions") == 32
        records.append(dict(unit=unit, analysis_dir=str(directory), status="completed" if completed else "pending",
                            completion_sha256=sha(completion) if completion.exists() else None,
                            recorded_status=value.get("status", "missing")))
    return records[0], records[1:]


def _csv(path):
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def summarize_cohorts(historical_record, new_records, outdir):
    """Pure saved-table orchestration with injectable records for small tests."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    generated_utc = datetime.now(timezone.utc).isoformat()
    if len(new_records) != 4 or len({r["unit"] for r in new_records}) != 4:
        raise ValueError("Exactly four distinct newly registered units are required")
    if historical_record["unit"] in {r["unit"] for r in new_records}:
        raise ValueError("Historical exploratory unit must be distinct from the new four")
    cohorts = (("new_four", list(new_records), "Four newly registered repetitions; historical exploratory unit excluded"),
               ("all_five", [historical_record] + list(new_records), "Historical exploratory unit plus the same four new repetitions"))
    result = {}
    lines = ["# PermWorld repetition cohorts", "", "Snapshot UTC: " + generated_utc + ".", "",
        "The primary presentation separates the four new repetitions from the combined five-unit description. The historical exploratory unit is included only in `all_five`; these are overlapping cohorts, not independent groups.", "",
        "Comparison A fixes a task pair and compares correct versus wrong input transformations. Its counts describe positive trained selectivity `Bt` and positive initialization-relative change `Delta`. Comparison B fixes the input transform and varies the right specialist; its counts describe designated scores above the mean of fourteen other candidates and uniquely first among all fifteen tasks. These four statistics are not interchangeable.", "",
        "Every count retains its denominator of 32 dependent conditions within a complete unit. Mean, sample SD, and range are calculated across completed units only; no pooling into a binomial trial is performed. Missing units remain in the manifest and are not negative results. With no completed units there are no estimated statistics; with one, sample SD is undefined.", ""]
    for name, records, definition in cohorts:
        destination = outdir / name
        summary = analysis.summarize_units(records, destination)
        summary.update(cohort=name, definition=definition, generated_utc=generated_utc, requested_unit_ids=[r["unit"] for r in records],
                       source_records=records, historical_exploratory_unit_included=name == "all_five")
        analysis.write_json(destination / "cohort_identity.json", summary)
        result[name] = summary
        lines += [f"## {name}: {definition}", "", f"Completed: **{summary['completed_units']}/{summary['listed_units']}**.", ""]
        if not summary["completed_units"]:
            lines += ["No completed new unit is available. The new-cohort means, sample standard deviations, ranges, and sign-consistency counts are not estimated. Pending entries are retained in `unit_status.csv`.", ""]
            continue
        rows = [r for r in _csv(destination / "unit_count_summary.csv") if r.get("control") == "answer_strata" and r.get("metric") == "linear_cka"]
        lines += ["| Statistic | Finite units | Mean | Sample SD | Min | Max |", "|---|---:|---:|---:|---:|---:|"]
        for row in rows:
            label = row["comparison"] + ": " + row["statistic"]
            cells = [row.get(k) or "undefined" for k in ("finite_units", "mean", "sample_sd", "minimum", "maximum")]
            lines.append("| " + " | ".join([label] + cells) + " |")
        lines += ["", "All unit-specific counts, continuous effects, finite-value counts, and descriptive direction agreement remain in this cohort's CSV files. Original metric precision is retained; exact signs introduce no practical-effect threshold.", ""]
    lines += ["These jointly varied data sampling, shared architecture-level initialization, batch order, and training streams. They do not isolate any one random factor, test independent task initialization within a unit, or identify a mathematical mechanism.", ""]
    (outdir / "report.md").write_text("\n".join(lines))
    analysis.write_json(outdir / "cohort_summary.json", result)
    return result


def summarize_registered_cohorts(protocol=DIAG / "protocol.json", outdir=DIAG / "cohort_summary"):
    config = read(protocol)
    if [u["unit"] for u in config["units"]] != ["unit02", "unit03", "unit04", "unit05"]:
        raise ValueError("The four newly registered units differ from the fixed manifest")
    original, new = _records(config, Path(protocol).parent)
    return summarize_cohorts(original, new, outdir)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DIAG / "protocol.json")
    parser.add_argument("--output-dir", type=Path, default=DIAG / "cohort_summary")
    args = parser.parse_args()
    result = summarize_registered_cohorts(args.protocol, args.output_dir)
    print(json.dumps({k: {f: v[f] for f in ("completed_units", "listed_units")} for k, v in result.items()}, indent=2))


if __name__ == "__main__":
    main()
