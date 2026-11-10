#!/usr/bin/env python3
"""Build the existing English paper from recorded repetition status only.

No training, feature extraction, geometry recomputation, Git mutation, or upload
is performed. ``--build-current`` permits an explicitly interim build while the
formal worker runs. ``--wait-pid`` waits for that specific worker to finish and
requires the complete five-unit record before producing a final-status build.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DIAG = ROOT / "diagnostics/permworld_repetition_20261106"
PAPER = ROOT / "manuscripts/representation_selectivity_20261012"
PROTOCOL = DIAG / "protocol.json"
ENGINE = ROOT / ".venv/tectonic_bin/tectonic"
EXTERNAL = Path("/mnt/yangx-projects-f/20261106/permworld_repetition")
BACKUPS = Path("/mnt/yangx-projects-f/20261106/paper_backups/representation_selectivity_20261012")
NEW_MEMBERS = (
    "sections/independent_repetition_methods.tex",
    "sections/independent_repetition_status.tex",
    "provenance/repetition_build_20261106.json",
)
FINAL_MEMBERS = ("sections/independent_repetition_results.tex", "figures/figure6_independent_repetition.png")
EXPECTED_UNITS = ("unit02", "unit03", "unit04", "unit05")
RECEIPT = DIAG / "paper_finalization.json"


def utc():
    return datetime.now(timezone.utc).isoformat()


def digest_bytes(value):
    return hashlib.sha256(value).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(2**20), b""):
            h.update(part)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def validate_scope():
    if PROTOCOL.resolve() != ROOT / "diagnostics/permworld_repetition_20261106/protocol.json":
        raise ValueError("Protocol must be the explicitly scoped repetition protocol")
    if PAPER.resolve() != ROOT / "manuscripts/representation_selectivity_20261012":
        raise ValueError("Paper directory escapes its explicit scope")
    config = read(PROTOCOL)
    if Path(config["output"]).resolve() != EXTERNAL.resolve():
        raise ValueError("External repetition root differs from the fixed scoped root")
    if tuple(u["unit"] for u in config["units"]) != EXPECTED_UNITS:
        raise ValueError("The bounded four-unit manifest changed")
    if len(config["tasks"]) != 16 or config["architectures"] != ["transformer", "gru"]:
        raise ValueError("The original sixteen-task/two-configuration scope changed")
    if config["encodings"] != ["one_line", "canonical_cycle"]:
        raise ValueError("The original two-encoding scope changed")
    return config


def _safe_member(name):
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or "\\" in name:
        raise ValueError("Unsafe archive member: " + name)
    if name.endswith("/"):
        raise ValueError("Explicit directory members are not source files")
    return name


def backup_originals():
    """Preserve the tracked historical artifacts, even if working PDF changed."""
    BACKUPS.mkdir(parents=True, exist_ok=True)
    baseline = BACKUPS / "baseline.json"
    if baseline.exists():
        record = read(baseline)
        for artifact in record["artifacts"].values():
            path = Path(artifact["backup_path"])
            if path.parent.resolve() != BACKUPS.resolve() or sha(path) != artifact["sha256"]:
                raise ValueError("Original artifact backup identity changed")
        names = record["original_source_members"]
        if len(names) != 42 or len(set(names)) != 42:
            raise ValueError("Original source inventory must contain forty-two members")
        return record
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    artifacts, zip_bytes = {}, None
    for name in ("manuscript.pdf", "manuscript_source.zip"):
        relative = str((PAPER / name).relative_to(ROOT))
        result = subprocess.run(["git", "show", f"{head}:{relative}"], cwd=ROOT, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, check=False)
        if result.returncode:
            raise RuntimeError("Cannot read tracked original " + name + ": " + result.stderr.decode(errors="replace"))
        payload = result.stdout
        digest = digest_bytes(payload)
        target = BACKUPS / f"{head[:12]}_{digest[:16]}_{name}"
        if target.exists() and sha(target) != digest:
            raise ValueError("Conflicting original artifact backup")
        if not target.exists():
            target.write_bytes(payload)
        artifacts[name] = dict(source=f"git show {head}:{relative}", backup_path=str(target), sha256=digest,
                               size_bytes=len(payload))
        if name.endswith(".zip"):
            zip_bytes = payload
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as archive:
        if archive.testzip() is not None:
            raise ValueError("Tracked original source archive is corrupt")
        names = [_safe_member(n) for n in archive.namelist()]
    if len(names) != 42 or len(set(names)) != 42:
        raise ValueError("Tracked baseline is not the original forty-two-member source archive")
    record = dict(saved_utc=utc(), git_head=head, artifacts=artifacts, original_source_members=names,
                  scope="Only the historical tracked paper PDF and source ZIP; no scientific artifact modified")
    write(baseline, record)
    return record


def completion_state(require_final=False):
    """Read existing completion records only; do not refresh or run analyses."""
    config_identity = digest_bytes(json.dumps(read(PROTOCOL), sort_keys=True).encode())
    rows = []
    for unit in EXPECTED_UNITS:
        path = EXTERNAL / unit / "analysis/completion.json"
        value = read(path) if path.is_file() else None
        complete = bool(value and value.get("status") == "completed" and value.get("expected_conditions") == 32)
        if complete and value.get("binding", {}).get("config_sha256") != config_identity:
            raise ValueError("Completed unit analysis is bound to a different configuration: " + unit)
        rows.append(dict(unit=unit, completed=complete, completion_path=str(path),
                         completion_sha256=sha(path) if path.exists() else None,
                         recorded_status=value.get("status") if value else "missing",
                         recorded_expected_conditions=value.get("expected_conditions") if value else None))
    summary_path = DIAG / "summary/summary_completion.json"
    summary = read(summary_path)
    actual = 1 + sum(r["completed"] for r in rows)
    if summary.get("listed_units") != 5 or summary.get("completed_units") != actual:
        raise ValueError("Summary and actual complete-unit records disagree")
    historical = DIAG / "historical_analysis/completion.json"
    if read(historical).get("status") != "completed":
        raise ValueError("Historical table verification is not complete")
    if require_final:
        if not all(r["completed"] for r in rows) or summary.get("completed_units") != 5:
            raise ValueError("Formal worker exited before all four new analyses and five-unit summary completed")
        exit_path = EXTERNAL / "pipeline_exit.json"
        if not exit_path.is_file() or read(exit_path).get("status") != "finished_requested_manifest":
            raise ValueError("Successful formal pipeline exit record missing")
        active = read(EXTERNAL / "active_process.json")
        exited_at = datetime.fromisoformat(read(exit_path)["utc"])
        started_at = datetime.fromisoformat(active["started_utc"])
        if exited_at < started_at or active.get("protocol_sha256") != sha(PROTOCOL):
            raise ValueError("Formal pipeline exit record is stale or belongs to another protocol")
    section = PAPER / NEW_MEMBERS[1]
    text = section.read_text()
    if f"{actual} of five units have complete main analyses" not in text:
        raise ValueError("Paper status section is stale relative to the existing summary")
    return dict(completed_units=actual, listed_units=5, additional_units=rows,
                summary_completion_sha256=sha(summary_path), historical_completion_sha256=sha(historical),
                summary=summary, final_complete=actual == 5)


def process_identity(pid):
    """Return /proc start ticks; treat zombies as stopped and detect PID reuse."""
    try:
        value = Path(f"/proc/{pid}/stat").read_text()
        fields = value[value.rfind(")") + 2:].split()
        if fields[0] in ("Z", "X"):
            return None
        return fields[19]  # field 22 (starttime); fields[0] is field 3
    except (FileNotFoundError, ProcessLookupError):
        return None


def wait_for_worker(pid, max_wait_hours, poll_seconds):
    launch = read(DIAG / "launch_record.json")
    if launch.get("pid") != pid:
        raise ValueError("Requested PID differs from the recorded formal worker PID")
    identity = process_identity(pid)
    if identity is not None:
        command = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
        if "experiments.permworld_repetition_20261106.runner" not in command:
            raise ValueError("Recorded PID is no longer the expected formal runner")
    deadline = time.monotonic() + max_wait_hours * 3600
    print(json.dumps(dict(event="waiting_for_formal_worker", pid=pid, maximum_wait_hours=max_wait_hours)), flush=True)
    while identity is not None and process_identity(pid) == identity:
        if time.monotonic() >= deadline:
            raise TimeoutError("Bounded formal-worker wait expired; no final completion claimed")
        time.sleep(min(poll_seconds, max(0.01, deadline - time.monotonic())))
    # Missing/failing exit receipts cannot leave a watcher hanging indefinitely.
    refresh_cohort_summaries()
    return completion_state(require_final=True)


def refresh_cohort_summaries():
    """Saved-table summaries only, keeping new four separate from old+new."""
    from tools.summarize_permworld_repetition_cohorts import summarize_registered_cohorts
    return summarize_registered_cohorts(PROTOCOL, DIAG / "cohort_summary")


def archive_dependencies(names):
    names = set(names)
    for name in names:
        if not name.endswith(".tex"):
            continue
        text = (PAPER / name).read_text()
        text = re.sub(r"(?<!\\)%[^\n]*", "", text)
        for reference in re.findall(r"\\(?:input|include)\s*\{([^}]+)\}", text):
            candidate = reference if Path(reference).suffix else reference + ".tex"
            if candidate not in names:
                raise ValueError(f"Source ZIP omits TeX input {candidate}, referenced by {name}")
        for reference in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\s*\{([^}]+)\}", text):
            possible = [reference] if Path(reference).suffix else [reference + ext for ext in (".pdf", ".png", ".jpg", ".jpeg", ".eps")]
            if not any(candidate in names for candidate in possible):
                raise ValueError(f"Source ZIP omits graphics input {reference}, referenced by {name}")
        for reference in re.findall(r"\\bibliography\s*\{([^}]+)\}", text):
            for part in reference.split(","):
                if part.strip() + ".bib" not in names:
                    raise ValueError("Source ZIP omits bibliography " + part)


def render_final_results(summary_dir, paper_dir, new_summary_dir=None):
    """Render all fixed primary summaries; never recompute scientific metrics.

    Exposed separately so fixture validation needs no real completion override.
    """
    summary_dir, paper_dir = Path(summary_dir), Path(paper_dir)
    new_summary_dir = Path(new_summary_dir) if new_summary_dir is not None else summary_dir.parent / "new_four"
    def rows(name, directory=summary_dir):
        with (directory / name).open(newline="") as stream:
            return list(csv.DictReader(stream))
    def integer(value):
        return int(float(value))
    def display(value):
        return "undefined" if value in (None, "", "None", "nan", "NaN") else f"{float(value):.6f}"
    def tex(value):
        return str(value).replace("_", r"\_").replace("%", r"\%")
    def primary_counts(directory, expected):
        found_rows = [r for r in rows("unit_count_summary.csv", directory) if r["control"] == "answer_strata" and r["metric"] == "linear_cka"]
        output = []
        for comparison, statistic, label in (("A", "Bt_positive", r"A: $B_t>0$"), ("A", "Delta_positive", r"A: $\Delta>0$"),
                                             ("B", "above_other_mean", "B: above other mean"), ("B", "unique_first", "B: unique first")):
            found = [r for r in found_rows if r["comparison"] == comparison and r["statistic"] == statistic]
            if len(found) != 1 or integer(found[0]["finite_units"]) != expected:
                raise ValueError(f"Final paper needs exactly {expected} complete primary count observations in {directory}")
            output.append((label, found[0]))
        return output
    selected = primary_counts(summary_dir, 5)
    new_selected = primary_counts(new_summary_dir, 4)
    effects = [r for r in rows("condition_effects_summary.csv") if r["control"] == "answer_strata" and
               r["metric"] == "linear_cka" and r["branch"] == "hidden"]
    specifications = (("A", "Bt", r"Comparison A: trained selectivity $B_t$"), ("A", "Delta", r"Comparison A: initialization-relative change $\Delta$"),
                      ("B", "trained_designated_minus_other_mean", "Comparison B: trained designated-minus-other-mean contrast"))
    groups, keys = [], None
    for comparison, effect, label in specifications:
        group = sorted((r for r in effects if r["comparison"] == comparison and r["effect"] == effect),
                       key=lambda r: (r["architecture"], r["encoding"], r["relation"]))
        identity = [(r["architecture"], r["encoding"], r["relation"]) for r in group]
        if len(group) != 32 or len(set(identity)) != 32 or (keys is not None and identity != keys):
            raise ValueError("Final paper needs the same complete thirty-two-condition inventory for all three effects")
        keys = identity
        groups.append((label, group))
    new_effects = [r for r in rows("condition_effects_summary.csv", new_summary_dir) if r["control"] == "answer_strata" and
                  r["metric"] == "linear_cka" and r["branch"] == "hidden"]
    new_groups = []
    for comparison, effect, label in specifications:
        group = sorted((r for r in new_effects if r["comparison"] == comparison and r["effect"] == effect),
                       key=lambda r: (r["architecture"], r["encoding"], r["relation"]))
        if [(r["architecture"], r["encoding"], r["relation"]) for r in group] != keys:
            raise ValueError("New-only and combined cohorts have different fixed condition inventories")
        new_groups.append((label, group))
    def count_table(observations):
        out = [r"\begin{center}\small\begin{tabular}{lrrrr}\toprule",
               r"Statistic & Mean & Sample SD & Minimum & Maximum\\\midrule"]
        for label, row in observations:
            out.append(label + " & " + " & ".join(display(row[k]) for k in ("mean", "sample_sd", "minimum", "maximum")) + r"\\")
        return out + [r"\bottomrule\end{tabular}\end{center}"]
    def direction_table(condition_groups, size):
        out = [r"\begin{center}\small\begin{tabular}{lrrrr}\toprule",
               f"Effect & Positive in all {size} & Negative in all {size} & Mixed or zero & Undefined unit(s)" + r"\\\midrule"]
        for label, group in condition_groups:
            if any(not 0 <= integer(r["finite_units"]) <= size for r in group):
                raise ValueError("Condition finite-unit count exceeds its explicitly selected cohort")
            positive = sum(integer(r["finite_units"]) == size and integer(r["positive_units"]) == size for r in group)
            negative = sum(integer(r["finite_units"]) == size and integer(r["negative_units"]) == size for r in group)
            undefined = sum(integer(r["finite_units"]) < size for r in group)
            short = r"A: $B_t$" if group[0]["effect"] == "Bt" else r"A: $\Delta$" if group[0]["effect"] == "Delta" else "B: trained contrast"
            out.append(f"{short} & {positive} & {negative} & {32-positive-negative-undefined} & {undefined}" + r"\\")
        return out + [r"\bottomrule\end{tabular}\end{center}"]
    text = [r"\paragraph{Four new repetitions, excluding the exploratory unit.}",
        "All four newly registered units have complete main analyses. They are presented first, without the historical exploratory unit. Each primary count retains its denominator of 32 dependent conditions, and each summary observation is a complete unit."]
    text += count_table(new_selected)
    text += ["The new-only exact sign agreement is reported separately below; agreement does not imply a practically large effect."]
    text += direction_table(new_groups, 4)
    text += [r"\paragraph{All five units, including the historical exploratory unit.}",
        "The combined description adds the original exploratory unit to the same four new repetitions. These overlapping cohorts are not independent groups, and the historical unit is not counted as a new replication."]
    text += count_table(selected)
    text += ["The combined sign summary again uses the original exact zero boundary; a near-zero sign change is not described as a substantive reversal."]
    text += direction_table(groups, 5)
    text += [
        "The repetitions jointly resample data, shared architecture-level initialization, batch order, and registered training streams. They do not isolate these factors, test independently initialized task specialists within a unit, or identify a mathematical mechanism. Neither the conditions nor pooled positive counts are independent binomial trials.",
        r"Figure~\ref{fig:independent-repetition} shows every unit's primary continuous effects. The following tables report all 32 conditions for each effect across all five units; six-decimal formatting is display-only. Separate new-four and all-five condition-effect and unit-count CSV files retain unrounded summaries, all unit-specific values, and finite-unit counts.",
        r"\begin{figure}[p]\centering",
        r"\includegraphics[width=\textwidth]{figures/figure6_independent_repetition.png}",
        r"\caption{All registered units' answer-controlled final-layer ordinary-CKA effects. Each marker is a unit-specific measurement of a dependent condition; no condition-level independence or practical-effect threshold is assumed.}\label{fig:independent-repetition}\end{figure}"]
    for index, (label, group) in enumerate(groups, 1):
        text += [r"\begin{table}[p]\centering\scriptsize",
            r"\begin{tabular}{llrrrrr}\toprule",
            r"Config./encoding & Relation & Finite units & Mean & Sample SD & Minimum & Maximum\\\midrule"]
        for row in group:
            config = ("TF" if row["architecture"] == "transformer" else "GRU") + "/" + ("line" if row["encoding"] == "one_line" else "cycle")
            text.append(config + " & " + tex(row["relation"]) + " & " + row["finite_units"] + " & " +
                        " & ".join(display(row[k]) for k in ("mean", "sample_sd", "minimum", "maximum")) + r"\\")
        text += [r"\bottomrule\end{tabular}", r"\caption{" + label + "; all fixed conditions. Statistics describe finite complete-unit observations; undefined values are not filled with zero.}",
                 rf"\label{{tab:independent-effect-{index}}}\end{{table}}"]
    destination = paper_dir / FINAL_MEMBERS[0]
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(text) + "\n")
    figure = paper_dir / FINAL_MEMBERS[1]
    figure.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(summary_dir / "condition_effects_by_unit.png", figure)
    status = paper_dir / NEW_MEMBERS[1]
    include = r"\input{sections/independent_repetition_results}"
    original = status.read_text()
    original = "\n".join(line for line in original.splitlines() if line.strip() != include)
    status.write_text(original.rstrip() + "\n" + include + "\n")
    return dict(primary_count_summary_sha256=sha(summary_dir / "unit_count_summary.csv"),
                condition_effect_summary_sha256=sha(summary_dir / "condition_effects_summary.csv"),
                source_figure_sha256=sha(summary_dir / "condition_effects_by_unit.png"),
                new_four_count_summary_sha256=sha(new_summary_dir / "unit_count_summary.csv"),
                new_four_condition_effect_summary_sha256=sha(new_summary_dir / "condition_effects_summary.csv"),
                generated_members=list(FINAL_MEMBERS), complete_condition_effect_rows=96)


def build(state, mode):
    original = backup_originals()
    if not ENGINE.is_file():
        raise FileNotFoundError("Existing local Tectonic engine is unavailable")
    final_results = render_final_results(DIAG / "cohort_summary/all_five", PAPER, DIAG / "cohort_summary/new_four") if state["final_complete"] else None
    names = list(original["original_source_members"]) + list(NEW_MEMBERS) + (list(FINAL_MEMBERS) if final_results else [])
    expected_count = 47 if final_results else 45
    if len(names) != expected_count or len(set(names)) != expected_count:
        raise ValueError("Source inventory differs from the explicitly scoped interim/final additions")
    for name in names:
        if name == NEW_MEMBERS[-1]:
            continue
        _safe_member(name)
        path = PAPER / name
        if path.resolve().parent != PAPER.resolve() and PAPER.resolve() not in path.resolve().parents:
            raise ValueError("Source resolves outside the paper directory")
        if not path.is_file():
            raise FileNotFoundError("Required source is missing: " + name)
    archive_dependencies(names)
    before = {name: sha(PAPER / name) for name in names if name != NEW_MEMBERS[-1]}
    log_path = DIAG / "paper_build.log"
    with log_path.open("wb") as stream:
        result = subprocess.run([str(ENGINE), "--keep-logs", "--keep-intermediates", "manuscript.tex"],
                                cwd=PAPER, stdout=stream, stderr=subprocess.STDOUT, check=False, timeout=600)
    if result.returncode:
        raise RuntimeError(f"Tectonic failed with exit code {result.returncode}; see {log_path}")
    if before != {name: sha(PAPER / name) for name in before}:
        raise ValueError("Paper source changed during build; refusing mismatched archive")
    current = completion_state(require_final=mode == "final_after_worker")
    if current != state:
        raise ValueError("Completion records changed during build; rerun the document-only finalizer")
    pdf = PAPER / "manuscript.pdf"
    if not pdf.is_file() or not pdf.read_bytes().startswith(b"%PDF-"):
        raise ValueError("Build did not produce a PDF")
    provenance = dict(build_utc=utc(), build_mode=mode, completed_units=state["completed_units"], listed_units=5,
        final_five_unit_result_available=state["final_complete"], protocol_sha256=sha(PROTOCOL),
        manuscript_sha256=sha(PAPER / "manuscript.tex"), PDF_sha256=sha(pdf),
        source_sha256=before, completion_state=state, final_results_rendering=final_results,
        original_artifact_backup=str(BACKUPS / "baseline.json"),
        source_archive_note="ZIP SHA is recorded outside this ZIP in diagnostics/permworld_repetition_20261106/paper_finalization.json to avoid circular hashing.",
        actions="Document compilation and source packaging only; no training, geometry recomputation, or upload")
    write(PAPER / NEW_MEMBERS[-1], provenance)
    package = PAPER / "manuscript_source.zip"
    handle, temporary_name = tempfile.mkstemp(prefix=".manuscript_source_", suffix=".zip", dir=PAPER)
    os.close(handle)
    temporary = Path(temporary_name)
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
            for name in names:
                archive.write(PAPER / name, arcname=name)
        with zipfile.ZipFile(temporary) as archive:
            if archive.namelist() != names or archive.testzip() is not None:
                raise ValueError("ZIP member inventory/CRC verification failed")
            for name in names:
                if archive.read(name) != (PAPER / name).read_bytes():
                    raise ValueError("ZIP source bytes differ: " + name)
        temporary.replace(package)
    finally:
        temporary.unlink(missing_ok=True)
    record = dict(status="completed", build_utc=utc(), build_mode=mode, completed_units=state["completed_units"],
        listed_units=5, final_five_unit_result_available=state["final_complete"], protocol_sha256=sha(PROTOCOL),
        manuscript_sha256=sha(PAPER / "manuscript.tex"), PDF_sha256=sha(pdf), ZIP_sha256=sha(package),
        provenance_sha256=sha(PAPER / NEW_MEMBERS[-1]), archive_members=names, archive_member_count=len(names),
        source_bytes_and_dependencies_verified=True, build_log=str(log_path),
        original_artifact_backup=str(BACKUPS / "baseline.json"),
        no_training_or_geometry_recomputation=True, no_upload_or_git_mutation=True)
    write(RECEIPT, record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--build-current", action="store_true")
    mode.add_argument("--wait-pid", type=int)
    parser.add_argument("--max-wait-hours", type=float, default=168,
                        help="Bound the worker wait (default seven days); timeout records pending_or_failed")
    parser.add_argument("--poll-seconds", type=float, default=30)
    args = parser.parse_args()
    try:
        if not 0 < args.max_wait_hours <= 336 or not .05 <= args.poll_seconds <= 60:
            raise ValueError("Wait must be positive and at most fourteen days; polling must be .05–60 seconds")
        if args.wait_pid is not None and args.wait_pid < 1:
            raise ValueError("Worker PID must be positive")
        validate_scope()
        # Save originals immediately, also for a watcher waiting on a long job.
        backup_originals()
        if args.wait_pid:
            state = wait_for_worker(args.wait_pid, args.max_wait_hours, args.poll_seconds)
        else:
            refresh_cohort_summaries()
            state = completion_state()
        record = build(state, "final_after_worker" if args.wait_pid else "current_status")
        print(json.dumps(record, indent=2), flush=True)
        return 0
    except Exception as error:
        record = dict(status="pending_or_failed", recorded_utc=utc(), error_type=type(error).__name__, error=str(error),
                      requested_mode="final_after_worker" if args.wait_pid else "current_status", wait_pid=args.wait_pid,
                      final_five_unit_result_claimed=False, no_training_or_geometry_recomputation=True,
                      no_upload_or_git_mutation=True)
        write(RECEIPT, record)
        print(json.dumps(record, indent=2), file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
