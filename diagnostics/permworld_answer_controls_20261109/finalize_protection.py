#!/usr/bin/env python3
"""Endpoint-only SHA audit after the checkpoint-only diagnostic has finished.

Reads the frozen source inventory and named protection receipt, never modifies
source files, and never imports model or metric code. NPZ hashes cover entire
containers; NPY hashes cover entire files, including headers. These are file
SHA256 checks, not independently recomputed tensor-value digests.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EXPECTED_INVENTORY_SHA = "32d2490a79d81574d2f64ad5a24c55de46e55256fbcfb02705cffbfbee3559e1"
EXPECTED_START_SHA = "f2e99e7a52b60ccc845695c26fdff62f969142503049bde10a9f3f54bf02b26b"
REQUIRED_FIELDS = ("prefix_final", "prediction", "output_prob")
OPTIONAL_FIELDS = ("answer_ce",)


def utc():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load(path, expected=None):
    if expected is not None and digest(path) != expected:
        raise ValueError(f"Frozen receipt changed: {path}")
    return json.loads(Path(path).read_text())


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(tmp, path)


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def identify(record):
    return {k: record[k] for k in
            ("unit", "architecture", "encoding", "task", "phase", "split")}


def payload_plan(inventory):
    """Record all logical fields but hash each physical file only once."""
    files, records, errors = {}, [], []
    for record in inventory["test_features"]:
        row = identify(record)
        row.update(format=record["format"], fields=[], physical_files=[])
        arrays = record["arrays"]
        for field in REQUIRED_FIELDS:
            if field not in arrays:
                errors.append(dict(**identify(record), reason="missing_required_field", field=field))
        selected = [field for field in REQUIRED_FIELDS + OPTIONAL_FIELDS if field in arrays]
        row["fields"] = selected
        row["optional_answer_ce_present"] = "answer_ce" in arrays
        if record["format"] == "npz_fields":
            entries = [(record["path"], record["expected_payload_sha256"], selected)]
        elif record["format"] == "npy_fields":
            entries = [(arrays[f]["path"], arrays[f]["expected_payload_sha256"], [f]) for f in selected]
        else:
            errors.append(dict(**identify(record), reason="unknown_inventory_format"))
            entries = []
        for path, expected, fields in entries:
            row["physical_files"].append(path)
            entry = files.setdefault(path, dict(expected_sha256=expected, uses=[]))
            if entry["expected_sha256"] != expected:
                errors.append(dict(path=path, reason="conflicting_payload_sha256"))
            entry["uses"].append(dict(**identify(record), fields=fields))
        records.append(row)
    return files, records, errors


def file_check(item):
    """Check one entire file; record instability during this read explicitly."""
    path, expected = item
    result = dict(path=path, expected_sha256=expected, checked_utc=utc())
    try:
        before = Path(path).stat()
        actual = digest(path)
        after = Path(path).stat()
        stable = ((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
                  == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns))
        result.update(actual_sha256=actual, size_bytes=after.st_size,
                      sha256_matches=actual == expected, stable_during_hash=stable,
                      mtime_ns_before=before.st_mtime_ns, mtime_ns_after=after.st_mtime_ns,
                      passed=actual == expected and stable)
    except (OSError, ValueError) as exc:
        result.update(passed=False, error_type=type(exc).__name__, error=str(exc))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "protection_end.json")
    parser.add_argument("--payload-output", type=Path, default=HERE / "activation_payload_end.json")
    parser.add_argument("--workers", type=int, default=2,
                        help="Concurrent streaming file hashes (default: 2).")
    parser.add_argument("--plan-only", action="store_true",
                        help="Print metadata-only scope; do not hash large files or write receipts.")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    inventory_path, start_path = HERE / "source_inventory.json", HERE / "protection_start.json"
    inventory = load(inventory_path, EXPECTED_INVENTORY_SHA)
    start = load(start_path, EXPECTED_START_SHA)
    protected = start["protected_files"]
    if len(protected) != start["protected_file_count"] or len(protected) != 1449:
        raise ValueError("Unexpected named protection inventory; no scope substitution allowed")
    payloads, logical_records, plan_errors = payload_plan(inventory)
    if len(logical_records) != inventory["logical_test_feature_record_count"] or len(logical_records) != 640:
        raise ValueError("Unexpected logical activation inventory")
    all_files = dict(protected)
    for path, record in payloads.items():
        if path in all_files and all_files[path] != record["expected_sha256"]:
            plan_errors.append(dict(path=path, reason="protected_and_payload_SHA_disagree"))
        all_files[path] = record["expected_sha256"]
    if args.plan_only:
        print(json.dumps(dict(protected_files=len(protected), logical_activation_records=len(logical_records),
                              activation_payload_files=len(payloads), unique_files_to_hash=len(all_files),
                              plan_errors=plan_errors, large_payloads_read=False), indent=2))
        return
    # This script is for the end of the complete pipeline, not its reproduction gate.
    last = load(HERE / "last_stage.json")
    summary = load(HERE / "summary/summary_completion.json")
    if not (last.get("stage") == "all" and last.get("completed") is True
            and summary.get("status") == "completed_saved_score_summary"):
        raise ValueError("Complete pipeline end markers are required before this audit")
    for path in (args.output, args.payload_output):
        if str(path.resolve()) in {str(Path(p).resolve()) for p in all_files}:
            raise ValueError("Audit output would overwrite a protected source")
        if path.resolve() in {inventory_path.resolve(), start_path.resolve(), Path(__file__).resolve()}:
            raise ValueError("Audit output would overwrite an audit dependency")
    if args.output.resolve() == args.payload_output.resolve():
        raise ValueError("Protection and payload receipts must have distinct paths")
    begun, timer = utc(), time.monotonic()
    head_before = git("rev-parse", "HEAD")
    checks = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for i, row in enumerate(pool.map(file_check, sorted(all_files.items())), 1):
            checks[row["path"]] = row
            if i % 100 == 0:
                print(json.dumps(dict(event="endpoint_hash_progress", files=i, total=len(all_files))), flush=True)
    head_after = git("rev-parse", "HEAD")
    protected_checks = [checks[path] for path in sorted(protected)]
    payload_checks = [dict(checks[path], uses=payloads[path]["uses"]) for path in sorted(payloads)]
    for record in logical_records:
        record["all_listed_payload_files_passed"] = bool(record["physical_files"]) and all(
            checks[path]["passed"] for path in record["physical_files"])
    passed_payload = not plan_errors and all(r["all_listed_payload_files_passed"] for r in logical_records)
    payload_audit = dict(schema="permworld-answer-control-activation-end/v1",
        status="passed" if passed_payload else "failed", started_utc=begun, completed_utc=utc(),
        source_inventory_sha256=EXPECTED_INVENTORY_SHA,
        logical_record_count=len(logical_records), logical_records_passed=sum(r["all_listed_payload_files_passed"] for r in logical_records),
        physical_file_count=len(payloads), physical_files_passed=sum(r["passed"] for r in payload_checks),
        physical_file_bytes_hashed=sum(r.get("size_bytes", 0) for r in payload_checks),
        hash_scope="Entire listed NPY files and NPZ containers; no separate ndarray-value digest or NPZ-member decompression audit.",
        required_fields=list(REQUIRED_FIELDS), optional_fields_when_present=list(OPTIONAL_FIELDS),
        plan_errors=plan_errors, logical_records=logical_records, file_checks=payload_checks,
        endpoint_only=True, no_training=True, no_neural_forward=True, no_activation_metrics=True)
    atomic_json(args.payload_output, payload_audit)
    passed = (passed_payload and all(r["passed"] for r in protected_checks)
              and head_before == head_after == start["git_head"])
    audit = dict(schema="permworld-answer-control-protection-end/v1",
        status="passed" if passed else "failed", started_utc=begun, completed_utc=utc(),
        protection_start_sha256=EXPECTED_START_SHA, source_inventory_sha256=EXPECTED_INVENTORY_SHA,
        protection_start_captured_utc=start["captured_utc"],
        scope="Comparison of the 1,449 explicitly named start SHA values with endpoint reads, plus listed activation-payload identity checks. It does not establish continuous immutability or cover every repository/F17 file.",
        protected_file_count=len(protected_checks), protected_files_passed=sum(r["passed"] for r in protected_checks),
        protected_bytes_hashed=sum(r.get("size_bytes", 0) for r in protected_checks),
        git_head_start=start["git_head"], git_head_audit_before=head_before, git_head_audit_after=head_after,
        git_head_matches_start=head_before == head_after == start["git_head"],
        worktree_status_start=start["worktree_status"], worktree_status_end=git("status", "--short"),
        worktree_status_scope="Recorded for review; new diagnostic files are expected. Equality is not required and no repository-wide absence-of-changes claim is made.",
        unique_files_hashed=len(checks), unique_bytes_hashed=sum(r.get("size_bytes", 0) for r in checks.values()),
        activation_payload_audit_path=str(args.payload_output.resolve()),
        activation_payload_audit_sha256=digest(args.payload_output), activation_payload_status=payload_audit["status"],
        endpoint_only=True, no_source_files_written=True, no_model_or_metric_code_imported=True,
        elapsed_seconds=time.monotonic()-timer, file_checks=protected_checks)
    atomic_json(args.output, audit)
    print(json.dumps(dict(status=audit["status"], protection_audit=str(args.output),
                          activation_payload_audit=str(args.payload_output),
                          protected_files_passed=audit["protected_files_passed"],
                          logical_activation_records_passed=payload_audit["logical_records_passed"],
                          unique_bytes_hashed=audit["unique_bytes_hashed"]), indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
