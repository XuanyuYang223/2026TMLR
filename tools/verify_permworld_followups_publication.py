#!/usr/bin/env python3
"""Read-only public-delivery audit for the six completed PermWorld follow-ups.

Run only after the release is published and the source commit is on main. This
checks publication bytes and provenance, never scientific results or models.
Remote content readback is deliberately bounded: the minimal artifact profile
and the complete reader ZIP. Other assets receive provider-metadata checks.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
META_REL = Path("artifacts/permworld_followups_20261116")
REPOSITORY = "XuanyuYang223/2026TMLR"
TAG = "permworld-followups-2026-11-16"
RELEASE_ID = 447841612
SCIENTIFIC_BASE = "6d745815c0a1a2a41ad5fc8c0aa6e018f9bab249"
BLOCK = 1024 * 1024
HEX40 = re.compile(r"^[0-9a-f]{40}$")


class VerificationError(Exception):
    """Controlled errors contain no credentials or redirected URLs."""


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def safe_relative(name):
    if (not isinstance(name, str) or not name or "\\" in name
            or PurePosixPath(name).is_absolute()
            or any(piece in ("", ".", "..") for piece in name.split("/"))):
        raise VerificationError("Unsafe publication-relative path")
    return name


def command(argv, *, cwd=ROOT):
    p = subprocess.run(argv, cwd=cwd, capture_output=True)
    if p.returncode:
        # In particular, do not expose gh stderr or an authenticated URL.
        raise VerificationError("Command failed: " + argv[0] + " (exit " + str(p.returncode) + ")")
    return p.stdout


def gh_json(endpoint):
    try:
        return json.loads(command(["gh", "api", endpoint]))
    except json.JSONDecodeError:
        raise VerificationError("GitHub metadata response was not JSON") from None


def release_assets(release_id):
    assets = []
    for page in range(1, 11):
        rows = gh_json(f"repos/{REPOSITORY}/releases/{release_id}/assets?per_page=100&page={page}")
        if not isinstance(rows, list):
            raise VerificationError("Invalid release asset page")
        assets.extend(rows)
        if len(rows) < 100:
            break
    else:
        raise VerificationError("Release asset inventory exceeded the bounded ten-page audit")
    if len({x["name"] for x in assets}) != len(assets):
        raise VerificationError("Repeated release asset name")
    return assets


def asset_metadata(asset):
    return {key: asset.get(key) for key in ("id", "name", "size", "digest", "state")}


def old_release_metadata(old):
    fresh = gh_json(f"repos/{REPOSITORY}/releases/{old['id']}")
    out = {key: fresh.get(key) for key in
           ("id", "tag_name", "draft", "prerelease", "target_commitish", "published_at")}
    out["assets"] = sorted((asset_metadata(x) for x in release_assets(old["id"])), key=lambda x: x["id"])
    return out


class Audit:
    def __init__(self, expected_commit):
        self.value = dict(status="running", expected_source_commit=expected_commit,
                          scientific_base_commit=SCIENTIFIC_BASE,
                          started_utc=datetime.now(timezone.utc).isoformat(),
                          checks=[], stages={}, scientific_calculations=0,
                          training_updates=0, model_or_probe_calls=0,
                          remote_mutations=0, credentials_or_redirect_urls_recorded=False)

    def check(self, condition, name, **fields):
        self.value["checks"].append(dict(check=name, passed=bool(condition), **fields))
        return bool(condition)

    def require(self, condition, name, **fields):
        if not self.check(condition, name, **fields):
            raise VerificationError(name)

    def stage(self, name, function):
        start = time.monotonic()
        try:
            result = function()
            self.value["stages"][name] = dict(status="passed", elapsed_seconds=time.monotonic()-start, result=result)
            return result
        except Exception as exc:
            message = str(exc) if isinstance(exc, VerificationError) else "Engineering failure (" + type(exc).__name__ + ")"
            self.value["stages"][name] = dict(status="failed", elapsed_seconds=time.monotonic()-start, error=message)
            self.check(False, "stage_" + name, error=message)
            return None


def read_inputs(audit):
    names = ("manifest.json", "file_inventory.json", "reader_archive.json",
             "older_releases_before.json", "source_snapshot.json")
    values, bindings = {}, []
    for name in names:
        path = ROOT / META_REL / name
        audit.require(path.is_file() and not path.is_symlink(), "regular_publication_input", path=str(META_REL/name))
        values[name] = json.loads(path.read_text())
        bindings.append(dict(path=str(META_REL/name), bytes=path.stat().st_size, sha256=sha(path)))
    manifest = values["manifest.json"]
    audit.require(manifest.get("repository") == REPOSITORY and manifest.get("release_tag") == TAG,
                  "manifest_repository_and_release")
    audit.require(manifest.get("scientific_source_commit") == SCIENTIFIC_BASE,
                  "analysis_time_base_Git_HEAD_preserved")
    audit.require(manifest.get("source_files_commit") == "4eeeacdfef526c5faa8be67b5212446d5f95f85e",
                  "formerly_uncommitted_source_bytes_have_separate_publication_commit")
    audit.value["input_bindings"] = bindings
    return values


def verify_new_release(audit, values):
    release = gh_json(f"repos/{REPOSITORY}/releases/{RELEASE_ID}")
    audit.require(release.get("id") == RELEASE_ID and release.get("tag_name") == TAG,
                  "stable_release_identity")
    audit.require(release.get("draft") is False and bool(release.get("published_at")), "release_is_published")
    manifest = values["manifest.json"]
    expected = {a["name"]: dict(bytes=a["bytes"], sha256=a["sha256"]) for a in manifest["assets"]}
    audit.require(len(expected) == len(manifest["assets"]), "manifest_asset_names_unique")
    reader = values["reader_archive.json"]
    audit.require(reader["name"] not in expected, "reader_name_distinct_from_packages")
    expected[reader["name"]] = dict(bytes=reader["bytes"], sha256=reader["sha256"])
    path = ROOT / META_REL / "manifest.json"
    expected["manifest.json"] = dict(bytes=path.stat().st_size, sha256=sha(path))
    actual = {a["name"]: a for a in release_assets(RELEASE_ID)}
    audit.require(set(actual) == set(expected), "exact_expected_release_asset_inventory",
                  expected=len(expected), actual=len(actual), missing=sorted(set(expected)-set(actual)),
                  unexpected=sorted(set(actual)-set(expected)))
    comparisons = []
    for name, record in sorted(expected.items()):
        asset = actual[name]
        passed = (asset.get("state") == "uploaded" and asset.get("size") == record["bytes"]
                  and asset.get("digest") == "sha256:" + record["sha256"])
        audit.check(passed, "release_provider_size_digest_state", asset_name=name)
        comparisons.append(dict(**asset_metadata(asset), expected_bytes=record["bytes"],
                                expected_sha256=record["sha256"], passed=passed))
    return dict(release_id=RELEASE_ID, tag=TAG, published_at=release["published_at"], assets=comparisons,
                scope="Provider inventory, size, SHA256 digest and uploaded state for every expected asset; not a download of all assets.")


def verify_older_releases(audit, values):
    snapshots = values["older_releases_before.json"]
    audit.require(len(snapshots) == 4 and sum(len(x["assets"]) for x in snapshots) == 117,
                  "four_older_releases_117_asset_snapshot")
    comparisons = []
    for old in snapshots:
        fresh = old_release_metadata(old)
        expected = dict(old)
        expected["assets"] = sorted(old["assets"], key=lambda x: x["id"])
        passed = fresh == expected
        audit.check(passed, "older_release_metadata_unchanged", release_id=old["id"], tag=old["tag_name"])
        comparisons.append(dict(release_id=old["id"], tag=old["tag_name"], assets=len(fresh["assets"]),
                                passed=passed, current=fresh))
    return dict(releases=comparisons,
                scope="Exact named release fields and all 117 asset IDs/names/sizes/provider digests/states versus the saved snapshot; no old asset download.")


def verify_source_manifests(audit, values):
    records = values["source_snapshot.json"]["original_manifest_bindings"]
    audit.require(len(records) == 6, "six_original_delivery_manifests")
    result = []
    for row in records:
        path = ROOT / safe_relative(row["path"])
        exists = path.is_file() and not path.is_symlink()
        actual = sha(path) if exists else None
        passed = exists and actual == row["sha256"]
        audit.check(passed, "original_delivery_manifest_SHA_unchanged", path=row["path"])
        result.append(dict(path=row["path"], expected_sha256=row["sha256"], current_sha256=actual, passed=passed))
    return dict(files=result, scope="Six explicitly named original delivery manifests; not all experiment files or whole-disk monitoring.")


def source_rows(values):
    return [r for r in values["file_inventory.json"]["files"] if r["git_included"]]


def verify_clone(audit, values, work, expected_commit):
    clone = work / "clean_clone"
    if not clone.exists():
        command(["git", "clone", "--depth", "1", "--branch", "main",
                 f"https://github.com/{REPOSITORY}.git", str(clone)])
        origin = "fresh_shallow_clone_created_this_run"
    else:
        audit.require(clone.is_dir() and not clone.is_symlink() and (clone/".git").is_dir(), "existing_clean_clone_is_real_directory")
        origin = "precreated_clone_checked_this_run"
    head = command(["git", "rev-parse", "HEAD"], cwd=clone).decode().strip()
    audit.require(head == expected_commit, "clean_clone_expected_commit", actual=head, expected=expected_commit)
    shallow = command(["git", "rev-parse", "--is-shallow-repository"], cwd=clone).decode().strip()
    audit.require(shallow == "true", "clean_clone_is_shallow")
    tracked = command(["git", "status", "--porcelain=v1", "--untracked-files=no"], cwd=clone)
    audit.require(not tracked.strip(), "clone_tracked_worktree_clean")
    listing = command(["git", "ls-tree", "-r", "-z", "HEAD"], cwd=clone)
    modes = {}
    for entry in listing.split(b"\0"):
        if entry:
            meta, path = entry.split(b"\t", 1)
            modes[path.decode()] = meta.split()[0].decode()
    absolute_links = []
    for name, mode in modes.items():
        if mode == "120000":
            target = command(["git", "show", expected_commit + ":" + name], cwd=clone).decode().strip()
            if target.startswith("/"):
                absolute_links.append(name)
    audit.require(not absolute_links, "clone_has_no_absolute_symlinks", paths=absolute_links)
    rows = source_rows(values)
    audit.require(len(rows) == 623 and len({r["repo_path"] for r in rows}) == 623,
                  "exact_623_selected_source_files")
    checks = []
    for row in rows:
        rel = safe_relative(row["repo_path"])
        path = clone / rel
        regular = path.is_file() and not path.is_symlink()
        actual = sha(path) if regular else None
        passed = (regular and modes.get(rel) in ("100644", "100755")
                  and path.stat().st_size == row["size_bytes"] and actual == row["sha256"])
        audit.check(passed, "clean_clone_selected_file_SHA", path=rel)
        checks.append(dict(path=rel, sha256=actual, expected_sha256=row["sha256"], passed=passed))
    # The full immutable diff is evaluated in the original repository, which
    # already has both commits; the fresh clone need not fetch old history.
    diff = command(["git", "diff", "--name-status", "--no-renames", "-z", SCIENTIFIC_BASE, expected_commit])
    parts = diff.split(b"\0")
    if parts and not parts[-1]:
        parts.pop()
    audit.require(len(parts) % 2 == 0, "bounded_diff_name_status_shape")
    changes = []
    for index in range(0, len(parts), 2):
        kind, name = parts[index].decode(), parts[index+1].decode()
        passed = kind == "A" or (kind == "M" and name in ("README.md", "CURRENT_RESEARCH.txt"))
        audit.check(passed, "old_scientific_tree_change_scope", change=kind, path=name)
        changes.append(dict(change=kind, path=name, allowed=passed))
    return dict(path=str(clone.relative_to(ROOT)), origin=origin, commit=head,
                selected_file_checks=checks, scientific_base_diff=changes,
                scope="All 623 inventory-selected source/result files from the public shallow main clone; immutable base-to-source diff permits additions and only README/CURRENT_RESEARCH modifications.")


def readback_minimal(audit, values, work):
    module_path = ROOT / "tools/download_artifacts.py"
    spec = importlib.util.spec_from_file_location("publication_download_only", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = module.run(ROOT/META_REL/"manifest.json", "minimal", work/"readback/minimal_downloads",
                        work/"readback/minimal_restored", materialize=False, allow_local_fixture=False)
    audit.check(result.get("status") == "passed_selected_artifact_restore", "minimal_profile_public_restore",
                status=result.get("status"))
    return dict(downloader_sha256=sha(module_path), downloader_audit=result,
                scope="Only the manifest's minimal profile: asset/stream/object hashes and safe CAS restoration. Remaining package bytes were not remotely reread by this check.")


def download_reader(reader, work):
    expected_url = f"https://github.com/{REPOSITORY}/releases/download/{TAG}/{reader['name']}"
    if reader.get("url") != expected_url:
        raise VerificationError("Reader URL is not the fixed public release asset")
    directory = work / "readback/reader"
    directory.mkdir(parents=True, exist_ok=True)
    if directory.is_symlink():
        raise VerificationError("Reader readback directory must not be a symlink")
    destination = directory / reader["name"]
    if destination.is_symlink():
        raise VerificationError("Reader readback destination must not be a symlink")
    fd, temp = tempfile.mkstemp(prefix=".reader-network-", dir=directory)
    count, digest = 0, hashlib.sha256()
    try:
        request = urllib.request.Request(expected_url, headers={"User-Agent": "permworld-publication-verification/1"})
        try:
            response = urllib.request.urlopen(request, timeout=120)
        except urllib.error.HTTPError as exc:
            raise VerificationError("Public reader download HTTP status " + str(exc.code)) from None
        except Exception:
            raise VerificationError("Public reader download failed; redirected URLs are not logged") from None
        with os.fdopen(fd, "wb") as out, response:
            fd = None
            while block := response.read(BLOCK):
                count += len(block)
                if count > reader["bytes"]:
                    raise VerificationError("Reader download exceeds declared size")
                digest.update(block)
                out.write(block)
        if count != reader["bytes"] or digest.hexdigest() != reader["sha256"]:
            raise VerificationError("Reader network bytes/SHA256 mismatch")
        if destination.exists():
            if destination.stat().st_size != count or sha(destination) != digest.hexdigest():
                raise VerificationError("Different existing reader download retained")
        else:
            with Path(temp).open("rb") as source, destination.open("xb") as out:
                while block := source.read(BLOCK):
                    out.write(block)
        return destination, count, digest.hexdigest()
    finally:
        if fd is not None:
            os.close(fd)
        Path(temp).unlink(missing_ok=True)


def readback_reader(audit, values, work):
    reader = values["reader_archive.json"]
    path, count, digest = download_reader(reader, work)
    rows = source_rows(values)
    expected = {r["repo_path"]: r for r in rows}
    extra = str(META_REL/"README.md")
    checks = []
    with zipfile.ZipFile(path) as archive:
        entries = archive.infolist()
        names = [z.filename for z in entries]
        audit.require(len(names) == reader["members"] and len(set(names)) == len(names), "reader_member_count_and_uniqueness")
        audit.require(set(names) == set(expected) | {extra}, "reader_exact_source_and_index_members")
        for entry in entries:
            safe_relative(entry.filename)
            audit.require(not entry.is_dir() and not stat.S_ISLNK(entry.external_attr >> 16),
                          "reader_regular_member_no_symlink", path=entry.filename)
        audit.require(archive.testzip() is None, "reader_full_CRC_testzip")
        for name, row in sorted(expected.items()):
            entry = archive.getinfo(name)
            h = hashlib.sha256()
            with archive.open(entry) as stream:
                while block := stream.read(BLOCK):
                    h.update(block)
            actual = h.hexdigest()
            passed = entry.file_size == row["size_bytes"] and actual == row["sha256"]
            audit.check(passed, "reader_selected_member_SHA", path=name)
            checks.append(dict(path=name, expected_sha256=row["sha256"], sha256=actual, passed=passed))
    return dict(name=reader["name"], bytes_read=count, sha256=digest, network_read=True,
                checked_source_members=len(checks), member_checks=checks, additional_CRC_checked_index=extra,
                no_ZIP_paths_extracted=True,
                scope="Complete public reader ZIP network bytes, CRC for every member, and all 623 selected member SHA256s. The extra publication README is covered by ZIP digest/CRC, not the original scientific-file inventory.")


def input_end_check(audit):
    result = []
    for row in audit.value.get("input_bindings", []):
        path = ROOT / row["path"]
        actual = sha(path)
        passed = actual == row["sha256"] and path.stat().st_size == row["bytes"]
        audit.check(passed, "publication_input_unchanged_at_end", path=row["path"])
        result.append(dict(path=row["path"], sha256=actual, passed=passed))
    return result


def save_audit(audit):
    payload = (json.dumps(audit.value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    attempts = ROOT / META_REL / "public_verification_attempts"
    attempts.mkdir(parents=True, exist_ok=True)
    attempt = attempts / (stamp + ".json")
    with attempt.open("xb") as stream:
        stream.write(payload)
    destination = ROOT / META_REL / "public_verification.json"
    fd, temporary = tempfile.mkstemp(prefix=".public-verification-", dir=destination.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
        os.replace(temporary, destination)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return destination, attempt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-commit", required=True,
                        help="Exact already-pushed main source commit; never a mutable branch name")
    args = parser.parse_args()
    if HEX40.fullmatch(args.expected_commit) is None:
        parser.error("--expected-commit requires a lowercase forty-character Git commit SHA")
    start = time.monotonic()
    audit = Audit(args.expected_commit)
    audit.value["verifier_sha256"] = sha(Path(__file__))
    work = ROOT / ".publication/permworld_followups_20261116"
    work.mkdir(parents=True, exist_ok=True)
    if work.is_symlink():
        parser.error("Publication workspace must not be a symlink")
    values = audit.stage("publication_inputs", lambda: read_inputs(audit))
    if values is not None:
        audit.stage("published_release_provider_metadata", lambda: verify_new_release(audit, values))
        audit.stage("old_release_preservation", lambda: verify_older_releases(audit, values))
        audit.stage("six_original_manifest_preservation", lambda: verify_source_manifests(audit, values))
        audit.stage("clean_public_main_clone", lambda: verify_clone(audit, values, work, args.expected_commit))
        audit.stage("minimal_artifact_public_readback", lambda: readback_minimal(audit, values, work))
        audit.stage("reader_ZIP_public_readback", lambda: readback_reader(audit, values, work))
        audit.stage("publication_input_end_SHA", lambda: input_end_check(audit))
    failures = sum(not r["passed"] for r in audit.value["checks"])
    audit.value.update(status="passed_publication_verification" if not failures else "failed_publication_verification",
                       checks_count=len(audit.value["checks"]), failed_checks=failures,
                       completed_utc=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.monotonic()-start,
                       remote_byte_scope="All reader ZIP bytes and the minimal package profile only; all other assets have provider size/digest/state confirmation, not a full remote download.")
    output, attempt = save_audit(audit)
    print(json.dumps(dict(status=audit.value["status"], checks=audit.value["checks_count"],
                          failures=failures, output=str(output.relative_to(ROOT)),
                          immutable_attempt=str(attempt.relative_to(ROOT)))))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
