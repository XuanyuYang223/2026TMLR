#!/usr/bin/env python3
"""Package saved Nov-10 results without changing or recomputing science.

Creates a separate release manifest; the original publication manifests and
frozen experiment receipts are never rewritten. No remote mutations here.
"""
from __future__ import annotations

import argparse
import collections
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
STAMP = "20261116"
TAG = "permworld-followups-2026-11-16"
REPO = "XuanyuYang223/2026TMLR"
META = ROOT / "artifacts" / f"permworld_followups_{STAMP}"
GROUPS = (
    "permworld_orthogonal_alignment_20261110",
    "permworld_exceedance_alignment_20261110",
    "permworld_lemma_composition_20261110",
    "permworld_local_causal_20261110",
    "permworld_local_timing_20261110",
    "permworld_closing_audit_20261110",
)
GIT_CAP = 8 * 1024**2  # Publication placement only, not a scientific cutoff.
FROOT = Path("/mnt/yangx-projects-f/20261110")


def canonical(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                       separators=(",", ":")) + "\n").encode()


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_once(path, value):
    data = canonical(value)
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError(f"Existing metadata differs: {path}; originals retained")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as handle:
            handle.write(data)


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT)


def rows(value):
    if isinstance(value, dict):
        return [(key, row) for key, row in value.items()]
    return [(row["path"], row) for row in value]


def prepare():
    META.mkdir(parents=True, exist_ok=True)
    candidates, exclusions, manifest_bindings = {}, [], []

    def add(path, group, expected=None):
        path = Path(path)
        if not path.is_absolute():
            path = ROOT / path
        real = path.resolve(strict=True)
        if not real.is_file():
            raise ValueError(f"Not a regular resolved input: {path}")
        size = real.stat().st_size
        if expected:
            if expected.get("size_bytes", size) != size:
                raise ValueError(f"Recorded size differs: {path}")
        key = str(path)
        sha = expected.get("sha256") if expected else None
        if key in candidates:
            old = candidates[key]
            if sha and old.get("recorded_sha256") not in (None, sha):
                raise ValueError(f"Conflicting recorded SHA: {path}")
            old["groups"].add(group)
            if sha:
                old["recorded_sha256"] = sha
            return
        if path.is_relative_to(ROOT):
            relative = path.relative_to(ROOT).as_posix()
            restore = "repo/" + relative
            git_included = not path.is_symlink() and size <= GIT_CAP
        elif real.is_relative_to(FROOT):
            relative = None
            restore = "external/20261110/" + real.relative_to(FROOT).as_posix()
            git_included = False
        else:
            raise ValueError(f"Out-of-scope source: {path}")
        candidates[key] = dict(
            logical_path=key, physical_path=str(real), repo_path=relative,
            restore_path=restore, size_bytes=size, recorded_sha256=sha,
            symlink_target=str(path.readlink()) if path.is_symlink() else None,
            git_included=git_included, groups={group},
        )

    for group in GROUPS:
        path = ROOT / "diagnostics" / group / "delivery_manifest.json"
        body = json.loads(path.read_text())
        manifest_bindings.append(dict(path=str(path.relative_to(ROOT)), sha256=digest(path)))
        for key, row in rows(body.get("files", body.get("inventory", []))):
            source = row.get("path", key)
            add(source, group, row)
        for field in ("recurrent_states", "token_state_shards"):
            for key, row in rows(body.get(field, {})):
                path = Path(row.get("path", key))
                exclusions.append(dict(path=str(path), size_bytes=row["size_bytes"],
                    recorded_sha256=row.get("sha256"), group=group, role=field,
                    reason="Intermediate input/state cache, excluded from this result release; retained locally",
                    fresh_hash_computed=False, exists_at_inventory=path.is_file()))

    # The old alignment manifest lists repository outputs only. Add its actual
    # fitted matrices/permutations and validation-format adapters explicitly.
    ortho = FROOT / "permworld_orthogonal_alignment"
    for sub in ("map_audits", "validation"):
        for path in sorted((ortho / sub).rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts:
                add(path, GROUPS[0])

    # Capture every not-yet-published repository file in the six bounded bundles,
    # including final self-excluded completions, scripts, figures and eight tests.
    untracked = [x.decode() for x in git("ls-files", "--others", "--exclude-standard", "-z").split(b"\0") if x]
    for rel in untracked:
        p = Path(rel)
        group = p.parts[1] if len(p.parts) > 1 else ""
        if p.parts[0] in ("diagnostics", "experiments") and group in GROUPS:
            add(ROOT / p, group)
        elif p.parts[0] == "tests" and p.name.startswith("test_permworld_"):
            add(ROOT / p, "permworld-tests")

    physical_hashes = {}
    inventory = []
    for item in sorted(candidates.values(), key=lambda x: x["logical_path"]):
        real = item["physical_path"]
        # Recorded hashes are checked against the bytes actually packed, then
        # again by the complete archive-stream verifier. Fresh hash here is only
        # needed when no prior receipt supplies one.
        sha = item["recorded_sha256"]
        if sha is None:
            if real not in physical_hashes:
                physical_hashes[real] = digest(real)
            sha = physical_hashes[real]
        previous = physical_hashes.get(real)
        if previous is not None and previous != sha:
            raise ValueError(f"Conflicting physical SHA: {real}")
        physical_hashes[real] = sha
        item["sha256"] = sha
        item["groups"] = sorted(item["groups"])
        inventory.append(item)

    merged = {}
    for item in inventory:
        sha = item["sha256"]
        if sha not in merged:
            merged[sha] = dict(sha256=sha, size_bytes=item["size_bytes"],
                source_realpath=item["physical_path"], object_member="objects/"+sha,
                logical_paths=[], experiments=[], roles=["saved_result_or_reproduction_source"],
                package_group=item["groups"][0].replace("_", "-"), release_object_required=True)
        obj = merged[sha]
        if obj["size_bytes"] != item["size_bytes"]:
            raise ValueError("SHA byte-count conflict")
        obj["logical_paths"].append(item["logical_path"])
        obj["experiments"] = sorted(set(obj["experiments"]) | set(item["groups"]))

    before = dict(source_commit=git("rev-parse", "HEAD").decode().strip(),
        status=git("status", "--porcelain=v1").decode(),
        original_manifest_bindings=manifest_bindings,
        scope="Six Nov-10 PermWorld result bundles; unrelated projects and existing manuscripts/releases excluded")
    write_once(META / "source_snapshot.json", before)
    write_once(META / "file_inventory.json", dict(files=inventory, git_file_cap_bytes=GIT_CAP))
    write_once(META / "objects_inventory.json", dict(objects=list(merged.values())))
    write_once(META / "excluded_intermediate_caches.json", dict(
        files=exclusions, count=len(exclusions), bytes=sum(x["size_bytes"] for x in exclusions),
        note="Not uploaded; prior checkpoints/input activations and third-party dependencies are also not republished. Historical absolute paths are provenance."))
    git_files = sorted(x["repo_path"] for x in inventory if x["git_included"])
    (META / "git_files.nul").write_bytes(b"\0".join(x.encode() for x in git_files)+b"\0")
    large = [x for x in inventory if x["repo_path"] and not x["git_included"]]
    (META / "release_only_repo_files.nul").write_bytes(b"\0".join(x["repo_path"].encode() for x in large)+b"\0")
    print(json.dumps(dict(logical_files=len(inventory), unique_objects=len(merged),
        unique_bytes=sum(x["size_bytes"] for x in merged.values()), git_files=len(git_files),
        release_only_repo_files=len(large), excluded_caches=len(exclusions),
        excluded_bytes=sum(x["size_bytes"] for x in exclusions))), flush=True)


def pack(output):
    spec = importlib.util.spec_from_file_location("saved_artifact_packer", ROOT/"tools/pack_artifacts.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.RELEASE_BASE = f"https://github.com/{REPO}/releases/download/{TAG}/"
    manifest, _ = mod.pack_inventory(META/"objects_inventory.json", output)
    audit = mod.verify_local(output/"package_manifest.json", output/"local_package_verification.json")
    data = json.loads((META/"file_inventory.json").read_text())["files"]
    manifest.update(repository=REPO, release_tag=TAG,
        scientific_source_commit=json.loads((META/"source_snapshot.json").read_text())["source_commit"],
        files=[dict(logical_path=x["logical_path"],sha256=x["sha256"],bytes=x["size_bytes"],
                    restore_path=x["restore_path"],experiments=x["groups"],
                    roles=["saved_result_or_reproduction_source"]) for x in data])
    all_packages = [x["package_id"] for x in manifest["packages"]]
    smallest = min(manifest["assets"], key=lambda x:x["bytes"])["package_id"]
    manifest["profiles"] = dict(minimal=[smallest], full=all_packages)
    manifest["publication_scope"] = (
        "All recorded result/probe/map/score/intervention/gradient outputs in six bundles; "
        "raw intermediate recurrent/token-state input caches and preexisting model/activation dependencies excluded")
    sys.path.insert(0,str(ROOT/"tools"))
    import download_artifacts
    download_artifacts.validate_manifest(manifest)
    write_once(META/"manifest.json",manifest)
    write_once(META/"local_package_verification.json",audit)
    write_once(META/"packaging_audit.json",json.loads((output/"packaging_audit.json").read_text()))
    print(json.dumps(dict(status=audit["status"],packages=len(all_packages),
        compressed_bytes=sum(x["bytes"] for x in manifest["assets"]))),flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command",choices=("prepare","pack"))
    ap.add_argument("--output",type=Path,default=ROOT/".publication"/f"permworld_followups_{STAMP}")
    args=ap.parse_args()
    if args.command=="prepare": prepare()
    else:pack(args.output)


if __name__=="__main__":
    main()
