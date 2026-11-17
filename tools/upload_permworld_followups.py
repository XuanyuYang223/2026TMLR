#!/usr/bin/env python3
"""Upload immutable packages concurrently to the existing bounded draft.

Never creates, deletes, clobbers or publishes assets/releases, or moves Git refs.
Only atomically finalized representation-*.tar.zst files are eligible. Provider
confirmation is separate from the packer's complete local stream verification;
this uploader neither performs nor asserts that scientific/package verification.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import threading
import time

REPO = "XuanyuYang223/2026TMLR"
TAG = "permworld-followups-2026-11-16"
RELEASE_ID = 447841612
ROOT = Path(__file__).resolve().parents[1]
FINAL_MANIFEST = ROOT / "artifacts/permworld_followups_20261116/manifest.json"


def gh(*args):
    process = subprocess.run(["gh", *args], cwd=ROOT, capture_output=True, text=True)
    if process.returncode:
        raise RuntimeError(process.stderr.strip()[-2000:])
    return process.stdout


def get_draft(release_id):
    """Read authenticated, stable-ID metadata and every current asset page."""
    release = json.loads(gh("api", f"repos/{REPO}/releases/{release_id}"))
    if release.get("id") != release_id or release.get("tag_name") != TAG or not release.get("draft"):
        raise ValueError("Expected the exact existing bounded draft; no mutation attempted")
    assets = []
    page = 1
    while True:
        batch = json.loads(gh("api", f"repos/{REPO}/releases/{release_id}/assets?per_page=100&page={page}"))
        if not isinstance(batch, list):
            raise ValueError("Unexpected authenticated asset inventory")
        assets.extend(batch)
        if len(batch) < 100:
            break
        page += 1
    names = [asset["name"] for asset in assets]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate remote asset names; all originals retained")
    return release, {asset["name"]: asset for asset in assets}


def refresh_draft(release_id, retries=3):
    for attempt in range(1, retries + 1):
        try:
            return get_draft(release_id)
        except RuntimeError:
            if attempt == retries:
                raise
            time.sleep(min(2 ** attempt, 8))
    raise AssertionError("Unreachable refresh retry exit")


def validate_asset(asset, name, size, sha):
    if (asset.get("name") != name or asset.get("state") != "uploaded"
            or asset.get("size") != size or asset.get("digest") != "sha256:" + sha):
        raise ValueError("Existing/partial asset lacks matching size, SHA and uploaded state: " + name)


def process_alive(pid):
    """A zombie has finished uploading even while its parent has not reaped it."""
    if pid is None:
        return False
    try:
        status = Path(f"/proc/{pid}/status").read_text()
    except FileNotFoundError:
        return False
    state = next((line.split()[1] for line in status.splitlines() if line.startswith("State:")), None)
    if state is None:
        raise ValueError("Cannot establish inherited uploader process state")
    return state != "Z"


def local_identity(path):
    if path.is_symlink() or not path.is_file():
        raise ValueError("Package must be a finalized regular file: " + path.name)
    before = path.stat()
    with path.open("rb") as stream:
        sha = hashlib.file_digest(stream, "sha256").hexdigest()
    after = path.stat()
    signature = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
    if signature(before) != signature(after):
        raise ValueError("Package changed while hashing; retained: " + path.name)
    return before.st_size, sha, signature(before)


class Progress:
    def __init__(self, path):
        self.path = path
        self.lock = threading.Lock()

    def write(self, event, **fields):
        row = dict(event=event, utc=datetime.now(timezone.utc).isoformat(), **fields)
        with self.lock:
            with self.path.open("a") as handle:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
        return row


def upload_one(path, release_id, progress, retries=3):
    started = time.monotonic()
    size, sha, stat_signature = local_identity(path)
    progress.write("asset_claim", name=path.name, bytes=size, sha256=sha)
    # No stale shared cache: re-read this draft before every claim/resume/retry.
    for attempt in range(1, retries + 1):
        _, assets = refresh_draft(release_id)
        existing = assets.get(path.name)
        if existing is not None:
            validate_asset(existing, path.name, size, sha)
            progress.write("asset_confirmed", name=path.name, bytes=size, sha256=sha,
                           status="provider_confirmed_existing", attempt=attempt,
                           elapsed_seconds=time.monotonic() - started)
            return dict(name=path.name, bytes=size, sha256=sha)
        now = path.stat()
        current = (now.st_dev, now.st_ino, now.st_size, now.st_mtime_ns, now.st_ctime_ns)
        if current != stat_signature:
            raise ValueError("Package changed after hashing: " + path.name)
        progress.write("upload_started", name=path.name, attempt=attempt, bytes=size)
        try:
            # Deliberately no --clobber. gh resolves the authenticated draft.
            gh("release", "upload", TAG, str(path), "--repo", REPO)
        except RuntimeError as error:
            progress.write("upload_attempt_failed", name=path.name, attempt=attempt, error=str(error))
            # A lost response may follow a successful upload. Confirm before a
            # retry; any starter/partial/different asset is retained and fails.
            _, assets = refresh_draft(release_id)
            existing = assets.get(path.name)
            if existing is not None:
                validate_asset(existing, path.name, size, sha)
                progress.write("asset_confirmed", name=path.name, bytes=size, sha256=sha,
                               status="provider_confirmed_after_network_error", attempt=attempt,
                               elapsed_seconds=time.monotonic() - started)
                return dict(name=path.name, bytes=size, sha256=sha)
            if attempt == retries:
                raise
            time.sleep(min(2 ** attempt, 8))
            continue
        _, assets = refresh_draft(release_id)
        existing = assets.get(path.name)
        if existing is None:
            raise ValueError("Upload returned without a provider asset: " + path.name)
        validate_asset(existing, path.name, size, sha)
        progress.write("asset_confirmed", name=path.name, bytes=size, sha256=sha,
                       status="uploaded_provider_size_digest_verified", attempt=attempt,
                       elapsed_seconds=time.monotonic() - started)
        return dict(name=path.name, bytes=size, sha256=sha)
    raise AssertionError("Unreachable retry exit")


def read_final_manifest(path):
    if not path.exists():
        return None
    try:
        manifest = json.loads(path.read_text())
    except json.JSONDecodeError:
        # write_once may have created the file milliseconds before completing
        # its bytes. It is not a completion signal until the JSON is readable.
        return None
    if manifest.get("repository") != REPO or manifest.get("release_tag") != TAG:
        raise ValueError("Unexpected final manifest repository/release")
    assets = manifest["assets"]
    result = {a["name"]: a for a in assets}
    if len(result) != len(assets) or not result:
        raise ValueError("Empty/repeated final package names")
    return result


def run(args):
    args.packages.mkdir(parents=True, exist_ok=True)
    progress = Progress(args.packages / "upload_progress.jsonl")
    refresh_draft(args.release_id)
    progress.write("manager_started", workers=args.workers, release_id=args.release_id,
                   inflight_pid=args.inflight_pid, inflight_name=args.inflight_name,
                   local_full_stream_verification_asserted=False)
    confirmed, failures, active = {}, {}, {}
    inflight = []
    if args.inflight_pid:
        inflight.append({"pid": args.inflight_pid, "name": args.inflight_name})
    if getattr(args, "inflight_file", None):
        extra = json.loads(args.inflight_file.read_text())
        if not isinstance(extra, list):
            raise ValueError("Inflight file must contain a list")
        for item in extra:
            if not isinstance(item.get("pid"), int) or item["pid"] <= 0:
                raise ValueError("Invalid inherited PID")
            name = item.get("name", "")
            if Path(name).name != name or not name.startswith("representation-") or not name.endswith(".tar.zst"):
                raise ValueError("Invalid inherited package name")
            inflight.append(item)
    inflight_logged = set()
    with ThreadPoolExecutor(max_workers=args.workers, thread_name_prefix="draft-asset") as pool:
        while True:
            for future, path in list(active.items()):
                if not future.done():
                    continue
                del active[future]
                try:
                    record = future.result()
                    confirmed[path.name] = record
                    print(json.dumps(dict(status="draft_asset_ready", **record, count=len(confirmed))), flush=True)
                except Exception as error:
                    failures[path.name] = str(error)
                    progress.write("asset_failed", name=path.name, error=str(error))
                    print(json.dumps(dict(status="asset_failed_retained", name=path.name, error=str(error))), flush=True)
            if failures:
                if active:
                    time.sleep(args.poll_seconds)
                    continue
                raise RuntimeError("Uploader stopped; failed assets retained: " + ", ".join(sorted(failures)))
            inherited_names = {x["name"] for x in inflight if process_alive(x["pid"])}
            for name in inherited_names - inflight_logged:
                progress.write("inherited_upload_wait", name=name)
            for name in inflight_logged - inherited_names:
                progress.write("inherited_upload_finished", name=name)
            inflight_logged = inherited_names
            final = read_final_manifest(args.manifest)
            active_names = {path.name for path in active.values()}
            for path in sorted(args.packages.glob("representation-*.tar.zst")):
                if len(active) >= args.workers:
                    break
                if path.name in confirmed or path.name in active_names:
                    continue
                if path.name in inherited_names:
                    continue
                if final is not None and path.name not in final:
                    raise ValueError("Local package is absent from final manifest: " + path.name)
                active[pool.submit(upload_one, path, args.release_id, progress)] = path
            if final is not None and not active and set(final).issubset(confirmed):
                _, assets = refresh_draft(args.release_id)
                for name, expected in final.items():
                    local = confirmed[name]
                    if local["bytes"] != expected["bytes"] or local["sha256"] != expected["sha256"]:
                        raise ValueError("Final manifest differs from uploaded package: " + name)
                    remote = assets.get(name)
                    if remote is None:
                        raise ValueError("Final provider inventory lost an asset: " + name)
                    validate_asset(remote, name, expected["bytes"], expected["sha256"])
                progress.write("all_manifest_packages_provider_confirmed", count=len(final),
                               local_full_stream_verification_asserted=False)
                print(json.dumps(dict(status="all_manifest_packages_uploaded_to_draft", count=len(final),
                                      local_full_stream_verification_asserted=False)), flush=True)
                return
            time.sleep(args.poll_seconds)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--packages", type=Path, default=ROOT / ".publication/permworld_followups_20261116")
    ap.add_argument("--poll-seconds", type=float, default=3)
    ap.add_argument("--workers", type=int, default=3, help="Use 1 to retain serial scheduling")
    ap.add_argument("--release-id", type=int, default=RELEASE_ID)
    ap.add_argument("--manifest", type=Path, default=FINAL_MANIFEST)
    ap.add_argument("--inflight-pid", type=int)
    ap.add_argument("--inflight-name")
    ap.add_argument("--inflight-file", type=Path, help="JSON list of active PID/name bindings adopted without interrupting their uploads")
    args = ap.parse_args()
    if args.workers < 1 or args.poll_seconds <= 0:
        ap.error("Positive workers and poll seconds required")
    if bool(args.inflight_pid) != bool(args.inflight_name):
        ap.error("Supply both --inflight-pid and --inflight-name")
    if args.inflight_name and (Path(args.inflight_name).name != args.inflight_name
                              or not args.inflight_name.startswith("representation-")
                              or not args.inflight_name.endswith(".tar.zst")):
        ap.error("Invalid inflight package basename")
    run(args)


if __name__ == "__main__":
    main()
