#!/usr/bin/env python3
"""Verify multipart release assets and restore immutable content-addressed files.

Engineering only: no scientific calculations, experiment imports or forwards.
Requirements: standard library plus zstandard==0.25.0.
"""
from __future__ import annotations
import argparse
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import tarfile
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request

HEX = re.compile(r'^[0-9a-f]{64}$')
CHUNK = 1024 * 1024


class ArtifactError(Exception):
    """Only controlled, URL-free error messages may be exposed."""


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(CHUNK), b''):
            h.update(chunk)
    return h.hexdigest()


def require(condition, message):
    if not condition:
        raise ArtifactError(message)


def safe_relative(value):
    require(isinstance(value, str) and value and '\\' not in value, 'Unsafe relative restoration path')
    p = PurePosixPath(value)
    require(not p.is_absolute() and all(x not in ('', '.', '..') for x in value.split('/')),
            'Absolute or traversing restoration path is forbidden')
    return p


def checked_root(value):
    path = Path(value).absolute()
    require(not path.is_symlink(), 'A download/restore root must not itself be a symlink')
    path.mkdir(parents=True, exist_ok=True)
    require(path.is_dir(), 'A download/restore root must be a directory')
    return path.resolve()


def safe_destination(root, relative, create_parent=True):
    relative = safe_relative(relative); root = Path(root).resolve()
    target = root.joinpath(*relative.parts)
    cursor = root
    for piece in relative.parts:
        cursor = cursor / piece
        require(not cursor.is_symlink(), 'Symlinks within the restoration/download tree are forbidden')
    require(target.resolve().is_relative_to(root), 'Restoration escaped its selected root')
    if create_parent:
        target.parent.mkdir(parents=True, exist_ok=True)
    return target


def matching_file(path, size, digest):
    path = Path(path)
    require(not path.is_symlink(), 'An existing target is a symlink')
    if not path.exists():
        return False
    require(path.is_file() and path.stat().st_size == size and sha(path) == digest,
            'Existing different file is retained; choose a new empty directory or resolve the collision')
    return True


def immutable_commit(temporary, destination, size, digest):
    """Publish with O_EXCL; no overwrite even if another process created it."""
    if matching_file(destination, size, digest):
        return False
    try:
        out = destination.open('xb')
    except FileExistsError:
        matching_file(destination, size, digest)
        return False
    try:
        with out, Path(temporary).open('rb') as src:
            shutil.copyfileobj(src, out, CHUNK)
    except Exception:
        # Only the new file created by this invocation is removed on failure.
        destination.unlink(missing_ok=True)
        raise
    require(destination.stat().st_size == size and sha(destination) == digest,
            'Committed object/download did not retain its verified bytes')
    return True


def validate_manifest(manifest):
    require(manifest.get('schema_version') == 1, 'Unsupported artifact manifest schema')
    for key in ('repository', 'release_tag', 'scientific_source_commit', 'assets', 'packages', 'objects', 'files', 'profiles'):
        require(key in manifest, 'Required artifact manifest field is missing')
    assets, packages, objects = {}, {}, {}
    for a in manifest['assets']:
        require(all(k in a for k in ('name', 'url', 'bytes', 'sha256', 'package_id', 'part_index')), 'Incomplete asset entry')
        require(isinstance(a['name'], str) and '/' not in a['name'] and '\\' not in a['name'] and a['name'] not in ('', '.', '..'), 'Unsafe asset name')
        require(a['name'] not in assets, 'Repeated asset name')
        require(HEX.fullmatch(a['sha256']) is not None and type(a['bytes']) is int and a['bytes'] > 0 and type(a['part_index']) is int and a['part_index'] >= 0,
                'Invalid asset size/hash/part index')
        u = urllib.parse.urlsplit(a['url'])
        require(u.username is None and u.password is None, 'Credential-bearing asset URL is forbidden')
        assets[a['name']] = a
    for p in manifest['packages']:
        require(all(k in p for k in ('package_id', 'parts', 'compressed_stream_sha256', 'object_sha256')), 'Incomplete package entry')
        require(p['package_id'] not in packages and p['parts'] and len(set(p['parts'])) == len(p['parts']), 'Invalid package part list')
        require(HEX.fullmatch(p['compressed_stream_sha256']) is not None, 'Invalid combined compressed-stream hash')
        require(all(n in assets and assets[n]['package_id'] == p['package_id'] for n in p['parts']), 'Package part references do not match assets')
        indexes = [assets[n]['part_index'] for n in p['parts']]
        require(indexes == sorted(set(indexes)), 'The explicit package parts are not in increasing unique part-index order')
        require(len(set(p['object_sha256'])) == len(p['object_sha256']), 'Repeated object within a package')
        packages[p['package_id']] = p
    for obj in manifest['objects']:
        require(all(k in obj for k in ('sha256', 'bytes', 'package_id', 'archive_member', 'experiments', 'roles')), 'Incomplete content object')
        digest = obj['sha256']
        require(HEX.fullmatch(digest) is not None and digest not in objects and type(obj['bytes']) is int and obj['bytes'] >= 0, 'Invalid/repeated object SHA or size')
        require(obj['package_id'] in packages and obj['archive_member'] == 'objects/' + digest, 'Unsafe or unexpected object archive member')
        objects[digest] = obj
    for key, p in packages.items():
        require(set(p['object_sha256']) == {s for s, o in objects.items() if o['package_id'] == key}, 'Package object list is incomplete or inconsistent')
    require(all(a['package_id'] in packages and a['name'] in packages[a['package_id']]['parts'] for a in assets.values()), 'Unreferenced or misassigned asset')
    logicals, restore_paths = {}, {}
    for f in manifest['files']:
        require(all(k in f for k in ('logical_path', 'sha256', 'bytes', 'restore_path', 'experiments', 'roles')), 'Incomplete logical file mapping')
        require(isinstance(f['logical_path'], str) and f['logical_path'] not in logicals, 'Repeated or invalid historical logical path')
        require(f['sha256'] in objects and type(f['bytes']) is int and f['bytes'] == objects[f['sha256']]['bytes'], 'Logical file does not match its content object')
        safe_relative(f['restore_path'])
        if f['restore_path'] in restore_paths:
            require(restore_paths[f['restore_path']] == f['sha256'], 'One materialized path maps to different content')
        logicals[f['logical_path']] = f; restore_paths[f['restore_path']] = f['sha256']
    for name in ('minimal', 'full'):
        require(name in manifest['profiles'] and isinstance(manifest['profiles'][name], list), 'Missing download profile')
        chosen = manifest['profiles'][name]
        require(len(set(chosen)) == len(chosen) and all(x in packages for x in chosen), 'Invalid/repeated profile package ID')
    return assets, packages, objects


def fetch_asset(asset, directory, audit, allow_local_fixture=False):
    target = safe_destination(directory, asset['name'])
    if matching_file(target, asset['bytes'], asset['sha256']):
        audit['assets'].append(dict(name=asset['name'], bytes=asset['bytes'], sha256=asset['sha256'],
                                    source='verified_download_cache', network_read=False, verified=True))
        return target
    url = urllib.parse.urlsplit(asset['url'])
    require(url.scheme in ('https', 'http') or (allow_local_fixture and url.scheme == 'file'),
            'Only HTTP(S) release downloads are allowed outside the synthetic fixture')
    if url.scheme == 'file':
        require(url.netloc in ('', 'localhost'), 'Synthetic local URL must be a local file')
    record = dict(name=asset['name'], bytes_expected=asset['bytes'], bytes_read=0,
                  source='local_synthetic_fixture' if url.scheme == 'file' else 'remote_network',
                  network_read=url.scheme != 'file', verified=False)
    audit['assets'].append(record)
    fd, temp = tempfile.mkstemp(prefix='.' + asset['name'] + '.partial-', dir=directory)
    temporary = Path(temp)
    try:
        request = urllib.request.Request(asset['url'], headers={'User-Agent': 'representation-artifacts/1'})
        try:
            response = urllib.request.urlopen(request, timeout=120)
        except urllib.error.HTTPError as exc:
            raise ArtifactError('Asset HTTP request failed (status ' + str(exc.code) + '); no redirect URL or credentials are logged') from None
        except Exception:
            raise ArtifactError('Asset network request failed; no redirect URL or credentials are logged') from None
        digest = hashlib.sha256()
        with os.fdopen(fd, 'wb') as out, response:
            fd = None
            while True:
                block = response.read(CHUNK)
                if not block: break
                record['bytes_read'] += len(block)
                require(record['bytes_read'] <= asset['bytes'], 'Asset exceeds its declared size')
                digest.update(block); out.write(block)
        require(record['bytes_read'] == asset['bytes'] and digest.hexdigest() == asset['sha256'], 'Downloaded asset bytes/SHA256 mismatch')
        immutable_commit(temporary, target, asset['bytes'], asset['sha256'])
        record.update(verified=True, sha256=asset['sha256'])
    finally:
        if fd is not None: os.close(fd)
        temporary.unlink(missing_ok=True)
    return target


class JoinedReader(io.RawIOBase):
    """Concatenates split parts without creating another full compressed file."""
    def __init__(self, paths):
        self.paths = iter(paths); self.current = None
    def readable(self): return True
    def read(self, size=-1):
        if size < 0:
            return b''.join(iter(lambda: self.read(CHUNK), b''))
        result = bytearray()
        while len(result) < size:
            if self.current is None:
                try: self.current = Path(next(self.paths)).open('rb')
                except StopIteration: break
            data = self.current.read(size-len(result))
            if data: result.extend(data)
            else: self.current.close(); self.current = None
        return bytes(result)
    def readinto(self, target):
        data = self.read(len(target)); target[:len(data)] = data; return len(data)
    def close(self):
        if self.current is not None: self.current.close(); self.current = None
        super().close()


def combined_sha(paths):
    digest = hashlib.sha256()
    for path in paths:
        with Path(path).open('rb') as stream:
            for chunk in iter(lambda: stream.read(CHUNK), b''): digest.update(chunk)
    return digest.hexdigest()


def extract_package(package, paths, object_table, restore, audit):
    import zstandard
    require(zstandard.__version__ == '0.25.0', 'Artifact decoder requires zstandard==0.25.0')
    expected = {s: object_table[s] for s in package['object_sha256']}
    existing = {s: matching_file(safe_destination(restore, 'objects/' + s), o['bytes'], s) for s, o in expected.items()}
    # Existing matching objects skip writes, not archive-safety checks: a newly
    # selected package cannot hide a dangerous member merely because every
    # expected CAS object is already present.
    found, index_seen = set(), False
    # read_across_frames handles both one compressed frame split at arbitrary
    # byte boundaries and an ordered stream of multiple valid Zstd frames.
    with JoinedReader(paths) as joined, zstandard.ZstdDecompressor().stream_reader(joined, read_across_frames=True) as decoded:
        with tarfile.open(fileobj=decoded, mode='r|', ignore_zeros=True, errorlevel=2) as archive:
            for member in archive:
                require(member.isreg() and member.type in (tarfile.REGTYPE, tarfile.AREGTYPE), 'Tar contains a nonregular entry; directories, links and devices are forbidden')
                name = member.name
                require('\\' not in name and name in ({'objects/' + s for s in expected} | {'PACKAGE_INDEX.json'}), 'Tar contains an unsafe or unexpected member')
                require(not member.pax_headers or all(k not in ('path', 'linkpath') or member.pax_headers[k] == name for k in member.pax_headers), 'Tar path override is inconsistent')
                source = archive.extractfile(member)
                require(source is not None, 'Tar regular member has no readable content')
                if name == 'PACKAGE_INDEX.json':
                    require(not index_seen and 0 <= member.size <= 16 * CHUNK, 'Repeated or oversized PACKAGE_INDEX')
                    blob = source.read(16 * CHUNK + 1)
                    require(len(blob) == member.size, 'Package index size differs from tar header')
                    json.loads(blob.decode('utf-8')); index_seen = True
                    continue
                digest = name.split('/')[1]; obj = expected[digest]
                require(digest not in found and member.size == obj['bytes'], 'Repeated content object or wrong tar member size')
                target = safe_destination(restore, name)
                fd, temp = (None, None) if existing[digest] else tempfile.mkstemp(prefix='.' + digest + '.partial-', dir=target.parent)
                h = hashlib.sha256(); count = 0
                try:
                    output = os.fdopen(fd, 'wb') if fd is not None else None
                    fd = None
                    try:
                        while True:
                            block = source.read(CHUNK)
                            if not block: break
                            count += len(block); require(count <= obj['bytes'], 'Object exceeds manifest size')
                            h.update(block)
                            if output is not None: output.write(block)
                    finally:
                        if output is not None: output.close()
                    require(count == obj['bytes'] and h.hexdigest() == digest, 'Content object bytes/SHA256 mismatch')
                    if temp is not None: immutable_commit(temp, target, obj['bytes'], digest)
                finally:
                    if fd is not None: os.close(fd)
                    if temp is not None: Path(temp).unlink(missing_ok=True)
                found.add(digest)
                audit['objects'].append(dict(sha256=digest, bytes=count, source='verified_object_cache' if existing[digest] else 'verified_tar_object', verified=True))
        # Consume the decoder to validate frame EOF/checksums; no new file is
        # accepted merely because the tar payload's first members looked valid.
        while decoded.read(CHUNK): pass
    require(index_seen and found == set(expected), 'Package lacks its index or one or more expected objects')
    audit['packages'].append(dict(package_id=package['package_id'], compressed_stream_verified=True,
                                  extraction='verified_regular_expected_members_only', objects=len(found), package_index_present=True))


def path_map_and_materialize(manifest, selected_objects, restore, materialize, audit):
    path = safe_destination(restore, 'path_map.json')
    prior = json.loads(path.read_text()) if path.exists() else {}
    require(isinstance(prior, dict) and all(isinstance(k, str) and isinstance(v, str) and re.fullmatch(r'objects/[0-9a-f]{64}', v) for k, v in prior.items()), 'Existing path map is invalid; no overwrite performed')
    mapping = dict(prior)
    for record in manifest['files']:
        digest = record['sha256']
        if digest not in selected_objects: continue
        object_relative = 'objects/' + digest
        require(record['logical_path'] not in mapping or mapping[record['logical_path']] == object_relative, 'Existing logical path maps to different content; no overwrite performed')
        mapping[record['logical_path']] = object_relative
        if materialize:
            source = safe_destination(restore, object_relative)
            target = safe_destination(restore, record['restore_path'])
            require(target != path and not record['restore_path'].startswith('objects/'), 'Materialized path collides with the content-addressed store or path map')
            reused = matching_file(target, record['bytes'], digest)
            if not reused:
                immutable_commit(source, target, record['bytes'], digest)
            audit['materialized'].append(dict(restore_path=record['restore_path'], sha256=digest, bytes=record['bytes'],
                                             reused_existing_identical=reused, regular_file=True, symlink=False))
    # Only append new, consistently mapped logical keys. Scientific files are
    # never overwritten; this small mapping is explicitly versioned metadata.
    if mapping != prior or not path.exists():
        fd, temporary = tempfile.mkstemp(prefix='.path_map-', dir=restore)
        try:
            with os.fdopen(fd, 'w') as f: json.dump(mapping, f, indent=2, sort_keys=True, ensure_ascii=False); f.write('\n')
            os.replace(temporary, path)
        finally: Path(temporary).unlink(missing_ok=True)
    audit['path_map'] = dict(path=str(path), sha256=sha(path), logical_paths=len(mapping), current_profile_paths=sum(f['sha256'] in selected_objects for f in manifest['files']),
                              semantics='Historical logical_path -> restore-root relative objects/<sha>; appended keys only, no changed existing mapping.')


def run(manifest_path, profile, directory, restore_dir, materialize=False, allow_local_fixture=False):
    start = time.monotonic()
    audit = {'status': 'running', 'manifest_sha256': sha(manifest_path), 'profile': profile,
             'assets': [], 'packages': [], 'objects': [], 'materialized': [],
             'source_model_or_metric_calls': 0, 'training_updates': 0, 'credentials_or_redirect_URLs_recorded': False,
             'materialize_requested': materialize, 'scientific_verification_status_not_modified': True}
    try:
        manifest = json.loads(Path(manifest_path).read_text())
        assets, packages, objects = validate_manifest(manifest)
        require(profile in ('minimal', 'full'), 'Unknown artifact profile')
        chosen = manifest['profiles'][profile]
        directory, restore = checked_root(directory), checked_root(restore_dir)
        selected_objects = {s for p in chosen for s in packages[p]['object_sha256']}
        audit['selected_package_ids'] = chosen
        audit['storage_estimate'] = dict(download_asset_bytes=sum(assets[n]['bytes'] for p in chosen for n in packages[p]['parts']),
                                        unique_object_bytes=sum(objects[s]['bytes'] for s in selected_objects),
                                        additional_materialized_logical_copy_bytes=sum(f['bytes'] for f in manifest['files'] if f['sha256'] in selected_objects) if materialize else 0,
                                        compressed_join_file_bytes=0,
                                        note='Upper bound counts regular copies separately; transient per-file verified staging may temporarily require another asset/object size. No hard links or symlinks are materialized.')
        for package_id in chosen:
            package = packages[package_id]
            paths = [fetch_asset(assets[n], directory, audit, allow_local_fixture) for n in package['parts']]
            require(combined_sha(paths) == package['compressed_stream_sha256'], 'Joined compressed-stream SHA256 mismatch')
            extract_package(package, paths, objects, restore, audit)
        path_map_and_materialize(manifest, selected_objects, restore, materialize, audit)
        audit.update(status='passed_selected_artifact_restore', selected_packages=len(chosen), selected_unique_objects=len(selected_objects),
                     remote_assets_read=[a['name'] for a in audit['assets'] if a['network_read']],
                     reused_download_assets=[a['name'] for a in audit['assets'] if a['source'] == 'verified_download_cache'],
                     remote_bytes_read=sum(a.get('bytes_read', 0) for a in audit['assets'] if a['network_read']),
                     scope='Size/SHA of selected downloaded or reused assets, ordered joined stream SHA, and selected content bytes. Remote-readback coverage is only remote_assets_read; cached local bytes and unpublished/unselected assets are not claimed remotely re-read.')
    except ArtifactError as exc:
        audit.update(status='failed_artifact_restore', error=str(exc))
    except Exception as exc:
        audit.update(status='failed_artifact_restore', error='Engineering failure (' + type(exc).__name__ + '); raw URL/redirect/error detail is not logged')
    audit['elapsed_seconds'] = time.monotonic() - start
    return audit


def self_test():
    import zstandard
    require(zstandard.__version__ == '0.25.0', 'Synthetic fixture requires zstandard==0.25.0')
    results = []
    with tempfile.TemporaryDirectory(prefix='representation-artifact-fixture-') as td:
        root = Path(td); blobs = (b'finite fixture bytes\x00\x01', b'unchanged scientific object\n' * 25)
        digests = [hashlib.sha256(b).hexdigest() for b in blobs]
        def make_manifest(name, malicious=None, multi_frame=False):
            raw = io.BytesIO()
            with tarfile.open(fileobj=raw, mode='w') as tar:
                for digest, blob in zip(digests, blobs):
                    member = tarfile.TarInfo('objects/' + digest); member.size = len(blob); tar.addfile(member, io.BytesIO(blob))
                index = json.dumps({'package_id': 'fixture', 'objects': digests}).encode()
                member = tarfile.TarInfo('PACKAGE_INDEX.json'); member.size = len(index); tar.addfile(member, io.BytesIO(index))
                if malicious:
                    member = tarfile.TarInfo(malicious)
                    if malicious == 'objects/link': member.type = tarfile.SYMTYPE; member.linkname = '../../outside'
                    else: member.size = 3
                    tar.addfile(member, None if member.issym() else io.BytesIO(b'bad'))
            data = raw.getvalue()
            compressed = (zstandard.ZstdCompressor().compress(data[:len(data)//2]) +
                          zstandard.ZstdCompressor().compress(data[len(data)//2:])) if multi_frame else zstandard.ZstdCompressor().compress(data)
            parts = [compressed[:len(compressed)//2], compressed[len(compressed)//2:]]
            entries = []
            for i, part in enumerate(parts):
                path = root / f'{name}.part{i}'; path.write_bytes(part)
                entries.append(dict(name=path.name, url=path.as_uri(), bytes=len(part), sha256=sha(path), package_id='fixture', part_index=i))
            manifest = dict(schema_version=1, repository='synthetic-fixture', release_tag='fixture-only', scientific_source_commit='not-a-scientific-run',
                assets=entries, packages=[dict(package_id='fixture', parts=[x['name'] for x in entries], compressed_stream_sha256=hashlib.sha256(compressed).hexdigest(), object_sha256=digests)],
                objects=[dict(sha256=s, bytes=len(b), package_id='fixture', archive_member='objects/' + s, experiments=['fixture'], roles=['fixture']) for s, b in zip(digests, blobs)],
                files=[dict(logical_path='/historical/fixture/' + str(i), sha256=s, bytes=len(b), restore_path='legacy/fixture/' + str(i), experiments=['fixture'], roles=['fixture']) for i, (s, b) in enumerate(zip(digests, blobs))],
                profiles={'minimal': ['fixture'], 'full': ['fixture']})
            path = root / (name + '.json'); path.write_text(json.dumps(manifest)); return path
        manifest = make_manifest('good')
        a = run(manifest, 'minimal', root / 'downloads', root / 'restored', True, True)
        require(a['status'].startswith('passed') and a['selected_unique_objects'] == 2, 'Correct multipart synthetic package failed')
        require(all((root / 'restored/legacy/fixture' / str(i)).read_bytes() == b for i, b in enumerate(blobs)), 'Materialized bytes changed')
        results.append({'check': 'correct_split_stream_object_SHA_regular_materialization', 'passed': True})
        again = run(manifest, 'minimal', root / 'downloads', root / 'restored', True, True)
        require(again['status'].startswith('passed') and len(again['reused_download_assets']) == 2 and all(x['source'] == 'verified_object_cache' for x in again['objects']), 'Repeated exact-cache reuse failed')
        results.append({'check': 'idempotent_download_and_object_cache_reuse', 'passed': True})
        wrong_asset = root / 'downloads' / 'good.part0'; wrong_asset.write_bytes(b'existing different asset retained')
        asset_collision = run(manifest, 'minimal', root / 'downloads', root / 'restored', False, True)
        require(asset_collision['status'].startswith('failed') and wrong_asset.read_bytes() == b'existing different asset retained', 'Existing different downloaded asset was overwritten')
        results.append({'check': 'different_existing_download_not_overwritten', 'passed': True})
        mapping_target = root / 'restored/legacy/fixture/0'; mapping_target.write_bytes(b'existing different materialization retained')
        materialized_collision = run(manifest, 'minimal', root / 'other-downloads', root / 'restored', True, True)
        require(materialized_collision['status'].startswith('failed') and mapping_target.read_bytes() == b'existing different materialization retained', 'Existing different materialization was overwritten')
        results.append({'check': 'different_materialized_file_not_overwritten', 'passed': True})
        multi = make_manifest('multiple-frames', multi_frame=True)
        multiple = run(multi, 'minimal', root / 'multiple-downloads', root / 'multiple-restored', False, True)
        require(multiple['status'].startswith('passed'), 'Ordered concatenated Zstd frames failed')
        results.append({'check': 'multiple_Zstd_frames_and_arbitrary_split_boundaries', 'passed': True})
        bad = root / 'restored/objects' / digests[0]; bad.write_bytes(b'leave this existing different file intact')
        # This download cache still matches. The earlier deliberately corrupted
        # cache must not mask the CAS collision this check intends to exercise.
        collision = run(manifest, 'minimal', root / 'other-downloads', root / 'restored', False, True)
        require(collision['status'].startswith('failed') and bad.read_bytes() == b'leave this existing different file intact', 'Existing different object was overwritten')
        results.append({'check': 'different_existing_object_not_overwritten', 'passed': True})
        for name, malicious in (('traversal', '../escape'), ('symlink', 'objects/link')):
            path = make_manifest(name, malicious)
            outcome = run(path, 'minimal', root / (name + '-downloads'), root / (name + '-restore'), False, True)
            require(outcome['status'].startswith('failed') and not (root / 'escape').exists(), 'Unsafe tar member accepted')
            results.append({'check': 'reject_tar_' + name, 'passed': True})
        unsafe_with_existing = make_manifest('unsafe-existing', '../escape')
        unsafe_cache = run(unsafe_with_existing, 'minimal', root / 'unsafe-existing-downloads', root / 'multiple-restored', False, True)
        require(unsafe_cache['status'].startswith('failed'), 'Existing valid objects bypassed unsafe archive rejection')
        results.append({'check': 'cached_objects_do_not_bypass_archive_safety', 'passed': True})
        m = json.loads(manifest.read_text()); m['files'][0]['restore_path'] = '../escape'
        invalid = root / 'bad-manifest.json'; invalid.write_text(json.dumps(m))
        outcome = run(invalid, 'minimal', root / 'invalid-downloads', root / 'invalid-restore', True, True)
        require(outcome['status'].startswith('failed'), 'Manifest path traversal accepted')
        results.append({'check': 'reject_manifest_restore_path_traversal', 'passed': True})
    return {'status': 'passed_synthetic_artifact_fixture', 'checks': results, 'count': len(results),
            'remote_network_reads': 0, 'science_files_or_models_used': 0, 'zstandard_version': zstandard.__version__}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=Path('artifacts_manifest.json'))
    parser.add_argument('--profile', choices=('minimal', 'full'), default='minimal')
    parser.add_argument('--directory', type=Path, default=Path('artifacts/downloads'))
    parser.add_argument('--restore-dir', type=Path, default=Path('artifacts/restored'))
    parser.add_argument('--materialize', action='store_true'); parser.add_argument('--output', type=Path)
    parser.add_argument('--self-test', action='store_true'); args = parser.parse_args()
    try:
        result = self_test() if args.self_test else run(args.manifest, args.profile, args.directory, args.restore_dir, args.materialize)
    except ArtifactError as exc:
        result = {'status': 'failed_artifact_restore', 'error': str(exc)}
    except Exception as exc:
        result = {'status': 'failed_artifact_restore', 'error': 'Engineering failure (' + type(exc).__name__ + '); redirect URLs are not logged'}
    if args.output:
        require(args.output.resolve() != args.manifest.resolve(), 'Audit output cannot overwrite the manifest')
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    print(json.dumps({k: result[k] for k in ('status', 'count', 'selected_packages', 'selected_unique_objects', 'remote_bytes_read', 'error') if k in result}, ensure_ascii=False))
    raise SystemExit(not result['status'].startswith('passed'))
