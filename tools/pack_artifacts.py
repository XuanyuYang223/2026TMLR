#!/usr/bin/env python3
"""Content-addressed, deterministic release packing and full stream verification.

Only explicitly inventoried regular source files are read. Scientific files
are never changed. Package members are regular PACKAGE_INDEX.json and
objects/<sha256>; source paths are provenance, not archive member paths.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import tarfile
import tempfile
import time

import zstandard as zstd

RAW_CAP = 5*1024**3//4
ASSET_CAP = 3*1024**3//2
CHUNK = 1024**2
RELEASE_BASE = 'https://github.com/XuanyuYang223/2026TMLR/releases/download/representation-selectivity-2026-10-11-caveat/'
SHA_PATTERN = re.compile(r'^[0-9a-f]{64}$')
SCHEMA = 1


def canonical(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=False,
                       separators=(',', ':'), allow_nan=False)+'\n').encode()


def sha_file(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def file_stat(value):
    return dict(device=value.st_dev, inode=value.st_ino, size_bytes=value.st_size,
                mtime_ns=value.st_mtime_ns, ctime_ns=value.st_ctime_ns,
                mode=stat.S_IFMT(value.st_mode))


def require_sha(value):
    if not isinstance(value, str) or not SHA_PATTERN.fullmatch(value):
        raise ValueError('Expected lowercase SHA256: '+repr(value))
    return value


def source_stat(obj):
    path=Path(obj['source_realpath'])
    if not path.is_absolute() or path.is_symlink() or path.resolve()!=path:
        raise ValueError('source_realpath must be the resolved regular source: '+str(path))
    status=path.stat()
    if not stat.S_ISREG(status.st_mode) or status.st_size!=obj['size_bytes']:
        raise ValueError('Source file type/byte count differs from inventory: '+str(path))
    return file_stat(status)


class SourceReader:
    """Hash bytes actually supplied to tar, reject a source changing mid-read."""
    def __init__(self, obj):
        self.obj=obj; self.before=source_stat(obj)
        fd=os.open(obj['source_realpath'],os.O_RDONLY|getattr(os,'O_NOFOLLOW',0))
        self.handle=os.fdopen(fd,'rb'); self.digest=hashlib.sha256(); self.bytes=0
        if file_stat(os.fstat(self.handle.fileno()))!=self.before:
            self.handle.close();raise ValueError('Source binding changed before read')
    def read(self, size=-1):
        result=self.handle.read(size);self.digest.update(result);self.bytes+=len(result)
        return result
    def finish(self):
        after=file_stat(os.fstat(self.handle.fileno()));self.handle.close()
        path_after=source_stat(self.obj)
        if self.before!=after or after!=path_after:
            raise ValueError('Source stat changed while reading: '+self.obj['source_realpath'])
        if self.bytes!=self.obj['size_bytes'] or self.digest.hexdigest()!=self.obj['sha256']:
            raise ValueError('Source byte/SHA256 mismatch: '+self.obj['source_realpath'])
        return dict(sha256=self.obj['sha256'],source_realpath=self.obj['source_realpath'],
                    bytes_read=self.bytes,source_stat_before=self.before,source_stat_after=after,
                    source_sha256_verified=True)
    def abort(self):
        if not self.handle.closed:self.handle.close()


class OutputWriter:
    def __init__(self, handle, cap):
        self.handle=handle;self.cap=cap;self.digest=hashlib.sha256();self.bytes=0
    def write(self, blob):
        if self.bytes+len(blob)>self.cap:
            raise ValueError('Compressed asset exceeds 1.5 GiB; no scientific file split/alteration is allowed')
        written=self.handle.write(blob)
        if written!=len(blob):raise OSError('Short package write')
        self.digest.update(blob);self.bytes+=written;return written
    def flush(self):self.handle.flush()


class InputReader:
    def __init__(self, handle):
        self.handle=handle;self.digest=hashlib.sha256();self.bytes=0
    def read(self, size=-1):
        blob=self.handle.read(size);self.digest.update(blob);self.bytes+=len(blob);return blob
    def readinto(self, buffer):
        blob=self.read(len(buffer));buffer[:len(blob)]=blob;return len(blob)


def normalized_objects(payload, raw_cap=RAW_CAP):
    objects=payload.get('objects') if isinstance(payload,dict) else payload
    if not isinstance(objects,list):raise ValueError('Inventory needs objects[]')
    merged={}
    for item in objects:
        if not isinstance(item,dict):raise ValueError('Object must be a mapping')
        required=item.get('release_object_required')
        if type(required) is not bool:raise ValueError('release_object_required must be boolean')
        if not required:continue
        digest=require_sha(item['sha256']);size=item['size_bytes']
        if type(size) is not int or size<0:raise ValueError('Invalid source byte count')
        if size>raw_cap:
            raise ValueError(f'Single source exceeds raw cap: {item["source_realpath"]}, {size} bytes > {raw_cap}; stop without altering original')
        if item.get('object_member')!='objects/'+digest:
            raise ValueError('Inventory object_member must equal objects/<SHA256>')
        group=item['package_group']
        if not isinstance(group,str) or not group:raise ValueError('Missing package_group')
        for field in ('logical_paths','experiments','roles'):
            if not isinstance(item[field],list) or not all(isinstance(v,str) for v in item[field]):
                raise ValueError('Inventory needs string list: '+field)
        obj={k:item[k] for k in ('sha256','size_bytes','source_realpath','object_member','package_group')}
        obj.update({k:sorted(set(item[k])) for k in ('logical_paths','experiments','roles')})
        source_stat(obj)  # Metadata only; do not hash scientific objects twice here.
        if digest in merged:
            old=merged[digest]
            if old['size_bytes']!=size or old['package_group']!=group:
                raise ValueError('Conflicting SHA256 size/group; resolve inventory before packing')
            for key in ('logical_paths','experiments','roles'):
                old[key]=sorted(set(old[key])|set(obj[key]))
            # Physical duplicates stay explicit provenance; one source supplies bytes.
            old['source_realpaths']=sorted(set(old['source_realpaths'])|{obj['source_realpath']})
            old['source_realpath']=old['source_realpaths'][0]
        else:
            obj['source_realpaths']=[obj['source_realpath']];merged[digest]=obj
    return [merged[digest] for digest in sorted(merged)]


def slug(group):
    clean=re.sub('[^a-z0-9-]+','-',group.lower()).strip('-') or 'group'
    if clean!=group or len(clean)>64:
        clean=clean[:48]+'-'+hashlib.sha256(group.encode()).hexdigest()[:10]
    return clean


def package_index(group, bin_index, objects):
    records=[dict(sha256=o['sha256'],bytes=o['size_bytes'],archive_member=o['object_member'],
                  logical_paths=o['logical_paths'],experiments=o['experiments'],roles=o['roles'])
             for o in objects]
    identity=dict(schema_version=SCHEMA,package_group=group,bin_index=bin_index,objects=records)
    short=hashlib.sha256(canonical(identity)).hexdigest()[:16]
    package_id=f'{slug(group)}-{bin_index:03d}-{short}'
    index=dict(identity,package_id=package_id,
               archive_format='tar.zst regular content-addressed objects, original scientific bytes')
    return index,'representation-'+package_id+'.tar.zst'


def plan(objects, raw_cap=RAW_CAP):
    groups=defaultdict(list)
    for obj in objects:groups[obj['package_group']].append(obj)
    packages=[]
    for group, members in sorted(groups.items()):
        bins=[];current=[];total=0
        for obj in sorted(members,key=lambda x:x['sha256']):
            if current and total+obj['size_bytes']>raw_cap:
                bins.append(current);current=[];total=0
            current.append(obj);total+=obj['size_bytes']
        if current:bins.append(current)
        for i, members in enumerate(bins,1):
            index,name=package_index(group,i,members)
            packages.append(dict(package_id=index['package_id'],name=name,index=index,objects=members,
                                 raw_bytes=sum(o['size_bytes'] for o in members),package_group=group))
    digests=[o['sha256'] for p in packages for o in p['objects']]
    if len(digests)!=len(set(digests)) or set(digests)!=set(o['sha256'] for o in objects):
        raise AssertionError('Every selected SHA256 must be packaged exactly once globally')
    return packages


def tar_info(name, size):
    info=tarfile.TarInfo(name);info.size=size;info.type=tarfile.REGTYPE
    info.mtime=0;info.uid=info.gid=0;info.uname=info.gname='';info.mode=0o644
    return info


def verify_package(path, asset, expected_objects, expected_index=None):
    """Hash all decompressed regular members and all compressed input bytes once."""
    path=Path(path)
    if path.is_symlink() or not path.is_file():raise ValueError('Package must be a regular local file')
    before=file_stat(path.stat())
    if before['size_bytes']!=asset['bytes'] or asset['bytes']>ASSET_CAP:
        raise ValueError('Package bytes differ from manifest or exceed asset cap')
    require_sha(asset['sha256']);wanted={o['sha256']:o for o in expected_objects}
    if len(wanted)!=len(expected_objects):raise ValueError('Duplicate expected objects')
    seen=set();members=[];index=None
    with path.open('rb') as raw:
        source=InputReader(raw)
        with zstd.ZstdDecompressor().stream_reader(source,read_across_frames=True,closefd=False) as reader:
            with tarfile.open(fileobj=reader,mode='r|') as archive:
                for info in archive:
                    name=info.name;logical=PurePosixPath(name)
                    if name in seen or logical.is_absolute() or '..' in logical.parts or str(logical)!=name:
                        raise ValueError('Duplicate or unsafe archive member: '+name)
                    if not info.isreg():raise ValueError('Only regular archive members allowed: '+name)
                    if info.uid!=0 or info.gid!=0 or info.mtime!=0 or info.uname or info.gname or info.mode!=0o644:
                        raise ValueError('Noncanonical archive ownership/time/mode: '+name)
                    seen.add(name);stream=archive.extractfile(info)
                    if stream is None:raise ValueError('Missing regular member stream')
                    if name=='PACKAGE_INDEX.json':
                        if len(seen)!=1 or info.size>16*1024**2:raise ValueError('Index must be first and bounded')
                        blob=stream.read();index=json.loads(blob)
                        if canonical(index)!=blob:raise ValueError('Noncanonical PACKAGE_INDEX.json')
                        if index.get('schema_version')!=SCHEMA or index.get('package_id')!=asset['package_id']:
                            raise ValueError('Package index schema/id mismatch')
                        if expected_index is not None and index!=expected_index:raise ValueError('Existing package index differs from current plan')
                        records=index.get('objects')
                        if not isinstance(records,list) or len(records)!=len(wanted):raise ValueError('Index object count mismatch')
                        listed={o['sha256']:o for o in records}
                        if len(listed)!=len(records) or set(listed)!=set(wanted):raise ValueError('Index object SHA256 set mismatch')
                        for digest, record in listed.items():
                            require_sha(digest)
                            if record.get('archive_member')!='objects/'+digest or record.get('bytes')!=wanted[digest]['bytes']:
                                raise ValueError('Index member/byte schema mismatch')
                            for field in ('logical_paths','experiments','roles'):
                                if not isinstance(record.get(field),list) or not all(isinstance(v,str) for v in record[field]):
                                    raise ValueError('Index metadata list schema mismatch: '+field)
                        continue
                    if index is None or not name.startswith('objects/'):
                        raise ValueError('Unexpected archive member: '+name)
                    digest=require_sha(name[len('objects/'):])
                    if digest not in wanted or info.size!=wanted[digest]['bytes']:
                        raise ValueError('Extra or wrong-size object member: '+name)
                    h=hashlib.sha256();size=0
                    while True:
                        blob=stream.read(CHUNK)
                        if not blob:break
                        h.update(blob);size+=len(blob)
                    if size!=info.size or h.hexdigest()!=digest:
                        raise ValueError('Decompressed object byte/SHA256 mismatch: '+name)
                    members.append(dict(sha256=digest,bytes=size,archive_member=name))
                # tar stops at its end marker; drain its own buffered stream too,
                # rejecting nonzero tails or hidden concatenated archives.
                while True:
                    tail=archive.fileobj.read(CHUNK)
                    if not tail:break
                    if any(tail):raise ValueError('Nonzero data after tar end marker')
            while reader.read(CHUNK):
                raise ValueError('Unconsumed decompressed data after tar stream')
        if source.bytes!=asset['bytes'] or source.digest.hexdigest()!=asset['sha256']:
            raise ValueError('Complete compressed stream SHA256/bytes mismatch')
    if file_stat(path.stat())!=before:raise ValueError('Package changed during local verification')
    if seen!={'PACKAGE_INDEX.json',*('objects/'+digest for digest in wanted)}:
        raise ValueError('Archive coverage differs from expected object set')
    return dict(status='passed_full_stream_verification',name=asset['name'],bytes=asset['bytes'],sha256=asset['sha256'],
                package_id=asset['package_id'],object_count=len(members),objects=members,
                regular_members_only=True,extra_members=False,unsafe_members=False,
                decompressed_duplicate_files_written=0)


def source_hash_check(obj):
    reader=SourceReader(obj)
    try:
        while reader.read(CHUNK):pass
        return reader.finish()
    except BaseException:
        reader.abort();raise


def write_package(package, output, asset_cap=ASSET_CAP):
    output=Path(output);target=output/package['name'];index_blob=canonical(package['index'])
    source_checks=[]
    if target.exists():
        # Existing immutable packages may be reused only after both source
        # bindings and all package member bytes are actually checked again.
        for obj in package['objects']:source_checks.append(source_hash_check(obj))
        asset=dict(name=target.name,url=RELEASE_BASE+target.name,bytes=target.stat().st_size,
                   sha256=sha_file(target),package_id=package['package_id'],part_index=1)
        verify=verify_package(target,asset,[dict(sha256=o['sha256'],bytes=o['size_bytes']) for o in package['objects']],package['index'])
        return asset,dict(package_id=package['package_id'],status='existing_same_package_verified_and_reused',source_checks=source_checks,verification=verify)
    temporary=None
    try:
        with tempfile.NamedTemporaryFile(prefix='.'+target.name+'.',suffix='.partial',dir=output,delete=False) as handle:
            temporary=Path(handle.name);sink=OutputWriter(handle,asset_cap)
            with zstd.ZstdCompressor(level=3,threads=2).stream_writer(sink,closefd=False) as compressed:
                with tarfile.open(fileobj=compressed,mode='w|',format=tarfile.USTAR_FORMAT) as archive:
                    archive.addfile(tar_info('PACKAGE_INDEX.json',len(index_blob)),io.BytesIO(index_blob))
                    for obj in package['objects']:
                        reader=SourceReader(obj)
                        try:
                            archive.addfile(tar_info(obj['object_member'],obj['size_bytes']),reader)
                            source_checks.append(reader.finish())
                        except BaseException:
                            reader.abort();raise
            sink.flush();os.fsync(handle.fileno())
            size=sink.bytes;digest=sink.digest.hexdigest()
        asset=dict(name=target.name,url=RELEASE_BASE+target.name,bytes=size,sha256=digest,
                   package_id=package['package_id'],part_index=1)
        # Link is an exclusive atomic publication; never replace a same-name
        # different package. Temporary is solely created by this invocation.
        os.link(temporary,target);temporary.unlink();temporary=None
        return asset,dict(package_id=package['package_id'],status='packed_sources_SHA_verified',source_checks=source_checks)
    finally:
        if temporary is not None and temporary.exists():temporary.unlink()


def manifest_schema(manifest):
    if manifest.get('schema_version')!=SCHEMA:raise ValueError('Unsupported package manifest schema')
    assets=manifest.get('assets');packages=manifest.get('packages');objects=manifest.get('objects')
    if not all(isinstance(v,list) for v in (assets,packages,objects)):raise ValueError('Manifest needs assets/packages/objects arrays')
    if len({a['name'] for a in assets})!=len(assets) or len({p['package_id'] for p in packages})!=len(packages):
        raise ValueError('Duplicate asset/package identifiers')
    by_package={p['package_id']:p for p in packages};by_asset={a['name']:a for a in assets}
    object_set=set()
    for asset in assets:
        name=asset['name']
        if Path(name).name!=name or not re.fullmatch(r'representation-[a-z0-9-]+\.tar\.zst',name):raise ValueError('Unsafe asset filename')
        require_sha(asset['sha256'])
        if type(asset['bytes']) is not int or not 0<=asset['bytes']<=ASSET_CAP or asset['part_index']!=1:
            raise ValueError('Asset byte/part schema mismatch')
        package=by_package.get(asset['package_id'])
        if package is None or package['parts']!=[name] or package['compressed_stream_sha256']!=asset['sha256']:
            raise ValueError('Asset/package stream mapping mismatch')
        if asset['url']!=RELEASE_BASE+name:raise ValueError('Unexpected release URL')
    for obj in objects:
        digest=require_sha(obj['sha256'])
        if digest in object_set:raise ValueError('Object occurs more than once globally')
        object_set.add(digest)
        if obj['archive_member']!='objects/'+digest or obj['package_id'] not in by_package or type(obj['bytes']) is not int or obj['bytes']<0:
            raise ValueError('Object member/package/byte mapping mismatch')
    for package in packages:
        linked={o['sha256'] for o in objects if o['package_id']==package['package_id']}
        if len(package['parts'])!=1 or package['parts'][0] not in by_asset or set(package['object_sha256'])!=linked or len(package['object_sha256'])!=len(linked):
            raise ValueError('Package object coverage mismatch')
    return assets,packages,objects


def verify_local(manifest_path, output=None):
    manifest_path=Path(manifest_path);manifest=json.loads(manifest_path.read_text())
    assets,_,objects=manifest_schema(manifest);folder=manifest_path.parent;records=[]
    for asset in assets:
        expected=[o for o in objects if o['package_id']==asset['package_id']]
        records.append(verify_package(folder/asset['name'],asset,expected))
    result=dict(status='passed_full_local_stream_verification',package_manifest_sha256=sha_file(manifest_path),
        package_count=len(records),object_count=len(objects),compressed_bytes=sum(a['bytes'] for a in assets),
        original_payload_bytes=sum(o['bytes'] for o in objects),records=records,
        full_decompression_and_object_SHA256=True,actual_uncompressed_files_written=0,source_files_modified=0)
    if output:Path(output).write_bytes(canonical(result))
    return result


def pack_inventory(inventory_path, output, raw_cap=RAW_CAP, asset_cap=ASSET_CAP):
    begun=time.monotonic();inventory_path=Path(inventory_path);output=Path(output)
    payload=json.loads(inventory_path.read_text());objects=normalized_objects(payload,raw_cap);packages=plan(objects,raw_cap)
    output.mkdir(parents=True,exist_ok=True);assets=[];checks=[];manifest_objects=[]
    for package in packages:
        asset,check=write_package(package,output,asset_cap);assets.append(asset);checks.append(check)
        print(json.dumps(dict(status='package_source_verified',name=asset['name'],
            object_count=len(package['objects']),raw_bytes=package['raw_bytes'],
            compressed_bytes=asset['bytes'],sha256=asset['sha256'],
            elapsed_seconds=time.monotonic()-begun)),flush=True)
        for obj in package['objects']:
            manifest_objects.append(dict(sha256=obj['sha256'],bytes=obj['size_bytes'],package_id=package['package_id'],
                archive_member=obj['object_member'],experiments=obj['experiments'],roles=obj['roles'],logical_paths=obj['logical_paths']))
    manifest=dict(schema_version=SCHEMA,format='tar.zst level3 threads2 canonical regular members',
        release_base_url=RELEASE_BASE,inventory_sha256=sha_file(inventory_path),raw_bin_cap_bytes=raw_cap,
        asset_cap_bytes=asset_cap,assets=assets,
        packages=[dict(package_id=p['package_id'],parts=[p['name']],compressed_stream_sha256=asset['sha256'],
                       object_sha256=[o['sha256'] for o in p['objects']]) for p,asset in zip(packages,assets)],objects=manifest_objects)
    manifest_schema(manifest)
    manifest_path=output/'package_manifest.json';blob=canonical(manifest)
    if manifest_path.exists() and manifest_path.read_bytes()!=blob:
        raise ValueError('Existing package_manifest.json differs; use a new explicit output directory')
    if not manifest_path.exists():
        with manifest_path.open('xb') as handle:handle.write(blob)
    audit=dict(status='packed_source_checked_not_yet_full_local_verification',utc=datetime.now(timezone.utc).isoformat(),
        inventory_sha256=sha_file(inventory_path),package_manifest_sha256=sha_file(manifest_path),
        script_sha256=sha_file(__file__),zstandard_version=zstd.__version__,
        packages=len(assets),objects=len(objects),input_release_object_rows=sum(bool(o['release_object_required']) for o in (payload['objects'] if isinstance(payload,dict) else payload)),
        global_unique_SHA_once=True,raw_cap_bytes=raw_cap,asset_cap_bytes=asset_cap,
        raw_bytes=sum(o['bytes'] for o in manifest_objects),compressed_bytes=sum(a['bytes'] for a in assets),
        records=checks,source_files_modified=0,scientific_arrays_changed=0,scientific_metrics_computed=0,
        uploaded_assets=0,Git_mutations=0,elapsed_seconds=time.monotonic()-begun)
    (output/'packaging_audit.json').write_bytes(canonical(audit))
    return manifest,audit


def fixture(output):
    """Small independent engineering bytes only; no real artifact inventory."""
    output=Path(output);output.mkdir(parents=True,exist_ok=True);checks=[]
    def check(name,condition):
        if not condition:raise AssertionError(name)
        checks.append(dict(check=name,passed=True))
    with tempfile.TemporaryDirectory(prefix='pack-artifact-fixture-',dir=output) as tmp:
        root=Path(tmp);rows=[]
        for i,(size,group,needed) in enumerate([(700,'fixture-main',True),(700,'fixture-main',True),(20,'fixture-other',True),(15,'fixture-main',False)]):
            blob=bytes((j*37+i*31)%256 for j in range(size));path=root/f'source-{i}.bin';path.write_bytes(blob)
            digest=hashlib.sha256(blob).hexdigest()
            rows.append(dict(sha256=digest,size_bytes=size,source_realpath=str(path.resolve()),object_member='objects/'+digest,
                logical_paths=[f'fixture/source-{i}.bin'],experiments=['engineering_fixture_only'],roles=['synthetic_bytes'],package_group=group,release_object_required=needed))
        duplicate=dict(rows[0]);duplicate['logical_paths']=['fixture/duplicate.bin'];rows.append(duplicate)
        inv=root/'inventory.json';inv.write_bytes(canonical(dict(objects=rows)))
        first,audit=pack_inventory(inv,root/'first',raw_cap=1000)
        check('small raw-cap split produces3packages',len(first['assets'])==3)
        check('global SHA dedup selects3objects and preserves duplicate logical map',len(first['objects'])==3 and len(next(o for o in first['objects'] if o['sha256']==rows[0]['sha256'])['logical_paths'])==2)
        check('release_false_object omitted',rows[3]['sha256'] not in {o['sha256'] for o in first['objects']})
        actual=verify_local(root/'first/package_manifest.json')
        check('all compressed streams and regular object bytes verify',actual['status']=='passed_full_local_stream_verification' and actual['object_count']==3)
        second,_=pack_inventory(inv,root/'second',raw_cap=1000)
        check('canonical tar and Zstd deterministic across separate outputs',first['assets']==second['assets'])
        reused,reuse_audit=pack_inventory(inv,root/'first',raw_cap=1000)
        check('idempotent existing package validates source bindings and reuses',reused==first and all(r['status']=='existing_same_package_verified_and_reused' for r in reuse_audit['records']))
        source=Path(rows[0]['source_realpath']);old_stat=file_stat(source.stat());blob=source.read_bytes();source.write_bytes(bytes([blob[0]^1])+blob[1:])
        try:pack_inventory(inv,root/'first',raw_cap=1000)
        except ValueError as error:rejected='SHA256 mismatch' in str(error)
        else:rejected=False
        check('same-size source SHA mutation rejected before package reuse',rejected)
        check('existing package bytes unchanged after rejected source mutation',all(sha_file(root/'first'/a['name'])==a['sha256'] for a in first['assets']))
        source.write_bytes(blob)
        try:normalized_objects(dict(objects=[rows[0]]),raw_cap=699)
        except ValueError as error:rejected='Single source exceeds raw cap' in str(error)
        else:rejected=False
        check('oversized single source stops with exact path and bytes',rejected)
        # Tiny deliberately invalid packages are engineered anew, not copies of
        # any scientific archive. They must be rejected by full local checks.
        base_plan=plan(normalized_objects(dict(objects=[rows[2]])))[0]
        for kind in ['unsafe','extra','corrupt_object','symlink']:
            bad=root/f'{kind}.tar.zst'
            with bad.open('wb') as handle:
                with zstd.ZstdCompressor(level=3,threads=2).stream_writer(handle,closefd=False) as zhandle:
                    with tarfile.open(fileobj=zhandle,mode='w|',format=tarfile.USTAR_FORMAT) as archive:
                        index_blob=canonical(base_plan['index']);archive.addfile(tar_info('PACKAGE_INDEX.json',len(index_blob)),io.BytesIO(index_blob))
                        raw=Path(rows[2]['source_realpath']).read_bytes()
                        if kind=='corrupt_object':raw=bytes([raw[0]^1])+raw[1:]
                        archive.addfile(tar_info(rows[2]['object_member'],len(raw)),io.BytesIO(raw))
                        if kind in ('unsafe','extra'):archive.addfile(tar_info('../escape' if kind=='unsafe' else 'unlisted.txt',1),io.BytesIO(b'x'))
                        if kind=='symlink':
                            member=tar_info('link',0);member.type=tarfile.SYMTYPE;member.linkname='objects/'+rows[2]['sha256'];archive.addfile(member)
            asset=dict(name=bad.name,bytes=bad.stat().st_size,sha256=sha_file(bad),package_id=base_plan['package_id'])
            try:verify_package(bad,asset,[dict(sha256=rows[2]['sha256'],bytes=rows[2]['size_bytes'])])
            except ValueError:rejected=True
            else:rejected=False
            check('full verifier rejects_'+kind,rejected)
        # Same generated name carrying different bytes must never be overwritten.
        poisoned=root/'poisoned';poisoned.mkdir();first_plan=plan(normalized_objects(dict(objects=rows),1000),1000)[0]
        poison=poisoned/first_plan['name'];poison.write_bytes(b'preexisting different package');saved=poison.read_bytes()
        try:pack_inventory(inv,poisoned,raw_cap=1000)
        except (ValueError,zstd.ZstdError,tarfile.TarError):rejected=True
        else:rejected=False
        check('different preexisting same-name asset rejected and retained',rejected and poison.read_bytes()==saved)
    result=dict(status='passed_synthetic_packaging_fixture_only',checks=checks,count=len(checks),
        source_script_sha256=sha_file(__file__),zstandard_version=zstd.__version__,
        real_scientific_input_files_read=0,original_scientific_files_modified=0,
        synthetic_temp_directory_removed=True,new_forward=0,scientific_metrics=0,uploads=0,Git_mutations=0)
    (output/'packaging_engineering_fixture.json').write_bytes(canonical(result));return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--objects',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--verify-local',action='store_true')
    parser.add_argument('--manifest',type=Path)
    parser.add_argument('--fixture',action='store_true')
    args=parser.parse_args()
    if args.fixture:
        result=fixture(args.output)
    elif args.verify_local:
        result=verify_local(args.manifest or args.output/'package_manifest.json',args.output/'local_package_verification.json')
    else:
        if args.objects is None:parser.error('--objects is required for packing')
        _,result=pack_inventory(args.objects,args.output)
    print(json.dumps({k:result[k] for k in ['status','packages','objects','package_count','object_count','count','elapsed_seconds'] if k in result}))
