"""Run the original full package verifier once, with separate timing evidence.

No packing, uploading, Git action, source-file writes or scientific metric
operations. Each original verify_package call is unchanged; aggregation is
the original verify_local schema, so no second decompression pass is needed.
"""
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import time

ROOT=Path('/mnt/yangx-projects-f/20261011/representation_publication')
PACKAGES=ROOT/'packages'
SOURCE=ROOT/'publication_tools/pack_artifacts.py'
SOURCE_SHA='c6c550900aeae2ef9e8e71e3df631f041637c30347513e08c3b12412412752d0'
MANIFEST=PACKAGES/'package_manifest.json'
TIMELINE=ROOT/'local_package_verification_timeline.jsonl'
AUDIT=ROOT/'local_package_verification_execution.json'


def utc():return datetime.now(timezone.utc).isoformat()
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def main():
    started=time.monotonic();begin_utc=utc()
    assert sha(SOURCE)==SOURCE_SHA
    spec=importlib.util.spec_from_file_location('unchanged_original_package_verifier',SOURCE)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    manifest_before=sha(MANIFEST);manifest=json.loads(MANIFEST.read_text())
    assets,_,objects=module.manifest_schema(manifest)
    assert len(assets)==89 and len(objects)==4959
    before_receipts=[]
    receipts=ROOT/'upload_receipts.jsonl'
    if receipts.exists():before_receipts=[json.loads(line) for line in receipts.read_text().splitlines() if line]
    already_uploaded={r['name'] for r in before_receipts if r.get('status') in
        ('existing_identical_asset_reused','uploaded_provider_size_and_digest_verified')}
    audit=dict(status='running_full_local_stream_verification',started_utc=begin_utc,
        started_monotonic=started,package_manifest_sha256=manifest_before,verifier_source_sha256=SOURCE_SHA,
        wrapper_sha256=sha(__file__),planned_packages=len(assets),planned_objects=len(objects),
        pre_verification_upload_receipt_count=len(before_receipts),
        asset_names_with_receipts_before_verification=sorted(already_uploaded),
        timeline_path=str(TIMELINE),source_scientific_files_modified=0,archives_repacked=0,
        new_forward=0,scientific_metrics=0,Git_mutations=0,uploads=0,
        evidence_scope='One unchanged original verify_package call per existing package, complete compressed stream and every decompressed regular object; no duplicate full pass.')
    write(AUDIT,audit);records=[];timing=[]
    with TIMELINE.open('x') as timeline:
        for i,asset in enumerate(assets,1):
            package_started=time.monotonic();package_start_utc=utc()
            expected=[obj for obj in objects if obj['package_id']==asset['package_id']]
            event=dict(event='verification_started',index=i,total=len(assets),name=asset['name'],
                package_id=asset['package_id'],utc=package_start_utc,monotonic=package_started,
                bytes=asset['bytes'],objects=len(expected),
                upload_receipt_precedes_this_verification=asset['name'] in already_uploaded)
            timeline.write(json.dumps(event)+'\n');timeline.flush();print(json.dumps(event),flush=True)
            try:
                record=module.verify_package(PACKAGES/asset['name'],asset,expected)
            except BaseException as error:
                end_utc=utc();failed=dict(event='verification_failed',index=i,name=asset['name'],utc=end_utc,
                    monotonic=time.monotonic(),error_type=type(error).__name__,error=str(error))
                timeline.write(json.dumps(failed)+'\n');timeline.flush();print(json.dumps(failed),flush=True)
                audit.update(status='failed_full_local_stream_verification',failed_asset=asset['name'],
                    error_type=type(error).__name__,error=str(error),completed_packages=len(records),
                    finished_utc=end_utc,elapsed_seconds=time.monotonic()-started)
                write(AUDIT,audit);raise
            end=time.monotonic();end_utc=utc();records.append(record)
            finished=dict(event='verification_completed',index=i,total=len(assets),name=asset['name'],
                package_id=asset['package_id'],started_utc=package_start_utc,utc=end_utc,
                started_monotonic=package_started,monotonic=end,elapsed_seconds=end-package_started,
                status=record['status'],bytes=asset['bytes'],objects=record['object_count'],
                upload_receipt_precedes_this_verification=asset['name'] in already_uploaded)
            timing.append(finished);timeline.write(json.dumps(finished)+'\n');timeline.flush();print(json.dumps(finished),flush=True)
    assert sha(SOURCE)==SOURCE_SHA and sha(MANIFEST)==manifest_before
    assert len(records)==len(assets)==89 and sum(r['object_count'] for r in records)==len(objects)==4959
    result=dict(status='passed_full_local_stream_verification',package_manifest_sha256=manifest_before,
        package_count=len(records),object_count=len(objects),compressed_bytes=sum(a['bytes'] for a in assets),
        original_payload_bytes=sum(o['bytes'] for o in objects),records=records,
        full_decompression_and_object_SHA256=True,actual_uncompressed_files_written=0,source_files_modified=0)
    # Publish the same success schema only after every stream passed. The upload
    # gate can never consume an early per-package event as whole-set success.
    out=PACKAGES/'local_package_verification.json'
    if out.exists():raise FileExistsError('Retain an existing verification record; do not overwrite it')
    temporary=out.with_suffix('.json.pending')
    with temporary.open('x') as handle:json.dump(result,handle,indent=2,ensure_ascii=False,allow_nan=False);handle.write('\n')
    temporary.rename(out)
    upload_prior_times=[r for r in timing if r['upload_receipt_precedes_this_verification']]
    audit.update(status='passed_full_local_stream_verification',finished_utc=utc(),
        elapsed_seconds=time.monotonic()-started,completed_packages=89,verified_objects=4959,
        compressed_bytes=result['compressed_bytes'],payload_bytes=result['original_payload_bytes'],
        output_path=str(out),output_sha256=sha(out),timeline_sha256=sha(TIMELINE),
        first_actual_package_start_utc=timing[0]['started_utc'],first_actual_package_completed_utc=timing[0]['utc'],
        earliest_completed_verification_of_previously_uploaded_asset_utc=upload_prior_times[0]['utc'] if upload_prior_times else None,
        earlier_uploads_were_not_retroactively_declared_preverified=True,
        per_package_verification_count=89,duplicate_complete_passes=0,
        wall_utc_minus_monotonic_seconds=(datetime.fromisoformat(audit['finished_utc'])-datetime.fromisoformat(begin_utc)).total_seconds()-(time.monotonic()-started))
    write(AUDIT,audit);print(json.dumps({k:audit[k] for k in ['status','completed_packages','verified_objects','elapsed_seconds','first_actual_package_completed_utc']}),flush=True)


if __name__=='__main__':main()
