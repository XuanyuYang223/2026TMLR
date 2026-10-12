"""Wait for all package uploads, independently gate, publish only authorized new release.

All local writes are outside the scientific Git tree. No upload retry, asset
deletion, tag mutation, Git push, training, model import or scientific change.
"""
from __future__ import annotations
from collections import Counter
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import urllib.error
import urllib.request

B=Path('/mnt/yangx-projects-f/20261011/representation_publication')
OUT=B/'publication_finalizer_v2'
REPO='XuanyuYang223/2026TMLR'
RID=410204953
TAG='representation-selectivity-2026-10-11-caveat'
SOURCE='83325c2b6a2e41c69fc8b2e8f25e0cb58b80aca2'
METADATA='e66b561e88394340fc1a43bc48c639e36d57eb6a'
MAIN='29ec0802987fda8d1c5b9986426233d4ceb4b4c8'
READY='all_package_assets_uploaded_provider_verified'
SUCCESS='published_prerelease_provider_verified'
PATCHED=False

def stamp():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name('.'+path.name+'.tmp')
    temp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n');os.replace(temp,path)
def require(value,message):
    if not value:raise RuntimeError(message)
def api(path,method='GET',payload=None,optional_404=False):
    cmd=['gh','api',path]
    if method!='GET':cmd+=['--method',method]
    if payload is not None:cmd+=['--input',str(payload)]
    result=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,timeout=120)
    if result.returncode:
        if optional_404:
            try:
                if json.loads(result.stdout).get('status')=='404':return None
            except (ValueError,AttributeError):pass
            if 'HTTP 404' in result.stderr:return None
        # Never echo stderr, auth headers or signed redirects.
        raise RuntimeError('GitHub API request failed: '+method+' '+path+' (exit '+str(result.returncode)+')')
    return json.loads(result.stdout)
def commit_ref(path):
    obj=api(path)['object']
    for _ in range(5):
        if obj['type']=='commit':return obj['sha']
        require(obj['type']=='tag','Unsupported Git ref object type')
        obj=api(f'repos/{REPO}/git/tags/{obj["sha"]}')['object']
    raise RuntimeError('Annotated tag chain exceeds fixed safety bound')
def assets_all():
    result=[]
    for page in range(1,10):
        rows=api(f'repos/{REPO}/releases/{RID}/assets?per_page=100&page={page}')
        require(isinstance(rows,list),'Asset API schema differs')
        result.extend(rows)
        if len(rows)<100:return result
    raise RuntimeError('Unexpected asset count exceeds explicit release scope')

def old_preservation(phase):
    code=B/'publication_tools/remote_state_checks.py'
    with (OUT/f'old_preservation_{phase}.log').open('w') as stream:
        result=subprocess.run(['/home/yangx/ICML/.venv/bin/python','-B',str(code)],stdout=stream,stderr=subprocess.STDOUT,timeout=240)
    require(result.returncode==0,'Existing release/tag preservation tool failed')
    value=read(B/'remote_preservation_provider_audit.json')
    require(value['old_releases_tags_unchanged'],'One or more old release/tag metadata fields differ')
    require(value['all_package_provider_metadata_pass'] and value['package_count']==89,'Existing comparison did not verify all89 package provider records')
    require(len(read(B/'state_before/remote_releases.json'))==3 and len(read(B/'state_before/remote_tags.json'))==3,'Protected baseline does not contain the expected3 releases/3 tags')
    write(OUT/f'old_preservation_{phase}.json',value)
    return value

def gates(package,metadata,phase):
    old=old_preservation(phase)
    repository=api(f'repos/{REPO}')
    require(repository['default_branch']=='main' and not repository['private'],'Target default branch/public visibility differs')
    main=commit_ref(f'repos/{REPO}/git/ref/heads/main')
    tag=commit_ref(f'repos/{REPO}/git/ref/tags/{TAG}')
    require(main==MAIN,'Remote main SHA differs from the authorized finalization boundary')
    require(tag==SOURCE,'Scientific source tag no longer points to the authorized source commit')
    release=api(f'repos/{REPO}/releases/{RID}')
    require(release['id']==RID and release['tag_name']==TAG,'New-release identity differs')
    actual=assets_all();names=Counter(a['name'] for a in actual)
    require(all(v==1 for v in names.values()),'Duplicate remote asset names')
    expected=package['assets']+metadata['files']
    require(len(package['assets'])==89 and len(metadata['files'])==2,'Expected asset scope differs from89+2')
    require(set(names)=={a['name'] for a in expected},'Remote asset-name set differs from the exact89+2 package/metadata set')
    remote={a['name']:a for a in actual};checks=[]
    for e in expected:
        r=remote[e['name']]
        url=e.get('url') or f'https://github.com/{REPO}/releases/download/{TAG}/{e["name"]}'
        # Draft release APIs expose an id-bound temporary tag slug. Only the
        # before-publication gate uses that actual draft slug; published URLs
        # must match the immutable manifest exactly.
        provider_expected_url=url
        if phase=='before_publication' and release['draft'] is True:
            html_prefix=f'https://github.com/{REPO}/releases/tag/'
            require(release['html_url'].startswith(html_prefix),'Draft HTML URL outside this repository')
            draft_slug=release['html_url'][len(html_prefix):]
            require(draft_slug==TAG or (draft_slug.startswith('untagged-') and
                    draft_slug.removeprefix('untagged-').isalnum()),'Unexpected draft tag slug')
            provider_expected_url=f'https://github.com/{REPO}/releases/download/{draft_slug}/{e["name"]}'
        else:
            require(release['draft'] is False,'After-publication gate still sees a draft')
        row=dict(name=e['name'],asset_id=r['id'],state=r['state'],bytes=r['size'],sha256=e['sha256'],provider_digest=r.get('digest'),
                 browser_download_url=r['browser_download_url'],is_metadata=e in metadata['files'],
                 state_match=r['state']=='uploaded',size_match=r['size']==e['bytes'],digest_match=r.get('digest')=='sha256:'+e['sha256'],url_match=r['browser_download_url']==provider_expected_url,declared_public_URL=url,expected_current_provider_URL=provider_expected_url)
        require(all(row[k] for k in ('state_match','size_match','digest_match','url_match')),'Asset provider metadata mismatch: '+e['name'])
        checks.append(row)
    latest=api(f'repos/{REPO}/releases/latest',optional_404=True)
    latest_id=None if latest is None else latest['id']
    require(latest_id is None or latest_id in {r['id'] for r in read(B/'state_before/remote_releases.json')},'Current latest release is outside protected old releases')
    result=dict(status='all_provider_identity_gates_passed',recorded_utc=stamp(),phase=phase,repository=REPO,release_id=RID,tag_name=TAG,
                remote_main_sha=main,tag_commit=tag,old_releases_tags_unchanged=True,protected_release_count=3,protected_tag_count=3,
                package_count=89,metadata_count=2,unique_remote_names=len(names),asset_checks=checks,latest_release_id=latest_id,
                provider_metadata_checks_not_full_remote_byte_readback=True)
    write(OUT/f'provider_gates_{phase}.json',result)
    return result,release

def public_get():
    url=f'https://api.github.com/repos/{REPO}/releases/tags/{TAG}'
    # Explicitly unauthenticated urllib request. Retry only visibility404/503
    # after the already-authorized publication, never issue another PATCH.
    for attempt in range(1,6):
        request=urllib.request.Request(url,headers={'User-Agent':'representation-publication-finalizer/1','Accept':'application/vnd.github+json'})
        try:
            with urllib.request.urlopen(request,timeout=60) as response:
                require(response.status==200,'Public tag HTTP response is not200')
                blob=json.loads(response.read())
            require(blob['id']==RID and blob['tag_name']==TAG and blob['draft'] is False and blob['prerelease'] is True,'Unauthenticated public release fields differ')
            return dict(HTTP_status=200,attempt=attempt,url=url,release_id=blob['id'],html_url=blob['html_url'],authenticated=False,
                        tag_name=blob['tag_name'],draft=blob['draft'],prerelease=blob['prerelease'])
        except urllib.error.HTTPError as exc:
            if exc.code not in (404,503) or attempt==5:raise RuntimeError('Unauthenticated public tag verification failed (HTTP '+str(exc.code)+')') from None
            time.sleep(5)
    raise RuntimeError('Public tag verification did not complete')

def main():
    global PATCHED
    OUT.mkdir(parents=True,exist_ok=True)
    require(not (B/'publication_release_live.json').exists(),'Publication live certificate already exists; inspect rather than repeat publication')
    package_path=B/'packages/package_manifest.json';meta_path=B/'release_metadata/metadata_binding.json';notes_path=B/'release_notes.txt'
    pins={str(p):sha(p) for p in (package_path,meta_path,notes_path,B/'publication_tools/remote_state_checks.py')}
    start=time.monotonic();polls=0
    while True:
        require(not (B/'publication_finalize_cancel.json').exists(),'Root/user cancellation sentinel is present; no publication')
        progress=read(B/'upload_progress_verified.json') if (B/'upload_progress_verified.json').exists() else {}
        polls+=1
        waiting=dict(status='waiting_for_exact_provider_verified_all89',poll_interval_seconds=45,polls=polls,recorded_utc=stamp(),
                     uploaded_verified_assets=progress.get('verified_assets'),expected_packages=progress.get('expected_packages'),
                     progress_status=progress.get('status'),release_id=RID,tag_name=TAG,publication_patch_performed=False)
        write(OUT/'wait_state.json',waiting)
        print(json.dumps(waiting),flush=True)
        if progress.get('status')==READY:
            # The existing uploader omits expected_packages in its final
            # receipt. Count89 plus the immutable manifest/API checks are
            # required; the missing optional display field is not invented.
            require(progress.get('verified_assets')==89 and progress.get('expected_packages',89)==89,'Ready status lacks exact89 completion counts')
            break
        require(progress.get('status') not in ('failed','error','upload_error','cancelled','stopped'),'Upload entered explicit failure/cancellation state; no publication')
        time.sleep(45)
    require(all(sha(p)==digest for p,digest in pins.items()),'Bound package/metadata/body/protection checker changed while waiting')
    package=read(package_path);metadata=read(meta_path)
    require(metadata['scientific_source_commit']==SOURCE and metadata['publication_metadata_commit']==METADATA,'Metadata commit binding differs')
    before,release=gates(package,metadata,'before_publication')
    require(release['draft'] is True,'New release is already published; no implicit repeat publication')
    body=notes_path.read_text().rstrip()+(
        '\n\nPublication metadata commit: '+METADATA+'. Current default-branch main commit: '+MAIN+'.\n'
        'Saved-table check scope: PermWorld and GraphWorld fixed-transform per-length and complete-weight aggregation/designated comparisons; '
        'MatrixGroupWorld primary R1/R3 B0/Bt/dc/du/Delta decomposition and the retained numerical verification status. '
        'These are saved-CSV/JSON checks, not new training or model forwards. The independent public clean-clone/minimal-asset byte-readback audit '
        'runs after publication and is recorded separately; this release text does not claim that later download has already completed.\n'
        'MatrixGroupWorld remains 64 passed, 2 failed, 6 undefined_consistent; completed_with_verification_caveat and scientific_analysis_complete=false. '
        'The two failures concern the original fixed unit001 R1/n6/first64 output-probability kNN subset. No tolerance, metric or original result is changed.\n')
    payload=dict(tag_name=TAG,target_commitish=SOURCE,draft=False,prerelease=True,make_latest='false',body=body)
    request=OUT/'publish_authorized_new_release_request.json';write(request,payload)
    require(set(payload)=={'tag_name','target_commitish','draft','prerelease','make_latest','body'},'Unexpected PATCH scope')
    require(not (B/'publication_finalize_cancel.json').exists(),'Cancellation appeared before the only authorized PATCH')
    published=api(f'repos/{REPO}/releases/{RID}',method='PATCH',payload=request);PATCHED=True
    write(OUT/'publish_authorized_new_release_response.json',published)
    require(published['id']==RID and published['tag_name']==TAG and published['draft'] is False and published['prerelease'] is True,'PATCH response identity/state differs')
    require(published['target_commitish']==SOURCE,'PATCH did not retain the scientific source target')
    after,_=gates(package,metadata,'after_publication')
    require(after['latest_release_id']==before['latest_release_id'],'Publication changed the previous latest release')
    public=public_get()
    live=dict(status=SUCCESS,public=True,repository=REPO,release_id=RID,tag_name=TAG,
        scientific_source_commit=SOURCE,tag_commit=after['tag_commit'],publication_metadata_commit=METADATA,
        remote_default_branch='main',remote_main_sha=after['remote_main_sha'],draft=False,prerelease=True,
        make_latest_requested='false',previous_latest_release_id=before['latest_release_id'],latest_release_unchanged=True,
        package_assets_provider_verified=89,metadata_assets_provider_verified=2,unique_asset_names=91,
        package_asset_bytes=sum(a['bytes'] for a in package['assets']),metadata_asset_bytes=sum(a['bytes'] for a in metadata['files']),
        old_releases_tags_unchanged=True,protected_old_release_count=3,protected_old_tag_count=3,
        release_html_url=public['html_url'],unauthenticated_public_tag_HTTP_status=public['HTTP_status'],public_tag_GET=public,
        recorded_utc=stamp(),elapsed_wait_and_finalization_seconds=time.monotonic()-start,
        package_manifest_sha256=pins[str(package_path)],metadata_binding_sha256=pins[str(meta_path)],
        publication_request_sha256=sha(request),original_release_notes_sha256=pins[str(notes_path)],
        before_provider_audit_path=str(OUT/'provider_gates_before_publication.json'),after_provider_audit_path=str(OUT/'provider_gates_after_publication.json'),
        MatrixGroupWorld_status='completed_with_verification_caveat',Matrix_scientific_analysis_complete=False,
        Matrix_numerical_status={'passed':64,'failed':2,'undefined_consistent':6},
        audit_scope='API asset identity/state/size/digest/URL and unauthenticated public release GET. Provider digest is not a claim of full byte readback; separate clean-clone/minimal readback audit follows.',
        Git_or_old_release_tag_mutations=False,new_model_forward_or_training=False,new_scientific_metrics=False)
    write(B/'publication_release_live.json',live)
    print(json.dumps({'status':live['status'],'release_html_url':live['release_html_url'],'package_assets_provider_verified':89,'metadata_assets_provider_verified':2}),flush=True)

if __name__=='__main__':
    try:main()
    except Exception as exc:
        OUT.mkdir(parents=True,exist_ok=True)
        value=dict(status='publication_finalization_failed_no_success_claim',recorded_utc=stamp(),release_id=RID,tag_name=TAG,
                   only_new_release_PATCH_performed=PATCHED,error_type=type(exc).__name__,error=str(exc),
                   old_release_tag_mutations_attempted=False,asset_deletion_or_retry_or_training_attempted=False,
                   corrective_actions_require_root_review=True)
        write(B/'publication_release_live_failed_v2.json',value)
        print(json.dumps(value),flush=True)
        raise SystemExit(2)
