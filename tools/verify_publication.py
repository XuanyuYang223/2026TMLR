#!/usr/bin/env python3
"""Post-publication clean-clone qualification; engineering only.

Default invocation does nothing remote. --run --release-published is explicit.
No experiment imports, model calls, hidden metrics, optimizers or training.
"""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request

REPOSITORY = 'https://github.com/XuanyuYang223/2026TMLR.git'
REPOSITORY_NAME = 'XuanyuYang223/2026TMLR'
EXPECTED_TOOLS = {
    'tools/recompute_tables.py': 'b37053658f993ddbb29220b08fe10a465b8ca17c1022ce32022a2f94467d30de',
    'tools/table_checksums.json': '3871b334d0ea47d25c48a31d2c11a7250df293effa8575f0d19101dcd900b3b6',
    'tools/download_artifacts.py': '745b6c15843e71f4ca7903bd97f1501ac42e9ba5a13da1e59f34841c02b11333',
    'requirements-artifacts.txt': 'da56c02977e00d88d897af40d227d773822aa7134d30e1d40c68e1518d666026',
}
HEX40 = re.compile(r'^[0-9a-f]{40}$')
HEX64 = re.compile(r'^[0-9a-f]{64}$')

class QualificationError(Exception):
    pass

def require(condition, message):
    if not condition:
        raise QualificationError(message)

def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()

def fresh_environment():
    # No inherited auth tokens, PYTHONPATH, pip cache/config, Git URL rewrites
    # or experiment environment. A system interpreter/stdlib remains required.
    return {
        'PATH': os.defpath,
        'LANG': 'C.UTF-8', 'LC_ALL': 'C.UTF-8',
        'PYTHONNOUSERSITE': '1', 'PYTHONDONTWRITEBYTECODE': '1',
        'PIP_CONFIG_FILE': os.devnull, 'PIP_NO_CACHE_DIR': '1',
        'PIP_DISABLE_PIP_VERSION_CHECK': '1',
        'GIT_CONFIG_GLOBAL': os.devnull, 'GIT_CONFIG_NOSYSTEM': '1',
        'GIT_TERMINAL_PROMPT': '0',
    }

def command(args, cwd, log_dir, phase, audit, timeout=1800):
    start = time.monotonic()
    print('phase=' + phase + ' started', flush=True)
    log = log_dir / (phase + '.log')
    with log.open('xb') as output:
        completed = subprocess.run(args, cwd=cwd, env=fresh_environment(),
                                   stdout=output, stderr=subprocess.STDOUT,
                                   timeout=timeout, check=False)
    audit['phases'].append({'phase': phase, 'command': [str(x) for x in args],
                            'wall_seconds': time.monotonic() - start,
                            'exit_code': completed.returncode,
                            'log_path': str(log), 'log_sha256': digest(log)})
    require(completed.returncode == 0, 'Qualification phase failed: ' + phase)
    print('phase=' + phase + ' completed', flush=True)

def local_regular(root, relative):
    p = Path(relative)
    require(not p.is_absolute() and '..' not in p.parts, 'Unsafe repository-relative input')
    value = root / p
    require(value.is_file() and not value.is_symlink() and value.resolve().is_relative_to(root.resolve()),
            'Required clone input is absent, nonregular or outside the clone')
    return value

def read_json(root, relative):
    path = local_regular(root, relative)
    return json.loads(path.read_text()), {'path': relative, 'sha256': digest(path), 'bytes': path.stat().st_size}

def check_matrix_caveat(root):
    completion, a = read_json(root, 'diagnostics/matrixgroupworld_20261011/completion.json')
    numerical, b = read_json(root, 'diagnostics/matrixgroupworld_20261011/actual_numerical_verification.json')
    wanted = {'passed': 64, 'failed': 2, 'undefined_consistent': 6}
    require(completion['status'] == 'completed_with_verification_caveat' and
            completion['scientific_analysis_complete'] is False and
            completion['tables_complete'] is True and completion['formal_models'] == 8 and
            completion['independent_data_initialization_units'] == 2 and
            completion['formal_updates'] == 160000,
            'Original Matrix completeness/caveat fields changed')
    receipts = completion['receipts']
    expected_runs = {u + '_' + f for u in ('unit001', 'unit002') for f in ('F1','F2','F3','F4')}
    require(len(receipts) == 8 and {x['run_id'] for x in receipts} == expected_runs and
            all(x['step'] == 20000 and x['actual_initial_identity_verified'] is True for x in receipts),
            'Eight saved genuine Matrix 20k endpoint receipts were not preserved')
    actual = dict(Counter(x['status'] for x in numerical['records']))
    require(actual == wanted and numerical['status_counts'] == wanted and
            completion['numerical_status_counts'] == wanted and numerical['failure_count'] == 2 and
            completion['numerical_failure_count'] == 2,
            'Original numerical 64/2/6 record counts changed')
    failed = [x for x in numerical['records'] if x['status'] == 'failed']
    require({x['role'] for x in failed} == {'correct', 'wrong'} and
            all(x['unit'] == 'unit001' and x['relation'] == 'R1' and x['n'] == 6 and
                x['control'] == 'raw_within_n' and x['branch'] == 'output_prob' and
                x['phase'] == 'trained' and x['metric'] == 'knn_overlap' and
                x['fit_count'] == 64 and x['analysis_count'] == 64 and x['tolerance'] == 1e-8
                for x in failed), 'Preserved Matrix failed-check scope changed')
    require(numerical['selection_check_count'] == 288 and
            len(numerical['selection_checks']) == 288 and
            all(x['passed'] is True for x in numerical['selection_checks']),
            'Saved selector-check completeness changed')
    require(completion['stopped'] is True and completion['automatic_next_stage'] is False,
            'Saved Matrix stop boundary changed')
    return {'status': 'passed_preserved_state_check_with_original_failures', 'inputs': [a,b],
            'formal_models': 8, 'real_20k_receipts': 8, 'independent_units': 2,
            'status_counts': wanted, 'failed_records': failed,
            'scientific_analysis_complete': False,
            'scope': 'Reads saved JSON state/receipt fields only; not model, activation or training verification.'}

def check_public_release(tag):
    require(re.fullmatch(r'[A-Za-z0-9._-]+', tag) is not None, 'Unsafe release tag')
    url = 'https://api.github.com/repos/' + REPOSITORY_NAME + '/releases/tags/' + tag
    request = urllib.request.Request(url, headers={'Accept':'application/vnd.github+json',
                                                  'User-Agent':'representation-clean-clone/1'})
    try:
        with urllib.request.urlopen(request, timeout=45) as response:
            blob = response.read(16 * 1024 * 1024 + 1)
    except Exception:
        raise QualificationError('Public release API read failed; credentials/redirect URLs are not logged') from None
    require(len(blob) <= 16 * 1024 * 1024, 'Public release response exceeds its bounded size')
    release = json.loads(blob)
    require(release['draft'] is False and release['published_at'] is not None and release['tag_name'] == tag,
            'Release is not actually publicly published at the requested tag')
    return release, {'release_id': release['id'], 'tag_name': tag, 'draft': False,
                     'published_at': release['published_at'], 'api_bytes':len(blob),
                     'api_response_sha256': hashlib.sha256(blob).hexdigest(),
                     'scope':'Public metadata read; not asset-byte remote readback.'}

def check_manifest(root, source_commit, tag, expected_sha, release):
    manifest, identity = read_json(root, 'artifacts_manifest.json')
    require(identity['sha256'] == expected_sha, 'Final clone manifest SHA256 does not match the publication owner binding')
    require(manifest['schema_version'] == 1 and manifest['release_tag'] == tag and
            manifest['scientific_source_commit'] == source_commit,
            'Clone manifest schema/tag/scientific-source binding changed')
    require(manifest['repository'].rstrip('/').removesuffix('.git') in
            (REPOSITORY_NAME, 'https://github.com/' + REPOSITORY_NAME),
            'Clone manifest points at a different repository')
    require(manifest['profiles']['minimal'] and manifest['profiles']['full'], 'Required explicit artifact profile is empty')
    package_ids={p['package_id'] for p in manifest['packages']}
    require(set(manifest['profiles']['full']) == package_ids and
            set(manifest['profiles']['minimal']).issubset(package_ids), 'Minimal/full profiles do not cover the declared package IDs')
    public_assets={a['name']:a for a in release['assets']}
    require(len(public_assets)==len(release['assets']), 'Public release has duplicate asset names')
    for asset in manifest['assets']:
        name=asset['name']; public=public_assets.get(name)
        require(public is not None and public['size']==asset['bytes'] and
                public['browser_download_url']==asset['url'], 'A manifest asset is missing/mismatched in the actual public release')
        expected_url='https://github.com/'+REPOSITORY_NAME+'/releases/download/'+tag+'/'
        require(asset['url'].startswith(expected_url) and '/' not in asset['url'][len(expected_url):],
                'Manifest asset URL is outside the expected public release')
    return manifest, identity

def qualify(args):
    require(args.release_published, 'The release-published signal is required; no remote operation performed')
    require(HEX40.fullmatch(args.expected_commit or '') is not None and
            HEX40.fullmatch(args.scientific_source_commit or '') is not None and
            HEX64.fullmatch(args.expected_manifest_sha256 or '') is not None,
            'Expected main/source commits and exact manifest SHA256 must be supplied')
    require(args.python is not None and Path(args.python).is_absolute() and
            not Path(args.python).absolute().is_relative_to(Path('/home/yangx/ICML')),
            'An explicit base interpreter outside the original research workspace must be supplied')
    work = args.directory.absolute()
    require(not work.exists() and not work.is_symlink(), 'Qualification needs a brand-new directory; existing cache is not reused')
    require(not work.resolve().is_relative_to(Path('/home/yangx/ICML')),
            'Qualification directory must be outside the original research workspace')
    work.mkdir(parents=True, exist_ok=False); logs=work/'logs';logs.mkdir()
    audit={'status':'running_clean_clone_qualification','utc':datetime.now(timezone.utc).isoformat(),
           'repository':REPOSITORY,'requested_main_commit':args.expected_commit,
           'scientific_source_commit':args.scientific_source_commit,
           'release_tag':args.release_tag,'phases':[],
           'original_host_science_inputs_opened':0,'model_forwards':0,'training_updates':0,
           'qualification_directory_new':True,'child_auth_tokens_inherited':False,
           'public_api_authorization_header_used':False,
           'pip_cache_enabled':False,'inherited_python_path':False}
    report=work/'clean_clone_audit.json';start=time.monotonic()
    try:
        release,public_identity=check_public_release(args.release_tag);audit['public_release']=public_identity
        clone=work/'clone'
        command(['git','clone','--no-local','--no-tags','--depth','1','--branch','main',REPOSITORY,str(clone)],
                work,logs,'fresh_public_main_clone',audit)
        head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=clone,env=fresh_environment()).decode().strip()
        origin=subprocess.check_output(['git','remote','get-url','origin'],cwd=clone,env=fresh_environment()).decode().strip()
        require(head==args.expected_commit and origin==REPOSITORY,'Fresh clone main/origin does not match the declared final publication')
        audit['actual_clone_commit']=head
        audit['tool_hashes']={}
        for relative,expected in EXPECTED_TOOLS.items():
            value=digest(local_regular(clone,relative)); require(value==expected,'Portable tool bytes differ from their stable publication binding')
            audit['tool_hashes'][relative]=value
        manifest,identity=check_manifest(clone,args.scientific_source_commit,args.release_tag,args.expected_manifest_sha256,release)
        audit['manifest']=identity
        interpreter_code='import json,sys,ensurepip;print(json.dumps(dict(executable=sys.executable,version=sys.version,prefix=sys.prefix,base_prefix=sys.base_prefix,ensurepip_available=True)))'
        command([args.python,'-I','-B','-c',interpreter_code],work,logs,'base_interpreter_identity',audit)
        interpreter=json.loads((logs/'base_interpreter_identity.log').read_text())
        require(interpreter['prefix']==interpreter['base_prefix'], 'Use an explicit base interpreter, not an already activated virtual environment')
        audit['base_interpreter']=dict(interpreter,scope='Interpreter/stdlib only; no base scientific packages copied or inherited by the new venv.')
        venv=work/'minimal_venv'
        command([args.python,'-I','-B','-m','venv','--copies',str(venv)],work,logs,'fresh_minimal_venv',audit)
        python=str(venv/'bin/python')
        command([python,'-I','-B','-m','pip','install','--no-cache-dir','--disable-pip-version-check',
                 '-r','requirements-artifacts.txt'],clone,logs,'install_artifact_only_dependency',audit)
        command([python,'-I','-B','tools/recompute_tables.py','--self-test'],clone,logs,'portable_fixture',audit)
        download_audit=work/'public_minimal_download_audit.json'
        command([python,'-I','-B','tools/download_artifacts.py','--manifest','artifacts_manifest.json',
                 '--profile','minimal','--directory',str(work/'fresh_downloads'),
                 '--restore-dir',str(work/'fresh_restored'),'--output',str(download_audit)],
                clone,logs,'actual_public_minimal_asset_restore',audit,timeout=14400)
        downloaded=json.loads(download_audit.read_text())
        selected=set(manifest['profiles']['minimal'])
        names={a['name'] for a in manifest['assets'] if a['package_id'] in selected}
        require(downloaded['status']=='passed_selected_artifact_restore' and
                set(downloaded['remote_assets_read'])==names and
                downloaded['reused_download_assets']==[] and
                all(x['source']=='remote_network' and x['verified'] is True for x in downloaded['assets']) and
                all(x['source']=='verified_tar_object' and x['verified'] is True for x in downloaded['objects']),
                'Minimal readback reused a pre-existing cache or lacks all selected remote bytes')
        audit['actual_remote_minimal_readback']={'audit_sha256':digest(download_audit),
            'asset_names':sorted(names),'remote_bytes_read':downloaded['remote_bytes_read'],
            'selected_packages':downloaded['selected_packages'],'selected_unique_objects':downloaded['selected_unique_objects'],
            'scope':'Actual public bytes for minimal profile only; other assets metadata/size checked, not independently re-downloaded.'}
        replay=work/'saved_table_check.json'
        command([python,'-I','-B','tools/recompute_tables.py','--repo-root','.', '--output',str(replay)],
                clone,logs,'saved_table_arithmetic_and_integrity',audit)
        table=json.loads(replay.read_text())
        require(table['status']=='passed_saved_table_arithmetic_with_preserved_verification_caveat',
                'Portable saved-table arithmetic did not retain the original caveat')
        audit['saved_table_check']={'path':str(replay),'sha256':digest(replay),'status':table['status']}
        audit['matrix_preserved_caveat']=check_matrix_caveat(clone)
        require(audit['matrix_preserved_caveat']['inputs'][1]['sha256'] ==
                'c2e79900b87018a0fc23ab13565446ab67273229c4f44894ed0e322144450d9f',
                'Original frozen Matrix numerical-audit bytes changed')
        caveat=work/'matrix_preserved_caveat_check.json'
        caveat.write_text(json.dumps(audit['matrix_preserved_caveat'],indent=2,allow_nan=False)+'\n')
        audit['matrix_preserved_caveat_check_sha256']=digest(caveat)
        audit['status']='passed_actual_public_clean_clone_minimal_readback_with_preserved_scientific_caveat'
        audit['scope']='Fresh public main clone, fresh minimal venv, public minimal byte downloads/restoration, saved-table replay and unchanged saved Matrix state. No checkpoint loading, model forward, hidden-metric recomputation, statistical replication or training.'
    except QualificationError as exc:
        audit.update(status='failed_clean_clone_qualification',error=str(exc))
    except Exception as exc:
        audit.update(status='failed_clean_clone_qualification',error='Engineering failure ('+type(exc).__name__+'); raw URLs/credentials omitted')
    audit['elapsed_process_wall_seconds']=time.monotonic()-start
    report.write_text(json.dumps(audit,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    print(json.dumps({'status':audit['status'],'report':str(report),'sha256':digest(report)}),flush=True)
    return audit

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',action='store_true')
    parser.add_argument('--release-published',action='store_true')
    parser.add_argument('--directory',type=Path)
    parser.add_argument('--expected-commit')
    parser.add_argument('--scientific-source-commit')
    parser.add_argument('--expected-manifest-sha256')
    parser.add_argument('--release-tag',default='representation-selectivity-2026-10-11-caveat')
    parser.add_argument('--python',help='Explicit base Python with ensurepip; original research .venv is forbidden')
    args=parser.parse_args()
    if not args.run:
        print('Prepared only: no Git operation, remote request, artifact download, model call or training. Run only after Root confirms release publication and final main metadata push.')
        raise SystemExit(0)
    require(args.directory is not None,'An unused outside-repository directory must be supplied')
    result=qualify(args)
    raise SystemExit(not result['status'].startswith('passed_'))
