"""Inventory completed local artifacts, preserving the scope of hash checks."""
from pathlib import Path
from datetime import datetime, timezone
from importlib.metadata import version
import hashlib
import json
import platform
import subprocess

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
PACKAGE=ROOT/'experiments/permworld_exceedance_alignment_20261110'
OUT=Path('/mnt/yangx-projects-f/20261110/permworld_exceedance_alignment')


def read(path):
    return json.loads(path.read_text())


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def write(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def main():
    protocol=read(HERE/'protocol.json')
    main_freeze=read(HERE/'freeze.json')
    token_freeze=read(HERE/'token_results/freeze.json')
    audit=read(HERE/'final_verification.json')
    protection=read(HERE/'protection_end.json')
    token_done=read(HERE/'token_results/run_completion.json')
    assert audit['status']==protection['status']=='passed'
    assert not audit['failures'] and not protection['failed_files']
    assert token_done['status']=='complete'
    assert token_done['units']==[u['unit'] for u in protocol['units']]
    for snapshot in [main_freeze['source_sha256'],token_freeze['source_sha256']]:
        for name,digest in snapshot.items():
            path=Path(name) if Path(name).is_absolute() else ROOT/name
            assert sha(path)==digest,path
    assert sha(HERE/'protocol.json')==main_freeze['protocol_sha256']
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    assert head==protocol['git_head']
    assert not subprocess.check_output(['git','status','--porcelain=v1','--untracked-files=no'],cwd=ROOT,text=True).strip()
    environment=dict(python=platform.python_version(),packages={name:version(name) for name in [
        'numpy','scipy','scikit-learn','pandas','matplotlib','torch','pytest']},
        dependency_target='/mnt/yangx-projects-f/20261109/permworld_peaks_probe/dependencies',
        no_dependencies_installed=True)
    write(HERE/'environment.json',environment)
    # Token-state payloads already have creation-time hashes and a separate
    # fixed-subset/header audit. Do not re-read all ~40GB merely for inventory.
    feature_root=OUT/'token_diagnostic/features'
    states={}
    for receipt_path in sorted(feature_root.rglob('completion.json')):
        receipt=read(receipt_path)
        for item in receipt['shards']:
            path=receipt_path.parent/item['filename']
            states[str(path)]=dict(path=str(path),sha256=item['sha256'],size_bytes=path.stat().st_size,
                digest_scope='Computed at extraction completion; not rehashed by delivery inventory.',
                extraction_receipt=str(receipt_path),shape=item['shape'],dtype=item['dtype'])
    files={}
    paths=set([*HERE.rglob('*'),*PACKAGE.rglob('*'),
               *ROOT.joinpath('tests').glob('test_permworld_exceedance*.py'),*OUT.rglob('*')])
    for path in sorted(paths):
        if not path.is_file() or '__pycache__' in path.parts or path.suffix=='.pyc':
            continue
        if path.parent==HERE and path.name in ['completion.json','delivery_manifest.json']:
            continue
        if str(path) in states:
            continue
        try:
            key=str(path.relative_to(ROOT))
        except ValueError:
            key=str(path)
        files[key]=dict(path=str(path),sha256=sha(path),size_bytes=path.stat().st_size,
                        digest_scope='Rehashed at delivery inventory.')
    write(HERE/'delivery_manifest.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),
        git_head=head,main_protocol_sha256=main_freeze['protocol_sha256'],
        token_freeze_sha256=sha(HERE/'token_results/freeze.json'),
        files=files,token_state_shards=states,
        scope='New experiment files plus external scores/parameters/read-only-forward caches. Self/completion excluded. Token state digests are extraction-time records; independent final audit covers all headers and explicit fixed numerical subsets, not every payload byte.'))
    main_receipts=[]
    for unit in protocol['units']:
        name=unit['unit']
        main_receipts.append(dict(unit=name,historical=unit['historical'],
            main=read(HERE/'results'/name/'completion.json'),
            pv_score_rebuild=read(HERE/'results'/name/'pv_score_rebuild.json'),
            token=read(HERE/'token_results'/name/'completion.json')))
    write(HERE/'completion.json',dict(status='completed',completed_utc=datetime.now(timezone.utc).isoformat(),
        git_head=head,main_protocol_sha256=main_freeze['protocol_sha256'],
        token_freeze_sha256=sha(HERE/'token_results/freeze.json'),
        new_units=4,historical_reference_units=1,encodings=protocol['encodings'],lengths=protocol['lengths'],
        primary_support='full eligible',sensitivity='pair-specific original-style count matching',
        main_result='Predicted encoding crossover did not occur: PV one-line margin larger4/4; ED cycle margin smaller4/4; joint predicted directions0/4 full and matched.',
        token_result='Separately reported local explicit-edge scalar readout; never substituted for final-state endpoint.',
        no_base_training=True,main_used_only_existing_prefix_caches=True,
        token_stage_used_frozen_forwards=True,no_new_world=True,
        no_automatic_publication=True,no_paper_or_F17_edit=True,
        statistical_unit='Four independent data/initialization/training bundles; controls and positions not extra units.',
        final_verification_status=audit['status'],verification_failures=0,
        protected_file_status=protection['status'],protected_named_start_paths=protection['original_start_files'],
        whole_token_payloads_rehashed_at_delivery=0,token_state_shard_count=len(states),
        independently_rehashed_delivery_files=len(files),
        delivery_manifest_sha256=sha(HERE/'delivery_manifest.json'),
        receipts=main_receipts,stopped_after_requested_two_stages=True))
    print(json.dumps(dict(status='completed',units=len(main_receipts),files=len(files),token_shards=len(states))))


if __name__=='__main__':
    main()
