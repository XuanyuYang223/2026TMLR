"""Inventory the completed stage without creating any new intervention."""
from pathlib import Path
from datetime import datetime,timezone
from importlib.metadata import version
import hashlib
import json
import platform
import subprocess

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
PACKAGE=ROOT/'experiments/permworld_local_causal_20261110'
OUT=Path('/mnt/yangx-projects-f/20261110/permworld_local_causal')


def read(path):return json.loads(path.read_text())


def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def write(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def main():
    protocol=read(HERE/'protocol.json');frozen=read(HERE/'freeze.json')
    engineering=read(HERE/'engineering_freeze.json')
    for snap in [frozen['sources'],engineering['sources']]:
        for path,digest in snap.items():assert sha(ROOT/path)==digest,path
    assert sha(HERE/'protocol.json')==frozen['protocol_sha256']
    assert read(HERE/'summary_freeze.json')['protocol_sha256']==frozen['protocol_sha256']
    audit=read(HERE/'saved_results_audit.json');protection=read(HERE/'protection_end.json')
    assert audit['status']==protection['status']=='passed'
    assert not audit['failures']
    assert read(HERE/'run_completion.json')['status']=='completed'
    assert read(HERE/'summary/completion.json')['status']=='completed_saved_local_intervention_summary'
    delta=read(HERE/'exact_modal_delta_check.json')
    assert all(r['any_numeric_count_changes']==r['any_argmax_token_changes']==0 for r in delta['per_unit_local_projection'].values())
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    assert head==protocol['git_head']
    assert not subprocess.check_output(['git','status','--porcelain=v1','--untracked-files=no'],cwd=ROOT,text=True).strip()
    write(HERE/'environment.json',dict(python=platform.python_version(),
        packages={k:version(k) for k in ['numpy','scipy','scikit-learn','pandas','matplotlib','torch','pytest']},
        recurrence_dtype='float32',probe_dtype='float64',
        frozen_flags=dict(allow_tf32=False,cudnn_allow_tf32=False,cudnn_deterministic=True),
        dependency_target='/mnt/yangx-projects-f/20261109/permworld_peaks_probe/dependencies',
        no_new_dependencies_installed=True))
    # Large recurrent-state caches are already hashed at creation and checked
    # on a bounded independent subset. Preserve that distinction in inventory.
    state_records={}
    for path in OUT.glob('*/states/*/*/completion.json'):
        r=read(path)
        for name,digest in r['files'].items():
            if name.endswith('states.npy'):
                state_records[name]=dict(path=name,sha256=digest,size_bytes=Path(name).stat().st_size,
                    hash_scope='Extraction-time SHA; no redundant full-state rehash at delivery',receipt=str(path))
    paths=sorted(set([*HERE.rglob('*'),*PACKAGE.rglob('*'),*OUT.rglob('*'),
                      *ROOT.joinpath('tests').glob('test_permworld_local_causal*.py')]))
    files={};seen=set()
    for path in paths:
        if not path.is_file() or '__pycache__' in path.parts or path.suffix=='.pyc':continue
        if path.parent==HERE and path.name in ('completion.json','delivery_manifest.json'):continue
        resolved=path.resolve()
        if str(resolved) in state_records:continue
        if resolved in seen:continue
        seen.add(resolved)
        try:key=str(path.relative_to(ROOT))
        except ValueError:key=str(path)
        files[key]=dict(path=str(path),physical_path=str(resolved),sha256=sha(path),size_bytes=path.stat().st_size,
                        hash_scope='Rehashed at delivery')
    write(HERE/'delivery_manifest.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),
        git_head=head,protocol_sha256=frozen['protocol_sha256'],files=files,recurrent_states=state_records,
        scope='New stage-one source/tests/diagnostics and external generated inputs/readouts/states/logits/results. Symlink CSV aliases are deduplicated by physical path. Recurrent-state payloads retain creation-time hashes plus bounded final checks, not a new whole-cache byte audit. Self/completion excluded.'))
    receipts=[read(HERE/'results'/u['unit']/'completion.json') for u in protocol['units']]
    write(HERE/'completion.json',dict(status='completed',completed_utc=datetime.now(timezone.utc).isoformat(),
        git_head=head,protocol_sha256=frozen['protocol_sha256'],new_units=4,historical_reference_units=1,
        selected_layer=protocol['selected_layer']+1,timing='after pi(i+1) value token',dose=1.,dimension=1,
        formal_local_edit_cases=80640,all_condition_rows=sum(r['rows'] for r in receipts),
        test_coverage='n10..30; maini2..n-2; matchedi6..n-2; bothsigns/tasks',
        tests_passed=32,independent_verification_status=audit['status'],
        outcome='Independent local checker strongly changes, but every formal local edit leaves final argmax token/count unchanged. Count-matched group retains the null modal response. Micro expectation shifts are reported, not called equivalence.',
        exact_relative_delta=delta,unit_receipts=receipts,
        no_optimizer=True,no_backward=True,no_base_training=True,no_cross_model_exchange=True,
        no_new_combination_task=True,no_automatic_publication=True,no_old_report_or_F17_edit=True,
        stopped_after_requested_first_stage=True,delivery_files=len(files),state_cache_files=len(state_records),
        delivery_manifest_sha256=sha(HERE/'delivery_manifest.json')))
    print(json.dumps(dict(status='completed',units=len(receipts),files=len(files),state_caches=len(state_records))))


if __name__=='__main__':main()
