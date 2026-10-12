"""Read-only source/protection snapshots. No models, arrays or metrics."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

WORKSPACE = Path('/home/yangx/ICML')
HERE = Path(__file__).resolve().parent
OLD = WORKSPACE / 'diagnostics/permworld_eight_relations_20261010'
GRAPH = WORKSPACE / 'diagnostics/graphworld_pilot_20261010'

def read(p): return json.loads(Path(p).read_text())
def sha(p):
    with Path(p).open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()
def write(p, value): Path(p).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
def git():
    def call(*args):
        return subprocess.check_output(['git', *args], cwd=WORKSPACE)
    return dict(HEAD=call('rev-parse', 'HEAD').decode().strip(),
                worktree_diff=call('diff', '--binary', 'HEAD').decode(),
                index_diff=call('diff', '--cached', '--binary', 'HEAD').decode(),
                status=call('status', '--porcelain=v1').decode(),
                scope='Actual beginning/end snapshots, not a claim about all intermediate states or all workspace files.')

def scope():
    complete=read(OLD/'completion.json')
    with (OLD/'completed_checkpoint_manifest.csv').open() as f: perm=list(csv.DictReader(f))
    graph=read(OLD/'final_review/graph_completed_checkpoint_manifest.json')['records']
    assert len(perm)==64 and len(graph)==4
    assert all(int(r['step'])==20000 for r in perm+graph)
    sources=[]
    for domain, rows in [('permworld',perm),('graphworld',graph)]:
        for r in rows:
            receipt=read(r['completed_path'])
            assert receipt['step']==20000 and receipt['finite_gradient_updates']==20000
            assert receipt['checkpoint_sha256']==r['checkpoint_sha256']
            sources.append(dict(domain=domain, run_id=r['run_id'], task=r['task'],
                checkpoint_path=r['checkpoint_path'], checkpoint_sha256=r['checkpoint_sha256'],
                initial_checkpoint_path=r['initial_checkpoint_path'],
                initial_checkpoint_sha256=r['initial_checkpoint_sha256'],
                identity='Saved genuine step0; matched cache and container identity verified in the completed source audit. No constructor/seed reconstruction here.',
                step=20000))
    pp=read(OLD/'protocol.json'); gp=read(GRAPH/'protocol.json')
    pi=read(pp['evaluation_index_path'])
    gi=read('/mnt/yangx-projects-f/20261010/graphworld_pilot/evaluation_cache/index.json')
    assert len(pi['records'])==256 and len(gi['records'])==16
    assert len(pp['relations'])==8 and len(gp['relations'])==4
    value=dict(status='actual_scope_verified', registered_utc=datetime.now(timezone.utc).isoformat(),
        expected_and_actual=dict(PermWorld_models=64, PermWorld_relations=8, PermWorld_main_conditions=32,
                                 GraphWorld_models=4, GraphWorld_relations=4),
        source_completion_status=complete['status'], sources=sources,
        cache_logical_records=dict(permworld=256,graphworld=16),
        source_scope_differences=[], missing=[], new_training=0, new_forward=0,
        historical_reconstructed_initializations_included=False)
    write(HERE/'source_scope.json',value)
    return value

def start():
    assert not (HERE/'protection_start.json').exists(), 'Do not rebind the original start snapshot.'
    begun=time.monotonic(); source_scope=scope()
    ledger=read(OLD/'final_review/physical_file_manifest.json')['records']
    expected={r['path']:r['sha256'] for r in ledger}
    binding=read(OLD/'completion.json')['artifact_bindings']
    for path,m in binding.items():
        if path in expected: assert expected[path]==m['sha256'], path
        expected[path]=m['sha256']
    paths=set(expected)
    roots=[OLD,GRAPH,WORKSPACE/'experiments/permworld_eight_relations_20261010',
           WORKSPACE/'experiments/graphworld_pilot_20261010',
           WORKSPACE/'experiments/permworld_encoding',
           WORKSPACE/'experiments/permworld_encoding_archdev_20261009']
    for root in roots:
        paths.update(str(p.resolve()) for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix not in {'.pyc','.pyo'})
    paths.update(str((WORKSPACE/n).resolve()) for n in ('CURRENT_RESEARCH.txt','archive_index.csv','cleanup_candidates.csv'))
    snapshot=git(); write(HERE/'git_start.json', snapshot)
    write(HERE/'protection_start_pending.json',dict(status='hashing',files=len(paths),new_cross_task_computation_must_wait=True))
    records=[]
    for i,name in enumerate(sorted(paths)):
        p=Path(name); before=p.stat(); digest=sha(p); after=p.stat()
        assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns),name
        if name in expected: assert digest==expected[name], 'Prior bound source changed: '+name
        records.append(dict(path=name,sha256=digest,bytes=before.st_size,mtime_ns=before.st_mtime_ns,
                            compared_to_prior_bound_SHA=name in expected))
        if (i+1)%400==0: print(f'protection start {i+1}/{len(paths)}',flush=True)
    result=dict(status='passed',registered_utc=datetime.now(timezone.utc).isoformat(),records=records,
                files=len(records),old_bound_files_checked=len(expected),git=snapshot,
                actual_source_counts=source_scope['expected_and_actual'],wall_seconds=time.monotonic()-begun,
                protected_scope='Every reference in the previous physical scientific audit, previous final delivery bindings, complete old frozen Perm/Graph diagnostic and source directories, old encoding source, and current entry/archive files. Includes protected F17 references; excludes new supplement directory and regenerable bytecode.',
                limitation='Only the named start/end byte identities and tracked Git snapshots are checked. This does not establish every file or the entire execution history was unchanged.',
                arrays_loaded=0,models_loaded=0,forward=0,training=0)
    write(HERE/'protection_start.json',result)
    print(json.dumps(dict(status='passed',stage='start',files=len(records),wall_seconds=result['wall_seconds'])))

def end():
    assert not (HERE/'protection_end.json').exists(), 'Do not refresh a completed end snapshot.'
    begun=time.monotonic(); initial=read(HERE/'protection_start.json'); checks=[]
    for i,r in enumerate(initial['records']):
        p=Path(r['path']); digest=sha(p); checks.append(dict(path=r['path'],sha256=digest,
            unchanged=digest==r['sha256'] and p.stat().st_size==r['bytes']))
        if (i+1)%400==0: print(f'protection end {i+1}/{len(initial["records"])}',flush=True)
    current=git();before=initial['git']
    same=all(current[k]==before[k] for k in ('HEAD','worktree_diff','index_diff'))
    old_status=set(before['status'].splitlines()); new_status=set(current['status'].splitlines())
    expected_new='?? diagnostics/representation_pairing_review_20261011/'
    unexplained_added=sorted(new_status-old_status-{expected_new}); removed=sorted(old_status-new_status)
    passed=all(x['unchanged'] for x in checks) and same and not unexplained_added and not removed
    write(HERE/'git_end.json',current)
    result=dict(status='passed' if passed else 'failed',registered_utc=datetime.now(timezone.utc).isoformat(),
        protection_start_sha256=sha(HERE/'protection_start.json'),files=len(checks),checks=checks,
        all_protected_bytes_match_start=all(x['unchanged'] for x in checks),tracked_Git_snapshots_equal=same,
        unexplained_added_status=unexplained_added,removed_status=removed,wall_seconds=time.monotonic()-begun,
        training=0,forward=0,models_loaded=0,
        limitation=initial['limitation'])
    write(HERE/'protection_end.json',result); assert passed, 'Protected dependencies or Git changed'
    print(json.dumps(dict(status=result['status'],stage='end',files=len(checks),wall_seconds=result['wall_seconds'])))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['start','end'],required=True)
    {'start':start,'end':end}[parser.parse_args().stage]()
