"""Hash concrete previous deliverables before/after this isolated new study."""
from __future__ import annotations
import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

WORK=Path('/home/yangx/ICML');HERE=Path(__file__).resolve().parent
PREV=WORK/'diagnostics/representation_pairing_review_20261011'
def read(p):return json.loads(Path(p).read_text())
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(name,o):(HERE/name).write_text(json.dumps(o,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def git():
    def run(*args):return subprocess.check_output(['git',*args],cwd=WORK).decode()
    return dict(HEAD=run('rev-parse','HEAD').strip(),status=run('status','--porcelain=v1'),
                worktree_diff=run('diff','--binary','HEAD'),index_diff=run('diff','--cached','--binary','HEAD'))
def start():
    assert not (HERE/'protection_start.json').exists(),'Do not rebind actual task beginning'
    begun=time.monotonic();prev=read(PREV/'protection_start.json')
    expected={r['path']:r['sha256'] for r in prev['records']}
    completed=read(PREV/'completion.json')
    for name,m in completed['artifact_bindings'].items():expected[str(PREV/name)]=m['sha256']
    expected[str(PREV/'completion.json')]=sha(PREV/'completion.json')
    additional=[WORK/'experiments/permworld_scope_inventory_20261009',WORK/'experiments/permworld_eight_relations_20261010',
                WORK/'experiments/graphworld_pilot_20261010']
    for root in additional:
        if root.exists():
            for p in root.rglob('*'):
                if p.is_file() and '__pycache__' not in p.parts and p.suffix not in ('.pyc','.pyo'):
                    expected.setdefault(str(p),sha(p))
    records=[];g=git()
    for i,(path,h) in enumerate(sorted(expected.items())):
        p=Path(path);before=p.stat();actual=sha(p);after=p.stat()
        assert before.st_size==after.st_size and before.st_mtime_ns==after.st_mtime_ns,path
        assert actual==h,('Previous protected source already differs',path)
        records.append(dict(path=path,sha256=actual,bytes=before.st_size,mtime_ns=before.st_mtime_ns))
        if (i+1)%400==0:print('protection start',i+1,'/',len(expected),flush=True)
    write('git_start.json',g)
    write('protection_start.json',dict(status='passed',files=len(records),records=records,git=g,
        wall_seconds=time.monotonic()-begun,
        scope='Previous2580-file explicit protected scope (including F17 references), entire prior pairing-review delivery bindings and its completion, selected old experiment source directories.',
        limitation='Beginning/end SHA checks establish identity only for these named files, not all files or all intermediate execution states. Current research entry protected and left unchanged this task.'))
    print(json.dumps({'status':'passed','files':len(records),'wall_seconds':time.monotonic()-begun}))
def end():
    assert not (HERE/'protection_end.json').exists(),'Do not silently refresh protection completion'
    begun=time.monotonic();start=read(HERE/'protection_start.json');checks=[]
    for i,r in enumerate(start['records']):
        p=Path(r['path']);h=sha(p);checks.append(dict(path=r['path'],sha256=h,unchanged=h==r['sha256'] and p.stat().st_size==r['bytes']))
        if (i+1)%400==0:print('protection end',i+1,'/',len(start['records']),flush=True)
    now=git();same=all(now[k]==start['git'][k] for k in ('HEAD','worktree_diff','index_diff'))
    allowed={'?? experiments/matrixgroupworld_20261011/','?? diagnostics/matrixgroupworld_20261011/'}
    added=sorted(set(now['status'].splitlines())-set(start['git']['status'].splitlines())-allowed)
    removed=sorted(set(start['git']['status'].splitlines())-set(now['status'].splitlines()))
    passed=all(r['unchanged'] for r in checks) and same and not added and not removed
    write('git_end.json',now);write('protection_end.json',dict(status='passed' if passed else 'failed',files=len(checks),
        checks=checks,tracked_Git_snapshots_equal=same,unexpected_status_added=added,status_removed=removed,
        wall_seconds=time.monotonic()-begun,limitation=start['limitation']))
    assert passed,'Named protected files or tracked Git snapshots changed'
    print(json.dumps({'status':'passed','files':len(checks),'wall_seconds':time.monotonic()-begun}))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',required=True,choices=['start','end']);a=p.parse_args()
    {'start':start,'end':end}[a.stage]()
