"""Isolated sources and explicit stage boundaries. Importing this module is inert."""
from __future__ import annotations
import csv
import hashlib
import json
import time
from pathlib import Path

WORKSPACE=Path('/home/yangx/ICML')
ROOT=Path('/mnt/yangx-projects-f/20261011/matrixgroupworld')
DIAG=WORKSPACE/'diagnostics/matrixgroupworld_20261011'
CONFIG=DIAG/'protocol.json'
SOURCE=Path(__file__).resolve().parent

def read(path):return json.loads(Path(path).read_text())
def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def json_write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def csv_write(path,rows,fields=None):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if fields is None:fields=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def payload_sha(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def verify():
    c=read(CONFIG);f=read(DIAG/'freeze_manifest.json')
    assert sha(CONFIG)==f['protocol_sha256'],'Formal protocol changed after freeze'
    for path,h in f['source_sha256'].items():assert sha(path)==h,('Frozen implementation changed',path)
    assert len(c['training_manifest'])==8 and c['training']['updates']==20000
    assert c['model']['vocab_size']==188 and c['lengths']==[6,8,10]
    return c
def source_version():return payload_sha(read(DIAG/'freeze_manifest.json')['source_sha256'])
def gpu_deadline():return float(read(DIAG/'gpu_budget.json')['deadline_monotonic'])
def gpu_remaining():return max(0.,gpu_deadline()-time.monotonic())
