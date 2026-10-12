"""Exact original model/optimizer interface with separate extension artifacts."""
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import numpy as np
import torch

from experiments.permworld_encoding.core import (
    ACTIONS, ENCODINGS, RELATIONS, TASKS, WORKSPACE, TOKEN_TO_ID, PROPERTY_FUNCTIONS,
    batch, decode, encode, fingerprint, orbit_key, parameter_hash, sha, loss, json_write,
)
from experiments.permworld_encoding_archdev_20261009 import common as original
from experiments.permworld_encoding.train import atomic_checkpoint, rng_hash, rng_state, restore_rng

ROOT = Path('/mnt/yangx-projects-f/20261010/permworld_eight_relations')
DIAG = WORKSPACE/'diagnostics/permworld_eight_relations_20261010'
CONFIG = DIAG/'protocol.json'
OLD_ANALYSIS = WORKSPACE/'diagnostics/permworld_controlled_comparison_20261010'
SOURCE_ROOT = original.ROOT
ARCHITECTURES = original.ARCHITECTURES
model = original.model
device_setup = original.device_setup
metrics = original.metrics
FIELDS = ['initial_correct','initial_wrong','initial_identity','trained_correct','trained_wrong','trained_identity',
          'B0','Bt','dc','dw','di','Delta']

def js(path): return json.loads(Path(path).read_text())
def rows(path):
    import csv
    with Path(path).open(newline='') as f: return list(csv.DictReader(f))
def csv_write(path, records):
    import csv
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    assert records
    keys=list(dict.fromkeys(k for r in records for k in r))
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(records)
def number(x): return float('nan') if x in (None,'','NaN','nan') else float(x)
def equal_length(values):
    assert len(values)==21
    return math.fsum(values)/21 if all(math.isfinite(x) for x in values) else float('nan')
def array_sha(a):
    a=np.ascontiguousarray(a);h=hashlib.sha256()
    h.update(str(a.dtype).encode());h.update(str(a.shape).encode());h.update(a.tobytes());return h.hexdigest()
def verify():
    c=js(CONFIG);f=js(DIAG/'training_freeze.json')
    assert sha(CONFIG)==f['protocol_sha256']
    for p,h in f['source_sha256'].items(): assert sha(p)==h,('Frozen dependency changed',p)
    return c
def source_version():
    f=js(DIAG/'training_freeze.json')
    return hashlib.sha256(json.dumps(f['source_sha256'],sort_keys=True,separators=(',',':')).encode()).hexdigest()
def load_data(c):
    assert sha(c['data_path'])==c['data_sha256']
    assert sha(c['batch_schedule_path'])==c['batch_schedule_sha256']
    with np.load(c['data_path'],allow_pickle=False) as f:data={k:f[k].copy() for k in f.files}
    return data,np.load(c['batch_schedule_path'],mmap_mode='r',allow_pickle=False)
def initial(c,arch):
    ref=c['initializations'][arch]
    assert sha(ref['path'])==ref['sha256']
    saved=torch.load(ref['path'],map_location='cpu',weights_only=False)
    assert saved['vocabulary']==dict(TOKEN_TO_ID) and saved['model_config']==c['model']
    assert saved['parameter_sha256']==ref['parameter_sha256']
    return saved,ref
def start_rng(c,arch):
    ref=c['new_task_start_rng'][arch];assert sha(ref['path'])==ref['sha256']
    saved=torch.load(ref['path'],map_location='cpu',weights_only=False)
    assert rng_hash(saved['rng'])==ref['rng_hash']==saved['rng_hash']
    assert saved['parameter_sha256']==c['initializations'][arch]['parameter_sha256']
    return saved['rng']

def selections(data,relation,split,n,control):
    ns=data[split+'_lengths']; labels=data[split+'_labels'].astype(np.int64,copy=False)
    yl=labels[:,0,TASKS.index(relation['left'])]
    yr=labels[:,:,TASKS.index(relation['right'])]-int(relation['right_label_offset'])
    keys=np.column_stack([ns,yl,yr]);use=ns==n
    if control=='answer_strata':
        _,inv,count=np.unique(keys,axis=0,return_inverse=True,return_counts=True)
        use&=count[inv]>=3
    elif control!='raw_within_length':raise ValueError(control)
    return np.flatnonzero(use),keys[use] if control=='answer_strata' else None
