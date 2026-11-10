"""Read-only adapters for the five actual PermWorld source units."""
from __future__ import annotations
from collections import OrderedDict
import csv
import hashlib
import json
import os
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
DIAG=ROOT/'diagnostics/permworld_answer_controls_20261109'
DEFAULT_OUTPUT=Path('/mnt/yangx-projects-f/20261109/permworld_answer_controls')
CONFIG=DIAG/'protocol.json'
REPETITION=ROOT/'diagnostics/permworld_repetition_20261106'


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def read(path):return json.loads(Path(path).read_text())


def safe(x):
    if isinstance(x,dict):return {str(k):safe(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [safe(v) for v in x]
    if isinstance(x,np.generic):x=x.item()
    if isinstance(x,float) and not np.isfinite(x):return None
    return x


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(safe(value),indent=2,allow_nan=False)+'\n');os.replace(tmp,path)


def csv_write(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    names=list(dict.fromkeys(k for r in rows for k in r)) or ['status']
    tmp=path.with_suffix(path.suffix+'.tmp')
    with tmp.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=names);w.writeheader();w.writerows(safe(rows))
    os.replace(tmp,path)


def csv_read(path):
    with Path(path).open(newline='') as f:return list(csv.DictReader(f))


def load_config():
    c=read(CONFIG);f=read(DIAG/'freeze.json')
    assert sha(CONFIG)==f['protocol_sha256']
    for p,h in f['source_sha256'].items():assert sha(ROOT/p)==h,('Frozen dependency changed',p)
    return c


class Sources:
    """No model constructor; cache identity is checked on first access."""
    def __init__(self,c):
        self.c=c;self.cache=OrderedDict();self.checked={}
        self.old_index=read(c['historical_activation_index'])
        self.old={(r['architecture'],r['encoding'],r['task'],r['condition'],r['split']):r for r in self.old_index['records']}
        self.old_models={(r['architecture'],r['encoding'],r['task']):r for r in read(REPETITION/'original_audit.json')['models']}
        self.unit_map={r['unit']:r for r in c['units']}

    def data_path(self,unit):return Path(self.unit_map[unit]['data_path'])

    def data(self,unit):
        path=self.data_path(unit);self.verify_file(path,self.unit_map[unit]['data_sha256'])
        with np.load(path,allow_pickle=False) as f:
            return {k:f[k].copy() for k in f.files if k.endswith(('_labels','_lengths','_permutations'))}

    def verify_file(self,path,expected):
        path=Path(path);sig=(path.stat().st_size,path.stat().st_mtime_ns)
        if str(path) not in self.checked:
            assert sha(path)==expected,('Input SHA mismatch',path)
            self.checked[str(path)]=dict(sha256=expected,size_bytes=sig[0],mtime_ns=sig[1])
        else:assert sig==(self.checked[str(path)]['size_bytes'],self.checked[str(path)]['mtime_ns']),('Input changed',path)

    def checkpoint(self,unit,arch,enc,task,phase='trained'):
        u=self.unit_map[unit]
        if phase=='initial':
            a=u['initializations'][arch];return Path(a['path']),a['sha256']
        if unit=='archdev001':
            a=self.old_models[arch,enc,task]['checkpoint_path'];return Path(a['path']),a['sha256']
        root=Path(u['root'])/'models'/f'{unit}_{arch}_{enc}_{task}'
        done=read(root/'completed.json');assert done['step']==done['finite_gradient_updates']==20000
        return root/'checkpoint.pt',done['checkpoint_sha256']

    def get(self,unit,arch,enc,task,phase,split,field):
        key=(unit,arch,enc,task,phase,split,field)
        if key in self.cache:self.cache.move_to_end(key);return self.cache[key]
        if unit=='archdev001':
            r=self.old[arch,enc,task,phase,split]
            if field not in r['arrays']:raise KeyError(field)
            a=r['arrays'][field];path=Path(a['path']);self.verify_file(path,a['sha256'])
            result=np.load(path,mmap_mode='r',allow_pickle=False)
        elif field in ('answer_ce','numeric_mass'):
            path=Path(self.c['output'])/'full_nll'/unit/f'{arch}_{enc}_{task}'/(field+'.npy')
            meta=read(path.parent/'completion.json');self.verify_file(path,meta['arrays'][field]['sha256'])
            result=np.load(path,mmap_mode='r',allow_pickle=False)
        else:
            path=Path(self.unit_map[unit]['root'])/'features'/f'{arch}_{enc}_{task}_{phase}_{split}.npz'
            meta=read(path.with_suffix('.json'));assert meta['signature']['phase']==phase and not meta['signature']['engineering']
            self.verify_file(path,meta['sha256'])
            with np.load(path,allow_pickle=False) as f:result=f[field].copy()
        self.cache[key]=result
        while len(self.cache)>24:self.cache.popitem(last=False)
        return result

    def feature_stack(self,unit,arch,enc,phase):
        arrays=[self.get(unit,arch,enc,t,phase,'test','prefix_final') for t in self.c['tasks']]
        assert all(a.ndim==3 and a.shape[1]==3 for a in arrays)
        assert len({a.shape for a in arrays})==1
        # Saved arrays are [anchor,action,feature]; kernels use [task,action,anchor,feature].
        return np.stack(arrays).transpose(0,2,1,3)

    def getter(self,unit):
        return lambda arch,enc,task,phase,split,field:self.get(unit,arch,enc,task,phase,split,field)
