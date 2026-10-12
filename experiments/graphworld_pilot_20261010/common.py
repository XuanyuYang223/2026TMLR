"""Fixed sources, artifact identities and side-effect-free true-step0 loading."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from experiments.permworld_encoding_archdev_20261009.common import model,device_setup,metrics
from experiments.permworld_encoding.core import WORKSPACE,sha,json_write,parameter_hash
from experiments.permworld_encoding.train import atomic_checkpoint,rng_hash,rng_state,restore_rng
from .core import *

ROOT=Path('/mnt/yangx-projects-f/20261010/graphworld_pilot')
DIAG=WORKSPACE/'diagnostics/graphworld_pilot_20261010'
CONFIG=DIAG/'protocol.json'
SOURCE_ROOT=Path('/mnt/yangx-projects-f/20261009/permworld_new_artifacts/encoding_archdev')
SOURCE_CONFIG=SOURCE_ROOT/'protocol.json'

def read(path):return json.loads(Path(path).read_text())
def csv_write(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def payload_sha(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def verify():
    c=read(CONFIG);f=read(DIAG/'freeze_manifest.json')
    assert sha(CONFIG)==f['protocol_sha256']
    for path,h in f['source_sha256'].items():assert sha(path)==h,('Source changed',path)
    for path,h in c['source_artifacts_sha256'].items():assert sha(path)==h,('Original dependency changed',path)
    assert c['training']['updates']==20000 and len(c['training_manifest'])==4 and c['model']['vocab_size']==188
    return c

def source_version():return payload_sha(read(DIAG/'freeze_manifest.json')['source_sha256'])

def initial(c):
    """Never construct or reconstruct an initial checkpoint."""
    path=Path(c['initial_checkpoint_path']);assert path.is_file()
    assert sha(path)==c['initial_checkpoint_sha256']
    saved=torch.load(path,map_location='cpu',weights_only=False)
    assert saved['parameter_sha256']==c['initial_parameter_sha256']
    assert saved['vocabulary']==TOKEN_TO_ID and saved['model_config']==c['model']
    return saved

def starting_rng(c):
    path=Path(c['starting_rng_path']);assert path.is_file() and sha(path)==c['starting_rng_sha256']
    saved=torch.load(path,map_location='cpu',weights_only=False)
    assert saved['parameter_sha256']==c['initial_parameter_sha256'] and rng_hash(saved['rng'])==c['starting_rng_hash']
    return saved['rng']

def load_data(c,splits=('train','validation')):
    audit=read(ROOT/'world/data_audit.json')
    assert sha(ROOT/'world/data.npz')==audit['data_sha256']
    assert sha(ROOT/'world/batch_indices.npy')==audit['batch_indices_sha256']
    with np.load(ROOT/'world/data.npz',allow_pickle=False) as ar:
        data={k:ar[k] for k in ar.files if any(k.startswith(s+'_') for s in splits)}
    return data,np.load(ROOT/'world/batch_indices.npy',mmap_mode='r'),audit

def check_protected():
    protection=read(DIAG/'protected_files.json')
    for path,h in protection['files'].items():assert sha(path)==h,('Protected old artifact changed',path)
    return len(protection['files'])
