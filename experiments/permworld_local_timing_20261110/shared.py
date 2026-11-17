from pathlib import Path
import csv
import hashlib
import json
import os

ROOT=Path(__file__).resolve().parents[2]
PACKAGE=Path(__file__).resolve().parent
DIAG=ROOT/'diagnostics/permworld_local_timing_20261110'
OUT=Path('/mnt/yangx-projects-f/20261110/permworld_local_timing')
PRIOR=ROOT/'diagnostics/permworld_local_causal_20261110'
PRIOR_OUT=Path('/mnt/yangx-projects-f/20261110/permworld_local_causal')
TASKS=('peaks','valleys')
SEEDS=tuple(range(1026,1034))
COMMON=('unedited','local_projection','same_label_projection',*[f'random_direction_{s}' for s in SEEDS])
AUX_LATE=tuple(f'late_layer_swap_{j}' for j in range(1,5))


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def read(path):return json.loads(Path(path).read_text())


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');os.replace(tmp,path)


def write_csv(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
