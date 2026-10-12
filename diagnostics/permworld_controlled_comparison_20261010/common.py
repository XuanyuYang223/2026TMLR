"""Read-only scientific helpers for the fixed20k controlled comparison."""
from pathlib import Path
import csv
import hashlib
import json
import math
import sys
import numpy as np

DIAG=Path(__file__).resolve().parent
WORKSPACE=DIAG.parents[1]
sys.path.insert(0,str(WORKSPACE))
from experiments.permworld_encoding_archdev_20261009.common import ROOT as SOURCE_ROOT, CONFIG as SOURCE_CONFIG, metrics
from experiments.permworld_encoding.core import ACTIONS,TASKS,TOKEN_TO_ID,sha,json_write,orbit_key,fingerprint,encode,decode
from experiments.permworld_encoding_archdev_20261009.evaluation import selections

FROOT=Path('/mnt/yangx-projects-f/20261010/permworld_controlled_comparison')
FIELDS=['initial_correct','initial_wrong','initial_identity','trained_correct','trained_wrong','trained_identity',
        'B0','Bt','dc','dw','di','Delta']
def js(path):return json.loads(Path(path).read_text())
def array_sha(a):
    a=np.ascontiguousarray(a);h=hashlib.sha256()
    h.update(str(a.dtype).encode());h.update(str(a.shape).encode());h.update(a.tobytes());return h.hexdigest()
def payload_sha(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def rows(path):
    with Path(path).open(newline='') as f:return list(csv.DictReader(f))
def csv_write(path,records):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if not records:raise ValueError('Do not silently emit an empty table')
    fields=list(dict.fromkeys(k for row in records for k in row))
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(records)
def number(x):return float(x) if x not in (None,'','NaN','nan') else float('nan')
def equal_length(values):
    assert len(values)==21
    return math.fsum(values)/21 if all(math.isfinite(x) for x in values) else float('nan')
def verify():
    protocol=js(DIAG/'protocol.json');freeze=js(DIAG/'freeze.json')
    assert sha(DIAG/'protocol.json')==freeze['protocol_sha256']
    assert sha(DIAG/'comparison_manifest_before_extraction.csv')==freeze['comparison_manifest_sha256']
    assert sha(DIAG/'protected_before.json')==freeze['protected_manifest_sha256']
    for path,digest in freeze['source_sha256'].items():assert sha(path)==digest,('Frozen source changed',path)
    return protocol
