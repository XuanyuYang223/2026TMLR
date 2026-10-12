"""Checkpoint-only utilities; no optimizer, loss or training entry point."""
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent.parent
REPO = Path('/home/yangx/neurips')
sys.path.insert(0, str(WORKSPACE))
sys.path.insert(0, str(WORKSPACE / 'external/neurips/src'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(2**20), b''):
            h.update(b)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def write_csv(path, rows):
    rows = list(rows)
    with Path(path).open('w', newline='') as f:
        fields = list(dict.fromkeys(k for row in rows for k in row))
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def load_csv(path):
    with Path(path).open() as f:
        return list(csv.DictReader(f))


def centered(x, keys=None):
    x = np.asarray(x, float).copy()
    if keys is None:
        return x-x.mean(0)
    _, inv = np.unique(np.asarray(keys), axis=0, return_inverse=True)
    for group in np.unique(inv):
        ids = inv == group; x[ids] -= x[ids].mean(0)
    return x


def alignment(k, l):
    denominator = np.sqrt(np.square(k).sum()*np.square(l).sum())
    return float(np.sum(k*l)/denominator) if denominator > 1e-20 else None


def u_center(k):
    k = np.asarray(k, float).copy(); n = len(k)
    if n < 4:
        return None
    np.fill_diagonal(k, 0)
    sums = k.sum(1)
    k -= sums[:, None]/(n-2)+sums[None, :]/(n-2)
    k += sums.sum()/((n-1)*(n-2))
    np.fill_diagonal(k, 0)
    return k


def gram(x):
    x = centered(x)
    return x@x.T


def weighted_bootstrap_cka(k, l, weights):
    """Biased CKA after explicit paired resampling, via its multiplicities."""
    p = weights/weights.sum(1, keepdims=True)
    kp = p@k.T; lp = p@l.T
    def dot(a, b, ap, bp):
        return ((p@(a*b))*p).sum(1)-2*(p*ap*bp).sum(1)+(p*ap).sum(1)*(p*bp).sum(1)
    num = dot(k, l, kp, lp)
    den = np.sqrt(np.maximum(dot(k, k, kp, kp)*dot(l, l, lp, lp), 0))
    return np.divide(num, den, out=np.full(len(p), np.nan), where=den>1e-20)


def weight_hash(state):
    h=hashlib.sha256()
    for name, value in sorted(state.items()):
        a=value.detach().cpu().contiguous().numpy()
        h.update(name.encode());h.update(str(a.dtype).encode());h.update(str(a.shape).encode());h.update(a.tobytes())
    return h.hexdigest()


def complete_mean(values):
    return float(np.mean(values)) if all(v is not None and np.isfinite(v) for v in values) else None
