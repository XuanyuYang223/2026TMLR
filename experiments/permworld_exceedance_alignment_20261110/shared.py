"""Paths and read-only data access shared by the two separate diagnostic stages."""
from pathlib import Path
import csv
import hashlib
import json
import os
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = Path(__file__).resolve().parent
DIAG = ROOT / 'diagnostics/permworld_exceedance_alignment_20261110'
OUT = Path('/mnt/yangx-projects-f/20261110/permworld_exceedance_alignment')
PV = ROOT / 'diagnostics/permworld_orthogonal_alignment_20261110'
VENDOR = ROOT / 'experiments/permworld_peaks_probe_20261109/vendor'
ALIGN_VENDOR = ROOT / 'experiments/permworld_orthogonal_alignment_20261110/vendor'
ENCODINGS = ('canonical_cycle', 'one_line')
LENGTHS = tuple(range(10, 31))
SHUFFLE_SEEDS = tuple(range(1026, 1034))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def array_sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    os.replace(temporary, path)


def write_csv(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def read_csv(path):
    with Path(path).open() as stream:
        return list(csv.DictReader(stream))


def vendor():
    sys.path.insert(0, str(VENDOR))
    sys.path.insert(0, str(ALIGN_VENDOR))
    from analyze import fit_probe, read_length, example_auc, average, position_prior
    from alignment import rotation, shuffled_indices
    return fit_probe, read_length, example_auc, average, position_prior, rotation, shuffled_indices


def load_data(unit):
    assert sha(unit['data_path']) == unit['data_sha256']
    with np.load(unit['data_path'], allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def descriptor(unit, encoding, task, phase, split):
    matches = [r for r in unit['raw_features'] if r['encoding'] == encoding
               and r['task'] == task and r['phase'] == phase and r['split'] == split]
    if len(matches) != 1:
        raise ValueError(('Raw descriptor not unique', unit['unit'], encoding, task, phase, split, len(matches)))
    return matches[0]


def raw_arrays(unit, encoding, task, phase, split):
    record = descriptor(unit, encoding, task, phase, split)
    assert sha(record['path']) == record['sha256']
    if record['field']:
        with np.load(record['path'], allow_pickle=False) as archive:
            hidden = archive[record['field']].copy()
            prediction = archive[record.get('prediction_field') or 'prediction'].copy()
    else:
        hidden = np.load(record['path'], mmap_mode='r', allow_pickle=False)
        path = record['prediction_path']
        if record.get('prediction_sha256'):
            assert sha(path) == record['prediction_sha256']
        prediction = np.load(path, mmap_mode='r', allow_pickle=False)
    assert hidden.dtype == np.float32 and hidden.shape == tuple(record['shape'])
    return hidden, prediction, record


def split_rows(data):
    from .oracles import exceedance_mask, deficiency_mask, inverse_permutation
    _, original_split, *_ = vendor()
    result = {}
    for n in LENGTHS:
        fit, dev, eligible, _, *_ = original_split(data, n)
        val_global = np.flatnonzero(data['validation_lengths'] == n)
        test_global = np.flatnonzero(data['test_lengths'] == n)
        pv = data['validation_permutations'][val_global, :n]
        pt = data['test_permutations'][test_global, :n]
        e_val = np.array([exceedance_mask(p) for p in pv], dtype=bool)
        d_val = np.array([deficiency_mask(p) for p in pv], dtype=bool)
        e = np.array([exceedance_mask(p) for p in pt], dtype=bool)
        dt = [np.array([deficiency_mask(p) for p in values], dtype=bool)
              for values in [pt, n+1-pt, np.argsort(pt, axis=1)+1]]
        assert np.array_equal(np.take_along_axis(dt[2], pt-1, axis=1), e)
        matched = eligible & (dt[0].sum(1) == e.sum(1)) & (dt[1].sum(1) == e.sum(1))
        result[n] = dict(fit=fit, dev=dev, eligible=eligible, matched=matched,
                         val_indices=val_global, test_indices=test_global,
                         val_permutations=pv, test_permutations=pt,
                         source_val_labels=e_val, target_val_labels=d_val,
                         source_test_labels=e, target_test_labels=dt)
    return result
