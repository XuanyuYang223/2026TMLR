"""Read a completed repetition unit's saved validation/test feature files."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from analyze import ENCODINGS


def sha(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def load_cache(unit, data_sha, encoding, task, phase, split):
    path = unit / 'features' / f'gru_{encoding}_{task}_{phase}_{split}.npz'
    meta = json.loads(path.with_suffix('.json').read_text())
    signature = meta['signature']
    expected = dict(data_sha256=data_sha, architecture='gru', encoding=encoding,
                    task=task, phase=phase, split=split, engineering=False,
                    point='input end before task token, final_norm')
    for key, value in expected.items():
        if signature.get(key) != value:
            raise ValueError(f'{path}: invalid {key} in metadata')
    if sha(path) != meta['sha256']:
        raise ValueError(f'{path}: feature payload hash mismatch')
    with np.load(path, allow_pickle=False) as f:
        return f['prefix_final'], f['prediction'], signature['checkpoint_sha256']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--unit-dir', required=True, type=Path)
    parser.add_argument('--features', required=True, type=Path)
    args = parser.parse_args()
    unit = args.unit_dir
    data_sha = sha(unit / 'data.npz')
    audit = json.loads((unit / 'data_audit.json').read_text())
    if data_sha != audit['data_sha256']:
        raise ValueError(f'{unit}: data payload hash mismatch')
    args.features.mkdir(parents=True, exist_ok=True)
    with np.load(unit / 'data.npz', allow_pickle=False) as data:
        splits = {s: len(data[s + '_lengths']) for s in ('validation', 'test')}

    for enc in ENCODINGS:
        for name in ('initial', 'peaks', 'valleys', 'descents', 'recoils'):
            task, phase = ('peaks', 'initial') if name == 'initial' else (name, 'trained')
            arrays, hashes = {}, set()
            for split, n in splits.items():
                arrays[f'{split}_indices'] = np.arange(n)
                if split == 'validation' and task in ('descents', 'recoils'):
                    continue
                features, prediction, digest = load_cache(unit, data_sha, enc, task, phase, split)
                if features.shape != (n, 3, 256) or prediction.shape != (n, 3):
                    raise ValueError(f'{unit}/{enc}/{task}/{phase}/{split}: feature shape mismatch')
                hashes.add(digest)
                if split == 'validation' or name == 'peaks':
                    actions = (0,)
                elif name in ('descents', 'recoils'):
                    actions = (1,)
                else:
                    actions = (0, 1, 2)
                for a in actions:
                    arrays[f'{split}_{a}_hidden'] = features[:, a]
                    arrays[f'{split}_{a}_prediction'] = prediction[:, a]
            if len(hashes) != 1:
                raise ValueError(f'{unit}/{enc}/{name}: checkpoint changed between splits')
            arrays['checkpoint_sha256'] = np.asarray(hashes.pop())
            output = args.features / f'{enc}-{name}.npz'
            partial = output.with_suffix('.partial.npz')
            np.savez(partial, **arrays)
            partial.replace(output)
            print(output, flush=True)


if __name__ == '__main__':
    main()
