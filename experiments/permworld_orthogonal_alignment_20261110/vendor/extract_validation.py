"""Extract only the validation actions missing from the historical probe cache."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from infer import extract, load_definition


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', required=True, type=Path)
    parser.add_argument('--original-code', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--encoding', required=True, choices=('one_line', 'canonical_cycle'))
    parser.add_argument('--threads', default=3, type=int)
    parser.add_argument('--controls', action='store_true', help='Extract complement for descents/recoils')
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    args.out.mkdir(parents=True, exist_ok=True)
    inventory = json.loads((args.inputs / 'source_inventory.json').read_text())
    model_config = json.loads((args.inputs / 'protocol.json').read_text())['model']
    source = args.inputs / 'data.npz'
    if digest(source) != inventory[source.name]['sha256']:
        raise ValueError('Released data hash mismatch')
    with np.load(source, allow_pickle=False) as data:
        lengths = data['validation_lengths']
        prefixes = data[f'validation_{args.encoding}_input']
    gru = load_definition(args.original_code / 'archdev_common.py', 'CausalGRU')
    batch = load_definition(args.original_code / 'encoding_core.py', 'batch')
    if args.controls:
        plan = [(task, f'{args.encoding}-{task}.pt', task, (1,))
                for task in ('descents', 'recoils')]
    else:
        plan = [('initial', 'initial.pt', 'valleys', (1, 2)),
                ('trained', f'{args.encoding}-valleys.pt', 'valleys', (1, 2))]
    for phase, name, task, actions in plan:
        path = args.inputs / name
        source_hash = digest(path)
        if source_hash != inventory[name]['sha256']:
            raise ValueError(f'{name}: released checkpoint hash mismatch')
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        assert checkpoint['architecture'] == 'gru'
        if phase != 'initial':
            assert checkpoint['step'] == checkpoint['finite_gradient_updates'] == 20_000
            assert checkpoint['data_sha256'] == inventory[source.name]['sha256']
            assert checkpoint['initial_checkpoint_sha256'] == inventory['initial.pt']['sha256']
        net = gru(model_config).eval()
        net.load_state_dict(checkpoint['model'], strict=True)
        target = args.out / f'{args.encoding}-{phase}.npz'
        arrays = {'checkpoint_sha256': np.asarray(source_hash),
                  'validation_indices': np.arange(len(lengths))}
        for action in actions:
            hidden, _ = extract(net, batch, prefixes[:, action], lengths,
                                task, 'cpu', 128)
            arrays[f'validation_{action}_hidden'] = hidden
            print(f'{args.encoding} {phase} action {action}', flush=True)
        temp = target.with_suffix('.partial.npz')
        np.savez(temp, **arrays)
        temp.replace(target)


if __name__ == '__main__':
    main()
