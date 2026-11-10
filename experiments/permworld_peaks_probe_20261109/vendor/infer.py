"""Forward existing GRU checkpoints and export read-only probe features."""

import argparse
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from passage import TOKEN_TO_ID


def load_definition(path, name):
    source = ast.parse(path.read_text(), filename=str(path))
    node = next(item for item in source.body if getattr(item, 'name', '') == name)
    namespace = dict(globals())
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace[name]


def extract(net, batch, inputs, lengths, task, device, batch_size):
    features = np.empty((len(lengths), net.config.d_model), dtype=np.float32)
    predictions = np.empty(len(lengths), dtype=np.int16)
    captured = []

    def save_hidden(_layer, _args, output):
        captured.append(output)

    hook = net.final_norm.register_forward_hook(save_hidden)
    try:
        with torch.inference_mode():
            for n in np.unique(lengths):
                rows = np.flatnonzero(lengths == n)
                for start in range(0, len(rows), batch_size):
                    use = rows[start:start + batch_size]
                    ids, mask, _, end, query = batch(inputs[use], lengths[use], task, device=device)
                    out = net(ids, mask)
                    row = torch.arange(len(use), device=device)
                    features[use] = captured.pop()[row, end].cpu().numpy()
                    predictions[use] = out[row, query].argmax(dim=-1).cpu().numpy()
    finally:
        hook.remove()
    return features, predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--features', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--threads', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lengths', nargs='+', type=int, default=list(range(10, 31)))
    parser.add_argument('--encodings', nargs='+', choices=['canonical_cycle', 'one_line'],
                        default=['canonical_cycle', 'one_line'])
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    here = Path(__file__).resolve().parent
    gru = load_definition(here / 'archdev_common.py', 'CausalGRU')
    batch = load_definition(here / 'encoding_core.py', 'batch')
    model = json.loads((args.inputs / 'protocol.json').read_text())['model']
    sources = json.loads((args.inputs / 'source_inventory.json').read_text())
    args.features.mkdir(parents=True, exist_ok=True)
    plan = {'initial': (0, 1, 2), 'peaks': (0,), 'valleys': (0, 1, 2),
            'descents': (1,), 'recoils': (1,)}
    with np.load(args.inputs / 'data.npz', allow_pickle=False) as raw:
        data = {key: raw[key] for key in raw.files if not key.startswith('train_')}
    assert set(args.lengths) <= set(data['test_lengths'])
    for encoding in args.encodings:
        for task, actions in plan.items():
            name = 'initial.pt' if task == 'initial' else f'{encoding}-{task}.pt'
            target = args.features / f'{encoding}-{task}.npz'
            if target.exists():
                print(f'Reusing {target}', flush=True)
                continue
            path = args.inputs / name
            with path.open('rb') as handle:
                digest = hashlib.file_digest(handle, 'sha256').hexdigest()
            if digest != sources[name]['sha256']:
                raise ValueError(f'Checkpoint SHA mismatch for {path}')
            # Hash-verified checkpoints come from the author's published release.
            checkpoint = torch.load(path, map_location='cpu', weights_only=False)
            assert checkpoint['architecture'] == 'gru'
            if task != 'initial':
                assert checkpoint['step'] == checkpoint['finite_gradient_updates'] == 20_000
                assert checkpoint['run_id'].endswith(f'_gru_{encoding}_{task}')
            net = gru(model).to(args.device).eval()
            net.load_state_dict(checkpoint['model'], strict=True)
            prompt_task = 'peaks' if task == 'initial' else task
            arrays = {'checkpoint_sha256': np.asarray(digest)}
            for split in ('validation', 'test'):
                idx = np.flatnonzero(np.isin(data[f'{split}_lengths'], args.lengths))
                arrays[f'{split}_indices'] = idx
                lengths = data[f'{split}_lengths'][idx]
                prefixes = data[f'{split}_{encoding}_input'][idx]
                for action in actions:
                    # Fitting uses only each specialist's own identity input.
                    if split == 'validation' and action != 0:
                        continue
                    if split == 'validation' and task in {'descents', 'recoils'}:
                        continue
                    hidden, prediction = extract(net, batch, prefixes[:, action], lengths,
                                                 prompt_task, args.device, args.batch_size)
                    arrays[f'{split}_{action}_hidden'] = hidden
                    arrays[f'{split}_{action}_prediction'] = prediction
            temp = target.with_suffix('.partial.npz')
            np.savez(temp, **arrays)
            temp.replace(target)
            print(f'Finished {target}', flush=True)


if __name__ == '__main__':
    main()
