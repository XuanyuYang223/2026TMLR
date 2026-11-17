"""Frozen per-edge final-norm states; no training or optimizer construction."""
from __future__ import annotations

import gc
import hashlib
from pathlib import Path
import sys
import time

import numpy as np
import torch

from .shared import DIAG, OUT, VENDOR, LENGTHS, sha, array_sha, read, write, descriptor, raw_arrays
from .oracles import edge_token_indices

BASE = OUT / 'token_diagnostic'
CONDITIONS = {
    'source': ('exceedances', 'trained', 0),
    'target_identity': ('deficiencies', 'trained', 0),
    'target_complement': ('deficiencies', 'trained', 1),
    'target_inverse': ('deficiencies', 'trained', 2),
    'initial_inverse': ('deficiencies', 'initial', 2),
    'fixed_inverse': ('fixed_points', 'trained', 2),
}
KIND_NAMES = ('one_line_value', 'cycle_internal', 'cycle_closure', 'cycle_singleton')


def parameter_hash(model):
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous().numpy()
        for piece in (name, str(value.dtype), str(value.shape)):
            digest.update(piece.encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def native_permutations(permutations, action):
    if action == 0:
        return permutations
    if action == 1:
        return permutations.shape[1] + 1 - permutations
    if action == 2:
        return np.argsort(permutations, axis=1) + 1
    raise ValueError(action)


def points(permutations, encoding):
    records = [edge_token_indices(p, encoding) for p in permutations]
    return {
        'before': np.stack([r['before'] for r in records]),
        'after': np.stack([r['after'] for r in records]),
        'kind': np.asarray([[KIND_NAMES.index(k) for k in r['kind']] for r in records], dtype=np.int8),
    }


def directory(unit, encoding, condition, split):
    return BASE/'features'/unit/encoding/condition/split


def load_states(unit, encoding, condition, split, n):
    folder = directory(unit, encoding, condition, split)
    return np.load(folder/f'n{n}.npy', mmap_mode='r', allow_pickle=False)


def load_net(record, device):
    sys.path.insert(0, str(VENDOR))
    from infer import load_definition
    gru = load_definition(VENDOR/'archdev_common.py', 'CausalGRU')
    batch = load_definition(VENDOR/'encoding_core.py', 'batch')
    assert sha(record['checkpoint_path']) == record['checkpoint_sha256']
    checkpoint = torch.load(record['checkpoint_path'], map_location='cpu', weights_only=False)
    net = gru(read(DIAG/'protocol.json')['model']).to(device).eval()
    net.load_state_dict(checkpoint['model'], strict=True)
    before = parameter_hash(net)
    assert before == record['parameter_sha256'], ('Checkpoint parameter binding', before, record['parameter_sha256'])
    return net, batch, before


def forward_edges(net, batch, inputs, n, task, coords, device):
    captured = []
    hook = net.final_norm.register_forward_hook(lambda _m, _i, value: captured.append(value))
    try:
        with torch.inference_mode():
            ids, mask, _, end, _ = batch(inputs, np.full(len(inputs), n), task, device=device)
            net(ids, mask)
            sequence = captured.pop()
            row = torch.arange(len(inputs), device=device)[:, None]
            state = torch.stack([sequence[row, torch.as_tensor(coords[k], device=device)]
                                 for k in ('before', 'after')], dim=2)
            return state.cpu().numpy(), sequence[torch.arange(len(inputs), device=device), end].cpu().numpy()
    finally:
        hook.remove()


def setup_torch(threads=3):
    torch.set_num_threads(threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def preflight(unit, data, encoding='one_line', count=8):
    start = time.perf_counter()
    device = setup_torch()
    record = descriptor(unit, encoding, 'exceedances', 'trained', 'validation')
    net, batch, before = load_net(record, device)
    n = LENGTHS[0]
    indices = np.flatnonzero(data['validation_lengths'] == n)[:count]
    permutations = data['validation_permutations'][indices, :n]
    coords = points(permutations, encoding)
    states, end = forward_edges(net, batch, data[f'validation_{encoding}_input'][indices, 0], n,
                               'exceedances', coords, device)
    reference, _, _ = raw_arrays(unit, encoding, 'exceedances', 'trained', 'validation')
    error = float(np.abs(end-reference[indices, 0]).max())
    after = parameter_hash(net)
    assert error <= 2e-5 and before == after
    assert states.shape == (len(indices), n, 2, 256) and np.isfinite(states).all()
    result = dict(status='passed', unit=unit['unit'], encoding=encoding, n=n, anchors=len(indices),
                  device=device, maximum_prefix_error=error, tolerance=2e-5,
                  parameter_sha256_before=before, parameter_sha256_after=after,
                  seconds=time.perf_counter()-start, input_sha256=array_sha(data[f'validation_{encoding}_input'][indices, 0]),
                  data_sha256=unit['data_sha256'], checkpoint_sha256=record['checkpoint_sha256'],
                  no_optimizer=True, no_backward=True, no_test_scoring=True)
    write(DIAG/'token_results'/'preflight.json', result)
    del net
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def extract_condition(unit, data, encoding, condition, batch_size=128):
    task, phase, action = CONDITIONS[condition]
    pending = []
    for split in ('validation', 'test'):
        folder = directory(unit['unit'], encoding, condition, split)
        complete = folder/'completion.json'
        if complete.exists():
            meta = read(complete)
            assert meta['data_sha256'] == unit['data_sha256']
            assert meta['checkpoint_sha256'] == descriptor(unit, encoding, task, phase, split)['checkpoint_sha256']
            for item in meta['shards']:
                assert sha(folder/item['filename']) == item['sha256']
            continue
        pending.append(split)
    if not pending:
        return
    device = setup_torch()
    record = descriptor(unit, encoding, task, phase, pending[0])
    net, batch, before = load_net(record, device)
    for split in pending:
        started = time.perf_counter()
        folder = directory(unit['unit'], encoding, condition, split)
        folder.mkdir(parents=True, exist_ok=True)
        reference, _, binding = raw_arrays(unit, encoding, task, phase, split)
        assert binding['checkpoint_sha256'] == record['checkpoint_sha256']
        shards, max_error = [], 0.
        for n in LENGTHS:
            indices = np.flatnonzero(data[f'{split}_lengths'] == n)
            p = native_permutations(data[f'{split}_permutations'][indices, :n], action)
            coords = points(p, encoding)
            path = folder/f'n{n}.npy'
            temporary = path.with_suffix('.partial.npy')
            target = np.lib.format.open_memmap(temporary, mode='w+', dtype=np.float32,
                                               shape=(len(indices), n, 2, 256))
            local_error = 0.
            for start in range(0, len(indices), batch_size):
                use = slice(start, min(start+batch_size, len(indices)))
                selected = indices[use]
                hidden, end = forward_edges(net, batch, data[f'{split}_{encoding}_input'][selected, action],
                                            n, task, {k:v[use] for k,v in coords.items()}, device)
                local_error = max(local_error, float(np.abs(end-reference[selected, action]).max()))
                assert local_error <= 2e-5, (unit['unit'], encoding, condition, split, n, local_error)
                assert np.isfinite(hidden).all()
                target[use] = hidden
            target.flush()
            del target
            temporary.replace(path)
            max_error = max(max_error, local_error)
            shards.append(dict(n=n, filename=path.name, shape=[len(indices),n,2,256], dtype='float32',
                               sha256=sha(path), anchor_indices_sha256=array_sha(indices),
                               input_sha256=array_sha(data[f'{split}_{encoding}_input'][indices, action]),
                               maximum_prefix_error=local_error))
        after = parameter_hash(net)
        assert before == after and sha(record['checkpoint_path']) == record['checkpoint_sha256']
        write(folder/'completion.json', dict(status='complete', unit=unit['unit'], encoding=encoding,
              condition=condition, task=task, phase=phase, action=action, split=split,
              data_sha256=unit['data_sha256'], checkpoint_sha256=record['checkpoint_sha256'],
              checkpoint_path=record['checkpoint_path'], parameter_sha256_before=before,
              parameter_sha256_after=after, oracle_sha256=sha(Path(__file__).with_name('oracles.py')),
              maximum_prefix_error=max_error, tolerance=2e-5, shards=shards,
              elapsed_seconds=time.perf_counter()-started, no_optimizer=True, no_backward=True,
              feature='FP32 final_norm sequence, native-edge order, point axis before/after'))
        print(f'TOKEN EXTRACT {unit["unit"]} {encoding} {condition} {split}: {max_error:.3g}', flush=True)
    del net
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
