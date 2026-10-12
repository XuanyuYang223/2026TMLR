"""Exact GL(n,F2) objects; row-major tokens from the unchanged 188-token vocabulary.

No Torch import or model execution is needed for mathematics or data preparation.
Weight is ordinary integer Hamming weight, never a sum modulo two.
"""
from functools import lru_cache
import hashlib
import itertools
from pathlib import Path
import sys

import numpy as np

WORKSPACE = Path(__file__).resolve().parents[2]
UPSTREAM = WORKSPACE / 'external/neurips/src'
if str(UPSTREAM) not in sys.path:
    sys.path.insert(0, str(UPSTREAM))
from neurips_permutations.passage import VOCABULARY, TOKEN_TO_ID, ID_TO_TOKEN

TASKS = ('F1', 'F2', 'F3', 'F4')
LENGTHS = (6, 8, 10)
ACTIONS = ('identity', 'transpose', 'P_transpose', 'Qt_transpose',
           'right_Q', 'right_Qt', 'left_Qt', 'left_Q')
TASK_TOKENS = {task: '<EXCEEDANCES>' for task in TASKS}
RELATIONS = (
    dict(relation='R1', left='F1', right='F2', correct='transpose', wrong='P_transpose',
         main=True, mapping='identity', relation_type='transpose'),
    dict(relation='R2', left='F3', right='F4', correct='transpose', wrong='Qt_transpose',
         main=False, mapping='identity', relation_type='transpose'),
    dict(relation='R3', left='F1', right='F3', correct='right_Q', wrong='right_Qt',
         main=True, mapping='identity', relation_type='fixed_right_action'),
    dict(relation='R4', left='F2', right='F4', correct='left_Qt', wrong='left_Q',
         main=False, mapping='identity', relation_type='fixed_left_action'),
)


def binary(a, square=True):
    a = np.asarray(a)
    if a.ndim != 2 or (square and (a.shape[0] != a.shape[1] or len(a) < 2)):
        raise ValueError('Expected a square matrix of order at least two')
    if a.dtype.kind not in 'bui' or not np.isin(a, (0, 1)).all():
        raise ValueError('Expected exact binary integer entries')
    return a.astype(np.uint8, copy=False)


def rank(a):
    """Exact XOR elimination; no floating point rank or tolerance."""
    m = binary(a, square=False).copy()
    pivot = 0
    for j in range(m.shape[1]):
        candidates = np.flatnonzero(m[pivot:, j])
        if not len(candidates):
            continue
        k = pivot + int(candidates[0])
        m[[pivot, k]] = m[[k, pivot]]
        rows = np.flatnonzero(m[:, j])
        rows = rows[rows != pivot]
        m[rows] ^= m[pivot]
        pivot += 1
        if pivot == m.shape[0]:
            break
    return pivot


def validate(a):
    a = binary(a)
    if rank(a) != len(a):
        raise ValueError('Expected an invertible matrix over F2')
    return a


def matmul(a, b):
    a, b = binary(a, square=False), binary(b, square=False)
    if a.shape[1] != b.shape[0]:
        raise ValueError('Incompatible GF2 multiplication shapes')
    # The frozen largest inner dimension is 10; uint16 avoids uint8 overflow.
    return ((a.astype(np.uint16) @ b.astype(np.uint16)) & 1).astype(np.uint8)


def generators(n):
    if not isinstance(n, (int, np.integer)) or n < 2:
        raise ValueError('n must be an integer at least two')
    p = np.eye(n, dtype=np.uint8)
    p[[0, 1]] = p[[1, 0]]
    q = np.eye(n, dtype=np.uint8)
    q[1, 0] = 1
    return p, q


def labels(a):
    a = validate(a)
    return np.array((a[:, 0].sum(), a[0].sum(),
                     np.bitwise_xor(a[:, 0], a[:, 1]).sum(),
                     np.bitwise_xor(a[0], a[1]).sum()), dtype=np.int16)


def transform(a, action):
    a = validate(a)
    b = a.copy()
    if action == 'identity':
        return b
    if action == 'transpose':
        return b.T.copy()
    if action == 'P_transpose':
        b = b.T.copy(); b[[0, 1]] = b[[1, 0]]
    elif action == 'Qt_transpose':
        b = b.T.copy(); b[0] ^= b[1]
    elif action == 'right_Q':
        b[:, 0] ^= b[:, 1]
    elif action == 'right_Qt':
        b[:, 1] ^= b[:, 0]
    elif action == 'left_Qt':
        b[0] ^= b[1]
    elif action == 'left_Q':
        b[1] ^= b[0]
    else:
        raise ValueError(f'Unknown frozen action: {action}')
    return b


def mapped_right(y, n, relation):
    return np.asarray(y)


@lru_cache(None)
def small_group():
    result = []
    for values in itertools.product((0, 1), repeat=4):
        a = np.array(values, dtype=np.uint8).reshape(2, 2)
        if rank(a) == 2:
            a.flags.writeable = False
            result.append(a)
    if len(result) != 6:
        raise AssertionError('GL(2,F2) must have six elements')
    return tuple(result)


def orbit_candidates(a):
    """All 72 LAR/L A^T R candidates, with duplicates deliberately retained."""
    a = validate(a)
    h = np.stack(small_group()).astype(np.uint16)
    result = []
    for source in (a, a.T):
        right = np.repeat(source[None], 6, axis=0)
        right[:, :, :2] = np.matmul(source[:, :2].astype(np.uint16)[None], h) & 1
        both = np.tile(right, (6, 1, 1))
        left = np.repeat(h, 6, axis=0)
        both[:, :2] = np.matmul(left, both[:, :2].astype(np.uint16)) & 1
        result.append(both)
    return np.concatenate(result, axis=0).astype(np.uint8, copy=False)


def key(a):
    a = binary(a)
    # Big endian row-major bits, zero padded at the low end. Python bytes/int
    # retain all 100 bits for n10; no uint64 truncation is permitted.
    return bytes([len(a)]) + np.packbits(a.reshape(-1), bitorder='big').tobytes()


def orbit(a):
    members = {}
    for b in orbit_candidates(a):
        members[key(b)] = b
    return tuple(members[k] for k in sorted(members))


def orbit_key(a):
    candidates = orbit_candidates(a)
    packed = np.packbits(candidates.reshape(72, -1), axis=1, bitorder='big')
    return bytes([len(a)]) + min(row.tobytes() for row in packed)


def orbit_id(a):
    return orbit_key(a).hex()


def input_id(a):
    return hashlib.sha256(key(a)).hexdigest()


def prefix_length(n):
    return int(n) ** 2 + int(n) + 4


def encode(a):
    a = validate(a); n = len(a)
    if n not in (2, 3, *LENGTHS):
        raise ValueError('Only audit n2/3 and frozen n6/8/10 are supported')
    tokens = ['<BOS>', '<SIZE>', f'{n:02d}', '<ARG_START>']
    for i in range(n):
        tokens.extend(f'{int(v):02d}' for v in a[i])
        if i != n - 1:
            tokens.append('|')
    tokens.append('<ARG_END>')
    ids = tuple(TOKEN_TO_ID[t] for t in tokens)
    if len(ids) != prefix_length(n) or len(ids) + 4 > 128:
        raise AssertionError('Invalid or truncated MatrixGroupWorld grammar')
    return ids


def decode(ids):
    try:
        tokens = tuple(ID_TO_TOKEN[int(t)] for t in ids)
    except (ValueError, TypeError, KeyError):
        raise ValueError('Invalid token IDs') from None
    if len(tokens) < 5 or tokens[:2] != ('<BOS>', '<SIZE>') or tokens[3] != '<ARG_START>':
        raise ValueError('Invalid prefix boundaries')
    try:
        n = int(tokens[2])
    except ValueError:
        raise ValueError('Invalid matrix order') from None
    if n not in (2, 3, *LENGTHS) or len(tokens) != prefix_length(n) or tokens[-1] != '<ARG_END>':
        raise ValueError('Invalid length, order, or end boundary')
    rows = [[]]
    for token in tokens[4:-1]:
        if token == '|':
            rows.append([])
        elif token in ('00', '01'):
            rows[-1].append(int(token))
        else:
            raise ValueError('Expected binary entry or row separator')
    if len(rows) != n or any(len(row) != n for row in rows):
        raise ValueError('Invalid row lengths')
    a = validate(np.array(rows, dtype=np.uint8))
    if encode(a) != tuple(map(int, ids)):
        raise ValueError('Noncanonical token sequence')
    return a


def batch(prefixes, ns, task, answers=None, device='cpu'):
    """Original shifted answer/EOS supervision; query is common to all tasks."""
    import torch
    if task not in TASKS:
        raise ValueError(task)
    ns = np.asarray(ns, dtype=np.int64)
    if ns.ndim != 1 or not len(ns) or not np.isin(ns, LENGTHS).all():
        raise ValueError('Invalid batch matrix orders')
    lengths = ns * ns + ns + 4
    width = int(lengths.max()) + 2 + (2 if answers is not None else 0)
    if width > 128:
        raise ValueError('Sequence cannot be truncated')
    ids = torch.full((len(ns), width), TOKEN_TO_ID['<PAD>'], dtype=torch.long, device=device)
    target = torch.full_like(ids, -100)
    end = torch.as_tensor(lengths - 1, dtype=torch.long, device=device)
    if len(prefixes) != len(ns):
        raise ValueError('Prefix/sample count mismatch')
    for i, length in enumerate(lengths):
        row = np.asarray(prefixes[i][:int(length)], dtype=np.int64)
        if len(row) != length or row[-1] != TOKEN_TO_ID['<ARG_END>']:
            raise ValueError('Invalid prefix row')
        ids[i, :int(length)] = torch.as_tensor(row.copy(), device=device)
    rows = torch.arange(len(ns), device=device)
    ids[rows, end + 1] = TOKEN_TO_ID[TASK_TOKENS[task]]
    ids[rows, end + 2] = TOKEN_TO_ID['=']
    if answers is not None:
        answers = np.asarray(answers)
        if answers.shape != ns.shape or answers.dtype.kind not in 'iu' or ((answers < 1) | (answers > ns)).any():
            raise ValueError('Invalid GL weight labels')
        y = torch.as_tensor([TOKEN_TO_ID[f'{int(v):02d}'] for v in answers], device=device)
        ids[rows, end + 3] = y
        ids[rows, end + 4] = TOKEN_TO_ID['<EOS>']
        target[rows, end + 3] = y
        target[rows, end + 4] = TOKEN_TO_ID['<EOS>']
    return ids, ids.ne(TOKEN_TO_ID['<PAD>']), target, end, end + 2
