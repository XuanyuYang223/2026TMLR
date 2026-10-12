"""Independent bit-row mathematics audit, separate from all formal inputs.

Reference arithmetic uses Python integers/XOR, rather than NumPy elimination or
NumPy multiplication from core. It never generates final analysis labels.
"""
import argparse
from collections import Counter
import itertools
import json
from math import comb
from pathlib import Path
import time

import numpy as np

from . import core


def rows(a):
    return tuple(sum(int(a[i, j]) << j for j in range(len(a))) for i in range(len(a)))


def array(r):
    n = len(r)
    return np.array([[(int(v) >> j) & 1 for j in range(n)] for v in r], dtype=np.uint8)


def rank_bits(r, n=None):
    n = len(r) if n is None else n
    basis = {}
    for value in r:
        value = int(value)
        while value:
            bit = value.bit_length() - 1
            if bit not in basis:
                basis[bit] = value; break
            value ^= basis[bit]
    return len(basis)


def multiply_bits(a, b):
    result = []
    for row in a:
        value = 0
        for j in range(len(b)):
            if (row >> j) & 1:
                value ^= b[j]
        result.append(value)
    return tuple(result)


def transpose_bits(a):
    n = len(a)
    return tuple(sum(((a[i] >> j) & 1) << i for i in range(n)) for j in range(n))


def labels_bits(a):
    col1 = sum((row & 1) for row in a)
    col2 = sum(((row >> 1) & 1) for row in a)
    xor_col = sum(((row & 1) ^ ((row >> 1) & 1)) for row in a)
    return (col1, a[0].bit_count(), xor_col, (a[0] ^ a[1]).bit_count())


def generators_bits(n):
    p = tuple(2 if i == 0 else 1 if i == 1 else 1 << i for i in range(n))
    q = tuple(3 if i == 1 else 1 << i for i in range(n))
    return p, q


def transform_bits(a, action):
    p, q = generators_bits(len(a)); qt = transpose_bits(q); at = transpose_bits(a)
    if action == 'identity': return a
    if action == 'transpose': return at
    if action == 'P_transpose': return multiply_bits(p, at)
    if action == 'Qt_transpose': return multiply_bits(qt, at)
    if action == 'right_Q': return multiply_bits(a, q)
    if action == 'right_Qt': return multiply_bits(a, qt)
    if action == 'left_Qt': return multiply_bits(qt, a)
    if action == 'left_Q': return multiply_bits(q, a)
    raise ValueError(action)


def key_bits(a):
    value = 0; n = len(a)
    for row in a:
        for j in range(n):
            value = (value << 1) | ((row >> j) & 1)
    padding = (-n * n) % 8
    return bytes([n]) + (value << padding).to_bytes((n * n + 7) // 8, 'big')


def small_group_bits():
    return tuple((a, b) for a, b in itertools.product(range(4), repeat=2) if rank_bits((a, b)) == 2)


def orbit_bits(a):
    n = len(a); hs = tuple((h[0], h[1], *(1 << i for i in range(2, n))) for h in small_group_bits())
    return {multiply_bits(multiply_bits(l, source), r)
            for source in (a, transpose_bits(a)) for l in hs for r in hs}


def exact_matrices(n):
    return tuple(a for a in itertools.product(range(1 << n), repeat=n) if rank_bits(a) == n)


def run(output=None, random_samples=32, seed=98000021):
    start = time.monotonic(); checked = 0; orbit_checked = 0; counters = Counter(); summaries = []
    rng = np.random.default_rng(seed); examples = {}; wrong_disagreements = Counter(); distributions = []

    def require(condition, name):
        nonlocal checked
        checked += 1
        if not condition:
            raise AssertionError(name)
        counters[name] += 1

    groups = {2: exact_matrices(2), 3: exact_matrices(3)}
    require(len(groups[2]) == 6, 'GL2_size_six')
    require(len(groups[3]) == 168, 'GL3_size_168')
    require({rows(h) for h in core.small_group()} == set(small_group_bits()), 'independent_H_enumeration')
    for n in core.LENGTHS:
        candidates = []; proposals = 0
        while len(candidates) < random_samples:
            a = rng.integers(0, 2, (n, n), dtype=np.uint8); proposals += 1; r = rows(a)
            require(core.rank(a) == rank_bits(r), 'independent_rank_random_proposals')
            if rank_bits(r) == n:
                candidates.append(r)
        groups[n] = tuple(candidates)
        summaries.append(dict(n=n, audit_samples=random_samples, iid_proposals=proposals,
                              independent_from_formal_worlds=True))
    for n in (2, 3, *core.LENGTHS):
        ys = []
        for ar in groups[n]:
            a = array(ar); expected = labels_bits(ar); ys.append(expected)
            require(core.rank(a) == rank_bits(ar) == n, 'independent_exact_full_rank')
            require(tuple(map(int, core.labels(a))) == expected, 'independent_four_labels')
            require(np.array_equal(core.decode(core.encode(a)), a), 'encoding_roundtrip')
            require(core.key(a) == key_bits(ar), 'independent_big_packed_key')
            p, q = generators_bits(n); identity = tuple(1 << i for i in range(n))
            require(multiply_bits(q, q) == identity, 'Q_square_identity')
            require(multiply_bits(p, p) == identity, 'P_square_identity')
            require(tuple((row & 1) ^ ((row >> 1) & 1) for row in q) ==
                    tuple(int(i == 0) for i in range(n)), 'Q_s_equals_e1')
            for action in core.ACTIONS:
                b = core.transform(a, action); br = transform_bits(ar, action)
                require(rows(b) == br, 'independent_frozen_transform')
                require(core.rank(b) == rank_bits(br) == n, 'frozen_transform_preserves_rank')
                require(tuple(map(int, core.labels(b))) == labels_bits(br), 'transformed_labels_independent')
            for r in core.RELATIONS:
                left = core.TASKS.index(r['left']); right = core.TASKS.index(r['right'])
                require(labels_bits(transform_bits(ar, r['correct']))[right] == expected[left],
                        f'{r["relation"]}_correct_formula')
                wrong = labels_bits(transform_bits(ar, r['wrong']))[right]
                formula = {'R1': sum((row >> 1) & 1 for row in ar),
                           'R2': expected[0],
                           'R3': sum((row >> 1) & 1 for row in ar),
                           'R4': ar[1].bit_count()}[r['relation']]
                require(wrong == formula, f'{r["relation"]}_wrong_formula')
                if wrong != expected[left]:
                    wrong_disagreements[r['relation']] += 1
                    examples.setdefault(r['relation'], {'n': n, 'matrix_rows': list(ar),
                                                         'left_answer': expected[left], 'wrong_answer': wrong})
            ref_members = orbit_bits(ar); primary_members = core.orbit(a)
            require({rows(b) for b in primary_members} == ref_members, 'independent_complete_72_orbit')
            canonical = min(map(key_bits, ref_members))
            require(core.orbit_key(a) == canonical, 'independent_canonical_key')
            require(len(ref_members) <= 72, 'orbit_at_most_72')
            # P,Q generate H. Closure under left/right P,Q and transpose proves
            # closure of the full finite orbit without a redundant 72^2 scan.
            for br in ref_members:
                for brr in (multiply_bits(p, br), multiply_bits(q, br),
                            multiply_bits(br, p), multiply_bits(br, q), transpose_bits(br)):
                    require(brr in ref_members, 'H_left_right_transpose_closure')
                require(core.orbit_key(array(br)) == canonical, 'canonical_invariant_all_members')
                require(rank_bits(br) == n, 'orbit_member_exact_full_rank')
                orbit_checked += 1
            # Multiplication implementation and transpose anti-automorphism.
            partner = groups[n][0]
            require(rows(core.matmul(a, array(partner))) == multiply_bits(ar, partner),
                    'independent_GF2_multiplication')
            require(transpose_bits(multiply_bits(ar, partner)) ==
                    multiply_bits(transpose_bits(partner), transpose_bits(ar)), 'transpose_anti_automorphism')
        for j, task in enumerate(core.TASKS):
            counts = Counter(y[j] for y in ys)
            require(len(counts) >= 2, 'nonconstant_audit_task_labels')
            if n in (2, 3):
                for k in range(1, n + 1):
                    require(counts[k] * ((1 << n) - 1) == len(groups[n]) * comb(n, k),
                            'exhaustive_theoretical_label_distribution')
                distributions.append(dict(n=n, task=task, total=len(groups[n]), histogram=dict(counts),
                                          theoretical='C(n,k)/(2^n-1)', exact_match=True))
    for r in core.RELATIONS:
        require(wrong_disagreements[r['relation']] > 0, f'{r["relation"]}_wrong_not_identical')
    require(int.from_bytes(core.key(np.eye(10, dtype=np.uint8))[1:], 'big').bit_length() > 64,
            'n10_key_retains_bits_above_64')
    for n in core.LENGTHS:
        require(len(core.encode(np.eye(n, dtype=np.uint8))) + 4 == n*n+n+8 <= 128,
                'teacher_forcing_grammar_within_128')
    invalid = (np.zeros((6, 6), dtype=np.uint8), np.eye(6, dtype=float),
               np.array([[1, 2], [0, 1]]), np.ones((2, 3), dtype=np.uint8),
               np.ones((1, 1), dtype=np.uint8))
    for bad in invalid:
        try:
            core.encode(bad)
        except ValueError:
            require(True, 'invalid_matrix_rejected')
        else:
            require(False, 'invalid_matrix_rejected')
    valid_ids = core.encode(np.eye(6, dtype=np.uint8))
    bad_ids = (valid_ids[:-1], (999, *valid_ids[1:]), (*valid_ids[:4], core.TOKEN_TO_ID['02'], *valid_ids[5:]))
    for bad in bad_ids:
        try:
            core.decode(bad)
        except ValueError:
            require(True, 'invalid_encoding_rejected')
        else:
            require(False, 'invalid_encoding_rejected')
    result = {'status': 'passed', 'checks': checked, 'check_counts': dict(counters),
              'exhaustive_GL_sizes': {'2': 6, '3': 168}, 'random_audit_seed': seed,
              'random_large_n': summaries, 'orbit_members_checked': orbit_checked,
              'wrong_formula_counterexamples': examples,
              'exact_label_distribution': distributions,
              'independent_implementation': 'Python integer bit rows, XOR-basis rank, XOR-selected row multiplication, independent row-major bit packing and independently enumerated GL2 subgroup.',
              'sampling_scope': 'All n2/3 invertible matrices and 32 independently seeded audit GL matrices for each n6/8/10; these are not formal inputs or final analysis labels.',
              'elapsed_process_wall_seconds': time.monotonic() - start,
              'vocabulary_size': len(core.VOCABULARY),
              'prefix_lengths': {str(n): core.prefix_length(n) for n in core.LENGTHS},
              'teacher_forcing_lengths': {str(n): core.prefix_length(n) + 4 for n in core.LENGTHS},
              'grammar': '<BOS> <SIZE> nn <ARG_START> n rows of n 00/01 tokens separated by | <ARG_END> <EXCEEDANCES> = answer <EOS>',
              'no_Torch_or_model_calls': True, 'no_formal_analysis_label_calls': True}
    if len(core.VOCABULARY) != 188:
        raise RuntimeError('Existing vocabulary is not 188 tokens')
    if output is not None:
        path = Path(output); path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    return result


def audit_world_overlap(root, output, seed=98000021, random_samples=32):
    """Record actual audit/formal intersections; never exclude or resample worlds.

    Independent random seeds do not prove disjoint inputs. Reconstruct only the
    already-fixed audit matrices, then compare their raw and canonical keys with
    saved worlds. No labels (particularly no analysis labels) are evaluated here.
    """
    root = Path(root); rng = np.random.default_rng(seed); audit = {}; proposals = {}
    for n in core.LENGTHS:
        chosen = []; count = 0
        while len(chosen) < random_samples:
            a = rng.integers(0, 2, (n, n), dtype=np.uint8); count += 1
            if rank_bits(rows(a)) == n:
                chosen.append(a)
        audit[n] = chosen; proposals[str(n)] = count
    raw = {core.key(a) for chosen in audit.values() for a in chosen}
    orbits = {core.orbit_key(a) for chosen in audit.values() for a in chosen}
    result_rows = []
    for directory in sorted((root / 'worlds').iterdir()):
        path = directory / 'data.npz'
        if not path.exists():
            continue
        with np.load(path, allow_pickle=False) as ar:
            for split in ('train', 'validation', 'analysis'):
                raw_intersection = set(); orbit_intersection = set()
                for n, a, k in zip(ar[f'{split}_lengths'], ar[f'{split}_matrices'], ar[f'{split}_orbit_ids']):
                    a = a[:int(n), :int(n)]; input_key = core.key(a); orbit_key = bytes.fromhex(str(k))
                    if input_key in raw:
                        raw_intersection.add(input_key.hex())
                    if orbit_key in orbits:
                        orbit_intersection.add(orbit_key.hex())
                result_rows.append({'unit': directory.name, 'split': split,
                                    'raw_anchor_overlap': len(raw_intersection),
                                    'canonical_orbit_overlap': len(orbit_intersection),
                                    'raw_overlap_keys': sorted(raw_intersection),
                                    'orbit_overlap_keys': sorted(orbit_intersection)})
    result = {'status': 'recorded', 'audit_seed': seed, 'audit_gl_samples_per_n': random_samples,
              'iid_proposals_per_n': proposals, 'audit_input_key_count': len(raw),
              'audit_orbit_key_count': len(orbits), 'formal_world_intersections': result_rows,
              'source_isolation': 'Different fixed seed; actual input/orbit intersections recorded here rather than inferred from seed identity.',
              'no_world_modification_or_resampling': True, 'no_label_calls': True}
    path = Path(output); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--output', required=True)
    args = p.parse_args(); r = run(args.output)
    print(json.dumps({k: r[k] for k in ('status', 'checks', 'orbit_members_checked', 'elapsed_process_wall_seconds')}))
