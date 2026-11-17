"""Exact exceedance/inverse-deficiency labels and explicit-edge token points.

All mathematical positions/values are one-based. Hidden-state indices are
zero-based positions in the real encoded prefix (including BOS/SIZE/header).
"Before" means the state after the immediately preceding token; "after" is
the state after the local serialization explicitly reveals the edge. This is
not the earliest logical entailment over all valid permutation completions.
"""
from __future__ import annotations

from numbers import Integral

import numpy as np


def validate_permutation(permutation):
    p = tuple(permutation)
    if (not p or any(isinstance(v, (bool, np.bool_)) or not isinstance(v, Integral) for v in p)
            or sorted(p) != list(range(1, len(p) + 1))):
        raise ValueError("A nonempty integer permutation of 1..n is required")
    return tuple(int(v) for v in p)


def inverse_permutation(permutation):
    p = validate_permutation(permutation)
    inverse = [0] * len(p)
    for position, value in enumerate(p, start=1):
        inverse[value - 1] = position
    return tuple(inverse)


def labels(permutation):
    """Position labels; fixed points are negative in both binary tasks."""
    p = np.asarray(validate_permutation(permutation), dtype=np.int64)
    position = np.arange(1, len(p) + 1)
    return dict(exceedances=p > position, deficiencies=p < position,
                fixed_points=p == position)


def exceedance_mask(permutation):
    return labels(permutation)['exceedances']


def deficiency_mask(permutation):
    return labels(permutation)['deficiencies']


def batch_labels(permutations):
    p = np.asarray(permutations)
    if p.ndim != 2 or p.shape[1] < 1 or p.dtype.kind not in 'iu':
        raise ValueError("Expected an integer matrix of permutations [N,n]")
    positions = np.arange(1, p.shape[1] + 1)
    if not np.all(np.sort(p, axis=1) == positions):
        raise ValueError("Every row must be a permutation of 1..n")
    return dict(exceedances=p > positions, deficiencies=p < positions,
                fixed_points=p == positions)


def pullback_target_owned(values, permutation):
    """Gather target-owned positions j=pi(i) exactly once along the last axis.

    Use for a deficiency probe/prior indexed by positions of inverse(pi).
    DO NOT use on a transferred source probe's output: that output is already
    indexed by source i, so applying this gather again changes its semantics.
    """
    p = validate_permutation(permutation)
    values = np.asarray(values)
    if values.ndim < 1 or values.shape[-1] != len(p):
        raise ValueError("Target-owned final axis must have length n")
    return np.take(values, np.asarray(p) - 1, axis=-1)


def pullback_target_owned_batch(values, permutations):
    """Batch version for target-owned scores/labels with shape [N,n]."""
    values = np.asarray(values)
    p = np.asarray(permutations)
    batch_labels(p)  # Validate permutations without inspecting score values.
    if values.shape != p.shape:
        raise ValueError("Target-owned values and permutations must share shape [N,n]")
    return np.take_along_axis(values, p.astype(np.int64) - 1, axis=1)


def canonical_cycles(permutation):
    """Independent traversal: follow pi, minimum-first, cycles by minima."""
    p = validate_permutation(permutation)
    visited = set()
    cycles = []
    for first in range(1, len(p) + 1):
        if first in visited:
            continue
        current, cycle = first, []
        while current not in visited:
            visited.add(current)
            cycle.append(current)
            current = p[current - 1]
        cycles.append(tuple(cycle))
    return tuple(cycles)


def encoded_tokens(permutation, encoding):
    """Independent implementation of the inspected frozen n<=30 grammar."""
    p = validate_permutation(permutation)
    n = len(p)
    if n > 30:
        raise ValueError("The frozen model protocol supports n<=30")
    tokens = ['<BOS>', '<SIZE>', f'{n:02d}']
    if encoding == 'one_line':
        tokens.append('<ONE_START>')
        for index, value in enumerate(p):
            if index:
                tokens.append(',')
            tokens.append(f'{value:02d}')
        tokens.append('<ONE_END>')
    elif encoding == 'canonical_cycle':
        tokens.append('<CYCLE_START>')
        for cycle_index, cycle in enumerate(canonical_cycles(p)):
            if cycle_index:
                tokens.append(';')
            for index, value in enumerate(cycle):
                if index:
                    tokens.append(',')
                tokens.append(f'{value:02d}')
        tokens.append('<CYCLE_END>')
    else:
        raise ValueError("Encoding must be one_line or canonical_cycle")
    assert len(tokens) == 2 * n + 4
    return tuple(tokens)


def edge_token_indices(permutation, encoding):
    """Return raw edge identities and causal extraction indices in source order.

    Nonclosing cycle edges i->j become explicit at the numeric token for j;
    their preceding state is at the comma. A closing edge becomes explicit
    only at ';' or CYCLE_END, with its preceding state at the last numeric
    token. Singleton cycles use that same delimiter rule but receive their
    own kind and have negative labels. One-line source i is implicit in its
    slot, hence source_token=-1 and after is the value pi(i)'s token.
    """
    p = validate_permutation(permutation)
    n = len(p)
    tokens = encoded_tokens(p, encoding)
    source = np.arange(1, n + 1, dtype=np.int64)
    destination = np.asarray(p, dtype=np.int64)
    before = np.empty(n, dtype=np.int64)
    after = np.empty(n, dtype=np.int64)
    source_token = np.full(n, -1, dtype=np.int64)
    destination_token = np.empty(n, dtype=np.int64)
    closing_token = np.full(n, -1, dtype=np.int64)
    kinds = [''] * n
    if encoding == 'one_line':
        after[:] = 4 + 2 * np.arange(n)
        before[:] = after - 1
        destination_token[:] = after
        kinds = ['one_line_value'] * n
    else:
        cycles = canonical_cycles(p)
        numeric = {}
        cursor = 4
        for cycle in cycles:
            for value in cycle:
                numeric[value] = cursor
                cursor += 2
        for cycle in cycles:
            for k, value in enumerate(cycle):
                row = value - 1
                source_token[row] = numeric[value]
                destination_token[row] = numeric[p[row]]
                if k + 1 < len(cycle):
                    after[row] = numeric[cycle[k + 1]]
                    kinds[row] = 'cycle_internal'
                    assert tokens[after[row] - 1] == ','
                else:
                    after[row] = numeric[value] + 1
                    closing_token[row] = after[row]
                    kinds[row] = 'cycle_singleton' if len(cycle) == 1 else 'cycle_closure'
                    assert tokens[after[row]] in (';', '<CYCLE_END>')
                before[row] = after[row] - 1
    assert np.all(before >= 0) and np.all(after < len(tokens))
    return dict(source=source, destination=destination, before=before, after=after,
                source_token=source_token, destination_token=destination_token,
                closing_token=closing_token, kind=np.asarray(kinds), tokens=tokens)


def paired_edge_token_indices(permutation, encoding):
    """Pair forward i->j with inverse j->i, returning target indices in i order."""
    p = validate_permutation(permutation)
    inverse = inverse_permutation(p)
    forward = edge_token_indices(p, encoding)
    target = edge_token_indices(inverse, encoding)
    order = np.asarray(p, dtype=np.int64) - 1
    source_labels = labels(p)
    target_labels = labels(inverse)
    aligned_labels = target_labels['deficiencies'][order]
    assert np.array_equal(source_labels['exceedances'], aligned_labels)
    assert not aligned_labels[source_labels['fixed_points']].any()
    return dict(forward=forward, target=target, target_position_for_source=order + 1,
                target_before_in_source_order=target['before'][order],
                target_after_in_source_order=target['after'][order],
                target_kind_in_source_order=target['kind'][order],
                source_exceedance=source_labels['exceedances'],
                target_own_deficiency=target_labels['deficiencies'],
                target_deficiency_in_source_order=aligned_labels,
                fixed_points=source_labels['fixed_points'])
