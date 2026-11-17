"""Fixed downstream rules whose only inputs are predicted masks and length.

Masks are one-dimensional full-length Boolean arrays. Their array indices are
zero-based; returned segment endpoints are inclusive and one-based. Endpoint
bits in a turning mask are ignored, because endpoints are boundary sentinels
rather than genuine peaks or valleys. No permutation or answer is accepted.
"""
from __future__ import annotations

from numbers import Integral

import numpy as np


def _length(n):
    if isinstance(n, bool) or not isinstance(n, Integral) or n < 1:
        raise ValueError("Length must be a positive integer")
    return int(n)


def _mask(mask, n, name):
    n = _length(n)
    array = np.asarray(mask)
    if array.dtype != np.bool_ or array.shape != (n,):
        raise ValueError(f"{name} must be a Boolean array of shape ({n},)")
    return array


def maximal_runs_from_mask(turning_mask, n):
    """Construct consecutive-boundary candidate intervals, including ends."""
    n = _length(n)
    turning = _mask(turning_mask, n, 'turning_mask')
    if n == 1:
        return [(1, 1)]
    boundaries = [1, *(int(index) + 2 for index in np.flatnonzero(turning[1:-1])), n]
    return list(zip(boundaries[:-1], boundaries[1:]))


def _leftmost_longest(intervals):
    return min(intervals, key=lambda interval: (-(interval[1] - interval[0] + 1), interval[0]))


def longest_segment(turning_mask, n):
    return _leftmost_longest(maximal_runs_from_mask(turning_mask, n))


def longest_fixed_segment(turning_mask, fixed_mask, n):
    _mask(turning_mask, n, 'turning_mask')
    fixed = _mask(fixed_mask, n, 'fixed_mask')
    if not fixed.any():
        return None
    eligible = [interval for interval in maximal_runs_from_mask(turning_mask, n)
                if fixed[interval[0] - 1:interval[1]].any()]
    return _leftmost_longest(eligible)


def turns_before_first_fixed(turning_mask, fixed_mask, n):
    turning = _mask(turning_mask, n, 'turning_mask')
    fixed = _mask(fixed_mask, n, 'fixed_mask')
    points = np.flatnonzero(fixed)
    if not len(points):
        return None
    first = int(points[0])
    # Strictly before the first fixed position, excluding both sentinels.
    return int(turning[1:min(first, n - 1)].sum())


def compose(turning_mask, fixed_mask, n):
    """Use the identical masks for the two segment tasks and the count task."""
    _mask(turning_mask, n, 'turning_mask')
    _mask(fixed_mask, n, 'fixed_mask')
    return dict(longest=longest_segment(turning_mask, n),
                fixed_longest=longest_fixed_segment(turning_mask, fixed_mask, n),
                turns_before_fixed=turns_before_first_fixed(turning_mask, fixed_mask, n))


def compose_batch(turning_masks, fixed_masks, n):
    """Vectorized mask-only rules, with ``-1`` sentinels for NONE answers.

    Args:
        turning_masks: Boolean matrix ``[examples, n]``. Endpoint bits are ignored.
        fixed_masks: Boolean matrix with the identical shape.
        n: Positive common length of the batch.

    Returns:
        ``longest`` and ``fixed_longest`` are int64 ``[examples, 2]`` arrays
        of inclusive one-based endpoints. Missing constrained segments use
        ``(-1, -1)``. ``turns_before_fixed`` is int64 ``[examples]`` and uses
        ``-1`` when no fixed point was predicted. Inputs are not modified.

    Candidate intervals are all consecutive predicted boundaries. Prefix
    maxima locate their starts, and prefix sums test fixed-point inclusion.
    Argmax chooses the earliest candidate on ties, hence the leftmost start.
    No loop over examples, permutation, or truth information is used.
    """
    n = _length(n)
    turning = np.asarray(turning_masks)
    fixed = np.asarray(fixed_masks)
    if (turning.dtype != np.bool_ or turning.ndim != 2 or turning.shape[1] != n
            or fixed.dtype != np.bool_ or fixed.shape != turning.shape):
        raise ValueError("Both masks must be Boolean matrices with identical shape (N, n)")
    count = turning.shape[0]
    indices = np.arange(count)
    has_fixed = fixed.any(axis=1)
    if n == 1:
        longest = np.ones((count, 2), dtype=np.int64)
        return dict(longest=longest,
                    fixed_longest=np.where(has_fixed[:, None], longest, -1),
                    turns_before_fixed=np.where(has_fixed, 0, -1).astype(np.int64))

    boundaries = turning.copy()
    boundaries[:, 0] = True
    boundaries[:, -1] = True
    positions = np.arange(n, dtype=np.int64)
    preceding = np.maximum.accumulate(np.where(boundaries, positions, 0), axis=1)
    starts = preceding[:, :-1]
    right_positions = positions[None, 1:]
    lengths = right_positions - starts + 1
    candidates = boundaries[:, 1:]
    candidate_lengths = np.where(candidates, lengths, 0)
    best = candidate_lengths.argmax(axis=1)
    longest = np.column_stack((starts[indices, best] + 1, best + 2))

    fixed_prefix = np.concatenate((np.zeros((count, 1), dtype=np.int64),
                                   fixed.cumsum(axis=1, dtype=np.int64)), axis=1)
    fixed_counts = fixed_prefix[:, 2:] - np.take_along_axis(fixed_prefix, starts, axis=1)
    constrained_lengths = np.where(candidates & (fixed_counts > 0), lengths, 0)
    constrained_best = constrained_lengths.argmax(axis=1)
    constrained = np.column_stack((starts[indices, constrained_best] + 1, constrained_best + 2))
    constrained = np.where(has_fixed[:, None], constrained, -1)

    first_fixed = fixed.argmax(axis=1)
    before = (turning[:, 1:-1]
              & (positions[None, 1:-1] < first_fixed[:, None])).sum(axis=1, dtype=np.int64)
    before = np.where(has_fixed, before, -1)
    return dict(longest=longest, fixed_longest=constrained, turns_before_fixed=before)
