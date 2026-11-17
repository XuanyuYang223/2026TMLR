"""Independent raw-permutation oracles for the three downstream questions.

This module deliberately does not import the prediction-mask combiners. Runs
are found by directly scanning the signs of adjacent differences. Endpoints
are inclusive, use one-based positions, and equal-length answers choose the
leftmost start. A shared turning-point endpoint belongs to both adjacent runs.
"""
from __future__ import annotations

from numbers import Integral


def _permutation(values):
    values = list(values)
    if not values or any(isinstance(x, bool) or not isinstance(x, Integral) for x in values):
        raise ValueError("A nonempty integer permutation is required")
    values = [int(x) for x in values]
    if sorted(values) != list(range(1, len(values) + 1)):
        raise ValueError("Values must be a permutation of 1 through n")
    return values


def monotone_runs(permutation):
    """Return maximal strict monotone intervals by scanning adjacent slopes."""
    values = _permutation(permutation)
    n = len(values)
    if n == 1:
        return [(1, 1)]
    runs = []
    start = 1
    previous_increasing = values[1] > values[0]
    for right_index in range(2, n):
        increasing = values[right_index] > values[right_index - 1]
        if increasing != previous_increasing:
            # right_index is the one-based position of the shared extremum.
            runs.append((start, right_index))
            start = right_index
        previous_increasing = increasing
    runs.append((start, n))
    return runs


def oracle_tasks(permutation):
    """Solve all tasks directly from a permutation, without predicted masks."""
    values = _permutation(permutation)
    runs = monotone_runs(values)
    longest = min(runs, key=lambda interval: (-(interval[1] - interval[0] + 1), interval[0]))
    fixed = [position for position, value in enumerate(values, start=1) if value == position]
    if not fixed:
        return dict(longest=longest, fixed_longest=None, turns_before_fixed=None)

    eligible = [interval for interval in runs if any(interval[0] <= point <= interval[1] for point in fixed)]
    fixed_longest = min(eligible, key=lambda interval: (-(interval[1] - interval[0] + 1), interval[0]))
    first_fixed = fixed[0]
    # Count directly from local comparisons, never from run boundaries or masks.
    before = 0
    for position in range(2, len(values)):
        if position >= first_fixed:
            break
        a, b, c = values[position - 2:position + 1]
        if (a < b > c) or (a > b < c):
            before += 1
    return dict(longest=longest, fixed_longest=fixed_longest, turns_before_fixed=before)


def constraint_changes_answer(permutation):
    """Scoring-only subset: a real fixed point changes the longest answer."""
    answer = oracle_tasks(permutation)
    return answer['fixed_longest'] is not None and answer['fixed_longest'] != answer['longest']
