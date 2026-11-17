from itertools import permutations

import numpy as np
import pytest

from experiments.permworld_lemma_composition_20261110.combiners import (
    compose, compose_batch, longest_fixed_segment, longest_segment, maximal_runs_from_mask,
    turns_before_first_fixed,
)
from experiments.permworld_lemma_composition_20261110.oracles import (
    constraint_changes_answer, monotone_runs, oracle_tasks,
)


def true_masks(p):
    """Test-only independent local labels; not reused by either implementation."""
    turning = np.zeros(len(p), dtype=bool)
    for j in range(1, len(p) - 1):
        turning[j] = ((p[j] > p[j - 1] and p[j] > p[j + 1])
                      or (p[j] < p[j - 1] and p[j] < p[j + 1]))
    fixed = np.array([value == j for j, value in enumerate(p, start=1)], dtype=bool)
    return turning, fixed


@pytest.mark.parametrize('n', range(1, 9))
def test_exhaustive_true_mask_composition_agrees_with_independent_oracle(n):
    for p in permutations(range(1, n + 1)):
        turning, fixed = true_masks(p)
        assert compose(turning, fixed, n) == oracle_tasks(p), p


@pytest.mark.parametrize('p,expected', [
    ((1,), {'longest': (1, 1), 'fixed_longest': (1, 1), 'turns_before_fixed': 0}),
    ((1, 2), {'longest': (1, 2), 'fixed_longest': (1, 2), 'turns_before_fixed': 0}),
    ((2, 1), {'longest': (1, 2), 'fixed_longest': None, 'turns_before_fixed': None}),
    ((1, 2, 3, 4), {'longest': (1, 4), 'fixed_longest': (1, 4), 'turns_before_fixed': 0}),
    ((4, 3, 2, 1), {'longest': (1, 4), 'fixed_longest': None, 'turns_before_fixed': None}),
    ((5, 4, 3, 2, 1), {'longest': (1, 5), 'fixed_longest': (1, 5), 'turns_before_fixed': 0}),
    ((2, 1, 4, 3), {'longest': (1, 2), 'fixed_longest': None, 'turns_before_fixed': None}),
    ((4, 1, 3, 2, 5), {'longest': (1, 2), 'fixed_longest': (2, 3), 'turns_before_fixed': 1}),
])
def test_explicit_oracle_boundaries(p, expected):
    assert oracle_tasks(p) == expected
    assert compose(*true_masks(p), len(p)) == expected


def test_leftmost_tie_and_shared_turning_endpoint_fixed_point():
    # Both (1,3) and (3,5) contain the same fixed turning point and tie.
    turning = np.array([False, False, True, False, False])
    fixed = np.array([False, False, True, False, False])
    assert maximal_runs_from_mask(turning, 5) == [(1, 3), (3, 5)]
    assert longest_segment(turning, 5) == (1, 3)
    assert longest_fixed_segment(turning, fixed, 5) == (1, 3)
    assert turns_before_first_fixed(turning, fixed, 5) == 0


def test_first_and_last_fixed_positions_and_fake_turning_sentinels():
    turning = np.array([True, True, False, True, True])
    first = np.array([True, False, False, False, False])
    last = np.array([False, False, False, False, True])
    assert turns_before_first_fixed(turning, first, 5) == 0
    assert turns_before_first_fixed(turning, last, 5) == 2
    assert longest_fixed_segment(turning, first, 5) == (1, 2)
    assert longest_fixed_segment(turning, last, 5) == (4, 5)


def test_diagnostic_subset_excludes_no_fixed_points_and_matches_definition():
    examples = [(2, 1, 4, 3), (1, 2, 3, 4), (2, 3, 1, 4, 5)]
    for p in examples:
        answer = oracle_tasks(p)
        assert constraint_changes_answer(p) == (
            answer['fixed_longest'] is not None and answer['fixed_longest'] != answer['longest'])
    found = [p for p in permutations(range(1, 6)) if constraint_changes_answer(p)]
    assert found


@pytest.mark.parametrize('bad', [[], [1, 1], [0, 1], [True], [1.0, 2.0]])
def test_oracle_rejects_non_permutations(bad):
    with pytest.raises(ValueError):
        oracle_tasks(bad)


def test_combiners_accept_only_full_boolean_masks_and_positive_length():
    with pytest.raises(ValueError):
        compose(np.zeros(3, dtype=bool), np.zeros(3, dtype=bool), 4)
    with pytest.raises(ValueError):
        longest_segment([0, 1, 0], 3)
    with pytest.raises(ValueError):
        longest_segment(np.zeros(0, dtype=bool), 0)


def test_raw_oracle_matches_independent_enumeration_of_all_subintervals():
    # This check does not use extrema, changes of direction, or mask boundaries.
    # It also verifies that restricting to maximal runs loses no constrained optimum.
    for n in range(1, 7):
        for p in permutations(range(1, n + 1)):
            monotone = []
            fixed = [j for j, value in enumerate(p, start=1) if value == j]
            for left in range(n):
                for right in range(left, n):
                    sub = p[left:right + 1]
                    increasing = all(sub[k] < sub[k + 1] for k in range(len(sub) - 1))
                    decreasing = all(sub[k] > sub[k + 1] for k in range(len(sub) - 1))
                    if increasing or decreasing:
                        monotone.append((left + 1, right + 1))
            ordering = lambda interval: (-(interval[1] - interval[0] + 1), interval[0])
            expected = min(monotone, key=ordering)
            actual = oracle_tasks(p)
            assert actual['longest'] == expected
            if fixed:
                constrained = [segment for segment in monotone
                               if any(segment[0] <= j <= segment[1] for j in fixed)]
                assert actual['fixed_longest'] == min(constrained, key=ordering)
            else:
                assert actual['fixed_longest'] is None


def _array_answers(answers):
    return dict(longest=np.asarray([a['longest'] for a in answers], dtype=np.int64).reshape(-1, 2),
                fixed_longest=np.asarray([a['fixed_longest'] if a['fixed_longest'] is not None else (-1, -1)
                                          for a in answers], dtype=np.int64).reshape(-1, 2),
                turns_before_fixed=np.asarray([a['turns_before_fixed'] if a['turns_before_fixed'] is not None else -1
                                               for a in answers], dtype=np.int64))


@pytest.mark.parametrize('n', range(1, 8))
def test_batch_matches_raw_oracle_for_every_permutation(n):
    perms = list(permutations(range(1, n + 1)))
    labels = [true_masks(p) for p in perms]
    turning = np.stack([a[0] for a in labels])
    fixed = np.stack([a[1] for a in labels])
    actual = compose_batch(turning, fixed, n)
    expected = _array_answers([oracle_tasks(p) for p in perms])
    for key in expected:
        np.testing.assert_array_equal(actual[key], expected[key])
        assert actual[key].dtype == np.int64


def test_batch_matches_scalar_for_arbitrary_predicted_masks_without_mutation():
    rng = np.random.default_rng(20261110)
    for n in range(1, 41):
        turning = rng.random((100, n)) < .35
        fixed = rng.random((100, n)) < .1
        fixed[0] = False
        fixed[1] = False; fixed[1, 0] = True
        fixed[2] = False; fixed[2, -1] = True
        fixed[3] = True
        turning[4] = False
        turning[5] = True
        turning_before, fixed_before = turning.copy(), fixed.copy()
        expected = _array_answers([compose(a, b, n) for a, b in zip(turning, fixed)])
        actual = compose_batch(turning, fixed, n)
        for key in expected:
            np.testing.assert_array_equal(actual[key], expected[key])
        np.testing.assert_array_equal(turning, turning_before)
        np.testing.assert_array_equal(fixed, fixed_before)


@pytest.mark.parametrize('n', [1, 2, 30])
def test_empty_batch(n):
    actual = compose_batch(np.zeros((0, n), dtype=bool), np.zeros((0, n), dtype=bool), n)
    assert actual['longest'].shape == actual['fixed_longest'].shape == (0, 2)
    assert actual['turns_before_fixed'].shape == (0,)


def test_batch_rejects_invalid_masks():
    with pytest.raises(ValueError):
        compose_batch(np.zeros((2, 4), dtype=bool), np.zeros((3, 4), dtype=bool), 4)
    with pytest.raises(ValueError):
        compose_batch(np.zeros((2, 4)), np.zeros((2, 4), dtype=bool), 4)
    with pytest.raises(ValueError):
        compose_batch(np.zeros((2, 4), dtype=bool), np.zeros((2, 4), dtype=bool), 3)
