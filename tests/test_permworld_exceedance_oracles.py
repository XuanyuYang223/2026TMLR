from itertools import permutations
from pathlib import Path
import sys

import numpy as np
import pytest

from experiments.permworld_exceedance_alignment_20261110.oracles import (
    batch_labels, canonical_cycles, edge_token_indices, encoded_tokens,
    inverse_permutation, labels, paired_edge_token_indices,
    pullback_target_owned, pullback_target_owned_batch,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'external/neurips/src'))
from neurips_permutations.passage import cycle_tokens, one_line_tokens
from neurips_permutations.math_ops import PROPERTY_FUNCTIONS


def explicitly_revealed_edges(tokens):
    """Independent streaming parser; never infer a forced missing value."""
    edges = {}
    if len(tokens) <= 3:
        return edges
    if tokens[3] == '<ONE_START>':
        slot = 1
        for token in tokens[4:]:
            if token.isdigit():
                edges[slot] = int(token)
                slot += 1
    else:
        current_cycle = []
        for token in tokens[4:]:
            if token.isdigit():
                value = int(token)
                if current_cycle:
                    edges[current_cycle[-1]] = value
                current_cycle.append(value)
            elif token in (';', '<CYCLE_END>'):
                assert current_cycle
                edges[current_cycle[-1]] = current_cycle[0]
                current_cycle = []
    return edges


@pytest.mark.parametrize('n', range(1, 9))
def test_exhaustive_inverse_relation_and_fixed_negatives(n):
    for p in permutations(range(1, n + 1)):
        inv = inverse_permutation(p)
        forward, target = labels(p), labels(inv)
        assert inverse_permutation(inv) == p
        pulled = pullback_target_owned(target['deficiencies'], p)
        np.testing.assert_array_equal(forward['exceedances'], pulled)
        assert not pulled[forward['fixed_points']].any()
        assert not forward['deficiencies'][forward['fixed_points']].any()
        assert int(forward['exceedances'].sum()) == PROPERTY_FUNCTIONS['exceedances'](p)
        assert int(forward['deficiencies'].sum()) == PROPERTY_FUNCTIONS['deficiencies'](p)


@pytest.mark.parametrize('n', range(1, 8))
def test_exhaustive_token_indices_against_real_serializer_and_stream_parser(n):
    for p in permutations(range(1, n + 1)):
        for encoding, renderer in [('one_line', one_line_tokens), ('canonical_cycle', cycle_tokens)]:
            actual_tokens = ('<BOS>', '<SIZE>', f'{n:02d}', *renderer(p))
            assert encoded_tokens(p, encoding) == actual_tokens
            layout = edge_token_indices(p, encoding)
            assert len(actual_tokens) == 2*n + 4
            for i, destination in enumerate(p, start=1):
                before, after = int(layout['before'][i-1]), int(layout['after'][i-1])
                assert before + 1 == after
                assert i not in explicitly_revealed_edges(actual_tokens[:before+1])
                assert explicitly_revealed_edges(actual_tokens[:after+1])[i] == destination


def test_cycle_closures_singletons_and_inverse_pairing():
    p = (2, 3, 1, 5, 4, 6)
    assert canonical_cycles(p) == ((1, 2, 3), (4, 5), (6,))
    layout = edge_token_indices(p, 'canonical_cycle')
    assert layout['tokens'] == ('<BOS>', '<SIZE>', '06', '<CYCLE_START>',
                                '01', ',', '02', ',', '03', ';', '04', ',', '05', ';', '06', '<CYCLE_END>')
    np.testing.assert_array_equal(layout['after'], [6, 8, 9, 12, 13, 15])
    np.testing.assert_array_equal(layout['before'], [5, 7, 8, 11, 12, 14])
    assert list(layout['kind']) == ['cycle_internal', 'cycle_internal', 'cycle_closure',
                                   'cycle_internal', 'cycle_closure', 'cycle_singleton']
    pair = paired_edge_token_indices(p, 'canonical_cycle')
    inv = inverse_permutation(p)
    for i, j in enumerate(p, start=1):
        index = pair['target_after_in_source_order'][i-1]
        revealed = explicitly_revealed_edges(pair['target']['tokens'][:int(index)+1])
        assert inv[j-1] == i == revealed[j]
    assert pair['target_kind_in_source_order'][0] == 'cycle_closure'
    assert pair['target_kind_in_source_order'][2] == 'cycle_internal'
    assert pair['target_kind_in_source_order'][5] == 'cycle_singleton'


def test_one_line_values_and_before_indices():
    layout = edge_token_indices((3, 1, 2), 'one_line')
    np.testing.assert_array_equal(layout['before'], [3, 5, 7])
    np.testing.assert_array_equal(layout['after'], [4, 6, 8])
    assert layout['tokens'][-1] == '<ONE_END>'
    assert all(t == -1 for t in layout['source_token'])


def test_target_owned_scores_reorder_once_source_index_scores_do_not():
    p = (2, 3, 1)
    own = labels(inverse_permutation(p))['deficiencies']
    transferred_source_index = labels(p)['exceedances']
    np.testing.assert_array_equal(pullback_target_owned(own, p), transferred_source_index)
    assert not np.array_equal(pullback_target_owned(pullback_target_owned(own, p), p), transferred_source_index)
    assert not np.array_equal(pullback_target_owned(transferred_source_index, p), transferred_source_index)
    scores = np.array([.1, .9, .8])
    np.testing.assert_array_equal(pullback_target_owned(scores, p), [.9, .8, .1])


def test_batch_labels_and_target_owned_gather():
    p = np.array([(2, 3, 1), (1, 3, 2)], dtype=np.int16)
    inv = np.argsort(p, axis=1) + 1
    own = batch_labels(inv)['deficiencies']
    np.testing.assert_array_equal(pullback_target_owned_batch(own, p), batch_labels(p)['exceedances'])
    np.testing.assert_array_equal(batch_labels(p)['fixed_points'], p == np.arange(1, 4))


def test_matches_real_full_encoder_without_model_construction():
    from experiments.permworld_encoding.core import encode, ID_TO_TOKEN
    for p in [(1,), (2, 1), (2, 3, 1, 5, 4, 6), tuple(range(30, 0, -1))]:
        for encoding in ['one_line', 'canonical_cycle']:
            real = tuple(ID_TO_TOKEN[i] for i in encode(p, encoding))
            assert encoded_tokens(p, encoding) == real


@pytest.mark.parametrize('p', [(), (1, 1), (0, 1), (True,), (1., 2.)])
def test_invalid_permutation_rejected(p):
    with pytest.raises(ValueError):
        labels(p)


def test_shape_and_protocol_guards():
    with pytest.raises(ValueError):
        encoded_tokens(tuple(range(1, 32)), 'one_line')
    with pytest.raises(ValueError):
        encoded_tokens((1,), 'other')
    with pytest.raises(ValueError):
        pullback_target_owned([.1], (2, 1))
    with pytest.raises(ValueError):
        pullback_target_owned_batch(np.zeros((2, 3)), np.array([[1, 2, 3]]))
