import numpy as np
import pytest
from experiments.permworld_lemma_composition_20261110.calibration import select_threshold,micro_f1


def test_threshold_matches_bruteforce_source_only():
    rng=np.random.default_rng(20261110)
    for _ in range(100):
        scores=rng.integers(-4,5,size=(12,9)).astype(float)/3
        labels=rng.random((12,9))<.2
        threshold,info=select_threshold(scores,labels)
        candidates=sorted(np.unique(scores),reverse=True)
        scored=[(micro_f1(scores>=x,labels),x) for x in candidates]
        expected=max(scored)
        assert threshold==expected[1]
        assert info['source_micro_f1']==pytest.approx(expected[0])


def test_empty_source_positives_is_frozen_empty_mask():
    scores=np.array([[1.,2.],[3.,4.]])
    threshold,info=select_threshold(scores,np.zeros_like(scores,dtype=bool))
    assert np.isposinf(threshold) and not (scores>=threshold).any()
    assert info['source_predicted_positions']==0


def test_ties_are_not_split_and_invalid_inputs_rejected():
    t,_=select_threshold(np.ones((2,4)),np.array([[1,0,0,0],[0,1,0,0]],dtype=bool))
    assert t==1
    with pytest.raises(ValueError):select_threshold(np.ones((2,4)),np.ones((2,3),dtype=bool))
    with pytest.raises(ValueError):select_threshold(np.array([[np.nan]]),np.ones((1,1),dtype=bool))
