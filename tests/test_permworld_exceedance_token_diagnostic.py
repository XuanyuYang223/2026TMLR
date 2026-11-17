"""Synthetic checks only: no real checkpoint or experiment outcome access."""
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

from experiments.permworld_exceedance_alignment_20261110.token_diagnostic import (
    masked_auc, fit_scalar, predict, rotation, pull_states,
)


def test_masked_auroc_matches_independent_sklearn_with_undefined():
    rng=np.random.default_rng(17)
    y=rng.integers(0,2,size=(40,19)).astype(bool)
    score=rng.normal(size=y.shape)
    mask=rng.random(y.shape)>.3
    mask[0]=False
    y[1]=False
    result=masked_auc(score,y,mask)
    for i in range(len(y)):
        own=y[i,mask[i]]
        if len(own) and own.any() and (~own).any():
            np.testing.assert_allclose(result[i],roc_auc_score(own,score[i,mask[i]]),atol=1e-14,rtol=0)
        else:
            assert np.isnan(result[i])


def test_serialized_scalar_preserves_actual_float32_scaler_path():
    rng=np.random.default_rng(333)
    x=rng.normal(size=(31,12,7)).astype(np.float32)
    labels=rng.integers(0,2,size=(31,12)).astype(bool)
    fit=np.arange(31)<23
    dev=~fit
    saved,grid=fit_scalar(x,labels,fit,dev)
    scaler=StandardScaler().fit(x[fit].reshape(-1,7))
    ridge=Ridge(alpha=saved['alpha'],solver='cholesky').fit(
        scaler.transform(x[fit].reshape(-1,7)).astype(np.float64),labels[fit].reshape(-1).astype(float))
    independent=ridge.predict(scaler.transform(x.reshape(-1,7)).astype(np.float64)).reshape(labels.shape)
    np.testing.assert_array_equal(predict(saved,x),independent)
    assert [row['alpha'] for row in grid]==[1.,10.,100.,1000.]


def test_effective_transformed_scalar_equals_explicit_map_then_probe():
    rng=np.random.default_rng(44)
    x=rng.normal(size=(30,11,9)).astype(np.float32)
    y=rng.integers(0,2,size=(30,11)).astype(bool)
    probe,_=fit_scalar(x,y,np.arange(30)<22,np.arange(30)>=22)
    target=rng.normal(size=(20,11,2,9)).astype(np.float32)
    q,_=np.linalg.qr(rng.normal(size=(9,9)))
    mx=rng.normal(size=9);my=rng.normal(size=9)
    explicit=predict(probe,(target-my)@q+mx)
    w=q@probe['weight']
    efficient=target.astype(np.float64)@w+mx@probe['weight']-my@w+probe['bias']
    np.testing.assert_allclose(explicit,efficient,atol=2e-12,rtol=0)


def test_inverse_native_features_gather_once_preserves_point_and_hidden_axes():
    p=np.array([[2,3,1],[3,1,2]])
    value=np.arange(2*3*2*4).reshape(2,3,2,4)
    actual=pull_states(value,p)
    expected=np.stack([row[order-1] for row,order in zip(value,p)])
    np.testing.assert_array_equal(actual,expected)
    assert not np.array_equal(pull_states(actual,p),expected)


def test_rotation_orientation_is_target_to_source():
    rng=np.random.default_rng(81)
    source=rng.normal(size=(200,12))
    q,_=np.linalg.qr(rng.normal(size=(12,12)))
    target=source@q
    fitted=rotation(target.T@source)
    np.testing.assert_allclose(target@fitted,source,atol=2e-12,rtol=0)
