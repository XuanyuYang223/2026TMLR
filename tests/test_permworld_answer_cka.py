import importlib.util
from pathlib import Path
import numpy as np
import torch
from experiments.permworld_answer_controls_20261109.cka import (
    all_scores,dual_answer_basis,residual_scores,grouped_original_scores,
)
from experiments.permworld_answer_controls_20261109.common import Sources


def test_saved_activation_adapter_keeps_anchor_and_action_axes_distinct():
    source=Sources.__new__(Sources);source.c={'tasks':['left','right']}
    arrays={t:np.arange(7*3*5).reshape(7,3,5)+(1000 if t=='right' else 0) for t in source.c['tasks']}
    source.get=lambda unit,arch,enc,task,phase,split,field:arrays[task]
    h=source.feature_stack('u','a','e','trained')
    assert h.shape==(2,3,7,5)
    assert np.array_equal(h[1,2],arrays['right'][:,2])


def original():
    path=Path(__file__).resolve().parents[1]/'diagnostics/permworld_scope_inventory_20261009/metrics.py'
    spec=importlib.util.spec_from_file_location('answer_test_original_metrics',path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def test_all_pairs_matches_original_gram_cka():
    rng=np.random.default_rng(119);h=rng.normal(size=(4,3,35,9));metric=original()
    for debiased in (False,True):
        got=all_scores(torch.tensor(h),debiased).numpy()
        for a,b,t in [(0,1,0),(3,0,2),(2,1,1),(1,1,0)]:
            assert abs(got[t,a,b]-metric.cka(h[a,0],h[b,t],debiased))<1e-12


def test_shared_answer_basis_equals_additive_least_squares():
    rng=np.random.default_rng(121);a=rng.integers(0,4,80);b=rng.integers(0,3,80);x=rng.normal(size=(80,7))
    q,meta=dual_answer_basis(a,b)
    design=np.column_stack((np.ones(80),np.eye(4)[a],np.eye(3)[b]))
    residual=x-design@np.linalg.lstsq(design,x,rcond=None)[0]
    assert np.max(np.abs(residual-(x-q@(q.T@x))))<1e-12
    q2,m2=dual_answer_basis(a,a+1);q1,m1=dual_answer_basis(a,a)
    assert m2['design_rank']==m1['design_rank']==4 and m1['answers_identical']
    assert np.max(np.abs(q1@q1.T-q2@q2.T))<1e-12
    assert meta['residual_df']==80-meta['design_rank']


def test_residual_cka_matches_independent_lstsq_and_null_is_undefined():
    rng=np.random.default_rng(122);h=rng.normal(size=(2,3,60,8));a=rng.integers(0,4,60);b=rng.integers(0,3,60)
    q,_=dual_answer_basis(a,b);got,diag=residual_scores(torch.tensor(h),[(0,1,2)],[q]);metric=original()
    design=np.column_stack((np.ones(60),np.eye(4)[a],np.eye(3)[b]))
    x=h[0,0]-design@np.linalg.lstsq(design,h[0,0],rcond=None)[0]
    y=h[1,2]-design@np.linalg.lstsq(design,h[1,2],rcond=None)[0]
    assert abs(got[0]-metric.cka(x,y,True))<1e-12
    h[0,0]=np.eye(4)[a]@rng.normal(size=(4,8))
    got,_=residual_scores(torch.tensor(h),[(0,1,2)],[q]);assert np.isnan(got[0])


def test_original_joint_answer_control_reproduces_actual_sample_dropping():
    rng=np.random.default_rng(123);h=rng.normal(size=(3,3,35,6));inv=np.asarray([0]*10+[1]*12+[2]*11+[3,4]);count=np.bincount(inv)
    got=grouped_original_scores(torch.tensor(h),[(0,2,1)],[(inv,count)])[0]
    use=count[inv]>=3;keys=inv[use,None];metric=original()
    x=metric.center(h[0,0,use],keys);y=metric.center(h[2,1,use],keys)
    assert abs(got-metric.cka(x,y))<1e-12
def test_numeric_mass_stays_probability_and_matches_independent_softmax():
    from experiments.permworld_answer_controls_20261109.run import numeric_support_mass
    generator=np.random.default_rng(20261109)
    logits=generator.normal(size=(128,188)).astype(np.float32)+32
    logits[:,31:]-=45
    logits[0]=generator.normal(size=188).astype(np.float32)
    stable=logits.astype(np.float64)
    stable-=stable.max(1,keepdims=True)
    probability=np.exp(stable);probability/=probability.sum(1,keepdims=True)
    got=numeric_support_mass(torch.as_tensor(logits)).numpy()
    assert got.dtype==np.float64
    assert np.all((got>=0)&(got<=1))
    np.testing.assert_allclose(got,probability[:,:31].sum(1),rtol=0,atol=2e-15)

