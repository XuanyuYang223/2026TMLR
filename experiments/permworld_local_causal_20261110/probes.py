"""Independent fit halves for editing, manipulation checks and count diagnosis."""
import hashlib
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

from .shared import OUT,DIAG,write,sha


def fit_linear(x,y):
    x=np.asarray(x,dtype=np.float64)
    scaler=StandardScaler().fit(x)
    ridge=Ridge(alpha=10.,solver='cholesky').fit(scaler.transform(x),np.asarray(y,dtype=np.float64))
    weight=ridge.coef_/scaler.scale_
    bias=float(ridge.intercept_-scaler.mean_@weight)
    np.testing.assert_allclose(x[:8]@weight+bias,ridge.predict(scaler.transform(x[:8])),atol=1e-10,rtol=0)
    return dict(weight=weight,bias=np.asarray(bias),scaler_mean=scaler.mean_,scaler_scale=scaler.scale_,
                ridge_coef=ridge.coef_,ridge_intercept=np.asarray(ridge.intercept_))


def predict(parameters,x):
    return np.asarray(x,dtype=np.float64)@parameters['weight']+parameters['bias']


def fit_bundle(states,nodes,layer,point,unit,task,save=True):
    fit_a=np.array([r['pool']=='fit' and r['fit_partition']=='A' for r in nodes])
    fit_b=np.array([r['pool']=='fit' and r['fit_partition']=='B' for r in nodes])
    assert min(fit_a.sum(),fit_b.sum())>=256
    labels=np.array([r['label'] for r in nodes])
    counter=np.array([r['resolved_count']-(r['label'] if point==0 else 0) for r in nodes])
    x=states[:,point,layer]
    edit=fit_linear(x[fit_a],labels[fit_a])
    checker=fit_linear(x[fit_b],labels[fit_b])
    count=fit_linear(x[fit_b],counter[fit_b])
    v=edit['weight']/np.linalg.norm(edit['weight'])
    bundle=dict(edit=edit,checker=checker,count=count,direction=v,layer=layer,point=point)
    if save:
        path=OUT/unit/'probes'/task/f'layer{layer+1}_point{point}.npz'
        path.parent.mkdir(parents=True,exist_ok=True)
        arrays={f'{role}_{key}':value for role,model in [('edit',edit),('checker',checker),('count',count)] for key,value in model.items()}
        arrays.update(direction=v,fit_a_node_ids=np.array([r['node_id'] for r,k in zip(nodes,fit_a) if k]),
                      fit_b_node_ids=np.array([r['node_id'] for r,k in zip(nodes,fit_b) if k]))
        np.savez_compressed(path,**arrays)
        bundle['path']=str(path);bundle['sha256']=sha(path)
    return bundle


def edit_state(recipient,donor,direction,dose=1.):
    r=np.asarray(recipient,dtype=np.float64)
    d=np.asarray(donor,dtype=np.float64)
    coordinate=(d-r)@direction
    changed=(r+dose*coordinate[...,None]*direction).astype(np.float32)
    return changed,coordinate
