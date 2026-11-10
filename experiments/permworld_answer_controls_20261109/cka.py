"""FP64 CKA and shared dual-answer OLS. No optimizer/backward operations."""
from __future__ import annotations
import numpy as np
import torch


def dual_answer_basis(a,b):
    """SVD basis for intercept + both one-hot answers on the same anchor rows."""
    a=np.asarray(a);b=np.asarray(b);n=len(a)
    aa=np.eye(len(np.unique(a)),dtype=np.float64)[np.unique(a,return_inverse=True)[1]]
    blocks=[np.ones((n,1)),aa]
    identical=bool(np.array_equal(a,b))
    if not identical:blocks.append(np.eye(len(np.unique(b)),dtype=np.float64)[np.unique(b,return_inverse=True)[1]])
    design=np.concatenate(blocks,axis=1)
    u,s,_=np.linalg.svd(design,full_matrices=False)
    tolerance=np.finfo(np.float64).eps*max(design.shape)*s[0]
    rank=int((s>tolerance).sum());q=u[:,:rank]
    return q,dict(design_columns=design.shape[1],design_rank=rank,residual_df=n-rank,answers_identical=identical,
                  rank_tolerance=tolerance,rank_rule='eps(float64)*max(N,p)*largest_singular_value')


def center(x):return x-x.mean(dim=-2,keepdim=True)


def u_center(k):
    n=k.shape[-1]
    if n<4:raise ValueError('U-centering needs at least four rows')
    a=k.clone();a.diagonal(dim1=-2,dim2=-1).zero_();s=a.sum(-1)
    a-=s[..., :,None]/(n-2)+s[...,None,:]/(n-2)
    a+=s.sum(-1)[...,None,None]/((n-1)*(n-2))
    a.diagonal(dim1=-2,dim2=-1).zero_();return a


def feature_cka(x,y):
    x=center(x);y=center(y)
    xy=x.transpose(-1,-2)@y;xx=x.transpose(-1,-2)@x;yy=y.transpose(-1,-2)@y
    den=(xx.square().sum((-2,-1))*yy.square().sum((-2,-1))).sqrt()
    value=xy.square().sum((-2,-1))/den
    return torch.where(den>1e-20,value,torch.full_like(value,float('nan')))


def all_scores(h,debiased=True):
    """h[TASK,ACTION,N,D]; returns[ACTION,leftTASK,rightTASK]."""
    tasks,actions,n,d=h.shape
    hh=center(h)
    if debiased:
        k=u_center(hh@hh.transpose(-1,-2));flat=k.reshape(tasks,actions,-1)
        norms=flat.square().sum(-1).sqrt();out=[]
        for action in range(actions):
            numerator=flat[:,0]@flat[:,action].T;den=norms[:,0,None]*norms[None,:,action]
            out.append(torch.where(den>1e-20,numerator/den,torch.full_like(numerator,float('nan'))))
        return torch.stack(out)
    x=hh[:,0].permute(1,0,2).reshape(n,tasks*d)
    xx=x.T@x;norms=[]
    for i in range(tasks):norms.append(xx[i*d:(i+1)*d,i*d:(i+1)*d].square().sum())
    normx=torch.stack(norms).sqrt();out=[]
    for action in range(actions):
        y=hh[:,action].permute(1,0,2).reshape(n,tasks*d);xy=x.T@y;yy=y.T@y
        normy=torch.stack([yy[i*d:(i+1)*d,i*d:(i+1)*d].square().sum() for i in range(tasks)]).sqrt()
        numerator=xy.reshape(tasks,d,tasks,d).square().sum((1,3));den=normx[:,None]*normy[None,:]
        out.append(torch.where(den>1e-20,numerator/den,torch.full_like(numerator,float('nan'))))
    return torch.stack(out)


def residual_scores(h,pairs,bases,batch_size=16):
    """Direct FP64 residual features; padded orthonormal bases have zero columns."""
    scores=[];diagnostics=[];n=h.shape[-2]
    for start in range(0,len(pairs),batch_size):
        group=pairs[start:start+batch_size];qs=bases[start:start+batch_size]
        rank=max(q.shape[1] for q in qs);q=np.zeros((len(qs),n,rank),np.float64)
        for i,a in enumerate(qs):q[i,:,:a.shape[1]]=a
        q=torch.as_tensor(q,device=h.device)
        x=torch.stack([h[a,0] for a,b,t in group]);y=torch.stack([h[b,t] for a,b,t in group])
        xx=x-q@(q.transpose(-1,-2)@x);yy=y-q@(q.transpose(-1,-2)@y)
        k=u_center(center(xx)@center(xx).transpose(-1,-2));l=u_center(center(yy)@center(yy).transpose(-1,-2))
        den=(k.square().sum((-2,-1))*l.square().sum((-2,-1))).sqrt();value=(k*l).sum((-2,-1))/den
        value=torch.where(den>1e-20,value,torch.full_like(value,float('nan')))
        # Numerical annihilation is undefined, not evidence of zero alignment.
        ex=xx.square().sum((-2,-1));ey=yy.square().sum((-2,-1));tx=x.square().sum((-2,-1));ty=y.square().sum((-2,-1))
        vanished=(ex<=torch.maximum(tx*1e-24,torch.full_like(tx,1e-20)))|(ey<=torch.maximum(ty*1e-24,torch.full_like(ty,1e-20)))
        value=torch.where(vanished,torch.full_like(value,float('nan')),value)
        scores.extend(value.cpu().numpy().tolist())
        diagnostics.extend(zip((ex/tx).cpu().numpy().tolist(),(ey/ty).cpu().numpy().tolist()))
    return np.asarray(scores),list(diagnostics)


def grouped_original_scores(h,pairs,groups,batch_size=16):
    """Reproduce joint-strata means and >=3 retained rows without changing CKA."""
    result=[];n=h.shape[-2];d=h.shape[-1]
    for start in range(0,len(pairs),batch_size):
        items=pairs[start:start+batch_size];gg=groups[start:start+batch_size]
        inv=np.stack([a[0] for a in gg]);count=np.stack([a[1][a[0]] for a in gg]);valid=count>=3
        x=torch.stack([h[a,0] for a,b,t in items]);y=torch.stack([h[b,t] for a,b,t in items])
        ids=torch.as_tensor(inv,device=h.device,dtype=torch.long)
        mask=torch.as_tensor(valid,device=h.device)[...,None];cnt=torch.as_tensor(count,device=h.device)[...,None]
        def residual(z):
            sums=torch.zeros_like(z);sums.scatter_add_(1,ids[...,None].expand(-1,-1,d),z)
            return (z-sums.gather(1,ids[...,None].expand(-1,-1,d))/cnt)*mask
        v=feature_cka(residual(x),residual(y));selected=torch.as_tensor(valid.sum(1),device=h.device)
        v=torch.where(selected>=2,v,torch.full_like(v,float('nan')))
        result.extend(v.cpu().numpy().tolist())
    return np.asarray(result)
