"""Exact graph operations and a separate serializer using the existing vocabulary."""
from functools import lru_cache
import hashlib
import itertools
import random
import numpy as np
import torch
from experiments.permworld_encoding.core import TOKEN_TO_ID, ID_TO_TOKEN, loss

TASKS=('max_out','min_out','max_in','min_in')
ACTIONS=('identity','R','WR','C','WC')
TASK_TOKENS={'max_out':'<EXCEEDANCES>','min_out':'<DEFICIENCIES>',
             'max_in':'<LEFT_TO_RIGHT_MAXIMA>','min_in':'<LEFT_TO_RIGHT_MINIMA>'}
RELATIONS=(
    dict(relation='reverse_max',left='max_out',right='max_in',correct='R',wrong='WR',mapping='identity'),
    dict(relation='reverse_min',left='min_out',right='min_in',correct='R',wrong='WR',mapping='identity'),
    dict(relation='complement_out',left='max_out',right='min_out',correct='C',wrong='WC',mapping='n_minus_1_minus_y'),
    dict(relation='complement_in',left='max_in',right='min_in',correct='C',wrong='WC',mapping='n_minus_1_minus_y'))

def validate(a):
    a=np.asarray(a)
    if a.ndim!=2 or a.shape[0]!=a.shape[1] or not np.isin(a,[0,1]).all() or np.diag(a).any():
        raise ValueError('Expected square binary adjacency with zero diagonal')
    return a.astype(np.uint8,copy=False)

@lru_cache(None)
def pairs(n):return tuple(itertools.combinations(range(n),2))

@lru_cache(None)
def wrong_table(n):
    """Frozen shuffled adjacent swaps of unordered pairs; orientation preserved."""
    original=pairs(n); shuffled=list(original);random.Random(20261010+n).shuffle(shuffled)
    mapping={p:p for p in original}
    for k in range(0,len(shuffled)-1,2):
        p,q=shuffled[k:k+2];mapping[p]=q;mapping[q]=p
    return tuple(mapping[p] for p in original)

def transform(a,action):
    a=validate(a);n=len(a)
    if action=='identity':return a.copy()
    if action=='R':return a.T.copy()
    if action=='C':
        b=1-a;np.fill_diagonal(b,0);return b
    if action=='W':
        b=np.zeros_like(a)
        for (i,j),(k,l) in zip(pairs(n),wrong_table(n)):
            b[k,l]=a[i,j];b[l,k]=a[j,i]
        return b
    if action=='WR':return transform(transform(a,'R'),'W')
    if action=='WC':return transform(transform(a,'C'),'W')
    raise ValueError(action)

def labels(a):
    a=validate(a);out=a.sum(1);inn=a.sum(0)
    return np.array([out.max(),out.min(),inn.max(),inn.min()],dtype=np.int16)

def mapped_right(y,n,relation):
    return n-1-np.asarray(y) if relation['mapping']=='n_minus_1_minus_y' else np.asarray(y)

def key(a):
    a=validate(a);n=len(a)
    return bytes([n])+np.packbits(a[~np.eye(n,dtype=bool)]).tobytes()

def orbit(a):
    members={}
    for r,c,w in itertools.product((0,1),repeat=3):
        b=validate(a).copy()
        for flag,action in ((r,'R'),(c,'C'),(w,'W')):
            if flag:b=transform(b,action)
        members[key(b)]=b
    return tuple(members.values())

def orbit_key(a):return min(map(key,orbit(a)))
def orbit_id(a):return hashlib.sha256(orbit_key(a)).hexdigest()

def encode(a):
    a=validate(a);n=len(a)
    if n not in (6,7,8):raise ValueError('Only frozen n6/7/8 graph pilot')
    tokens=['<BOS>','<SIZE>',f'{n:02d}','<ARG_START>']
    for i in range(n):
        tokens.extend(f'{int(a[i,j]):02d}' for j in range(n) if i!=j)
        if i<n-1:tokens.append('|')
    tokens.append('<ARG_END>')
    ids=tuple(TOKEN_TO_ID[t] for t in tokens)
    assert len(ids)==n*n+4
    return ids

def decode(ids):
    try:tokens=tuple(ID_TO_TOKEN[int(x)] for x in ids)
    except (KeyError,ValueError,TypeError):raise ValueError('Invalid token') from None
    if len(tokens)<4 or tokens[:2]!=('<BOS>','<SIZE>') or tokens[3]!='<ARG_START>':raise ValueError('Invalid prefix')
    try:n=int(tokens[2])
    except ValueError:raise ValueError('Invalid n') from None
    if n not in (6,7,8) or len(tokens)!=n*n+4 or tokens[-1]!='<ARG_END>':raise ValueError('Invalid boundaries/length')
    groups=[];row=[]
    for token in tokens[4:-1]:
        if token=='|':groups.append(row);row=[]
        elif token in ('00','01'):row.append(int(token))
        else:raise ValueError('Expected binary entry or row boundary')
    groups.append(row)
    if len(groups)!=n or any(len(r)!=n-1 for r in groups):raise ValueError('Invalid rows')
    a=np.zeros((n,n),np.uint8)
    for i,row in enumerate(groups):a[i,np.arange(n)!=i]=row
    if encode(a)!=tuple(map(int,ids)):raise ValueError('Noncanonical graph encoding')
    return a

def batch(prefixes,ns,task,answers=None,device='cpu'):
    """Graph-specific prompt with the original two-token shifted CE convention."""
    ns=np.asarray(ns);width=int(ns.max())**2+6+(2 if answers is not None else 0)
    ids=torch.full((len(ns),width),TOKEN_TO_ID['<PAD>'],dtype=torch.long,device=device)
    target=torch.full_like(ids,-100);end=torch.as_tensor(ns*ns+3,dtype=torch.long,device=device)
    for i,n in enumerate(ns):ids[i,:int(n*n+4)]=torch.as_tensor(np.array(prefixes[i,:int(n*n+4)],copy=True),device=device)
    rows=torch.arange(len(ns),device=device)
    ids[rows,end+1]=TOKEN_TO_ID[TASK_TOKENS[task]];ids[rows,end+2]=TOKEN_TO_ID['=']
    if answers is not None:
        y=torch.as_tensor([TOKEN_TO_ID[f'{int(x):02d}'] for x in answers],device=device)
        ids[rows,end+3]=y;ids[rows,end+4]=TOKEN_TO_ID['<EOS>']
        target[rows,end+3]=y;target[rows,end+4]=TOKEN_TO_ID['<EOS>']
    assert width<=128
    return ids,ids.ne(TOKEN_TO_ID['<PAD>']),target,end,end+2
