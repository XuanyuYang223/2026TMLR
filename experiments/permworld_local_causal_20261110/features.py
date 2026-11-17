"""Read actual recurrent states at the edge update, leaving all weights fixed."""
from pathlib import Path
from collections import defaultdict
import time

import numpy as np
import torch

from .shared import DIAG,OUT,sha,read,write,ids_for,catalog
from .recurrent import run_prefix,resume,parameter_hash


def collect(unit,task,model,pools,batch_size=128):
    from .shared import families
    selected=families(unit)
    nodes,lookup=catalog(selected,task,set(pools))
    tag='fit_dev' if set(pools)=={'fit','dev'} else 'test'
    folder=OUT/unit/'states'/task/tag
    folder.mkdir(parents=True,exist_ok=True)
    receipt=folder/'completion.json'
    if receipt.exists():
        old=read(receipt)
        assert old['model_parameter_sha256']==parameter_hash(model)
        assert old['node_ids']==[r['node_id'] for r in nodes]
        for path,digest in old['files'].items():assert sha(path)==digest
        return np.load(folder/'states.npy',mmap_mode='r'),nodes,lookup,old
    started=time.perf_counter()
    before_hash=parameter_hash(model)
    temporary=folder/'states.partial.npy'
    states=np.lib.format.open_memmap(temporary,mode='w+',dtype=np.float32,shape=(len(nodes),2,4,256))
    native=np.full((len(nodes),188),np.nan,np.float32)
    noop=np.full_like(native,np.nan)
    grouping=defaultdict(list)
    for index,row in enumerate(nodes):grouping[row['n'],row['position']].append(index)
    max_logit_error=0.;argmax_mismatches=0
    device=next(model.parameters()).device
    with torch.inference_mode():
        for (n,i),indices in sorted(grouping.items()):
            cut=4+2*i
            for start in range(0,len(indices),batch_size):
                use=np.array(indices[start:start+batch_size])
                ids=torch.as_tensor(ids_for([nodes[k]['permutation'] for k in use],task),device=device)
                old=run_prefix(model,ids,cut-1)
                after=resume(model,ids[:,cut:cut+1],cut,old).states
                states[use,0]=old.cpu().numpy();states[use,1]=after.cpu().numpy()
                if tag=='test':
                    original=model(ids)[:,-1].cpu().numpy()
                    continuation=resume(model,ids[:,cut+1:],cut+1,after).logits[:,-1].cpu().numpy()
                    error=float(np.max(np.abs(original-continuation)))
                    max_logit_error=max(max_logit_error,error)
                    argmax_mismatches+=int(np.count_nonzero(original.argmax(1)!=continuation.argmax(1)))
                    assert error<=1e-4,('Native/no-op continuation mismatch',unit,task,n,i,error)
                    native[use]=original;noop[use]=continuation
            if i==n-2:print(f'STATES {unit} {task} {tag} n={n}',flush=True)
    states.flush();del states
    temporary.replace(folder/'states.npy')
    assert before_hash==parameter_hash(model)
    files={str(folder/'states.npy'):sha(folder/'states.npy')}
    if tag=='test':
        np.savez_compressed(folder/'native_logits.npz',native_logits=native,noop_logits=noop)
        files[str(folder/'native_logits.npz')]=sha(folder/'native_logits.npz')
    write(receipt,dict(status='completed',unit=unit,task=task,pools=list(pools),node_ids=[r['node_id'] for r in nodes],
        node_count=len(nodes),files=files,model_parameter_sha256=before_hash,
        maximum_original_vs_noop_logit_difference=max_logit_error,
        original_vs_noop_argmax_mismatches=argmax_mismatches,
        original_vs_noop_logit_tolerance=1e-4,actual_state='GRU h_n per layer, before/after right-value update',
        state_axis=['node','point_before_after','layer','coordinate'],elapsed_seconds=time.perf_counter()-started,
        no_optimizer=True,no_backward=True))
    return np.load(folder/'states.npy',mmap_mode='r'),nodes,lookup,read(receipt)
