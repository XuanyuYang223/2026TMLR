"""Preparation-stage integration checks, without a trained research model."""
import argparse
import copy
import json
import tempfile
from pathlib import Path
import numpy as np
import torch
from .core import *
from .data import contains,generate_world
from .evaluate import alignment,center,components,token_gram,u_center
from .train import rng_state,restore_rng,step0


def run(candidate_path,output):
    config=load_config(candidate_path);rng=np.random.default_rng(73061905);results={}
    # Membership includes an all-zero byte digest and a digest ending in zero.
    hashes=[b'\0'*16,b'123456789012345\0',fingerprint((2,1))]
    index=np.unique(np.array(hashes,dtype='V16'))
    assert contains(index,[*hashes,b'abcdefabcdefabcd']).tolist()==[True,True,True,False]
    results['hash_index_boundary_cases']=True
    # Zero-change synthetic states must yield Delta=0 even with initial/trained preferences nonzero.
    values={'initial':[.2,.7,.9],'trained':[.2,.7,.9]};c=components(values,0,1)
    assert c['delta']==0 and c['initial_preference']==c['trained_preference'] and abs(c['initial_preference']+.5)<1e-14
    for _ in range(100):
        a=rng.random(3);b=rng.random(3);d=components({'initial':a,'trained':b},1,2)
        assert abs(d['delta']-(d['change_correct']-d['change_wrong']))<1e-14
    # Demonstrate the requested non-flip: trained preference stays +.2 while Delta changes sign.
    d0=components({'initial':[.9,.1,.1],'trained':[.6,.4,.1]},0,1)
    d1=components({'initial':[.1,.9,.1],'trained':[.6,.4,.1]},0,1)
    assert d0['delta']<0<d1['delta'] and d0['trained_preference']==d1['trained_preference']>0
    results['delta_sign_flip_without_trained_preference_flip']=True
    # Kernel matches an independent explicit categorical feature matrix with within-stratum centering.
    ids=rng.integers(0,5,(30,8));keys=np.repeat(np.arange(10),3)[:,None]
    explicit=center(np.eye(5)[ids].reshape(30,-1),keys)
    assert np.max(np.abs(token_gram(ids,keys)-explicit@explicit.T))<1e-12
    results['fixed_token_kernel_independently_verified']=True
    torch.set_num_threads(2)
    with tempfile.TemporaryDirectory(prefix='permworld_encoding_fixture_') as temporary:
        root=Path(temporary);cfg=copy.deepcopy(config)
        cfg.update(lengths=[10,11],data={'examples_per_length':{'train':8,'validation':4,'test':8},'maximum_attempts_per_world':10000})
        cfg['training'].update(updates=5,effective_batch=4)
        unit={'unit':'fixture','data_seed':73061906,'batch_seed':73061907,'model_seed':73061908,'dropout_seed_base':73061909}
        generate_world(cfg,unit,np.empty(0,dtype='V16'),{'fixture_only':True,'sha256':'empty'},root)
        data=dict(np.load(root/'data.npz'));seen={}
        for split in ['train','validation','test']:
            for row,n in zip(data[split+'_permutations'],data[split+'_lengths']):
                p=tuple(map(int,row[:n]));key=orbit_key(p);assert key not in seen;seen[key]=split
            for e in ENCODINGS:
                for i,n in enumerate(data[split+'_lengths']):assert decode(data[f'{split}_{e}_input'][i,0,:2*n+4],e)==tuple(data[split+'_permutations'][i,:n])
        schedule=np.load(root/'batch_indices.npy');assert schedule.shape==(5,4)
        for row in schedule:assert len(set(data['train_lengths'][row]))==1 and len(set(row))==4
        chosen=schedule[0];x,mask,y,end,eq=batch(data['train_one_line_input'][chosen,0],data['train_lengths'][chosen],TASKS[0],data['train_labels'][chosen,0,0])
        assert end.dtype==torch.long and (y!=-100).sum()==2*len(chosen)
        assert torch.all(x[torch.arange(len(chosen)),eq]==TOKEN_TO_ID['='])
        # Snapshot and clone integration: construct but do not optimize any model.
        freeze=root/'dummy_freeze.json';json_write(freeze,{'code_sha256':{}});cfg['freeze_manifest']=str(freeze)
        saved,meta=step0(cfg,unit,root,'fixture_protocol')
        assert sha(root/'step0.pt')==meta['checkpoint_sha256']
        a=make_model(cfg,unit['model_seed']);b=make_model(cfg,unit['model_seed']+1)
        a.load_state_dict(saved['model']);b.load_state_dict(saved['model'])
        assert parameter_hash(a)==parameter_hash(b)==meta['parameter_sha256']
        s=rng_state();expected=torch.rand(10);restore_rng(s);assert torch.equal(expected,torch.rand(10))
        # Temporary fixture removed on context exit; no research step-0/data/checkpoint remains.
        results.update(fixture_orbit_split_and_batch_identity=True,real_step0_save_load_clone_and_rng_replay=True,
            fixture_actual_optimizer_updates=0,fixture_research_models_saved=False)
    json_write(output,dict(status='passed',checks=results,formal_training_started=False))
    print(results,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--output',required=True);a=p.parse_args();run(a.config,a.output)
