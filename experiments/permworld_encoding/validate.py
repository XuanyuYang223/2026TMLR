"""Meaningful exhaustive encoding/label/transform and causality checks."""
import argparse
from itertools import permutations
import numpy as np
import torch
from .core import *
from experiments.longrun_attention import accelerate


def run(output):
    count=0;relations=0;maxwidth=0
    rng=np.random.default_rng(73061901)
    examples=[]
    for n in range(1,8):examples.extend(permutations(range(1,n+1)))
    for n in range(10,31):examples.extend(tuple(map(int,rng.permutation(n)+1)) for _ in range(256))
    for p in examples:
        original=labels(p)
        rendered={e:encode(p,e) for e in ENCODINGS}
        assert len(rendered['one_line'])==len(rendered['canonical_cycle'])==2*len(p)+4
        for e,x in rendered.items():
            q=decode(x,e);assert q==p;assert labels(q)==original
            other=ENCODINGS[1-ENCODINGS.index(e)]
            assert decode(encode(q,other),other)==p
        for action in ACTIONS:
            q=transform(p,action)
            for e in ENCODINGS:assert decode(encode(q,e),e)==q
        for r in RELATIONS:
            q=transform(p,r['input_transform'])
            assert PROPERTY_FUNCTIONS[r['right']](q)==PROPERTY_FUNCTIONS[r['left']](p)+r['right_label_offset']
            wrong='complement' if r['input_transform']=='inverse' else 'inverse'
            # Wrong means the original comparator, not a guarantee of wrong labels on every sample.
            for e in ENCODINGS:
                assert decode(encode(transform(p,wrong),e),e)==transform(p,wrong)
            relations+=1
        count+=1;maxwidth=max(maxwidth,len(rendered['one_line']))
    # Independent known cycle example fixes orientation, rotation and cycle ordering.
    p=(2,3,1,5,4,6)
    assert canonical_cycles(p)==((1,2,3),(4,5),(6,))
    assert decode(encode(p,'canonical_cycle'),'canonical_cycle')==p
    invalid=0
    for e in ENCODINGS:
        good=list(encode(p,e))
        bads=[good[:-1],good+[TOKEN_TO_ID[',']],good[:4]+[TOKEN_TO_ID['01']]*len(good[4:-1])+good[-1:]]
        if e=='canonical_cycle':
            bad=good.copy();bad[4],bad[6]=bad[6],bad[4];bads.append(bad)
        for bad in bads:
            try:decode(bad,e)
            except ValueError:invalid+=1
            else:raise AssertionError('Accepted invalid encoding')
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    config={'model':{'max_seq_len':128,'d_model':256,'layers':4,'heads':8,'dropout':.1,'mlp_ratio':4,'tie_embeddings':True}}
    model=make_model(config,73061902).eval();same=make_model(config,73061902).eval()
    assert parameter_hash(model)==parameter_hash(same)
    causal_error=0.;acceleration_error=0.
    for e in ENCODINGS:
        seq=encode(p,e);prefix=np.full((1,64),TOKEN_TO_ID['<PAD>']);prefix[0,:len(seq)]=seq
        x,mask,target,end,eq=batch(prefix,[len(p)],TASKS[0],answers=[labels(p)[0]])
        captured=[];hook=model.final_norm.register_forward_hook(lambda m,i,o:captured.append(o.detach().clone()))
        with torch.inference_mode():full=model(x,mask);fullhidden=captured[-1][0,end[0]];only=model(x[:,:len(seq)],mask[:,:len(seq)]);pre=captured[-1][0,-1]
        causal_error=max(causal_error,float((pre-fullhidden).abs().max()));hook.remove()
        assert causal_error<1e-5
        assert (target!=-100).sum()==2 and int(end[0])==len(seq)-1
        assert target[0,eq[0]+1]==x[0,eq[0]+1] and target[0,eq[0]+2]==TOKEN_TO_ID['<EOS>']
        accelerated=accelerate(make_model(config,73061902).eval())
        with torch.inference_mode():fast=accelerated(x,mask)
        acceleration_error=max(acceleration_error,float((full-fast).abs().max()));assert acceleration_error<1e-5
    result=dict(status='passed',permutations=count,exhaustive_max_n=7,random_per_length=256,random_lengths=list(range(10,31)),
        relation_identity_checks=relations,invalid_serializations_rejected=invalid,
        matched_token_lengths=True,prefix_max_tokens=maxwidth,supervised_max_tokens=maxwidth+4,
        vocabulary_size=len(VOCABULARY),tasks=list(TASKS),relations=RELATIONS,
        canonical_rule='Follow pi, start each cycle at its minimum, order cycles by minima, keep singleton cycles; comma inside, semicolon between.',
        prefix_causal_max_abs_error=causal_error,sdpa_max_abs_error=acceleration_error,
        shared_initial_parameter_identity_checked=True,training_started=False)
    json_write(output,result);print(result|{'relations':'eight fixed relations','tasks':'sixteen fixed tasks'},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();run(a.output)
