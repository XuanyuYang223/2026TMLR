"""All eight fixed relations, corresponding initialization pairs, no inference/training."""
import json
from itertools import product
import numpy as np

from common import ROOT, WORKSPACE, centered, alignment, u_center, write_csv, write_json, load_csv, complete_mean, weighted_bootstrap_cka
from experiments.specialist_input_adjustment import adjust

SEEDS=[17,42,101]
PAIRS=list(product(SEEDS,repeat=2))
LENGTHS=list(range(10,31))
METRICS=['ordinary','debiased','input_adjusted']
PARENT=WORKSPACE/'results/specialist_cka_controls'


def sub(a,b):
    return None if a is None or b is None else float(a-b)


def run():
    sig=json.loads((PARENT/'protocol.json').read_text())['signature']
    ds=dict(np.load(PARENT/'dataset.npz')); test=ds['split']==2
    d={k:v[test] for k,v in ds.items() if k not in ['tasks','transforms']}
    tasks=list(ds['tasks']); actions=list(ds['transforms']); ns=d['lengths']
    old={}
    for row in load_csv(WORKSPACE/'results/specialist_input_adjustment/per_length_scores.csv'):
        if row['branch']=='prefix_final_norm' and row['condition'] in ['trained','matched_init']:
            key=(row['relation'],int(row['seed']),row['condition'],row['control'],int(row['length']),row['transform'])
            old[key]=(float(row['ordinary_cka']),float(row['input_adjusted_cka']))
    # These arrays, not fresh model runs, feed all new comparisons.
    features={}; outputs={}
    for task in tasks:
        for seed in SEEDS:
            for cond in ['trained','matched_init']:
                with np.load(PARENT/'features'/f'{cond}_{task}_s{seed}.npz') as a:
                    features[task,seed,cond]=a['prefix_final_norm'][test].astype(float)
                    outputs[task,seed,cond]=a['output_prob'][test].astype(float)
    rng=np.random.default_rng(2026100901)
    weights={n:np.array([np.bincount(rng.integers(0,128,128),minlength=128) for _ in range(256)],float) for n in LENGTHS}
    rows=[]; baselines=[]; inputrows=[]; permrows=[]; nulls={}; bootstrap={}; stratumrows=[]; max_reuse_error=0.
    for ri,r in enumerate(sig['relations']):
        relation=r['pair_id']; ya=d['labels'][:,0,tasks.index(r['left'])]; yb=d['labels'][:,:,tasks.index(r['right'])]
        correct=actions.index(r['input_transform']); wrong=1 if correct==2 else 2
        keys=np.column_stack([ns,ya,yb]); _,inv,counts=np.unique(keys,axis=0,return_inverse=True,return_counts=True)
        keep=counts[inv]>=3; selected=keys[keep]
        _,sinv=np.unique(selected,axis=0,return_inverse=True)
        prng=np.random.default_rng(2026100702+ri); permutations=[]
        for _ in range(32):
            order=np.arange(len(selected))
            for group in np.unique(sinv):
                ids=np.flatnonzero(sinv==group);order[ids]=prng.permutation(ids)
            permutations.append(order)
        permutations=np.array(permutations)
        # Seeds are fixed; all input bootstrap draws are shared across all relations.
        boot=np.zeros((len(PAIRS),256)); nuls=np.zeros((len(PAIRS),len(LENGTHS),32))
        for ni,n in enumerate(LENGTHS):
            for control in ['length','answer_strata']:
                use=(ns==n)&(keep if control=='answer_strata' else True)
                key=keys[use] if control=='answer_strata' else None
                m=int(use.sum()); stratumrows.append(dict(relation=relation,length=n,control=control,anchors=m,
                    common_strata=int(len(np.unique(keys[use],axis=0))) if control=='answer_strata' else ''))
                inputs=np.eye(n)[d['permutations'][use,0,:n]-1].reshape(m,n*n)
                z=centered(inputs,key);g=z@z.T
                for action in range(3):
                    other=centered(np.eye(n)[d['permutations'][use,action,:n]-1].reshape(m,n*n),key)
                    np.testing.assert_allclose(g,other@other.T,atol=1e-12)
                    if control=='length' and ri==0:
                        v=centered(d['permutations'][use,0,:n]/n);w=centered(d['permutations'][use,action,:n]/n)
                        inputrows.append(dict(length=n,transform=actions[action],numeric_cka=alignment(v@v.T,w@w.T),
                            permutation_matrix_cka=alignment(g,other@other.T),anchors=m))
                # Small shared kernels; no hidden state pooling across lengths.
                kernels={}; adjusted={}; debiased={}; outkernels={}
                for cond in ['trained','matched_init']:
                    for seed in SEEDS:
                        for side,task in [('left',r['left']),('right',r['right'])]:
                            for action in ([0] if side=='left' else range(3)):
                                ident=(cond,seed,side,action)
                                z=centered(features[task,seed,cond][use,action],key);k=z@z.T
                                kernels[ident]=k;debiased[ident]=u_center(k);adjusted[ident]=adjust(k,g)[0]
                                o=centered(outputs[task,seed,cond][use,action],key);outkernels[ident]=o@o.T
                # Existing gold baselines, now summarized within each original length as well.
                for branch in ['answer_onehot','answer_scalar']:
                    x=np.eye(31)[ya[use]] if branch=='answer_onehot' else ya[use,None]
                    x=centered(x,key);k=x@x.T
                    vals=[]
                    for a in range(3):
                        y=np.eye(31)[yb[use,a]] if branch=='answer_onehot' else yb[use,a,None]
                        y=centered(y,key);vals.append(alignment(k,y@y.T))
                    baselines.append(dict(relation=relation,length=n,control=control,branch=branch,left_seed='',right_seed='',
                        trained_correct=vals[correct],trained_wrong=vals[wrong],trained_identity=vals[0],
                        init_correct='',init_wrong='',init_identity='',trained_contrast=sub(vals[correct],vals[wrong]),
                        init_contrast='',delta='',anchors=m))
                if control=='answer_strata':
                    pos=np.flatnonzero(selected[:,0]==n); mapper=np.full(len(selected),-1);mapper[pos]=np.arange(m)
                    lp=mapper[permutations[:,pos]];assert (lp>=0).all()
                    assert all(np.array_equal(np.sort(p),np.arange(m)) for p in lp)
                for pi,(sa,sb) in enumerate(PAIRS):
                    typ='same_seed' if sa==sb else 'cross_seed'
                    for metric in METRICS:
                        mats={'ordinary':kernels,'debiased':debiased,'input_adjusted':adjusted}[metric]
                        values={}
                        for cond in ['trained','matched_init']:
                            k=mats[cond,sa,'left',0]
                            vals=[alignment(k,mats[cond,sb,'right',a]) if k is not None and mats[cond,sb,'right',a] is not None else None for a in range(3)]
                            if sa==sb and metric in ['ordinary','input_adjusted']:
                                prior=[old[relation,sa,cond,control,n,actions[a]][0 if metric=='ordinary' else 1] for a in range(3)]
                                max_reuse_error=max(max_reuse_error,max(abs(a-b) for a,b in zip(vals,prior)))
                                np.testing.assert_allclose(vals,prior,atol=2e-12,rtol=0)
                                vals=prior
                            values[cond]=vals
                        tv=values['trained'];iv=values['matched_init']
                        tc=sub(tv[correct],tv[wrong]);ic=sub(iv[correct],iv[wrong])
                        rows.append(dict(relation=relation,left_task=r['left'],right_task=r['right'],correct_transform=actions[correct],
                            wrong_transform=actions[wrong],left_seed=sa,right_seed=sb,seed_pair_type=typ,length=n,anchors=m,control=control,metric=metric,
                            trained_correct=tv[correct],trained_wrong=tv[wrong],trained_identity=tv[0],
                            init_correct=iv[correct],init_wrong=iv[wrong],init_identity=iv[0],
                            trained_contrast=tc,init_contrast=ic,delta=sub(tc,ic),
                            initialization_status='program_rebuilt_original_identity_unverified',
                            score_origin='reused_per_length_parent' if sa==sb and metric!='debiased' else 'new_cached_feature_comparison'))
                    ov={cond:[alignment(outkernels[cond,sa,'left',0],outkernels[cond,sb,'right',a]) for a in range(3)] for cond in ['trained','matched_init']}
                    tv=ov['trained'];iv=ov['matched_init'];tc=sub(tv[correct],tv[wrong]);ic=sub(iv[correct],iv[wrong])
                    baselines.append(dict(relation=relation,length=n,control=control,branch='output_prob',left_seed=sa,right_seed=sb,
                        trained_correct=tv[correct],trained_wrong=tv[wrong],trained_identity=tv[0],
                        init_correct=iv[correct],init_wrong=iv[wrong],init_identity=iv[0],trained_contrast=tc,init_contrast=ic,delta=sub(tc,ic),anchors=m))
                    if control=='length':
                        bv={}
                        for cond in ['trained','matched_init']:
                            k=kernels[cond,sa,'left',0]
                            bv[cond]=weighted_bootstrap_cka(k,kernels[cond,sb,'right',correct],weights[n])-weighted_bootstrap_cka(k,kernels[cond,sb,'right',wrong],weights[n])
                        boot[pi]+=(bv['trained']-bv['matched_init'])/len(LENGTHS)
                    else:
                        nv={}
                        for cond in ['trained','matched_init']:
                            k=kernels[cond,sa,'left',0];cv=[]
                            for a in [correct,wrong]:
                                l=kernels[cond,sb,'right',a];sh=l[lp[:,:,None],lp[:,None,:]]
                                den=np.sqrt(np.square(k).sum()*np.square(l).sum())
                                cv.append(np.einsum('ij,bij->b',k,sh)/den)
                            nv[cond]=cv[0]-cv[1]
                        nuls[pi,ni]=nv['trained']-nv['matched_init']
                        observed=next(x['delta'] for x in reversed(rows) if x['metric']=='ordinary')
                        vals=nuls[pi,ni]
                        permrows.append(dict(relation=relation,left_seed=sa,right_seed=sb,length=n,anchors=m,
                            observed_delta=observed,null_mean=float(vals.mean()),null_min=float(vals.min()),null_max=float(vals.max()),
                            descriptive_upper_rank=(1+int((vals>=observed).sum()))/33))
            # Optional progress is sparse and contains no outcome-based decision.
        bootstrap[relation]=boot;nulls[relation]=nuls
        print({'relation_complete':relation,'ordered_seed_pairs':len(PAIRS),'lengths':len(LENGTHS)},flush=True)
    write_csv(ROOT/'relation_length_cka.csv',rows);write_csv(ROOT/'answer_output_length_baselines.csv',baselines)
    write_csv(ROOT/'input_length_baselines.csv',inputrows);write_csv(ROOT/'answer_strata_audit.csv',stratumrows)
    write_csv(ROOT/'permutation_length_calibration.csv',permrows)
    np.savez_compressed(ROOT/'input_bootstrap_delta.npz',**bootstrap)
    np.savez_compressed(ROOT/'permutation_delta.npz',**nulls)
    summarize(rows,baselines,bootstrap,nulls)
    write_json(ROOT/'relation_execution.json',dict(rows=len(rows),relations=8,ordered_seed_pairs=9,lengths=21,
        reused_diagonal_scores=8*3*21*2*2*2*3,maximum_reuse_discrepancy=max_reuse_error,
        features_only=True,training_started=False))


def summarize(rows,baselines,bootstrap,nulls):
    from collections import defaultdict
    groups=defaultdict(list)
    for r in rows:groups[r['relation'],r['left_seed'],r['right_seed'],r['control'],r['metric']].append(r)
    src=[]
    for (rel,sa,sb,con,met),rs in groups.items():
        ds=[r['delta'] for r in rs]; finite=[v for v in ds if v is not None]
        row=dict(relation=rel,left_seed=sa,right_seed=sb,seed_pair_type='same_seed' if sa==sb else 'cross_seed',control=con,metric=met,
            delta=complete_mean(ds),length_median_delta=float(np.median(finite)) if finite else None,
            positive_lengths=sum(v>0 for v in finite),available_lengths=len(finite),
            length_min_delta=min(finite) if finite else None,length_max_delta=max(finite) if finite else None)
        for field in ['trained_correct','trained_wrong','trained_identity','init_correct','init_wrong','init_identity','trained_contrast','init_contrast']:
            row[field]=complete_mean([r[field] for r in rs])
        src.append(row)
    write_csv(ROOT/'relation_seed_summary.csv',src)
    ag=defaultdict(list)
    for r in src:ag[r['relation'],r['seed_pair_type'],r['control'],r['metric']].append(r)
    summary=[];bootrows=[];perms=[]
    for (rel,typ,con,met),rs in ag.items():
        vals=[r['delta'] for r in rs]; finite=[v for v in vals if v is not None]
        row=dict(relation=rel,seed_pair_type=typ,control=con,metric=met,delta_mean=complete_mean(vals),
            delta_median=float(np.median(finite)) if finite else None,positive_pairs=sum(v>0 for v in finite),pairs=len(rs),
            seed_pair_min=min(finite) if finite else None,seed_pair_max=max(finite) if finite else None,
            min_positive_lengths=min(r['positive_lengths'] for r in rs),max_positive_lengths=max(r['positive_lengths'] for r in rs))
        for field in ['trained_correct','trained_wrong','trained_identity','init_correct','init_wrong','init_identity','trained_contrast','init_contrast']:
            row[field]=complete_mean([r[field] for r in rs])
        summary.append(row)
        indices=[i for i,(a,b) in enumerate(PAIRS) if (a==b)==(typ=='same_seed')]
        if con=='length' and met=='ordinary':
            draws=bootstrap[rel][indices].mean(0)
            bootrows.append(dict(relation=rel,seed_pair_type=typ,observed_delta=row['delta_mean'],
                input_bootstrap_low=float(np.quantile(draws,.025)),input_bootstrap_high=float(np.quantile(draws,.975)),
                inference_scope='input anchors conditional on these fixed source models; not independent relationship or seed confidence'))
        if con=='answer_strata' and met=='ordinary':
            draws=nulls[rel][indices].mean((0,1))
            perms.append(dict(relation=rel,seed_pair_type=typ,observed_delta=row['delta_mean'],null_mean=float(draws.mean()),
                null_min=float(draws.min()),null_max=float(draws.max()),descriptive_upper_rank=(1+int((draws>=row['delta_mean']).sum()))/33))
    write_csv(ROOT/'relation_summary.csv',summary);write_csv(ROOT/'input_bootstrap_summary.csv',bootrows)
    write_csv(ROOT/'permutation_summary.csv',perms)
    macro=[];lor=[]
    for typ,con,met in product(['same_seed','cross_seed'],['length','answer_strata'],METRICS):
        rs=[r for r in summary if (r['seed_pair_type'],r['control'],r['metric'])==(typ,con,met)]
        vals=[r['delta_mean'] for r in rs]
        macro.append(dict(seed_pair_type=typ,control=con,metric=met,relation_count=8,delta_mean=complete_mean(vals),
            delta_median=float(np.median(vals)),positive_relations=sum(v>0 for v in vals)))
        for omitted in rs:
            ds=[r['delta_mean'] for r in rs if r['relation']!=omitted['relation']]
            lor.append(dict(seed_pair_type=typ,control=con,metric=met,omitted_relation=omitted['relation'],remaining_relations=7,
                delta_mean=complete_mean(ds),delta_median=float(np.median(ds)),positive_relations=sum(v>0 for v in ds)))
    write_csv(ROOT/'macro_summary.csv',macro);write_csv(ROOT/'leave_one_relation.csv',lor)
    lol=[]
    for key,rs in groups.items():
        rel,sa,sb,con,met=key
        for omit in LENGTHS:
            lol.append(dict(relation=rel,left_seed=sa,right_seed=sb,control=con,metric=met,omitted_length=omit,
                delta=complete_mean([r['delta'] for r in rs if r['length']!=omit])))
    write_csv(ROOT/'leave_one_length.csv',lol)
    macroboot=[]
    for typ in ['same_seed','cross_seed']:
        ids=[i for i,(a,b) in enumerate(PAIRS) if (a==b)==(typ=='same_seed')]
        b=np.stack([v[ids].mean(0) for v in bootstrap.values()]).mean(0)
        macroboot.append(dict(seed_pair_type=typ,input_bootstrap_low=float(np.quantile(b,.025)),input_bootstrap_high=float(np.quantile(b,.975)),
            fixed_relations=8,draws=256,shared_anchor_resampling=True))
    write_csv(ROOT/'macro_input_bootstrap.csv',macroboot)
    bg=defaultdict(list)
    for r in baselines:bg[r['relation'],r['control'],r['branch'],r['left_seed'],r['right_seed']].append(r)
    bs=[]
    for (rel,con,branch,sa,sb),rs in bg.items():
        row=dict(relation=rel,control=con,branch=branch,left_seed=sa,right_seed=sb)
        for f in ['trained_correct','trained_wrong','trained_identity','trained_contrast','init_contrast','delta']:
            vs=[r[f] if r[f]!='' else None for r in rs];row[f]=complete_mean(vs)
        bs.append(row)
    write_csv(ROOT/'answer_output_summary.csv',bs)


if __name__=='__main__':run()
