"""Property32 cache-only drift diagnosis for all historical pool splits."""
import json
from collections import defaultdict
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from common import ROOT, REPO, load_csv, write_csv, write_json, centered, alignment, complete_mean, weighted_bootstrap_cka


def cka(x,y):
    x=centered(x);y=centered(y)
    den=np.sqrt(np.square(x.T@x).sum()*np.square(y.T@y).sum())
    return float(np.square(x.T@y).sum()/den) if den>1e-20 else None


def run():
    data=np.load(ROOT/'k_probe_inputs.npz');ns=data['lengths']
    jobs=[r for r in load_csv(ROOT/'model_inventory.csv') if r['family']=='k_series']
    kernels={};values={}
    inits={s:np.load(ROOT/f'k_initial_reference_seed{s}.npy').astype(float) for s in [17,42,101]}
    grouped=defaultdict(dict)
    for j in jobs:
        ident=(j['replicate'],int(j['k']));grouped[ident][j['pool']]=j
        values[j['model_id']]=torch.load(j['activation_cache'],map_location='cpu',weights_only=False)['layers']['final_norm'].numpy().astype(float)
    rng=np.random.default_rng(2026100902)
    weights={n:np.array([np.bincount(rng.integers(0,int((ns==n).sum()),int((ns==n).sum())),minlength=int((ns==n).sum())) for _ in range(256)],float) for n in range(2,31)}
    initgrams={(s,n):(z:=centered(a[ns==n]))@z.T for s,a in inits.items() for n in range(2,31)}
    rows=[];ownrows=[];pooled=[];shared=[];bootrows=[];bootstrap={}
    for (rep,k),pool in grouped.items():
        a=pool['a'];b=pool['b'];seed=int(a['seed']);assert int(b['seed'])==seed
        x=values[a['model_id']];y=values[b['model_id']];init=inits[seed]
        identical=a['initial_weight_hash_rebuilt']==b['initial_weight_hash_rebuilt']
        assert identical
        shared.append(dict(replicate=rep,k=k,seed=seed,pool_a_checkpoint_sha256=a['checkpoint_sha256'],pool_b_checkpoint_sha256=b['checkpoint_sha256'],
            rebuilt_weights_identical=identical,pool_a_initial_weight_hash=a['initial_weight_hash_rebuilt'],pool_b_initial_weight_hash=b['initial_weight_hash_rebuilt'],
            original_shared_initialization_verified=False,initialization_status=a['initial_status']))
        boot=np.zeros((3,256))
        for n in range(2,31):
            use=ns==n;xx=centered(x[use]);yy=centered(y[use]);kx=xx@xx.T;ky=yy@yy.T;ki=initgrams[seed,n]
            inter=alignment(kx,ky);selfa=alignment(kx,ki);selfb=alignment(ky,ki)
            rows.append(dict(replicate=rep,k=k,seed=seed,length=n,anchors=int(use.sum()),intermodel_cka=inter,
                pool_a_self_initial_cka=selfa,pool_b_self_initial_cka=selfb,mean_self_initial_cka=complete_mean([selfa,selfb]),
                original_initial_identity_verified=False))
            for job,selfvalue in [(a,selfa),(b,selfb)]:
                ownrows.append(dict(replicate=rep,k=k,seed=seed,pool=job['pool'],model_id=job['model_id'],length=n,anchors=int(use.sum()),
                    self_initial_cka=selfvalue,initial_status=job['initial_status']))
            for i,(ka,kb) in enumerate([(kx,ky),(kx,ki),(ky,ki)]):
                boot[i]+=weighted_bootstrap_cka(ka,kb,weights[n])/29
        pooled.append(dict(replicate=rep,k=k,seed=seed,anchors=len(ns),intermodel_cka=cka(x,y),
            pool_a_self_initial_cka=cka(x,init),pool_b_self_initial_cka=cka(y,init)))
        bootstrap[f'{rep}_k{k}']=boot
        for metric,ds in [('intermodel',boot[0]),('mean_self_initial',(boot[1]+boot[2])/2)]:
            bootrows.append(dict(replicate=rep,k=k,seed=seed,metric=metric,input_bootstrap_low=float(np.quantile(ds,.025)),
                input_bootstrap_high=float(np.quantile(ds,.975)),scope='paired length-stratified input sampling conditional on fixed models'))
        print({'k_complete':rep,'k':k},flush=True)
    summary=[]
    for (rep,k),pool in grouped.items():
        rs=[r for r in rows if r['replicate']==rep and r['k']==k]
        row=dict(replicate=rep,k=k,seed=int(pool['a']['seed']),lengths=29,fixed_weight='1/29',
            repetition_type='source_seed_run' if rep in ['r0','r1','r2'] else 'subset_sensitivity_fixed_seed17')
        for f in ['intermodel_cka','pool_a_self_initial_cka','pool_b_self_initial_cka','mean_self_initial_cka']:
            row[f]=complete_mean([r[f] for r in rs]);row[f+'_length_median']=float(np.median([r[f] for r in rs]))
        summary.append(row)
    trend=[]
    for rep in ['r0','r1','r2','r3','r4']:
        rs=sorted((r for r in summary if r['replicate']==rep),key=lambda r:r['k'])
        for metric in ['intermodel_cka','mean_self_initial_cka']:
            vs=np.array([r[metric] for r in rs]);ds=np.diff(vs)
            trend.append(dict(replicate=rep,metric=metric,k1=float(vs[0]),k16=float(vs[-1]),k16_minus_k1=float(vs[-1]-vs[0]),
                increasing_adjacent_steps=int((ds>0).sum()),adjacent_steps=4,
                spearman_k=float(np.corrcoef(np.arange(5),np.argsort(np.argsort(vs)))[0,1]),
                interpretation='descriptive fixed-run trajectory, not independent five-point inference'))
    write_csv(ROOT/'k_length_cka.csv',rows);write_csv(ROOT/'k_model_self_initial_length.csv',ownrows)
    write_csv(ROOT/'k_pooled_control.csv',pooled);write_csv(ROOT/'k_summary.csv',summary)
    write_csv(ROOT/'k_shared_initialization_audit.csv',shared);write_csv(ROOT/'k_input_bootstrap.csv',bootrows)
    write_csv(ROOT/'k_trend.csv',trend);np.savez_compressed(ROOT/'k_input_bootstrap_draws.npz',**bootstrap)
    fig,axs=plt.subplots(1,2,figsize=(12,4.5),sharex=True,sharey=True)
    colors=plt.cm.tab10(np.arange(5))
    for i,rep in enumerate(['r0','r1','r2','r3','r4']):
        rs=sorted((r for r in summary if r['replicate']==rep),key=lambda r:r['k'])
        label=f'{rep.upper()}, seed {rs[0]["seed"]}'+(' (subset)' if i>=3 else '')
        for ax,metric,bmetric in zip(axs,['intermodel_cka','mean_self_initial_cka'],['intermodel','mean_self_initial']):
            lows=[r['input_bootstrap_low'] for r in bootrows if r['replicate']==rep and r['metric']==bmetric]
            highs=[r['input_bootstrap_high'] for r in bootrows if r['replicate']==rep and r['metric']==bmetric]
            ax.plot([r['k'] for r in rs],[r[metric] for r in rs],marker='o',linestyle='--' if i>=3 else '-',color=colors[i],label=label)
            # Raw bootstrap percentiles have finite-sample shifts; do not draw nominal CI bands.
    axs[0].set_title('Between task pools A and B');axs[1].set_title('Retention relative to own rebuilt initialization')
    for ax in axs:
        ax.set_xscale('log',base=2);ax.set_xticks([1,2,4,8,16],[1,2,4,8,16]);ax.set_ylim(0,1);ax.set_xlabel('Number of tasks k');ax.grid(alpha=.2)
    axs[0].set_ylabel('Mean of 29 within-length linear CKA values');axs[1].legend(fontsize=8,loc='best')
    fig.suptitle('Exploratory checkpoint diagnosis; original initialization identity unverified',fontsize=10)
    fig.tight_layout();fig.savefig(ROOT/'k_drift.png',dpi=180);fig.savefig(ROOT/'k_drift.pdf');plt.close(fig)
    write_json(ROOT/'k_execution.json',dict(models=50,pool_pairs=25,per_length_rows=len(rows),original_probe_reused=True,
        equal_length_weights=True,pooled_control_separate=True,training_started=False))


if __name__=='__main__':run()
