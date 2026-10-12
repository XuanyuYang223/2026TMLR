"""Independent numerical checks and protected-artifact byte verification."""
import json
import subprocess
from collections import defaultdict
import numpy as np
import torch
from common import ROOT, WORKSPACE, load_csv, write_json, sha, centered, alignment, gram, u_center, weighted_bootstrap_cka


def feature_cka(x,y):
    x=centered(x);y=centered(y)
    den=np.linalg.norm(x.T@x)*np.linalg.norm(y.T@y)
    return float(np.linalg.norm(x.T@y)**2/den) if den>1e-20 else None


def independent_debiased(k,l):
    """Unbiased HSIC via trace/row sums, independently of U-centering code."""
    n=len(k)
    def hsic(a,b):
        a=a.copy();b=b.copy();np.fill_diagonal(a,0);np.fill_diagonal(b,0)
        return (np.trace(a@b)+a.sum()*b.sum()/((n-1)*(n-2))-2*np.dot(a.sum(1),b.sum(1))/(n-2))/(n*(n-3))
    den=np.sqrt(max(hsic(k,k)*hsic(l,l),0))
    return float(hsic(k,l)/den) if den>1e-20 else None


def run():
    tests=[];maxerr=0.
    def check(name,a,b,tol=3e-11):
        nonlocal maxerr
        if a is None or b is None:assert a is None and b is None,(name,a,b);return
        err=abs(float(a)-float(b));assert err<tol,(name,a,b,err)
        maxerr=max(maxerr,err);tests.append(dict(check=name,max_abs_error=err))
    rng=np.random.default_rng(101)
    for n in [7,31,128]:
        x=rng.normal(size=(n,23));y=rng.normal(size=(n,11));k=gram(x);l=gram(y)
        check('U_center_vs_unbiased_HSIC_trace',alignment(u_center(k),u_center(l)),independent_debiased(k,l))
        for _ in range(3):
            ids=rng.integers(0,n,n);w=np.bincount(ids,minlength=n)[None,:]
            check('weighted_CKA_vs_explicit_resampling',weighted_bootstrap_cka(k,l,w)[0],feature_cka(x[ids],y[ids]))
        q=np.linalg.qr(rng.normal(size=(23,23)))[0]
        check('orthogonal_coordinate_invariance',feature_cka(x,y),feature_cka(x@q,y))
    assert alignment(gram(np.ones((10,4))),gram(rng.normal(size=(10,8)))) is None
    # Exhaustive arithmetic and fixed-weight checks on all saved cells.
    rows=load_csv(ROOT/'relation_length_cka.csv');assert len(rows)==9072
    groups=defaultdict(list)
    for r in rows:
        ds=float(r['trained_correct'])-float(r['trained_wrong'])-float(r['init_correct'])+float(r['init_wrong'])
        check('per_cell_matched_initial_delta',ds,float(r['delta']))
        groups[r['relation'],r['left_seed'],r['right_seed'],r['control'],r['metric']].append(r)
        assert r['initialization_status']=='program_rebuilt_original_identity_unverified'
    for r in load_csv(ROOT/'relation_seed_summary.csv'):
        rs=groups[r['relation'],r['left_seed'],r['right_seed'],r['control'],r['metric']]
        assert sorted(int(v['length']) for v in rs)==list(range(10,31))
        check('equal_length_weight_1_over_21',float(r['delta']),sum(float(v['delta'])/21 for v in rs))
    # Evaluate selected real cells through feature-space products and trace-based HSIC.
    parent=WORKSPACE/'results/specialist_cka_controls';data=dict(np.load(parent/'dataset.npz'));test=data['split']==2
    ns=data['lengths'][test];tasks=list(data['tasks']);sig=json.loads((parent/'protocol.json').read_text())['signature']
    for rel in sig['relations']:
        ya=data['labels'][test,0,tasks.index(rel['left'])];yb=data['labels'][test,:,tasks.index(rel['right'])]
        aidx=list(data['transforms']).index(rel['input_transform'])
        assert np.array_equal(ya+rel['right_label_offset'],yb[:,aidx])
        keys=np.column_stack([ns,ya,yb]);_,inv,cnt=np.unique(keys,axis=0,return_inverse=True,return_counts=True);keep=cnt[inv]>=3
        for sa,sb in [(17,17),(17,42),(101,42)]:
            for con in ['length','answer_strata']:
                for cond in ['trained','matched_init']:
                    with np.load(parent/'features'/f'{cond}_{rel["left"]}_s{sa}.npz') as a, np.load(parent/'features'/f'{cond}_{rel["right"]}_s{sb}.npz') as b:
                        xa=a['prefix_final_norm'][test,0];ybv=b['prefix_final_norm'][test]
                    for n in [10,20,30]:
                        use=(ns==n)&(keep if con=='answer_strata' else True);key=keys[use] if con=='answer_strata' else None
                        x=centered(xa[use],key)
                        for act,field in [(aidx,'correct'),(1 if aidx==2 else 2,'wrong'),(0,'identity')]:
                            y=centered(ybv[use,act],key)
                            for met,value in [('ordinary',feature_cka(x,y)),('debiased',independent_debiased(x@x.T,y@y.T))]:
                                r=next(v for v in groups[rel['pair_id'],str(sa),str(sb),con,met] if int(v['length'])==n)
                                check('saved_cell_independent_'+met,value,float(r[('trained' if cond=='trained' else 'init')+'_'+field]))
    # Verify all historical pooled inter-pool CKA values were retained exactly.
    sourcejobs=load_csv(ROOT/'model_inventory.csv')
    for r in load_csv(ROOT/'k_pooled_control.csv'):
        job=next(j for j in sourcejobs if j['family']=='k_series' and j['replicate']==r['replicate'] and j['k']==r['k'] and j['pool']=='a')
        from pathlib import Path
        parentcka=Path(job['activation_cache']).parent.parent
        old=next(v for v in load_csv(parentcka/'pairwise_layer_cka.csv') if v['comparison']=='disjoint_pools_equal_k' and v['task_count_a']==r['k'] and v['layer']=='final_norm')
        check('historical_pooled_k_CKA',float(r['intermodel_cka']),float(old['linear_cka']))
    kdata=np.load(ROOT/'k_probe_inputs.npz');kns=kdata['lengths'];krows=load_csv(ROOT/'k_length_cka.csv')
    for r in krows:
        if int(r['length']) not in [2,16,30]:continue
        jobs=[j for j in sourcejobs if j['family']=='k_series' and j['replicate']==r['replicate'] and j['k']==r['k']]
        a=next(j for j in jobs if j['pool']=='a');b=next(j for j in jobs if j['pool']=='b');use=kns==int(r['length'])
        x=torch.load(a['activation_cache'],map_location='cpu',weights_only=False)['layers']['final_norm'].numpy()[use]
        y=torch.load(b['activation_cache'],map_location='cpu',weights_only=False)['layers']['final_norm'].numpy()[use]
        init=np.load(ROOT/f'k_initial_reference_seed{r["seed"]}.npy')[use]
        for name,xx,yy in [('intermodel_cka',x,y),('pool_a_self_initial_cka',x,init),('pool_b_self_initial_cka',y,init)]:
            check('within_length_k_independent_feature_CKA',float(r[name]),feature_cka(xx,yy))
    audit=json.loads((ROOT/'initialization_audit.json').read_text())
    assert audit['logical_models']==98 and audit['unique_checkpoints']==94
    assert audit['original_initial_weights_found']==[]
    assert all(not r['original_initial_identity_verified'] for r in audit['model_configs_and_rng'])
    for seed in [17,42,101]:
        assert len({r['initial_weight_hash_rebuilt'] for r in sourcejobs if int(r['seed'])==seed})==1
    shared=load_csv(ROOT/'k_shared_initialization_audit.csv');assert len(shared)==25
    assert all(r['rebuilt_weights_identical']=='True' and r['original_shared_initialization_verified']=='False' for r in shared)
    assert len(load_csv(ROOT/'k_model_self_initial_length.csv'))==1450
    # The protected snapshot includes F17 publication files and the parent diagnostics.
    protected=json.loads((ROOT/'protected_before.json').read_text());changes=[]
    for path,digest in protected['files'].items():
        full=WORKSPACE/path
        if not full.exists() or sha(full)!=digest:changes.append(path)
    assert not changes,changes
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=WORKSPACE,text=True).strip();assert head==protected['git_head']
    tracked=subprocess.check_output(['git','diff','--name-only','HEAD'],cwd=WORKSPACE,text=True).strip();assert tracked=='',tracked
    write_json(ROOT/'verification.json',dict(status='passed',numeric_checks=len(tests),maximum_absolute_discrepancy=maxerr,
        checks_by_name={name:sum(t['check']==name for t in tests) for name in sorted({t['check'] for t in tests})},
        original_initial_identity_claims_checked=True,protected_files_checked=len(protected['files']),protected_files_changed=changes,
        git_head_unchanged=True,tracked_worktree_unchanged=True,training_started=False))
    print({'verification':'passed','numeric_checks':len(tests),'protected_files':len(protected['files']),'max_error':maxerr},flush=True)


if __name__=='__main__':run()
