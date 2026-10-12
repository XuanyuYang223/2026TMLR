"""One frozen endpoint, three original metrics and fixed output/input controls."""
import math
import time
from torch.nn import functional as F
from .common import *

def selections(data,r,split,n,control):
    ns=data[split+'_lengths'];ys=data[split+'_labels'];ai=[0,ACTIONS.index(r['correct']),ACTIONS.index(r['wrong'])]
    left=ys[:,0,TASKS.index(r['left'])]
    right=mapped_right(ys[:,ai,TASKS.index(r['right'])],ns[:,None],r)
    keys=np.column_stack([ns,left,right]);use=ns==n
    if control=='answer_density_strata':keys=np.column_stack([keys,data[split+'_edges'][:,0]])
    if control!='raw_within_n':
        _,inverse,count=np.unique(keys,axis=0,return_inverse=True,return_counts=True);use&=count[inverse]>=3
    return use,keys[use] if control!='raw_within_n' else None

def components(v,static=False):
    row={f'{c}_{role}':v[c][role] for c in ('initial','trained') for role in ('correct','wrong','identity')}
    def sub(a,b):return a-b if a is not None and b is not None else None
    row['B0']=sub(row['initial_correct'],row['initial_wrong']);row['Bt']=sub(row['trained_correct'],row['trained_wrong'])
    for role,field in [('correct','dc'),('wrong','dw'),('identity','di')]:row[field]=sub(row['trained_'+role],row['initial_'+role])
    row['Delta']=sub(row['Bt'],row['B0'])
    if row['Delta'] is not None:assert abs(row['Delta']-(row['dc']-row['dw']))<1e-10
    if static:assert all(row[k] is None for k in ('initial_correct','initial_wrong','initial_identity','B0','dc','dw','di','Delta'))
    return row

def cache_model(c,job,condition,split,data,device):
    directory=ROOT/'evaluation_cache';directory.mkdir(exist_ok=True)
    path=directory/f"{job['task']}_{condition}_{split}.npz"
    if condition=='initial':cp_path=Path(c['initial_checkpoint_path']);cp_hash=c['initial_checkpoint_sha256'];cp=initial(c)
    else:
        done=read(ROOT/'models'/job['run_id']/'completed.json');cp_path=ROOT/'models'/job['run_id']/'checkpoint.pt';cp_hash=sha(cp_path)
        assert done['step']==20000 and done['checkpoint_sha256']==cp_hash and done['protocol_sha256']==sha(CONFIG)
        assert done['source_version_sha256']==source_version()
        cp=torch.load(cp_path,map_location='cpu',weights_only=False)
        for k,v in dict(step=20000,run_id=job['run_id'],protocol_sha256=sha(CONFIG),source_version_sha256=source_version(),
                       initial_parameter_sha256=c['initial_parameter_sha256'],starting_rng_hash=c['starting_rng_hash']).items():assert cp[k]==v
        assert cp['data_sha256']==read(ROOT/'world/data_audit.json')['data_sha256']
    signature=dict(protocol_sha256=sha(CONFIG),data_sha256=read(ROOT/'world/data_audit.json')['data_sha256'],
                   source_version_sha256=source_version(),checkpoint_sha256=cp_hash,task=job['task'],split=split,condition=condition)
    if path.exists():
        meta=read(path.with_suffix('.json'));assert meta['signature']==signature and sha(path)==meta['sha256']
        with np.load(path,allow_pickle=False) as ar:return {k:ar[k] for k in ar.files},meta
    net=model(c,'transformer',c['model_seed'],device);net.load_state_dict(cp['model'],strict=True);net.eval()
    before=parameter_hash(net);captured=[];hook=net.final_norm.register_forward_hook(lambda m,i,o:captured.append(o))
    arrays={};ns=data[split+'_lengths'];N=len(ns)
    arrays['prefix_final']=np.empty((N,5,256),np.float32);arrays['output_prob']=np.empty((N,5,31),np.float32)
    arrays['prediction']=np.empty((N,5),np.int16);arrays['answer_ce']=np.empty((N,5),np.float64)
    manual_records=[];begun=time.monotonic()
    from experiments.permworld_encoding_archdev_20261009.evaluation import manual_final
    try:
        with torch.inference_mode():
            for start in range(0,N,128):
                use=np.arange(start,min(start+128,N));nn=ns[use]
                for action in range(5):
                    x,m,_,end,eq=batch(data[split+'_input'][use,action],nn,job['task'],device=device)
                    captured.clear();z=net(x,m);rows=torch.arange(len(use),device=device)
                    h=captured[-1][rows,end].float();score=z[rows,eq]
                    arrays['prefix_final'][use,action]=h.cpu().numpy();arrays['output_prob'][use,action]=score[:,:31].softmax(-1).cpu().numpy()
                    arrays['prediction'][use,action]=score.argmax(-1).cpu().numpy()
                    gold=torch.as_tensor(data[split+'_labels'][use,action,TASKS.index(job['task'])].astype(np.int64),device=device)
                    arrays['answer_ce'][use,action]=F.cross_entropy(score,gold,reduction='none').cpu().double().numpy()
                    # Fixed first128 batch of each n; same shape, not smaller-prefix raw layers.
                    if split=='validation' and start in (0,256,512):
                        explicit=manual_final(net,x,m,'transformer')[rows,end]
                        difference=float((explicit[:8]-h[:8]).abs().max());assert difference<=2e-5
                        manual_records.append(dict(n=int(nn[0]),action=ACTIONS[action],vectors=8,max_abs=difference,same_shape=True))
        assert all(np.isfinite(a).all() for a in arrays.values()) and parameter_hash(net)==before
        assert all(p.grad is None for p in net.parameters())
    finally:hook.remove()
    np.savez_compressed(path,**arrays)
    meta=dict(signature=signature,sha256=sha(path),checkpoint_path=str(cp_path),shape={k:list(v.shape) for k,v in arrays.items()},
              actual_initial_identity_verified=True,parameter_sha256_before=before,parameter_sha256_after=before,
              evaluation='eval/inference_mode/FP32/no autocast/TF32 disabled; final_norm at graph input-end before task',
              manual_extraction_checks=manual_records,wall_seconds=time.monotonic()-begun,EOS_evaluated=False,source_updates=0)
    json_write(path.with_suffix('.json'),meta);return arrays,meta

def metric_values(xf,yf,xt,yt,kf,kt,ids,orbits):
    xf=metrics.center(xf,kf);yf=metrics.center(yf,kf);xt=metrics.center(xt,kt);yt=metrics.center(yt,kt)
    return metrics.scores(xf,yf,xt,yt,ids=ids,duplicate_keys=orbits)

def geometry(c,data,features):
    rows=[];independent=[];names=['linear_cka','knn_overlap','procrustes','debiased_cka']
    for r in RELATIONS:
        for n in c['lengths']:
            for control in c['evaluation']['controls']:
                fit,kf=selections(data,r,'validation',n,control);test,kt=selections(data,r,'analysis',n,control)
                ids=np.flatnonzero(test);orbits=data['analysis_orbit_id'][test,None]
                actions={'correct':ACTIONS.index(r['correct']),'wrong':ACTIONS.index(r['wrong']),'identity':0}
                for branch in ('prefix_final','output_prob','gold_onehot','gold_scalar','input_onehot'):
                    static=branch in ('gold_onehot','gold_scalar','input_onehot')
                    values={m:{cnd:{role:None for role in actions} for cnd in ('initial','trained')} for m in names}
                    errors={cnd:{role:None for role in actions} for cnd in ('initial','trained')};rank=None
                    for cond in (('trained',) if static else ('initial','trained')):
                        for role,ai in actions.items():
                            if branch=='input_onehot':
                                # Any fixed R/C/W permutes edge slots/categories orthogonally.
                                # Shared centered Grams are exact, eliminating floating tie artifacts.
                                from diagnostics.permworld_layers_20261009.input_geometry import scores_from_grams
                                def get_input(split,use,keys):
                                    graphs=data[split+'_graphs'][use,0,:n,:n]
                                    edge=graphs[:,~np.eye(n,dtype=bool)];x=np.eye(2)[edge].reshape(len(edge),2*n*(n-1))
                                    return metrics.center(x,keys)
                                xf=get_input('validation',fit,kf);xt=get_input('analysis',test,kt)
                                result,rank=scores_from_grams(xf@xf.T,xf@xf.T,xt@xt.T,xt@xt.T,xt@xf.T,xt@xf.T,ids=ids)
                            else:
                                def get(split,use,is_right):
                                    task=r['right'] if is_right else r['left'];action=ai if is_right else 0
                                    if static:
                                        y=data[split+'_labels'][use,action,TASKS.index(task)]
                                        if is_right:y=mapped_right(y,n,r)
                                        return np.eye(31)[y] if branch=='gold_onehot' else y[:,None]
                                    return features[task,cond,split][branch][use,action]
                                xf=get('validation',fit,False);yf=get('validation',fit,True);xt=get('analysis',test,False);yt=get('analysis',test,True)
                                result,rank=metric_values(xf,yf,xt,yt,kf,kt,ids,orbits)
                                if r['relation']=='reverse_max' and n==6 and branch in ('prefix_final','output_prob'):
                                    # Independent feature covariance formula with same fixed centers.
                                    a=metrics.center(xt,kt);b=metrics.center(yt,kt)
                                    den=np.linalg.norm(a.T@a)*np.linalg.norm(b.T@b)
                                    expected=float(np.square(a.T@b).sum()/den) if len(a)>=2 and den>1e-20 else None
                                    got=result['linear_cka'][0]
                                    assert (expected is None)==(got is None)
                                    error=abs(expected-got) if expected is not None else None
                                    if error is not None:assert error<=1e-8
                                    independent.append(dict(relation=r['relation'],n=n,control=control,branch=branch,condition=cond,role=role,
                                                            covariance_cka=expected,gram_cka=got,error=error))
                            for metric in names:values[metric][cond][role]=result[metric][0]
                            errors[cond][role]=result['procrustes'][1]
                    for metric in names:
                        row=dict(relation=r['relation'],left=r['left'],right=r['right'],correct_transform=r['correct'],wrong_transform=r['wrong'],
                            n=n,control=control,branch=branch,metric=metric,fit_count=int(fit.sum()),analysis_count=int(test.sum()),
                            procrustes_rank_cap=8,static_reference=static,**components(values[metric],static))
                        if metric=='procrustes':row.update({f'{cond}_{role}_error':errors[cond][role] for cond in errors for role in actions})
                        rows.append(row)
    assert len(rows)==720
    csv_write(DIAG/'geometry_per_n.csv',rows)
    fields=c['evaluation']['fields'];groups={}
    for row in rows:
        k=tuple(row[f] for f in ('relation','control','branch','metric'));groups.setdefault(k,[]).append(row)
    summaries=[]
    for key,rr in groups.items():
        assert sorted(r['n'] for r in rr)==[6,7,8]
        s=dict(zip(('relation','control','branch','metric'),key));s.update(n_count=3,worlds=1,task_models=4,static_reference=rr[0]['static_reference'])
        for field in fields:
            vals=[r[field] for r in rr];s[field]=math.fsum(vals)/3 if all(v is not None and math.isfinite(v) for v in vals) else None
        summaries.append(s)
    assert len(summaries)==240;csv_write(DIAG/'geometry_summary.csv',summaries)
    json_write(DIAG/'independent_metrics.json',dict(status='passed',records=independent,source_updates=0))
    return rows,summaries

def evaluate():
    c=verify();assert read(ROOT/'training_complete.json')['status']=='completed'
    data,_,audit=load_data(c,('train','validation','analysis'));device=device_setup();features={};index=[];learning=[];begun=time.monotonic()
    for job in c['training_manifest']:
        for cond in ('initial','trained'):
            for split in ('validation','analysis'):
                arrays,meta=cache_model(c,job,cond,split,data,device);features[job['task'],cond,split]=arrays
                index.append(dict(task=job['task'],condition=cond,split=split,path=str(ROOT/'evaluation_cache'/f"{job['task']}_{cond}_{split}.npz"),metadata=meta))
                for n in c['lengths']:
                    train=data['train_labels'][data['train_lengths']==n,0,TASKS.index(job['task'])]
                    mode=int(np.argmax(np.bincount(train,minlength=8)));use=data[split+'_lengths']==n
                    for ai,action in enumerate(ACTIONS):
                        y=data[split+'_labels'][use,ai,TASKS.index(job['task'])]
                        learning.append(dict(task=job['task'],condition=cond,split=split,n=n,action=action,samples=int(use.sum()),
                            answer_accuracy=float((arrays['prediction'][use,ai]==y).mean()),answer_ce=float(arrays['answer_ce'][use,ai].mean()),
                            training_mode=mode,mode_accuracy=float((y==mode).mean()),edge_mean=float(data[split+'_edges'][use,ai].mean()),
                            density_mean=float(data[split+'_edges'][use,ai].mean()/(n*(n-1))),EOS_evaluated=False))
    assert len(index)==16 and len(learning)==240
    # Genuine step0 states share all task-prefix hidden arrays, but outputs stay task-specific.
    for split in ('validation','analysis'):
        h=features[TASKS[0],'initial',split]['prefix_final']
        for task in TASKS:assert np.array_equal(h,features[task,'initial',split]['prefix_final'])
    json_write(ROOT/'evaluation_cache/index.json',dict(status='completed',protocol_sha256=sha(CONFIG),source_version_sha256=source_version(),data_sha256=audit['data_sha256'],
          records=index,actions=list(ACTIONS),logical_cache_count=16,actual_initial_identity_verified=True,initial_hidden_task_independent_bitwise=True,source_updates=0))
    from threadpoolctl import threadpool_limits
    csv_write(DIAG/'source_learning.csv',learning)
    with threadpool_limits(limits=2):rows,summaries=geometry(c,data,features)
    protected=check_protected()
    json_write(DIAG/'evaluation_complete.json',dict(status='completed',protocol_sha256=sha(CONFIG),source_version_sha256=source_version(),
         per_n_rows=len(rows),summary_rows=len(summaries),source_learning_rows=len(learning),cache_count=len(index),old_protected_files=protected,
         independent_metrics_status='passed',wall_seconds=time.monotonic()-begun,source_updates=0))
