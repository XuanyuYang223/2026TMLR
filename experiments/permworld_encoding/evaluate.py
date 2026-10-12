"""Fixed checkpoint evaluation and encoding-interaction decomposition."""
import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
import numpy as np
import torch
from scipy.stats import t as student_t
from .core import *
from experiments.longrun_attention import accelerate
from experiments.specialist_input_adjustment import adjust


def write_csv(path,rows):
    with Path(path).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def center(x,keys=None):
    x=np.asarray(x,np.float64).copy()
    if keys is None:return x-x.mean(0)
    _,inv=np.unique(keys,axis=0,return_inverse=True)
    for group in np.unique(inv):
        ids=inv==group;x[ids]-=x[ids].mean(0)
    return x


def alignment(k,l):
    den=np.linalg.norm(k)*np.linalg.norm(l)
    return float(np.sum(k*l)/den) if den>1e-20 else None


def u_center(k):
    n=len(k)
    if n<4:return None
    k=k.copy();np.fill_diagonal(k,0);s=k.sum(1)
    k-=s[:,None]/(n-2)+s[None,:]/(n-2);k+=s.sum()/((n-1)*(n-2));np.fill_diagonal(k,0);return k


def token_gram(ids,keys=None):
    """The pre-fixed categorical token-position kernel; no token numeric ordering."""
    k=np.zeros((len(ids),len(ids)),float)
    for pos in range(ids.shape[1]):k+=(ids[:,pos,None]==ids[None,:,pos])
    if keys is None:return k-k.mean(0,keepdims=True)-k.mean(1,keepdims=True)+k.mean()
    _,inv=np.unique(keys,axis=0,return_inverse=True)
    for group in np.unique(inv):
        use=inv==group;k[use]-=k[use].mean(0)
    for group in np.unique(inv):
        use=inv==group;k[:,use]-=k[:,use].mean(1,keepdims=True)
    return k


def components(values,correct,wrong):
    out={}
    for cond in ['trained','initial']:
        for role,a in [('correct',correct),('wrong',wrong),('identity',0)]:out[f'{cond}_{role}']=values[cond][a]
        a,b=values[cond][correct],values[cond][wrong];out[f'{cond}_preference']=a-b if a is not None and b is not None else None
    for role in ['correct','wrong','identity','preference']:
        a,b=out['trained_'+role],out['initial_'+role];out['change_'+role]=a-b if a is not None and b is not None else None
    out['delta']=out['change_preference'];return out


@torch.inference_mode()
def extract(model,data,encoding,task,device):
    model=accelerate(model).to(device).eval();features=[];probs=[];predictions=[];captures=[]
    hook=model.final_norm.register_forward_hook(lambda m,i,o:captures.append(o))
    ns=data['test_lengths'];inputs=data[f'test_{encoding}_input'];batchsize=128
    try:
        for start in range(0,len(ns),batchsize):
            sl=slice(start,start+batchsize);feat=[];prob=[];pred=[]
            for action in range(3):
                ids,mask,_,end,eq=batch(inputs[sl,action],ns[sl],task,device=device);captures.clear();logits=model(ids,mask)
                rows=torch.arange(len(ids),device=device);feat.append(captures[-1][rows,end].float().cpu().numpy())
                score=logits[rows,eq];prob.append(score[:,:31].softmax(-1).float().cpu().numpy());pred.append(score.argmax(-1).cpu().numpy())
            features.append(np.stack(feat,axis=1));probs.append(np.stack(prob,axis=1));predictions.append(np.stack(pred,axis=1))
    finally:hook.remove();model.to('cpu')
    return dict(prefix_final=np.concatenate(features),output_prob=np.concatenate(probs),prediction_full_vocab=np.concatenate(predictions))


def cache(root,name,model,data,encoding,task,device,signature):
    path=root/f'{name}.npz';meta=path.with_suffix('.json')
    if path.exists():
        m=json.loads(meta.read_text());assert m['signature']==signature and m['sha256']==sha(path)
        return dict(np.load(path))
    arrays=extract(model,data,encoding,task,device);np.savez_compressed(path,**arrays);json_write(meta,dict(signature=signature,sha256=sha(path)));return arrays


def evaluate_unit(config_path,unitid):
    config=verify_frozen(config_path);unit=next(u for u in config['units'] if u['unit']==unitid);root=WORKSPACE/config['output']/unitid
    protocol_sha=sha(config_path);audit=json.loads((root/'data_audit.json').read_text());assert sha(root/'data.npz')==audit['data_sha256']
    initial=torch.load(root/'step0.pt',map_location='cpu',weights_only=False);meta0=json.loads((root/'step0.json').read_text())
    assert sha(root/'step0.pt')==meta0['checkpoint_sha256'];assert initial['protocol_sha256']==protocol_sha
    data=dict(np.load(root/'data.npz'));ns=data['test_lengths'];features={};accuracies=[]
    cache_root=root/'evaluation_cache';cache_root.mkdir(exist_ok=True);device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    for job in [j for j in config['training_manifest'] if j['unit']==unitid]:
        enc,task=job['encoding'],job['task'];tdir=root/job['run_id'];finished=json.loads((tdir/'completed.json').read_text())
        assert finished['step']==config['training']['updates'] and finished['protocol_sha256']==protocol_sha
        assert finished['step0_parameter_sha256']==meta0['parameter_sha256'];assert sha(tdir/'checkpoint.pt')==finished['checkpoint_sha256']
        for cond in ['initial','trained']:
            cp=initial if cond=='initial' else torch.load(tdir/'checkpoint.pt',map_location='cpu',weights_only=False)
            model=make_model(config,unit['model_seed']);model.load_state_dict(cp['model'],strict=True)
            if cond=='initial':assert parameter_hash(model)==meta0['parameter_sha256']
            sig=dict(protocol_sha256=protocol_sha,data_sha256=audit['data_sha256'],weights_sha256=meta0['parameter_sha256'] if cond=='initial' else finished['checkpoint_sha256'],task=task,encoding=enc,condition=cond)
            a=cache(cache_root,f'{cond}_{enc}_{task}',model,data,enc,task,device,sig);features[cond,enc,task]=a
            for n in config['lengths']:
                use=ns==n
                for action in range(3):
                    y=data['test_labels'][use,action,TASKS.index(task)]
                    accuracies.append(dict(unit=unitid,encoding=enc,condition=cond,task=task,length=n,action=ACTIONS[action],samples=int(use.sum()),
                        full_vocab_accuracy=float(np.mean(a['prediction_full_vocab'][use,action]==y)),
                        numeric_conditional_accuracy=float(np.mean(a['output_prob'][use,action].argmax(-1)==y))))
            del cp,model
        print({'eval_checkpoint':job['run_id']},flush=True)
    # Shared saved weights imply identical task-free initial prefix states, not identical encoded geometry.
    for enc in ENCODINGS:
        base=features['initial',enc,TASKS[0]]['prefix_final']
        for task in TASKS:np.testing.assert_allclose(base,features['initial',enc,task]['prefix_final'],atol=1e-5,rtol=0)
    rows=[];baseline_rows=[];inputrows=[]
    for relation in RELATIONS:
        rel=relation['pair_id'];correct=ACTIONS.index(relation['input_transform']);wrong=1 if correct==2 else 2
        yleft=data['test_labels'][:,0,TASKS.index(relation['left'])];yright=data['test_labels'][:,:,TASKS.index(relation['right'])]
        keys=np.column_stack([ns,yleft,yright]);_,inv,counts=np.unique(keys,axis=0,return_inverse=True,return_counts=True);keep=counts[inv]>=3
        for n in config['lengths']:
            for control in ['raw_within_length','answer_strata']:
                use=(ns==n)&(keep if control=='answer_strata' else True);key=keys[use] if control=='answer_strata' else None
                m=int(use.sum());object_features=np.eye(n)[data['test_permutations'][use,:n]-1].reshape(m,n*n)
                z=center(object_features,key);g=z@z.T
                for enc in ENCODINGS:
                    ik=[token_gram(data[f'test_{enc}_input'][use,a,:2*n+4],key) for a in range(3)]
                    inputrows.append(dict(unit=unitid,relation=rel,encoding=enc,length=n,anchors=m,control=control,
                        correct_token_position_cka=alignment(ik[0],ik[correct]),wrong_token_position_cka=alignment(ik[0],ik[wrong]),identity_token_position_cka=alignment(ik[0],ik[0]),
                        object_permutation_matrix_correct_cka=alignment(g,g),object_permutation_matrix_wrong_cka=alignment(g,g)))
                    vals={metric:{} for metric in ['ordinary','debiased','input_adjusted']};ov={}
                    for cond in ['trained','initial']:
                        a=features[cond,enc,relation['left']];b=features[cond,enc,relation['right']]
                        x=center(a['prefix_final'][use,0],key);k=x@x.T
                        l=[]
                        for action in range(3):
                            y=center(b['prefix_final'][use,action],key);l.append(y@y.T)
                        vals['ordinary'][cond]=[alignment(k,j) for j in l]
                        uk=u_center(k);vals['debiased'][cond]=[alignment(uk,u_center(j)) if uk is not None else None for j in l]
                        if np.square(g).sum()>1e-20:
                            pk=adjust(k,g)[0];vals['input_adjusted'][cond]=[alignment(pk,adjust(j,g)[0]) for j in l]
                        else:vals['input_adjusted'][cond]=[None]*3
                        x=center(a['output_prob'][use,0],key);kk=x@x.T;ov[cond]=[]
                        for action in range(3):
                            y=center(b['output_prob'][use,action],key);ov[cond].append(alignment(kk,y@y.T))
                    for metric,v in vals.items():
                        rows.append(dict(unit=unitid,relation=rel,encoding=enc,length=n,anchors=m,control=control,metric=metric,**components(v,correct,wrong)))
                    baseline_rows.append(dict(unit=unitid,relation=rel,encoding=enc,length=n,anchors=m,control=control,baseline='output_prob',**components(ov,correct,wrong)))
                    for baseline in ['true_answer_onehot','true_answer_scalar']:
                        x=center(np.eye(31)[yleft[use]] if baseline.endswith('onehot') else yleft[use,None],key);k=x@x.T;vv=[]
                        for action in range(3):
                            y=center(np.eye(31)[yright[use,action]] if baseline.endswith('onehot') else yright[use,action,None],key);vv.append(alignment(k,y@y.T))
                        baseline_rows.append(dict(unit=unitid,relation=rel,encoding=enc,length=n,anchors=m,control=control,baseline=baseline,
                            **components({'trained':vv,'initial':[None]*3},correct,wrong)))
    write_csv(root/'evaluation_length.csv',rows);write_csv(root/'baseline_length.csv',baseline_rows);write_csv(root/'input_baseline_length.csv',inputrows);write_csv(root/'task_accuracy_length.csv',accuracies)
    json_write(root/'evaluation_complete.json',dict(protocol_sha256=protocol_sha,source_models=32,rows=len(rows),no_checkpoint_selection=True,
        interpretation='All paired encoding components, eight relations, fixed primary two inverse interactions; no learned-mechanism conclusion from CKA alone.'))


def summarize(config_path):
    config=verify_frozen(config_path);output=WORKSPACE/config['output'];groups=defaultdict(list)
    for unit in config['units']:
        root=output/unit['unit'];assert json.loads((root/'evaluation_complete.json').read_text())['protocol_sha256']==sha(config_path)
        with (root/'evaluation_length.csv').open() as f:
            for r in csv.DictReader(f):groups[r['unit'],r['relation'],r['encoding'],r['control'],r['metric']].append(r)
    units=[];index={}
    fields=['initial_preference','trained_preference','change_correct','change_wrong','change_identity','delta',
        'initial_correct','initial_wrong','initial_identity','trained_correct','trained_wrong','trained_identity']
    for key,rows in groups.items():
        assert sorted(int(r['length']) for r in rows)==config['lengths'];u,rel,enc,control,metric=key
        row=dict(unit=u,relation=rel,encoding=enc,control=control,metric=metric)
        for field in fields:row[field]=float(np.mean([float(r[field]) for r in rows])) if all(r[field]!='' for r in rows) else None
        units.append(row);index[key]=row
    interactions=[]
    for unit in config['units']:
        for rel in RELATIONS:
            for control in ['raw_within_length','answer_strata']:
                for metric in ['ordinary','debiased','input_adjusted']:
                    a=index[unit['unit'],rel['pair_id'],'one_line',control,metric];b=index[unit['unit'],rel['pair_id'],'canonical_cycle',control,metric]
                    row=dict(unit=unit['unit'],relation=rel['pair_id'],control=control,metric=metric)
                    for field,name in [('initial_preference','gamma_initial'),('trained_preference','gamma_trained'),('change_correct','gamma_correct'),('change_wrong','gamma_wrong'),('delta','gamma_delta')]:
                        row[name]=b[field]-a[field] if a[field] is not None and b[field] is not None else None
                    if row['gamma_delta'] is not None:
                        assert abs(row['gamma_delta']-(row['gamma_trained']-row['gamma_initial']))<1e-10
                        assert abs(row['gamma_delta']-(row['gamma_correct']-row['gamma_wrong']))<1e-10
                    interactions.append(row)
    stats=[]
    for rel in ['descent_inverse','exceedance_inverse']:
        rs=[r for r in interactions if r['relation']==rel and r['control']=='answer_strata' and r['metric']=='ordinary']
        for field in ['gamma_delta','gamma_initial','gamma_trained','gamma_correct','gamma_wrong']:
            complete=all(r[field] is not None for r in rs);values=np.array([r[field] for r in rs],float) if complete else None
            row=dict(relation=rel,contrast=field,primary_test=field=='gamma_delta',planned_units=len(config['units']),
                available_units=sum(r[field] is not None for r in rs),mean=None,sd=None,ci_low=None,ci_high=None,
                interval_scope='simultaneous95_for_two_primary_relations' if field=='gamma_delta' else 'pointwise95_descriptive',p_two_sided=None,bonferroni_p=None)
            if complete:
                mean=float(values.mean());sd=float(values.std(ddof=1));se=sd/math.sqrt(len(values));critical=float(student_t.ppf(.9875 if field=='gamma_delta' else .975,len(values)-1))
                p=float(2*student_t.sf(abs(mean/se),len(values)-1)) if se>0 else (1. if mean==0 else 0.)
                row.update(mean=mean,sd=sd,ci_low=mean-critical*se,ci_high=mean+critical*se,
                    p_two_sided=p if field=='gamma_delta' else None,bonferroni_p=min(1,2*p) if field=='gamma_delta' else None)
            stats.append(row)
    write_csv(output/'unit_components.csv',units);write_csv(output/'encoding_interactions.csv',interactions);write_csv(output/'primary_statistics.csv',stats)
    baseline_groups=defaultdict(list);accuracy_groups=defaultdict(list)
    for unit in config['units']:
        root=output/unit['unit']
        with (root/'baseline_length.csv').open() as f:
            for r in csv.DictReader(f):baseline_groups[r['unit'],r['relation'],r['encoding'],r['control'],r['baseline']].append(r)
        with (root/'task_accuracy_length.csv').open() as f:
            for r in csv.DictReader(f):accuracy_groups[r['unit'],r['encoding'],r['condition'],r['task'],r['action']].append(r)
    baseline_components=[];baseline_index={}
    for key,rs in baseline_groups.items():
        assert sorted(int(r['length']) for r in rs)==config['lengths'];u,rel,enc,control,baseline=key
        row=dict(unit=u,relation=rel,encoding=enc,control=control,baseline=baseline)
        for field in fields:row[field]=float(np.mean([float(r[field]) for r in rs])) if all(r[field]!='' for r in rs) else None
        baseline_components.append(row);baseline_index[key]=row
    baseline_interactions=[]
    for key,a in baseline_index.items():
        u,rel,enc,control,baseline=key
        if enc!='one_line':continue
        b=baseline_index[u,rel,'canonical_cycle',control,baseline];row=dict(unit=u,relation=rel,control=control,baseline=baseline)
        for field,name in [('initial_preference','gamma_initial'),('trained_preference','gamma_trained'),('change_correct','gamma_correct'),('change_wrong','gamma_wrong'),('delta','gamma_delta')]:
            row[name]=b[field]-a[field] if a[field] is not None and b[field] is not None else None
        baseline_interactions.append(row)
    acc=[]
    for (u,enc,cond,task,action),rs in accuracy_groups.items():
        assert sorted(int(r['length']) for r in rs)==config['lengths']
        acc.append(dict(unit=u,encoding=enc,condition=cond,task=task,action=action,
            full_vocab_accuracy=float(np.mean([float(r['full_vocab_accuracy']) for r in rs])),
            numeric_conditional_accuracy=float(np.mean([float(r['numeric_conditional_accuracy']) for r in rs]))))
    write_csv(output/'baseline_unit_components.csv',baseline_components);write_csv(output/'baseline_encoding_interactions.csv',baseline_interactions)
    write_csv(output/'task_accuracy_summary.csv',acc)
    json_write(output/'analysis_complete.json',dict(protocol_sha256=sha(config_path),planned_units=len(config['units']),
        missing_units_not_silently_dropped=True,primary_hypotheses=2,
        inference_scope='Independent jointly resampled source/data-world units conditional on these two fixed relations and the fixed mathematical domain; Bonferroni paired t procedure assumes adequate unit-effect distribution.'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--unit');p.add_argument('--summarize',action='store_true');a=p.parse_args()
    if a.summarize:summarize(a.config)
    elif a.unit:evaluate_unit(a.config,a.unit)
    else:p.error('Specify --unit or --summarize; no training is launched automatically.')
