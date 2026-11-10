"""Fixed, resumable existing-checkpoint analysis. Never creates an optimizer."""
from __future__ import annotations
import argparse
from collections import defaultdict
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from .common import *
from .cka import all_scores,dual_answer_basis,grouped_original_scores,residual_scores


def configure():
    if CONFIG.exists():return load_config()
    prior=read(REPETITION/'protocol.json');audit=read(REPETITION/'original_audit.json')
    assert (DIAG/'math_classification.json').exists()
    source=Path('/mnt/yangx-projects-f/20261009/permworld_new_artifacts/encoding_archdev')
    units=[dict(unit='archdev001',root=str(source),data_path=str(source/'worlds/archdev001/data.npz'),
                data_sha256=audit['large_source_file_sha256'][str(source/'worlds/archdev001/data.npz')],
                initializations={r['architecture']:dict(path=r['path'],sha256=r['sha256'],parameter_sha256=r['parameter_sha256']) for r in audit['saved_initializations']})]
    for u in prior['units']:
        base=Path(prior['output'])/u['unit'];a=read(base/'data_audit.json');initial={}
        for arch in prior['architectures']:
            path=base/'initializations'/arch/'step0.pt';m=read(path.with_suffix('.json'))
            initial[arch]=dict(path=str(path),sha256=m['checkpoint_sha256'],parameter_sha256=m['parameter_sha256'])
        units.append(dict(**u,root=str(base),data_path=str(base/'data.npz'),data_sha256=a['data_sha256'],initializations=initial))
    config=dict(version='permworld-dual-answer-checkpoint-diagnostic/v1',created_utc=datetime.now(timezone.utc).isoformat(),
        output=str(DEFAULT_OUTPUT),units=units,tasks=prior['tasks'],relations=prior['relations'],
        architectures=prior['architectures'],encodings=prior['encodings'],actions=prior['actions'],
        lengths=prior['lengths'],length_weights=prior['length_weights'],model=prior['model'],
        historical_activation_index='/mnt/yangx-projects-f/20261010/permworld_eight_relations/evaluation/index.json',
        phases=['initial','trained'],versions=['pooled_raw','within_length_debiased','within_length_dual_answer_debiased'],
        control_definition=dict(pooled_raw='Ordinary linear CKA pooled over all original lengths; NEW length sensitivity, not the original primary analysis.',
             within_length_debiased='Per-n U-centered linear Gram CKA; equal min count per length, original512 so no anchors removed; equal1/21 aggregate.',
             within_length_dual_answer_debiased='Within each n use shared intercept+onehot(A(x))+onehot(B(Tx)) design for both feature matrices; deduplicate identical answer arrays, SVD handles dependent columns; same-sample residuals then U-centered CKA.'),
        controls_attribution_bridge='Per-length ordinary CKA saved separately to distinguish length separation from bias correction; not an outcome-selected fourth primary version.',
        numerical=dict(dtype='float64',comparison_tolerance=1e-8,ordinary_denominator_floor=1e-20,residual_energy_relative_floor=1e-24,
                       residual_energy_absolute_floor=1e-20,OLS_rank='eps(float64)*max(N,p)*largest_singular_value'),
        alternative_exact_transforms={r['pair_id']:[] for r in prior['relations']},
        transform_advantage='S(designated T)-max(S(other eligible T)), including identity; not original Bt or original training Delta; undefined required controls propagate.',
        initialization='All five units have saved genuine step0. No seed-based reconstructed initialization will be substituted.',
        known_initial_identity_reference='Within architecture+encoding+unit, pre-task identity states are identical across savedinitial specialists. Their raw and shared-design residual identity CKA is1 when defined. Thus the new strongest-alternative contrast can start negative; its increase alone is not evidence that correct-transform CKA increased.',
        simplicity={'one_line/complement':'simple','canonical_cycle/inverse':'simple','one_line/inverse':'scrambling','canonical_cycle/complement':'scrambling','one_line/identity':'identity','canonical_cycle/identity':'identity'},
        bootstrap_seed=20261109,bootstrap_replicates=10000,bootstrap_confidence=.95,
        statistical_unit='8 prespecified disjoint task pairs; retain all architecture/encoding/unit entries within resampled pair; new4 separate from all5. Conditional catalog sensitivity, not a proof of population-level independence of related pairs.',
        decision_policy='No new practical-zero/equivalence bound. Report continuous effects, task-pair interval, seed ranges, original exact zero directions and deletions. CI crossingzero is not evidence of no learning. Residual linear answer control does not remove nonlinear answer/confidence information.',
        learning_phases=['trained'],learning_splits=['test'],numeric_classes=31,baseline_smoothing_alpha=.5,learning_prior_alpha=.5,
        learning_policy='Train-only per-length frequency prior withfixed.5smoothing over31validanswerIDs. Full188vocabulary NLL from read-only forward when needed; conditional31 NLL separately. Spearman uses conditionalnumericexpectation; no posthoc nearbaseline cutoff.',
        allowed_operations='Existing inputs/weights/activations and necessary forward/linear-algebra analysis only. No optimizer, backward, probe fit, new model, new unit, task budget or publication.',
        scientific_protocol_changed=False,prior_results_seen=True,exploratory=True,
        original_primary_definition='Per-length final_norm ordinary CKA with joint-answer strata>=3; NOT pooled. Original A uses one fixed wrong input. These are reproduced before new versions.',
        code_parent_git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
    files=list((ROOT/'experiments/permworld_answer_controls_20261109').glob('*.py'))
    files.extend([ROOT/'experiments/permworld_repetition_20261106/analysis.py',ROOT/'diagnostics/permworld_scope_inventory_20261009/metrics.py',
                  DIAG/'math_classification.json',DIAG/'formula_validation.json'])
    # Read-only forward uses the same model/batch constructor dependencies.
    oldfreeze=read(REPETITION/'freeze.json')
    hashes={p:sha(ROOT/p) for p in oldfreeze['source_sha256']}
    hashes.update({str(p.relative_to(ROOT)):sha(p) for p in files})
    write(CONFIG,config);write(DIAG/'freeze.json',dict(protocol_sha256=sha(CONFIG),source_sha256=hashes,registered_before_new_CKA=True))
    Path(config['output']).mkdir(parents=True,exist_ok=True);write(Path(config['output'])/'protocol.json',config)
    return config


def device():
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def number(x):
    try:v=float(x)
    except (ValueError,TypeError):return np.nan
    return v


def selections(data,task_index,relation,right,n):
    labels=data['test_labels'];use=data['test_lengths']==n
    key=np.column_stack([labels[use,0,task_index[relation['left']]],labels[use,:,task_index[right]]])
    _,inv,count=np.unique(key,axis=0,return_inverse=True,return_counts=True)
    return inv,count


def expected_tables(unit,c):
    if unit=='archdev001':
        ap=ROOT/'diagnostics/permworld_eight_relations_20261010/per_length_results.csv'
        bp=ROOT/'diagnostics/representation_pairing_review_20261011/pair_per_length.csv'
    else:
        base=Path(next(u['root'] for u in c['units'] if u['unit']==unit))/'analysis'
        ap=base/'comparison_a_per_length.csv';bp=base/'comparison_b_per_length.csv'
    a={(r['architecture'],r['encoding'],r['relation'],r['control'],int(r['length'])):r for r in csv_read(ap)
       if r['layer']=='final_norm' and r['branch']=='hidden' and r['metric']=='linear_cka'}
    b={(r['architecture'],r['encoding'],r['relation'],r['right_task'],r['control'],int(r['length'])):r for r in csv_read(bp)
       if r.get('domain','permworld')=='permworld'}
    return a,b


def reproduce(c,sources):
    """Recompute original raw/joint-answer primary CKA from all saved states."""
    root=Path(c['output'])/'reproduction';root.mkdir(parents=True,exist_ok=True)
    done=DIAG/'reproduction.json'
    if done.exists():assert read(done)['status']=='passed';return
    ix={t:i for i,t in enumerate(c['tasks'])};dev=device();records=[];worst=0.;total=0
    for u in c['units']:
        unit=u['unit'];data=sources.data(unit);aa,bb=expected_tables(unit,c)
        pairs=[];spec=[]
        for relation in c['relations']:
            left=ix[relation['left']];right=ix[relation['right']];action=c['actions'].index(relation['input_transform'])
            for task in c['tasks']:
                if task==relation['left']:continue
                pairs.append((left,ix[task],action));spec.append((relation,task,'B',None))
            for role,t in [('wrong',1 if action==2 else 2),('identity',0)]:
                pairs.append((left,right,t));spec.append((relation,relation['right'],'A',role))
        for arch in c['architectures']:
            for enc in c['encodings']:
                for phase in c['phases']:
                    path=root/f'{unit}_{arch}_{enc}_{phase}.json'
                    if path.exists():
                        r=read(path);assert r['status']=='passed';records.append(r);continue
                    h=sources.feature_stack(unit,arch,enc,phase);local=[];count=0;maximum=0.;accumulated=defaultdict(list)
                    for n in c['lengths']:
                        mask=data['test_lengths']==n;hh=torch.as_tensor(h[:,:,mask],dtype=torch.float64,device=dev)
                        raw=all_scores(hh,False).cpu().numpy()
                        groups=[selections(data,ix,rel,right,n) for rel,right,typ,role in spec]
                        joint=grouped_original_scores(hh,pairs,groups)
                        for control in ['raw_within_length','answer_strata']:
                            for i,((left,right,t),(rel,task,typ,role)) in enumerate(zip(pairs,spec)):
                                observed=raw[t,left,right] if control=='raw_within_length' else joint[i]
                                if typ=='B':expected=number(bb[arch,enc,rel['pair_id'],task,control,n]['fixed_transform_score_'+phase])
                                else:expected=number(aa[arch,enc,rel['pair_id'],control,n][phase+'_'+role])
                                ok=(np.isnan(expected) and np.isnan(observed)) or (np.isfinite(expected) and np.isfinite(observed) and abs(expected-observed)<=1e-8)
                                error=float(abs(expected-observed)) if np.isfinite(expected) and np.isfinite(observed) else None
                                if error is not None:maximum=max(maximum,error)
                                if not ok:
                                    failure=dict(status='failed',unit=unit,arch=arch,encoding=enc,phase=phase,n=n,control=control,source_task=c['tasks'][left],right_task=task,transform=c['actions'][t],expected=expected,observed=observed,tolerance=1e-8)
                                    write(DIAG/'reproduction_failure.json',failure);raise AssertionError(failure)
                                accumulated[rel['pair_id'],task,control,typ,role].append(float(observed))
                                count+=1
                        local.append(dict(length=n,comparisons=2*len(pairs)))
                        del hh
                    aggregate=[]
                    for (relation,right,control,typ,role),values in accumulated.items():
                        assert len(values)==21
                        aggregate.append(dict(relation=relation,right_task=right,control=control,type=typ,role=role,mean=float(np.mean(values)) if np.isfinite(values).all() else None))
                    r=dict(status='passed',unit=unit,architecture=arch,encoding=enc,phase=phase,comparisons=count,max_absolute_difference=maximum,per_length=local,aggregates=aggregate)
                    write(path,r);records.append(r);print(json.dumps(dict(event='original_reproduced',**r),default=str),flush=True)
                    total+=count;worst=max(worst,maximum);del h
    # Counts/ranks from the already independently audited summaries are also
    # checked against the reproduced original scoring population.
    audit=read(REPETITION/'final_saved_results_audit.json');assert audit['status']=='passed'
    count_checks=[]
    for u in c['units']:
        unit=u['unit'];lookup={}
        for r in records:
            if r['unit']!=unit:continue
            for a in r['aggregates']:
                lookup[r['architecture'],r['encoding'],r['phase'],a['relation'],a['right_task'],a['control'],a['type'],a['role']]=number(a['mean'])
        expected_path=(REPETITION/'historical_analysis/unit_counts.csv' if unit=='archdev001' else Path(u['root'])/'analysis/unit_counts.csv')
        references=csv_read(expected_path)
        for control in ['raw_within_length','answer_strata']:
            bt=[];delta=[];above=first=0
            for arch in c['architectures']:
                for enc in c['encodings']:
                    for rel in c['relations']:
                        vals={}
                        for phase in c['phases']:
                            correct=lookup[arch,enc,phase,rel['pair_id'],rel['right'],control,'B',None]
                            wrong=lookup[arch,enc,phase,rel['pair_id'],rel['right'],control,'A','wrong']
                            vals[phase]=correct-wrong
                            if phase=='trained':
                                others=[lookup[arch,enc,phase,rel['pair_id'],t,control,'B',None] for t in c['tasks'] if t not in [rel['left'],rel['right']]]
                                above+=int(correct>np.mean(others));first+=int(all(correct>v for v in others))
                        bt.append(vals['trained']);delta.append(vals['trained']-vals['initial'])
            actual=dict(Bt_positive=sum(v>0 for v in bt),Delta_positive=sum(v>0 for v in delta),above_other_mean=above,unique_first=first)
            ra=next(r for r in references if r['comparison']=='A' and r['control']==control and r['metric']=='linear_cka')
            rb=next(r for r in references if r['comparison']=='B' and r['control']==control and r['metric']=='linear_cka')
            expected={k:int((ra if k in ['Bt_positive','Delta_positive'] else rb)[k]) for k in actual}
            assert actual==expected,('Reproduced count/rank differs',unit,control,actual,expected)
            count_checks.append(dict(unit=unit,control=control,**actual,passed=True))
    result=dict(status='passed',records=records,total_comparisons=sum(r['comparisons'] for r in records),
                max_absolute_difference=max(r['max_absolute_difference'] for r in records),counts_and_ranks_reference_sha256=sha(REPETITION/'final_saved_results_audit.json'),
                reproduced_count_and_rank_checks=count_checks,scope='All five units, architectures, encodings, phases, original 21length ordinary hidden CKA underraw/joint-answer controls, all original15 candidate comparisons plus wrong/identity roles; counts/ranks recomputed from reproduced scores. No new metrics computed before passing.',old_reports_unchanged=True)
    write(done,result);write(root/'completion.json',result)


def all_pairs(c):return [(a,b,t) for a in range(len(c['tasks'])) for b in range(len(c['tasks'])) if a!=b for t in range(3)]


def bases_for(c,data,n,path):
    pairs=all_pairs(c);labels=data['test_labels'][data['test_lengths']==n]
    if path.exists():
        with np.load(path,allow_pickle=False) as z:
            packed=z['bases'];ranks=z['ranks'];diag=json.loads(str(z['metadata']))
            return [packed[i,:,:r] for i,r in enumerate(ranks)],diag
    bases=[];diag=[]
    with threadpool_limits(limits=2):
        for a,b,t in pairs:
            q,meta=dual_answer_basis(labels[:,0,a],labels[:,t,b]);bases.append(q);diag.append(meta)
    maximum=max(q.shape[1] for q in bases);packed=np.zeros((len(bases),len(labels),maximum),np.float64)
    for i,q in enumerate(bases):packed[i,:,:q.shape[1]]=q
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.stem+'.tmp.npz')
    np.savez_compressed(tmp,bases=packed,ranks=np.asarray([q.shape[1] for q in bases]),metadata=np.asarray(json.dumps(diag)));os.replace(tmp,path)
    return bases,diag


def calculate(c,sources):
    assert read(DIAG/'reproduction.json')['status']=='passed'
    root=Path(c['output']);pairs=all_pairs(c);dev=device();aggregates=[]
    for u in c['units']:
        unit=u['unit'];data=sources.data(unit);ns=data['test_lengths'];minimum=min(int((ns==n).sum()) for n in c['lengths'])
        # Original counts are equal, so the mandated minimum changes no inputs.
        assert minimum==512 and all(int((ns==n).sum())==minimum for n in c['lengths'])
        for arch in c['architectures']:
            for enc in c['encodings']:
                for phase in c['phases']:
                    label=f'{unit}_{arch}_{enc}_{phase}';out=root/'scores'/label;out.mkdir(parents=True,exist_ok=True)
                    if (out/'aggregate.csv').exists() and (out/'completion.json').exists():
                        receipt=read(out/'completion.json');assert receipt['protocol_sha256']==sha(CONFIG) and receipt['aggregate_sha256']==sha(out/'aggregate.csv')
                        aggregates.extend(csv_read(out/'aggregate.csv'));continue
                    h=sources.feature_stack(unit,arch,enc,phase);start=time.monotonic()
                    pooledpath=out/'pooled.npy'
                    if pooledpath.exists():pooled=np.load(pooledpath)
                    else:
                        hh=torch.as_tensor(h,dtype=torch.float64,device=dev);pooled=all_scores(hh,False).cpu().numpy();np.save(pooledpath,pooled);del hh;torch.cuda.empty_cache()
                    per_length=[];score_rows=[]
                    for n in c['lengths']:
                        path=out/f'n{n}.csv'
                        if path.exists():per_length.extend(csv_read(path));continue
                        mask=ns==n;hh=torch.as_tensor(h[:,:,mask],dtype=torch.float64,device=dev)
                        unbiased=all_scores(hh,True).cpu().numpy();ordinary=all_scores(hh,False).cpu().numpy()
                        bases,meta=bases_for(c,data,n,root/'answer_bases'/unit/f'n{n}.npz')
                        residual,energies=residual_scores(hh,pairs,bases,batch_size=16)
                        rows=[]
                        for i,(a,b,t) in enumerate(pairs):
                            common=dict(unit=unit,phase=phase,architecture=arch,encoding=enc,source_task=c['tasks'][a],candidate_task=c['tasks'][b],transform=c['actions'][t],length=n,samples=minimum)
                            rows.append(dict(**common,cka_version='within_length_debiased',cka=unbiased[t,a,b],ordinary_bridge=ordinary[t,a,b],undefined_reason='' if np.isfinite(unbiased[t,a,b]) else 'degenerate_U_centered_energy'))
                            rows.append(dict(**common,cka_version='within_length_dual_answer_debiased',cka=residual[i],**meta[i],left_residual_energy_fraction=energies[i][0],right_residual_energy_fraction=energies[i][1],undefined_reason='' if np.isfinite(residual[i]) else 'numerically_annihilated_or_degenerate_U_centered_residuals'))
                        csv_write(path,rows);per_length.extend(rows)
                        print(json.dumps(dict(event='new_scores',unit=unit,architecture=arch,encoding=enc,phase=phase,length=n,elapsed_seconds=time.monotonic()-start)),flush=True)
                        del hh,bases;torch.cuda.empty_cache()
                    grouped=defaultdict(list)
                    for row in per_length:grouped[row['source_task'],row['candidate_task'],row['transform'],row['cka_version']].append(row)
                    for i,(a,b,t) in enumerate(pairs):
                        common=dict(unit=unit,phase=phase,architecture=arch,encoding=enc,source_task=c['tasks'][a],candidate_task=c['tasks'][b],transform=c['actions'][t],samples=minimum*len(c['lengths']),length_count=len(c['lengths']))
                        score_rows.append(dict(**common,cka_version='pooled_raw',cka=pooled[t,a,b],finite_lengths=len(c['lengths']) if np.isfinite(pooled[t,a,b]) else 0,undefined_reason='' if np.isfinite(pooled[t,a,b]) else 'degenerate_pooled_energy'))
                        for version in c['versions'][1:]:
                            selected=grouped[c['tasks'][a],c['tasks'][b],c['actions'][t],version]
                            assert len(selected)==len(c['lengths'])
                            values=np.asarray([number(r['cka']) for r in selected]);value=float(values.mean()) if np.isfinite(values).all() else None
                            score_rows.append(dict(**common,cka_version=version,cka=value,finite_lengths=int(np.isfinite(values).sum()),undefined_reason='' if value is not None else 'undefined_length_preserved_no_reweight'))
                    csv_write(out/'aggregate.csv',score_rows);write(out/'completion.json',dict(protocol_sha256=sha(CONFIG),aggregate_sha256=sha(out/'aggregate.csv'),elapsed_seconds=time.monotonic()-start))
                    aggregates.extend(score_rows);del h
    csv_write(DIAG/'cka_results.csv',aggregates)
    write(DIAG/'cka_completion.json',dict(status='completed',rows=len(aggregates),expected_rows=5*4*2*16*15*3*3,CSV_sha256=sha(DIAG/'cka_results.csv')))
    return aggregates


def numeric_support_mass(logits, classes=31):
    """FP64 probability sum; clamp only the bounded reduction roundoff."""
    mass=logits.double().softmax(-1)[:,:classes].sum(-1)
    tolerance=4*logits.shape[-1]*torch.finfo(torch.float64).eps
    assert torch.isfinite(mass).all() and (mass>=0).all() and (mass<=1+tolerance).all()
    return mass.clamp(0,1)


def full_nll(c,sources):
    """Necessary identity-only forward; no feature fitting or optimizer."""
    from experiments.permworld_encoding_archdev_20261009.common import model,device_setup
    from experiments.permworld_encoding.core import batch,parameter_hash
    import torch.nn.functional as F
    dev=device_setup();records=[]
    for u in c['units']:
        unit=u['unit']
        if unit=='archdev001':continue # actual saved full CE is available
        sources.verify_file(u['data_path'],u['data_sha256'])
        with np.load(u['data_path'],allow_pickle=False) as z:
            ns=z['test_lengths'];inputs={enc:z[f'test_{enc}_input'][:,0].copy() for enc in c['encodings']};labels=z['test_labels'][:,0].copy()
        for arch in c['architectures']:
            for enc in c['encodings']:
                for column,task in enumerate(c['tasks']):
                    path=Path(c['output'])/'full_nll'/unit/f'{arch}_{enc}_{task}';done=path/'completion.json'
                    cpfile,digest=sources.checkpoint(unit,arch,enc,task);sources.verify_file(cpfile,digest)
                    if done.exists() and read(done).get('numeric_mass_computation')=='FP64_full_softmax_support_sum_v1':
                        meta=read(done);assert meta['checkpoint_sha256']==digest and meta['protocol_sha256']==sha(CONFIG)
                        for desc in meta['arrays'].values():sources.verify_file(desc['path'],desc['sha256'])
                        records.append(meta);continue
                    path.mkdir(parents=True,exist_ok=True);cp=torch.load(cpfile,map_location='cpu',weights_only=False)
                    net=model(c,arch,u['architecture_model_seeds'][arch],dev);net.load_state_dict(cp['model']);net.eval();before=parameter_hash(net)
                    assert before==read(cpfile.with_name('completed.json'))['parameter_sha256']
                    ce=[];mass=[];predictions=[];conditional=[]
                    with torch.inference_mode():
                        for start in range(0,len(ns),128):
                            sl=slice(start,start+128);ids,mask,_,_,eq=batch(inputs[enc][sl],ns[sl],task,device=dev)
                            z=net(ids,mask)[torch.arange(len(ids),device=dev),eq].float();truth=torch.as_tensor(labels[sl,column].astype(np.int64),device=dev)
                            ce.append(F.cross_entropy(z,truth,reduction='none').cpu().numpy());mass.append(numeric_support_mass(z).cpu().numpy())
                            predictions.append(z.argmax(-1).cpu().numpy());conditional.append(z[:,:31].softmax(-1).cpu().numpy())
                    assert parameter_hash(net)==before and sha(cpfile)==digest
                    saved_prediction=sources.get(unit,arch,enc,task,'trained','test','prediction')[:,0]
                    saved_probability=sources.get(unit,arch,enc,task,'trained','test','output_prob')[:,0]
                    prediction_mismatches=int((np.concatenate(predictions)!=saved_prediction).sum())
                    probability_error=float(np.max(np.abs(np.concatenate(conditional)-saved_probability)))
                    assert prediction_mismatches==0 and probability_error<=2e-5,('Readout reproduction failed',unit,arch,enc,task,prediction_mismatches,probability_error)
                    if (path/'answer_ce.npy').exists():
                        assert np.array_equal(np.concatenate(ce),np.load(path/'answer_ce.npy')),('Existing valid full CE changed',unit,arch,enc,task)
                    arrays={}
                    for name,val in [('answer_ce',np.concatenate(ce)),('numeric_mass',np.concatenate(mass))]:
                        np.save(path/(name+'.npy'),val);arrays[name]=dict(path=str(path/(name+'.npy')),sha256=sha(path/(name+'.npy')),shape=list(val.shape))
                    meta=dict(status='completed',unit=unit,architecture=arch,encoding=enc,task=task,checkpoint_sha256=digest,protocol_sha256=sha(CONFIG),data_sha256=u['data_sha256'],parameter_sha256_before=before,parameter_sha256_after=before,arrays=arrays,saved_prediction_mismatches=prediction_mismatches,saved_conditional_probability_max_error=probability_error,numeric_mass_computation='FP64_full_softmax_support_sum_v1',forward='FP32 eval identity only, query answer logit, full188token CE; FP64 support probability sum; no optimizer/backward')
                    write(done,meta);records.append(meta);print(json.dumps(dict(event='full_NLL',unit=unit,architecture=arch,encoding=enc,task=task)),flush=True)
                    del net,cp;torch.cuda.empty_cache()
    write(DIAG/'full_nll_forward.json',dict(status='completed',records=records,no_optimizer=True,no_backward=True))


def learning(c,sources):
    from .learning import evaluate_learning
    rows=[]
    for u in c['units']:
        unit=u['unit'];cc=dict(c,unit=unit)
        result=evaluate_learning(cc,sources.data(unit),sources.getter(unit),Path(c['output'])/'learning'/unit)
        p=Path(c['output'])/'learning'/unit/'learning_summary.csv';rows.extend(csv_read(p))
    csv_write(DIAG/'learning_summary.csv',rows)
    return rows


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['freeze','reproduce','calculate','full-nll','learning','summary','all']);a=p.parse_args()
    if a.stage=='freeze':configure();return
    c=load_config();sources=Sources(c)
    with threadpool_limits(limits=2):
        if a.stage in ('reproduce','all'):reproduce(c,sources)
        if a.stage in ('full-nll','all'):full_nll(c,sources)
        if a.stage in ('learning','all'):learning(c,sources)
        if a.stage in ('calculate','all'):calculate(c,sources)
        if a.stage in ('summary','all'):
            from .summary import summarize
            summarize(c,csv_read(DIAG/'cka_results.csv'),csv_read(DIAG/'learning_summary.csv'),DIAG/'summary')
    write(DIAG/'last_stage.json',dict(stage=a.stage,utc=datetime.now(timezone.utc).isoformat(),completed=True,no_training=True))


if __name__=='__main__':main()
