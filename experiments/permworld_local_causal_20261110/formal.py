"""Fixed within-model recurrent interventions; all conditions retain one endpoint."""
from collections import defaultdict
import csv
from pathlib import Path
import time

import numpy as np
import torch

from .shared import DIAG,OUT,TASKS,RANDOM_SEEDS,CONDITIONS,sha,read,write,families,ids_for,transformed
from .features import collect
from .probes import fit_bundle,predict,edit_state
from .recurrent import load_model,resume,parameter_hash

RESUME_CONDITIONS=['local_projection','same_label_projection',*[f'random_direction_{s}' for s in RANDOM_SEEDS],'full_state_swap']


def output_metrics(logits):
    z=torch.as_tensor(logits,dtype=torch.float64)
    ids=z.argmax(1).cpu().numpy()
    mass=z.softmax(1)[:,:31].sum(1).cpu().numpy()
    distribution=z[:,:31].softmax(1)
    expectation=(distribution*torch.arange(31,device=z.device,dtype=torch.float64)).sum(1).cpu().numpy()
    return dict(argmax_token=ids,argmax_count=np.where(ids<=30,ids,-1),numeric_mass=mass,expectation=expectation)


def query_batched(model,ids,states,start,limit=512):
    pieces=[]
    device=next(model.parameters()).device
    with torch.inference_mode():
        for low in range(0,len(ids),limit):
            high=min(low+limit,len(ids))
            result=resume(model,torch.as_tensor(ids[low:high],device=device),start,
                          torch.as_tensor(states[low:high],device=device))
            pieces.append(result.logits[:,-1].cpu().numpy())
    return np.concatenate(pieces)


def execute(unit,protocol):
    unit_name=unit['unit'];dest=DIAG/'results'/unit_name;dest.mkdir(parents=True,exist_ok=True)
    done=dest/'completion.json'
    if done.exists():
        r=read(done);assert r['protocol_sha256']==sha(DIAG/'protocol.json')
        for path,digest in r['files'].items():assert sha(path)==digest
        print('FORMAL REUSE',unit_name,flush=True);return r
    started=time.perf_counter()
    chosen_layer=int(protocol['selected_layer']);point=int(protocol['point']);dose=float(protocol['dose'])
    assert point==1 and dose==1 and 0<=chosen_layer<=3
    all_families=families(unit_name)
    selected=[f for f in all_families if f['split']=='test']
    output=OUT/unit_name/'results';output.mkdir(parents=True,exist_ok=True)
    temporary=output/'pair_results.partial.csv';final=output/'pair_results.csv'
    stream=temporary.open('w',newline='');writer=None;row_count=0;task_audits=[]
    probability_artifacts=[]
    try:
        for task_index,task in enumerate(TASKS):
            device='cuda' if torch.cuda.is_available() else 'cpu'
            model=load_model(unit['models'][task],device,protocol['model'])
            parameter_before=parameter_hash(model)
            fit_states,fit_nodes,_,fit_receipt=collect(unit_name,task,model,('fit','dev'))
            bundle=fit_bundle(fit_states,fit_nodes,chosen_layer,point,unit_name,task)
            state,nodes,lookup,state_receipt=collect(unit_name,task,model,('test',))
            with np.load(OUT/unit_name/'states'/task/'test/native_logits.npz',allow_pickle=False) as z:
                native=z['native_logits'].copy();noop=z['noop_logits'].copy()
            native_metrics=output_metrics(native);base_metrics=output_metrics(noop)
            direction=bundle['direction']
            unit_index=0 if unit['historical'] else int(unit_name[-2:])
            random={seed:np.random.default_rng(np.random.SeedSequence([seed,unit_index,task_index])).normal(size=256) for seed in RANDOM_SEEDS}
            random={seed:value/np.linalg.norm(value) for seed,value in random.items()}
            grouping=defaultdict(list)
            for f in selected:grouping[f['n'],f['center']].append(f)
            max_swap_error=0.;max_same_state_error=0.;max_random_norm_error=0.;max_edit_donor_error=0.
            task_rows=0;prob_dir=OUT/unit_name/'interventions'/task;prob_dir.mkdir(parents=True,exist_ok=True)
            for (n,i),family_group in sorted(grouping.items()):
                cases=[]
                for f in family_group:
                    controls={r['control_for_pair_id']:r for r in f['pairs'] if r['kind']=='same_label'}
                    for r in f['pairs']:
                        if r['kind'] not in ('simple','count_matched'):continue
                        same=controls.get(r['pair_id'])
                        find=lambda values:lookup[n,i,transformed(values,task)]
                        cases.append(dict(family=f,record=r,same=same,recipient=find(r['recipient']),donor=find(r['donor']),
                                          actual_cf=find(r['actual_local_counterfactual']),
                                          same_index=find(same['donor']) if same else None))
                ri=np.array([c['recipient'] for c in cases]);di=np.array([c['donor'] for c in cases]);ci=np.array([c['actual_cf'] for c in cases])
                si=np.array([c['same_index'] if c['same_index'] is not None else c['recipient'] for c in cases])
                hr=np.asarray(state[ri,point]).copy();hd=np.asarray(state[di,point]);hs=np.asarray(state[si,point]);hc=np.asarray(state[ci,point])
                edited,coordinate=edit_state(hr[:,chosen_layer],hd[:,chosen_layer],direction,dose)
                same_edited,same_coordinate=edit_state(hr[:,chosen_layer],hs[:,chosen_layer],direction,dose)
                planned=np.abs(coordinate)*dose
                carried={}
                for condition in RESUME_CONDITIONS:
                    h=hr.copy()
                    if condition=='local_projection':h[:,chosen_layer]=edited
                    elif condition=='same_label_projection':h[:,chosen_layer]=same_edited
                    elif condition=='full_state_swap':h=hd.copy()
                    else:
                        seed=int(condition.rsplit('_',1)[1]);q=random[seed]
                        h[:,chosen_layer]=(hr[:,chosen_layer].astype(np.float64)+dose*coordinate[:,None]*q).astype(np.float32)
                    carried[condition]=h
                target_actual=np.linalg.norm(carried['local_projection'][:,chosen_layer].astype(np.float64)-hr[:,chosen_layer],axis=1)
                for seed in RANDOM_SEEDS:
                    condition=f'random_direction_{seed}'
                    norm=np.linalg.norm(carried[condition][:,chosen_layer].astype(np.float64)-hr[:,chosen_layer],axis=1)
                    error=np.max(np.abs(norm-target_actual));max_random_norm_error=max(max_random_norm_error,float(error))
                    assert np.all(np.abs(norm-target_actual)<=1e-5+1e-5*target_actual)
                ids=ids_for([nodes[k]['permutation'] for k in ri],task);cut=4+2*i
                stacked=np.concatenate([carried[c] for c in RESUME_CONDITIONS])
                suffix=np.concatenate([ids[:,cut+1:]]*len(RESUME_CONDITIONS))
                resumed=query_batched(model,suffix,stacked,cut+1).reshape(len(RESUME_CONDITIONS),len(cases),188)
                logits={c:resumed[j] for j,c in enumerate(RESUME_CONDITIONS)}
                logits['unedited']=noop[ri];logits['actual_counterfactual']=native[ci]
                swap_error=float(np.max(np.abs(logits['full_state_swap']-native[di])))
                max_swap_error=max(max_swap_error,swap_error)
                assert swap_error<=1e-4,('Full-state donor wiring',unit_name,task,n,i,swap_error)
                checker_before=predict(bundle['checker'],hr[:,chosen_layer]);edit_before=predict(bundle['edit'],hr[:,chosen_layer])
                checker_donor=predict(bundle['checker'],hd[:,chosen_layer]);edit_donor=predict(bundle['edit'],hd[:,chosen_layer])
                prefix_before=predict(bundle['count'],hr[:,chosen_layer])
                order=list(CONDITIONS)
                packed_logits=np.stack([logits[c] for c in order])
                score_file=prob_dir/f'n{n}_i{i}.npz'
                np.savez_compressed(score_file,logits=packed_logits,pair_ids=np.array([c['record']['pair_id'] for c in cases]),
                    condition_names=np.array(order),recipient_node_ids=np.array([nodes[k]['node_id'] for k in ri]),
                    donor_node_ids=np.array([nodes[k]['node_id'] for k in di]),actual_cf_node_ids=np.array([nodes[k]['node_id'] for k in ci]),
                    coordinate=coordinate,random_directions=np.stack([random[s] for s in RANDOM_SEEDS]),
                    direction=direction,recipient_states=hr,donor_states=hd,same_label_states=hs,
                    source_checkpoint_sha256=np.asarray(unit['models'][task]['checkpoint_sha256']),
                    data_sha256=np.asarray(unit['data_sha256']),protocol_sha256=np.asarray(sha(DIAG/'protocol.json')))
                probability_artifacts.append(score_file)
                for condition in CONDITIONS:
                    h=hr if condition=='unedited' else hc if condition=='actual_counterfactual' else carried[condition]
                    after=output_metrics(logits[condition])
                    check_after=predict(bundle['checker'],h[:,chosen_layer]);edit_after=predict(bundle['edit'],h[:,chosen_layer]);prefix_after=predict(bundle['count'],h[:,chosen_layer])
                    if condition=='local_projection':
                        max_edit_donor_error=max(max_edit_donor_error,float(np.max(np.abs(edit_after-edit_donor))))
                    for j,case in enumerate(cases):
                        r=case['record'];same=case['same'];recipient=case['recipient'];donor=case['donor'];cf=case['actual_cf']
                        total=r['recipient_total_count'];desired=total+r['direction']
                        available=condition!='same_label_projection' or same is not None
                        if condition=='full_state_swap':oracle_count=r['donor_total_count'];intervention_donor=donor
                        elif condition=='actual_counterfactual':oracle_count=desired;intervention_donor=cf
                        elif condition=='same_label_projection':oracle_count=total;intervention_donor=case['same_index']
                        elif condition=='unedited':oracle_count=total;intervention_donor=recipient
                        else:oracle_count=desired;intervention_donor=donor if condition=='local_projection' else None
                        actual_delta=h[j].astype(np.float64)-hr[j]
                        actual_l2=float(np.linalg.norm(actual_delta))
                        plan=float(planned[j]) if condition=='local_projection' or condition.startswith('random_direction_') else float(abs(same_coordinate[j])*dose) if condition=='same_label_projection' else actual_l2
                        row=dict(unit=unit_name,task=task,group='main' if r['kind']=='simple' else 'matched',sign=r['direction'],
                            pair_id=r['pair_id'],family_id=r['family_id'],component_id=case['family']['component_id'],condition=condition,
                            condition_available=available,missing_reason='' if available else 'No valid same-label donor under bounded frozen search',
                            length=n,position=i,layer=chosen_layer+1,timing='after',dose=dose,
                            cut_token_index=cut,recipient_node_id=nodes[recipient]['node_id'],
                            donor_id=nodes[donor]['node_id'],pair_donor_node_id=nodes[donor]['node_id'],
                            donor_fields_scope='fixed_label_changing_pair_donor',
                            intervention_donor_id=nodes[intervention_donor]['node_id'] if intervention_donor is not None else '',
                            recipient_local_label=r['recipient_label'],desired_local_label=r['donor_label'],
                            intervention_donor_local_label=nodes[intervention_donor]['label'] if intervention_donor is not None else -1,
                            intervention_donor_resolved_count=nodes[intervention_donor]['resolved_count'] if intervention_donor is not None else '',
                            intervention_donor_full_true_count=nodes[intervention_donor]['total_count'] if intervention_donor is not None else '',
                            probe_boundary=.5,recipient_local_score=float(checker_before[j]),edited_local_score=float(check_after[j]) if available else np.nan,
                            checker_donor_score=float(checker_donor[j]),edit_probe_recipient_score=float(edit_before[j]),
                            edit_probe_edited_score=float(edit_after[j]) if available else np.nan,edit_probe_donor_score=float(edit_donor[j]),
                            recipient_prefix_count_prediction=float(prefix_before[j]),edited_prefix_count_prediction=float(prefix_after[j]) if available else np.nan,
                            recipient_resolved_count=r['recipient_prefix_count'],donor_resolved_count=r['donor_prefix_count'],
                            recipient_true_count=total,desired_true_count=desired,donor_full_true_count=r['donor_total_count'],condition_oracle_count=oracle_count,
                            recipient_argmax_count=int(base_metrics['argmax_count'][recipient]),edited_argmax_count=int(after['argmax_count'][j]) if available else -1,
                            recipient_argmax_token=int(base_metrics['argmax_token'][recipient]),edited_argmax_token=int(after['argmax_token'][j]) if available else -1,
                            recipient_numeric_expectation=float(base_metrics['expectation'][recipient]),edited_numeric_expectation=float(after['expectation'][j]) if available else np.nan,
                            recipient_numeric_mass=float(base_metrics['numeric_mass'][recipient]),edited_numeric_mass=float(after['numeric_mass'][j]) if available else np.nan,
                            planned_edit_l2=plan if available else np.nan,edit_l2=actual_l2 if available else np.nan,
                            recipient_correct=int(base_metrics['argmax_count'][recipient])==total,
                            same_label_variant=same.get('same_label_variant','') if same else '',earlier_center=r['earlier_center'] or '')
                        if writer is None:writer=csv.DictWriter(stream,fieldnames=list(row));writer.writeheader()
                        writer.writerow(row);row_count+=1;task_rows+=1
                if i==n-2:print(f'INTERVENTIONS {unit_name} {task} n={n}',flush=True)
            assert parameter_hash(model)==parameter_before==unit['models'][task]['parameter_sha256']
            task_audits.append(dict(task=task,rows=task_rows,selected_layer=chosen_layer+1,point=point,dose=dose,
                prefix_feature_receipt=state_receipt,probe_file=bundle['path'],probe_sha256=bundle['sha256'],
                maximum_fullswap_vs_native_donor_logit_difference=max_swap_error,
                maximum_actual_random_vs_treatment_norm_difference=max_random_norm_error,
                maximum_defining_probe_donor_score_difference=max_edit_donor_error,
                parameter_sha256_before=parameter_before,parameter_sha256_after=parameter_hash(model),
                all_conditions_use_actual_recurrent_state=True))
            del model,fit_states,state,native,noop
            if torch.cuda.is_available():torch.cuda.empty_cache()
    finally:stream.close()
    temporary.replace(final)
    link=dest/'pair_results.csv'
    if not link.exists():link.symlink_to(final)
    assert link.resolve()==final
    receipt=dict(status='completed',unit=unit_name,historical_reference=unit['historical'],
        protocol_sha256=sha(DIAG/'protocol.json'),rows=row_count,selected_layer=chosen_layer+1,point=point,dose=dose,
        task_audits=task_audits,files={str(final):sha(final)},
        intervention_files=[dict(path=str(p),sha256=sha(p),size_bytes=p.stat().st_size) for p in probability_artifacts],
        elapsed_seconds=time.perf_counter()-started,no_optimizer=True,no_backward=True,no_base_training=True,
        no_cross_model_exchange=True,no_new_combination_task=True)
    write(done,receipt)
    print(dict(unit=unit_name,status='completed',rows=row_count,elapsed_seconds=receipt['elapsed_seconds']),flush=True)
    return receipt
