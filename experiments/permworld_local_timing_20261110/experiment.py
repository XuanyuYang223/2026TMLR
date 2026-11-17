"""The same lower-state edit, with and without current-token upper replay."""
from collections import defaultdict
import csv
import hashlib
from pathlib import Path
import time

import numpy as np
import torch

from .shared import ROOT,DIAG,OUT,PRIOR,PRIOR_OUT,TASKS,SEEDS,COMMON,AUX_LATE,sha,read,write
from .replay import replay_upper
from experiments.permworld_local_causal_20261110.shared import families,catalog,ids_for
from experiments.permworld_local_causal_20261110.formal import output_metrics,query_batched
from experiments.permworld_local_causal_20261110.probes import edit_state,predict
from experiments.permworld_local_causal_20261110.recurrent import load_model,parameter_hash

ORDER=[('late',c) for c in COMMON]+[('early',c) for c in COMMON]
ORDER += [('late',c) for c in AUX_LATE]+[('early','early_full_layer1'),('diagnostic','full_state_swap'),('diagnostic','actual_counterfactual')]


def bit_hash(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def load_probe(path):
    with np.load(path,allow_pickle=False) as z:
        return {role:{key:z[f'{role}_{key}'].copy() for key in ['weight','bias']} for role in ['edit','checker','count']},z['direction'].copy()


def execute(unit,protocol):
    name=unit['unit'];dest=DIAG/'results'/name;dest.mkdir(parents=True,exist_ok=True)
    done=dest/'completion.json'
    if done.exists():
        r=read(done);assert r['protocol_sha256']==sha(DIAG/'protocol.json')
        for path,digest in r['files'].items():assert sha(path)==digest
        print('TIMING REUSE',name,flush=True);return r
    started=time.perf_counter()
    audit_units={u['unit']:u for u in read(DIAG/'input_audit.json')['units']}
    aud=audit_units[name]
    selected=[f for f in families(name) if f['split']=='test']
    records={r['pair_id']:(f,r) for f in selected for r in f['pairs'] if r['kind'] in ('simple','count_matched')}
    case_folder=OUT/name;case_folder.mkdir(parents=True,exist_ok=True)
    output=case_folder/'pair_results.csv';temporary=case_folder/'pair_results.partial.csv'
    artifacts=[];task_reports=[];row_count=0
    stream=temporary.open('w',newline='');writer=None
    try:
        for task in TASKS:
            device='cuda' if torch.cuda.is_available() else 'cpu'
            model=load_model(unit['models'][task],device,protocol['model'])
            parameter_before=parameter_hash(model)
            probe_path=PRIOR_OUT/name/'probes'/task/'layer1_point1.npz'
            assert sha(probe_path)==aud['probes'][task]['sha256']
            params,probe_direction=load_probe(probe_path)
            nodes,lookup=catalog(selected,task,{'test'})
            old_state_folder=PRIOR_OUT/name/'states'/task/'test'
            cache_receipt=read(old_state_folder/'completion.json')
            assert cache_receipt['node_ids']==[r['node_id'] for r in nodes]
            state_path=old_state_folder/'states.npy'
            assert sha(state_path)==cache_receipt['files'][str(state_path)]
            states=np.load(state_path,mmap_mode='r',allow_pickle=False)
            by_id={r['node_id']:index for index,r in enumerate(nodes)}
            with np.load(old_state_folder/'native_logits.npz',allow_pickle=False) as z:
                native=z['native_logits'].copy()
            maxima=dict(noedit_state=0.,noedit_logits=0.,early_noedit_native_logits=0.,late_old_logits=0.,fullswap_native_logits=0.,
                        paired_h1=0.,paired_checker=0.,random_L1_norm=0.)
            noop_mode_mismatch=0;late_mode_mismatch=0;task_rows=0
            cells=sorted((PRIOR_OUT/name/'interventions'/task).glob('n*_i*.npz'),
                         key=lambda p:tuple(map(int,p.stem[1:].split('_i'))))
            expected_cells={(c['n'],c['i']):c for c in aud['cells'][task]}
            assert len(cells)==len(expected_cells)==357
            cell_receipts=[]
            for source_path in cells:
                n,i=tuple(map(int,source_path.stem[1:].split('_i')))
                source_digest=sha(source_path)
                expected_cell=expected_cells[n,i]
                assert source_digest==expected_cell['sha256']
                with np.load(source_path,allow_pickle=False) as z:
                    old={key:z[key].copy() for key in z.files}
                assert old['source_checkpoint_sha256'].item()==unit['models'][task]['checkpoint_sha256']
                assert old['data_sha256'].item()==unit['data_sha256']
                assert np.array_equal(old['direction'],probe_direction)
                pair_ids=list(map(str,old['pair_ids']))
                info=[records[pair_id] for pair_id in pair_ids]
                ri=np.array([by_id[str(k)] for k in old['recipient_node_ids']])
                di=np.array([by_id[str(k)] for k in old['donor_node_ids']])
                ci=np.array([by_id[str(k)] for k in old['actual_cf_node_ids']])
                hr=old['recipient_states'];hd=old['donor_states'];hs=old['same_label_states']
                previous=np.asarray(states[ri,0]).copy()
                assert np.array_equal(states[ri,1],hr) and np.array_equal(states[di,1],hd)
                cf_states=np.asarray(states[ci,1])
                B=len(ri);coordinate=old['coordinate'];direction=old['direction']
                lower={'unedited':hr[:,0].copy(),
                       'local_projection':(hr[:,0].astype(np.float64)+coordinate[:,None]*direction).astype(np.float32),
                       'same_label_projection':edit_state(hr[:,0],hs[:,0],direction,1.)[0]}
                for seed,q in zip(SEEDS,old['random_directions']):
                    lower[f'random_direction_{seed}']=(hr[:,0].astype(np.float64)+coordinate[:,None]*q).astype(np.float32)
                carried={}
                for condition in COMMON:
                    h=hr.copy();h[:,0]=lower[condition];carried['late',condition]=h
                replay_names=[*COMMON,'early_full_layer1']
                replay_input=np.concatenate([lower[c] for c in COMMON]+[hd[:,0]])
                prev_batch=np.concatenate([previous]*len(replay_names))
                early=replay_upper(model,torch.as_tensor(prev_batch,device=device),torch.as_tensor(replay_input,device=device)).cpu().numpy()
                early=early.reshape(len(replay_names),B,4,256)
                for k,condition in enumerate(replay_names):carried['early',condition]=early[k]
                for condition in COMMON:
                    assert np.array_equal(carried['late',condition][:,0],carried['early',condition][:,0]),'H1 timing identity'
                for j,condition in enumerate(AUX_LATE):
                    h=hr.copy();h[:,j]=hd[:,j];carried['late',condition]=h
                assert np.array_equal(carried['late','late_layer_swap_1'][:,0],carried['early','early_full_layer1'][:,0])
                carried['diagnostic','full_state_swap']=hd.copy()
                carried['diagnostic','actual_counterfactual']=cf_states.copy()
                state_error=float(np.max(np.abs(carried['early','unedited']-hr)))
                maxima['noedit_state']=max(maxima['noedit_state'],state_error)
                assert state_error<=2e-5,('Unedited current-token states',name,task,n,i,state_error)
                target_norm=np.linalg.norm(carried['late','local_projection'][:,0].astype(np.float64)-hr[:,0],axis=1)
                for timing in ('late','early'):
                    for seed in SEEDS:
                        norm=np.linalg.norm(carried[timing,f'random_direction_{seed}'][:,0].astype(np.float64)-hr[:,0],axis=1)
                        norm_error=float(np.max(np.abs(norm-target_norm)))
                        maxima['random_L1_norm']=max(maxima['random_L1_norm'],norm_error)
                        assert np.all(np.abs(norm-target_norm)<=1e-5+1e-5*target_norm)
                current_ids=ids_for([nodes[k]['permutation'] for k in ri],task);cut=4+2*i
                run_keys=[key for key in ORDER if key!=('diagnostic','actual_counterfactual')]
                stacked=np.concatenate([carried[key] for key in run_keys])
                suffix=np.concatenate([current_ids[:,cut+1:]]*len(run_keys))
                predictions=query_batched(model,suffix,stacked,cut+1).reshape(len(run_keys),B,188)
                logits={key:predictions[j] for j,key in enumerate(run_keys)}
                old_names=list(map(str,old['condition_names']))
                logits['diagnostic','actual_counterfactual']=old['logits'][old_names.index('actual_counterfactual')]
                for condition in COMMON:
                    err=float(np.max(np.abs(logits['late',condition]-old['logits'][old_names.index(condition)])))
                    maxima['late_old_logits']=max(maxima['late_old_logits'],err)
                    assert err<=1e-4,('Late replication',name,task,n,i,condition,err)
                    late_mode_mismatch+=int(np.count_nonzero(logits['late',condition].argmax(1)!=old['logits'][old_names.index(condition)].argmax(1)))
                noedit_error=float(np.max(np.abs(logits['early','unedited']-logits['late','unedited'])))
                maxima['noedit_logits']=max(maxima['noedit_logits'],noedit_error)
                assert noedit_error<=1e-4,('Unedited replay query',name,task,n,i,noedit_error)
                native_noedit_error=float(np.max(np.abs(logits['early','unedited']-native[ri])))
                maxima['early_noedit_native_logits']=max(maxima['early_noedit_native_logits'],native_noedit_error)
                assert native_noedit_error<=1e-4,('Unedited replay versus original native model',name,task,n,i,native_noedit_error)
                noop_mode_mismatch+=int(np.count_nonzero(logits['early','unedited'].argmax(1)!=logits['late','unedited'].argmax(1)))
                swap_error=float(np.max(np.abs(logits['diagnostic','full_state_swap']-native[di])))
                maxima['fullswap_native_logits']=max(maxima['fullswap_native_logits'],swap_error)
                assert swap_error<=1e-4,('Full-state donor wiring',name,task,n,i,swap_error)
                baseline=output_metrics(logits['late','unedited'])
                measures={key:output_metrics(logits[key]) for key in ORDER}
                checker_before=predict(params['checker'],hr[:,0]);edit_before=predict(params['edit'],hr[:,0])
                checker_donor=predict(params['checker'],hd[:,0]);edit_donor=predict(params['edit'],hd[:,0])
                checker_after={key:predict(params['checker'],carried[key][:,0]) for key in ORDER}
                edit_after={key:predict(params['edit'],carried[key][:,0]) for key in ORDER}
                count_before=predict(params['count'],hr[:,0])
                count_after={key:predict(params['count'],carried[key][:,0]) for key in ORDER}
                hashes={key:[bit_hash(row) for row in carried[key][:,0]] for key in ORDER}
                for condition in COMMON:
                    assert hashes['late',condition]==hashes['early',condition]
                    assert np.array_equal(checker_after['late',condition],checker_after['early',condition])
                cell_dir=OUT/name/'cells'/task;cell_dir.mkdir(parents=True,exist_ok=True)
                output_path=cell_dir/source_path.name
                np.savez_compressed(output_path,logits=np.stack([logits[key] for key in ORDER]),
                    timings=np.array([key[0] for key in ORDER]),conditions=np.array([key[1] for key in ORDER]),
                    pair_ids=old['pair_ids'],recipient_node_ids=old['recipient_node_ids'],donor_node_ids=old['donor_node_ids'],
                    actual_cf_node_ids=old['actual_cf_node_ids'],recipient_previous_states=previous,
                    injected_layer1_states=np.stack([lower[c] for c in COMMON]+[hd[:,0]]),injected_conditions=np.array(replay_names),
                    early_current_upper_states=early[:,:,1:],source_cell_sha256=np.asarray(source_digest),
                    source_checkpoint_sha256=old['source_checkpoint_sha256'],data_sha256=old['data_sha256'],
                    protocol_sha256=np.asarray(sha(DIAG/'protocol.json')))
                artifacts.append(output_path)
                cell_receipts.append(dict(path=str(output_path),sha256=sha(output_path),source_cell=str(source_path),n=n,i=i,cases=B))
                # Case-major rows allow bounded independent and summary joins.
                for j,(f,r) in enumerate(info):
                    total=r['recipient_total_count'];desired=total+r['direction']
                    for key in ORDER:
                        timing,condition=key;h=carried[key];m=measures[key]
                        delta=h[j].astype(np.float64)-hr[j]
                        norm=float(np.linalg.norm(delta[0]));full_norm=float(np.linalg.norm(delta))
                        plan=float(abs(coordinate[j])) if condition=='local_projection' or condition.startswith('random_direction_') else norm
                        oracle=total if condition in ('unedited','same_label_projection') else r['donor_total_count'] if condition=='full_state_swap' else desired if condition=='actual_counterfactual' else ''
                        intervention_layer=int(condition.rsplit('_',1)[1]) if condition in AUX_LATE else 1 if timing!='diagnostic' else 0
                        payload_dimension=1 if condition in COMMON and condition!='unedited' else 256 if condition in AUX_LATE or condition=='early_full_layer1' else 1024 if condition=='full_state_swap' else 0
                        row=dict(unit=name,task=task,group='main' if r['kind']=='simple' else 'matched',sign=r['direction'],
                            pair_id=r['pair_id'],family_id=r['family_id'],component_id=f['component_id'],timing=timing,condition=condition,
                            condition_available=True,missing_reason='',length=n,position=i,layer=intervention_layer,
                            probe_layer=1,payload_dimension=payload_dimension,dose=1.,cut_token_index=cut,
                            recipient_node_id=nodes[ri[j]]['node_id'],pair_donor_id=nodes[di[j]]['node_id'],
                            first_layer_state_sha256=hashes[key][j],
                            recipient_local_label=r['recipient_label'],desired_local_label=r['donor_label'],probe_boundary=.5,
                            recipient_local_score=float(checker_before[j]),edited_local_score=float(checker_after[key][j]),
                            checker_donor_score=float(checker_donor[j]),edit_probe_recipient_score=float(edit_before[j]),
                            edit_probe_edited_score=float(edit_after[key][j]),edit_probe_donor_score=float(edit_donor[j]),
                            recipient_prefix_count_prediction=float(count_before[j]),edited_prefix_count_prediction=float(count_after[key][j]),
                            recipient_resolved_count=r['recipient_prefix_count'],donor_resolved_count=r['donor_prefix_count'],
                            recipient_true_count=total,desired_true_count=desired,donor_full_true_count=r['donor_total_count'],condition_oracle_count=oracle,
                            recipient_argmax_count=int(baseline['argmax_count'][j]),edited_argmax_count=int(m['argmax_count'][j]),
                            recipient_argmax_token=int(baseline['argmax_token'][j]),edited_argmax_token=int(m['argmax_token'][j]),
                            recipient_numeric_expectation=float(baseline['expectation'][j]),edited_numeric_expectation=float(m['expectation'][j]),
                            recipient_numeric_mass=float(baseline['numeric_mass'][j]),edited_numeric_mass=float(m['numeric_mass'][j]),
                            planned_edit_l2=plan,layer1_edit_l2=norm,edit_l2=norm,full_state_edit_l2=full_norm,carried_state_l2norm=full_norm,
                            recipient_correct=int(baseline['argmax_count'][j])==total,
                            source_cell_sha256=source_digest,earlier_center=r['earlier_center'] or '')
                        if writer is None:writer=csv.DictWriter(stream,fieldnames=list(row));writer.writeheader()
                        writer.writerow(row);row_count+=1;task_rows+=1
                if i==n-2:print(f'TIMING {name} {task} n={n}',flush=True)
            after_hash=parameter_hash(model)
            assert after_hash==parameter_before==unit['models'][task]['parameter_sha256']
            task_reports.append(dict(task=task,rows=task_rows,maxima=maxima,early_noop_argmax_mismatches=noop_mode_mismatch,
                rerun_late_old_argmax_mismatches=late_mode_mismatch,parameter_sha256_before=parameter_before,parameter_sha256_after=after_hash,
                identical_layer1_bytes_all_common_cases=True,no_probe_fitting=True,cells=cell_receipts))
            del model,states,native
            if torch.cuda.is_available():torch.cuda.empty_cache()
    finally:stream.close()
    temporary.replace(output)
    alias=dest/'pair_results.csv'
    if not alias.exists():alias.symlink_to(output)
    assert alias.resolve()==output
    receipt=dict(status='completed',unit=name,historical_reference=unit['historical'],rows=row_count,
        protocol_sha256=sha(DIAG/'protocol.json'),files={str(output):sha(output)},task_reports=task_reports,
        elapsed_seconds=time.perf_counter()-started,no_probe_fitting=True,no_new_data=True,no_base_training=True,
        no_DAS=True,no_cross_model=True,no_new_generalization_task=True)
    write(done,receipt)
    print(dict(unit=name,status='completed',rows=row_count,elapsed_seconds=receipt['elapsed_seconds']),flush=True)
    return receipt
