"""Fixed two-relation geometry with independent fit/test inputs."""
import csv
import json
from pathlib import Path
import numpy as np
import torch

from .common import *
from .training import load_data, source_version


def write_csv(path, rows):
    with Path(path).open('w', newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


@torch.inference_mode()
def extract(net,data,encoding,task,device,split):
    values=[];outputs=[];predictions=[];captures=[]
    net.eval();hook=net.final_norm.register_forward_hook(lambda m,i,o:captures.append(o))
    ns=data[split+'_lengths'];prefix=data[f'{split}_{encoding}_input']
    try:
        for start in range(0,len(ns),128):
            sl=slice(start,start+128);features=[];probs=[];pred=[]
            for action in range(3):
                ids,mask,_,end,eq=batch(prefix[sl,action],ns[sl],task,device=device)
                captures.clear();logits=net(ids,mask);rows=torch.arange(len(ids),device=device)
                features.append(captures[-1][rows,end].float().cpu().numpy())
                score=logits[rows,eq];probs.append(score[:,:31].softmax(-1).float().cpu().numpy())
                pred.append(score.argmax(-1).cpu().numpy())
            values.append(np.stack(features,1));outputs.append(np.stack(probs,1));predictions.append(np.stack(pred,1))
    finally:hook.remove()
    return dict(prefix_final=np.concatenate(values),output_prob=np.concatenate(outputs),prediction=np.concatenate(predictions))


def cache(path,net,data,encoding,task,device,split,signature):
    if path.exists():
        metadata=json.loads(path.with_suffix('.json').read_text())
        assert metadata['signature']==signature and metadata['sha256']==sha(path)
        return dict(np.load(path))
    arrays=extract(net,data,encoding,task,device,split)
    np.savez_compressed(path,**arrays)
    json_write(path.with_suffix('.json'),dict(signature=signature,sha256=sha(path)))
    return arrays


def components(values):
    result={}
    for condition in ['initial','trained']:
        for role in ['correct','wrong','identity']:
            result[condition+'_'+role]=values[condition][role]
        a=values[condition]['correct'];b=values[condition]['wrong']
        result['B0' if condition=='initial' else 'Bt']=a-b if a is not None and b is not None else None
    for role in ['correct','wrong','identity']:
        a=result['trained_'+role];b=result['initial_'+role]
        result['change_'+role]=a-b if a is not None and b is not None else None
    a=result['Bt'];b=result['B0'];result['Delta']=a-b if a is not None and b is not None else None
    if result['Delta'] is not None:
        assert abs(result['Delta']-(result['change_correct']-result['change_wrong']))<1e-10
    return result


def selections(data,relation,split,n,control):
    ns=data[split+'_lengths'];labels=data[split+'_labels']
    yl=labels[:,0,TASKS.index(relation['left'])]
    yr=labels[:,:,TASKS.index(relation['right'])]
    keys=np.column_stack([ns,yl,yr]);use=ns==n
    if control=='answer_strata':
        _,inv,count=np.unique(keys,axis=0,return_inverse=True,return_counts=True)
        use &= count[inv]>=3
    return use,keys[use] if control=='answer_strata' else None


def metric_values(xfit,yfit,xtest,ytest,keysfit,keystest,ids,duplicate_keys):
    xfit=metrics.center(xfit,keysfit);yfit=metrics.center(yfit,keysfit)
    xtest=metrics.center(xtest,keystest);ytest=metrics.center(ytest,keystest)
    values,_=metrics.scores(xfit,yfit,xtest,ytest,ids=ids,duplicate_keys=duplicate_keys)
    return values


def manual_final(net, ids, mask, architecture):
    """Explicit hidden forward without the activation hook or output head."""
    positions=torch.arange(ids.shape[1],device=ids.device)
    hidden=net.token_embedding(ids)+net.position_embedding(positions)[None]
    hidden=net.embedding_dropout(hidden)*mask[...,None]
    if architecture=='transformer':
        for block in net.blocks:hidden=block(hidden,mask)
    else:
        for i,recurrent in enumerate(net.recurrent_layers):
            hidden,_=recurrent(hidden)
            if i<len(net.interlayer_dropout):hidden=net.interlayer_dropout[i](hidden)
    return net.final_norm(hidden)*mask[...,None]


def independent_center(x,keys):
    x=np.asarray(x,np.float64)
    if keys is None:return x-x.mean(axis=0,keepdims=True)
    _,group=np.unique(keys,axis=0,return_inverse=True)
    design=np.eye(int(group.max())+1)[group]
    return x-design@((design.T@x)/design.sum(0)[:,None])


def independent_evaluation(config,data,features,rows,device):
    """Check actual saved states and primary CKA by separate paths."""
    activation_max=0.;suffix_max=0.;vectors=0;activation_checks=[]
    for job in config['training_manifest']:
        architecture,encoding,task=job['architecture'],job['encoding'],job['task']
        for condition in ['initial','trained']:
            cpfile=ROOT/'initializations'/architecture/'step0.pt' if condition=='initial' else ROOT/'models'/job['run_id']/'checkpoint.pt'
            cp=torch.load(cpfile,map_location='cpu',weights_only=False)
            net=model(config,architecture,job['model_seed'],device).eval();net.load_state_dict(cp['model'])
            for n in [10,20,30]:
                chosen=np.flatnonzero(data['test_lengths']==n)[:8]
                assert len(chosen)==8
                prefix=data[f'test_{encoding}_input'][chosen,:,:2*n+4]
                ids=torch.as_tensor(prefix.reshape(24,2*n+4).astype('int64'),device=device)
                mask=ids.ne(TOKEN_TO_ID['<PAD>'])
                with torch.inference_mode():
                    actual=manual_final(net,ids,mask,architecture)[:,-1].cpu().numpy().reshape(8,3,256)
                    full,fullmask,_,end,_=batch(prefix.reshape(24,2*n+4),np.repeat(n,24),task,device=device)
                    fullhidden=manual_final(net,full,fullmask,architecture)[torch.arange(24,device=device),end].cpu().numpy().reshape(8,3,256)
                cached=features[architecture,encoding,task,condition,'test']['prefix_final'][chosen]
                error=float(np.max(np.abs(actual-cached)));causal=float(np.max(np.abs(actual-fullhidden)))
                assert error<=2e-5 and causal<=2e-5, (job['run_id'],condition,n,error,causal)
                activation_max=max(activation_max,error);suffix_max=max(suffix_max,causal);vectors+=24
                activation_checks.append(dict(run_id=job['run_id'],condition=condition,length=n,
                    vectors=24,prefix_only_cache_max_difference=error,suffix_causal_max_difference=causal))
            del net,cp
    # Fixed first relationship/length; no cell selection by effect size.
    relation=config['relations'][0];n=config['lengths'][0]
    correct=ACTIONS.index(relation['input_transform']);wrong=1 if correct==2 else 2
    bykey={(r['architecture'],r['encoding'],r['relation'],r['length'],r['control'],r['branch'],r['metric']):r for r in rows}
    cka_max=0.;cka_checks=[]
    for architecture in ARCHITECTURES:
        for encoding in ENCODINGS:
            for control in config['evaluation']['controls']:
                use,keys=selections(data,relation,'test',n,control)
                for branch in ['prefix_final','output_prob']:
                    recorded=bykey[architecture,encoding,relation['pair_id'],n,control,branch,'linear_cka']
                    for condition in ['initial','trained']:
                        x=independent_center(features[architecture,encoding,relation['left'],condition,'test'][branch][use,0],keys)
                        for role,ai in [('correct',correct),('wrong',wrong),('identity',0)]:
                            y=independent_center(features[architecture,encoding,relation['right'],condition,'test'][branch][use,ai],keys)
                            den=np.linalg.norm(x.T@x)*np.linalg.norm(y.T@y)
                            independent=float(np.square(x.T@y).sum()/den) if len(x)>=2 and den>1e-20 else None
                            existing=recorded[condition+'_'+role]
                            if independent is None:assert existing is None
                            else:
                                assert existing is not None
                                error=abs(independent-existing);assert error<=1e-8
                                cka_max=max(cka_max,error)
                            cka_checks.append(dict(architecture=architecture,encoding=encoding,control=control,branch=branch,
                                condition=condition,role=role,samples=int(use.sum()),independent_cka=independent,recorded_cka=existing))
    json_write(ROOT/'independent_evaluation.json',dict(status='passed',protocol_sha256=sha(CONFIG),
        execution_source_version_sha256=source_version(),activation_vectors_checked=vectors,
        max_saved_activation_difference=activation_max,max_prefix_suffix_causal_difference=suffix_max,
        max_cka_covariance_difference=cka_max,activation_checks=activation_checks,cka_checks=cka_checks,
        methods='Explicit embedding/positions->blocks or GRU->norm on prefix-only input; suffix causal comparison. Feature covariance CKA with independent strata design-matrix centering.',
        fixed_lengths=[10,20,30],fixed_samples_per_length=8,fixed_cka_relation=relation['pair_id'],fixed_cka_length=n))


def evaluate():
    config=verify(); assert (ROOT/'training_complete.json').exists()
    data,_,audit=load_data(config);device=device_setup();features={};accuracy=[]
    cache_root=ROOT/'evaluation_cache';cache_root.mkdir(exist_ok=True)
    for job in config['training_manifest']:
        architecture,encoding,task=job['architecture'],job['encoding'],job['task']
        completion=json.loads((ROOT/'models'/job['run_id']/'completed.json').read_text())
        cp_path=ROOT/'models'/job['run_id']/'checkpoint.pt'
        assert completion['step']==config['training']['updates'] and sha(cp_path)==completion['checkpoint_sha256']
        assert completion['protocol_sha256']==sha(CONFIG)
        assert completion['execution_source_version_sha256']==source_version()
        zero,zero_meta=initial(config,architecture)
        for condition in ['initial','trained']:
            cp=zero if condition=='initial' else torch.load(cp_path,map_location='cpu',weights_only=False)
            net=model(config,architecture,job['model_seed'],device).eval();net.load_state_dict(cp['model'])
            if condition=='initial':assert parameter_hash(net)==zero_meta['parameter_sha256']
            for split in ['validation','test']:
                signature=dict(protocol_sha256=sha(CONFIG),data_sha256=audit['data_sha256'],split=split,
                    checkpoint_sha256=zero_meta['checkpoint_sha256'] if condition=='initial' else completion['checkpoint_sha256'])
                path=cache_root/f'{architecture}_{encoding}_{task}_{condition}_{split}.npz'
                arrays=cache(path,net,data,encoding,task,device,split,signature)
                features[architecture,encoding,task,condition,split]=arrays
                if split=='test':
                    for n in config['lengths']:
                        use=data['test_lengths']==n
                        for ai,action in enumerate(ACTIONS):
                            truth=data['test_labels'][use,ai,TASKS.index(task)]
                            accuracy.append(dict(architecture=architecture,encoding=encoding,task=task,
                                condition=condition,length=n,action=action,samples=int(use.sum()),
                                full_vocab_accuracy=float(np.mean(arrays['prediction'][use,ai]==truth)),
                                numeric_accuracy=float(np.mean(arrays['output_prob'][use,ai].argmax(-1)==truth))))
            del net,cp
        print({'evaluated':job['run_id']},flush=True)
    for architecture in ARCHITECTURES:
        for encoding in ENCODINGS:
            for split in ['validation','test']:
                base=features[architecture,encoding,DEV_TASKS[0],'initial',split]['prefix_final']
                for task in DEV_TASKS:
                    np.testing.assert_allclose(base,features[architecture,encoding,task,'initial',split]['prefix_final'],atol=1e-5,rtol=0)
    rows=[];goldrows=[]
    for relation in config['relations']:
        correct=ACTIONS.index(relation['input_transform']);wrong=1 if correct==2 else 2
        roles={'correct':correct,'wrong':wrong,'identity':0}
        for n in config['lengths']:
            for control in config['evaluation']['controls']:
                fit,kfit=selections(data,relation,'validation',n,control)
                test,ktest=selections(data,relation,'test',n,control)
                ids=np.flatnonzero(test);duplicates=data['test_permutations'][test]
                for architecture in ARCHITECTURES:
                    for encoding in ENCODINGS:
                        for branch in ['prefix_final','output_prob']:
                            results={metric:{} for metric in ['linear_cka','debiased_cka','knn_overlap','procrustes']}
                            errors={metric:{} for metric in results}
                            for condition in ['initial','trained']:
                                for metric in results:results[metric][condition]={};errors[metric][condition]={}
                                xfit=features[architecture,encoding,relation['left'],condition,'validation'][branch][fit,0]
                                xtest=features[architecture,encoding,relation['left'],condition,'test'][branch][test,0]
                                for role,ai in roles.items():
                                    yfit=features[architecture,encoding,relation['right'],condition,'validation'][branch][fit,ai]
                                    ytest=features[architecture,encoding,relation['right'],condition,'test'][branch][test,ai]
                                    values=metric_values(xfit,yfit,xtest,ytest,kfit,ktest,ids,duplicates)
                                    for metric,(value,error) in values.items():
                                        results[metric][condition][role]=value;errors[metric][condition][role]=error
                            for metric,value in results.items():
                                rows.append(dict(architecture=architecture,encoding=encoding,relation=relation['pair_id'],
                                    length=n,control=control,branch=branch,metric=metric,fit_samples=int(fit.sum()),test_samples=int(test.sum()),
                                    **components(value),initial_correct_error=errors[metric]['initial']['correct'],
                                    trained_correct_error=errors[metric]['trained']['correct']))
                        # Static true answers have no trained/initial contrast.
                        for baseline in ['true_answer_onehot','true_answer_scalar']:
                            for role,ai in roles.items():
                                values=[]
                                for split,use in [('validation',fit),('test',test)]:
                                    labels=data[split+'_labels'];left=labels[use,0,TASKS.index(relation['left'])]
                                    right=labels[use,ai,TASKS.index(relation['right'])]
                                    values.extend([np.eye(31)[left] if baseline.endswith('onehot') else left[:,None],
                                                   np.eye(31)[right] if baseline.endswith('onehot') else right[:,None]])
                                scores=metric_values(*values,kfit,ktest,ids,duplicates)
                                for metric,(value,error) in scores.items():
                                    goldrows.append(dict(architecture=architecture,encoding=encoding,relation=relation['pair_id'],
                                        length=n,control=control,baseline=baseline,role=role,metric=metric,value=value,error=error,
                                        fit_samples=int(fit.sum()),test_samples=int(test.sum()),static_no_training_delta=True))
        print({'geometry_relation_complete':relation['pair_id']},flush=True)
    independent_evaluation(config,data,features,rows,device)
    write_csv(ROOT/'geometry_length.csv',rows);write_csv(ROOT/'gold_baseline_length.csv',goldrows)
    write_csv(ROOT/'task_accuracy_length.csv',accuracy)
    json_write(ROOT/'evaluation_complete.json',dict(protocol_sha256=sha(CONFIG),geometry_rows=len(rows),gold_rows=len(goldrows),
        models=len(config['training_manifest']),relations=len(config['relations']),metric_checks=metrics.verify(),
        source_unseen_geometry_test=True,final_layer_primary=True,formal_training_started=False))


def report():
    config=verify();assert json.loads((ROOT/'evaluation_complete.json').read_text())['protocol_sha256']==sha(CONFIG)
    with (ROOT/'geometry_length.csv').open() as f:rows=list(csv.DictReader(f))
    fields=['B0','Bt','change_correct','change_wrong','change_identity','Delta',
            'initial_correct','initial_wrong','initial_identity','trained_correct','trained_wrong','trained_identity']
    groups={}
    for row in rows:
        key=tuple(row[k] for k in ['architecture','encoding','relation','control','branch','metric'])
        groups.setdefault(key,[]).append(row)
    aggregate=[];index={}
    for key,values in groups.items():
        assert sorted(int(r['length']) for r in values)==config['lengths']
        record=dict(zip(['architecture','encoding','relation','control','branch','metric'],key))
        for field in fields:
            record[field]=float(np.mean([float(r[field]) for r in values])) if all(r[field]!='' for r in values) else None
        record['lengths']=len(values);record['undefined_lengths']=sum(r['Delta']=='' for r in values)
        aggregate.append(record);index[key]=record
    interactions=[]
    for key,left in index.items():
        architecture,encoding,relation,control,branch,metric=key
        if encoding!='one_line':continue
        right=index[architecture,'canonical_cycle',relation,control,branch,metric]
        record=dict(architecture=architecture,relation=relation,control=control,branch=branch,metric=metric,development_only=True)
        for field,name in [('B0','Gamma0'),('Bt','Gammat'),('change_correct','Gamma_correct'),('change_wrong','Gamma_wrong'),('Delta','GammaDelta')]:
            record[name]=right[field]-left[field] if right[field] is not None and left[field] is not None else None
        interactions.append(record)
    write_csv(ROOT/'geometry_summary.csv',aggregate);write_csv(ROOT/'encoding_interactions.csv',interactions)
    data,_,_=load_data(config);device=device_setup();learning=[]
    from .training import accuracy
    for job in config['training_manifest']:
        saved,meta=initial(config,job['architecture']);net=model(config,job['architecture'],job['model_seed'],device)
        net.load_state_dict(saved['model']);before=accuracy(net,data,job['encoding'],job['task'],device)
        done=json.loads((ROOT/'models'/job['run_id']/'completed.json').read_text());after=done['curve'][-1]['answer_accuracy']
        learning.append(dict(run_id=job['run_id'],architecture=job['architecture'],encoding=job['encoding'],task=job['task'],
            parameters=done['parameters'],initial_validation_accuracy=before['answer_accuracy'],final_validation_accuracy=after,
            gain=after-before['answer_accuracy'],learning_gate_passed=after-before['answer_accuracy']>=config['engineering']['source_validation_gain_over_initial_min']))
        del net
    write_csv(ROOT/'source_learning.csv',learning)
    gates={name:json.loads((ROOT/name).read_text())['status'] for name in ['cpu_check.json','overfit_summary.json','resume_summary.json','independent_evaluation.json']}
    status='passed' if all(v=='passed' for v in gates.values()) and all(r['learning_gate_passed'] for r in learning) else 'failed'
    json_write(ROOT/'engineering_complete.json',dict(status=status,gates=gates,models=16,development_only=True,
        geometry_not_used_for_gate=True,formal_training_started=False,stop_after_report=True,
        no_confirmation_pvalues=True,parameter_counts_not_matched=True))
    lines=['PermWorld Encoding  × Architecture-development experiment',f'Engineering status: {status}',
        'New data world/model seeds; four tasks × two encodings × two architectures, totaling 16 models; fixed 20k updates.',
        'Genuine step-0 is shared within architectures; parameter counts are not matched. Development results do not enter formal confirmation or trigger expansion.',
        'Three fixed readouts per length; Procrustes is fitted only on validation and evaluated on test.debiased CKA is a sensitivity.',
        'Answer control uses exact answer strata, without removing all confidence/output information.CKA does not independently identify a mathematical mechanism.',
        'The complete B0/Bt, correct/wrong changes, Delta, and encoding interactions: CSV; retain all lengths and undefined cells.',
        'Source-learning gates assess task learning and engineering completeness, not geometric-effect direction or significance.',
        'Stop after this round report; no not_started formal manifest, added seeds, GroupWorld, F17, or larger models.','',
        *[str(r) for r in learning], '', *[str(r) for r in interactions if r['branch']=='prefix_final' and r['control']=='answer_strata']]
    (ROOT/'report.txt').write_text('\n'.join(lines)+'\n')
    print({'development_complete':status,'formal_training_started':False},flush=True)
