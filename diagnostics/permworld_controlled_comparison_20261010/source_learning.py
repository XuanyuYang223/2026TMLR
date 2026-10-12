"""Full-vocabulary source diagnostics on the original fixed evaluation anchors."""
from common import *

def main():
    protocol=verify();layerindex=js(FROOT/'layers/index.json')
    assert layerindex['status']=='completed'
    rec={r['source_key']:r for r in layerindex['records']}
    with np.load(protocol['source_data_path'],allow_pickle=False) as z:
        data={s+'_'+k:z[s+'_'+k] for s in ['validation','test'] for k in ['lengths','labels']}
    baseline_source=WORKSPACE/'diagnostics/permworld_layers_20261009/encoding_development/length_baseline_accuracy_supplemental.csv'
    baselines={(r['task'],int(r['length']),r['action']):r for r in rows(baseline_source)}
    records=[];audit=[]
    for job in protocol['checkpoint_inventory']:
        complete=js(job['completed_path']);old_val=complete['curve'][-1]
        assert old_val['step']==20000
        for condition in ['initial','trained']:
            for split in ['validation','test']:
                logical='|'.join([job['run_id'],condition,split])
                old=layerindex['logical_old_cache_map'][logical];assert sha(old['path'])==old['sha256']
                with np.load(old['path'],allow_pickle=False) as z:pred=z['prediction']
                ns=data[split+'_lengths'];labels=data[split+'_labels'][:,:,TASKS.index(job['task'])]
                assert pred.shape==labels.shape
                if condition=='trained':
                    row=rec[layerindex['job_condition_split_map'][logical]]
                    ce=np.load(row['answer_ce_path'],mmap_mode='r');newpred=np.load(row['answer_argmax_path'],mmap_mode='r')
                    assert sha(row['answer_ce_path'])==row['answer_ce_sha256']
                    assert sha(row['answer_argmax_path'])==row['answer_argmax_sha256']
                    assert np.array_equal(newpred,pred) and np.isfinite(ce).all()
                else:ce=None
                for n in protocol['lengths']:
                    use=ns==n;count=int(use.sum());assert count==(256 if split=='validation' else 512)
                    for ai,action in enumerate(ACTIONS):
                        acc=int((pred[use,ai]==labels[use,ai]).sum())/count
                        b=baselines[job['task'],n,action]
                        majority=int(b['training_majority_label'])
                        base=int((labels[use,ai]==majority).sum())/count
                        if split=='test':assert base==float(b['length_majority_accuracy'])
                        value=math.fsum(map(float,ce[use,ai]))/count if ce is not None else float('nan')
                        records.append(dict(run_id=job['run_id'],architecture=job['architecture'],encoding=job['encoding'],
                            task=job['task'],condition=condition,split=split,action=action,length=n,samples=count,
                            answer_accuracy=acc,answer_ce=value,answer_ce_reason='' if ce is not None else 'Not saved originally; no task-specific initial CE newly evaluated',
                            length_majority_accuracy=base,accuracy_minus_length_mode=acc-base,training_majority_label=majority,
                            majority_source='original source train labels; smallest-label tie rule',
                            eos_accuracy=float('nan'),eos_reason='No EOS appended in prompt-only extraction; old teacher-forced validation endpoint given separately',
                            evaluated_token='answer at2n+5, argmax and CE over all188 vocabulary logits',parameters=job['parameters']))
                if condition=='trained' and split=='validation':
                    identity=float(np.mean(pred[:,0]==labels[:,0]))
                    audit.append(dict(run_id=job['run_id'],new_original_validation_answer_accuracy=identity,
                        old_validation_answer_accuracy=old_val['answer_accuracy'],
                        signed_accuracy_difference_from_teacher_forced_old_curve=identity-old_val['answer_accuracy'],
                        old_curve_accuracy_exactly_reproduced=(identity==old_val['answer_accuracy']),
                        old_validation_teacher_forced_eos_accuracy=old_val['eos_accuracy'],
                        old_validation_samples=old_val['samples'],eos_not_newly_evaluated=True,predictions_exact=True))
    assert len(records)==4032 and len(audit)==16
    csv_write(DIAG/'source_learning.csv',records)
    csv_write(DIAG/'source_validation_eos_original.csv',audit)
    groups={}
    for r in records:
        key=tuple(r[k] for k in ['architecture','encoding','task','condition','split','action'])
        groups.setdefault(key,[]).append(r)
    summary=[]
    for key,g in groups.items():
        assert sorted(r['length'] for r in g)==protocol['lengths']
        r=dict(zip(['architecture','encoding','task','condition','split','action'],key))
        for field in ['answer_accuracy','answer_ce','length_majority_accuracy','accuracy_minus_length_mode']:
            r[field]=equal_length([number(x[field]) for x in g])
        r.update(length_count=21,weights='fixed1/21',source_task_scope='original evaluation anchors; no accuracy-based selection')
        summary.append(r)
    csv_write(DIAG/'source_learning_summary.csv',summary)
    json_write(DIAG/'source_learning_audit.json',dict(status='passed',rows=len(records),source_conditions=16,
        length_count=21,summary_rows=len(summary),trained_predictions_exactly_equal_old_cache=True,
        original_validation_accuracy_exactly_reproduced=all(r['old_curve_accuracy_exactly_reproduced'] for r in audit),
        old_curve_comparison_scope='Old answer+EOS teacher-forced full sequence versus original prompt-only cached predictions; signed differences retained',
        length_mode_test_exactly_reused=True,
        initial_CE_missing=True,reason='Shared initial hidden extraction is not shared task-specific logits',
        new_source_training=False,new_evaluation_anchors=False,old8k_excluded=True,
        baseline_source=str(baseline_source),baseline_source_sha256=sha(baseline_source)))
    print(dict(source_learning_rows=len(records),length_summary_rows=len(summary)),flush=True)

if __name__=='__main__':main()
