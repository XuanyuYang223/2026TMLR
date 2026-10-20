"""Read-only export of the encoding × architecture development experiment.

This file is independent of both frozen evaluators. It never trains, loads model
weights, or changes the F-drive experiment. --fixture uses only synthetic CPU
tables in a temporary directory. Real export requires all sixteen fixed final
models and the completed evaluator/engineering report.
"""
import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import html
import json
import math
from pathlib import Path
import shutil
import tempfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE=Path(__file__).resolve().parent
DEFAULT_SOURCE=Path('/mnt/yangx-projects-f/20261009/permworld_new_artifacts/encoding_archdev')
DEFAULT_OUTPUT=HERE/'encoding_development'
ARCHS=('transformer','gru')
ENCODINGS=('one_line','canonical_cycle')
RELATIONS=('descent_inverse','peak_complement')
CONTROLS=('raw_within_length','answer_strata')
METRICS=('linear_cka','knn_overlap','procrustes','debiased_cka')
BRANCHES=('prefix_final','output_prob')
TASKS=('descents','recoils','peaks','valleys')
DATA_LABEL_TASKS=('anti_fixed_points','components','deficiencies','descents','exceedances','fixed_points',
    'global_descents','lds_length','left_to_right_maxima','left_to_right_minima','lis_length',
    'longest_decreasing_run','longest_increasing_run','peaks','recoils','valleys')
LENGTHS=tuple(range(10,31))
COMPONENTS=('B0','Bt','change_correct','change_wrong','Delta')
GAMMAS=('Gamma0','Gammat','GammaDelta')
LABELS={'B0':'Initialization preference B₀','Bt':'Trained preference Bₜ',
        'change_correct':'Correct-score change δc',
        'change_wrong':'Wrong-score change δw','Delta':'Training increment Δ',
        'Gamma0':'Encoding interaction Γ₀','Gammat':'Encoding interaction Γₜ',
        'GammaDelta':'Encoding interaction ΓΔ'}
COLORS=('#355f9c','#78a7db','#aa5b20','#e3a16d')


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(2**20),b''):digest.update(block)
    return digest.hexdigest()


def load_json(path):return json.loads(Path(path).read_text())


def load_csv(path):
    with Path(path).open(newline='') as stream:return list(csv.DictReader(stream))


def write_csv(path,rows):
    with Path(path).open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def number(value):
    if value in (None,''):return None
    parsed=float(value)
    assert math.isfinite(parsed), f'Nonfinite stored result: {value}'
    return parsed


def format_value(value):
    if value in (None,''):return 'undefined'
    if isinstance(value,float):return f'{value:.5f}'
    try:
        v=float(value)
        if math.isfinite(v):return f'{v:.5f}' if any(c in str(value) for c in '.eE') else str(value)
    except (ValueError,TypeError):pass
    return str(value)


def table(rows,columns):
    head=''.join(f'<th>{html.escape(k)}</th>' for k in columns)
    body=''.join('<tr>'+''.join(f'<td>{html.escape(format_value(row.get(k)))}</td>' for k in columns)+'</tr>' for row in rows)
    return '<div class="scroll"><table><thead><tr>'+head+'</tr></thead><tbody>'+body+'</tbody></table></div>'


def validate_results(source):
    required=['protocol.json','freeze_manifest.json','geometry_length.csv','geometry_summary.csv',
              'encoding_interactions.csv','source_learning.csv','task_accuracy_length.csv',
              'gold_baseline_length.csv','engineering_complete.json','evaluation_complete.json',
              'training_complete.json','independent_evaluation.json','cpu_check.json',
              'overfit_summary.json','resume_summary.json','data_ready.json']
    missing=[name for name in required if not (source/name).is_file()]
    assert not missing, f'Experiment incomplete; missing: {missing}'
    protocol=load_json(source/'protocol.json');summary=load_csv(source/'geometry_summary.csv')
    interactions=load_csv(source/'encoding_interactions.csv');learning=load_csv(source/'source_learning.csv')
    accuracy=load_csv(source/'task_accuracy_length.csv');length=load_csv(source/'geometry_length.csv')
    complete=load_json(source/'engineering_complete.json')
    assert protocol['lengths']==list(LENGTHS)
    assert tuple(protocol['tasks'])==TASKS and tuple(protocol['architectures'])==ARCHS
    assert tuple(r['pair_id'] for r in protocol['relations'])==RELATIONS
    assert len(protocol['training_manifest'])==16
    assert protocol['training']['updates']==20000
    assert complete['models']==16 and complete['development_only'] is True
    assert complete['formal_training_started'] is False
    # A failed engineering gate is also a valid completed negative development
    # report. Export preserves failures and never drops or reruns weak models.
    assert complete['status'] in ('passed','failed')
    keys={(r['architecture'],r['encoding'],r['relation'],r['control'],r['branch'],r['metric']) for r in summary}
    expected={(a,e,r,c,b,m) for a in ARCHS for e in ENCODINGS for r in RELATIONS
              for c in CONTROLS for b in BRANCHES for m in METRICS}
    assert keys==expected and len(summary)==len(expected)==128
    ikeys={(r['architecture'],r['relation'],r['control'],r['branch'],r['metric']) for r in interactions}
    assert ikeys=={(a,r,c,b,m) for a in ARCHS for r in RELATIONS for c in CONTROLS for b in BRANCHES for m in METRICS}
    assert len(interactions)==64
    for row in summary:
        if number(row['Delta']) is not None:
            assert abs(number(row['Delta'])-(number(row['Bt'])-number(row['B0'])))<1e-8
            assert abs(number(row['Delta'])-(number(row['change_correct'])-number(row['change_wrong'])))<1e-8
    for row in interactions:
        if number(row['GammaDelta']) is not None:
            assert abs(number(row['GammaDelta'])-(number(row['Gammat'])-number(row['Gamma0'])))<1e-8
    grouped=defaultdict(list)
    for row in length:
        grouped[tuple(row[k] for k in ['architecture','encoding','relation','control','branch','metric'])].append(row)
    assert set(grouped)==expected
    for key,values in grouped.items():
        assert sorted(int(v['length']) for v in values)==list(LENGTHS)
        aggregate=next(row for row in summary if tuple(row[k] for k in ['architecture','encoding','relation','control','branch','metric'])==key)
        for field in COMPONENTS:
            values_field=[number(row[field]) for row in values]
            existing=number(aggregate[field])
            if any(v is None for v in values_field):assert existing is None
            else:assert abs(existing-float(np.mean(values_field)))<1e-8
    accuracy_keys={(r['architecture'],r['encoding'],r['task'],r['condition'],int(r['length']),r['action']) for r in accuracy}
    assert accuracy_keys=={(a,e,t,c,n,g) for a in ARCHS for e in ENCODINGS for t in TASKS
                           for c in ['initial','trained'] for n in LENGTHS for g in ['identity','complement','inverse']}
    assert len(accuracy)==2016 and len(learning)==16
    curves=[]
    for job in protocol['training_manifest']:
        done=load_json(source/'models'/job['run_id']/'completed.json')
        assert done['step']==20000 and done['curve'][-1]['step']==20000
        assert done['architecture']==job['architecture'] and done['encoding']==job['encoding'] and done['task']==job['task']
        for row in done['curve']:
            curves.append(dict(run_id=job['run_id'],architecture=job['architecture'],encoding=job['encoding'],
                task=job['task'],step=row['step'],train_loss=row['train_loss'],validation_answer_accuracy=row['answer_accuracy'],
                validation_numeric_accuracy=row.get('numeric_accuracy'),validation_eos_accuracy=row.get('eos_accuracy')))
    return dict(protocol=protocol,summary=summary,interactions=interactions,learning=learning,
                accuracy=accuracy,length=length,complete=complete,curves=curves)


def bars(axis,values,labels,colors):
    positions=np.arange(len(values))
    for x,value,color in zip(positions,values,colors):
        if value is None:
            axis.plot([x],[0.],marker='x',color='#999');axis.annotate('undefined',(x,0.),xytext=(0,8),textcoords='offset points',ha='center',fontsize=7)
        else:
            axis.bar(x,value,color=color,width=.7)
    axis.axhline(0,color='#777',linewidth=.6)
    axis.set_xticks(positions,labels,rotation=35,ha='right',fontsize=8)
    axis.grid(axis='y',alpha=.15)


def save_figure(fig,path):
    fig.tight_layout();fig.savefig(path,dpi=145,bbox_inches='tight');plt.close(fig)


def component_plots(bundle,output):
    index={tuple(r[k] for k in ['architecture','encoding','relation','control','branch','metric']):r for r in bundle['summary']}
    interactions={tuple(r[k] for k in ['architecture','relation','control','branch','metric']):r for r in bundle['interactions']}
    conditions=[(a,e) for a in ARCHS for e in ENCODINGS]
    condition_labels=['TF / line','TF / cycle','GRU / line','GRU / cycle']
    plots=[]
    for branch in BRANCHES:
        for control in CONTROLS:
            for metric in METRICS:
                fig,axes=plt.subplots(2,5,figsize=(16,6.6),squeeze=False)
                for ri,relation in enumerate(RELATIONS):
                    for ci,component in enumerate(COMPONENTS):
                        values=[number(index[a,e,relation,control,branch,metric][component]) for a,e in conditions]
                        bars(axes[ri,ci],values,condition_labels,COLORS)
                        axes[ri,ci].set_title(LABELS[component],fontsize=9)
                        if ci==0:axes[ri,ci].set_ylabel(relation,fontsize=10)
                title=f'{branch} | {control} | {metric}: all two relations and four conditions'
                fig.suptitle(title,fontsize=12);name=f'components_{branch}_{control}_{metric}.png';save_figure(fig,output/name)
                plots.append(dict(kind='components',branch=branch,control=control,metric=metric,path=name,title=title))
                fig,axes=plt.subplots(2,3,figsize=(11,6.6),squeeze=False)
                for ri,relation in enumerate(RELATIONS):
                    for gi,gamma in enumerate(GAMMAS):
                        values=[number(interactions[a,relation,control,branch,metric][gamma]) for a in ARCHS]
                        bars(axes[ri,gi],values,['Transformer','GRU'],['#355f9c','#aa5b20'])
                        axes[ri,gi].set_title(LABELS[gamma],fontsize=10)
                        if gi==0:axes[ri,gi].set_ylabel(relation,fontsize=10)
                title=f'{branch} | {control} | {metric}: cycle minus line encoding interaction'
                fig.suptitle(title,fontsize=11);name=f'interactions_{branch}_{control}_{metric}.png';save_figure(fig,output/name)
                plots.append(dict(kind='interactions',branch=branch,control=control,metric=metric,path=name,title=title))
    return plots


def length_baseline_supplement(bundle,source,output,fixture=False):
    """Post-protocol descriptive check; labels from source train choose modes."""
    protocol=bundle['protocol'];unit=protocol['units'][0]['unit']
    local=source/'worlds'/unit/'data.npz'
    data_path=local if local.exists() else Path(protocol['output'])/unit/'data.npz'
    data_sha=None
    if data_path.exists():
        audit_path=data_path.with_name('data_audit.json')
        data_sha=sha(data_path)
        if audit_path.exists():assert data_sha==load_json(audit_path)['data_sha256']
        with np.load(data_path,allow_pickle=False) as archive:
            assert archive['train_labels'].shape[-1]==len(DATA_LABEL_TASKS)
            assert archive['test_labels'].shape[-1]==len(DATA_LABEL_TASKS)
            counts=[]
            for split,actions in [('train',['identity']),('test',['identity','complement','inverse'])]:
                ns=archive[split+'_lengths'];answers=archive[split+'_labels']
                for task in TASKS:
                    task_index=DATA_LABEL_TASKS.index(task)
                    for n in LENGTHS:
                        use=ns==n;assert use.any()
                        for ai,action in enumerate(actions):
                            labels=answers[use,ai,task_index]
                            assert np.max(labels)<=30
                            histogram=np.bincount(labels,minlength=31)
                            for label,count in enumerate(histogram):
                                counts.append(dict(split=split,task=task,length=n,action=action,label=label,count=int(count)))
    else:
        # All modes/accuracies can be independently reconstructed from the small
        # exported count table without recovering the binary object corpus.
        candidates=[source/'label_counts_length_baseline_supplemental.csv',source.parent/'label_counts_length_baseline_supplemental.csv']
        available=next((p for p in candidates if p.exists()),None)
        assert available is not None, 'Missing labels and previously exported supplemental counts'
        counts=load_csv(available)
    grouped=defaultdict(dict)
    for row in counts:
        key=(row['split'],row['task'],int(row['length']),row['action'])
        label=int(row['label']);assert label not in grouped[key];grouped[key][label]=int(row['count'])
    assert len(counts)==10416
    baseline=[];baseline_index={}
    for task in TASKS:
        for n in LENGTHS:
            train=grouped['train',task,n,'identity'];assert sorted(train)==list(range(31))
            largest=max(train.values());mode=min(label for label,count in train.items() if count==largest)
            for action in ['identity','complement','inverse']:
                test=grouped['test',task,n,action];assert sorted(test)==list(range(31))
                samples=sum(test.values());assert samples>0
                row=dict(task=task,length=n,action=action,training_examples=sum(train.values()),
                    training_majority_label=mode,training_majority_count=largest,testing_examples=samples,
                    length_majority_accuracy=test[mode]/samples,test_correct=test[mode],tie_policy='smallest_label',
                    supplemental_post_protocol=True)
                baseline.append(row);baseline_index[task,n,action]=row
    differences=[]
    for row in bundle['accuracy']:
        reference=baseline_index[row['task'],int(row['length']),row['action']]
        full=number(row['full_vocab_accuracy']);numeric=number(row['numeric_accuracy']);base=reference['length_majority_accuracy']
        differences.append(dict(architecture=row['architecture'],encoding=row['encoding'],task=row['task'],
            condition=row['condition'],length=int(row['length']),action=row['action'],
            testing_examples=reference['testing_examples'],training_majority_label=reference['training_majority_label'],
            full_vocab_accuracy=full,numeric_accuracy=numeric,length_majority_accuracy=base,
            full_vocab_gain_over_length_majority_pp=100*(full-base),numeric_gain_over_length_majority_pp=100*(numeric-base),
            supplemental_post_protocol=True))
    summaries=[];groups=defaultdict(list)
    for row in differences:
        groups[tuple(row[k] for k in ['architecture','encoding','task','condition','action'])].append(row)
    for key,values in groups.items():
        assert sorted(r['length'] for r in values)==list(LENGTHS)
        row=dict(zip(['architecture','encoding','task','condition','action'],key))
        for field in ['full_vocab_accuracy','numeric_accuracy','length_majority_accuracy',
                      'full_vocab_gain_over_length_majority_pp','numeric_gain_over_length_majority_pp']:
            row[field]=float(np.mean([r[field] for r in values]))
        row['fixed_length_weights']='1/21';row['supplemental_post_protocol']=True;summaries.append(row)
    write_csv(output/'label_counts_length_baseline_supplemental.csv',counts)
    write_csv(output/'length_baseline_accuracy_supplemental.csv',baseline)
    write_csv(output/'model_minus_length_baseline_supplemental.csv',differences)
    write_csv(output/'model_minus_length_baseline_summary_supplemental.csv',summaries)
    note=dict(status='complete',added_utc=datetime.now(timezone.utc).isoformat(),added_client_date='2026-10-10',
        client_timezone='America/Los_Angeles',post_protocol_descriptive=True,fixture_only=fixture,
        reason='Engineering review noted that step-0+10pp gate and roughly30% early accuracy can reflect length-dependent label majority; added to describe content-blind baseline, not modify main metrics or gates.',
        fixed_training_majority_rule='For each task/length use only source train label counts; ties choose smallest label. Apply same frozen predictor to every test action and every model.',
        tasks=list(TASKS),actions=['identity','complement','inverse'],label_array_order=list(DATA_LABEL_TASKS),
        source_data_sha256=data_sha,statistic='Model accuracy minus training-defined length-majority accuracy, fixed1/21 average; no p-values.',
        main_metric_or_gate_or_budget_changed=False,no_selected_samples_or_retraining=True,counts_rows=len(counts),
        complete_model_length_rows=len(differences),interpretation='Above this baseline shows more than this particular length-only predictor; below does not prove absence of permutation-content use.')
    (output/'length_baseline_protocol_supplemental.json').write_text(json.dumps(note,indent=2,ensure_ascii=False)+'\n')
    bundle['length_baseline']=baseline;bundle['length_baseline_summary']=summaries
    return note


def learning_plots(bundle,output):
    accuracy=bundle['accuracy'];curves=bundle['curves'];plots=[]
    for action in ['identity','complement','inverse']:
        for endpoint in ['full_vocab_accuracy','numeric_accuracy']:
            fig,axes=plt.subplots(4,2,figsize=(12,11),sharex=True,sharey=True,squeeze=False)
            for ti,task in enumerate(TASKS):
                for ai,architecture in enumerate(ARCHS):
                    axis=axes[ti,ai]
                    for ei,encoding in enumerate(ENCODINGS):
                        for condition,style in [('initial','--'),('trained','-')]:
                            selected=sorted([r for r in accuracy if r['task']==task and r['architecture']==architecture
                                and r['encoding']==encoding and r['condition']==condition and r['action']==action],key=lambda r:int(r['length']))
                            axis.plot([int(r['length']) for r in selected],[float(r[endpoint]) for r in selected],style,
                                color=COLORS[ei],label=f'{encoding} {condition}',linewidth=1.3)
                    reference=sorted([r for r in bundle.get('length_baseline',[]) if r['task']==task and r['action']==action],key=lambda r:r['length'])
                    if reference:axis.plot([r['length'] for r in reference],[r['length_majority_accuracy'] for r in reference],':',color='#444',label='train-defined length majority',linewidth=1.5)
                    axis.set_ylim(0,1);axis.grid(alpha=.15);axis.set_title(f'{task} / {architecture}',fontsize=10)
                    if ti==3:axis.set_xlabel('Permutation length')
                    if ai==0:axis.set_ylabel('Accuracy')
            axes[0,0].legend(fontsize=7);title=f'All sixteen model conditions: {endpoint}, {action}'
            fig.suptitle(title,fontsize=12);name=f'accuracy_{action}_{endpoint}.png';save_figure(fig,output/name)
            plots.append(dict(kind='accuracy',path=name,title=title))
    for endpoint in ['validation_answer_accuracy','train_loss']:
        fig,axes=plt.subplots(4,2,figsize=(12,11),sharex=True,squeeze=False)
        for ti,task in enumerate(TASKS):
            for ai,architecture in enumerate(ARCHS):
                axis=axes[ti,ai]
                for ei,encoding in enumerate(ENCODINGS):
                    selected=sorted([r for r in curves if r['task']==task and r['architecture']==architecture and r['encoding']==encoding],key=lambda r:int(r['step']))
                    axis.plot([int(r['step']) for r in selected],[float(r[endpoint]) for r in selected],color=COLORS[ei],label=encoding,linewidth=1.3)
                axis.grid(alpha=.15);axis.set_title(f'{task} / {architecture}',fontsize=10)
                if endpoint.endswith('accuracy'):axis.set_ylim(0,1)
                if ti==3:axis.set_xlabel('Fixed training updates')
                if ai==0:axis.set_ylabel(endpoint)
        axes[0,0].legend(fontsize=8);title=f'All source curves: {endpoint}; fixed final20k, no best-checkpoint selection'
        fig.suptitle(title,fontsize=12);name=f'learning_{endpoint}.png';save_figure(fig,output/name)
        plots.append(dict(kind='learning',path=name,title=title))
    return plots


def copy_small_artifacts(source,output):
    files=[]
    for path in source.iterdir():
        if path.is_file() and path.suffix in ('.csv','.json','.txt'):files.append(path)
    files.extend((source/'models').glob('*/completed.json'))
    files.extend((source/'initializations').glob('*.json'))
    files.extend((source/'initializations').glob('*/*.json'))
    files.extend((source/'worlds').glob('archdev001/data_audit.json'))
    files.extend((source/'worlds').glob('historical_input_hashes.json'))
    files.extend((source/'overfit').glob('*/result.json'))
    files.extend((source/'resume').glob('*/result.json'))
    copied=[]
    for path in sorted(set(files)):
        assert path.stat().st_size<50*2**20, f'Unexpected large metadata artifact: {path}'
        relative=path.relative_to(source);target=output/'artifacts'/relative;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(path,target);assert sha(path)==sha(target)
        copied.append(dict(source_relative=str(relative),export_relative=str(target.relative_to(output)),bytes=path.stat().st_size,sha256=sha(path)))
    return copied


def render_report(bundle,plots,copied,output,source,fixture=False):
    complete=bundle['complete'];parts=['<!doctype html><html lang="en"><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width,initial-scale=1">',
        '<title>PermWorld encoding and architecture development results</title><style>',
        'body{font:16px system-ui,sans-serif;color:#243040;margin:32px auto;max-width:1220px;padding:0 20px;line-height:1.6}',
        'h1,h2,h3{line-height:1.3}.note{padding:18px;background:#edf3fa;border-left:4px solid #355f9c}',
        '.scroll{overflow-x:auto}table{border-collapse:collapse;font-size:12px;white-space:nowrap;width:100%}',
        'th,td{border-bottom:1px solid #ddd;text-align:right;padding:7px}th{background:#f3f5f7}td:first-child,th:first-child{text-align:left}',
        'figure{margin:24px 0}img{max-width:100%;height:auto}summary{cursor:pointer;font-weight:600}',
        'code{word-break:break-all}a{color:#355f9c}.failed{color:#a13f32}</style>',
        '<h1>PermWorld: encoding × architecture development results</h1>']
    if fixture:parts.append('<p class="note">Synthetic fixture only: the following figures/tables are code-validation data, not experimental results.</p>')
    parts.append('<p class="note">One development data world/source initialization unit: an engineering and exploratory experiment, without confirmatory significance or equivalence claims.'
        'Four tasks × two encodings × two architectures = 16 models; all retained. Transformer: width 256, 4 layers, 8 heads, 3,240,448 parameters;'
        'GRU: width 256, 4 layers, 1,660,416 parameters; capacity is unmatched. Both encodings copy the exact initialization within each architecture; weights differ across architectures.</p>')
    parts.append(f'<p>Engineering completion status:<strong>{html.escape(complete["status"])}</strong>.'
        'Fixed 20,000 updates, identical underlying permutations and batches, and identical answer+EOS supervision; no best-checkpoint selection or scaling/extensions for weak tasks.'
        'Differences in source-task mastery affect geometric readouts and cannot be directly attributed to architecture alone.</p>')
    parts.append('<h2>Interpreting encoding interactions</h2><p>Each metric separately displays initialization preference B₀, trained preference Bₜ, correct-pair change δc,'
        'wrong-pair change δw, and Δ. Δ = Bₜ − B₀ = δc − δw. Encoding interactions are cycle minus one-line: ΓΔ = Γₜ − Γ₀.</p>'
        '<p><strong>A sign change in Δ across encodings alone is not a reversal of learned relation preference.</strong>If Bₜ is similar but B₀ differs,'
        'the change concerns training increments relative to initialization. Correct-pair improvements and wrong-pair declines are also reported separately. Hidden minus output metrics do not identify an additional mechanism.</p>'
        '<p>Lengths 10–30 are computed separately with fixed 1/21 aggregation; undefined lengths are retained and aggregate weights are not renormalized. Neighborhood k=10; Procrustes is fitted on orbit-disjoint'
        'validation inputs and evaluated on test inputs, with negative prediction error as its score. Debiased CKA is a sensitivity analysis. Answer stratification cannot remove all confidence/output information.</p>')
    parts.append('<h2>Complete model learning performance and engineering gates</h2>')
    parts.append(table(bundle['learning'],['run_id','parameters','initial_validation_accuracy','final_validation_accuracy','gain','learning_gate_passed']))
    parts.append('<p><strong>Improving at least 10 percentage points over genuine initialization is only an engineering learning gate.</strong>It does not guarantee learning of permutation content,'
        'or support a strong architectural conclusion. The added length-mode diagnostic is a descriptive supplement after protocol freezing: each task/length mode uses training labels only,'
        'resolving ties to the smallest label and then scoring every test action and model; primary metrics, gates, samples, and budgets remain unchanged.</p>')
    parts.append('<details><summary>All models relative to the training-defined length baseline (post-hoc supplement)</summary>')
    parts.append(table(bundle.get('length_baseline_summary',[]),['architecture','encoding','task','condition','action',
        'full_vocab_accuracy','numeric_accuracy','length_majority_accuracy','full_vocab_gain_over_length_majority_pp','numeric_gain_over_length_majority_pp']))
    parts.append('</details><p>Exceeding this baseline means this specific length-only predictor cannot reproduce the performance; falling below it does not mean the model uses no permutation content.'
        'Complete per-length differences and training/test label counts can be independently recomputed; see<a href="model_minus_length_baseline_supplemental.csv">Model-by-length table</a>, '
        '<a href="label_counts_length_baseline_supplemental.csv">Label counts</a>and'
        '<a href="length_baseline_protocol_supplemental.json">Supplementary-analysis note</a>.</p>')
    parts.append('<h3>Engineering repairs and boundaries</h3><p>The historical development NPZ contains three prefixed permutation arrays that the generic reader previously skipped. The repair explicitly excludes'
        '102,144 historical development orbits and 817,152 orbit members; historical/new development orbit overlap is 0, with regeneration retaining the fixed data seed.'
        'Earlier data and audits are retained as snapshots; no source training used the incorrect data.</p>'
        '<p>Internal dropout state in the original multilayer cuDNN GRU cannot be restored through ordinary RNG checkpoints; an independent CUDA toy check found this problem.'
        'Four single-layer GRUs and three explicit Dropout(0.1) modules passed CUDA restoration checks. Original initialization is migrated tensor by tensor with identical canonical-coordinate hashes,'
        'and hidden-state/logit checks pass; migration is labeled a no-training rename of genuine original initialization, not falsely described as a new constructor initialization. Transformer and data remain unchanged.</p>'
        '<p>Final real-data audits include a continuous single-process 0→2000 reference versus independent-process 1000→2000 restoration, finite gradients and learning gates for all source models,'
        'prefix-only handwritten forward checks against initial/trained saved activations, suffix causality, and independent covariance-CKA verification. Specific statuses and values follow; failures are not described as passes.</p>')
    audit_names=['cpu_check.json','overfit_summary.json','resume_summary.json','independent_evaluation.json',
                 'development_exclusion_final_audit.json','data_ready.json','engineering_complete.json',
                 'explicit_dropout_repair_log.json','initializations/execution_binding_v4.json']
    for name in audit_names:
        path=source/name
        if path.exists():
            data=load_json(path);keys=['status','source_version_sha256','execution_source_version_sha256','models','source_optimizer_updates',
                'prior_orbits','new_orbits','prior_new_orbit_intersection','prior_development_orbit_members_appended',
                'activation_vectors_checked','max_saved_activation_difference','max_prefix_suffix_causal_difference','max_cka_covariance_difference',
                'all_corresponding_tensors_bitwise_equal','canonical_coordinate_sha256','formal_training_started','gates']
            selected={k:data[k] for k in keys if k in data}
            parts.append(f'<details><summary>{html.escape(name)}</summary><pre>{html.escape(json.dumps(selected,indent=2,ensure_ascii=False))}</pre>'
                f'<a href="artifacts/{html.escape(name,quote=True)}">Complete original JSON</a></details>')
    parts.append('<h2>Global geometry and output baselines: all prespecified conditions</h2>')
    for branch in BRANCHES:
        parts.append(f'<h3>{html.escape(branch)}</h3>')
        for control in CONTROLS:
            for metric in METRICS:
                parts.append(f'<details><summary>{html.escape(control)} / {html.escape(metric)}</summary>')
                for plot in plots:
                    if plot.get('branch')==branch and plot.get('control')==control and plot.get('metric')==metric:
                        parts.append(f'<figure><img loading="lazy" src="plots/{plot["path"]}" alt="{html.escape(plot["title"],quote=True)}"><figcaption>{html.escape(plot["title"])}</figcaption></figure>')
                selected=[r for r in bundle['summary'] if r['branch']==branch and r['control']==control and r['metric']==metric]
                parts.append(table(selected,['architecture','encoding','relation','B0','Bt','change_correct','change_wrong','Delta','undefined_lengths']))
                selected=[r for r in bundle['interactions'] if r['branch']==branch and r['control']==control and r['metric']==metric]
                parts.append(table(selected,['architecture','relation','Gamma0','Gammat','Gamma_correct','Gamma_wrong','GammaDelta']))
                parts.append('</details>')
    parts.append('<h2>Training curves and per-length accuracy for every model</h2>')
    for plot in plots:
        if plot['kind'] in ('accuracy','learning'):
            parts.append(f'<details><summary>{html.escape(plot["title"])}</summary><figure><img loading="lazy" src="plots/{plot["path"]}" alt="{html.escape(plot["title"],quote=True)}"></figure></details>')
    parts.append('<h2>Small artifacts for recomputation</h2><p>All per-length data, static-answer baselines, protocols, frozen hashes, repair records, and learning curves are exported.'
        'Large data, activations, and weights remain on drive F; this script does not read model weights, train, or recompute frozen scientific metrics.</p><ul>')
    for item in copied:
        parts.append(f'<li><a href="{html.escape(item["export_relative"],quote=True)}">{html.escape(item["source_relative"])}</a> ({item["bytes"]:,} bytes)</li>')
    parts.extend(['</ul><p>Development results cannot replace formal multi-unit verification or identify mathematical-encoding mechanisms from geometric similarity alone. Stop after the engineering report; no automatic seeds, systems, or training scale increases.</p></html>'])
    (output/'report.html').write_text('\n'.join(parts))


def export(source,output,fixture=False):
    assert source.resolve()!=output.resolve()
    bundle=validate_results(source)
    if not fixture:
        exclusion=load_json(source/'development_exclusion_final_audit.json')
        repair=load_json(source/'explicit_dropout_repair_log.json')
        assert exclusion['status']=='passed' and exclusion['prior_new_orbit_intersection']==0
        assert exclusion['prior_development_orbit_members_appended']==817152
        assert repair['all_corresponding_tensors_bitwise_equal'] and repair['head_embedding_tied']
        assert repair['parameters']==1660416
    output.mkdir(parents=True,exist_ok=True);plots_dir=output/'plots';plots_dir.mkdir(exist_ok=True)
    baseline_note=length_baseline_supplement(bundle,source,output,fixture)
    copied=copy_small_artifacts(source,output)
    write_csv(output/'source_training_curves.csv',bundle['curves'])
    plots=component_plots(bundle,plots_dir)+learning_plots(bundle,plots_dir)
    assert len(plots)==40
    render_report(bundle,plots,copied,output,source,fixture)
    evidence=dict(status='passed',fixture_only=fixture,source_directory=str(source),source_models=16,
        fixed_final_updates=20000,component_rows=128,interaction_rows=64,length_rows=len(bundle['length']),
        accuracy_length_rows=2016,plots=len(plots),all_four_conditions_two_relations_controls_metrics_present=True,
        original_undefined_values_retained=True,no_confirmatory_inference=True,no_model_weights_read=True,
        no_GPU_or_training_started=True,original_source_artifacts_unmodified=True,
        engineering_status=bundle['complete']['status'],source_small_artifacts=copied,script_sha256=sha(__file__))
    evidence['length_baseline_supplement']=baseline_note
    (output/'export_audit.json').write_text(json.dumps(evidence,indent=2,ensure_ascii=False)+'\n')
    return evidence


def fixture():
    with tempfile.TemporaryDirectory(prefix='permworld_encoding_export_fixture_') as temporary:
        root=Path(temporary);source=root/'source';source.mkdir();output=root/'output'
        jobs=[dict(run_id=f'fixture_{a}_{e}_{t}',architecture=a,encoding=e,task=t) for a in ARCHS for e in ENCODINGS for t in TASKS]
        protocol=dict(lengths=list(LENGTHS),tasks=list(TASKS),architectures=list(ARCHS),training_manifest=jobs,
                      relations=[{'pair_id':r} for r in RELATIONS],training={'updates':20000},
                      units=[{'unit':'archdev001'}],output=str(source/'worlds'))
        world=source/'worlds/archdev001';world.mkdir(parents=True)
        ns=np.repeat(LENGTHS,4);train=np.zeros((len(ns),1,16),dtype='uint8');test=np.zeros((len(ns),3,16),dtype='uint8')
        for task in TASKS:
            index=DATA_LABEL_TASKS.index(task);train[:,0,index]=np.tile([1,1,2,2],len(LENGTHS))
            test[:,0,index]=np.tile([1,1,1,2],len(LENGTHS));test[:,1,index]=np.tile([1,2,2,2],len(LENGTHS));test[:,2,index]=np.tile([1,1,2,2],len(LENGTHS))
        np.savez(world/'data.npz',train_lengths=ns,train_labels=train,test_lengths=ns,test_labels=test)
        (source/'protocol.json').write_text(json.dumps(protocol))
        complete=dict(status='passed',models=16,development_only=True,formal_training_started=False)
        for name in ['freeze_manifest','engineering_complete','evaluation_complete','training_complete','independent_evaluation',
                     'cpu_check','overfit_summary','resume_summary','data_ready']:
            (source/f'{name}.json').write_text(json.dumps(complete))
        rows=[];lengthrows=[];interactions=[]
        for a in ARCHS:
            for r in RELATIONS:
                for c in CONTROLS:
                    for b in BRANCHES:
                        for m in METRICS:
                            # Deliberate Delta sign flip with equal Bt: input
                            # initialization changes, trained preference does not.
                            for e,B0 in [('one_line',-.4),('canonical_cycle',.4)]:
                                dc=.01;Bt=.2;delta=Bt-B0;dw=dc-delta
                                record=dict(architecture=a,encoding=e,relation=r,control=c,branch=b,metric=m,
                                            B0=B0,Bt=Bt,change_correct=dc,change_wrong=dw,Delta=delta,undefined_lengths=0)
                                if a=='gru' and r=='peak_complement' and c=='answer_strata' and b=='output_prob' and m=='procrustes':
                                    for k in COMPONENTS:record[k]=None
                                    record['undefined_lengths']=21
                                rows.append(record)
                                for n in LENGTHS:lengthrows.append(dict(record,length=n))
                            gamma=dict(architecture=a,relation=r,control=c,branch=b,metric=m,Gamma0=.8,Gammat=0.,
                                       Gamma_correct=0.,Gamma_wrong=.8,GammaDelta=-.8)
                            if record['Delta'] is None:
                                for k in [*GAMMAS,'Gamma_correct','Gamma_wrong']:gamma[k]=None
                            interactions.append(gamma)
        write_csv(source/'geometry_summary.csv',rows);write_csv(source/'geometry_length.csv',lengthrows)
        write_csv(source/'encoding_interactions.csv',interactions)
        write_csv(source/'gold_baseline_length.csv',[{'baseline':'true_answer_onehot','value':None,'static_no_training_delta':True}])
        learning=[];accuracy=[]
        for job in jobs:
            path=source/'models'/job['run_id'];path.mkdir(parents=True)
            curves=[dict(step=k,train_loss=1/(k/1000),answer_accuracy=.1+.6*k/20000,numeric_accuracy=.7,eos_accuracy=1.) for k in range(1000,20001,1000)]
            (path/'completed.json').write_text(json.dumps(dict(job,step=20000,curve=curves)))
            learning.append(dict(job,parameters=3240448 if job['architecture']=='transformer' else 1660416,
                initial_validation_accuracy=.1,final_validation_accuracy=.7,gain=.6,learning_gate_passed=True))
            for condition in ['initial','trained']:
                for n in LENGTHS:
                    for action in ['identity','complement','inverse']:
                        accuracy.append(dict(architecture=job['architecture'],encoding=job['encoding'],task=job['task'],
                            condition=condition,length=n,action=action,full_vocab_accuracy=.1 if condition=='initial' else .7,numeric_accuracy=.75))
        write_csv(source/'source_learning.csv',learning);write_csv(source/'task_accuracy_length.csv',accuracy)
        evidence=export(source,output,fixture=True)
        text=(output/'report.html').read_text()
        assert 'Synthetic fixture only' in text and 'undefined' in text and 'ΓΔ' in text
        assert len(list((output/'plots').glob('*.png')))==40
        assert evidence['component_rows']==128 and evidence['interaction_rows']==64
        supplemental=load_csv(output/'length_baseline_accuracy_supplemental.csv')
        assert len(supplemental)==252 and all(int(r['training_majority_label'])==1 for r in supplemental)
        assert all(abs(float(r['length_majority_accuracy'])-.75)<1e-12 for r in supplemental if r['action']=='identity')
        assert len(load_csv(output/'model_minus_length_baseline_supplemental.csv'))==2016
        # Missing a condition must fail, rather than silently produce a report.
        write_csv(source/'geometry_summary.csv',rows[:-1])
        try:validate_results(source)
        except AssertionError:missing_rejected=True
        else:missing_rejected=False
        assert missing_rejected
        return dict(status='passed',synthetic_only=True,complete_condition_and_length_check=True,
                    delta_flip_without_Bt_flip_fixture=True,undefined_preserved=True,missing_condition_rejected=True,
                    train_only_length_baseline_tie_policy_verified=True,all_model_actions_baseline_reported=True,
                    rendered_plots=40,read_real_experiment_outputs=False,GPU_started=False)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--source',type=Path,default=DEFAULT_SOURCE)
    parser.add_argument('--output',type=Path,default=DEFAULT_OUTPUT);parser.add_argument('--fixture',action='store_true')
    args=parser.parse_args()
    if args.fixture:print(json.dumps(fixture(),ensure_ascii=False,indent=2))
    else:print(json.dumps(export(args.source,args.output),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
