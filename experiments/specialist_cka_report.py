"""Readable report of all frozen specialist controls and the small ablation."""
from collections import defaultdict
import csv
from datetime import datetime, timezone
import html
import json
from pathlib import Path

import numpy as np

from .analysis import write_csv
from .longrun_engine import atomic_json

ROOT=Path('results/specialist_cka_controls')
ADJUST=Path('results/specialist_input_adjustment')
ABLATION=Path('results/frozen_operator_constraint_ablation')
NAMES={'descent_inverse':'Descents--inverse descents (inversion)','peak_complement':'Peaks--valleys (complement)',
    'fixed_complement':'Fixed points--anti-fixed points (complement)','exceedance_inverse':'Exceedances--deficiencies (inversion)',
    'subsequence_complement':'LIS—LDS (complement)','run_complement':'Increasing--decreasing run (complement)',
    'record_complement':'Left-to-right maxima--minima (complement)','decomposition_complement':'Global descents--components (complement)'}


def read_csv(path):
    with path.open() as f:return list(csv.DictReader(f))


def stats(rows,key):
    values=[float(r[key]) for r in rows if r[key] not in ['',None]]
    if not values:return {'count':0,'mean':None,'positive':0}
    return {'count':len(values),'mean':float(np.mean(values)),'min':min(values),'max':max(values),'positive':sum(v>0 for v in values)}


def table(rows,columns):
    return '<table><thead><tr>'+''.join('<th>'+html.escape(title)+'</th>' for key,title in columns)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(row.get(key,'')))+'</td>' for key,title in columns)+'</tr>' for row in rows)+'</tbody></table>'


def fmt(x):return 'undefined' if x is None else f'{x:+.4f}'


def run():
    contrasts=read_csv(ROOT/'contrasts.csv');null=read_csv(ROOT/'conditional_null.csv')
    adjusted=read_csv(ADJUST/'per_source_contrasts.csv');ablation=read_csv(ABLATION/'per_source_results.csv')
    protocol=json.loads((ROOT/'protocol.json').read_text());sig=protocol['signature'];rels=[r['pair_id'] for r in sig['relations']]
    summary={'scope':sig['plan']['inference_scope'],'primary_position':'task-free ONE_END, final_norm',
        'test_anchors':2688,'original_source_checkpoints':48,'source_seeds':[17,42,101],'source_retraining':0,
        'core_protocol_registered_utc':protocol['registered_utc'],'controls':{},'initialization_increments':{},'conditional_permutation_controls':{}}
    for branch in ['prefix_final_norm','query_final_norm','output_prob','answer_onehot','answer_scalar']:
        summary['controls'][branch]={}
        for control in ['raw','length','answer_strata','confidence']:
            summary['controls'][branch][control]={cond:stats([r for r in contrasts if r['branch']==branch and r['control']==control and r['condition']==cond],'correct_minus_wrong') for cond in ['trained','matched_init','independent_init']}
    for branch in ['prefix_final_norm','query_final_norm']:
        summary['initialization_increments'][branch]={}
        for control in ['raw','length','answer_strata','confidence']:
            rows=[r for r in contrasts if r['branch']==branch and r['control']==control]
            for cond in ['matched_init','independent_init']:
                result=[]
                for x in rows:
                    if x['condition']!='trained':continue
                    y=next(y for y in rows if y['condition']==cond and y['relation']==x['relation'] and y['seed']==x['seed'])
                    result.append({'relation':x['relation'],'seed':x['seed'],'increment':float(x['correct_minus_wrong'])-float(y['correct_minus_wrong'])})
                summary['initialization_increments'][branch][control+'_'+cond]=stats(result,'increment')
        a=[r for r in null if r['branch']==branch and r['condition']=='trained']
        summary['conditional_permutation_controls'][branch]={'shuffle_contrast':stats(a,'shuffle_mean'),
            'excess_above_shuffle_mean':stats(a,'contrast_minus_shuffle_mean'),
            'above_all_32_shuffles':sum(float(x['observed_contrast'])>float(x['shuffle_max']) for x in a)}
    summary['supplemental_input_adjustment']={'scope':json.loads((ADJUST/'protocol.json').read_text())['signature']['plan']['registration_scope'],'controls':{}}
    for branch in ['prefix_final_norm','query_final_norm']:
        summary['supplemental_input_adjustment']['controls'][branch]={}
        for control in ['length','answer_strata']:
            selected=[r for r in adjusted if r['branch']==branch and r['control']==control]
            summary['supplemental_input_adjustment']['controls'][branch][control]={co:stats([r for r in selected if r['condition']==co],'input_adjusted_correct_minus_wrong') for co in ['trained','matched_init','independent_init']}
            summary['supplemental_input_adjustment']['controls'][branch][control]['ordinary_per_length']={co:stats([r for r in selected if r['condition']==co],'ordinary_correct_minus_wrong') for co in ['trained','matched_init','independent_init']}
            for cond in ['matched_init','independent_init']:
                diffs=[];ordinary_diffs=[]
                for x in selected:
                    if x['condition']=='trained':
                        y=next(y for y in selected if y['relation']==x['relation'] and y['seed']==x['seed'] and y['condition']==cond)
                        diffs.append({'increment':float(x['input_adjusted_correct_minus_wrong'])-float(y['input_adjusted_correct_minus_wrong'])})
                        ordinary_diffs.append({'increment':float(x['ordinary_correct_minus_wrong'])-float(y['ordinary_correct_minus_wrong'])})
                summary['supplemental_input_adjustment']['controls'][branch][control]['increment_vs_'+cond]=stats(diffs,'increment')
                summary['supplemental_input_adjustment']['controls'][branch][control]['ordinary_increment_vs_'+cond]=stats(ordinary_diffs,'increment')
    summary['constraint_ablation']={}
    for method in dict.fromkeys(x['method'] for x in ablation):
        summary['constraint_ablation'][method]={word:{key:stats([x for x in ablation if x['method']==method and x['word']==word],key) for key in ['answer_accuracy','both_correct','neither_correct','orientation_excess','displacement_nmse']} for word in ['ci','ici']}
        summary['constraint_ablation'][method]['word_average']={key:stats([x for x in ablation if x['method']==method],key)['mean'] for key in ['answer_accuracy','both_correct','displacement_nmse']}
    per_relation=[];per_seed=[]
    def cell(rel,seed,branch,control,condition='trained'):
        return next(x for x in contrasts if x['relation']==rel and x['seed']==str(seed) and x['branch']==branch and x['control']==control and x['condition']==condition)
    for rel in rels:
        audit=json.loads((ROOT/'evaluations'/f'{rel}_s17.json').read_text())['stratum_audit']
        raw=[float(cell(rel,s,'prefix_final_norm','raw')['correct_minus_wrong']) for s in [17,42,101]]
        matched=[float(cell(rel,s,'prefix_final_norm','answer_strata')['correct_minus_wrong']) for s in [17,42,101]]
        init=[float(cell(rel,s,'prefix_final_norm','answer_strata','matched_init')['correct_minus_wrong']) for s in [17,42,101]]
        out=[float(cell(rel,s,'output_prob','answer_strata')['correct_minus_wrong']) for s in [17,42,101]]
        adj=[float(x['input_adjusted_correct_minus_wrong']) for x in adjusted if x['relation']==rel and x['branch']=='prefix_final_norm' and x['control']=='answer_strata' and x['condition']=='trained']
        adj_init=[float(x['input_adjusted_correct_minus_wrong']) for x in adjusted if x['relation']==rel and x['branch']=='prefix_final_norm' and x['control']=='answer_strata' and x['condition']=='matched_init']
        per_relation.append({'relation':rel,'name':NAMES[rel],'matched_anchors':audit['retained'],
            'raw_prefix_mean':float(np.mean(raw)),'answer_matched_prefix_mean':float(np.mean(matched)),
            'answer_matched_prefix_seed17':matched[0],'answer_matched_prefix_seed42':matched[1],'answer_matched_prefix_seed101':matched[2],
            'answer_matched_increment_vs_matched_init_mean':float(np.mean(np.array(matched)-init)),
            'answer_matched_output_mean':float(np.mean(out)),'per_length_input_adjusted_prefix_mean':float(np.mean(adj)),
            'per_length_input_adjusted_increment_vs_matched_init_mean':float(np.mean(adj)-np.mean(adj_init))})
        for s in [17,42,101]:
            row={'relation':rel,'seed':s,'matched_anchors':audit['retained']}
            for view in ['prefix_final_norm','query_final_norm','output_prob']:
                for control in ['raw','length','answer_strata']:
                    row[view+'_'+control]=float(cell(rel,s,view,control)['correct_minus_wrong'])
            row['prefix_matched_init']=float(cell(rel,s,'prefix_final_norm','answer_strata','matched_init')['correct_minus_wrong'])
            row['prefix_independent_init']=float(cell(rel,s,'prefix_final_norm','answer_strata','independent_init')['correct_minus_wrong'])
            per_seed.append(row)
    write_csv(ROOT/'per_relation_summary.csv',per_relation);write_csv(ROOT/'per_relation_seed_summary.csv',per_seed)
    summary['per_relation']=per_relation
    accuracy=read_csv(ROOT/'source_accuracy.csv');task_accuracy=[]
    for task in dict.fromkeys(x['task'] for x in accuracy):
        values=[float(x['test_numeric_accuracy']) for x in accuracy if x['task']==task]
        task_accuracy.append({'task':task,'mean':float(np.mean(values)),'min':min(values),'max':max(values)})
    summary['source_task_accuracy']=task_accuracy
    atomic_json(ROOT/'summary.json',summary)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4.7),constrained_layout=True)
    for ax,branch,title in zip(axes,['prefix_final_norm','query_final_norm'],['Pooled lengths: task-free ONE_END','Pooled lengths: task query (=)']):
        for co,color,offset in [('trained','#2166ac',-.15),('matched_init','#b35806',0),('independent_init','#777777',.15)]:
            for j,rel in enumerate(rels):
                values=[float(x['correct_minus_wrong']) for x in contrasts if x['relation']==rel and x['branch']==branch and x['control']=='answer_strata' and x['condition']==co]
                ax.scatter([j+offset]*3,values,color=color,s=18,alpha=.65)
                ax.scatter(j+offset,np.mean(values),color=color,marker='D',s=35,label=co if j==0 else None)
        ax.axhline(0,color='black',lw=.8);ax.set_xticks(range(8),[r.replace('_','\n') for r in rels],fontsize=7)
        ax.set(title=title,ylabel='Correct - wrong CKA, common answer strata');ax.legend(fontsize=8)
    for ext in ['png','pdf']:fig.savefig(ROOT/f'answer_matched_cka.{ext}',dpi=180)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.7),constrained_layout=True)
    for ax,branch,title in zip(axes,['prefix_final_norm','query_final_norm'],['Per-length input adjustment: ONE_END','Per-length input adjustment: query']):
        for co,color,offset in [('trained','#2166ac',-.15),('matched_init','#b35806',0),('independent_init','#777777',.15)]:
            for j,rel in enumerate(rels):
                values=[float(x['input_adjusted_correct_minus_wrong']) for x in adjusted if x['relation']==rel and x['branch']==branch and x['control']=='answer_strata' and x['condition']==co]
                ax.scatter([j+offset]*3,values,color=color,s=18,alpha=.65)
                ax.scatter(j+offset,np.mean(values),color=color,marker='D',s=35,label=co if j==0 else None)
        ax.axhline(0,color='black',lw=.8);ax.set_xticks(range(8),[r.replace('_','\n') for r in rels],fontsize=7)
        ax.set(title=title,ylabel='Correct - wrong, mean of 21 lengths');ax.legend(fontsize=8)
    for ext in ['png','pdf']:fig.savefig(ROOT/f'per_length_input_adjustment.{ext}',dpi=180)
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4),constrained_layout=True)
    methods=list(summary['constraint_ablation'])
    for i,word in enumerate(['ci','ici']):
        for j,m in enumerate(methods):
            values=[100*float(x['answer_accuracy']) for x in ablation if x['method']==m and x['word']==word]
            ax.scatter([j+(i-.5)*.16]*3,values,color=['#2166ac','#b35806'][i],alpha=.65,label=word.upper() if j==0 else None)
    ax.axhline(50,color='black',ls='--',lw=.8);ax.set_xticks(range(5),[m.replace('_','\n') for m in methods],fontsize=8)
    ax.set(ylabel='Collision answer accuracy (%)',title='Frozen-source constraint ablation, 3 source seeds');ax.legend()
    for ext in ['png','pdf']:fig.savefig(ROOT/f'constraint_ablation.{ext}',dpi=180)
    plt.close(fig)
    relation_display=[{'name':r['name'],'anchors':r['matched_anchors'],'raw':fmt(r['raw_prefix_mean']),
        'seeds':' / '.join(fmt(r[f'answer_matched_prefix_seed{s}']) for s in [17,42,101]),
        'increment':fmt(r['answer_matched_increment_vs_matched_init_mean']),'output':fmt(r['answer_matched_output_mean']),
        'adjust':fmt(r['per_length_input_adjusted_prefix_mean']),
        'adjust_increment':fmt(r['per_length_input_adjusted_increment_vs_matched_init_mean'])} for r in per_relation]
    global_display=[]
    for branch,name in [('prefix_final_norm','Pre-task  ONE_END'),('query_final_norm','Query position '),('output_prob','31 -class numeric output'),('answer_onehot','Gold answers onehot')]:
        for control,cname in [('raw','original '),('length','Length-centered'),('answer_strata','Joint-answer strata'),('confidence','FIT Confidence regression')]:
            val=summary['controls'][branch][control]['trained']
            if val['count'] or control=='answer_strata':global_display.append({'branch':name,'control':cname,'mean':fmt(val['mean']),'positive':f"{val['positive']}/{val['count']}" if val['count'] else ' is constant; CKA undefined'})
    ab_display=[]
    for method in methods:
        row={'method':method}
        for word in ['ci','ici']:
            row[word]=f"{100*summary['constraint_ablation'][method][word]['answer_accuracy']['mean']:.2f}%"
            row[word+'_both']=f"{100*summary['constraint_ablation'][method][word]['both_correct']['mean']:.2f}%"
        ab_display.append(row)
    seed_display=[{'relation':NAMES[r['relation']],'seed':r['seed'],'prefix_raw':fmt(r['prefix_final_norm_raw']),
        'prefix_length':fmt(r['prefix_final_norm_length']),'prefix_answer':fmt(r['prefix_final_norm_answer_strata']),
        'query_answer':fmt(r['query_final_norm_answer_strata']),'output_answer':fmt(r['output_prob_answer_strata']),
        'init':fmt(r['prefix_matched_init'])} for r in per_seed]
    adj=summary['supplemental_input_adjustment']['controls']['prefix_final_norm']['answer_strata']['trained']
    length_comparison=[]
    for branch,label in [('prefix_final_norm','Pre-task  ONE_END'),('query_final_norm','Query position ')]:
        values=summary['supplemental_input_adjustment']['controls'][branch]['answer_strata']
        for kind,title in [('ordinary_per_length','Per-length ordinary CKA'),('adjusted','Per-length input-adjusted CKA')]:
            co=values[kind] if kind=='ordinary_per_length' else values
            inc=values['ordinary_increment_vs_matched_init'] if kind=='ordinary_per_length' else values['increment_vs_matched_init']
            length_comparison.append({'branch':label,'method':title,'trained':fmt(co['trained']['mean']),
                'matched':fmt(co['matched_init']['mean']),'independent':fmt(co['independent_init']['mean']),
                'increment':fmt(inc['mean']),'positive':f"{inc['positive']}/{inc['count']}"})
    summary['interpretation']={'original_correct_transform_signal_reproduced':True,
        'pooled_answer_matched_signal_positive':True,
        'uniform_training_added_relation_geometry_supported':False,
        'reason':'Per-length ordinary and invariant-input-adjusted CKA both exceed matched initialization in only 6/24 answer-matched cells, all six from the two inverse relations. Complement signals also occur strongly in random networks. Pooled length centering does not equal per-length CKA averaging.',
        'output_independent_algebraic_mechanism_confirmed':False,'stable_hidden_composition_confirmed':False}
    atomic_json(ROOT/'summary.json',summary)
    report=f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>Original cross-model  CKA: new tests and answer controls</title>
<style>body{{max-width:1250px;margin:36px auto;padding:0 22px;font:16px/1.7 system-ui;color:#202733}}table{{border-collapse:collapse;width:100%;font-size:13px;margin:18px 0}}th,td{{border:1px solid #d7dde5;padding:7px;text-align:left}}th{{background:#edf2f7}}img{{max-width:100%}}.note{{padding:14px;background:#f1f5f9;border-left:4px solid #64748b}}a{{color:#2166ac}}</style>
<h1>Return to original cross-model  CKA: new tests and answer controls</h1>
<p><strong>original  CKA The phenomenon replicates, but initialization/per-length controls expose important boundaries.</strong>Eight relationships, three seeds, 48  existing specialists; no source retraining. Pooled-length correct-transform advantages are  24/24  positive, including after answer matching; per-length computation leaves only  6/24  answer-matched trained cells exceeding initialization, all from the two inversion relationships. The six complement relationships align more strongly even in random networks, precluding a general training-learned-relationship interpretation.</p>
<p>The main quantity is  ΔCKA = CKA(correct transformation) − CKA(original wrong transformation). Original wrong comparators remain: complement for inversion relationships, inversion for complement relationships; untransformed inputs are separate. Conditions and weights are identical except mathematical alignment.</p>
<p>Core comparison rules were saved at  {protocol['registered_utc']} , before new data, representations, and  CKA. New  2,688  test anchors plus  672 FIT  and  336  validation anchors. The original  1,600  ten-thousand parent inputs were scanned; all eight orbit states of candidates exclude the original data and earlier local inputs. After initial registration but before any selection/inference, the reader was corrected for compact  JSON  format; the failed registration is retained and the protocol was saved again.</p>
<h2>For every relationship and all seeds/</h2>
<p>joint strata use length, untransformed left answer, and the three right-side  e/C/I  answers. Retain strata with at least three anchors and subtract means within each branch/stratum; all conditions share retained rows. Deterministic gold encodings become constant within strata, so conditional  CKA undefined. The last column separately adds mean per-length input adjustment and cannot be subtracted directly from  pooled CKA  as the same statistic.</p>
{table(relation_display,[('name','Relationship'),('anchors','Matched anchors'),('raw','original  ONE_END Δ Mean'),('seeds','Pooled-length answer-matched Δ:17 / 42 / 101'),('increment','Pooled-length increment over initialization'),('output','Matched output Δ Mean'),('adjust','Per-length input-adjusted Δ'),('adjust_increment','Per-length adjusted increment over initialization')])}
<img src="answer_matched_cka.png"><p><a href="answer_matched_cka.pdf">Download PDF Figure</a> · <a href="per_relation_seed_summary.csv">Per relationship × seed/ CSV</a> · <a href="contrasts.csv">All branches and control comparisons</a> · <a href="scores.csv">Three transformations'  CKA original value</a></p>
{table(seed_display,[('relation','Relationship'),('seed','seed/'),('prefix_raw','ONE_END original  Δ'),('prefix_length','Length-centered Δ'),('prefix_answer','Answer-matched Δ'),('query_answer','Query-matched Δ'),('output_answer','Output-matched Δ'),('init','Same-seed initial matched Δ')])}
<h2>Where is the signal?</h2>
{table(global_display,[('branch','Branch'),('control','Control'),('mean','Δ Mean'),('positive','Positive cells')])}
<p>After answer matching, pooled-length pre-task representations have mean  Δ  is  {summary['controls']['prefix_final_norm']['answer_strata']['trained']['mean']:.4f}, with  23/24  exceeding same-seed initialization; independent-seed initials also give  23/24. The trained model's  24/24  conditional comparisons all exceed the prescribed  32  within-group shuffles. Shuffling preserves gold labels/strata but disrupts input correspondence. The signal is not pure deterministic gold encoding, yet learned relation structure and natural input geometry remain indistinguishable. Per-length controls below change the initialization conclusion. These cells share tasks, inputs, and seeds, rather than  24  independent trials.</p>
<p>Gold answers and model-output distributions alone reproduce large raw correct-transform advantages. Thus raw  CKA  cannot establish an internal algebraic mechanism. Conditional output geometry may retain confidence/other information; hidden  Δ  exceeding output  Δ  does not establish output independence. Fixed  FIT-only confidence regression is supplementary, not a causal mediation decomposition. Raw and matched analyses differ in sample support/centering, so their decrease does not quantify the fraction explained by answers.</p>
<h2>Input-structure controls and supplementary deconfounding</h2>
<p>Input-encoding baselines show  CKA  encoding sensitivity: length-centered numeric-position inputs for complement give  CKA=1, while permutation-matrix inputs for inversion give  CKA=1. This motivates initialization/within-group shuffling controls. Full input baselines:  <a href="input_baselines.csv">CSV</a>.</p>
<p>A linear positional permutation-matrix input kernel is also used per length. Its input  Gram  in  e/C/I  is identical across conditions, including after joint-answer centering. Following  <a href="https://proceedings.neurips.cc/paper_files/paper/2022/file/79cbf4f96c2bcc67267421154da689dd-Paper-Conference.pdf">Cui  et al. 2022</a>  regression and  PSD projection, this kernel is removed and every length/seed reported. The supplement was saved after core  CKA /input baselines but before adjusted scores, using centered kernels. It neither reproduces every preprocessing step of that paper nor excludes all input confounds.</p>
<p>Under joint-answer control, pre-task input-adjusted  Δ  has per-length mean  {adj['mean']:.4f}, positive  {adj['positive']}/{adj['count']}; see  <a href="../specialist_input_adjustment/per_source_contrasts.csv">Per-relationship/seed</a>, <a href="../specialist_input_adjustment/per_length_scores.csv">Per-length raw values</a>, <a href="../specialist_input_adjustment/protocol.json">Supplementary protocol</a>.</p>
{table(length_comparison,[('branch','Position'),('method','Statistics after answer matching'),('trained','training Δ Mean'),('matched','Same-seed initialization Δ'),('independent','Independent initialization Δ'),('increment','Trained increment over same-seed initialization'),('positive','Positive increments')])}
<img src="per_length_input_adjustment.png"><p><a href="per_length_input_adjustment.pdf">Per-length controls PDF Figure</a></p>
<p><strong>Per-length ordinary  CKA  and input-adjusted  CKA  both have only  6/24  exceeding initialization.</strong>These six cells are exactly the three seeds of descents--inverse descents and exceedances--deficiencies; all six complement relationships are below initialization. Independent initials yield the same  6/24. Most change comes from computing per length then averaging, not kernel regression: pooled  CKA after subtracting length means differs from averaging each length's  CKA . Therefore pooled-length  23/24 alone does not imply general training gains. Inversion is worth pursuing, but fixed-transform other-task comparisons and fuller output controls are needed for relationship specificity.</p>
<h2>Frozen-source composition-constraint ablation</h2>
<p>Three additional prespecified source seeds  8111/9127/10141 freeze the same source model/original numeric readout. Fitting uses only  e↔C  and  e↔I  known edges; hyperparameters depend only on visible-state validation error, with 400  fixed updates. New collision tests contain  1,344  pairs with identical three known answers and different CI/ICI  hidden answers. Operators receive norm constraints, affine round-trip constraints, or both. Original operators and equal-budget unconstrained refits are retained to separate constraints from extra optimization. Inspired by  <a href="https://proceedings.mlr.press/v119/azencot20a/azencot20a.pdf">Azencot  et al. 2020</a>  consistency, directly constrain the  C/I  involution relationships without training a  Koopman  autoencoder.</p>
{table(ab_display,[('method','Method'),('ci','CI Accuracy'),('ici','ICI Accuracy'),('ci_both','CI Both correct'),('ici_both','ICI Both correct')])}
<img src="constraint_ablation.png"><p>Both-correct means both inputs in a collision pair are correct; CI/ICI  are separate computational paths, not the two pair members.50%  reference applies only to deterministic predictors of the known scalar answers under this collision selection, not a neural-information bound. Round-trip constraints give only a tiny increment over equal-budget unconstrained refits; norm/combined constraints perform worse under the fixed selection rule. Conclusions apply only to composition generalization with supplied known constraints.</p>
<p><a href="../frozen_operator_constraint_ablation/per_source_results.csv">All seeds × Method × CI/ICI</a> · <a href="../frozen_operator_constraint_ablation/protocol.json">Ablation protocol</a> · <a href="constraint_ablation.pdf">PDF Figure</a></p>
<h2>Proficiency, evidence limits, and verification</h2>
<p>Source-task proficiency varies, especially low long-permutation accuracy for  peaks  and  recoils ; all are retained without result-based selection. Task means describe extracted transformation/seed conditions, not proficiency-matched balanced controls. Per-task/condition accuracies:  <a href="source_accuracy.csv">CSV</a>.</p>
{table([{'task':r['task'],'mean':f"{100*r['mean']:.2f}%",'min':f"{100*r['min']:.2f}%",'max':f"{100*r['max']:.2f}%"} for r in task_accuracy],[('task','Source task'),('mean','Mean numeric accuracy'),('min','Minimum condition'),('max','Maximum condition')])}
<div class="note">Supported conclusion: correct mathematical transforms can increase cross-model  CKA, depending on relation family, input geometry, and length aggregation. Both inversion relationships show consistent per-length, answer-matched, initialization-relative increments; complement raw positives do not identify training-formed structure. This establishes neither full algebraic representations, output-independent mechanisms, nor stable unseen-composition inference. All results are conditional on these eight relationships, three seeds, existing data world, and specified analyses.</div>
<p>Independent sample  Gram  formulas verify  1,944  core  CKA  values; raw-forward spot checks cover all prefix layers/numeric outputs; separately checked ablation  30  endpoints, all  48  candidate visible-validation scores, and frozen readouts. Check counts describe coverage, not independent effect replication.<a href="verification.json">CKA Verification</a> · <a href="extraction_reference_check.json">Raw-forward comparison</a> · <a href="../frozen_operator_constraint_ablation/verification.json">Ablation verification</a> · <a href="../specialist_input_adjustment/verification.json">Input-adjustment verification</a> · <a href="summary.json">Machine-readable summary</a> · <a href="protocol.json">Core protocol</a> · <a href="completion.json">Completion and hash records</a></p>
</html>'''
    (ROOT/'report.html').write_text(report)
    print(json.dumps({'report':str(ROOT/'report.html'),'primary_answer_matched':summary['controls']['prefix_final_norm']['answer_strata']['trained'],
        'input_adjusted':adj,'ablation_mean':{m:summary['constraint_ablation'][m]['word_average'] for m in methods}},indent=2))


if __name__=='__main__':run()
