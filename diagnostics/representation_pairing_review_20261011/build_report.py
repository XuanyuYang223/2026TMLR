"""Reader report from saved review tables. No new CKA, model or fitting."""
from __future__ import annotations
import os
os.environ['OPENBLAS_NUM_THREADS']='2'
os.environ['OMP_NUM_THREADS']='2'
import csv
import hashlib
import html
import json
import math
from collections import Counter
from pathlib import Path

HERE=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text())
def rows(n):
    with (HERE/n).open() as f:return list(csv.DictReader(f))
def num(x):
    if x in (None,'','None','NaN','nan'):return None
    y=float(x);return y if math.isfinite(y) else None
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def stats(rr,field):
    x=[num(r[field]) for r in rr];z=[v for v in x if v is not None]
    return dict(total=len(x),finite=len(z),undefined=len(x)-len(z),positive=sum(v>0 for v in z),negative=sum(v<0 for v in z),zero=sum(v==0 for v in z))
def table(rr,fields):
    return '<div class="table"><table><thead><tr>'+''.join('<th>'+html.escape(k)+'</th>' for k in fields)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(r.get(k,'')))+'</td>' for k in fields)+'</tr>' for r in rr)+'</tbody></table></div>'

def run():
    old=read(HERE/'existing_results_review.json');new=read(HERE/'analysis_audit.json')
    protected=read(HERE/'protection_end.json');reader=read(HERE/'result_reader_audit.json')
    assert old['status']==new['status']==protected['status']==reader['status']=='passed'
    p=read(HERE/'protocol.json');pairs=rows('pair_summary.csv');comparisons=rows('designated_category_comparisons.csv')
    change=rows('common_support_training_comparisons.csv');negative=rows('six_negative_bt.csv');w=rows('W_actual_summary.csv')
    relevant=[r for r in comparisons if r['phase']=='trained' and r['comparison_category']=='not_designated']
    summary={}
    for domain in p['systems']:
        summary[domain]={}
        for control in p['systems'][domain]['controls']:
            rr=[r for r in relevant if r['domain']==domain and r['control']==control]
            summary[domain][control]=stats(rr,'designated_minus_category_mean_state')
    known=rows('classification_manifest.csv')
    assert Counter(r['category'] for r in known)=={'designated':12,'not_designated':120}
    missing=[r for r in pairs if any(num(r['fixed_transform_score_'+phase]) is None for phase in ('initial','trained'))]
    # Plot fixed scores on the full frozen candidate grid; missing and excluded
    # cells remain masked. No best-model/task/layer subset is selected.
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    relation_names={'descent_inverse':'DESC / RECOIL','peak_complement':'PEAK / VALLEY','fixed_complement':'FIXED / ANTI',
                    'exceedance_inverse':'EXCEED / DEF','subsequence_complement':'LIS / LDS','run_complement':'INC / DEC RUN',
                    'record_complement':'MAX / MIN RECORD','decomposition_complement':'GLOBAL DESC / COMP'}
    perm_protocol=read(p['systems']['permworld']['protocol_path']); tasks=perm_protocol['label_task_names']; relations=[r['pair_id'] for r in perm_protocol['relations']]
    cmap=plt.get_cmap('viridis').copy();cmap.set_bad('#dedede')
    fig,axes=plt.subplots(4,2,figsize=(18,17),layout='constrained')
    for i,(a,e) in enumerate(( (a,e) for a in perm_protocol['architectures'] for e in perm_protocol['encodings'])):
        for j,control in enumerate(('raw_within_length','answer_strata')):
            ax=axes[i,j];m=np.full((8,16),np.nan);stars=[]
            for r in pairs:
                if (r['domain'],r['architecture'],r['encoding'],r['control'])!=('permworld',a,e,control):continue
                y,x=relations.index(r['relation']),tasks.index(r['right_task']);v=num(r['fixed_transform_score_trained'])
                if v is not None:m[y,x]=v
                if r['category']=='designated':stars.append((x,y))
            im=ax.imshow(m,cmap=cmap,vmin=0,vmax=1,aspect='auto');ax.set_title(f'{a} / {e} / {control}')
            ax.set_xticks(range(16),tasks,rotation=70,ha='right',fontsize=8);ax.set_yticks(range(8),[relation_names[x] for x in relations],fontsize=9)
            if stars:ax.scatter(*zip(*stars),marker='*',c='white',edgecolors='black',s=70,linewidths=.7)
    fig.colorbar(im,ax=axes,label='Fixed-transform trained ordinary CKA (each candidate uses all21 lengths)');fig.suptitle('Full PermWorld right-task control; white star = designated. Grey = excluded/undefined; see CSV for distinction.',fontsize=14)
    fig.savefig(HERE/'perm_all_candidate_scores.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(15,7),layout='constrained');configs=[(a,e) for a in perm_protocol['architectures'] for e in perm_protocol['encodings']]
    for j,control in enumerate(('raw_within_length','answer_strata')):
        m=np.full((8,4),np.nan)
        for r in relevant:
            if r['domain']=='permworld' and r['control']==control:m[relations.index(r['relation']),configs.index((r['architecture'],r['encoding']))]=num(r['designated_minus_category_mean_state'])
        im=axes[j].imshow(m,cmap='coolwarm',vmin=-1,vmax=1,aspect='auto');axes[j].set_title(control)
        axes[j].set_yticks(range(8),[relation_names[x] for x in relations]);axes[j].set_xticks(range(4),[f'{a}\n{e}' for a,e in configs],fontsize=8)
        for y,x in np.ndindex(m.shape):
            if np.isfinite(m[y,x]):axes[j].text(x,y,f'{m[y,x]:+.3f}',ha='center',va='center',fontsize=9)
    fig.colorbar(im,ax=axes,label='Designated score - finite mean of not_designated candidates');fig.suptitle('Same input transform, different task specialist. Descriptive, candidate-specific strata.')
    fig.savefig(HERE/'perm_designated_contrasts.png',dpi=180);plt.close(fig)
    graph_relations=[r['relation'] for r in read(p['systems']['graphworld']['protocol_path'])['relations']]
    fig,axes=plt.subplots(1,3,figsize=(17,5),layout='constrained')
    graph_tasks=['max_out','min_out','max_in','min_in']
    for j,control in enumerate(p['systems']['graphworld']['controls']):
        m=np.full((4,4),np.nan);stars=[]
        for r in pairs:
            if r['domain']!='graphworld' or r['control']!=control:continue
            y,x=graph_relations.index(r['relation']),graph_tasks.index(r['right_task']);v=num(r['fixed_transform_score_trained'])
            if v is not None:m[y,x]=v
            if r['category']=='designated':stars.append((x,y))
        im=axes[j].imshow(m,cmap=cmap,vmin=0,vmax=1);axes[j].set_title(control,fontsize=10)
        axes[j].set_xticks(range(4),graph_tasks,rotation=35,ha='right');axes[j].set_yticks(range(4),graph_relations)
        if stars:axes[j].scatter(*zip(*stars),marker='*',c='white',edgecolors='black',s=100)
    fig.colorbar(im,ax=axes,label='Fixed-transform trained CKA, equal n6/7/8');fig.suptitle('GraphWorld12 fixed candidate pairs; density is sensitivity, not a replacement primary control')
    fig.savefig(HERE/'graph_all_candidate_scores.png',dpi=180);plt.close(fig)
    lines=['PermWorld / GraphWorld supplementary review of current results (2026-10-11; saved-cache analysis, no new training/forward passes)','',
        'What this round resolved: verification of 64+4 genuine 20k endpoints; independent arithmetic checks of the previous 32 primary and six negative cells; 480+12 controls varying the right task under a fixed input transformation; and checks of how often original wrong transformations distinguish answer formulas on actual analysis inputs. All new cross-task metrics are exploratory analyses after inspecting previous results. The protocol was frozen before the new CKA computation, not retrospectively preregistered as confirmation.','',
        'Previous conclusion: in the answer-stratified, final-layer primary analysis with equal weighting by length, 26 of 32 PermWorld conditions exhibit positive pairing selectivity. Selectivity varies with encoding, layer, and control; training increments relative to matched initialization require separate interpretation. Correct mathematical correspondences often accompany geometric selectivity, but the current evidence alone does not identify a mathematical-structure mechanism.',
        'B0=S_correct,initial−S_wrong,initial; Bt=S_correct,trained−S_wrong,trained; delta_correct and delta_wrong are their respective training changes; Delta=Bt−B0=delta_correct−delta_wrong. For a fixed trained result, changing the initialization reference cannot alter Bt; it alters Delta and the increment interpretation. This round lacks multiple independent initializations to identify an initialization effect on Bt.',
        'Saved-table arithmetic yields primary Bt 26 positive/6 negative and Delta 18 positive/14 negative, specifically for answer strata, prefix_final final norm, ordinary CKA, and equal 1/21 weights over n10–30. All six negative cells use canonical-cycle encoding; see six_negative_bt.csv for both source accuracies/train-defined per-length modes/per-length signs. Bt directions agree across the three primary metrics in 32/32 cells and Delta in 30/32. The two exceptions are one-line GRU peaks/valleys and runs, with negative kNN Delta but positive CKA/Procrustes. This is agreement among geometric readings of shared activations, not independent replication or functional evidence.',
        'Weak tasks and capacity differences matter but do not explain everything: canonical-cycle GRU global-descents/components sources reach approximately 99.414%/99.991%, yet Bt is negative. GRU exceedances/deficiencies are nearly perfect in both encodings, with positive Bt in both but a Delta reversal: Gamma0=1.09407, Gamma_t=0.06432, Gamma_Delta=−1.02975. Exceeding the mode does not establish mastery.','',
        'New control: fix left task f, original correct T, encoding/configuration, inputs, and final extraction point, while varying right specialist g. This differs from fixing a task pair and changing correct/wrong input roles; do not combine counts. Fields are fixed_transform_score_initial/trained, not original Bt or Delta. Both controls retain original per-length weights: 1/21 for Perm n10–30 and 1/3 for Graph n6/7/8. Any undefined length makes its candidate aggregate undefined; do not fill with zero or reweight.',
        'The classification catalog asserts only proved designated formulas. other_known_exact is empty here and remains a separate empty category. not_designated means not listed for this correspondence, not mathematically unrelated. Candidates were not selected using CKA. Answer strata retain the original joint-answer key, including answers for the original right-task input roles, without computing new wrong-role CKA. Complete raw anchors are shared, but candidate answer masks and strata differ; group counts/residual df/IDs/key hashes appear in pair_masks.csv. Designated-minus-category-mean differences are descriptive; task identity, label distributions, mastery, and stratification support still vary, so these are not pure causal effects of mathematical relations.','',
        'Under a fixed transformation, does the designated correspondence exceed nondesignated candidates (trained designated score minus the mean finite complete-length score for that category; positive/negative/undefined are condition counts, not independent replications):']
    for domain,cc in summary.items():
        for control,s in cc.items():lines.append(f'{domain} / {control}: {s["positive"]} positive, {s["negative"]} negative, {s["zero"]} zero, {s["undefined"]} undefined; total{s["total"]} coupled conditions.')
    lines+=['Answer control changes the new observation: conditions exceeding the nondesignated-candidate mean change from raw30/32 to answer-stratified29/32 for Perm and from raw4/4 to answer-stratified3/4 for Graph; adding the original exact-edge-count control gives 2/4. Exceeding the category mean does not mean ranking first among all candidates.',
        'The three negative new answer-controlled Perm contrasts are canonical-cycle Transformer fixed/anti-fixed points, GRU fixed/anti-fixed points, and Transformer global-descents/components. They are distinct from the six original negative Bt cells. The new Graph reverse_min contrast changes from raw +0.31530 to answer-controlled −0.01319; complement_in changes from answer-controlled +0.00901 to density-sensitivity −0.00845. Near-zero signs do not establish significant positive or negative effects.',
        'Positive new Gcat counts are reported separately: Perm raw30/32, answer-controlled27/32; Graph raw4/4, answer-controlled2/4, density-sensitivity2/4. These are changes relative to initialization over common finite candidate sets, not original relation Delta or counts of additional independent evidence.']
    lines+=['','Save all candidate raw scores, category finite means/medians/counts, and initial/trained support lists for each condition. Genuine step0 parameters are shared within configuration, so raw initial right-task scores coincide. Answer-stratified initial differences can arise from candidate-specific grouping/support, not different initial weights. The new initialization-relative contrast uses only the intersection C of candidates finite in both phases: Gcat=(S_designated,t−mean_C S_g,t)−(S_designated,0−mean_C S_g,0), with C listed. This is not original Delta; each candidate phase must itself contain all 21/3 lengths.',
        f'All-candidate summaries: {len(pairs)} rows (including control versions); rows undefined in at least one phase: {len(missing)}; 480+12 counts candidate pairs, not additional models or independent samples.','',
        'Actual discrimination by wrong transformations: equal-weight accidental formula-preservation rates on all original analysis anchors are 56.771%, 54.818%, 56.771%, 59.701% for the four Graph relations; approximately 57.8%–61.6% under original answer control and 66.9%–76.5% under original density control. These inputs differ from the previous random-math51.9%–65.57% batch. Original W does not guarantee breaking the formula on every sample, limiting discrimination without invalidating the experiment. Retain all matching rows, W, and original results.',
        'All 1536 raw Boolean events for reverse_max and complement_out coincide (both are max_out(A)=max_out(W(A))); they are not independent relations, although answer selectors differ. Original inverse-mismatch preservation for Perm global-descents/components is approximately 77.0% under raw equal weights. W_actual_per_anchor.csv has 92160 relation–input rows but only 10752 Perm+1536 Graph distinct underlying objects; 372 original stratified sample-count checks passed. Report equal-weight rates separately from pooled numerators/denominators.','',
        'Independent checks and protection: previous-table arithmetic has maximum discrepancy 0 across 256022 per-length/12120 aggregate comparisons. New designated32+4 phase/control/length values directly match original correct scores. Another 40 independent Gram/covariance computations cover the first length of 10 fixed pairs. All use saved activations without repeated forward passes. result_reader_audit additionally checks a fixed small set of aggregates, categories, and differences from saved scores. Retain undefined entries and reasons; calibration refers to a null hypothesis, not a functional measure.',
        f'Start/end protection covers {protected["files"]} explicitly named files, all byte-identical, with unchanged Git HEAD and tracked worktree/index snapshots. These are scoped start/end checks, not proof that all files remained unchanged throughout execution. F17, previous frozen reports/protocols/caches, the current entry point, and previous archives are preserved. Engineering repairs are recorded separately; executed SHA values identify frozen scientific source.','',
        'Unresolved: each system still has one world and a shared genuine step0 per configuration, with unmatched capacities. Labels, confidence/prediction errors, and learning can still explain some geometric patterns. The new control supplies descriptive task-correspondence comparisons; it identifies neither an independent mathematical-structure mechanism nor functional benefits.','',
        'Limited replication proposal (not executed): nearly mastered GRU exceedances/deficiencies encoding pairs, GRU maxima/minima-record complement pairs, and Transformer fixed/anti-fixed-point pairs including canonical-cycle negative selectivity. Each independent world+true-step0 unit uses 12 unique specialists with paired encodings, fixed original 20k budget, and primary analysis; 1–2 units are cost scenarios only. Historical matching-source process references are approximately 2.727/5.455 hours including validation IO, not active GPU time or guarantees of future speed/power. Future cache costs remain Unknown and separate. No requirement makes all three domains positive; deliver weak tasks without automatic accuracy-target tuning. Graph relation extensions/GroupWorld/a third domain remain next-phase candidates only; see next_phase_candidates.txt. No data preparation or execution occurred.','',
        'Conclusion and stopping: this scoped review uses previous models, inputs, and saved activations, with no new training, optimizer, model, forward pass, probe, transfer, world, layer/metric expansion, upload, deletion, or movement. Stop after reporting and protection audits.']
    zh='In the answer-stratified, final-layer analysis with equal weighting across lengths, 26 of 32 PermWorld conditions exhibit positive pairing selectivity. Selectivity varies with encoding, layer and control, while training increments relative to matched saved initializations require separate interpretation. In the new fixed-transform, varying-right-task control, designated correspondences exceed the mean of candidates not designated for that correspondence in 29/32 PermWorld and 3/4 GraphWorld answer-stratified conditions. These conditions share models and inputs and are not independent replications. The new score contrasts are distinct from the original Bt and Delta and do not identify a pure causal effect of the mathematical relation. Exact mathematical correspondences often accompany geometric selectivity, but these exploratory observations alone do not identify a mathematical-structure mechanism or functional benefit.'
    en='In the answer-stratified, final-layer analysis with equal weighting across lengths, 26 of 32 PermWorld conditions exhibit positive pairing selectivity. Selectivity varies with encoding, layer and control, while training increments relative to matched saved initializations require separate interpretation. In the new fixed-transform, varying-right-task control, designated correspondences exceed the mean of candidates not designated for that correspondence in 29/32 PermWorld and 3/4 GraphWorld answer-stratified conditions. These conditions share models and inputs and are not independent replications. The new score contrasts are distinct from the original Bt and Delta and do not identify a pure causal effect of the mathematical relation. Exact mathematical correspondences often accompany geometric selectivity, but these exploratory observations alone do not identify a mathematical-structure mechanism or functional benefit.'
    (HERE/'results_paragraph_zh_en.txt').write_text('English translation\n'+zh+'\n\nEnglish\n'+en+'\n')
    text='\n'.join(lines)+'\n';(HERE/'reader_report.txt').write_text(text)
    extra=['<h2>Six negative Bt conditions (original primary analysis)</h2>',table(negative,['architecture','encoding','relation','B0','Bt','Delta','dc','dw','left_test_answer_accuracy','left_test_length_majority_accuracy','right_test_answer_accuracy','right_test_length_majority_accuracy','Bt_length_signs']),
        '<h2>New control: designated correspondence minus prespecified category by condition</h2>',table(comparisons,['domain','relation','architecture','encoding','control','phase','comparison_category','designated_score','finite_mean','finite_median','defined_count','total_count','designated_minus_category_mean_state','finite_support_right_tasks']),
        '<h2>New training changes on shared finite support (not original Delta)</h2>',table([r for r in change if r['category']!='designated'],['domain','relation','architecture','encoding','control','category','finite_common_pair_count','designated_minus_common_category_initial','designated_minus_common_category_trained','Gcat','common_support_right_tasks']),
        '<h2>All right tasks (raw values)</h2>']
    for domain in p['systems']:
        for control in p['systems'][domain]['controls']:
            extra.append('<details><summary>'+html.escape(domain+' / '+control)+'</summary>'+table([r for r in pairs if r['domain']==domain and r['control']==control],['relation','architecture','encoding','right_task','category','fixed_transform_score_initial','fixed_transform_score_trained','minimum_selected_count','maximum_selected_count','initial_undefined_lengths','trained_undefined_lengths'])+'</details>')
    images=['perm_all_candidate_scores.png','perm_designated_contrasts.png','graph_all_candidate_scores.png']
    csvs=sorted(x.name for x in HERE.glob('*.csv'))
    document='<!doctype html><html lang="en"><meta charset="utf-8"><title>PermWorld / GraphWorld supplementary review</title><style>body{font:16px/1.6 system-ui,sans-serif;margin:32px auto;max-width:1400px;padding:0 20px;color:#17252b}pre{white-space:pre-wrap;font:inherit}.table{overflow:auto;margin:16px 0}table{border-collapse:collapse;font-size:12px}th,td{padding:7px;border:1px solid #ddd;white-space:nowrap}th{background:#e9f0f3;position:sticky;top:0}img{max-width:100%}details{padding:12px;border:1px solid #ccc;margin:12px 0}</style><h1>PermWorld / GraphWorld supplementary review of current results</h1><pre>'+html.escape(text)+'</pre>'+''.join('<figure><img src="'+n+'"><figcaption>'+n+'</figcaption></figure>' for n in images)+''.join(extra)+'<h2>Complete CSV and reproduction entry points</h2><ul>'+''.join('<li><a href="'+n+'">'+n+'</a></li>' for n in csvs)+'</ul><p><a href="REPRODUCE.txt">REPRODUCE.txt</a> · <a href="protocol.json">Frozen protocol</a> · <a href="results_paragraph_zh_en.txt">Results statements</a> · <a href="replication_candidates.txt">Limited replication candidates</a></p></html>'
    (HERE/'reader_report.html').write_text(document)
    input_names=['existing_results_review.json','analysis_audit.json','protection_end.json','result_reader_audit.json','pair_summary.csv','designated_category_comparisons.csv','common_support_training_comparisons.csv','six_negative_bt.csv','W_actual_summary.csv','classification_manifest.csv','protocol.json','freeze.json']
    outputs=['reader_report.txt','reader_report.html','results_paragraph_zh_en.txt']+images
    audit=dict(status='completed_reader_report',writer_sha256=sha(__file__),summary=summary,
        candidate_summary_rows=len(pairs),candidate_phase_undefined_rows=len(missing),
        inputs_sha256={n:sha(HERE/n) for n in input_names},outputs_sha256={n:sha(HERE/n) for n in outputs},
        mathematical_metrics_recomputed=0,models=0,forward=0,training=0,old_writes=0,statistical_tests=0)
    (HERE/'report_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status=audit['status'],summary=summary),ensure_ascii=False))

if __name__=='__main__':run()
