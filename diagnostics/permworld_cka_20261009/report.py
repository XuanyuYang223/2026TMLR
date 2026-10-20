"""Readable bounded diagnostic report, figures, and reproduction manifest."""
from datetime import datetime, timezone
import html
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import ROOT, WORKSPACE, load_csv, write_csv, write_json, sha

NAMES={'descent_inverse':'Descents--inverse descents (inversion)','peak_complement':'Peaks--valleys (complement)',
 'fixed_complement':'Fixed points--anti-fixed points (complement)','exceedance_inverse':'Exceedances--deficiencies (inversion)',
 'subsequence_complement':'LIS—LDS (complement)','run_complement':'Longest increasing--decreasing run (complement)',
 'record_complement':'Left-to-right maxima--minima (complement)','decomposition_complement':'Global descents--components (complement)'}


def table(headers,rows):
    return '<table><thead><tr>'+''.join('<th>'+html.escape(str(h))+'</th>' for h in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+str(c)+'</td>' for c in row)+'</tr>' for row in rows)+'</tbody></table>'


def fmt(x,digits=3):return f'{float(x):+.{digits}f}' if x not in ['',None] else 'undefined'


def link(file,label=None):return f'<a href="{file}">{html.escape(label or file)}</a>'


def run():
    summary=load_csv(ROOT/'relation_summary.csv');src=load_csv(ROOT/'relation_seed_summary.csv')
    cells=load_csv(ROOT/'relation_length_cka.csv');lor=load_csv(ROOT/'leave_one_relation.csv')
    k=load_csv(ROOT/'k_summary.csv');kpool=load_csv(ROOT/'k_pooled_control.csv')
    audit=json.loads((ROOT/'initialization_audit.json').read_text());ver=json.loads((ROOT/'verification.json').read_text())
    inventory=json.loads((ROOT/'artifact_inventory.json').read_text())
    relations=list(NAMES)
    def sel(rel,typ='same_seed',control='answer_strata',metric='ordinary'):
        return next(r for r in summary if (r['relation'],r['seed_pair_type'],r['control'],r['metric'])==(rel,typ,control,metric))
    # Report finite-sample resampling shifts explicitly. These are distributions, not nominal population CIs.
    b=np.load(ROOT/'input_bootstrap_delta.npz');resampling=[]
    for rel in relations:
        for typ,ids in [('same_seed',[0,4,8]),('cross_seed',[1,2,3,5,6,7])]:
            ds=b[rel][ids].mean(0);obs=float(sel(rel,typ,'length')['delta_mean'])
            resampling.append(dict(family='specialist',relation_or_replicate=rel,k='',metric='delta',seed_pair_type=typ,
                observed=obs,resampling_mean=float(ds.mean()),resampling_shift=float(ds.mean()-obs),resampling_sd=float(ds.std(ddof=1)),
                resampling_quantile_025=float(np.quantile(ds,.025)),resampling_quantile_975=float(np.quantile(ds,.975)),
                interpretation='descriptive paired input-resampling distribution; finite-sample shift is not removed and percentiles are not a nominal population CI'))
    kb=np.load(ROOT/'k_input_bootstrap_draws.npz')
    for r in k:
        draws=kb[f'{r["replicate"]}_k{r["k"]}']
        for metric,ds in [('intermodel_cka',draws[0]),('mean_self_initial_cka',(draws[1]+draws[2])/2)]:
            obs=float(r[metric]);resampling.append(dict(family='k_series',relation_or_replicate=r['replicate'],k=r['k'],metric=metric,
                seed_pair_type='',observed=obs,resampling_mean=float(ds.mean()),resampling_shift=float(ds.mean()-obs),
                resampling_sd=float(ds.std(ddof=1)),resampling_quantile_025=float(np.quantile(ds,.025)),resampling_quantile_975=float(np.quantile(ds,.975)),
                interpretation='descriptive paired input-resampling distribution; no source-seed/world population inference'))
    write_csv(ROOT/'resampling_diagnostics.csv',resampling)
    families=[]
    for typ in ['same_seed','cross_seed']:
        for control in ['length','answer_strata']:
            for metric in ['ordinary','debiased','input_adjusted']:
                for family in ['inverse','complement']:
                    rs=[r for r in summary if r['seed_pair_type']==typ and r['control']==control and r['metric']==metric and r['relation'].endswith(family)]
                    families.append(dict(seed_pair_type=typ,control=control,metric=metric,transform_family=family,fixed_relations=len(rs),
                        delta_mean=float(np.mean([float(r['delta_mean']) for r in rs])),delta_median=float(np.median([float(r['delta_mean']) for r in rs]))))
    write_csv(ROOT/'transform_family_summary.csv',families)
    # The points are seed summaries, not confidence intervals or independent relation replicates.
    fig,axs=plt.subplots(1,2,figsize=(12,4.8),sharey=True)
    for ax,control,title in zip(axs,['length','answer_strata'],['Within length, original inputs','Within length, common answer strata']):
        for typ,offset,color in [('same_seed',-.12,'#1565c0'),('cross_seed',.12,'#d55e00')]:
            rs=[sel(rel,typ,control) for rel in relations]
            mean=np.array([float(r['delta_mean']) for r in rs]);lo=np.array([float(r['seed_pair_min']) for r in rs]);hi=np.array([float(r['seed_pair_max']) for r in rs])
            ax.errorbar(np.arange(8)+offset,mean,yerr=[mean-lo,hi-mean],fmt='o',capsize=3,color=color,label=typ.replace('_',' '))
        ax.axhline(0,color='black',lw=.8);ax.set_xticks(range(8),['descent I','peak C','fixed C','exceed I','LIS C','run C','record C','decomp C'],rotation=50,ha='right')
        ax.set_title(title);ax.grid(axis='y',alpha=.2)
    axs[0].set_ylabel('Delta of correct-minus-wrong linear CKA');axs[1].legend(fontsize=9)
    fig.suptitle('Mean over 21 lengths; bars show fixed seed-pair range, not confidence intervals',fontsize=10)
    fig.tight_layout();fig.savefig(ROOT/'relation_delta.png',dpi=180);fig.savefig(ROOT/'relation_delta.pdf');plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(13,4.7),sharey=True)
    for ax,control,title in zip(axs,['length','answer_strata'],['Original inputs','Common answer strata']):
        vals=np.array([[np.mean([float(r['delta']) for r in cells if r['relation']==rel and r['control']==control and r['metric']=='ordinary' and r['seed_pair_type']=='same_seed' and int(r['length'])==n]) for n in range(10,31)] for rel in relations])
        im=ax.imshow(vals,aspect='auto',cmap='RdBu_r',vmin=-1.6,vmax=1.6);ax.set_xticks(range(0,21,2),range(10,31,2));ax.set_xlabel('Permutation length');ax.set_title(title)
        ax.set_yticks(range(8),['descent I','peak C','fixed C','exceed I','LIS C','run C','record C','decomp C'])
    fig.colorbar(im,ax=axs.ravel().tolist(),label='Delta, fixed same-seed mean');fig.subplots_adjust(right=.85,wspace=.12)
    fig.savefig(ROOT/'relation_length_heatmap.png',dpi=180,bbox_inches='tight');fig.savefig(ROOT/'relation_length_heatmap.pdf',bbox_inches='tight');plt.close(fig)
    # Replot the requested k curves without confusing resampling percentiles with confidence bands.
    fig,axs=plt.subplots(1,2,figsize=(12,4.5),sharex=True,sharey=True)
    colors=plt.cm.tab10(np.arange(5))
    for i,rep in enumerate(['r0','r1','r2','r3','r4']):
        rs=sorted([r for r in k if r['replicate']==rep],key=lambda r:int(r['k']))
        label=f'{rep.upper()}, seed {rs[0]["seed"]}'+(' (subset)' if i>=3 else '')
        for ax,metric in zip(axs,['intermodel_cka','mean_self_initial_cka']):
            ax.plot([int(r['k']) for r in rs],[float(r[metric]) for r in rs],marker='o',linestyle='--' if i>=3 else '-',color=colors[i],label=label)
    axs[0].set_title('Between task pools A and B');axs[1].set_title('Retention relative to own rebuilt initialization')
    for ax in axs:ax.set_xscale('log',base=2);ax.set_xticks([1,2,4,8,16],[1,2,4,8,16]);ax.set_ylim(0,1);ax.set_xlabel('Number of tasks k');ax.grid(alpha=.2)
    axs[0].set_ylabel('Mean of 29 within-length linear CKA values');axs[1].legend(fontsize=8)
    fig.suptitle('Original initialization identity unverified; R3/R4 are fixed-seed subset sensitivity',fontsize=10)
    fig.tight_layout();fig.savefig(ROOT/'k_drift.png',dpi=180);fig.savefig(ROOT/'k_drift.pdf');plt.close(fig)
    relationtable=[]
    for rel in relations:
        raw=sel(rel,control='length');ans=sel(rel);cross=sel(rel,'cross_seed')
        relationtable.append([html.escape(NAMES[rel]),fmt(raw['delta_mean']),fmt(ans['delta_mean']),fmt(ans['delta_median']),
            f"[{fmt(ans['seed_pair_min'])}, {fmt(ans['seed_pair_max'])}]",fmt(cross['delta_mean']),
            f"{ans['min_positive_lengths']}–{ans['max_positive_lengths']}/21"])
    seedtable=[]
    for rel in relations:
        vals=[next(r for r in src if r['relation']==rel and r['left_seed']==str(s) and r['right_seed']==str(s) and r['control']=='answer_strata' and r['metric']=='ordinary') for s in [17,42,101]]
        seedtable.append([html.escape(NAMES[rel])]+[fmt(r['delta']) for r in vals])
    sensitivity=[]
    for rel in relations:
        sensitivity.append([html.escape(NAMES[rel])]+[fmt(sel(rel,metric=m)['delta_mean']) for m in ['ordinary','debiased','input_adjusted']])
    rawtable=[]
    for rel in relations:
        r=sel(rel)
        rawtable.append([html.escape(NAMES[rel])]+[f'{float(r[f]):.3f}' for f in ['trained_correct','trained_wrong','trained_identity','init_correct','init_wrong','init_identity']])
    ktable=[]
    for rep in ['r0','r1','r2','r3','r4']:
        rs=sorted([r for r in k if r['replicate']==rep],key=lambda r:int(r['k']))
        ktable.append([rep.upper()+' / '+rs[0]['seed']]+[' / '.join(f'{float(r[f]):.3f}' for f in ['intermodel_cka','mean_self_initial_cka']) for r in rs])
    lo= [r for r in lor if r['seed_pair_type']=='same_seed' and r['metric']=='ordinary']
    lotable=[[NAMES[r['omitted_relation']],r['control'],fmt(r['delta_mean']),fmt(r['delta_median'])] for r in lo]
    loop=load_csv(ROOT/'leave_one_length.csv')
    inverse_ranges={rel:(min(float(r['delta']) for r in loop if r['relation']==rel and r['control']=='answer_strata'),max(float(r['delta']) for r in loop if r['relation']==rel and r['control']=='answer_strata')) for rel in ['descent_inverse','exceedance_inverse']}
    macro=load_csv(ROOT/'macro_summary.csv')
    paragraphs=[]
    paragraphs.append('<h1>PermWorld: existing-checkpoint  CKA diagnosis</h1><p>exploratory diagnosis · 2026-10-09 · This round is complete and stopped; no new not_started training.</p>')
    paragraphs.append('<p><strong>Main findings: </strong>Relative to references reconstructed with the original initialization procedure, the increments in the correct-minus-wrong contrast for both inversion relationships are positive across all available seed pairings, all 21 lengths, and all three fixed metrics. The six complement relationships do not show equally stable increments after answer control. This is evidence about representation geometry, not identification of an algebraic computation mechanism.</p>')
    paragraphs.append('<p><strong>Initialization limitation: </strong>98 logical models correspond to 94 distinct checkpoints, all classified as "Reconstructed with the original initialization procedure; original weights not directly verifiable”;0 have directly verifiable original initial weights; 0 were discarded as unreconstructible. Neither step-0 weights nor initial-weight hashes were found. Historical analysis code, model structure, 188-token vocabulary, standard initialization, and RNG call order were checked; the original training commit and PyTorch versions were not recorded. Same-seed or cache reproduction must not be described as verified initial-weight identity.</p>')
    paragraphs.append('<h2>Inventory and reuse scope</h2>')
    paragraphs.append(table(['Existing artifacts','Previously completed analysis','Treatment in this round'],[
      ['neurips Single-property models','16 tasks x 3 seeds, 48 20,000-step checkpoints; original eight-relationship CKA','Reuse weights, configurations, checkpoint SHA and completion markers; no retraining'],
      ['specialist_cka_controls','Fixed new inputs; pooled CKA after controlling length means, joint-answer strata, reconstructed initialization/Independent random references, output /answer geometry, 32 conditional permutations, supplementary confidence control','Reuse all data and prefix caches; pooled-length statistics do not replace the per-length primary analysis'],
      ['specialist_input_adjustment','21 Ordinary /input adjustment CKA,3 same-seed answer controls and initialization comparisons','Directly reuse same-seed values; maximum recomputation difference 2.22×10⁻¹⁶'],
      ['Property32 Task-count series','R0–R4, k=1/2/4/8/16, two pools totaling 50 models; 4,096 fixed prefixes; original pooled-length A/B CKA;seed17/42 random activations','Reuse all trained caches; forward extraction only for missing missing existing seed101 initialization references'],
      ['Added in this round','Matching cross-seed references, debiased and per-length permutations, shared-input resampling, self-initialization drift','All saved separately; F17 and the original CKA directory are read-only']]))
    paragraphs.append('<p>146 sampled cache checks agreed with original forwards before the task token: each specialist condition uses 10/20/30 lengths, Property32 uses 2/10/20/30 lengths; maximum absolute error within 8.89×10⁻⁶. This verifies consistency of the current program and cached extraction points, not historical original-weight identity.'+link('initialization_audit.json','Initialization and missing artifact audit')+';'+link('initialization_history.json','Historical source code/RNG audit')+';'+link('artifact_inventory.json','Version and data provenance')+';'+link('model_inventory.csv','Per-model checkpoint hashes')+'.</p>')
    paragraphs.append('<h2>Eight relationships: training increments under fixed length weights</h2><p>The state is final_norm at layer &lt;ONE_END&gt;, before the task prompt. The same 2,688 test orbit anchors, lengths 10–30 each 128; compute within each length and use fixed weight 1/21. Answer matching uses the original joint-label strata, retaining strata with at least 3 samples. Summarize the three same-seed and six ordered cross-seed pairs separately. Each cell subtracts its corresponding initial-model-pair contrast, not a shared constant.</p>')
    paragraphs.append('<p>Δ=(correct-minus-wrong)<sub>trained</sub>−(correct-minus-wrong)<sub>matching initialization</sub>. Delta is the difference between two CKA contrasts and can exceed 1; it is not an accuracy-percentage-point difference or an equal-sized increase of correct-pair CKA itself.</p>')
    paragraphs.append(table(['Relationship','Raw-input mean Delta','Answer-controlled mean Delta','Answer-controlled seed median','Same-seed Delta range','Cross-seed mean Delta','Same-seed positive-length count'],relationtable))
    paragraphs.append('<p>Both inversion relationships are positive across 9 pairings x 21 lengths under answer control. Same-seed Delta is approximately +0.589 for descents--inverse descents and +0.641 for exceedances--deficiencies. Raw-input longest increasing/decreasing-run +0.371 becomes −0.332 after answer control; peaks/valleys and LIS/LDS also show seed-dependent raw-input results. The complement relationships cannot all be summarized as obtaining training increments.</p>')
    paragraphs.append('<p>"Answer control" subtracts joint-label-stratum means. It does not remove all output information or make every same-label sample indistinguishable.43–128 anchors per length are retained; model probabilities, confidence, and input-related information may remain.</p>')
    paragraphs.append('<img src="relation_delta.png" alt="Eight-relationship training increments and seed-pair ranges">')
    paragraphs.append('<details><summary>Three same-seed results (all relationships)</summary>'+table(['Relationship','seed17','seed42','seed101'],seedtable)+'</details>')
    paragraphs.append('<details><summary>Retained raw CKA (answer-controlled, same-seed mean)</summary>'+table(['Relationship','Trained correct','Trained wrong','Trained identity','Initial correct','Initial wrong','Initial identity'],rawtable)+'</details>')
    paragraphs.append('<p><strong>Interpret the increment carefully: </strong>For descents--inverse descents, correct-pair CKA is 0.145, versus reconstructed initialization 0.179; wrong-pair CKA instead changes from 0.685 to 0.062. The positive Delta mainly describes a change in the correct/wrong relative contrast, not a general absolute increase in correct-pair similarity. For exceedances--deficiencies, correct-pair CKA from 0.261 to 0.281, while wrong-pair changes from 0.723 to 0.102. Identity-pair scores are also retained in full.</p>')
    paragraphs.append('<h2>Metrics, lengths, answer controls, and input controls</h2>')
    paragraphs.append(table(['Relationship','ordinary CKA Δ','debiased Δ','fixed-input Gram adjusted Δ'],sensitivity))
    paragraphs.append('<p>All three metrics retain the mean directions of both inversion and all six complement relationships; both inversion relationships remain positive for every seed pairing and length.U-centered The metric is a prescribed sensitivity. Answer-stratum residualization makes samples dependent, so no unbiased conditional-population estimate is claimed. Input adjustment retains the single positional permutation-matrix kernel, Gram regression, and PSD truncation; it excludes only components captured by this particular kernel, not every input confound.</p>')
    paragraphs.append('<p>Within each length, positional permutation-matrix input Gram for identity, inversion, and complement CKA is always 1, so the input-kernel correct-minus-wrong contrast is 0; the numeric-sequence complement in the kernel CKA is also 1 (fixed-length complementation is an orthogonal sign flip plus a constant), whereas inversion differs. Kernel definitions were fixed, not searched according to results.'+link('input_length_baselines.csv','Raw-input reference table')+'.</p>')
    paragraphs.append('<p>Gold answers one-hot under the correct transformation of every relationship have CKA equal to 1, versus approximately 0.018–0.047 for the wrong transformation, showing that output association alone can generate a strong correct-pair advantage. After joint-answer stratification, pure gold-answer encodings have variance 0,CKA and are recorded as undefined rather than filled with 0. Output probabilities also reproduce positive inversion Delta: same-seed values approximately +0.336 and +0.331. A larger hidden CKA value alone does not establish an additional mechanism.'+link('answer_output_summary.csv','Answer/output baselines')+';'+link('answer_output_length_baselines.csv','Per-length baselines')+'.</p>')
    paragraphs.append('<p>32 within-stratum permutations from the original protocol use identical index permutations for trained/initial and correct/wrong conditions. Compute per length, then apply fixed weights. Both answer-controlled inversion Deltas exceed all 32 shuffled values (maximum same-seed shuffled Delta approximately 0.008 and 0.030); the descriptive upper-tail rank is 1/33. Permutation controls also disrupt input correspondence, so they do not establish a mathematical mechanism and are not new confirmatory tests.'+link('permutation_summary.csv','Permutation summary')+'.</p>')
    paragraphs.append('<img src="relation_length_heatmap.png" alt="Complete per-length results">')
    paragraphs.append('<p>After leaving out any one length, both inversion relationships retain positive answer-controlled means across all pairings and three metrics: descents--inverse descents range '+f'[{inverse_ranges["descent_inverse"][0]:.3f}, {inverse_ranges["descent_inverse"][1]:.3f}]; exceedances--deficiencies range [{inverse_ranges["exceedance_inverse"][0]:.3f}, {inverse_ranges["exceedance_inverse"][1]:.3f}].'+link('leave_one_length.csv','Complete leave-one-length results')+'.</p>')
    paragraphs.append('<h2>Relationship dependence and uncertainty</h2><p>The eight fixed relationships comprise two inversion and six complement cases, sharing inputs, symmetry orbits, and transformation families. Three source seeds and six cross-seed pairs do not increase the number of independent relationships.R0/R3/R4 also share seed17; these are not five independent data worlds. Source /seed ranges describe variation among these source models only; leave-one-relationship results describe sensitivity of the fixed relationship set only.</p>')
    paragraphs.append('<p>ordinary CKA Same-seed macro mean Delta is +0.251, but the relationship median is −0.022; after answer control, macro mean is −0.109, median −0.290. Leaving out any single relationship gives negative answer-controlled macro means (−0.217 to −0.059). The two strong inversion results therefore do not imply a general alignment improvement across mathematical relationships.'+link('transform_family_summary.csv','Fixed transformation-family summary')+';'+link('macro_summary.csv','Relationship mean/Median')+'.</p>')
    paragraphs.append('<details><summary>All leave-one-relationship results (ordinary CKA, same-seed)</summary>'+table(['Omitted relationship','Control','Remaining mean','Remaining median'],lotable)+'</details>')
    paragraphs.append('<p>Input sampling uses 256 length-stratified paired orbit-anchor resamples; the same indices apply to all relationships/conditions/seeds, retaining shared-input dependence. Outputs include resampling SD, bias, and percentiles. Repeated sampling changes the finite-sample CKA diagonal-term weights. For these high-dimensional data, resampling distributions are clearly shifted from the original estimate; percentiles are explicitly descriptive sampling sensitivities, not unbiased population confidence intervals, and are not used to select relationships by significance.'+link('resampling_diagnostics.csv','Input-resampling standard deviation and bias')+'.</p>')
    paragraphs.append('<h2>Task-count series: between-model similarity and initialization retention</h2><p>All R0–R4, both pools, five k, totaling 50 existing models. Keep the original 4,096 prefixes, compute separately for lengths 2–30, and use fixed primary weight 1/29; the original pooled-length values remain a separate control.A/B reconstructed initial parameters agree within every group at every k, but bitwise identity of the original training starts remains unverified.'+link('k_shared_initialization_audit.csv','Shared-initialization audit')+'.</p>')
    paragraphs.append('<img src="k_drift.png" alt="Two parallel curves for the task-count series">')
    paragraphs.append(table(['Split / seed','k1: between-pool  / self-initialization','k2','k4','k8','k16'],ktable))
    paragraphs.append('<p> from k1 to k16; mean self-initialization CKA increases in all five groups, with R0/R1/R2 respectively 0.113→0.221, 0.078→0.170, 0.084→0.207.R0, R2, R4 increasing at every adjacent step; R1 from k8 to k16 decreases, R3 also decreases. This supports an exploratory association between more tasks and greater retention of initial geometry, not universal monotonicity. Between-pool trajectories are not wholly monotone either; the two similarities cannot exactly decompose learned and retained contributions.</p>')
    paragraphs.append('<p>Pooling lengths changes the magnitude: R0 k8 between-pool CKA original value 0.803, versus per-length primary value 0.214; its average self-initialization CKA is 0.740 when pooled, versus 0.193 per length. Both being high suggests retention of length/initial geometry, but does not identify contributions exactly. Shortest lengths have repeated inputs: n2 only 2 unique permutations (143 records), n3 only 6, n4 only 24; n2 rank-one geometry can yield a CKA ceiling. All lengths retain their fixed weights; none was removed according to results.'+link('k_pooled_control.csv','Pooled-length control')+';'+link('k_length_input_audit.csv','Duplicate-input audit')+'.</p>')
    paragraphs.append('<h2>Delivery, verification, and stopping</h2><p>'+f'{ver["numeric_checks"]:,} numerical comparisons passed: feature-space CKA rechecks, independent unbiased HSIC formula checks, explicit resampling and weighted Gram consistency, cell-wise Delta//equal length weights, original k-series reproduction, and initial-identity labels. Maximum error {ver["maximum_absolute_discrepancy"]:.2e}.{ver["protected_files_checked"]:,} protected-file SHA unchanged, Git HEAD and the tracked working tree unchanged in this diagnosis round.F17 paper, results, release assets, and original CKA artifacts were unchanged in this round.'+link('verification.json','Verification record')+'.</p>')
    paragraphs.append('<p>Reproducible code and audits are all in this directory.'+link('README.txt','Execution steps and artifact index')+';'+link('protocol.json','Fixed analysis definitions')+';'+link('relation_length_cka.csv','All 9,072 cell-wise CKA/Δ')+';'+link('k_model_self_initial_length.csv','50 Model-to-own-initialization per-length table')+';'+link('k_length_cka.csv','Task-count-series per-length table')+'.CSV Full original numerical values are retained rather than replaced with rounded report values.</p>')
    paragraphs.append('<h2>Final four questions</h2>')
    paragraphs.append('<ol><li><strong>Which relationships stably exceed the initialization reference?</strong>Both inversion relationships, descents--inverse descents and exceedances--deficiencies, have consistently positive Delta under answer and input-kernel sensitivities. This exceeds a program-reconstructed reference only; original initial-weight identity remains directly unverifiable.</li><li><strong>Does the result depend on task pair, length, or metric?</strong>Neither inversion relationship depends on a single length/seed/one of the three metrics, but the overall conclusion depends strongly on relationship family and answer control. The six complement cases lack equivalent advantages, and a macro mean does not imply a universal effect.</li><li><strong>Do more tasks accompany greater initialization retention?</strong>Across five splits, k16 all exceed k1; three full trajectories are monotone, while R1/R3 decrease at intermediate steps. Original shared initial weights remain unverified, and input/model pairs are dependent.</li><li><strong>Which open question can an encoding control address next?</strong>It can test whether the inversion-contrast increment depends on current numeric token, positions, and one-line sequence encoding or persists under another encoding preserving the exact mathematical relationships. Each encoding still needs its own initialization and answer/output references; CKA alone cannot establish an internal algebraic mechanism. That experiment is not run in this round.</li></ol><p><strong>Stopped as requested.</strong>No new model training, seeds, tasks, encoding experiments, null-space experiments, or finite-field/GELU composition experiments.</p>')
    style='body{font-family:system-ui,sans-serif;max-width:1180px;margin:32px auto;padding:0 20px;color:#16212c;line-height:1.65}table{border-collapse:collapse;width:100%;font-size:13px;margin:16px 0}th,td{padding:9px;border:1px solid #d3dce5;text-align:left}th{background:#eef4f8}img{width:100%;margin:10px 0}details{border:1px solid #d3dce5;padding:12px;margin:14px 0}summary{cursor:pointer}a{color:#075aa1}h2{margin-top:32px}strong{font-weight:650}'
    (ROOT/'report.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><title>PermWorld CKA checkpoint diagnosis</title><style>'+style+'</style><body>'+''.join(paragraphs)+'</body></html>')
    (ROOT/'summary.txt').write_text('''PermWorld Existing-checkpoint diagnosis: completed and stopped; no not_started new training.

Scope: eight relationships, three existing seeds and their nine ordered pairings, 21 fixed lengths; Property32 five splits, two pools, k=1/2/4/8/16, totaling 50 models.

Initialization identity: 98 logical models/94 unique checkpoints are all "Reconstructed with the original initialization procedure; original weights not directly verifiable”.missing original step-0 weights/hashes, training commit, and framework version. Sampled forwards from the original program agree with existing caches; this does not verify original initial-weight identity.

1. Which relationships have stable training increments?
For the two inversion relationships, descents--inverse descents and exceedances--deficiencies, answer-controlled same-seed mean Delta is respectively +0.589, +0.641; relative to matching program-reconstructed references, all 9 pairings, 21 lengths, and 3 fixed metrics are positive. The six complement relationships lack equally stable answer-controlled advantages.

2. Does the result depend on a task pair, length, or metric?
Each inversion relationship is robust to leave-one-length and three-metric checks; the overall claim depends on relationship type and answer control. Across eight relationships, ordinary CKA macro mean Delta is +0.251, median −0.022; answer-controlled mean −0.109, median −0.290, with a negative macro mean after omitting any relationship. Positive Delta is a change in the correct/wrong relative contrast, not an absolute correct-pair CKA increase. Neither can CKA identify an encoding or mathematical mechanism.

3. Do more tasks accompany stronger initialization retention?
In five splits, k16 average self-initialization CKA all exceed k1;R0/R2/R4 is monotone, while R1/R3 decrease at intermediate steps. Between-pool CKA is not universally monotone either. A/B reconstructed parameters agree, but original shared initial weights remain unverified; the two CKA measurements cannot exactly decompose similarity sources.

4. What can an encoding control resolve?
It tests whether the inversion-contrast increment depends on current numeric token, positions, and one-line sequence encoding or persists under another encoding preserving the mathematical relationships. Soft-output and general-input effects have not been fully excluded. No encoding experiment was conducted in this round.

Input resampling has finite-sample bias; percentiles are descriptive sensitivities only. Relationship, seed, and shared-input uncertainties are separate, without increasing the number of independent relationships. Full results, leave-one-relationship tables, k curves, and reproducible scripts: report.html and README.txt.

11,497 numerical comparisons passed; 1,214 protected-file SHA unchanged.F17 paper, results, release artifacts, and original CKA directory were unchanged in this round.
''')
    write_json(ROOT/'completion.json',dict(status='complete_stopped',completed_utc=datetime.now(timezone.utc).isoformat(),training_started=False,
        stable_positive_relations=['descent_inverse','exceedance_inverse'],original_initial_identity_verified=False,
        protected_files_unchanged=ver['protected_files_checked'],scope='existing PermWorld checkpoints only',
        unanswered=['Original initialization bitwise identity','All input-encoding/soft-output effects','Internal algebraic mechanism','New data-world replication']))
    # All executable inputs are immutable upstream references with hashes or local scripts.
    files={p.name:sha(p) for p in sorted(ROOT.iterdir()) if p.is_file() and p.name!='deliverable_manifest.json'}
    write_json(ROOT/'deliverable_manifest.json',dict(files=files,workspace_commit=inventory['workspace_commit'],upstream_commit=inventory['upstream_commit']))
    print({'report':str(ROOT/'report.html'),'status':'complete_stopped'},flush=True)


if __name__=='__main__':run()
