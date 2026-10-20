"""Frozen exploratory summaries and self-contained HTML report."""
from collections import defaultdict
import html
import json
import shutil
import time
import numpy as np
from inventory import ROOT,WS,csv_read,csv_write,js,save,sha

PRIMARY=['linear_cka','knn_overlap','procrustes']
FIELDS=['initial_correct','initial_wrong','initial_identity','trained_correct','trained_wrong','trained_identity',
        'B0','Bt','delta_correct','delta_wrong','delta',
        'initial_correct_raw_error','initial_wrong_raw_error','initial_identity_raw_error',
        'trained_correct_raw_error','trained_wrong_raw_error','trained_identity_raw_error']
KEYS=['domain','cohort','relation','family','kind','branch','control','metric']


def number(x):return float(x) if x not in ['',None] else None

def complete_mean(values):
    return float(np.mean(values)) if values and all(v is not None for v in values) else None


def table(rows,fields,limit=None):
    def fmt(x):
        if x is None or x=='':return 'undefined/missing'
        if isinstance(x,(float,np.floating)):return f'{x:.4f}'
        return html.escape(str(x))
    rows=rows[:limit] if limit else rows
    return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+html.escape(k)+'</th>' for k in fields)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+fmt(r.get(k))+'</td>' for k in fields)+'</tr>' for r in rows)+'</tbody></table></div>'


def run():
    started=time.monotonic();freeze=js(ROOT/'protocol_freeze.json')
    assert sha(ROOT/'summary_protocol.json')==freeze['summary_protocol_sha256']
    rows=csv_read(ROOT/'permutation_metric_results.csv')+csv_read(ROOT/'field_metric_results.csv')
    csv_write('metric_results.csv',rows)
    group=defaultdict(list)
    for r in rows:group[tuple(r[k] for k in KEYS)+ (r['source_unit'],r['seed'],r['world'])].append(r)
    units=[]
    for key,rs in group.items():
        expected=21 if rs[0]['domain']=='PermWorld' else 1
        assert len(rs)==expected
        out=dict(zip(KEYS+['source_unit','seed','world'],key),planned_lengths=expected)
        for f in FIELDS:
            v=[number(r[f]) for r in rs];out[f]=complete_mean(v)
            if f in ['Bt','delta']:out[f+'_defined_lengths']=sum(x is not None for x in v)
        for f in ['Bt','delta']:
            v=[number(r[f]) for r in rs];good=[x for x in v if x is not None]
            out[f+'_positive_lengths']=sum(x>0 for x in good)
            out[f+'_negative_lengths']=sum(x<0 for x in good)
            out[f+'_length_median']=float(np.median(good)) if len(good)==expected else None
            out[f+'_length_min']=min(good) if good else None;out[f+'_length_max']=max(good) if good else None
        units.append(out)
    csv_write('source_metric_summary.csv',units)
    rg=defaultdict(list)
    for r in units:rg[tuple(r[k] for k in KEYS)].append(r)
    relations=[]
    for key,rs in rg.items():
        out=dict(zip(KEYS,key),planned_sources=len(rs))
        for f in FIELDS:
            v=[r[f] for r in rs if r[f] is not None]
            out[f+'_sources']=len(v);out[f+'_mean']=float(np.mean(v)) if v else None
            out[f+'_median']=float(np.median(v)) if v else None
            if f in ['Bt','delta']:
                out[f+'_min']=min(v) if v else None;out[f+'_max']=max(v) if v else None
                out[f+'_source_sd']=float(np.std(v,ddof=1)) if len(v)>1 else None
                out[f+'_positive_sources']=sum(x>0 for x in v)
        relations.append(out)
    csv_write('relation_metric_summary.csv',relations)
    # Direction comparisons use the same source units for every primary metric.
    ag=defaultdict(dict)
    for r in units:
        if r['metric'] in PRIMARY:ag[tuple(r[k] for k in KEYS[:-1])+ (r['source_unit'],)].update({r['metric']:r})
    agreements=[]
    for key,ms in ag.items():
        out=dict(zip(KEYS[:-1]+['source_unit'],key))
        for f in ['Bt','delta']:
            good=all(m in ms and ms[m][f] is not None for m in PRIMARY)
            out[f+'_all_three_defined']=good
            for m in PRIMARY:out[f+'_'+m]=ms.get(m,{}).get(f)
            signs=[int(np.sign(ms[m][f])) for m in PRIMARY] if good else []
            out[f+'_all_three_direction_agree']=len(set(signs))==1 if good else None
            out[f+'_all_three_positive']=all(s>0 for s in signs) if good else None
        agreements.append(out)
    csv_write('direction_agreement.csv',agreements)
    # Equal relation means within a cohort; all shared families removed together.
    lg=defaultdict(list)
    for r in relations:
        if r['kind'] not in ['cross_task','cross_task_set']:continue
        lg[r['domain'],r['cohort'],r['branch'],r['control'],r['metric']].append(r)
    leave=[]
    for key,rs in lg.items():
        for family in ['none']+sorted({r['family'] for r in rs}):
            remain=[r for r in rs if family=='none' or r['family']!=family]
            row=dict(zip(['domain','cohort','branch','control','metric'],key),excluded_family=family,remaining_relations=len(remain))
            for f in ['Bt','delta','delta_correct','delta_wrong']:
                row[f]=complete_mean([r[f+'_mean'] for r in remain])
            leave.append(row)
    csv_write('leave_one_family.csv',leave)
    # Reuse old input and full-cache drift controls; never overwrite originals.
    parent=WS/'diagnostics/permworld_cka_20261009'
    reused=['input_length_baselines.csv','k_summary.csv','k_length_cka.csv',
            'k_pooled_control.csv','k_input_bootstrap.csv','input_bootstrap_summary.csv','permutation_summary.csv']
    for f in set(reused):shutil.copyfile(parent/f,ROOT/('inherited_'+f))
    save('inherited_control_audit.json',{'files':{f:dict(source=str(parent/f),sha256=sha(parent/f)) for f in set(reused)},
        'scope':'Historical controls retain their original estimator and uncertainty scope; not recomputed new-metric input uncertainty.'})
    # Conditional numeric answer accuracy, explicitly separate from full sequence scoring.
    data=dict(np.load(ROOT/'evaluation_inputs.npz'));tasks=list(data['tasks']);accuracy=[]
    records=js(ROOT/'source_forward_audit.json')['records']
    for r in records:
        if r['condition']!='trained':continue
        path=ROOT/'features'/f'{r["cohort"]}_{r["task"]}_s{r["seed"]}_trained.npz'
        with np.load(path) as a:
            pred=a['output_prob'][:,0].argmax(-1)
        gold=data['labels'][:,0,tasks.index(r['task'])]
        for n in range(10,31):
            ids=(data['lengths']==n)&(data['split']==1)
            accuracy.append(dict(cohort=r['cohort'],task=r['task'],seed=r['seed'],length=n,anchors=int(ids.sum()),
                accuracy=float((pred[ids]==gold[ids]).mean()),scope='Conditional numeric0..30 argmax, not full-vocabulary or sequence accuracy'))
    csv_write('source_answer_accuracy.csv',accuracy)
    # Approved additional training list is intentionally empty.
    csv_write('additional_training_manifest.csv',[],fields=['category','task','cohort','source_unit','encoding','reason','status'])
    future=[]
    future.append(dict(category='A',unique_tasks=0,source_units=0,encodings=0,models=0,reason='All current selected comparisons have existing complete weights; additional analysis only',status='completed_this_round'))
    future.append(dict(category='B_optional',unique_tasks=1,source_units=1,encodings=1,models=1,reason='Optional left_to_right_minima seed17 in six-hour matched protocol to fill min_record_reverse; could be unnecessary if relation omitted',status='candidate_not_scheduled; old-world protocol and exact seed alone do not establish new-world independence'))
    future.append(dict(category='C_candidate',unique_tasks=6,source_units='U_unfrozen',encodings=1,models='6*U',reason='Actual step0 plus new world/initialization source units; descents,recoils,peaks,valleys,Lmax,Rmax',status='not_authorized_to_run_this_round; U must follow precision/budget protocol'))
    future.append(dict(category='C_cost_scenario',unique_tasks=6,source_units=5,encodings=1,models=30,reason='Illustrative count only, not power or precision guarantee; independent engineering preflight excluded',status='candidate_only'))
    csv_write('future_training_candidates.csv',future)
    fwd=js(ROOT/'source_forward_audit.json');pa=js(ROOT/'permutation_evaluation_audit.json');fa=js(ROOT/'field_evaluation_audit.json');ka=js(ROOT/'k_evaluation_audit.json')
    cost=[dict(activity='existing_PermWorld_forward_and_cache',seconds=fwd['elapsed_seconds'],new_source_models=0,forward_sequences=fwd['forward_sequences_this_run'],
        reused_sequences=fwd['reused_sequences_this_run'],budget_type='GPU+cache IO wall time, RTX5070; reconstruction creates no trained source'),
        dict(activity='PermWorld_metric_analysis_PCA_orthogonal_fits',seconds=pa['elapsed_seconds'],new_source_models=0,forward_sequences=0,reused_sequences=0,budget_type='CPU analysis only; many single-purpose analytical fits'),
        dict(activity='F5_reuse_validation_and_metric_analysis',seconds=fa['elapsed_seconds'],new_source_models=0,forward_sequences=90*625,reused_sequences=90*625,budget_type='GPU feature audit + CPU maps;625 source-seen inputs'),
        dict(activity='Property32_cached_metrics',seconds=ka['elapsed_seconds'],new_source_models=0,forward_sequences=0,reused_sequences=50*4096,budget_type='CPU cached analysis only'),
        dict(activity='new_source_training',seconds=0,new_source_models=0,forward_sequences=0,reused_sequences=0,budget_type='No optimizer/source updates or paused resumes'),
        dict(activity='deferred_encoding768_plan',seconds='',new_source_models=0,forward_sequences=0,reused_sequences=0,budget_type='Old158–190GPU-hour allowance is a deferred scenario, not measured training in this task')]
    csv_write('cost_estimate.csv',cost)
    inventory=csv_read(ROOT/'training_inventory.csv');cohorts=csv_read(ROOT/'cohort_inventory_summary.csv')
    grade={g:sum(r['initialization_grade']==g for r in inventory) for g in ['A','B','C','D']}
    source_main=[r for r in relations if r['branch'] in ['prefix_final_norm','encoder_final'] and r['control']=='answer_strata' and r['metric'] in PRIMARY and r['kind']=='cross_task']
    ordinary_main=[r for r in relations if r['branch'] in ['prefix_final_norm','encoder_final'] and r['control']=='within_length' and r['metric'] in PRIMARY and r['kind'] in ['cross_task','cross_task_set']]
    direction_main=[r for r in agreements if r['domain']=='PermWorld' and r['branch']=='prefix_final_norm' and r['kind']=='cross_task' and r['control']=='answer_strata']
    counts={f:dict(available=sum(r[f+'_all_three_defined'] for r in direction_main),same=sum(r[f+'_all_three_direction_agree'] is True for r in direction_main),
        positive=sum(r[f+'_all_three_positive'] is True for r in direction_main),total=len(direction_main)) for f in ['Bt','delta']}
    save('summary_audit.json',dict(rows=len(rows),source_metric_summary_rows=len(units),relation_metric_summary_rows=len(relations),
        new_source_training=0,initialization_inventory_grades=grade,answer_control_direction_agreement=counts,
        elapsed_seconds=time.monotonic()-started))
    plots(relations)
    report(relations,units,agreements,leave,cohorts,cost,counts,grade)
    print('summary/report complete',len(rows),'metric cells',counts,flush=True)


def plots(relations):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    old=csv_read(ROOT/'inherited_k_summary.csv')
    fig,axes=plt.subplots(1,2,figsize=(11,4.1),sharey=True)
    for rep in ['r0','r1','r2','r3','r4']:
        rr=sorted((r for r in old if r['replicate']==rep),key=lambda r:int(r['k']))
        for ax,field in zip(axes,['intermodel_cka','mean_self_initial_cka']):
            ax.plot([int(r['k']) for r in rr],[float(r[field]) for r in rr],marker='o',linestyle='--' if rep in ['r3','r4'] else '-',label=rep+(' subset' if rep in ['r3','r4'] else ' source'))
            ax.set_xscale('log',base=2);ax.set_xticks([1,2,4,8,16],[1,2,4,8,16]);ax.set_xlabel('Number of tasks');ax.set_ylim(0,1)
    axes[0].set_title('Between pools: original full-cache length means');axes[1].set_title('Own reconstructed init: original length means')
    axes[0].set_ylabel('Linear CKA');axes[1].legend(fontsize=8);fig.tight_layout();fig.savefig(ROOT/'k_drift_reused.png',dpi=160);plt.close(fig)
    # All relation IDs retained; do not choose a subset with positive values.
    for cohort in ['specialist16','native_confirmation_single','six_hour_session_single']:
        rr=[r for r in relations if r['cohort']==cohort and r['kind']=='cross_task' and r['branch']=='prefix_final_norm' and r['control']=='answer_strata' and r['metric'] in PRIMARY]
        ids=list(dict.fromkeys(r['relation'] for r in rr));fig,axes=plt.subplots(1,3,figsize=(13, max(3,len(ids)*.3)),sharey=True)
        for ax,m in zip(axes,PRIMARY):
            lookup={r['relation']:r for r in rr if r['metric']==m};vals=[lookup[i]['delta_mean'] for i in ids]
            ax.barh(np.arange(len(ids)),[v or 0 for v in vals],color=['#147d64' if v is not None and v>0 else '#b85749' for v in vals])
            for pos,v in enumerate(vals):
                if v is None:ax.text(0,pos,'undefined',fontsize=8)
            ax.axvline(0,color='black',linewidth=.8);ax.set_title(m);ax.set_xlabel('Delta (metric-specific units)')
        axes[0].set_yticks(np.arange(len(ids)),ids);axes[0].invert_yaxis();fig.suptitle(cohort+' / pre-task / exact answer strata');fig.tight_layout();fig.savefig(ROOT/f'delta_{cohort}.png',dpi=160);plt.close(fig)


def report(relations,units,agreements,leave,cohorts,cost,counts,grade):
    primary=[r for r in relations if r['domain']=='PermWorld' and r['kind']=='cross_task' and r['branch']=='prefix_final_norm' and r['control']=='answer_strata' and r['metric'] in PRIMARY]
    compact=[]
    for r in primary:
        compact.append(dict(cohort=r['cohort'],relation=r['relation'],metric=r['metric'],sources=r['planned_sources'],defined=r['delta_sources'],
            B0=r['B0_mean'],Bt=r['Bt_mean'],dc=r['delta_correct_mean'],dw=r['delta_wrong_mean'],Delta=r['delta_mean'],median=r['delta_median'],
            min=r['delta_min'],max=r['delta_max'],positive_sources=r['delta_positive_sources']))
    missing=[r for r in csv_read(ROOT/'relation_task_graph.csv') if r['id']=='min_record_reverse']
    oldtrend=csv_read(WS/'diagnostics/permworld_cka_20261009/k_trend.csv')
    field=[r for r in relations if r['domain']=='F5' and r['branch']=='encoder_final' and r['control']=='within_length' and r['metric'] in PRIMARY]
    controls=[r for r in relations if r['branch'] in ['output_prob','answer_onehot','answer_scalar'] and r['control']=='within_length' and r['metric'] in PRIMARY]
    loo=[r for r in leave if r['branch']=='prefix_final_norm' and r['control']=='answer_strata' and r['metric'] in PRIMARY]
    s='''PermWorld Inventory and exploratory existing-model evaluation (2026-10-09)
New source-model training in this round: 0; no resumed paused tasks or publication. 624 logical inventory entries and 582 distinct weight-container SHA do not constitute independent replications.
Original 48 single-task +50 k positions: 98 logical positions, 94 distinct checkpoint, checked file by file via SHA.
k16 is 24 endpoints, including 6 reusing old v3 weights and 18 additional historical training runs. Small PermWorld1200-update pilot24, F5 rank3 72/rank4 45/matched27 are registered separately by cohort.
Relationships in this round: 19 mathematical certificates, 18 with matching completed weights; min_record_reverse lacks a same-cohort Lmin/Rmin.62 relationship-source pairs, 82 distinct PermWorld source positions; F5 only uses unified rank4 45 weights; k Reuse 50 positions.
Initialization: PermWorld references C, with constructor reconstruction not proving original-weight identity; F5 encoder has actual historical step0 activations/B, limited to 625 saved inputs and encoder positions; the initial output readout remains C. Not every state_dict was loaded for the inventory; unmeasured parameter hashes remain explicitly blank.
ordinary /Answer strata, gold answers, and conditional numeric output probabilities are retained. The primary point is before task token; ONE_END;query branches are separate. Fixed length 1/21;missing without reweighting.CKA, kNN(k10), 8-dimensional fitted PCA followed by orthogonal Procrustes was frozen before new results were viewed.
F5 All 625 source-training inputs were previously seen; 250/375 splits apply only to analysis-mapping fit/test. The four gold answers jointly identify each input, leaving no repeated strict answer groups and all values undefined; controls were not relaxed. Domain, architecture, and input encoding change together.
Statistics: exploratory, with no confirmatory p-values or equivalence claims. Source ranges/SDs describe initialization variation in a fixed world; relationships share tasks and orbits without adding independent relationships. Old input bootstrap is conditional only on the old fixed models/CKA protocol; no input-sampling intervals were manufactured for new metrics.
Minimal added training: current deliverables require 0,additional_training_manifest contains only a header. Optional gap 1 Lmin uses the old protocol seed17 (not new-world confirmation); for actual step0 and new-data confirmation, candidate 6 tasks x U×E, prioritizing E1,U is unfrozen; U5=30 is a cost scenario only. Future work first needs a separate engineering preflight, without choosing budgets by geometric results.
Encoding 768 models remain unexecuted. Current inventory supports extending relationships/changing metrics/a second instance. Freeze a smaller plan only after specifying the encoding intervention, representative relationships, precision, and resources. This round does not invoke encoding prepare/train/evaluate.
F17 is inventoried only; papers/results/releases were unchanged and it is not natural-geometry confirmation. Full protected-hash checks: audit.json.
Reproduction: inventory.py -> relations.py -> prepare_eval.py -> evaluate.py --stage features -> finalize_inventory.py -> evaluate.py --stage permutation-metrics/field-metrics;evaluate_k.py -> summarize.py -> verify_results.py. Fixed protocol/input files must not be overwritten after viewing results.
'''
    s+='\n Measured primary results: the original inversion relationships descents/recoils and exceedances/deficiencies have positive Delta under all three answer-controlled metrics, with all three sources agreeing.CKA Delta respectively 0.588965/0.641461,dc respectively -0.033851/+0.020471, dw respectively -0.622816/-0.620990, mainly from wrong-pair decreases. The six-hour cohort Lmax/Rmin inversion has only one positive source and is not a stable multi-source replication.\n'
    s+='For new reversal/Complement relationships have mean Bt mostly positive, while Delta is negative; this does not imply no learning. Max-record reversal has slightly negative native seed1009/3037 trained CKA preference but slightly positive other metrics. Double-rise/double-fall kNN has insufficient samples at two lengths, making the fixed aggregate undefined.\n'
    s+='For both old inversion relationships, across 3 source x 21 lengths, all three metrics and debiased CKA Delta are positive, without a single driving length. Max-record reversal native seed1009/3037 in debiased CKA has Bt become positive, showing metric-version dependence of weak trained preference; Delta remains negative.\n'
    s+='F5 For the first two relationship groups, CKA/Procrustes mean Delta is positive but kNN negative; other sources also change sign. Without strict answer controls or source-unseen inputs, the second instance does not support a unified mechanism claim.\n'
    s+='In the old full-cache k series, all five curves have k16 initialization retention exceeding k1,R0/R2/R4 is monotone, while R1/R3 but are not monotone. The new complete-orbit metric 29-length aggregate undefined.\n'
    (ROOT/'reuse_and_missing_report.txt').write_text(s)
    body='<h1>PermWorld: training inventory and minimal supplementation plan</h1><p class="lead">Completed weights were reused, relationships/metrics extended, and a  F5  second-instance exploratory evaluation completed. New source training: <strong>0</strong>. This report makes neither mechanism claims nor confirmatory inference.</p>'
    body+='<h2>1. Actually available inventory and initialization evidence</h2><p>The table lists logical entries and distinct weight containers, not independent experiments. Original 48+50 positions deduplicate to 94 checkpoint;k16 6 endpoints reuse old v3 weights.F17 containers are inventoried only. All selected completed weights are locally accessible, requiring no download/release retraining.</p>'
    body+=table(cohorts,['cohort','logical_positions','completed_records','accessible_weights','distinct_file_sha256','reused_endpoints','partial','plans_only'])
    body+='<p>Full statuses, configurations, hashes, caches, and gaps:  <a href="training_inventory.csv">training_inventory.csv</a>.PermWorld Initializations are all C: reconstructed using original constructors/vocabularies/RNG paths, without verification of original-weight identity.F5 encoder uses actual step-0 activations/B; the initial output head is reconstructed as C.'+html.escape(str(grade))+'. Unselected states were not all loaded to fill tables; absent parameter hashes remain explicit.</p>'
    body+='<h2>2. Which additional relationships need no source training?</h2><p>Exhaustive S1–S7 and fixed large-n random samples: 8,601 permutations x 19 identities all passed.18 have matching completed weights, forming 62 relationship/source pairings. Additional reversals of LIS/LDS, run, fixed/anti-fixed, left/right maxima, double rises/double falls; inversion of left maxima/right minima; and complement of double rises/double falls all reuse weights. Three inverse self-invariances are separate; left/cohort right-minimum relationships lacking a matched / are not spliced across cohorts.</p><p>Complement is also correct for some reversal task pairs, so it cannot serve as their wrong comparator. Wrong transformations were frozen before reading metrics using oracle counterexamples. Shared tasks/group actions carry dependency tags; 18 relationships are not 18 independent property families.</p>'
    body+='<p><a href="relation_task_graph.csv">Relationship coverage figure</a> · <a href="relationship_certificates.json">Mathematical certificates</a> · <a href="metric_protocol.json">Frozen metric protocol</a> · <a href="summary_protocol.json">Frozen summary rules</a></p>'
    body+='<h2>3. Do three metric families agree, and where does Delta arise?</h2><p>Compute each length independently and aggregate with fixed 1/21. The primary point is input-end, before the task prompt, final-layer state.B0=Initial correct-minus-wrong; Bt=trained correct-minus-wrong; dc/dw denote their respective training changes; Delta =Bt−B0=dc−dw.Procrustes score=−NRMSE, retaining raw errors. Positive Delta alone does not imply convergence of correct-pair representations.</p>'
    body+='<p>At the primary answer-stratified point, source/relationship cells with all three metrics defined have Bt direction agreement  '+str(counts['Bt']['same'])+'/'+str(counts['Bt']['available'])+'; Delta direction agreement  '+str(counts['delta']['same'])+'/'+str(counts['delta']['available'])+'. This descriptive coverage shares relationships//sources and cannot be counted as independent replications. All disagreements and negatives remain.</p>'
    body+='<p><strong>The most stable positive increments are the original inversion relationships </strong>:descents/recoils and exceedances/deficiencies: all three sources have positive Delta in all three metric families. In each 3 source x 21 length cell, all three metrics and debiased CKA have positive Delta, not driven by one length.CKA mean is 0.589/0.641,dc, respectively approximately −0.034/+0.020, dw approximately −0.623/−0.621, so the main pattern is reduced wrong-pair similarity. The six-hour left-maxima/right-minima inversion also has a positive direction, but only one source. The exceeded reference is reconstructed initialization C, not a verified increment from actual original initial weights.</p><p>For new reversal/complement relationships, mean Bt is mostly positive while Delta is negative; their initial preferences are higher, so negative Delta does not negate learning.native Left/right-maxima reversal seed1009 and 3037 Bt: ordinary CKA is slightly negative, neighborhood /Procrustes slightly positive; debiased CKA also becomes positive. This weak trained preference depends on metric version, while Delta remains negative. For double rises/double falls, the neighbor metric has insufficient samples at two lengths, leaving the strict 21-length aggregate undefined.</p>'
    body+=table(compact,['cohort','relation','metric','sources','defined','B0','Bt','dc','dw','Delta','median','min','max','positive_sources'])
    for c in ['specialist16','native_confirmation_single','six_hour_session_single']:body+=f'<img src="delta_{c}.png" alt="{c} all relation Delta">'
    body+='<p>ordinary CKA/debiased. Sensitivities, all sources, and per-length data:  <a href="metric_results.csv">metric_results.csv</a>, <a href="source_metric_summary.csv">Source-level summary</a>, <a href="direction_agreement.csv">Shared-source direction comparison</a>. Numerical magnitudes across metrics are not compared.8-dimensional PCA Procrustes evaluates only the selected subspace, not linear predictability of the entire hidden state.</p>'
    body+='<details><summary>Leave-one-dependency-family: all results retained</summary>'+table(loo,['cohort','metric','excluded_family','remaining_relations','Bt','delta','delta_correct','delta_wrong'])+'</details>'
    body+='<h2>4. Answer, output, input, and extraction-point controls</h2><p>Strict answer stratification subtracts group means separately for fit/test; this is an oracle measurement control, not a deployable predictor without answers. Gold-answer geometry is static, without a fictitious initialization or label-learning Delta. The output branch uses only numeric 0–30 conditional probabilities, retaining error/confidence structure.query is separate and cannot replace pre-task states or be concatenated into source_query_concat.</p><p>Historical input kernel controls and CKA permutation /input/bootstrap are reused within the scope documented in  <a href="inherited_control_audit.json">Inherited-control audit</a>. These original-protocol controls do not directly provide input-sampling intervals for new kNN or Procrustes and do not cover unmeasured reversal-input kernel. Hidden Delta minus output Delta does not identify an additional mechanism.</p>'
    body+='<details><summary>All output/gold baselines (length-stratified)</summary>'+table(controls,['domain','cohort','relation','branch','metric','B0_mean','Bt_mean','delta_correct_mean','delta_wrong_mean','delta_mean','delta_sources'])+'</details>'
    body+='<h2>5. k Task-count series: drift is an auxiliary clue</h2><p>Reuse 50 logical positions. R0–R2 denote distinct source seeds; R3/R4 fixed seed17 task-subset sensitivities. The two pools have matching reconstructed-initial parameter hashes, which does not verify historical shared initialization. The figure inherits fixed per-length CKA from complete caches, identical to the original report; this is not a new orbit-split metric.</p><img src="k_drift_reused.png" alt="Inter-pool CKA and own-init retention">'
    body+=table(oldtrend,['replicate','metric','k1','k16','k16_minus_k1','increasing_adjacent_steps','adjacent_steps'])
    body+='<p>Old per-length CKA shows all five initialization-retention curves have k16 exceeding k1;R0/R2/R4 increases step by step; R1/R3 has declines. This is a descriptive trend within a fixed world/protocol; R3/R4 in particular are not new source seeds.</p>'
    body+='<p>The new metric divides analysis fit/test by complete-orbit SHA parity using a fixed rule. n=2 lacks test, n=3 lacks fit, and fixed 1/29 plus complete-length requirements make the new aggregate undefined; it was not changed post hoc to average long permutations only. Per-length kNN/Procrustes/CKA remains in  <a href="k_metric_results.csv">k_metric_results.csv</a>; coverage:  <a href="k_metric_summary.csv">k_metric_summary.csv</a>.kNN Self-queries and repeated rows of the same input object are excluded; repeats of other objects retain their original cache sampling weights. High between-pool CKA together with high initialization retention suggests preserved initial geometry, without exactly identifying its sources.</p>'
    body+='<h2>6. Second domain: F5 what can be reused?</h2><p>Only the unified rank4 ordinary-training cohort is selected: 45 models (3 input/basis×3 initializations x 5 groups), 36 P/Mk comparisons, without mixing 144 heterogeneous sources. Correct transformation is the inverse task basis; wrong uses the prescribed adjacent basis inverse. All 625 inputs were seen in source training; 250 fit/375 test is only held out for analysis mappings. Actions preserve the first coordinate; fit/test orbit intersection 0. Raw step0 and final encoder activations, 90 in total, were independently recomputed; maximum difference 0.</p>'
    body+=table(field,['relation','metric','planned_sources','B0_mean','Bt_mean','delta_correct_mean','delta_wrong_mean','delta_mean','delta_median','delta_min','delta_max'])
    body+='<p>Mean Bt for four relationships is positive under all three metrics, but Delta directions differ: P/M1 and P/M2 CKA/Procrustes is positive, while kNN are respectively −0.0368/−0.0061.P/M3, P/M4 mean Delta is positive under all three, but many individual sources disagree; P/M3 Procrustes mean Delta is only 0.0010. The second instance shows metric and input/basis/initialization sensitivity, not uniform cross-domain replication.</p>'
    body+='<p>The complete rank4 answer tuple uniquely identifies each input, making strict answer strata all undefined. This limits exclusion of output association; controls are not relaxed to manufacture positives.F5 and PermWorld also changes mathematical domain, MLP/Transformer, and input representation; this is a second instance, not an effect attributable to domain alone.F5 Reinterpreting coordinates as polynomials does not constitute a third instance; F13/F17 intervention tasks are not confirmation of natural geometry.</p>'
    body+='<h2>7. Minimal new training and 768 encoding plan</h2><p>Currently required additions: 0.<a href="additional_training_manifest.csv">New-training manifest</a> is empty (header only). Optional coverage completion: under the same six-hour protocol, 1 Lmin seed17 fills the reversal-record relationship; the current scope can also be completed without it. For original-initialization identity/independent-new-world confirmation, candidate 6 tasks x U×E, first E=1;U is unfrozen; U=5 30 models are a cost scenario only, plus an independent engineering-development unit excluded from formal samples. Six tasks cover representative relationships only; making both original inversion pairs primary also requires exceedances/deficiencies, giving 8U (U5 with 40).<a href="minimal_confirmation_draft.json">Future minimal-confirmation draft</a> retains these open items and is not an executable frozen protocol.</p><p>768 models belong to an unexecuted encoding plan. Inventory already supports "extend relationships, change metrics, second instance" without a large encoding intervention. A future encoding intervention should answer whether it changes initial B0 or trained Bt, decomposing Gamma0/Gammat/GammaDelta;Delta reversal with similar Bt must not be described as reversed learned preference. Freeze a separate small protocol only after specifying the question, representative relationships, precision, and budget. No encoding prepare/train/evaluate or training continuation in this round.partial.</p>'
    body+=table(csv_read(ROOT/'future_training_candidates.csv'),['category','unique_tasks','source_units','encodings','models','reason','status'])
    body+='<h2>8. Cost, reproduction, and stopping</h2>'+table(cost,['activity','seconds','forward_sequences','reused_sequences','new_source_models','budget_type'])
    body+='<p>All artifacts from this round are in an independent directory. <a href="audit.json">Full verification/protection audit</a> · <a href="reuse_and_missing_report.txt">Short text report</a>. Reproduction scripts call neither optimizers nor source-training entries; analytical PCA/ orthogonal-mapping fit budgets are separate. Old CKA reports, F17 paper/results/release hashes were checked unchanged in that round. Stop after completion.</p>'
    (ROOT/'report.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>PermWorld inventory and exploratory metrics</title><style>body{font:16px/1.65 system-ui,sans-serif;max-width:1300px;margin:36px auto;padding:0 22px;color:#24313b}h1,h2{line-height:1.3}h2{margin-top:38px}.lead{background:#e8f3ee;padding:16px;border-radius:8px}.scroll{overflow:auto}table{border-collapse:collapse;font-size:12px;width:100%;margin:14px 0}th,td{border:1px solid #dce2e5;padding:7px;text-align:left;white-space:nowrap}th{background:#eef2f5}img{max-width:100%;margin:18px 0}a{color:#1456a0}details{margin:18px 0}</style>'+body+'</html>')


if __name__=='__main__':run()
