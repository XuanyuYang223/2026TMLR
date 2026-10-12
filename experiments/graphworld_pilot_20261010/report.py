"""Complete pilot delivery and explicit stop, including every undefined cell."""
import csv
import html
import math
from .common import *

def rows(path):
    with Path(path).open(newline='') as f:return list(csv.DictReader(f))
def number(v):return float(v) if v not in ('',None) else float('nan')

def report():
    c=verify();done=read(DIAG/'evaluation_complete.json');assert done['status']=='completed'
    summary=rows(DIAG/'geometry_summary.csv');lengths=rows(DIAG/'geometry_per_n.csv');learning=rows(DIAG/'source_learning.csv')
    assert len(summary)==240 and len(lengths)==720 and len(learning)==240
    coverage=[]
    for r in lengths:
        status='defined' if r['trained_correct']!='' and r['trained_wrong']!='' else 'undefined'
        coverage.append({k:r[k] for k in ('relation','n','control','branch','metric','fit_count','analysis_count')}
                         |dict(status=status,dependence='Shared4tasks,2transforms,1world,1trueinitial',reason='No threshold relaxation or length reweighting' if status=='undefined' else 'Exploratory cell'))
    csv_write(DIAG/'coverage_status.csv',coverage)
    main=[r for r in summary if r['branch']=='prefix_final' and r['metric']=='linear_cka']
    text=['GraphWorld four-task pilot: fixed-budget exploratory results.','',
        'Question: does hidden-state similarity change with the exact graph correspondence, relative to the wrong pairing and the actual initialization?',
        'Four source models use the original256-wide four-layer eight-head Transformer,188 tokens, cloned true step0 and the archived descents start RNG; only graph data/task semantics differ.',
        'The four relations share four tasks and two transformations. One graph world and one initialization do not establish universal effects or independent replications.',
        'Primary state is final_norm at the graph-end token before the task token. Similarity is higher-is-more-similar; Procrustes uses negative held-out normalized error.',
        'All3 n values have fixed1/3 weight. Missing strata/rank/variance leave the aggregate undefined. No model, relation, n or metric is selected by its outcome.','',
        'Primary CKA by relation/control: original similarities are preserved in the CSV; decompositions follow.']
    for r in main:
        text.append(' | '.join([r['relation'],r['control']]+[f'{k}={r[k] or "undefined"}' for k in c['evaluation']['fields']]))
    text+=['','Source task learning (trained,identity); full-vocabulary answer accuracy, answer CE and fixed training-mode baseline:']
    for r in learning:
        if r['condition']=='trained' and r['action']=='identity':
            text.append(' | '.join(f'{k}={r[k]}' for k in ('task','split','n','answer_accuracy','answer_ce','mode_accuracy','density_mean')))
    text+=['','Interpretation limits:',
        'Positive Delta can result from declining wrong-pair similarity; it does not require increasing correct-pair similarity. Bt and B0 must be read separately.',
        'Output probabilities remain a parallel reference; subtracting their Delta from hidden Delta is not a mechanism estimate.',
        'The positional binary onehot input kernel is invariant under the fixed slot/category isometries. It cannot distinguish correct from wrong in this pilot; its high similarity is an input reference.',
        'Density control adds exact anchor edge count to the same answer-stratum key. Too few samples remain undefined; the threshold3 is never loosened.',
        'Full<R,C,W> orbit exclusion prevents the specified transformations leaking across splits; arbitrary vertex-isomorphic graphs may still cross splits.',
        'Weak source-task performance remains in the report and does not trigger retraining or make an equal-learning mechanism comparison possible.',
        'This cross-system comparison also changes size, encoding, task and distribution. No domain-only causal effect is identified.',
        'EOS was measured during source validation under teacher-conditioned answer input, not during prompt-only geometry extraction.',
        'Completed fixed manifest; STOP. No further models, seeds, architectures, encodings or tuning are launched.']
    (DIAG/'report.txt').write_text('\n'.join(text)+'\n')
    from matplotlib import pyplot as plt
    fig,axs=plt.subplots(1,3,figsize=(15,4))
    for ax,metric in zip(axs,('linear_cka','knn_overlap','procrustes')):
        rr=[r for r in summary if r['branch']=='prefix_final' and r['control']=='answer_strata' and r['metric']==metric]
        for r in rr:ax.plot(['B0','Bt','dc','dw','Delta'],[number(r[k]) for k in ('B0','Bt','dc','dw','Delta')],marker='o',label=r['relation'])
        ax.axhline(0,color='gray',lw=.8);ax.set_title(metric);ax.set_ylabel('Within-metric contrast, fixed1/3 over n')
    axs[-1].legend(fontsize=8);fig.tight_layout();fig.savefig(DIAG/'answer_control_decomposition.png',dpi=150);plt.close(fig)
    (DIAG/'report.html').write_text('<!doctype html><meta charset="utf-8"><title>GraphWorld pilot</title><style>body{font:16px sans-serif;max-width:1200px;margin:32px auto}pre{white-space:pre-wrap}</style><img src="answer_control_decomposition.png" width="100%"><pre>'+html.escape('\n'.join(text))+'</pre>')
    records=[read(ROOT/'models'/j['run_id']/'completed.json') for j in c['training_manifest']]
    protected=check_protected()
    files=[p for p in ROOT.rglob('*') if p.is_file()]
    completion=dict(status='completed_and_stopped',protocol_sha256=sha(CONFIG),source_version_sha256=source_version(),
        new_source_models=4,source_updates=80000,engineering_updates=192,reused_actual_initial_states=1,
        independent_worlds=1,shared_initializations=1,relationship_instances=4,transform_types=2,n_values=3,
        sum_recorded_training_process_seconds=sum(r['elapsed_seconds'] for r in records),evaluation_wall_seconds=done['wall_seconds'],
        engineering_wall_seconds=read(DIAG/'gpu_engineering.json')['wall_seconds'],
        gpu_utilization_seconds_not_measured=True,cost_scope='Recorded process wall includes validation and checkpoint IO; not an independent measurement of GPU-active time.',
        peak_cuda_memory_bytes=max(r['peak_cuda_memory_bytes'] or 0 for r in records),storage_bytes=sum(p.stat().st_size for p in files),
        geometry_length_rows=720,geometry_summary_rows=240,source_learning_rows=240,
        undefined_length_cells=sum(r['status']=='undefined' for r in coverage),old_protected_files=protected,
        no_added_tasks_or_seeds=True,no_F17_changes=True,no_automatic_followup=True)
    json_write(DIAG/'completion.json',completion);return completion
