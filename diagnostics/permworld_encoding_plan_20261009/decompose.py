"""CSV-only addendum: no model loading, inference, CKA fitting, or training."""
import csv
from collections import defaultdict
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import html
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
WORKSPACE=ROOT.parent.parent
SOURCE=ROOT.parent/'permworld_cka_20261009'
D=Decimal


def load(path):
    with path.open() as f:return list(csv.DictReader(f))


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(2**20),b''):h.update(block)
    return h.hexdigest()


def write_json(name,data):
    (ROOT/name).write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def write_csv(name,rows):
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with (ROOT/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)


def mean(vs):return sum(vs,D(0))/D(len(vs)) if vs and all(v is not None for v in vs) else None


def median(vs):
    if not vs or any(v is None for v in vs):return None
    a=sorted(vs);m=len(a)//2
    return a[m] if len(a)%2 else (a[m-1]+a[m])/2


def component(r):
    result={key:D(r[key]) if r.get(key,'')!='' else None for key in [
        'init_correct','init_wrong','init_identity','trained_correct','trained_wrong','trained_identity']}
    def diff(a,b):return a-b if a is not None and b is not None else None
    result.update(initial_preference=diff(result['init_correct'],result['init_wrong']),
        trained_preference=diff(result['trained_correct'],result['trained_wrong']),
        delta_correct=diff(result['trained_correct'],result['init_correct']),
        delta_wrong=diff(result['trained_wrong'],result['init_wrong']),
        delta_identity=diff(result['trained_identity'],result['init_identity']))
    result['minus_delta_wrong']=-result['delta_wrong'] if result['delta_wrong'] is not None else None
    result['delta']=diff(result['trained_preference'],result['initial_preference'])
    return result


def execute():
    original_files={p.name:sha(p) for p in sorted(SOURCE.iterdir()) if p.is_file()}
    previous=json.loads((SOURCE/'deliverable_manifest.json').read_text())
    assert all(original_files[name]==h for name,h in previous['files'].items())
    source_rows=load(SOURCE/'relation_length_cka.csv');assert len(source_rows)==9072
    checks=0;maxerr=D(0)
    def compare(a,b,name):
        nonlocal checks,maxerr
        if a is None or b is None:assert a is None and b is None,name;return
        err=abs(a-b);assert err<D('1e-12'),(name,err)
        checks+=1;maxerr=max(maxerr,err)
    output=[];by_seed=defaultdict(list)
    for r in source_rows:
        comp=component(r)
        compare(comp['delta'],D(r['delta']),'cell Delta')
        compare(comp['delta'],comp['delta_correct']-comp['delta_wrong'],'component identity')
        compare(comp['initial_preference'],D(r['init_contrast']),'initial preference')
        compare(comp['trained_preference'],D(r['trained_contrast']),'trained preference')
        row={**{k:r[k] for k in ['relation','left_task','right_task','left_seed','right_seed','seed_pair_type','length','anchors','control','metric','initialization_status']},
            'encoding':'one_line',**comp}
        output.append(row);by_seed[r['relation'],r['left_seed'],r['right_seed'],r['control'],r['metric']].append(row)
    fields=list(component(source_rows[0]));seed_rows=[]
    seed_lookup={(r['relation'],r['left_seed'],r['right_seed'],r['control'],r['metric']):r for r in load(SOURCE/'relation_seed_summary.csv')}
    for key,rows in by_seed.items():
        assert sorted(int(r['length']) for r in rows)==list(range(10,31))
        base=rows[0]
        result={k:base[k] for k in ['relation','encoding','left_seed','right_seed','seed_pair_type','control','metric','initialization_status']}
        for field in fields:result[field]=mean([r[field] for r in rows])
        result['length_median_delta']=median([r['delta'] for r in rows]);result['length_weight']='1/21'
        compare(result['delta'],D(seed_lookup[key]['delta']),'equal length aggregation')
        seed_rows.append(result)
    by_relation=defaultdict(list)
    for r in seed_rows:by_relation[r['relation'],r['seed_pair_type'],r['control'],r['metric']].append(r)
    original_summary={(r['relation'],r['seed_pair_type'],r['control'],r['metric']):r for r in load(SOURCE/'relation_summary.csv')}
    summary=[]
    for key,rows in by_relation.items():
        rel,typ,control,metric=key
        result=dict(relation=rel,encoding='one_line',seed_pair_type=typ,control=control,metric=metric,pairs=len(rows))
        for field in fields:
            result[field+'_mean']=mean([r[field] for r in rows]);result[field+'_median']=median([r[field] for r in rows])
        result['delta_min']=min(r['delta'] for r in rows);result['delta_max']=max(r['delta'] for r in rows)
        compare(result['delta_mean'],D(original_summary[key]['delta_mean']),'seed pair aggregation')
        for field in ['trained_correct','trained_wrong','trained_identity','init_correct','init_wrong','init_identity']:
            compare(result[field+'_mean'],D(original_summary[key][field]),'raw summary identity')
        compare(result['delta_mean'],result['delta_correct_mean']-result['delta_wrong_mean'],'mean components')
        summary.append(result)
    write_csv('decomposition_length.csv',output);write_csv('decomposition_seed.csv',seed_rows);write_csv('decomposition_summary.csv',summary)
    baseline_rows=[];baseline_groups=defaultdict(list)
    for r in load(SOURCE/'answer_output_length_baselines.csv'):
        comp=component(r)
        row={k:r[k] for k in ['relation','length','control','branch','left_seed','right_seed','anchors']}
        row.update(encoding='one_line',baseline_type='trained_and_initial_output' if r['branch']=='output_prob' else 'fixed_truth_geometry',**comp)
        baseline_rows.append(row)
        typ=('same_seed' if r['left_seed']==r['right_seed'] else 'cross_seed') if r['left_seed'] else 'not_applicable'
        baseline_groups[r['relation'],r['control'],r['branch'],typ,r['left_seed'],r['right_seed']].append(row)
        if comp['delta'] is not None:compare(comp['delta'],D(r['delta']),'output baseline arithmetic')
    baseline_seed=defaultdict(list)
    for (rel,control,branch,typ,sa,sb),rows in baseline_groups.items():
        assert sorted(int(r['length']) for r in rows)==list(range(10,31))
        baseline_seed[rel,control,branch,typ].append({field:mean([r[field] for r in rows]) for field in fields})
    baseline_summary=[]
    for (rel,control,branch,typ),rows in baseline_seed.items():
        baseline_summary.append(dict(relation=rel,encoding='one_line',control=control,branch=branch,seed_pair_type=typ,
            **{field+'_mean':mean([r[field] for r in rows]) for field in fields},
            semantics='Truth has no initialization/training increment; constant-stratum CKA stays undefined.' if branch!='output_prob' else 'Output readout geometry, not an estimate of hidden mechanism.'))
    write_csv('baseline_decomposition_length.csv',baseline_rows);write_csv('baseline_decomposition_summary.csv',baseline_summary)
    report(summary,baseline_summary)
    # Preservation check: the prior delivered diagnosis and all originally protected study files remain identical.
    assert all(sha(SOURCE/name)==digest for name,digest in original_files.items())
    protected=json.loads((SOURCE/'protected_before.json').read_text())['files']
    assert all(sha(WORKSPACE/path)==digest for path,digest in protected.items())
    write_json('audit.json',dict(scope='Independent Decimal arithmetic and fixed-weight aggregation of supplied CSVs; no independent rerun of raw model features/CKA/SVD or previous 11,497 checks.',
        arithmetic_checks=checks,maximum_arithmetic_discrepancy=str(maxerr),
        original_diagnostic_files_unchanged=len(original_files),protected_files_unchanged=len(protected),
        sources=original_files,training_started=False,original_initial_identity_verified=False,
        new_encoding_results_available=False,completed_utc=datetime.now(timezone.utc).isoformat()))
    write_json('manifest.json',dict(files={p.name:sha(p) for p in sorted(ROOT.iterdir()) if p.is_file() and p.name!='manifest.json'},
        workspace_commit='1073dcb3be4db90ec23476bab8f503c9e00f364b',status='planning_addendum_complete_no_training'))
    print(dict(rows=len(output),seed_summaries=len(seed_rows),summary_rows=len(summary),arithmetic_checks=checks,
        max_error=str(maxerr),training_started=False),flush=True)


def report(summary,baselines):
    chosen=[r for r in summary if r['seed_pair_type']=='same_seed' and r['control']=='answer_strata' and r['metric']=='ordinary']
    def fmt(x):return 'undefined' if x is None else f'{x:+.4f}'
    headers=['Relationship','Initial preference B₀','Trained preference Bₜ','Correct-score change δc','Wrong-score change δw','Δ=δc−δw']
    grid='<table><tr>'+''.join('<th>'+h+'</th>' for h in headers)+'</tr>'
    for r in chosen:
        grid+='<tr><td>'+html.escape(r['relation'])+'</td>'+''.join('<td>'+fmt(r[f+'_mean'])+'</td>' for f in ['initial_preference','trained_preference','delta_correct','delta_wrong','delta'])+'</tr>'
    grid+='</table>'
    output_table='<table><tr><th>Relationship</th><th>Output initial preference</th><th>Output trained preference</th><th>Output Delta</th></tr>'
    for r in baselines:
        if r['control']=='answer_strata' and r['branch']=='output_prob' and r['seed_pair_type']=='same_seed':
            output_table+='<tr><td>'+r['relation']+'</td>'+''.join('<td>'+fmt(r[f+'_mean'])+'</td>' for f in ['initial_preference','trained_preference','delta'])+'</tr>'
    output_table+='</table>'
    body='''<h1>PermWorld: pairing-selectivity decomposition and encoding-control plan</h1>
<p>This supplements existing exploratory findings, without altering the original report or constituting a new confirmation. Only existing CSV were read; no inference, recomputation of CKA, or new model training.</p>
<p><strong>Directly supported by existing findings: </strong>Training changes relative preferences between transformation pairings, with dependence on initialization geometry and relationship type. Delta does not directly denote substantial convergence of correct-pair representations. Original initial-weight identity remains unverifiable; this table uses program-reconstructed references.</p>
<p>The table uses answer control, three same-seed pairs, and 21 lengths with fixed equal weights. Delta components: c=correct-trained minus correct-initial; w=wrong-trained minus wrong-initial; Delta =δc−δw. Decomposition terms are arithmetic relationships, not causal contribution percentages.</p>'''+grid+'''
<p>For both inversion relationships, wrong-pair CKA decreases approximately 0.62; correct-score changes are respectively −0.0339 and +0.0205. Delta is negative for all six complement relationships, yet their trained correct-minus-wrong means are positive. This does not imply absence of learning; CKA cannot identify an algebraic mechanism.</p>
<p>Same-seed, answer-controlled output-probability results:</p>'''+output_table+'''
<p>Subtracting within-gold-label means does not remove probability, confidence, or prediction-error structure. Retain gold-answer//output references; hidden Delta minus output Delta must not estimate an extra mechanism. Full gold baselines and all seeds//lengths: baseline_decomposition_length.csv; gold answers have no initial state, and constant gold-answer CKA remains undefined.</p>
<p>Across encodings, jointly report  Γ₀=B₀(E₁)−B₀(E₀), Γₜ=Bₜ(E₁)−Bₜ(E₀), Γc=δc(E₁)−δc(E₀), Γw=δw(E₁)−δw(E₀), together with  ΓΔ=Γₜ−Γ₀=Γc−Γw. A Delta reversal without a trained-preference reversal must not be described as a reversal of learned relationship preference; nonsignificance is not equivalence.</p>
<p>R0/k8 between-pool CKA pooled-length 0.803, per-length 0.214 remain a length-mixing sensitivity example. These do not exactly decompose initialization//input//learning contributions.</p>
<p><strong>Next-round question: </strong>How does input encoding affect training-induced reshaping of transformation-pairing selectivity?Analysis fields and interpretation rules are fixed in protocol.json; particular encodings, model budgets, data worlds, and source counts remain unfrozen. Only one_line data exist; no cross-encoding interaction scores have been generated.</p>
<p>Before formal confirmation, freeze a complete execution protocol and save actual step-0 weights/hashes, using new confirmation inputs//orbits. Identical initial parameters do not imply identical initial geometry when encoded inputs differ; compute each encoding's own initial/trained correct/wrong//identity CKA. Additional differences in vocabulary, sequence length, or architecture must be explicit.</p>
<p>Deliverables: <a href="decomposition_summary.csv">Full mean//median decomposition</a>;<a href="decomposition_seed.csv">Per-seed decomposition</a>;<a href="decomposition_length.csv">Per-length decomposition</a>;<a href="protocol.json">Next-round analysis protocol</a>;<a href="audit.json">CSV Arithmetic audit and protection checks</a>. Reproduce from the workspace with  PYTHONDONTWRITEBYTECODE=1 .venv/bin/python diagnostics/permworld_encoding_plan_20261009/decompose.py.</p>
<p><strong>This supplementary round is complete and stopped.</strong>No new training or encoding experiment; F17 and the previous diagnostic artifacts were unchanged in this round.</p>'''
    style='body{font-family:system-ui,sans-serif;max-width:1100px;margin:32px auto;line-height:1.7;padding:0 18px}table{border-collapse:collapse;width:100%;font-size:13px;margin:20px 0}td,th{border:1px solid #cdd7e0;padding:9px;text-align:left}th{background:#eef4f8}a{color:#075ba1}'
    (ROOT/'addendum.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><title>PermWorld selection decomposition</title><style>'+style+'</style><body>'+body+'</body></html>')


if __name__=='__main__':execute()
