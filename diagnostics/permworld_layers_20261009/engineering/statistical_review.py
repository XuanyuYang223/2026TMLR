"""Independent read-only summary audit; does not import frozen summary code."""
from __future__ import annotations
import csv, json, math, statistics, hashlib
from pathlib import Path
from collections import defaultdict, Counter

ROOT = Path(__file__).resolve().parents[1]
LENGTHS = set(range(10, 31))
KEYS = ('domain','cohort','relation','family','kind','branch','control','metric')
UK = KEYS + ('source_unit','seed','world')
GK = ('domain','cohort','kind','branch','control','metric')
D = ('B0','Bt','delta_correct','delta_wrong','delta')
F = ('initial_correct','initial_wrong','initial_identity','trained_correct','trained_wrong','trained_identity') + D + ('initial_correct_raw_error','initial_wrong_raw_error','initial_identity_raw_error','trained_correct_raw_error','trained_wrong_raw_error','trained_identity_raw_error')
METRICS = ('linear_cka','knn_overlap','procrustes','debiased_cka')
MAIN = METRICS[:3]
NEURAL = tuple(f'{l}_{p}' for l in ('block_01','block_02','block_03','block_04','final_norm') for p in ('end','value_mean'))
BRANCHES = NEURAL + ('embedding_end','embedding_value_mean','position_numeric_onehot','position_numeric_embedding','query_final_norm','output_prob','answer_onehot','answer_scalar')
COUNTS = Counter(); ERRORS=[]; MAXERR=defaultdict(float)

def read(name):
    with (ROOT/name).open(newline='') as h: return list(csv.DictReader(h))
def num(x):
    if x is None or x == '': return None
    z=float(x)
    if not math.isfinite(z): raise ValueError('nonfinite value')
    return z
def sign(x): return None if x is None else (x > 0) - (x < 0)
def group(rows, keys):
    out=defaultdict(list)
    for r in rows: out[tuple(r[k] for k in keys)].append(r)
    return out
def complete(v): return statistics.fmean(v) if v and all(x is not None for x in v) else None

def check(cat, loc, field, expected, observed):
    COUNTS[cat] += 1
    if isinstance(expected, bool): ok = str(observed) == str(expected)
    elif expected is None: ok = observed is None or observed == ''
    elif isinstance(expected, str): ok = str(observed) == expected
    else:
        actual=num(observed)
        err=abs(actual-expected) if actual is not None else float('inf')
        MAXERR[cat]=max(MAXERR[cat],err)
        ok=err <= 2e-11
    if not ok:
        COUNTS[cat+'_failures'] += 1
        if len(ERRORS)<30: ERRORS.append(dict(category=cat, location=list(loc) if isinstance(loc,tuple) else loc, field=field, expected=expected, observed=observed))

def stats(v):
    a=[x for x in v if x is not None]
    return dict(sources=len(a),mean=statistics.fmean(a) if a else None,median=statistics.median(a) if a else None,min=min(a) if a else None,max=max(a) if a else None,source_sd=statistics.stdev(a) if len(a)>1 else None,positive=sum(x>0 for x in a),negative=sum(x<0 for x in a),zero=sum(x==0 for x in a))

raw = read('layer_metric_results.csv')
sources = read('source_metric_summary.csv')
relations = read('relation_metric_summary.csv')
old_path=ROOT.parent/'permworld_scope_inventory_20261009'/'permutation_metric_results.csv'
with old_path.open(newline='') as h: old_rows=list(csv.DictReader(h))
old_keys=UK+('length',)
raw_lookup={tuple(r[k] for k in old_keys):r for r in raw}
check('inherited_exact','inherited','count',52080,len(old_rows))
for r in old_rows:
    fields=dict(r)
    if fields['branch']=='prefix_final_norm': fields['branch']='final_norm_end'
    key=tuple(fields[k] for k in old_keys)
    check('inherited_exact',key,'exists',True,key in raw_lookup)
    if key in raw_lookup:
        for f,v in fields.items(): check('inherited_exact',key,f,v,raw_lookup[key].get(f))
rg = group(raw, UK)
sg = group(sources, UK)
check('shape','raw','rows',187488,len(raw))
check('shape','source','rows',8928,len(sources))
check('shape','relation','rows',3312,len(relations))
check('shape','source','keys match',True,set(rg)==set(sg))

for i,r in enumerate(raw):
    q={f:num(r.get(f)) for f in F}
    for name,left,right in (('B0','initial_correct','initial_wrong'),('Bt','trained_correct','trained_wrong'),('delta_correct','trained_correct','initial_correct'),('delta_wrong','trained_wrong','initial_wrong'),('delta','Bt','B0')):
        expected=q[left]-q[right] if q[left] is not None and q[right] is not None else None
        check('raw_decomposition',i,name,expected,r[name])
    check('raw_decomposition',i,'delta dc-dw',q['delta_correct']-q['delta_wrong'] if q['delta_correct'] is not None and q['delta_wrong'] is not None else None,r['delta'])
    check('raw_decomposition',i,'trained_preference',q['Bt'],r['trained_preference'])
    check('raw_decomposition',i,'initial_preference',q['B0'],r['initial_preference'])
    check('grade',i,'grade','C',r['initialization_grade'])
    check('grade',i,'identity',False,r['original_initial_weight_identity_verified'])

recalc_source={}
for key,cells in rg.items():
    check('source_structure',key,'single row',1,len(sg[key]))
    row=sg[key][0]
    check('source_structure',key,'all lengths',True,len(cells)==21 and {int(r['length']) for r in cells}==LENGTHS)
    check('source_structure',key,'observed lengths',21,row['observed_lengths'])
    check('source_structure',key,'planned lengths',21,row['planned_lengths'])
    check('source_structure',key,'all_lengths_present',True,row['all_lengths_present'])
    check('source_structure',key,'grade','C',row['initialization_grade'])
    check('source_structure',key,'identity',False,row['original_initial_identity_verified'])
    agg=dict(zip(UK,key))
    for f in F:
        v=[num(c[f]) for c in cells]; a=[x for x in v if x is not None]
        val=complete(v)
        agg[f]=val
        check('source_mean_21',key,f,val,row[f])
        check('source_counts',key,f+'_defined_lengths',len(a),row[f+'_defined_lengths'])
        if f in D:
            for suffix,expected in [('positive_lengths',sum(x>0 for x in a)),('negative_lengths',sum(x<0 for x in a)),('zero_lengths',sum(x==0 for x in a)),('length_median',statistics.median(a) if len(a)==21 else None),('length_min',min(a) if a else None),('length_max',max(a) if a else None)]:
                agg[f+'_'+suffix]=expected
                check('source_length_descriptives',key,f+'_'+suffix,expected,row[f+'_'+suffix])
    recalc_source[key]=agg

rr = group(list(recalc_source.values()),KEYS)
rr_csv=group(relations,KEYS)
check('relation_structure','relations','keys match',True,set(rr)==set(rr_csv))
recalc_relation={}
for key,items in rr.items():
    check('relation_structure',key,'single row',1,len(rr_csv[key])); row=rr_csv[key][0]
    check('relation_structure',key,'planned_sources',len(items),row['planned_sources'])
    check('relation_structure',key,'source_units',';'.join(sorted(r['source_unit'] for r in items)),row['source_units'])
    check('relation_structure',key,'uncertainty','descriptive initialization variation; one fixed shared data world',row['uncertainty_scope'])
    agg=dict(zip(KEYS,key))
    for f in F:
        st=stats([r[f] for r in items])
        for suffix,expected in st.items():
            agg[f+'_'+suffix]=expected
            check('relation_stats',key,f+'_'+suffix,expected,row[f+'_'+suffix])
        check('relation_stats',key,f+'_all_planned_sources_defined',st['sources']==len(items),row[f+'_all_planned_sources_defined'])
    for f in D:
        for direction in ['positive','negative']:
            check('relation_stats',key,f+'_'+direction+'_all_lengths_sources',sum(r[f+'_'+direction+'_lengths']==21 for r in items),row[f+'_'+direction+'_all_lengths_sources'])
    recalc_relation[key]=agg

length_rows=read('length_metric_summary.csv'); length_groups=group(raw,KEYS+('length',))
length_lookup=group(length_rows,KEYS+('length',))
check('length_structure','keys','match',True,set(length_groups)==set(length_lookup))
for key,items in length_groups.items():
    check('length_structure',key,'single row',1,len(length_lookup[key])); row=length_lookup[key][0]
    check('length_structure',key,'planned_sources',len(items),row['planned_sources'])
    for f in D:
        for suffix,expected in stats([num(r[f]) for r in items]).items():
            check('length_stats',key,f+'_'+suffix,expected,row[f+'_'+suffix])

lr=group(list(recalc_relation.values()),GK)
leave=read('leave_one_relation_family.csv'); leave_keys=GK+('exclusion_level','excluded')
leave_lookup=group(leave,leave_keys)
expected_leave=set()
for key,items in lr.items():
    for level,excluded in [('none','')]+[('relation',v) for v in sorted({r['relation'] for r in items})]+[('family',v) for v in sorted({r['family'] for r in items})]:
        lk=key+(level,excluded); expected_leave.add(lk)
        check('leave_structure',lk,'single row',1,len(leave_lookup.get(lk,[])))
        if lk not in leave_lookup: continue
        row=leave_lookup[lk][0]
        retained=[r for r in items if level=='none' or r[level]!=excluded]
        check('leave_structure',lk,'planned',len(items),row['planned_relations'])
        check('leave_structure',lk,'remaining',len(retained),row['remaining_relations'])
        check('leave_structure',lk,'independent_relation_inference',False,row['independent_relation_inference'])
        for f in D:
            v=[r[f+'_mean'] for r in retained]
            check('leave_values',lk,f,complete(v),row[f])
            check('leave_values',lk,f+'_defined_relations',sum(x is not None for x in v),row[f+'_defined_relations'])
check('leave_structure','keys','match',True,set(leave_lookup)==expected_leave)

layer_rows=read('layer_metric_summary.csv'); layer_lookup=group(layer_rows,GK)
check('layer_structure','keys','match',True,set(layer_lookup)==set(lr))
for key,items in lr.items():
    check('layer_structure',key,'single row',1,len(layer_lookup[key])); row=layer_lookup[key][0]
    check('layer_structure',key,'planned',len(items),row['planned_relations'])
    check('layer_structure',key,'families',len({r['family'] for r in items}),row['families'])
    for f in D:
        v=[r[f+'_mean'] for r in items]; a=[x for x in v if x is not None]
        for suffix,expected in [('equal_relation_mean',complete(v)),('median_relation_mean',statistics.median(a) if a else None),('defined_relations',len(a)),('positive_relations',sum(x>0 for x in a)),('negative_relations',sum(x<0 for x in a))]:
            check('layer_values',key,f+'_'+suffix,expected,row[f+'_'+suffix])

AK=KEYS[:-1]+('source_unit','seed','world')
agroups=group(list(recalc_source.values()),AK)
agcsv=group(read('direction_agreement.csv'),AK)
dbcsv=group(read('debiased_sensitivity.csv'),AK)
for key,items in agroups.items():
    lookup={r['metric']:r for r in items}
    check('metric_completeness',key,'fixed four',True,set(lookup)==set(METRICS))
    check('direction_structure',key,'agreement row',1,len(agcsv[key]))
    check('direction_structure',key,'debiased row',1,len(dbcsv[key]))
    arow=agcsv[key][0]; brow=dbcsv[key][0]
    for f in D:
        vals=[lookup[m][f] for m in MAIN]; defined=all(x is not None for x in vals)
        check('direction',key,f+'_all_three_defined',defined,arow[f+'_all_three_defined'])
        check('direction',key,f+'_all_three_direction_agree',len({sign(x) for x in vals})==1 if defined else None,arow[f+'_all_three_direction_agree'])
        check('direction',key,f+'_all_three_positive',all(x>0 for x in vals) if defined else None,arow[f+'_all_three_positive'])
        for m,v in zip(MAIN,vals): check('direction',key,f+'_'+m,v,arow[f+'_'+m])
        a,b=lookup['linear_cka'][f],lookup['debiased_cka'][f]; defined=a is not None and b is not None
        for suffix,expected in [('ordinary',a),('debiased',b),('both_defined',defined),('direction_agree',sign(a)==sign(b) if defined else None)]:
            check('debiased',key,f+'_'+suffix,expected,brow[f+'_'+suffix])

basek=('domain','cohort','relation','family','kind','source_unit','seed','world')
for key,items in group(sources,basek).items():
    check('branch_completeness',key,'all branches',True,{r['branch'] for r in items}==set(BRANCHES))
    check('branch_completeness',key,'all metrics/controls combinations',True,len(items)==18*4*2 and len({(r['branch'],r['metric'],r['control']) for r in items})==144)
manifest=read('figure_manifest.csv')
check('plot_completeness','figures','count',30,len(manifest))
for r in manifest: check('plot_completeness',r['file'],'exists',True,(ROOT/r['file']).is_file())
check('plot_completeness','metric_figures','all cohorts/controls/metrics',True,{(r['cohort'],r['control'],r['metric']) for r in manifest if r['file'].startswith('layers_')}=={(c,x,m) for c in {r['cohort'] for r in raw} for x in ['answer_strata','within_length'] for m in METRICS})

# Every relation/layer/metric/length remains in the exported review, no selection.
profile=[]
for r in relations:
    if r['branch'] in NEURAL and r['control']=='answer_strata':
        profile.append({k:r[k] for k in ['cohort','relation','family','kind','branch','metric','planned_sources']+ [f+'_mean' for f in D]+['delta_min','delta_max','delta_positive','delta_negative','delta_sources','delta_positive_all_lengths_sources','delta_negative_all_lengths_sources']})

inverse=[]
for r in relations:
    if r['cohort']=='specialist16' and r['relation'] in ('descent_inverse','exceedance_inverse') and r['branch'] in NEURAL:
        inverse.append({k:r[k] for k in ['cohort','relation','branch','control','metric','planned_sources']+[f+'_mean' for f in F[:11]]+['delta_min','delta_max','delta_positive_all_lengths_sources','delta_negative_all_lengths_sources']})

missing=Counter((r['cohort'],r['branch'],r['control'],r['metric']) for r in sources if not r['delta'])
partial=Counter((r['cohort'],r['branch'],r['control'],r['metric']) for r in sources if 0<int(r['delta_defined_lengths'])<21)
review=dict(status='PASS' if not ERRORS else 'FAIL',read_only=True,new_source_training=0,independent_of_frozen_summary_implementation=True,length_weights={str(n):1/21 for n in range(10,31)},missing_length_reweighting=False,checks=dict(COUNTS),max_absolute_numeric_discrepancy=dict(MAXERR),first_errors=ERRORS,
            raw_rows=len(raw),source_rows=len(sources),relation_rows=len(relations),length_rows=len(length_rows),leave_rows=len(leave),layer_rows=len(layer_rows),fixed_metrics=list(MAIN),fixed_sensitivity_metric='debiased_cka',neural_branches=list(NEURAL),input_and_output_branches=list(BRANCHES[10:]),
            initialization_status='All historical sources are grade C constructor reconstructions; original step-0 identity unverified.',
            inference_scope='Descriptive source variation within shared cohort worlds; dependent relation/family removals are sensitivity only, not iid confidence intervals; fixed inputs, no fabricated input uncertainty.',
            inherited_reference_sha256=hashlib.sha256(old_path.read_bytes()).hexdigest(),inherited_metric_cells=52080,
            undefined_delta_source_cells=[dict(cohort=k[0],branch=k[1],control=k[2],metric=k[3],count=v) for k,v in sorted(missing.items())],
            partial_defined_lengths_source_cells=[dict(cohort=k[0],branch=k[1],control=k[2],metric=k[3],count=v) for k,v in sorted(partial.items())],
            full_neural_answer_control_profile=profile,two_inverse_decomposition_profile=inverse,
            sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in ['layer_metric_results.csv','source_metric_summary.csv','relation_metric_summary.csv','length_metric_summary.csv','leave_one_relation_family.csv','layer_metric_summary.csv','direction_agreement.csv','debiased_sensitivity.csv','figure_manifest.csv','protocol.json','summarize.py']})
(ROOT/'statistical_review.json').write_text(json.dumps(review,indent=2,ensure_ascii=False)+'\n')
print(json.dumps({k:review[k] for k in ['status','checks','max_absolute_numeric_discrepancy','first_errors','raw_rows','source_rows','relation_rows']},indent=2))
