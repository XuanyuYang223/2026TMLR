"""Streaming saved-output reader for the fixed local-state timing comparison.

This file fits no probe, changes no state, runs no model, and selects no
condition. Early/late effects are formed within the same pair before fixed
center/length aggregation. Eight random directions remain paired controls.
"""
from __future__ import annotations

import argparse
from array import array
from collections import defaultdict
import csv
from datetime import datetime, timezone
import gzip
from itertools import groupby,islice
import json
import hashlib
import math
from pathlib import Path
import statistics
import sys
import tempfile

HERE=Path(__file__).resolve().parent
UNITS=('archdev001','unit02','unit03','unit04','unit05')
FORMAL=UNITS[1:]
TASKS=('peaks','valleys')
GROUPS=('main','matched')
LENGTHS=tuple(range(10,31))
RANDOMS=tuple(f'random_direction_{s}' for s in range(1026,1034))
COMMON=('unedited','local_projection','same_label_projection',*RANDOMS)
EXPECTED={*(('late',c) for c in COMMON),*(('early',c) for c in COMMON),
          *(('late',f'late_layer_swap_{k}') for k in range(1,5)),
          ('early','early_full_layer1'),('diagnostic','full_state_swap'),
          ('diagnostic','actual_counterfactual')}
PRIMARY='Fixed 1/21 lengths after equal registered centers'
METRICS=('checker_boundary_crossing','checker_desired_side','checker_signed_shift',
         'relative_step_hit','desired_answer_hit','any_numeric_count_change',
         'other_numeric_count_change','any_argmax_token_change','numeric_argmax_coverage',
         'edited_nonnumeric_rate','signed_argmax_count_delta','signed_numeric_expectation_delta',
         'edited_numeric_mass','numeric_mass_delta','recipient_count_accuracy','reference_oracle_hit',
         'planned_edit_l2','edit_l2','carried_state_l2norm')
TOL=1e-10
NORM_ATOL=NORM_RTOL=1e-5


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def number(value):
    if value is None or str(value).strip().lower() in ('','none','nan','null'):return None
    result=float(value)
    return result if math.isfinite(result) else None


def boolean(value):return str(value).lower() not in ('false','0','none','nan','no','')


def delta(a,b):return a-b if a is not None and b is not None else None
def mean(values):return math.fsum(values)/len(values)


def read_json(path):return json.loads(Path(path).read_text())


def write_json(path,value):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def write_csv(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fields=list(dict.fromkeys(k for r in rows for k in r)) or ['status']
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)


def score(raw):
    r=dict(raw)
    for field in ('length','position','sign','recipient_local_label','desired_local_label',
                  'recipient_true_count','desired_true_count','recipient_argmax_count','edited_argmax_count',
                  'recipient_argmax_token','edited_argmax_token'):
        value=number(r[field])
        if value is None or value!=int(value):raise ValueError(('Invalid integer',field,r[field]))
        r[field]=int(value)
    for field in ('unit','task','group','timing','condition'):r[field]=sys.intern(r[field])
    if (r['unit'] not in UNITS or r['task'] not in TASKS or r['group'] not in GROUPS or
            r['sign'] not in (-1,1) or r['length'] not in LENGTHS or
            (r['timing'],r['condition']) not in EXPECTED or
            not (6 if r['group']=='matched' else 2)<=r['position']<=r['length']-2):
        raise ValueError(('Unregistered timing condition',r))
    if r['desired_local_label']-r['recipient_local_label']!=r['sign'] or r['desired_true_count']!=r['recipient_true_count']+r['sign']:
        raise ValueError('Local/answer endpoint differs from recipient+sign')
    if number(r['probe_boundary'])!=.5:raise ValueError('Frozen checker boundary is not0.5')
    available=boolean(r.get('condition_available','true'))
    before=number(r['recipient_local_score'])
    if before is None:raise ValueError('Independent baseline checker is missing')
    base=dict(unit=r['unit'],task=r['task'],group=r['group'],sign=r['sign'],pair_id=r['pair_id'],
              length=r['length'],position=r['position'],timing=r['timing'],condition=r['condition'],
              condition_available=available,missing_reason=r.get('missing_reason',''),
              checker_baseline_correct=int(before>=.5)==r['recipient_local_label'],
              recipient_final_correct=r['recipient_argmax_count']==r['recipient_true_count'])
    if not available:return dict(base,**{m:None for m in METRICS})
    after=number(r['edited_local_score'])
    if after is None:raise ValueError('Available edited checker is undefined')
    rb,eb=r['recipient_argmax_count'],r['edited_argmax_count']
    rt,et=r['recipient_argmax_token'],r['edited_argmax_token']
    if not (-1<=rb<=30 and -1<=eb<=30 and 0<=rt<188 and 0<=et<188):raise ValueError('Argmax outside frozen vocabulary')
    if rb!=(rt if rt<=30 else -1) or eb!=(et if et<=30 else -1):raise ValueError('Full-token and numeric argmax disagree')
    numeric=rb>=0 and eb>=0
    difference=eb-rb if numeric else None
    desired_side=int(after>=.5)==r['desired_local_label']
    e0,e1=number(r['recipient_numeric_expectation']),number(r['edited_numeric_expectation'])
    p0,p1=number(r['recipient_numeric_mass']),number(r['edited_numeric_mass'])
    if any(v is None or not -TOL<=v<=30+TOL for v in (e0,e1)):raise ValueError('Numeric expectation outside0..30')
    if any(v is None or not -TOL<=v<=1+TOL for v in (p0,p1)):raise ValueError('Numeric mass outside0..1')
    norm={m:number(r.get(m)) for m in ('planned_edit_l2','edit_l2','carried_state_l2norm')}
    if any(v is None or v<0 for v in norm.values()):raise ValueError('Available edit norms invalid')
    oracle=number(r.get('condition_oracle_count'))
    return dict(base,
        checker_boundary_crossing=float(base['checker_baseline_correct'] and desired_side),
        checker_desired_side=float(desired_side),checker_signed_shift=r['sign']*(after-before),
        relative_step_hit=float(numeric and difference==r['sign']),
        desired_answer_hit=float(eb==r['desired_true_count']),
        any_numeric_count_change=float(numeric and difference!=0),
        other_numeric_count_change=float(numeric and difference not in (0,r['sign'])),
        any_argmax_token_change=float(et!=rt),numeric_argmax_coverage=float(numeric),
        edited_nonnumeric_rate=float(eb<0),signed_argmax_count_delta=r['sign']*difference if numeric else None,
        signed_numeric_expectation_delta=r['sign']*(e1-e0),edited_numeric_mass=p1,numeric_mass_delta=p1-p0,
        recipient_count_accuracy=float(rb==r['recipient_true_count']),
        reference_oracle_hit=float(eb==oracle) if oracle is not None else None,**norm)


def same_value(a,b,field):
    x,y=number(a.get(field)),number(b.get(field))
    return (x is None and y is None) or (x is not None and y is not None and abs(x-y)<=TOL)


def pair_rows(raw):
    ix={(r['timing'],r['condition']):r for r in raw}
    if len(raw)!=29 or len(ix)!=29 or set(ix)!=EXPECTED:raise ValueError('Expected exactly29 distinct rows per pair')
    reference=ix['late','local_projection']
    for r in raw:
        for field in ('length','position','sign','recipient_local_label','desired_local_label','recipient_true_count',
                      'desired_true_count','recipient_argmax_count','recipient_argmax_token','recipient_numeric_expectation',
                      'recipient_local_score','probe_boundary'):
            if not same_value(r,reference,field):raise ValueError(('Paired recipient/endpoint mismatch',field))
    scored={k:score(v) for k,v in ix.items()}
    checks=[]
    for condition in COMMON:
        a,b=ix['early',condition],ix['late',condition]
        sa,sb=scored['early',condition],scored['late',condition]
        if sa['condition_available']!=sb['condition_available']:raise ValueError('Early/late availability differs')
        if not sa['condition_available']:continue
        if not a.get('first_layer_state_sha256') or a['first_layer_state_sha256']!=b['first_layer_state_sha256']:
            raise ValueError(('Injected L1 differs across timings',condition))
        for field in ('edited_local_score','edit_probe_edited_score','edited_prefix_count_prediction','planned_edit_l2','edit_l2'):
            if not same_value(a,b,field):raise ValueError(('Same L1 but checker/norm mismatch',condition,field))
        checks.append(dict(condition=condition,first_layer_sha_equal=True,checker_score_equal=True))
    if ix['early','early_full_layer1'].get('first_layer_state_sha256')!=ix['late','late_layer_swap_1'].get('first_layer_state_sha256'):
        raise ValueError('Auxiliary early/late full-L1 swap do not inject identical L1')
    for timing in ('early','late'):
        treatment=scored[timing,'local_projection']
        for random in RANDOMS:
            r=scored[timing,random]
            if not (r['condition_available'] and treatment['condition_available']):continue
            if abs(r['planned_edit_l2']-treatment['planned_edit_l2'])>TOL:raise ValueError('Planned random L1 norm not matched')
            if abs(r['edit_l2']-treatment['edit_l2'])>NORM_ATOL+NORM_RTOL*treatment['edit_l2']:raise ValueError('FP32 random L1 norm not matched')
    output=list(scored.values())
    for timing in ('early','late'):
        random=dict(scored[timing,RANDOMS[0]],condition='random_mean')
        random['condition_available']=all(scored[timing,c]['condition_available'] for c in RANDOMS)
        random['missing_reason']='' if random['condition_available'] else 'At least one of eight random controls unavailable'
        for m in METRICS:
            values=[scored[timing,c][m] for c in RANDOMS]
            random[m]=mean(values) if all(v is not None for v in values) else None
        scored[timing,'random_mean']=random;output.append(random)
    effects={}
    for condition in ('local_projection','same_label_projection','random_mean'):
        a,b=scored['early',condition],scored['late',condition]
        r=dict(a,timing='contrast',condition='early-minus-late__'+condition,
               condition_available=a['condition_available'] and b['condition_available'],
               missing_reason='' if a['condition_available'] and b['condition_available'] else 'Early/late not jointly available')
        for m in METRICS:r[m]=delta(a[m],b[m])
        effects[condition]=r;output.append(r)
    for control in ('same_label_projection','random_mean'):
        a,b=effects['local_projection'],effects[control]
        r=dict(a,condition='timing-did__local-minus-'+control,
               condition_available=a['condition_available'] and b['condition_available'],
               missing_reason='' if a['condition_available'] and b['condition_available'] else 'Four-way DiD support incomplete')
        for m in METRICS:r[m]=delta(a[m],b[m])
        output.append(r)
    a,b=scored['early','early_full_layer1'],scored['late','late_layer_swap_1']
    r=dict(a,timing='contrast',condition='early-minus-late__full-layer1',condition_available=a['condition_available'] and b['condition_available'])
    for m in METRICS:r[m]=delta(a[m],b[m])
    output.append(r)
    return output,checks


def supports(row):
    values=['all_pairs']
    if row['position']>=6:values.append('common_positions_ge6')
    if row['checker_baseline_correct']:values.append('checker_baseline_correct')
    if row['recipient_final_correct']:values.append('recipient_final_correct_diagnostic')
    return values


class Accumulator:
    __slots__=('pairs','available','sums','finite')
    def __init__(self):
        self.pairs=self.available=0;self.sums=array('d',[0.]*len(METRICS));self.finite=array('I',[0]*len(METRICS))
    def add(self,r):
        self.pairs+=1;self.available+=r['condition_available']
        for j,m in enumerate(METRICS):
            if r[m] is not None:self.sums[j]+=r[m];self.finite[j]+=1


def aggregate(cells,pooled,out):
    lengths={};position_count=0
    with gzip.open(Path(out)/'by_position.csv.gz','wt',newline='',compresslevel=1) as f:
        writer=None
        for key,a in sorted(cells.items()):
            base=dict(zip(('unit','task','group','sign','timing','condition','support','length','position'),key))
            row=dict(base,pairs=a.pairs,available_pairs=a.available)
            lkey=key[:-1]
            if lkey not in lengths:lengths[lkey]=dict(positions=set(),pairs=0,available=0,sums=[0.]*len(METRICS),finite_positions=[0]*len(METRICS),finite_pairs=[0]*len(METRICS))
            l=lengths[lkey];l['positions'].add(base['position']);l['pairs']+=a.pairs;l['available']+=a.available
            for j,m in enumerate(METRICS):
                v=a.sums[j]/a.finite[j] if a.finite[j] else None
                row[m]=v;row[m+'_finite_pairs']=a.finite[j];l['finite_pairs'][j]+=a.finite[j]
                if v is not None:l['sums'][j]+=v;l['finite_positions'][j]+=1
            if writer is None:writer=csv.DictWriter(f,fieldnames=list(row));writer.writeheader()
            writer.writerow(row);position_count+=1
    units=defaultdict(dict);length_rows=[]
    for key,a in sorted(lengths.items()):
        base=dict(zip(('unit','task','group','sign','timing','condition','support','length'),key))
        n=base['length'];lower=6 if base['group']=='matched' or base['support']=='common_positions_ge6' else 2
        expected=set(range(lower,n-1));present=a['positions']
        row=dict(base,pairs=a['pairs'],available_pairs=a['available'],registered_positions=len(expected),present_positions=len(present),missing_positions=';'.join(map(str,sorted(expected-present))))
        for j,m in enumerate(METRICS):
            count=a['finite_positions'][j]
            row[m]=a['sums'][j]/count if count==len(expected) and expected==present else None
            row[m+'_available_cell_mean']=a['sums'][j]/count if count else None
            row[m+'_finite_positions']=count;row[m+'_finite_pairs']=a['finite_pairs'][j]
        length_rows.append(row);units[key[:-1]][n]=row
    result=[]
    for key,group in sorted(units.items()):
        base=dict(zip(('unit','task','group','sign','timing','condition','support'),key))
        for j,m in enumerate(METRICS):
            values=[group[n][m] for n in LENGTHS if n in group and group[n][m] is not None]
            available=[group[n][m+'_available_cell_mean'] for n in LENGTHS if n in group and group[n][m+'_available_cell_mean'] is not None]
            meta=dict(base,metric=m,finite_lengths=len(values),missing_lengths=';'.join(str(n) for n in LENGTHS if n not in group),undefined_lengths=';'.join(str(n) for n in LENGTHS if n in group and group[n][m] is None),pairs=sum(r['pairs'] for r in group.values()),available_pairs=sum(r['available_pairs'] for r in group.values()),finite_pairs=sum(r[m+'_finite_pairs'] for r in group.values()),finite_positions=sum(r[m+'_finite_positions'] for r in group.values()))
            result.append(dict(meta,value=mean(values) if len(values)==21 else None,aggregation=PRIMARY))
            result.append(dict(meta,value=mean(available) if available else None,aggregation='Available-cell sensitivity'))
            p=pooled[key]
            result.append(dict(meta,value=p.sums[j]/p.finite[j] if p.finite[j] else None,aggregation='Pooled-pair sensitivity'))
    write_csv(Path(out)/'by_length.csv',length_rows)
    return result,position_count


def cohort(rows):
    keys=('task','group','sign','timing','condition','support','metric','aggregation')
    groups=defaultdict(dict)
    for r in rows:groups[tuple(r[k] for k in keys)][r['unit']]=r
    result=[]
    for key,g in sorted(groups.items()):
        chosen=[(u,g.get(u,{}).get('value')) for u in FORMAL];values=[v for _,v in chosen if v is not None];full=len(values)==4
        result.append(dict(zip(keys,key),registered_units=4,finite_units=len(values),mean=mean(values) if full else None,sample_sd=statistics.stdev(values) if full else None,minimum=min(values) if full else None,maximum=max(values) if full else None,positive_units=sum(v>0 for v in values),negative_units=sum(v<0 for v in values),zero_units=sum(v==0 for v in values),raw_unit_values=';'.join(u+':'+str(v) for u,v in chosen),undefined_reason='' if full else 'Missing/undefined unit; no reweighting'))
    return result


def derive(sources,out):
    """Sources is a sequence of (registered unit, iterable of raw rows)."""
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    cells=defaultdict(Accumulator);pooled=defaultdict(Accumulator);raw_count=pair_count=scored_count=hash_checks=0
    with gzip.open(out/'scored_pairs.csv.gz','wt',newline='',compresslevel=1) as f:
        writer=None
        for unit,source in sources:
            seen=set();blocks=set();previous_block=None
            pair_key=lambda r:(r['unit'],r['task'],r['group'],int(r['sign']),r['pair_id'])
            for key,values in groupby(source,pair_key):
                if key[0]!=unit or key in seen:raise ValueError(('Unexpected/repeated pair',key))
                seen.add(key);raw=list(values);raw_count+=len(raw)
                block=(raw[0]['task'],int(raw[0]['length']),int(raw[0]['position']))
                if block!=previous_block:
                    if block in blocks:raise ValueError('A task/n/i block recurred; source is not case-major')
                    blocks.add(block);previous_block=block
                rows,checks=pair_rows(raw);pair_count+=1;hash_checks+=len(checks)
                for r in rows:
                    if writer is None:writer=csv.DictWriter(f,fieldnames=list(r));writer.writeheader()
                    writer.writerow(r);scored_count+=1
                    for support in supports(r):
                        base=tuple(r[k] for k in ('unit','task','group','sign','timing','condition'))+(support,)
                        cells[base+(r['length'],r['position'])].add(r);pooled[base].add(r)
    units,position_count=aggregate(cells,pooled,out)
    stats=cohort(units)
    write_csv(out/'per_unit.csv',units);write_csv(out/'four_unit_summary.csv',stats)
    write_csv(out/'timing_contrasts_by_unit.csv',[r for r in units if r['timing']=='contrast'])
    write_csv(out/'timing_contrast_summary.csv',[r for r in stats if r['timing']=='contrast'])
    render_report(units,out)
    return dict(status='completed_saved_timing_summary',raw_rows=raw_count,pairs=pair_count,scored_rows=scored_count,position_rows=position_count,exact_l1_hash_comparisons=hash_checks,formal_units=list(FORMAL),historical_engineering_reference=UNITS[0],streamed_case_major=True,no_fits=True,no_forwards=True,no_metric_or_condition_selection=True,main_interpretation='Interlayer path at this current update, not precise accumulation time or a natural one-dimensional predicate mechanism')


def fmt(v):return 'undefined' if v is None else f'{v:.6g}'


def render_report(units,out):
    ix={tuple(r[k] for k in ('unit','task','group','sign','timing','condition','support','metric','aggregation')):r for r in units}
    lines=['# Local recurrent intervention: early versus late timing','',
        'Frozen old pairs, vectors, classifiers and dose are reused. Early and late inject **exactly the same first-layer state**, checked by SHA and checker-score equality. Late retains recipient upper-layer states after the current value token; early replays layers2–4 from their recipient previous-token states. Thus this compares one current interlayer path, not independent local signals or independent replications.','',
        'Primary relative step success is edited numeric argmax minus recipient numeric argmax equal to the requested sign. It is reported separately from predicting the true recipient answer plus that sign. Nonnumeric outputs, other count changes and full-token changes remain explicit. Per-pair early-minus-late and difference-in-differences against same-label and mean-eight-random controls are computed before fixed equal-center, then fixed1/21-length averaging. Missing required cells remain undefined.','',
        '|Unit|Task|Group|Sign|Late relative hit|Early relative hit|Late desired-answer hit|Early desired-answer hit|Early−late relative hit|DiD vs random|DiD vs same label|',
        '|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for u in UNITS:
        for t in TASKS:
            for g in GROUPS:
                for s in (-1,1):
                    get=lambda timing,condition,metric:ix.get((u,t,g,s,timing,condition,'all_pairs',metric,PRIMARY),{}).get('value')
                    values=[get('late','local_projection','relative_step_hit'),get('early','local_projection','relative_step_hit'),get('late','local_projection','desired_answer_hit'),get('early','local_projection','desired_answer_hit'),get('contrast','early-minus-late__local_projection','relative_step_hit'),get('contrast','timing-did__local-minus-random_mean','relative_step_hit'),get('contrast','timing-did__local-minus-same_label_projection','relative_step_hit')]
                    lines.append('|'+'|'.join([u,t,g,str(s),*(fmt(v) for v in values)])+'|')
    lines+=['','All original and auxiliary swaps, true-counterfactual/full-state wiring controls, checker/readout values, numeric mass, other-count/token changes, baseline-correct diagnostic subsets, available-cell and pooled sensitivities remain in the tables. Physical reference-oracle accuracy is separate from the mixed-state desired local endpoint. Higher-layer replay can change the total carried-state perturbation; only the injected first-layer norm is matched.','',
            'The local construction couples incoming token rank to the predicate label. An early effect would support a functionally effective current interlayer route under this edit, not isolate an abstract predicate or natural one-dimensional lemma. A late null effect alone cannot establish earlier irrelevance. No methods, doses, layers, positions, metrics or units are selected using these outcomes. The four units, not pairs/positions/timings/random directions, are repetitions. This stage stops after its report.','']
    (Path(out)/'report.md').write_text('\n'.join(lines))


def synthetic_sources():
    for unit in UNITS:
        def rows(u=unit):
            for task in TASKS:
                for n in LENGTHS:
                    for i in range(2,n-1):
                        for group in GROUPS:
                            if group=='matched' and i<6:continue
                            for sign in (-1,1):
                                for timing,condition in sorted(EXPECTED):
                                    changed=timing=='early' and condition in ('local_projection','early_full_layer1') or condition=='actual_counterfactual' or condition=='full_state_swap' and group=='main'
                                    local_changed=condition=='local_projection'
                                    label=int(sign<0);before=.8 if label else .2;after=1-before if local_changed else before
                                    arg=4+sign if changed else 4
                                    norm=1. if local_changed or condition in RANDOMS else 0.
                                    yield dict(unit=u,task=task,group=group,sign=sign,pair_id=f'{task}:{n}:{i}:{group}:{sign}',length=n,position=i,timing=timing,condition=condition,
                                               recipient_local_label=label,desired_local_label=1-label,recipient_true_count=4,desired_true_count=4+sign,
                                               recipient_local_score=before,edited_local_score=after,probe_boundary=.5,
                                               recipient_argmax_count=4,edited_argmax_count=arg,recipient_argmax_token=4,edited_argmax_token=arg,
                                               recipient_numeric_expectation=4.,edited_numeric_expectation=float(arg),recipient_numeric_mass=1.,edited_numeric_mass=1.,
                                               planned_edit_l2=norm,edit_l2=norm,carried_state_l2norm=norm+(1. if timing=='early' else 0.),
                                               edit_probe_edited_score=after,edited_prefix_count_prediction=3.,
                                               condition_oracle_count=4+sign if condition=='actual_counterfactual' or condition=='full_state_swap' and group=='main' else 4,
                                               first_layer_state_sha256='same-L1-'+condition if condition not in ('early_full_layer1','late_layer_swap_1') else 'same-full-L1')
        yield unit,rows()


def synthetic_checks():
    with tempfile.TemporaryDirectory(prefix='timing_summary_synthetic_') as tmp:
        result=derive(synthetic_sources(),tmp)
        with (Path(tmp)/'timing_contrast_summary.csv').open() as f:rows=list(csv.DictReader(f))
        selected=[r for r in rows if r['condition'].startswith('timing-did__') and r['metric']=='relative_step_hit']
        assert selected and all(float(r['mean'])==1 and float(r['sample_sd'])==0 for r in selected)
        assert all(r['registered_units']=='4' for r in selected)
    first=list(islice(next(synthetic_sources())[1],29))
    # Distinguish a one-step change from merely landing on the true-answer goal.
    r=next(dict(x) for x in first if x['condition']=='local_projection' and x['timing']=='early')
    r.update(recipient_argmax_count=2,recipient_argmax_token=2,edited_argmax_count=2+r['sign'],edited_argmax_token=2+r['sign'])
    s=score(r);assert s['relative_step_hit']==1 and s['desired_answer_hit']==0
    r.update(edited_argmax_count=-1,edited_argmax_token=90)
    s=score(r);assert s['relative_step_hit']==0 and s['numeric_argmax_coverage']==0 and s['any_argmax_token_change']==1
    broken=[dict(x) for x in first]
    next(x for x in broken if x['condition']=='local_projection' and x['timing']=='early')['first_layer_state_sha256']='wrong'
    try:pair_rows(broken)
    except ValueError as error:assert 'L1 differs' in str(error)
    else:raise AssertionError('L1 hash mismatch not rejected')
    return dict(status='passed',synthetic_only=True,source_sha256=sha(__file__),checks=['streamed29rowpairs','relative step differs from oracle answer','nonnumeric outcomes retained','identical L1 hash/checker across timings','within-pair random average','within-pair early/late then DiD','equal centers then fixed21 lengths','four new units only for SD'],example_result=result)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--synthetic',action='store_true');p.add_argument('--freeze',action='store_true');p.add_argument('--out',type=Path,default=HERE/'summary');p.add_argument('--protocol',type=Path,default=HERE/'protocol.json');args=p.parse_args()
    if args.synthetic:
        result=synthetic_checks();write_json(HERE/'summary_synthetic_verification.json',result);print(json.dumps(result,indent=2));return
    if not args.protocol.exists():raise FileNotFoundError('Freeze protocol before outcome access')
    if args.freeze:
        v=read_json(HERE/'summary_synthetic_verification.json');assert v['source_sha256']==sha(__file__) and v['status']=='passed'
        value=dict(source_sha256=sha(__file__),protocol_sha256=sha(args.protocol),created_utc=datetime.now(timezone.utc).isoformat(),synthetic_verification=v,no_new_outcome_access_during_development=True)
        path=HERE/'summary_freeze.json'
        if path.exists():
            old=read_json(path);assert all(old[k]==value[k] for k in ('source_sha256','protocol_sha256'));value=old
        else:write_json(path,value)
        print(json.dumps(value,indent=2));return
    frozen=read_json(HERE/'summary_freeze.json');assert frozen['source_sha256']==sha(__file__) and frozen['protocol_sha256']==sha(args.protocol)
    files=[(u,HERE/'results'/u/'pair_results.csv') for u in UNITS]
    handles=[];sources=[];provenance=[]
    try:
        for unit,path in files:
            digest=sha(path);f=path.open(newline='');handles.append(f);sources.append((unit,csv.DictReader(f)));provenance.append(dict(unit=unit,path=str(path),sha256=digest))
        result=derive(sources,args.out)
    finally:
        for f in handles:f.close()
    assert all(sha(r['path'])==r['sha256'] for r in provenance)
    result.update(completed_utc=datetime.now(timezone.utc).isoformat(),source_sha256=sha(__file__),protocol_sha256=sha(args.protocol),source_tables=provenance)
    write_json(args.out/'completion.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
