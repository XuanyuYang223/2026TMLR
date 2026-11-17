"""Frozen arithmetic-only reader for local recurrent-state interventions.

No model forwards, probe fits, layer selection, intervention construction or
outcome-dependent thresholding occur here. Random directions are paired
controls within recipients; the four new units are the replication units.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import tempfile

HERE = Path(__file__).resolve().parent
UNITS = ('archdev001', 'unit02', 'unit03', 'unit04', 'unit05')
FORMAL = UNITS[1:]
TASKS = ('peaks', 'valleys')
GROUPS = ('main', 'matched')
SIGNS = (-1, 1)
LENGTHS = tuple(range(10, 31))
RANDOMS = tuple(f'random_direction_{s}' for s in range(1026, 1034))
CONDITIONS = ('unedited', 'local_projection', 'same_label_projection', *RANDOMS,
              'full_state_swap', 'actual_counterfactual')
TOLERANCE = 1e-10
ACTUAL_NORM_ATOL = 1e-5
ACTUAL_NORM_RTOL = 1e-5
CROSSING_CRITERION = .8
PRIMARY_AGG = 'Fixed 1/21 length mean after equal positions; no missing-length reweighting'
METRICS = ('checker_boundary_crossing', 'checker_desired_side', 'checker_signed_shift',
           'checker_positive_shift', 'edit_probe_signed_shift', 'edit_probe_donor_score_error',
           'signed_numeric_expectation_delta', 'desired_count_hit', 'desired_count_hit_gain',
           'signed_argmax_count_delta', 'numeric_argmax_coverage', 'recipient_count_accuracy',
           'condition_oracle_count_hit', 'edited_numeric_mass', 'numeric_mass_delta', 'edit_l2', 'planned_edit_l2',
           'prefix_count_prediction_delta', 'signed_prefix_count_prediction_delta')


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def read_csv(path):
    with Path(path).open(newline='') as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for r in rows for k in r)) or ['status']
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def number(value):
    if value is None or str(value).strip().lower() in ('', 'nan', 'none', 'null'):
        return None
    result = float(value)
    if not math.isfinite(result):
        return None
    return result


def mean(values):
    return math.fsum(values) / len(values)


def delta(a, b):
    return a - b if a is not None and b is not None else None


def get(row, name):
    return number(row.get(name))


def boolean(value):
    return str(value).strip().lower() not in ('false','0','no','none','nan','')


def score_row(raw):
    """Score one condition against the same local-only recipient endpoint."""
    r = dict(raw)
    available=boolean(r.get('condition_available','true'))
    for field in ('length', 'position', 'sign', 'recipient_local_label', 'desired_local_label',
                  'recipient_true_count', 'desired_true_count', 'recipient_argmax_count', 'edited_argmax_count'):
        r[field] = int(r[field])
    if (r['unit'] not in UNITS or r['task'] not in TASKS or r['group'] not in GROUPS or
            r['sign'] not in SIGNS or r['condition'] not in CONDITIONS or r['length'] not in LENGTHS or
            not 2 <= r['position'] <= r['length'] - 2):
        raise ValueError(('Unregistered local-intervention row', r))
    if r['desired_local_label'] - r['recipient_local_label'] != r['sign']:
        raise ValueError('Local target is not a one-bit sign-consistent flip')
    if r['desired_true_count'] != r['recipient_true_count'] + r['sign']:
        raise ValueError('Local-only desired count must equal recipient + sign')
    if not 0 <= r['desired_true_count'] <= 30:
        raise ValueError('Desired count outside numeric vocabulary')
    if r['group'] == 'matched' and r['position'] < 6:
        raise ValueError('Matched group has an undeclared early position')
    if r['condition'] == 'local_projection':
        a, b = get(r, 'recipient_resolved_count'), get(r, 'donor_resolved_count')
        if a is None or b is None or b - a != (r['sign'] if r['group'] == 'main' else 0):
            raise ValueError(('Resolved-prefix group formula failed', r['unit'], r['pair_id']))
    before, after = get(r, 'recipient_local_score'), get(r, 'edited_local_score')
    boundary = get(r, 'probe_boundary')
    if before is None or boundary is None or (available and after is None):
        raise ValueError('Independent baseline checker and available edited scores are required')
    if abs(boundary - .5) > TOLERANCE:
        raise ValueError('Checker boundary differs from fixed 0.5')
    baseline_local_correct = int(before >= boundary) == r['recipient_local_label']
    base_meta=dict(unit=r['unit'],task=r['task'],group=r['group'],sign=r['sign'],pair_id=r['pair_id'],
                   condition=r['condition'],length=r['length'],position=r['position'],
                   checker_baseline_correct=baseline_local_correct,
                   recipient_final_correct=r['recipient_argmax_count'] == r['recipient_true_count'],
                   condition_available=available,missing_reason=r.get('missing_reason',''))
    if not available:
        return dict(base_meta,**{m:None for m in METRICS})
    desired_side = int(after >= boundary) == r['desired_local_label']
    shift = r['sign'] * (after - before)
    rec_arg, edit_arg = r['recipient_argmax_count'], r['edited_argmax_count']
    if rec_arg < -1 or rec_arg > 30 or edit_arg < -1 or edit_arg > 30:
        raise ValueError('Full-vocabulary argmax must be numeric0..30 or -1 nonnumeric')
    desired_hit = float(edit_arg == r['desired_true_count'])
    values = dict(checker_boundary_crossing=float(baseline_local_correct and desired_side),
                  checker_desired_side=float(desired_side), checker_signed_shift=shift,
                  checker_positive_shift=float(shift > 0),
                  edit_probe_signed_shift=(r['sign'] * (get(r, 'edit_probe_edited_score') - get(r, 'edit_probe_recipient_score')))
                      if get(r, 'edit_probe_edited_score') is not None and get(r, 'edit_probe_recipient_score') is not None else None,
                  edit_probe_donor_score_error=abs(get(r,'edit_probe_edited_score') - get(r,'edit_probe_donor_score'))
                      if get(r,'edit_probe_edited_score') is not None and get(r,'edit_probe_donor_score') is not None else None,
                  signed_numeric_expectation_delta=r['sign'] * (get(r,'edited_numeric_expectation') - get(r,'recipient_numeric_expectation'))
                      if get(r,'edited_numeric_expectation') is not None and get(r,'recipient_numeric_expectation') is not None else None,
                  desired_count_hit=desired_hit,
                  desired_count_hit_gain=desired_hit - float(rec_arg == r['desired_true_count']),
                  signed_argmax_count_delta=r['sign'] * (edit_arg - rec_arg) if rec_arg >= 0 and edit_arg >= 0 else None,
                  numeric_argmax_coverage=float(rec_arg >= 0 and edit_arg >= 0),
                  recipient_count_accuracy=float(rec_arg == r['recipient_true_count']),
                  condition_oracle_count_hit=float(edit_arg == int(r['condition_oracle_count']))
                      if r['condition'] in ('unedited','same_label_projection','full_state_swap','actual_counterfactual') and get(r,'condition_oracle_count') is not None else None,
                  edited_numeric_mass=get(r,'edited_numeric_mass'),
                  numeric_mass_delta=delta(get(r,'edited_numeric_mass'),get(r,'recipient_numeric_mass')),
                  edit_l2=get(r,'edit_l2'),
                  planned_edit_l2=get(r,'planned_edit_l2'),
                  prefix_count_prediction_delta=delta(get(r,'edited_prefix_count_prediction'),get(r,'recipient_prefix_count_prediction')),
                  signed_prefix_count_prediction_delta=r['sign'] * (get(r,'edited_prefix_count_prediction') - get(r,'recipient_prefix_count_prediction'))
                      if get(r,'edited_prefix_count_prediction') is not None and get(r,'recipient_prefix_count_prediction') is not None else None)
    if values['edit_l2'] is not None and values['edit_l2'] < 0:
        raise ValueError('Negative edit norm')
    for name in ('recipient_numeric_mass','edited_numeric_mass'):
        value = get(r,name)
        if value is not None and not 0 <= value <= 1 + TOLERANCE:
            raise ValueError('Numeric mass outside [0,1]')
    for name in ('recipient_numeric_expectation','edited_numeric_expectation'):
        value = get(r,name)
        if value is not None and not -TOLERANCE <= value <= 30 + TOLERANCE:
            raise ValueError('Conditional numeric expectation outside [0,30]')
    return dict(base_meta,**values)


def paired_scores(raw):
    rows = [score_row(r) for r in raw]
    groups = defaultdict(dict)
    raw_index = {}
    for r, original in zip(rows, raw):
        key = tuple(r[k] for k in ('unit','task','group','sign','pair_id'))
        if r['condition'] in groups[key]:
            raise ValueError(('Duplicate intervention condition', key, r['condition']))
        groups[key][r['condition']] = r
        raw_index[key, r['condition']] = original
    output = []
    for key, items in groups.items():
        if set(items) != set(CONDITIONS):
            raise ValueError(('Condition inventory differs for pair', key, sorted(set(CONDITIONS)-set(items))))
        reference = raw_index[key,'local_projection']
        for condition in CONDITIONS:
            r = raw_index[key,condition]
            for name in ('length','position','recipient_local_label','desired_local_label','recipient_true_count','desired_true_count','recipient_local_score','probe_boundary'):
                if number(r.get(name)) != number(reference.get(name)):
                    raise ValueError(('Recipient or endpoint differs across paired conditions', key, condition, name))
            if items[condition]['condition_available'] and items['local_projection']['condition_available']:
                for name in ('recipient_argmax_count','recipient_numeric_expectation'):
                    if number(r.get(name)) != number(reference.get(name)):
                        raise ValueError(('Recipient prediction differs across available conditions',key,condition,name))
        treatment = items['local_projection']
        for random in RANDOMS:
            norm = items[random]['edit_l2']
            if not (items[random]['condition_available'] and treatment['condition_available']):
                continue
            plan=items[random]['planned_edit_l2'];target_plan=treatment['planned_edit_l2']
            if plan is None or target_plan is None or abs(plan-target_plan)>TOLERANCE:
                raise ValueError(('Planned FP64 random perturbation norm is not matched',key,random))
            if norm is None or treatment['edit_l2'] is None or abs(norm-treatment['edit_l2']) > ACTUAL_NORM_ATOL+ACTUAL_NORM_RTOL*abs(treatment['edit_l2']):
                raise ValueError(('Random perturbation norm is not matched', key, random))
        for condition in CONDITIONS:
            output.append(items[condition])
        random_row = dict(items[RANDOMS[0]], condition='random_mean')
        random_row['condition_available']=all(items[c]['condition_available'] for c in RANDOMS)
        random_row['missing_reason']='' if random_row['condition_available'] else 'At least one of eight registered random controls unavailable'
        for metric in METRICS:
            values = [items[c][metric] for c in RANDOMS]
            random_row[metric] = mean(values) if all(v is not None for v in values) else None
        output.append(random_row)
        for control in ('unedited','same_label_projection','random_mean'):
            c=random_row if control=='random_mean' else items[control]
            paired=dict(treatment,condition='contrast:local_projection-minus-'+control,
                        condition_available=treatment['condition_available'] and c['condition_available'],
                        missing_reason='' if treatment['condition_available'] and c['condition_available'] else 'Treatment/control not jointly available')
            for metric in METRICS:
                paired[metric]=delta(treatment[metric],c[metric])
            output.append(paired)
    return output


def supports(row):
    selected = ['all_pairs']
    if row['checker_baseline_correct']:
        selected.append('checker_baseline_correct')
    if row['recipient_final_correct']:
        selected.append('recipient_final_correct_diagnostic')
    if row['position'] >= 6:
        selected.append('common_positions_ge6')
    return selected


def aggregate(rows, out=None):
    # Streaming accumulators avoid expanding every pair into 18 long-form
    # metric rows or repeatedly scanning the raw input for each unit cell.
    def accumulator():
        return dict(pairs=0,available_pairs=0,sums=[0.]*len(METRICS),finite=[0]*len(METRICS))
    cells=defaultdict(accumulator)
    pooled=defaultdict(accumulator)
    for r in rows:
        for support in supports(r):
            base=tuple(r[k] for k in ('unit','task','group','sign','condition'))+(support,)
            for a in (cells[base+(r['length'],r['position'])],pooled[base]):
                a['pairs']+=1
                a['available_pairs']+=int(r['condition_available'])
                for j,m in enumerate(METRICS):
                    if r[m] is not None:
                        a['sums'][j]+=r[m];a['finite'][j]+=1
    length_acc={}
    writer=stream=None
    if out is not None:
        stream=(Path(out)/'by_position.csv').open('w',newline='')
    try:
        for key,a in sorted(cells.items()):
            base=dict(zip(('unit','task','group','sign','condition','support','length','position'),key))
            pos=dict(base,pairs=a['pairs'],available_pairs=a['available_pairs'])
            lkey=key[:-1]
            if lkey not in length_acc:
                length_acc[lkey]=dict(positions=set(),pairs=0,available_pairs=0,
                                      sums=[0.]*len(METRICS),finite_positions=[0]*len(METRICS),finite_pairs=[0]*len(METRICS))
            length=length_acc[lkey]
            length['positions'].add(base['position']);length['pairs']+=a['pairs'];length['available_pairs']+=a['available_pairs']
            for j,m in enumerate(METRICS):
                v=a['sums'][j]/a['finite'][j] if a['finite'][j] else None
                pos[m]=v;pos[m+'_finite_pairs']=a['finite'][j]
                length['finite_pairs'][j]+=a['finite'][j]
                if v is not None:
                    length['sums'][j]+=v;length['finite_positions'][j]+=1
            if stream is not None:
                if writer is None:
                    writer=csv.DictWriter(stream,fieldnames=list(pos));writer.writeheader()
                writer.writerow(pos)
    finally:
        if stream is not None:stream.close()
    length_rows=[];by_unit=defaultdict(dict)
    for key,a in sorted(length_acc.items()):
        base=dict(zip(('unit','task','group','sign','condition','support','length'),key))
        n=base['length'];lower=6 if base['group']=='matched' or base['support']=='common_positions_ge6' else 2
        expected=set(range(lower,n-1));present=a['positions']
        row=dict(base,registered_positions=len(expected),present_positions=len(present),
                 missing_positions=';'.join(map(str,sorted(expected-present))),pairs=a['pairs'],available_pairs=a['available_pairs'])
        for j,m in enumerate(METRICS):
            count=a['finite_positions'][j]
            row[m]=a['sums'][j]/count if count==len(expected) and present==expected else None
            row[m+'_available_positions_mean']=a['sums'][j]/count if count else None
            row[m+'_finite_positions']=count;row[m+'_finite_pairs']=a['finite_pairs'][j]
        length_rows.append(row);by_unit[key[:-1]][n]=row
    units=[]
    for key,group in sorted(by_unit.items()):
        base=dict(zip(('unit','task','group','sign','condition','support'),key))
        registered=sum((n-7 if base['group']=='matched' or base['support']=='common_positions_ge6' else n-3) for n in LENGTHS)
        for j,m in enumerate(METRICS):
            primary=[group[n][m] for n in LENGTHS if n in group and group[n][m] is not None]
            available=[group[n][m+'_available_positions_mean'] for n in LENGTHS if n in group and group[n][m+'_available_positions_mean'] is not None]
            meta=dict(base,metric=m,finite_lengths=len(primary),
                      missing_lengths=';'.join(str(n) for n in LENGTHS if n not in group),
                      undefined_lengths=';'.join(str(n) for n in LENGTHS if n in group and group[n][m] is None),
                      pairs=sum(r['pairs'] for r in group.values()),available_pairs=sum(r['available_pairs'] for r in group.values()),
                      finite_pairs=sum(r[m+'_finite_pairs'] for r in group.values()),registered_positions=registered,
                      finite_positions=sum(r[m+'_finite_positions'] for r in group.values()))
            units.append(dict(meta,value=mean(primary) if len(primary)==21 else None,
                              aggregation=PRIMARY_AGG))
            units.append(dict(meta,value=mean(available) if available else None,
                              aggregation='Available-cell sensitivity: equal available positions then available lengths'))
            a=pooled[key]
            units.append(dict(meta,value=a['sums'][j]/a['finite'][j] if a['finite'][j] else None,
                              aggregation='Pooled pairs sensitivity'))
    return len(cells),length_rows,units


def paired_unit_effects(units):
    result=[]
    for r in units:
        if not r['condition'].startswith('contrast:'):
            continue
        result.append(dict(**{k:v for k,v in r.items() if k!='condition'},
                           contrast=r['condition'].split(':',1)[1],
                           paired_support='Within-pair differences only on jointly available treatment/control; random average requires all eight'))
    return result


def cohort(rows,keys):
    groups=defaultdict(dict)
    for r in rows:
        groups[tuple(r[k] for k in keys)][r['unit']]=r
    output=[]
    for key,group in sorted(groups.items()):
        chosen=[(u,group.get(u,{}).get('value')) for u in FORMAL]
        values=[v for _,v in chosen if v is not None]
        full=len(values)==4
        output.append(dict(zip(keys,key),registered_units=4,finite_units=len(values),
                           mean=mean(values) if full else None,sample_sd=statistics.stdev(values) if full else None,
                           minimum=min(values) if full else None,maximum=max(values) if full else None,
                           positive_units=sum(v>0 for v in values),negative_units=sum(v<0 for v in values),
                           zero_units=sum(v==0 for v in values),raw_unit_values=';'.join(u+':'+str(v) for u,v in chosen),
                           undefined_reason='' if full else 'Missing/undefined unit, not zero or reweighted'))
    return output


def derive(raw,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    scored=paired_scores(raw)
    position_count,lengths,units=aggregate(scored,out)
    effects=paired_unit_effects(units)
    unit_stats=cohort(units,('task','group','sign','condition','support','metric','aggregation'))
    effect_stats=cohort(effects,('task','group','sign','support','metric','aggregation','contrast'))
    for name,rows in [('scored_pairs',scored),('by_length',lengths),
                      ('per_unit',units),('paired_unit_effects',effects),('four_unit_summary',unit_stats),
                      ('four_unit_paired_summary',effect_stats)]:
        write_csv(out/(name+'.csv'),rows)
    render_report(units,effects,out)
    return dict(status='completed_saved_local_intervention_summary',raw_rows=len(raw),scored_rows=len(scored),position_rows=position_count,
                formal_units=list(FORMAL),historical_reference=UNITS[0],no_fits=True,no_forwards=True,
                randoms_averaged_within_pair_first=True,no_control_maximum=True,
                primary_aggregation='Fixed 1/21 lengths, equal positions, paired means within position',
                pooled_pair_sensitivity=True,no_count_outcome_selection=True)


def fmt(v):
    return 'undefined' if v is None else f'{v:.4f}'


def render_report(units,effects,out):
    index={tuple(r[k] for k in ('unit','task','group','sign','condition','support','metric','aggregation')):r for r in units}
    effect_index={tuple(r[k] for k in ('unit','task','group','sign','contrast','support','metric','aggregation')):r for r in effects}
    lines=['# Local recurrent-state intervention: saved-output summary','',
           'This analysis separates an independent local manipulation check from the final counting response. The local-only target is the recipient count plus the desired local sign in both main and count-matched groups. The matched donor complete count is not substituted for this endpoint. Historical data select geometry only; the four new units provide the formal paired repetitions.','',
           'Each reported primary cell averages pairs within a registered center, centers within length, and all n=10–30 with fixed weight 1/21. Missing required cells leave it undefined. Available-cell and pooled-pair sensitivities are separately labeled. The eight random controls are averaged within pair first; treatment-minus-control effects use common available pairs. Both signs and both tasks remain separate.','',
           '## Per-unit main questions','',
           'The checker was fitted on components independent of the edit-direction probe. True crossing requires the recipient to start on its own-label side of 0.5 and end on the opposite desired side. The descriptive manipulation criterion is at least 0.80 crossing and positive signed checker shift; count-output success never selects the edit. Conditional numeric expectation is read together with numeric probability mass and full-vocabulary argmax endpoints.','',
           '|Unit|Task|Group|Sign|Checker crossing|Checker signed shift|Manipulation criterion|Signed expectation change|Treatment−random expectation|Treatment−same-label expectation|Desired-count hit|Signed prefix-count decode change|Actual-CF oracle hit|Full-swap oracle hit|',
           '|---|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|']
    for u in UNITS:
        for t in TASKS:
            for g in GROUPS:
                for s in SIGNS:
                    get=lambda condition,metric:index.get((u,t,g,s,condition,'all_pairs',metric,PRIMARY_AGG),{}).get('value')
                    eff=lambda control,metric:effect_index.get((u,t,g,s,'local_projection-minus-'+control,'all_pairs',metric,PRIMARY_AGG),{}).get('value')
                    crossing=get('local_projection','checker_boundary_crossing');shift=get('local_projection','checker_signed_shift')
                    qualifies=crossing>=CROSSING_CRITERION and shift>0 if crossing is not None and shift is not None else None
                    values=[crossing,shift,qualifies,get('local_projection','signed_numeric_expectation_delta'),
                            eff('random_mean','signed_numeric_expectation_delta'),eff('same_label_projection','signed_numeric_expectation_delta'),
                            get('local_projection','desired_count_hit'),get('local_projection','signed_prefix_count_prediction_delta'),
                            get('actual_counterfactual','condition_oracle_count_hit'),get('full_state_swap','condition_oracle_count_hit')]
                    lines.append('|'+'|'.join([u,t,g,str(s),*(str(v) if isinstance(v,bool) else fmt(v) for v in values)])+'|')
    lines += ['', '## Scope and limitations','',
              'Main and matched families have different donor constructions and center availability, so their difference is not a pure factorial causal contrast. The common-center sensitivity i≥6 is retained. Same-label donors have natural perturbation magnitudes; random controls match treatment L2 pair by pair. Unavailable controls are retained with reasons and are never scored as −1 counts or zero success.','',
              'Count matching and an independent local checker do not uniquely identify a local semantic mechanism. A local probe direction can overlap a general count direction, and a mixed recurrent state can differ from naturally occurring states. The independently fitted immediate prefix-count decoder is diagnostic of this possibility; no new null-space or orthogonalization condition is introduced. A positive effect supports a functional effect of this specified state direction, not proof of natural lemma use, uniquely mathematical encoding, or ordinary compositional inference.','',
              'All signs, conditions, missingness, recipient-final-correct diagnostic subsets, numeric-mass checks, norm checks and positive controls remain in the CSVs. Statistics use four independent units; positions, pairs, random directions and the two tasks are not additional replication units. This stage stops after its stated report regardless of sign.','']
    (Path(out)/'report.md').write_text('\n'.join(lines))


def synthetic_checks():
    raw=[]
    # All registered lengths and positions; two tasks, both groups/signs and
    # all units. Synthetic outcomes only, with an exactly matched random norm.
    for unit in UNITS:
        for task in TASKS:
            for group in GROUPS:
                for sign in SIGNS:
                    for n in LENGTHS:
                        for i in range(6 if group=='matched' else 2,n-1):
                            label=int(sign<0); desired=1-label
                            before=.8 if label else .2
                            for condition in CONDITIONS:
                                changed=condition in ('local_projection','actual_counterfactual')
                                after=1-before if changed else before
                                base=dict(unit=unit,task=task,group=group,sign=sign,pair_id=f'{n}:{i}',condition=condition,
                                          length=n,position=i,layer=3,timing='after',dose=1,
                                          recipient_local_label=label,desired_local_label=desired,probe_boundary=.5,
                                          recipient_local_score=before,edited_local_score=after,
                                          edit_probe_recipient_score=before,edit_probe_edited_score=after,
                                          edit_probe_donor_score=1-before,
                                          recipient_true_count=4,desired_true_count=4+sign,
                                          donor_full_true_count=4+(sign if group=='main' else 0),
                                          recipient_argmax_count=4,edited_argmax_count=4+sign if changed else 4,
                                          recipient_numeric_expectation=4.,edited_numeric_expectation=4.+sign if changed else 4.,
                                          recipient_numeric_mass=1.,edited_numeric_mass=1.,edit_l2=1.,
                                          planned_edit_l2=1.,
                                          recipient_prefix_count_prediction=3.,edited_prefix_count_prediction=3.+sign if changed else 3.,
                                          recipient_resolved_count=3,donor_resolved_count=3+(sign if group=='main' else 0),
                                          condition_oracle_count=4+sign if condition=='actual_counterfactual' else 4)
                                raw.append(base)
    scored=paired_scores(raw)
    assert len(scored)==len(raw)//len(CONDITIONS)*(len(CONDITIONS)+4)
    local=next(r for r in scored if r['condition']=='local_projection')
    assert local['checker_boundary_crossing']==1 and local['desired_count_hit']==1
    random=next(r for r in scored if r['condition']=='random_mean')
    assert random['signed_numeric_expectation_delta']==0 and random['desired_count_hit']==0
    with tempfile.TemporaryDirectory(prefix='local_causal_synthetic_') as tmp:
        result=derive(raw,tmp)
        effects=read_csv(Path(tmp)/'four_unit_paired_summary.csv')
        chosen=[r for r in effects if r['contrast']=='local_projection-minus-random_mean' and r['metric']=='signed_numeric_expectation_delta']
        assert chosen and all(abs(float(r['mean'])-1)<TOLERANCE and float(r['sample_sd'])==0 for r in chosen)
        assert all(r['registered_units']=='4' for r in chosen)
    # A pre-existing desired-side prediction is not a boundary crossing.
    weak=dict(raw[0],recipient_local_score=.2 if raw[0]['desired_local_label']==0 else .8,
              edited_local_score=.2 if raw[0]['desired_local_label']==0 else .8)
    assert score_row(weak)['checker_boundary_crossing']==0 and score_row(weak)['checker_desired_side']==1
    # Unavailable is neither a wrong count nor a failed geometric edit.
    base=[dict(r) for r in raw if r['unit']=='unit02' and r['task']=='peaks' and r['group']=='matched'
          and r['sign']==1 and r['length']==10 and r['position']==6]
    assert len(base)==len(CONDITIONS)
    two=[]
    for index in range(2):
        for original in base:
            r=dict(original,pair_id='missing-check-'+str(index))
            if r['condition']=='local_projection':r['edited_numeric_expectation']=5.+2*index
            if index==1 and r['condition']=='same_label_projection':
                r.update(condition_available=False,missing_reason='synthetic unavailable real donor',
                         edited_local_score=None,edited_argmax_count=-1,edited_numeric_expectation=None)
            two.append(r)
    test=paired_scores(two)
    paired=[r for r in test if r['condition']=='contrast:local_projection-minus-same_label_projection']
    assert paired[0]['signed_numeric_expectation_delta']==1 and paired[1]['signed_numeric_expectation_delta'] is None
    unavailable=next(r for r in test if r['pair_id']=='missing-check-1' and r['condition']=='same_label_projection')
    assert unavailable['desired_count_hit'] is None and not unavailable['condition_available']
    _,_,small_units=aggregate(test)
    actual=next(r for r in small_units if r['condition']=='contrast:local_projection-minus-same_label_projection'
                and r['metric']=='signed_numeric_expectation_delta' and r['support']=='all_pairs'
                and r['aggregation'].startswith('Available-cell'))
    assert actual['value']==1 and actual['finite_pairs']==1 and actual['available_pairs']==1
    primary=next(r for r in small_units if r['condition']==actual['condition'] and r['metric']==actual['metric']
                 and r['support']=='all_pairs' and r['aggregation']==PRIMARY_AGG)
    assert primary['value'] is None and primary['finite_lengths']==0
    rounded=[dict(r) for r in base]
    next(r for r in rounded if r['condition']==RANDOMS[0])['edit_l2']=1.+1.5e-5
    paired_scores(rounded)  # Prespecified FP32 state-norm tolerance.
    broken=[dict(r) for r in base]
    next(r for r in broken if r['condition']==RANDOMS[0])['planned_edit_l2']=1.+1e-7
    try:paired_scores(broken)
    except ValueError as error:assert 'Planned FP64' in str(error)
    else:raise AssertionError('Planned-norm mismatch was not rejected')
    return dict(status='passed',synthetic_only=True,raw_rows=len(raw),source_sha256=sha(__file__),
                checks=['recipient+sign endpoint for matched donors','true checker boundary crossing',
                        'all eight random directions averaged within pair','paired unit effects',
                        'four formal units, historical excluded from SD','both signs separate',
                        'fixed lengths and equal positions','same-pair recipient consistency',
                        'random L2 matching guard','unavailable conditions not zero-filled',
                        'paired effects use common support before aggregation',
                        'missing registered cells make primary undefined with available sensitivity',
                        'planned FP64 versus actual FP32 norm tolerances'],example_result=result)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--synthetic',action='store_true')
    parser.add_argument('--freeze',action='store_true')
    parser.add_argument('--out',type=Path,default=HERE/'summary')
    parser.add_argument('--protocol',type=Path,default=HERE/'protocol.json')
    args=parser.parse_args()
    if args.synthetic:
        result=synthetic_checks();write_json(HERE/'summary_synthetic_verification.json',result)
        print(json.dumps(result,indent=2));return
    if not args.protocol.exists():
        raise FileNotFoundError('Protocol must be frozen before formal outcome access')
    if args.freeze:
        verification=load_json(HERE/'summary_synthetic_verification.json')
        assert verification['status']=='passed' and verification['source_sha256']==sha(__file__)
        result=dict(source_sha256=sha(__file__),protocol_sha256=sha(args.protocol),
                    created_utc=datetime.now(timezone.utc).isoformat(),synthetic_only_validation=True,
                    formal_outcomes_not_accessed_by_helper=True,verification=verification)
        path=HERE/'summary_freeze.json'
        if path.exists():
            old=load_json(path)
            assert all(old[k]==result[k] for k in ('source_sha256','protocol_sha256'))
            result=old
        else:write_json(path,result)
        print(json.dumps(result,indent=2));return
    frozen=load_json(HERE/'summary_freeze.json')
    assert sha(__file__)==frozen['source_sha256']
    assert sha(args.protocol)==frozen['protocol_sha256']
    raw,sources=[],[]
    for unit in UNITS:
        path=HERE/'results'/unit/'pair_results.csv'
        raw.extend(read_csv(path));sources.append(dict(path=str(path),sha256=sha(path)))
    result=derive(raw,args.out)
    result.update(completed_utc=datetime.now(timezone.utc).isoformat(),source_sha256=sha(__file__),
                  protocol_sha256=sha(args.protocol),source_tables=sources)
    write_json(args.out/'completion.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
