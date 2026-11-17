"""Derived tables from the completed frozen timing summary only.

No raw neural outputs, fits, forwards, metric changes or outcome selection.
All fields below are selected before new timing-summary outcomes are read.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime,timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import tempfile

HERE=Path(__file__).resolve().parent
UNITS=('archdev001','unit02','unit03','unit04','unit05')
FORMAL=UNITS[1:]
TASKS=('peaks','valleys')
GROUPS=('main','matched')
SIGNS=(-1,1)
PRIMARY='Fixed 1/21 lengths after equal registered centers'

MAIN_FIELDS=[
 ('late_relative_step',('late','local_projection','relative_step_hit')),
 ('early_relative_step',('early','local_projection','relative_step_hit')),
 ('late_desired_answer',('late','local_projection','desired_answer_hit')),
 ('early_desired_answer',('early','local_projection','desired_answer_hit')),
 ('late_other_numeric_change',('late','local_projection','other_numeric_count_change')),
 ('early_other_numeric_change',('early','local_projection','other_numeric_count_change')),
 ('late_any_numeric_change',('late','local_projection','any_numeric_count_change')),
 ('early_any_numeric_change',('early','local_projection','any_numeric_count_change')),
 ('late_any_token_change',('late','local_projection','any_argmax_token_change')),
 ('early_any_token_change',('early','local_projection','any_argmax_token_change')),
 ('late_numeric_coverage',('late','local_projection','numeric_argmax_coverage')),
 ('early_numeric_coverage',('early','local_projection','numeric_argmax_coverage')),
 ('late_signed_expectation_change',('late','local_projection','signed_numeric_expectation_delta')),
 ('early_signed_expectation_change',('early','local_projection','signed_numeric_expectation_delta')),
 ('late_checker_crossing',('late','local_projection','checker_boundary_crossing')),
 ('early_checker_crossing',('early','local_projection','checker_boundary_crossing')),
 ('late_checker_signed_shift',('late','local_projection','checker_signed_shift')),
 ('early_checker_signed_shift',('early','local_projection','checker_signed_shift')),
 ('late_injected_l1_norm',('late','local_projection','edit_l2')),
 ('early_injected_l1_norm',('early','local_projection','edit_l2')),
 ('late_full_carry_norm',('late','local_projection','carried_state_l2norm')),
 ('early_full_carry_norm',('early','local_projection','carried_state_l2norm')),
 ('early_minus_late_relative_step',('contrast','early-minus-late__local_projection','relative_step_hit')),
 ('early_minus_late_desired_answer',('contrast','early-minus-late__local_projection','desired_answer_hit')),
 ('early_minus_late_signed_expectation',('contrast','early-minus-late__local_projection','signed_numeric_expectation_delta')),
 ('did_relative_step_vs_random',('contrast','timing-did__local-minus-random_mean','relative_step_hit')),
 ('did_relative_step_vs_same_label',('contrast','timing-did__local-minus-same_label_projection','relative_step_hit')),
 ('did_desired_answer_vs_random',('contrast','timing-did__local-minus-random_mean','desired_answer_hit')),
 ('did_desired_answer_vs_same_label',('contrast','timing-did__local-minus-same_label_projection','desired_answer_hit')),
 ('did_expectation_vs_random',('contrast','timing-did__local-minus-random_mean','signed_numeric_expectation_delta')),
 ('did_expectation_vs_same_label',('contrast','timing-did__local-minus-same_label_projection','signed_numeric_expectation_delta')),
]
AUX_FIELDS=[]
for layer in range(1,5):
 for metric in ('relative_step_hit','desired_answer_hit','other_numeric_count_change','signed_numeric_expectation_delta'):
  AUX_FIELDS.append((f'late_whole_layer{layer}_{metric}',('late',f'late_layer_swap_{layer}',metric)))
for metric in ('relative_step_hit','desired_answer_hit','other_numeric_count_change','signed_numeric_expectation_delta'):
 AUX_FIELDS.append((f'early_whole_layer1_{metric}',('early','early_full_layer1',metric)))
 AUX_FIELDS.append((f'early_minus_late_whole_layer1_{metric}',('contrast','early-minus-late__full-layer1',metric)))
POSITIVE_FIELDS=[
 ('full4_donor_oracle_accuracy',('diagnostic','full_state_swap','reference_oracle_hit')),
 ('full4_local_desired_answer',('diagnostic','full_state_swap','desired_answer_hit')),
 ('full4_relative_step',('diagnostic','full_state_swap','relative_step_hit')),
 ('actual_cf_oracle_accuracy',('diagnostic','actual_counterfactual','reference_oracle_hit')),
 ('actual_cf_desired_answer',('diagnostic','actual_counterfactual','desired_answer_hit')),
 ('actual_cf_relative_step',('diagnostic','actual_counterfactual','relative_step_hit')),
 ('recipient_original_accuracy',('late','unedited','recipient_count_accuracy')),
 ('early_noop_relative_step',('early','unedited','relative_step_hit')),
 ('early_noop_any_token_change',('early','unedited','any_argmax_token_change')),
]


def sha(path):
 with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read_json(path):return json.loads(Path(path).read_text())
def write_json(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def read_csv(path):
 with Path(path).open(newline='') as f:return list(csv.DictReader(f))
def write_csv(path,rows):
 fields=list(dict.fromkeys(k for r in rows for k in r)) or ['status']
 with Path(path).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def value(x):
 if x is None or str(x).lower() in ('','none','nan','null'):return None
 v=float(x);return v if math.isfinite(v) else None
def mean(xs):return math.fsum(xs)/len(xs)


def index_units(rows):
 keys=('unit','task','group','sign','timing','condition','support','metric','aggregation')
 ix={}
 for r in rows:
  key=tuple(int(r[k]) if k=='sign' else r[k] for k in keys)
  if key in ix:raise ValueError(('Duplicate saved scalar',key))
  ix[key]=r
 return ix


def table(ix,spec):
 rows=[]
 for u in UNITS:
  for t in TASKS:
   for g in GROUPS:
    for sign in SIGNS:
     row=dict(unit=u,historical_reference=u==UNITS[0],task=t,group=g,sign=sign)
     for label,(timing,condition,metric) in spec:
      key=(u,t,g,sign,timing,condition,'all_pairs',metric,PRIMARY)
      if key not in ix:raise ValueError(('Missing registered headline field',label,key))
      row[label]=value(ix[key]['value'])
     rows.append(row)
 return rows


def summarize(rows,spec):
 ix={(r['unit'],r['task'],r['group'],r['sign']):r for r in rows}
 output=[]
 for t in TASKS:
  for g in GROUPS:
   for sign in SIGNS:
    row=dict(task=t,group=g,sign=sign,registered_units=4,
             primary_aggregation='Equal registered centers then all21 lengths with1/21; paired effects formed within pair first')
    for label,_ in spec:
     chosen=[(u,ix[u,t,g,sign][label]) for u in FORMAL]
     vs=[v for _,v in chosen if v is not None];full=len(vs)==4
     for k,v in dict(mean=mean(vs) if full else None,sample_sd=statistics.stdev(vs) if full else None,
                     minimum=min(vs) if full else None,maximum=max(vs) if full else None,
                     finite_units=len(vs),positive_units=sum(v>0 for v in vs),negative_units=sum(v<0 for v in vs),
                     raw_unit_values=';'.join(u+':'+str(v) for u,v in chosen)).items():
      row[label+'_'+k]=v
    output.append(row)
 return output


def crosscheck_existing(stats,spec,source_stats):
 ix={tuple(int(r[k]) if k=='sign' else r[k] for k in ('task','group','sign','timing','condition','support','metric','aggregation')):r for r in source_stats}
 count=0;maximum=0.
 for r in stats:
  for label,(timing,condition,metric) in spec:
   old=ix[r['task'],r['group'],r['sign'],timing,condition,'all_pairs',metric,PRIMARY]
   for field in ('mean','sample_sd','minimum','maximum'):
    a,b=r[label+'_'+field],value(old[field])
    if (a is None)!=(b is None):raise ValueError(('Undefined summary mismatch',label,field))
    if a is not None:
     error=abs(a-b);maximum=max(maximum,error)
     if error>1e-12:raise ValueError(('Summary arithmetic mismatch',label,field,error))
    count+=1
 return count,maximum


def fmt(v,percent=False):
 if v is None:return 'undefined'
 return f'{100*v:.2f}%' if percent else f'{v:.6g}'


def render(main,aux,pos,out):
 lines=['# Timing experiment: main edit and whole-layer diagnostics','',
  'This is an exploratory follow-up on the same five existing model/data units; their prior late-edit outcomes had been viewed. Historical results are engineering reference only. Four-unit summaries below retain both tasks, donor constructions and signs separately. Timings, positions and control directions are dependent comparisons within a unit.','',
  '## One-dimensional local edit: relative modal changes and oracle answers','',
  'Relative success means the edited numeric argmax differs from the recipient numeric argmax by exactly the requested sign. Oracle-answer success means the edited numeric answer equals the true recipient count plus that sign. These differ when the baseline is wrong. Both are reported; nonnumeric and other-count changes remain explicit in the CSVs.','',
  '|Task|Group|Sign|Late relative step|Early relative step|Late oracle answer|Early oracle answer|Early−late relative|DiD vs random|DiD vs same label|',
  '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
 for r in main:
  fields=['late_relative_step','early_relative_step','late_desired_answer','early_desired_answer',
          'early_minus_late_relative_step','did_relative_step_vs_random','did_relative_step_vs_same_label']
  lines.append('|'+'|'.join([r['task'],r['group'],str(r['sign']),*(fmt(r[x+'_mean'],True) for x in fields)])+'|')
 lines+=['','The injected first-layer state, local checker and one-dimensional dose are identical across early and late. Early replays upper current-token states; it can therefore change total carried-state norm even with identical injected L1 norm. Timing and DiD contrasts are formed within pair before equal-center and fixed1/21-length aggregation; random controls are averaged within pair, never maximized.','',
  '## Diagnostic 1: late replacement of each complete layer state','',
  'Each payload below is a complete 256-coordinate recurrent state. These replacements are distinct from the one-dimensional local edit and can transport history as well as local information. Columns show relative-step / oracle-answer success. They cannot identify a uniquely necessary counter layer.','',
  '|Task|Group|Sign|Whole L1 late|Whole L2 late|Whole L3 late|Whole L4 late|',
  '|---|---|---:|---:|---:|---:|---:|']
 for r in aux:
  vals=[fmt(r[f'late_whole_layer{k}_relative_step_hit_mean'],True)+' / '+fmt(r[f'late_whole_layer{k}_desired_answer_hit_mean'],True) for k in range(1,5)]
  lines.append('|'+'|'.join([r['task'],r['group'],str(r['sign']),*vals])+'|')
 lines+=['','## Diagnostic 2: complete L1 replacement before upper replay','',
  'Early complete-L1 replacement and late complete-L1 replacement inject byte-identical L1 states; their upper-state handling differs. The paired contrast below uses the same interface and pairs. A whole-state effect does not show that the chosen one-dimensional feature reproduces it.','',
  '|Task|Group|Sign|Whole L1 late relative|Whole L1 early relative|Paired early−late relative|Whole L1 early oracle answer|',
  '|---|---|---:|---:|---:|---:|---:|']
 for r in aux:
  fields=['late_whole_layer1_relative_step_hit','early_whole_layer1_relative_step_hit','early_minus_late_whole_layer1_relative_step_hit','early_whole_layer1_desired_answer_hit']
  lines.append('|'+'|'.join([r['task'],r['group'],str(r['sign']),*(fmt(r[x+'_mean'],True) for x in fields)])+'|')
 lines+=['','## Physical positive controls and local desired answer','',
  'The full-four-layer transplant has the donor whole-input oracle; the real local-only counterfactual has recipient+sign as its oracle. In the matched group those donor and local-only endpoints differ. Mixed single-layer states have no natural full-input oracle and are evaluated against the stated local target separately.','',
  '|Task|Group|Sign|Full-four-state donor oracle|Full-four-state local goal|Real counterfactual oracle|',
  '|---|---|---:|---:|---:|---:|']
 for r in pos:
  fields=['full4_donor_oracle_accuracy','full4_local_desired_answer','actual_cf_oracle_accuracy']
  lines.append('|'+'|'.join([r['task'],r['group'],str(r['sign']),*(fmt(r[x+'_mean'],True) for x in fields)])+'|')
 lines+=['','All means, sample SD, minima, maxima and four raw unit values are retained in `main_eight_cells.csv`, `auxiliary_eight_cells.csv`, and `positive_controls_eight_cells.csv`. The corresponding `*_by_unit.csv` files show individual formal units; historical files are separate. The original completed frozen summary retains other-change rates, numeric mass/coverage, signed count expectations, checker values, common-center/baseline-correct diagnostics, available-cell sensitivities and pooled sensitivities.','',
  'The input construction couples incoming token rank to the local label, and these are mixed-state interventions. Results can support the tested current interlayer route, not precise natural accumulation timing, an abstract predicate-specific axis, or a naturally reused one-dimensional lemma. No new units, models, readouts, doses, conditions, fits or forwards are introduced by this reader.','']
 (Path(out)/'reader_report.md').write_text('\n'.join(lines))


def build(unit_rows,source_stats,out,validate=True):
 out=Path(out);out.mkdir(parents=True,exist_ok=True)
 ix=index_units(unit_rows);checks=0;maximum=0.
 for name,spec in [('main',MAIN_FIELDS),('auxiliary',AUX_FIELDS),('positive_controls',POSITIVE_FIELDS)]:
  raw=table(ix,spec);stats=summarize(raw,spec)
  write_csv(out/(name+'_by_unit.csv'),[r for r in raw if not r['historical_reference']])
  write_csv(out/(name+'_historical_reference.csv'),[r for r in raw if r['historical_reference']])
  write_csv(out/(name+'_eight_cells.csv'),stats)
  if validate:
   n,e=crosscheck_existing(stats,spec,source_stats);checks+=n;maximum=max(maximum,e)
  if name=='main':main=stats
  elif name=='auxiliary':aux=stats
  else:positive=stats
 render(main,aux,positive,out)
 return dict(status='completed_derived_timing_headlines',formal_units=list(FORMAL),historical_reference=UNITS[0],
             main_cells=8,auxiliary_cells=8,positive_control_cells=8,scalar_checks=checks,maximum_summary_error=maximum,
             no_new_metrics=True,no_raw_output_access=True,no_fits=True,no_forwards=True)


def synthetic():
 specs=MAIN_FIELDS+AUX_FIELDS+POSITIVE_FIELDS;keys={spec for _,spec in specs};rows=[]
 for u_i,u in enumerate(UNITS):
  for t in TASKS:
   for g in GROUPS:
    for sign in SIGNS:
     for timing,condition,metric in keys:
      v=.1*u_i
      if metric=='desired_answer_hit':v+=.02
      if metric=='relative_step_hit':v+=.04
      rows.append(dict(unit=u,task=t,group=g,sign=sign,timing=timing,condition=condition,support='all_pairs',metric=metric,aggregation=PRIMARY,value=v))
 with tempfile.TemporaryDirectory(prefix='timing_headlines_synthetic_') as d:
  result=build(rows,[],d,False);main=read_csv(Path(d)/'main_eight_cells.csv')
  assert len(main)==8
  r=main[0];assert abs(float(r['early_relative_step_mean'])-.29)<1e-12
  assert abs(float(r['early_desired_answer_mean'])-.27)<1e-12
  assert abs(float(r['early_relative_step_sample_sd'])-statistics.stdev([.14,.24,.34,.44]))<1e-12
  assert len(read_csv(Path(d)/'auxiliary_by_unit.csv'))==32
  assert len(read_csv(Path(d)/'main_historical_reference.csv'))==8
 return dict(status='passed',synthetic_only=True,source_sha256=sha(__file__),checks=['distinct relative/oracle labels','four new units only','raw units/sample SD/range','all four complete-layer payloads','separate early complete-L1 paired contrast','historical reference separate'],example_result=result)


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--synthetic',action='store_true');p.add_argument('--freeze',action='store_true');p.add_argument('--out',type=Path,default=HERE/'headlines');args=p.parse_args()
 if args.synthetic:
  r=synthetic();write_json(HERE/'headline_synthetic_verification.json',r);print(json.dumps(r,indent=2));return
 if args.freeze:
  v=read_json(HERE/'headline_synthetic_verification.json');assert v['source_sha256']==sha(__file__) and v['status']=='passed'
  r=dict(source_sha256=sha(__file__),created_utc=datetime.now(timezone.utc).isoformat(),synthetic_verification=v,derived_only=True,before_actual_timing_summary_access=True)
  f=HERE/'headline_freeze.json'
  if f.exists():assert read_json(f)['source_sha256']==r['source_sha256'];r=read_json(f)
  else:write_json(f,r)
  print(json.dumps(r,indent=2));return
 frozen=read_json(HERE/'headline_freeze.json');assert frozen['source_sha256']==sha(__file__)
 root=HERE/'summary';receipt=read_json(root/'completion.json');assert receipt['status']=='completed_saved_timing_summary'
 paths=[root/'per_unit.csv',root/'four_unit_summary.csv',root/'completion.json']
 hashes={str(f):sha(f) for f in paths}
 r=build(read_csv(paths[0]),read_csv(paths[1]),args.out)
 r.update(completed_utc=datetime.now(timezone.utc).isoformat(),source_sha256=sha(__file__),source_tables=hashes,
          protocol_sha256=receipt['protocol_sha256'])
 write_json(args.out/'completion.json',r);print(json.dumps(r,indent=2))


if __name__=='__main__':main()
