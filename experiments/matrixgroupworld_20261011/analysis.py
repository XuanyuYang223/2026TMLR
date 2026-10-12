"""Frozen MatrixGroupWorld endpoint geometry and separate fixed-T task comparisons.

Only saved feature caches are consumed. No optimizer, model construction, input
generation, layer search, or statistical significance test occurs here.
"""
from __future__ import annotations
from collections import Counter, defaultdict
import importlib.util
import math
from pathlib import Path
import time
import numpy as np
from .common import DIAG, ROOT, CONFIG, csv_write, json_write, read, sha, verify, source_version
from .core import TASKS, ACTIONS, RELATIONS, TOKEN_TO_ID

METRIC_PATH=Path('/home/yangx/ICML/diagnostics/permworld_scope_inventory_20261009/metrics.py')
METRIC_SHA='c2a384a8212e661d1917c084a74699297bbef354f06dac42ac75ea90ce3d0b5c'
NAMES=('linear_cka','knn_overlap','procrustes')
CONTROLS=('raw_within_n','answer_strata')
BRANCHES=('prefix_final','output_prob','gold_onehot')
CATEGORIES=('designated','other_known_exact','not_designated')
PHASES=('initial','trained')
ROLES=('correct','wrong','identity')
PROOFS={
 'R1':'F2(A^T)=wt((A^T)^T e1)=wt(A e1)=F1(A).',
 'R2':'F4(A^T)=wt((A^T)^T s)=wt(A s)=F3(A).',
 'R3':'Q s=e1, hence F3(AQ)=wt(A Q s)=wt(A e1)=F1(A).',
 'R4':'Q s=e1, hence F4(Q^T A)=wt((Q^T A)^T s)=wt(A^T Q s)=F2(A).',
}

class AnalysisBudgetExhausted(RuntimeError):pass
def guard(deadline):
 if deadline is not None and time.monotonic()>=float(deadline):raise AnalysisBudgetExhausted('Fixed CPU-stage monotonic deadline reached')
def load_metrics():
 assert sha(METRIC_PATH)==METRIC_SHA,'Inherited scientific metric source changed'
 spec=importlib.util.spec_from_file_location('matrixgroupworld_frozen_metrics',METRIC_PATH)
 m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def array_sha(value):
 import hashlib
 a=np.ascontiguousarray(value);h=hashlib.sha256(str(a.dtype).encode()+repr(a.shape).encode());h.update(a.tobytes());return h.hexdigest()
def units(config):
 return [u['unit'] if isinstance(u,dict) else u for u in config['units']]

def prepare_candidates(config):
 """Pre-geometry finite known-exact catalogue; not a no-relation theorem."""
 math_audit=read(DIAG/'math_audit.json');assert math_audit['status']=='passed'
 rows=[]
 for unit in units(config):
  for r in RELATIONS:
   for task in TASKS:
    if task==r['left']:continue
    exact=task==r['right']
    rows.append(dict(unit=unit,relation=r['relation'],main=r['main'],left_task=r['left'],right_task=task,
     correct_transform=r['correct'],designated_right_task=r['right'],category='designated' if exact else 'not_designated',
     formula=f"{task}({r['correct']}(A)) = {r['left']}(A)" if exact else '',
     comparison_formula=f"{task}({r['correct']}(A)) = {r['left']}(A)",
     proof=PROOFS[r['relation']] if exact else '',
     pair_id=f"{unit}__{r['relation']}__{task}"))
 assert len(rows)==24 and len({r['pair_id'] for r in rows})==24
 csv_write(DIAG/'candidate_manifest.csv',rows)
 # Existing mathematical audit n2/3 scope, independent of final data/geometry:
 # finite examples only refute the specific equality, never prove independence.
 import itertools
 from .core import rank,labels,transform
 examples={};consistent=[]
 for n in [2,3]:
  for bits in itertools.product((0,1),repeat=n*n):
   A=np.array(bits,np.uint8).reshape(n,n)
   if rank(A)!=n:continue
   base=labels(A)
   for relation in RELATIONS:
    transformed=labels(transform(A,relation['correct']))
    for right in TASKS:
     if right==relation['left'] or right==relation['right']:continue
     k=(relation['relation'],right)
     if k not in examples and transformed[TASKS.index(right)]!=base[TASKS.index(relation['left'])]:
      examples[k]=dict(relation=relation['relation'],right_task=right,n=n,matrix=A.tolist(),
       left_answer=int(base[TASKS.index(relation['left'])]),transformed_candidate_answer=int(transformed[TASKS.index(right)]),
       scope='Counterexample only to candidate g(T(A))=f(A); not proof of absence of other relations.')
 for relation in RELATIONS:
  for right in TASKS:
   if right not in (relation['left'],relation['right']) and (relation['relation'],right) not in examples:
    consistent.append(dict(relation=relation['relation'],right_task=right,status='possible_not_proven'))
 json_write(DIAG/'candidate_classification_audit.json',dict(status='passed',rows=24,mathematical_types=12,
  categories=list(CATEGORIES),counts={c:sum(r['category']==c for r in rows) for c in CATEGORIES},
  designated_proofs=PROOFS,manifest_sha256=sha(DIAG/'candidate_manifest.csv'),math_audit_sha256=sha(DIAG/'math_audit.json'),
  source_sha256=sha(__file__),counterexamples=list(examples.values()),possible_not_proven=consistent,
  scope='Fixed known-exact binary catalogue; not_designated means not listed, not mathematically independent. No exhaustive absence claim. Small-n counterexamples can only disprove specific equations.',
  other_known_exact_is_empty=True,classification_fixed_before_geometry=True,no_geometry_read=True,no_models=True,
  shared_relations='R1/R2 share transpose; R3/R4 are fixed actions. Four relation instances are not four independent relation types.'))
 return rows

def selections(data,relation,split,n,control,right_task=None):
 """Joint key contains identity, correct and wrong right answers; not just T."""
 assert control in CONTROLS
 ns=np.asarray(data[split+'_lengths'],np.int64)
 ys=np.asarray(data[split+'_action_labels'],np.int64)
 right=right_task or relation['right'];li=TASKS.index(relation['left']);ri=TASKS.index(right)
 ai=[0,ACTIONS.index(relation['correct']),ACTIONS.index(relation['wrong'])]
 keys=np.column_stack([ns,ys[:,0,li],ys[:,ai,ri]])
 use=ns==int(n)
 if control=='answer_strata':
  _,inv,count=np.unique(keys,axis=0,return_inverse=True,return_counts=True);use &= count[inv]>=3
 ids=np.flatnonzero(use);chosen=keys[ids] if control=='answer_strata' else None
 if chosen is None:groups=1 if len(ids) else 0
 else:groups=len(np.unique(chosen,axis=0))
 audit=dict(selected_count=len(ids),stratum_count=groups,residual_df=len(ids)-groups,
  selection_ids_sha256=array_sha(ids),selection_keys_sha256=array_sha(chosen) if chosen is not None else 'raw',
  sample_ids_sha256=array_sha(np.asarray(data[split+'_ids'])[ids]))
 return ids,chosen,audit

def metric_values(xf,yf,xt,yt,kf,kt,ids,orbits,metrics=None):
 m=metrics or load_metrics()
 xf=m.center(xf,kf);yf=m.center(yf,kf);xt=m.center(xt,kt);yt=m.center(yt,kt)
 error,rank=m.procrustes(xf,yf,xt,yt,rank_cap=8)
 results={'linear_cka':(m.cka(xt,yt),None),'knn_overlap':(m.knn_overlap(xt,yt,k=10,ids=ids,duplicate_keys=orbits),None),
  'procrustes':(-error if error is not None else None,error)}
 reasons={}
 for name,(value,_) in results.items():
  if value is None:
   if len(xt)<2:reason='fewer_than_two_analysis_samples'
   elif np.square(m.center(xt)).sum()<=1e-20 or np.square(m.center(yt)).sum()<=1e-20:reason='constant_or_zero_centered_analysis_features'
   elif name=='knn_overlap':reason='fewer_than_k10_eligible_neighbors_or_zero_geometry'
   elif name=='procrustes':reason='fixed_rank8_insufficient_fit_rank_or_zero_scale_or_target_energy'
   else:reason='centered_Gram_norm_product_at_or_below_1e-20'
   reasons[name]=reason
 return results,rank,reasons

def decompose(values,static=False):
 row={f'{phase}_{role}':values[phase][role] for phase in PHASES for role in ROLES}
 def sub(a,b):return a-b if a is not None and b is not None else None
 row.update(B0=sub(row['initial_correct'],row['initial_wrong']),Bt=sub(row['trained_correct'],row['trained_wrong']),
  dc=sub(row['trained_correct'],row['initial_correct']),du=sub(row['trained_wrong'],row['initial_wrong']),
  di=sub(row['trained_identity'],row['initial_identity']))
 row['Delta']=sub(row['Bt'],row['B0'])
 if row['Delta'] is not None:assert abs(row['Delta']-(row['dc']-row['du']))<1e-10
 if static:assert all(row[k] is None for k in ['initial_correct','initial_wrong','initial_identity','B0','dc','du','di','Delta'])
 return row

def fixed_mean(values):
 assert len(values)==3
 return math.fsum(values)/3 if all(v is not None and math.isfinite(v) for v in values) else None
def finite_stats(values):
 finite=[v for v in values if v is not None and math.isfinite(v)]
 return dict(total_count=len(values),finite_count=len(finite),undefined_count=len(values)-len(finite),
  mean=math.fsum(finite)/len(finite) if finite else None,median=float(np.median(finite)) if finite else None)
def load_cache_index(index):
 if isinstance(index,(str,Path)):index=read(index)
 assert index['status']=='completed' and index['protocol_sha256']==sha(CONFIG)
 assert index['source_version_sha256']==source_version()
 arrays={}
 for r in index['records']:
  assert sha(r['path'])==r['metadata']['sha256']
  assert r['metadata']['signature']['protocol_sha256']==sha(CONFIG)
  assert r['metadata']['parameter_sha256_before']==r['metadata']['parameter_sha256_after']
  with np.load(r['path'],allow_pickle=False) as f:arrays[r['unit'],r['task'],r['condition'],r['split']]={k:f[k] for k in f.files}
 return index,arrays

def analyze(config,data_by_unit,index,cpu_deadline=None):
 assert verify()==config
 assert config['lengths']==[6,8,10]
 assert config['evaluation']['controls']==list(CONTROLS) and config['evaluation']['metrics']==list(NAMES)
 assert config['evaluation']['branches']==list(BRANCHES)
 assert len(config['training_manifest'])==8
 begun=time.monotonic();metrics=load_metrics();index,features=load_cache_index(index)
 import csv
 with (DIAG/'candidate_manifest.csv').open(newline='') as f:candidates=list(csv.DictReader(f))
 catalogue=read(DIAG/'candidate_classification_audit.json')
 assert catalogue['manifest_sha256']==sha(DIAG/'candidate_manifest.csv') and len(candidates)==24
 rows=[];masks=[];wrong_rows=[];pair_lengths=[]
 for unit in units(config):
  data=data_by_unit[unit]
  assert len(data['analysis_lengths'])==1536
  for r in RELATIONS:
   roles={'correct':ACTIONS.index(r['correct']),'wrong':ACTIONS.index(r['wrong']),'identity':0}
   for n in config['lengths']:
    for control in CONTROLS:
     guard(cpu_deadline)
     fit,kf,af=selections(data,r,'validation',n,control);test,kt,at=selections(data,r,'analysis',n,control)
     masks.append(dict(unit=unit,relation=r['relation'],n=n,control=control,split='validation',**af))
     masks.append(dict(unit=unit,relation=r['relation'],n=n,control=control,split='analysis',**at))
     truth=data['analysis_action_labels'];li=TASKS.index(r['left']);ri=TASKS.index(r['right'])
     wrong=truth[:,0,li]==truth[:,roles['wrong'],ri]
     correct=truth[:,0,li]==truth[:,roles['correct'],ri]
     assert correct.all(),'Exact designated formula failed on sealed data'
     wrong_rows.append(dict(unit=unit,relation=r['relation'],n=n,control=control,original_raw_count=int((data['analysis_lengths']==n).sum()),
      retained_count=len(test),wrong_formula_satisfied_count=int(wrong[test].sum()),
      wrong_formula_fraction=float(wrong[test].mean()) if len(test) else None,correct_formula_satisfied_count=int(correct[test].sum())))
     ids=test;orbits=np.asarray(data['analysis_orbit_ids'])[test,None]
     for branch in BRANCHES:
      static=branch=='gold_onehot'
      values={m:{phase:{role:None for role in ROLES} for phase in PHASES} for m in NAMES}
      errors={phase:{role:None for role in ROLES} for phase in PHASES};reasons={};rank=8
      for phase in ('trained',) if static else PHASES:
       for role,action in roles.items():
        guard(cpu_deadline)
        def get(split,use,right=False):
         task=r['right'] if right else r['left'];ai=action if right else 0
         if static:
          y=data[split+'_action_labels'][use,ai,TASKS.index(task)]
          token=np.array([TOKEN_TO_ID[f'{int(v):02d}'] for v in y]);return np.eye(188)[token]
         return features[unit,task,phase,split][branch][use,ai]
        result,rank,why=metric_values(get('validation',fit),get('validation',fit,True),get('analysis',test),get('analysis',test,True),kf,kt,ids,orbits,metrics)
        for metric in NAMES:values[metric][phase][role]=result[metric][0]
        errors[phase][role]=result['procrustes'][1]
        reasons[phase+'_'+role]=why
      for name in NAMES:
       row=dict(unit=unit,relation=r['relation'],main=r['main'],left=r['left'],right=r['right'],
        correct_transform=r['correct'],wrong_transform=r['wrong'],n=n,control=control,branch=branch,metric=name,
        fit_count=len(fit),analysis_count=len(test),fit_strata=af['stratum_count'],analysis_strata=at['stratum_count'],
        fit_residual_df=af['residual_df'],analysis_residual_df=at['residual_df'],static_reference=static,
        procrustes_rank_cap=8,**decompose(values[name],static))
       row['undefined_reasons']={phase_role:why[name] for phase_role,why in reasons.items() if name in why}
       if name=='procrustes':row.update({f'{phase}_{role}_error':errors[phase][role] for phase in PHASES for role in ROLES})
       rows.append(row)
  # Distinct fixed-T statistic: candidate task pairs, not correct-wrong Bt.
  for candidate in [p for p in candidates if p['unit']==unit]:
   r=next(r for r in RELATIONS if r['relation']==candidate['relation']);ai=ACTIONS.index(r['correct'])
   for n in config['lengths']:
    for control in CONTROLS:
     guard(cpu_deadline)
     fit,kf,af=selections(data,r,'validation',n,control,candidate['right_task']);test,kt,at=selections(data,r,'analysis',n,control,candidate['right_task'])
     row=dict(candidate,control=control,n=n,fit_count=len(fit),analysis_count=len(test),fit_strata=af['stratum_count'],analysis_strata=at['stratum_count'],
      fit_residual_df=af['residual_df'],analysis_residual_df=at['residual_df'],metric='ordinary_linear_cka',
      fit_ids_sha256=af['selection_ids_sha256'],analysis_ids_sha256=at['selection_ids_sha256'])
     for phase in PHASES:
      x=metrics.center(features[unit,r['left'],phase,'analysis']['prefix_final'][test,0],kt)
      y=metrics.center(features[unit,candidate['right_task'],phase,'analysis']['prefix_final'][test,ai],kt)
      row['fixed_transform_score_'+phase]=metrics.cka(x,y)
     a,b=(row['fixed_transform_score_'+p] for p in PHASES)
     row['training_score_change']=b-a if a is not None and b is not None else None
     pair_lengths.append(row)
 assert len(rows)==432 and len(masks)==96 and len(wrong_rows)==48 and len(pair_lengths)==144
 fields=[f'{phase}_{role}' for phase in PHASES for role in ROLES]+['B0','Bt','dc','du','di','Delta']
 groups=defaultdict(list)
 for row in rows:groups[tuple(row[k] for k in ['unit','relation','main','control','branch','metric'])].append(row)
 summaries=[]
 for key,rr in groups.items():
  assert sorted(r['n'] for r in rr)==[6,8,10]
  s=dict(zip(['unit','relation','main','control','branch','metric'],key),n_count=3,weight_each='1/3',static_reference=rr[0]['static_reference'])
  for field in fields:s[field]=fixed_mean([r[field] for r in rr]);s[field+'_undefined_n']=','.join(str(r['n']) for r in rr if r[field] is None)
  if rr[0]['metric']=='procrustes':
   for phase in PHASES:
    for role in ROLES:s[f'{phase}_{role}_error']=fixed_mean([r[f'{phase}_{role}_error'] for r in rr])
  summaries.append(s)
 assert len(summaries)==144
 pg=defaultdict(list)
 for p in pair_lengths:pg[p['pair_id'],p['control']].append(p)
 ps=[]
 for rr in pg.values():
  rr.sort(key=lambda r:r['n']);assert [r['n'] for r in rr]==[6,8,10]
  s={k:rr[0][k] for k in ['unit','relation','main','left_task','right_task','correct_transform','designated_right_task','category','pair_id']}
  s.update(control=rr[0]['control'],n_count=3,weights='1/3; complete three scales required')
  for phase in PHASES:
   f='fixed_transform_score_'+phase;s[f]=fixed_mean([r[f] for r in rr]);s[phase+'_finite_n']=sum(r[f] is not None for r in rr)
  a,b=(s['fixed_transform_score_'+p] for p in PHASES);s['training_score_change']=b-a if a is not None and b is not None else None
  ps.append(s)
 assert len(ps)==48
 category=[];comparisons=[];common_changes=[];ranking=[]
 pgroup=defaultdict(list)
 for s in ps:pgroup[s['unit'],s['relation'],s['control']].append(s)
 for key,rr in pgroup.items():
  identity=dict(zip(['unit','relation','control'],key));d=next(r for r in rr if r['category']=='designated')
  assert len(rr)==3 and sum(r['category']=='designated' for r in rr)==1
  for phase in PHASES:
   field='fixed_transform_score_'+phase;dv=d[field];st=finite_stats([r[field] for r in rr]);finite=[r for r in rr if r[field] is not None]
   # Competition rank with exact stored scores; all tie counts disclosed.
   rank=1+sum(r[field]>dv for r in finite) if dv is not None else None
   ties=sum(r[field]==dv for r in finite) if dv is not None else None
   others=[r[field] for r in rr if r['category']!='designated'];os=finite_stats(others)
   ranking.append(dict(**identity,phase=phase,designated_task=d['right_task'],designated_score=dv,designated_rank=rank,exact_score_tie_count=ties,
    candidate_total=3,candidate_finite=st['finite_count'],candidate_mean=st['mean'],candidate_median=st['median'],
    nondesignated_mean=os['mean'],nondesignated_median=os['median'],
    designated_minus_all_mean=dv-st['mean'] if dv is not None and st['mean'] is not None else None,
    designated_minus_nondesignated_mean=dv-os['mean'] if dv is not None and os['mean'] is not None else None,
    greater_than_all_other_finite=all(dv>v for v in others if v is not None) if dv is not None and os['finite_count']==len(others) else None))
   for name in CATEGORIES:
    cand=[r for r in rr if r['category']==name];stats=finite_stats([r[field] for r in cand])
    category.append(dict(**identity,phase=phase,category=name,**stats,status='empty_fixed_category' if not cand else 'finite_description' if stats['finite_count'] else 'no_finite_candidates',
     finite_support_right_tasks=';'.join(r['right_task'] for r in cand if r[field] is not None)))
    if name!='designated':
     comparisons.append(dict(**identity,phase=phase,comparison_category=name,designated_score=dv,**stats,
      designated_minus_category_mean=dv-stats['mean'] if dv is not None and stats['mean'] is not None else None,
      designated_minus_category_median=dv-stats['median'] if dv is not None and stats['median'] is not None else None))
  for name in CATEGORIES:
   cand=[r for r in rr if r['category']==name];common=[r for r in cand if all(r['fixed_transform_score_'+p] is not None for p in PHASES)]
   a=finite_stats([r['fixed_transform_score_initial'] for r in common])['mean'];b=finite_stats([r['fixed_transform_score_trained'] for r in common])['mean']
   d0,dt=(d['fixed_transform_score_'+p] for p in PHASES);r0=d0-a if d0 is not None and a is not None else None;rt=dt-b if dt is not None and b is not None else None
   common_changes.append(dict(**identity,category=name,candidate_total=len(cand),finite_common_count=len(common),common_initial_mean=a,common_trained_mean=b,
    designated_minus_common_initial=r0,designated_minus_common_trained=rt,Gcat=rt-r0 if r0 is not None and rt is not None else None,
    common_support_right_tasks=';'.join(r['right_task'] for r in common),not_original_Delta=True))
 assert len(category)==96 and len(comparisons)==64 and len(common_changes)==48 and len(ranking)==32
 # Full raw boolean bottom table; no accidental matches removed by geometry selectors.
 actual=[]
 for unit in units(config):
  data=data_by_unit[unit];ys=data['analysis_action_labels'];ns=data['analysis_lengths']
  for r in RELATIONS:
   left=ys[:,0,TASKS.index(r['left'])];correct=ys[:,ACTIONS.index(r['correct']),TASKS.index(r['right'])];wrong=ys[:,ACTIONS.index(r['wrong']),TASKS.index(r['right'])]
   assert np.array_equal(left,correct)
   for i in range(len(ns)):actual.append(dict(unit=unit,relation=r['relation'],n=int(ns[i]),anchor_row=i,anchor_id=str(data['analysis_ids'][i]),orbit_id=str(data['analysis_orbit_ids'][i]),
    left_answer=int(left[i]),correct_right_answer=int(correct[i]),wrong_right_answer=int(wrong[i]),wrong_formula_satisfied=int(left[i]==wrong[i])))
 assert len(actual)==12288
 wrong_summary=[]
 for unit in units(config):
  for relation in RELATIONS:
   for control in CONTROLS:
    rr=[w for w in wrong_rows if w['unit']==unit and w['relation']==relation['relation'] and w['control']==control]
    assert len(rr)==3
    numerator=sum(w['wrong_formula_satisfied_count'] for w in rr);denominator=sum(w['retained_count'] for w in rr)
    wrong_summary.append(dict(unit=unit,relation=relation['relation'],control=control,equal_n_fraction=fixed_mean([w['wrong_formula_fraction'] for w in rr]),
     pooled_numerator=numerator,pooled_denominator=denominator,pooled_fraction=numerator/denominator if denominator else None,
     unique_raw_anchors=1536,not_independent_relation_replications=True))
 paths=[]
 for filename,records in [('geometry_per_n.csv',rows),('geometry_summary.csv',summaries),('selection_masks.csv',masks),('W_actual_per_n.csv',wrong_rows),('W_actual_per_anchor.csv',actual),('W_actual_summary.csv',wrong_summary),
  ('pair_per_n.csv',pair_lengths),('pair_summary.csv',ps),('category_summary.csv',category),('designated_category_comparisons.csv',comparisons),('common_support_training_comparisons.csv',common_changes),('designated_rank.csv',ranking)]:
  guard(cpu_deadline);csv_write(DIAG/filename,records);paths.append(DIAG/filename)
 json_write(DIAG/'analysis_completion.json',dict(status='completed',protocol_sha256=sha(CONFIG),source_version_sha256=source_version(),
  cache_index_sha256=sha(ROOT/'evaluation_cache/index.json'),output_sha256={str(p):sha(p) for p in paths},
  geometry_per_n=432,geometry_summary=144,pair_per_n=144,pair_summary=48,categories=96,ranking_rows=32,source_models=8,units=2,
  wall_seconds=time.monotonic()-begun,models_constructed=0,new_training=0,new_forward=0,
  interpretation='Exploratory third-domain pilot, two independent data/init units; dependent tasks/relations/scales/metrics/candidates. No significance or universal mechanism claim.'))
 return dict(geometry=rows,geometry_summary=summaries,pairs=ps,categories=category,rank=ranking)
