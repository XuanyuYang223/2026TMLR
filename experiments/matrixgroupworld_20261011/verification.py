"""Independent fixed numerical and selection checks; no new source fitting.

The actual check plan is fixed here before final geometry: first unit, R1,n6,
raw/answer strata, first64 retained fit and analysis anchors, all three roles,
hidden/full-output branches. Unavailable checks stay unavailable. Truth onehot
constant/rank behavior is checked in fixtures and never changes frozen ties.
"""
from __future__ import annotations
from collections import Counter
import importlib.util
import math
from pathlib import Path
import time
import numpy as np
from .common import ROOT, DIAG, CONFIG, read, sha, csv_write, json_write, verify, source_version
from .core import TASKS, ACTIONS, RELATIONS, TOKEN_TO_ID, batch, encode
from .analysis import selections,metric_values,load_metrics,array_sha,units,load_cache_index,guard,NAMES

INDEPENDENT=Path('/home/yangx/ICML/diagnostics/permworld_controlled_comparison_20261010/metric_review/independent_metrics.py')
INDEPENDENT_SHA='6dbb804d22b7c4559bc8e3b41140efc50df5008c8a24698652525d57ebf0b8a3'

def independent_module():
 # Exact path and digest are bound in the formal protocol/source freeze as well.
 assert sha(INDEPENDENT)==INDEPENDENT_SHA,'Frozen independent numerical source changed'
 spec=importlib.util.spec_from_file_location('matrixgroupworld_independent_numerics',INDEPENDENT)
 m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def independent_selection(data,r,split,n,control,right_task=None):
 ns=data[split+'_lengths'];ys=data[split+'_action_labels'];right=right_task or r['right']
 ai=[0,ACTIONS.index(r['correct']),ACTIONS.index(r['wrong'])]
 keys=[(int(ns[i]),int(ys[i,0,TASKS.index(r['left'])]),*(int(ys[i,a,TASKS.index(right)]) for a in ai)) for i in range(len(ns))]
 count=Counter(keys)
 ids=np.array([i for i in range(len(ns)) if int(ns[i])==n and (control=='raw_within_n' or count[keys[i]]>=3)],np.int64)
 selected=np.array([keys[i] for i in ids],np.int64).reshape(-1,5) if control=='answer_strata' else None
 groups=len({keys[i] for i in ids}) if selected is not None else int(len(ids)>0)
 return ids,selected,len(ids)-groups

def compare(actual,expected,tolerance=1e-8):
 if actual is None or expected is None:return actual is None and expected is None,None
 error=abs(float(actual)-float(expected));return error<=tolerance,error

def fixture():
 """Synthetic-only CPU fixtures; no real final labels, checkpoints or features."""
 metrics=load_metrics();independent=independent_module();rng=np.random.default_rng(20261011317);records=[]
 xf=rng.normal(size=(64,12));yf=rng.normal(size=(64,13));xt=rng.normal(size=(64,12));yt=rng.normal(size=(64,13))
 for control in ['raw_within_n','answer_strata']:
  kf=None if control=='raw_within_n' else np.column_stack([np.arange(64)//4,np.ones(64)])
  kt=None if control=='raw_within_n' else np.column_stack([np.arange(64)//4,np.ones(64)])
  arrays=[independent.independent_center(z,k) for z,k in [(xf,kf),(yf,kf),(xt,kt),(yt,kt)]]
  ids=np.arange(64);orbits=np.arange(64)[:,None]
  result,rank,_=metric_values(xf,yf,xt,yt,kf,kt,ids,orbits,metrics)
  error,r=independent.independent_procrustes(*arrays)
  reference={'linear_cka':independent.covariance_cka(arrays[2],arrays[3]),
   'knn_overlap':independent.brute_overlap(arrays[2],arrays[3],k=10,ids=ids,duplicate_keys=orbits),
   'procrustes':-error if error is not None else None}
  assert rank==r==8
  for name in NAMES:
   passed,difference=compare(result[name][0],reference[name]);assert passed,(control,name,difference)
   records.append(dict(control=control,metric=name,actual=result[name][0],independent=reference[name],absolute_error=difference,passed=True))
 # A perfect held-out rotation becomes imperfect when only fit correspondences
 # are shuffled, demonstrating that fit data, not final data, determines Q.
 q=np.linalg.qr(rng.normal(size=(12,12)))[0]
 sf=metrics.center(xf);st=metrics.center(xt);syf=2*sf@q;syt=2*st@q
 e,_=metrics.procrustes(sf,syf,st,syt)
 shuffled,_=metrics.procrustes(sf,syf[rng.permutation(len(syf))],st,syt)
 assert e<1e-10 and shuffled>0.1
 records.append(dict(check='fit-only shuffle changes held-out map without refitting final',perfect_error=e,shuffled_fit_error=shuffled,passed=True))
 # Explicit small label table checks the >=3 joint selector with a genuinely
 # rare row; this is only a fabricated engineering table, not a data world.
 fake={}
 for split in ['validation','analysis']:
  ys=rng.integers(1,7,size=(80,8,4),dtype=np.int16);ys[:72]=np.repeat(ys[:18],4,axis=0)
  fake[split+'_action_labels']=ys;fake[split+'_lengths']=np.repeat(6,80);fake[split+'_ids']=np.arange(80)
  for r in RELATIONS:
   for control in ['raw_within_n','answer_strata']:
    a,k,audit=selections(fake,r,split,6,control);b,l,df=independent_selection(fake,r,split,6,control)
    assert np.array_equal(a,b) and ((k is None and l is None) or np.array_equal(k,l)) and audit['residual_df']==df
    records.append(dict(check='independent joint tuple Counter selection',split=split,relation=r['relation'],control=control,selected=len(a),residual_df=df,passed=True))
 # Correct shifted target locations and separately scored CE, with no model.
 import torch
 from torch.nn import functional as F
 A=np.eye(6,dtype=np.uint8);prefix=np.full((4,128),TOKEN_TO_ID['<PAD>'],np.int16);encoded=encode(A);prefix[:,:len(encoded)]=encoded
 for task in TASKS:
  answers=np.repeat(1 if task in ('F1','F2') else 2,4).astype(np.int16)
  ids,mask,target,end,eq=batch(prefix,np.repeat(6,4),task,answers=answers,device='cpu');rr=torch.arange(4)
  assert torch.all(target[rr,eq+1]==ids[rr,eq+1]) and torch.all(target[rr,eq+2]==TOKEN_TO_ID['<EOS>'])
  assert torch.all(target.ne(-100).sum(1)==2)
  logits=torch.from_numpy(rng.normal(size=(*ids.shape,188)).astype(np.float32))
  a=F.cross_entropy(logits[rr,eq],ids[rr,eq+1]);eos=F.cross_entropy(logits[rr,eq+1],target[rr,eq+2])
  combined=F.cross_entropy(logits[:,:-1].reshape(-1,188),target[:,1:].reshape(-1),ignore_index=-100)
  assert abs(float(combined-(a+eos)/2))<1e-6
  records.append(dict(check='exact shifted answer and EOS supervision/CE',task=task,passed=True))
 constants=np.ones((64,188));keys=np.column_stack([np.arange(64)//4,np.ones(64)])
 assert np.square(metrics.center(constants,keys)).sum()==0
 for name,v in metric_values(constants,constants,constants,constants,keys,keys,np.arange(64),np.arange(64)[:,None],metrics)[0].items():assert v[0] is None
 result=dict(status='passed',fixture_only=True,records=records,count=len(records),
  metric_source_sha256=sha(Path('/home/yangx/ICML/diagnostics/permworld_scope_inventory_20261009/metrics.py')),
  independent_source_sha256=sha(INDEPENDENT),source_updates=0,real_final_labels_read=False,models_constructed=0,new_forward=0,
  actual_plan=dict(unit='first manifest unit',relation='R1',n=6,fit_anchors=64,analysis_anchors=64,controls=['raw_within_n','answer_strata'],
   phases=['initial','trained'],roles=['correct','wrong','identity'],branches=['prefix_final','output_prob'],tolerance=1e-8,
   insufficient_support='unavailable; no batch replacement, threshold relaxation or rank adaptation'))
 json_write(DIAG/'numerical_fixture.json',result);return result

def verify_actual(config,data_by_unit,index,cpu_deadline=None):
 assert verify()==config
 metrics=load_metrics();independent=independent_module();index,features=load_cache_index(index)
 begun=time.monotonic();records=[];selection_checks=[];unit=units(config)[0];r=RELATIONS[0];n=6;data=data_by_unit[unit]
 assert r['relation']=='R1'
 # Independently verify all24 candidate selectors, not just the designated8.
 import csv
 with (DIAG/'candidate_manifest.csv').open(newline='') as f:candidates=list(csv.DictReader(f))
 for candidate in candidates:
  current=data_by_unit[candidate['unit']];relation=next(x for x in RELATIONS if x['relation']==candidate['relation'])
  for split in ['validation','analysis']:
   for nn in [6,8,10]:
    for control in ['raw_within_n','answer_strata']:
     guard(cpu_deadline)
     ids,keys,audit=selections(current,relation,split,nn,control,candidate['right_task'])
     explicit,explicitkeys,df=independent_selection(current,relation,split,nn,control,candidate['right_task'])
     assert np.array_equal(ids,explicit) and audit['residual_df']==df
     assert (keys is None and explicitkeys is None) or np.array_equal(keys,explicitkeys)
     selection_checks.append(dict(unit=candidate['unit'],relation=relation['relation'],right=candidate['right_task'],split=split,n=nn,control=control,
      selected=len(ids),residual_df=df,ids_sha256=array_sha(ids),passed=True))
 assert len(selection_checks)==288
 roles={'correct':ACTIONS.index(r['correct']),'wrong':ACTIONS.index(r['wrong']),'identity':0}
 for control in ['raw_within_n','answer_strata']:
  fit,kf,_=selections(data,r,'validation',n,control);test,kt,_=selections(data,r,'analysis',n,control)
  if len(fit)<64 or len(test)<64:
   records.append(dict(unit=unit,relation='R1',n=6,control=control,status='unavailable',reason='frozen selector has fewer than64 fit or analysis anchors',fit=len(fit),analysis=len(test)))
   continue
  fit=fit[:64];test=test[:64];kf=kf[:64] if kf is not None else None;kt=kt[:64] if kt is not None else None
  for branch in ['prefix_final','output_prob']:
   for phase in ('initial','trained'):
    for role,ai in roles.items():
     guard(cpu_deadline)
     def get(split,use,right=False):
      task=r['right'] if right else r['left'];action=ai if right else 0
      if branch=='gold_onehot':
       y=data[split+'_action_labels'][use,action,TASKS.index(task)]
       return np.eye(188)[[TOKEN_TO_ID[f'{int(v):02d}'] for v in y]]
      return features[unit,task,phase,split][branch][use,action]
     arrays=(get('validation',fit),get('validation',fit,True),get('analysis',test),get('analysis',test,True))
     independent_arrays=tuple(independent.independent_center(x,k) for x,k in zip(arrays,(kf,kf,kt,kt)))
     result,rank,_=metric_values(*arrays,kf,kt,test,np.asarray(data['analysis_orbit_ids'])[test,None],metrics)
     error,rank2=independent.independent_procrustes(*independent_arrays);assert rank==rank2==8
     expected={'linear_cka':independent.covariance_cka(independent_arrays[2],independent_arrays[3]),
      'knn_overlap':independent.brute_overlap(independent_arrays[2],independent_arrays[3],k=10,ids=test,duplicate_keys=np.asarray(data['analysis_orbit_ids'])[test,None]),
      'procrustes':-error if error is not None else None}
     for name in NAMES:
      passed,difference=compare(result[name][0],expected[name])
      records.append(dict(unit=unit,relation='R1',n=6,control=control,branch=branch,phase=phase,role=role,metric=name,
       status='passed' if passed and expected[name] is not None else 'undefined_consistent' if passed else 'failed',
       production=result[name][0],independent=expected[name],absolute_error=difference,tolerance=1e-8,
       fit_count=64,analysis_count=64,fit_ids_sha256=array_sha(fit),analysis_ids_sha256=array_sha(test),fixed_rank=rank))
 failures=[r for r in records if r['status']=='failed'];unavailable=[r for r in records if r['status']=='unavailable']
 counts=Counter(r['status'] for r in records)
 result=dict(status='failed' if failures else 'passed_with_unavailable_checks' if unavailable else 'passed',
  protocol_sha256=sha(CONFIG),source_version_sha256=source_version(),cache_index_sha256=sha(ROOT/'evaluation_cache/index.json'),
  independent_source_sha256=sha(INDEPENDENT),source_models=8,units=2,records=records,selection_checks=selection_checks,
  selection_check_count=288,status_counts=dict(counts),failure_count=len(failures),unavailable_count=len(unavailable),
  wall_seconds=time.monotonic()-begun,source_updates=0,models_constructed=0,new_forward=0,
  scope='Fixed64 engineering recomputation of saved arrays, not all432 production cells. Null metrics stay undefined and unavailable support is not called a passed metric.',
  cutoff_policy='No change to source samples, selector, k10, rank8, tolerance, metric or numerical method in response to results.')
 json_write(DIAG/'actual_numerical_verification.json',result)
 if failures:raise AssertionError('Fixed independent numerical verification failed; no tolerance relaxation')
 return result
