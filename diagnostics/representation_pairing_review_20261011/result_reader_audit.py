"""Independent arithmetic/scope reader of completed cache-only CKA tables.

No model, activation, metric producer, training, or inference import. This file
checks stored-score arithmetic, never calculates a new representation metric.
"""
import csv
import hashlib
import itertools
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

HERE=Path(__file__).resolve().parent
PHASES=('initial','trained')
CATEGORIES=('designated','other_known_exact','not_designated')
GROUP=('domain','relation','architecture','encoding','control')
TOL=1e-10

def read(path):return json.loads(Path(path).read_text())
def table(name):
 with open(HERE/name,newline='') as f:return list(csv.DictReader(f))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def num(value):
 if value is None or str(value).strip().lower() in ('','none','nan'):return None
 result=float(value)
 assert math.isfinite(result),('unexpected infinity',value)
 return result
def eq(actual,expected,tolerance=TOL):
 a=num(actual)
 assert (a is None)==(expected is None),(actual,expected)
 if a is not None:assert abs(a-expected)<=tolerance,(a,expected,tolerance)
def key(row,fields):return tuple(row[k] for k in fields)
def stats(values):
 finite=[x for x in values if x is not None]
 return dict(total_count=len(values),defined_count=len(finite),
  finite_mean=math.fsum(finite)/len(finite) if finite else None,
  finite_median=statistics.median(finite) if finite else None)
def check_stats(row,expected):
 for field in ('total_count','defined_count'):assert int(row[field])==expected[field]
 for field in ('finite_mean','finite_median'):eq(row[field],expected[field])

def main():
 completion=read(HERE/'analysis_completion.json')
 assert completion['status']=='completed_cache_only_fixed_pairing_analysis'
 protocol=read(HERE/'protocol.json');freeze=read(HERE/'freeze.json');audit=read(HERE/'analysis_audit.json')
 assert completion['analysis_audit_sha256']==sha(HERE/'analysis_audit.json')
 assert completion['protocol_sha256']==freeze['protocol_sha256']==audit['protocol_sha256']==sha(HERE/'protocol.json')
 assert audit['freeze_sha256']==sha(HERE/'freeze.json')
 assert audit['source_sha256']==completion['source_sha256']==sha(HERE/'fixed_transform_analysis.py')
 assert audit['status']=='passed' and audit['failure_count']==0
 for path,digest in freeze['source_sha256'].items():assert sha(path)==digest
 for path,digest in audit['input_sha256'].items():assert sha(path)==digest
 for path,digest in audit['output_sha256'].items():assert sha(path)==digest
 classification=table('classification_manifest.csv');candidates=table('candidate_pairs.csv')
 assert len(classification)==132 and len(candidates)==492
 assert sha(HERE/'classification_manifest.csv')==freeze['classification_manifest_sha256']==protocol['classification_manifest_sha256']
 assert sha(HERE/'classification_audit.json')==freeze['classification_audit_sha256']==protocol['classification_audit_sha256']
 assert sha(HERE/'candidate_pairs.csv')==freeze['candidate_pairs_sha256']==protocol['pair_manifest_sha256']
 assert Counter(c['domain'] for c in candidates)=={'permworld':480,'graphworld':12}
 classes={key(c,('domain','relation','right_task')):c for c in classification}
 assert len(classes)==132
 pairs={c['pair_id']:c for c in candidates};assert len(pairs)==492
 for c in candidates:
  assert c['right_task']!=c['left_task']
  original=classes[key(c,('domain','relation','right_task'))]
  for field in original:assert c[field]==original[field],(c['pair_id'],field)
 assert Counter(c['category'] for c in candidates)=={'designated':36,'not_designated':456}
 assert protocol['categories']==list(CATEGORIES)
 lengths=table('pair_per_length.csv');masks=table('pair_masks.csv');summary=table('pair_summary.csv')
 assert len(lengths)==len(masks)==20268 and len(summary)==996
 perkeys={key(r,('pair_id','control','length')):r for r in lengths};assert len(perkeys)==20268
 maskkeys={key(r,('pair_id','control','length')):r for r in masks};assert perkeys.keys()==maskkeys.keys()
 pergroup=defaultdict(list)
 for r in lengths:
  c=pairs[r['pair_id']];system=protocol['systems'][c['domain']]
  assert r['control'] in system['controls'] and int(r['length']) in system['lengths']
  for field in ('domain','relation','architecture','encoding','left_task','right_task','transform','category'):
   assert r[field]==c[field]
  assert int(r['selected_count'])==int(maskkeys[key(r,('pair_id','control','length'))]['selected_count'])
  if r['control'].startswith('raw_'):assert int(r['selected_count'])==512
  initial=num(r['fixed_transform_score_initial']);trained=num(r['fixed_transform_score_trained'])
  eq(r['training_score_change'],trained-initial if initial is not None and trained is not None else None)
  pergroup[key(r,('pair_id','control'))].append(r)
 summaries={key(r,('pair_id','control')):r for r in summary};assert len(summaries)==996 and summaries.keys()==pergroup.keys()
 selected_ids={c['pair_id'] for c in candidates if
  (c['domain']=='permworld' and c['relation']=='descent_inverse' and c['right_task'] in ['recoils','anti_fixed_points']) or
  (c['domain']=='graphworld' and c['relation']=='reverse_max' and c['right_task'] in ['max_in','min_in'])}
 assert len(selected_ids)==10
 selected_checks=[];missing_by_domain_control=defaultdict(Counter)
 for paircontrol,rr in pergroup.items():
  s=summaries[paircontrol];expected=protocol['systems'][s['domain']]['lengths']
  rr=sorted(rr,key=lambda r:int(r['length']));assert [int(r['length']) for r in rr]==expected
  weights=protocol['systems'][s['domain']]['weights'];assert len(weights)==len(expected)
  assert all(abs(w-1/len(expected))<1e-15 for w in weights)
  assert int(s['length_count'])==len(expected)
  assert int(s['minimum_selected_count'])==min(int(r['selected_count']) for r in rr)
  assert int(s['maximum_selected_count'])==max(int(r['selected_count']) for r in rr)
  for phase in PHASES:
   values=[num(r['fixed_transform_score_'+phase]) for r in rr]
   undefined=[str(r['length']) for r,v in zip(rr,values) if v is None]
   assert int(s[phase+'_defined_lengths'])==len(expected)-len(undefined)
   assert s[phase+'_undefined_lengths']==','.join(undefined)
   observed=num(s['fixed_transform_score_'+phase])
   assert (observed is None)==bool(undefined),'Missing lengths must not be silently reweighted'
   missing_by_domain_control[s['domain'],s['control']][phase+'_undefined_pairs']+=int(observed is None)
   missing_by_domain_control[s['domain'],s['control']][phase+'_undefined_length_cells']+=len(undefined)
   if s['pair_id'] in selected_ids:
    result=math.fsum(values)/len(expected) if not undefined else None
    eq(observed,result)
    selected_checks.append(dict(pair_id=s['pair_id'],control=s['control'],phase=phase,length_count=len(expected),
     recomputed_complete_equal_length_mean=result,stored_score=observed,undefined_lengths=undefined,passed=True))
  a=num(s['fixed_transform_score_initial']);b=num(s['fixed_transform_score_trained'])
  eq(s['training_score_change'],b-a if a is not None and b is not None else None)
 assert len(selected_checks)==44
 groups=defaultdict(list)
 for s in summary:groups[key(s,GROUP)].append(s)
 assert len(groups)==76
 cat=table('category_summary.csv');comparison=table('designated_category_comparisons.csv');gcat=table('common_support_training_comparisons.csv')
 assert len(cat)==456 and len(comparison)==304 and len(gcat)==228
 categoryindex={key(r,GROUP+('category','phase')):r for r in cat};assert len(categoryindex)==456
 comparisonindex={key(r,GROUP+('comparison_category','phase')):r for r in comparison};assert len(comparisonindex)==304
 gindex={key(r,GROUP+('category',)):r for r in gcat};assert len(gindex)==228
 empty_categories=0;undefined_categories=0;descriptive_categories=0;common_counts=Counter()
 for group,ss in groups.items():
  designated=[s for s in ss if s['category']=='designated'];assert len(designated)==1;designated=designated[0]
  for category in CATEGORIES:
   candidates_in_cat=[s for s in ss if s['category']==category]
   for phase in PHASES:
    values=[num(s['fixed_transform_score_'+phase]) for s in candidates_in_cat]
    st=stats(values);row=categoryindex[group+(category,phase)];check_stats(row,st)
    status='empty_fixed_category' if not candidates_in_cat else 'no_defined_candidates' if not st['defined_count'] else 'finite_candidate_description'
    assert row['status']==status
    empty_categories+=status=='empty_fixed_category';undefined_categories+=status=='no_defined_candidates';descriptive_categories+=status=='finite_candidate_description'
    support=[s for s in candidates_in_cat if num(s['fixed_transform_score_'+phase]) is not None]
    assert row['finite_support_right_tasks']==';'.join(s['right_task'] for s in support)
    assert row['finite_support_pair_ids']==';'.join(s['pair_id'] for s in support)
    assert row['independent_replications']=='False'
    if category!='designated':
     comp=comparisonindex[group+(category,phase)];check_stats(comp,st);d=num(designated['fixed_transform_score_'+phase]);eq(comp['designated_score'],d)
     assert comp['designated_pair_id']==designated['pair_id']
     eq(comp['designated_minus_category_mean_state'],d-st['finite_mean'] if d is not None and st['finite_mean'] is not None else None)
     eq(comp['designated_minus_category_median'],d-st['finite_median'] if d is not None and st['finite_median'] is not None else None)
   common=[s for s in candidates_in_cat if all(num(s['fixed_transform_score_'+p]) is not None for p in PHASES)]
   r=gindex[group+(category,)];assert int(r['candidate_total'])==len(candidates_in_cat) and int(r['finite_common_pair_count'])==len(common)
   a=stats([num(s['fixed_transform_score_initial']) for s in common])['finite_mean']
   b=stats([num(s['fixed_transform_score_trained']) for s in common])['finite_mean']
   changes=stats([num(s['training_score_change']) for s in common])
   eq(r['common_initial_mean'],a);eq(r['common_trained_mean'],b)
   eq(r['common_mean_training_score_change'],changes['finite_mean']);eq(r['common_median_training_score_change'],changes['finite_median'])
   d0=num(designated['fixed_transform_score_initial']);dt=num(designated['fixed_transform_score_trained'])
   r0=d0-a if d0 is not None and a is not None else None;rt=dt-b if dt is not None and b is not None else None
   eq(r['designated_initial_score'],d0);eq(r['designated_trained_score'],dt)
   eq(r['designated_minus_common_category_initial'],r0);eq(r['designated_minus_common_category_trained'],rt)
   eq(r['Gcat'],rt-r0 if r0 is not None and rt is not None else None)
   assert r['common_support_pair_ids']==';'.join(s['pair_id'] for s in common)
   assert r['common_support_right_tasks']==';'.join(s['right_task'] for s in common)
   assert r['not_original_Delta']==r['no_nan_length_reweighting']=='True'
   assert (r['designated_both_phases_finite']=='True')==(d0 is not None and dt is not None)
   common_counts[category]+=len(common)
 assert empty_categories==152
 # Read saved original scores only; no activations or new metric computation.
 original_check_count=0;original_maxerr=0.
 for domain in ['permworld','graphworld']:
  with open(protocol['systems'][domain]['original_results_path'],newline='') as f:old=list(csv.DictReader(f))
  original={}
  for r in old:
   if r['metric']!='linear_cka':continue
   if domain=='permworld':
    if r['branch']!='hidden' or r['layer']!='final_norm':continue
    k=(r['relation'],r['architecture'],r['encoding'],r['control'],int(r['length']))
   else:
    if r['branch']!='prefix_final':continue
    k=(r['relation'],'transformer','row_major_offdiagonal_binary',r['control'],int(r['n']))
   assert k not in original;original[k]=r
  for r in lengths:
   if r['domain']!=domain or r['category']!='designated':continue
   oldrow=original[(r['relation'],r['architecture'],r['encoding'],r['control'],int(r['length']))]
   for phase in PHASES:
    expected=num(oldrow[phase+'_correct']);got=num(r['fixed_transform_score_'+phase]);eq(got,expected,1e-8)
    if got is not None:original_maxerr=max(original_maxerr,abs(got-expected))
    original_check_count+=1
 assert original_check_count==2760
 verification_counts={}
 for name,expected_count in [('designated_endpoint_verification.csv',2760),('designated_aggregate_verification.csv',152),('independent_covariance_verification.csv',40)]:
  rr=table(name);assert len(rr)==expected_count and all(r['passed']=='True' for r in rr)
  verification_counts[name]=dict(rows=len(rr),passed=len(rr),maximum_stored_absolute_error=max((num(r['absolute_error']) or 0. for r in rr)))
 out=dict(status='passed',protocol_sha256=sha(HERE/'protocol.json'),analysis_completion_sha256=sha(HERE/'analysis_completion.json'),
  analysis_audit_sha256=sha(HERE/'analysis_audit.json'),reader_script_sha256=sha(__file__),
  candidate_scope=dict(mathematical_manifest_rows=132,permworld_candidate_pairs=480,graphworld_candidate_pairs=12,designated_model_pairs=36),
  coverage=dict(pair_length_control_rows=20268,pair_summary_rows=996,category_summary_rows=456,category_comparison_rows=304,Gcat_rows=228,group_count=76),
  stored_arithmetic_checks=dict(fixed_selected_pair_ids=sorted(selected_ids),fixed_selected_pair_phase_summaries=44,
   pair_length_training_changes=20268,pair_summary_training_changes=996,category_all_mean_median=456,category_comparisons=304,common_support_comparisons=228),
  fixed_selected_summary_checks=selected_checks,
  missing_support=[dict(domain=d,control=c,**v) for (d,c),v in sorted(missing_by_domain_control.items())],
  category_status_counts=dict(empty_fixed_category=empty_categories,no_defined_candidates=undefined_categories,finite_candidate_description=descriptive_categories),
  category_common_support_pair_exposures=dict(common_counts),
  original_designated_score_checks=dict(pair_count=36,phase_length_control_comparisons=original_check_count,maximum_absolute_error=original_maxerr,tolerance=1e-8),
  producer_verification_tables=verification_counts,
  weights='Exactly complete 21/3 length grids, fixed equal1/21 or1/3. Any undefined length leaves pair-phase aggregate undefined.',
  category_policy='Empty other_known_exact retained separately. Non-designated category uses finite complete-length candidate aggregates and records phase-specific supports; Gcat uses identical both-phase finite candidate support.',
  limits='Arithmetic/scope audit of saved scores only, not new CKA or new independent confirmation. Pair-specific answer selectors may differ between right tasks; categories are a limited known-exact catalogue, not algebraic independence. Candidate cells share specialists/inputs/world/init.',
  no_model_imports=True,no_forward=True,no_new_metrics=True,no_training=True,no_old_output_writes=True)
 (HERE/'result_reader_audit.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
 notes=['Independent completed-result arithmetic audit: PASSED.',
  '132 mathematical catalogue rows expand to480 PermWorld +12 GraphWorld candidate pairs; all right tasks exclude the left task. Classification is unchanged, and the empty other_known_exact category is retained.',
  '20268 unique length/control rows and996 pair/control summaries are complete. Raw and answer controls remain separate; Graph density is a third separate control.',
  'The fixed ten-pair reader plan independently recomputed44 initial/trained summaries from saved per-length scores. All use fixed equal weights; an undefined length is never dropped or reweighted.',
  'All456 category mean/median/count rows,304 designated-minus-category rows and228 common-support Gcat rows were independently reaggregated. Empty category state rows:152; no-defined nonempty category state rows:'+str(undefined_categories)+'.',
  'All36 designated model pairs reproduced the original saved per-length correct scores in2760 phase/length/control comparisons; maximum absolute difference:'+str(original_maxerr)+'. Producer independent covariance fixture:40 passed; this reader checks its saved verification and provenance, without rerunning that metric.',
  'Pair-phase missing support by domain/control:']
 notes.extend(json.dumps(r) for r in out['missing_support'])
 notes.extend(['The pair-specific answer masks and finite candidate supports differ between tasks. Neither a category mean nor Gcat is a pure relation intervention, original Bt/Delta, or independent-replication estimate.',
  'No model/activation/metric computation, training, additional input generation or old artifact editing was performed.'])
 (HERE/'result_reader_audit.txt').write_text('\n'.join(notes)+'\n')
 print(json.dumps({k:out[k] for k in ['status','coverage','category_status_counts','missing_support','original_designated_score_checks']}))

if __name__=='__main__':main()
