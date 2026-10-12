"""Independent read-only aggregation of the completed saved-anchor audit CSVs."""
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

HERE=Path(__file__).resolve().parent
def rows(name):
 with open(HERE/name,newline='') as f:return list(csv.DictReader(f))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

anchors=rows('W_actual_per_anchor.csv');by_unit={};counts=defaultdict(Counter);unique=defaultdict(set)
for r in anchors:
 unit=(r['domain'],r['relation'],int(r['n']),r['anchor_id'])
 assert unit not in by_unit;by_unit[unit]=r
 unique[r['domain']].add(r['anchor_id'])
 assert r['correct_formula_satisfied']=='1'
 assert (int(r['left_answer'])==int(r['mapped_right_wrong']))==(r['wrong_formula_satisfied']=='1')
 assert (int(r['left_answer'])==int(r['mapped_right_identity']))==(r['identity_formula_satisfied']=='1')
 assert (int(r['original_answer_stratum_count'])>=3)==(r['original_answer_strata_selected']=='1')
 controls=['raw_within_n' if r['domain']=='graphworld' else 'raw_within_length']
 if r['original_answer_strata_selected']=='1':controls.append('answer_strata')
 if r['domain']=='graphworld':
  assert (int(r['original_answer_density_stratum_count'])>=3)==(r['original_answer_density_strata_selected']=='1')
  if r['original_answer_density_strata_selected']=='1':controls.append('answer_density_strata')
 for c in controls:
  key=(r['domain'],r['relation'],int(r['n']),c)
  counts[key]['retained']+=1;counts[key]['wrong']+=int(r['wrong_formula_satisfied'])
  counts[key]['correct']+=int(r['correct_formula_satisfied']);counts[key]['identity']+=int(r['identity_formula_satisfied'])
lengths=rows('W_actual_per_length.csv')
assert len(lengths)==372
for r in lengths:
 key=(r['domain'],r['relation'],int(r['n']),r['control']);c=counts[key]
 for column,local in [('retained_anchor_count','retained'),('wrong_formula_satisfied_count','wrong'),('correct_formula_satisfied_count','correct'),('identity_formula_satisfied_count','identity')]:
  assert int(r[column])==c[local],(key,column)
 assert int(r['original_raw_anchor_count'])==512
 assert abs(float(r['wrong_formula_fraction'])-c['wrong']/c['retained'])<1e-14
summary=rows('W_actual_summary.csv')
assert len(summary)==28
for r in summary:
 sub=[s for s in lengths if all(s[k]==r[k] for k in ['domain','relation','control'])]
 numerator=sum(int(s['wrong_formula_satisfied_count']) for s in sub)
 denominator=sum(int(s['retained_anchor_count']) for s in sub)
 assert numerator==int(r['pooled_wrong_formula_numerator']) and denominator==int(r['pooled_retained_denominator'])
 assert abs(numerator/denominator-float(r['pooled_wrong_formula_fraction']))<1e-14
 equal=sum(float(s['wrong_formula_fraction']) for s in sub)/len(sub)
 assert abs(equal-float(r['fixed_equal_length_wrong_formula_fraction']))<1e-14
identical_graph_criterion=0
for (domain,relation,n,anchor),r in by_unit.items():
 if domain=='graphworld' and relation=='reverse_max':
  other=by_unit[(domain,'complement_out',n,anchor)]
  assert r['wrong_formula_satisfied']==other['wrong_formula_satisfied']
  identical_graph_criterion+=1
assert len(anchors)==92160 and {d:len(v) for d,v in unique.items()}=={'permworld':10752,'graphworld':1536}
assert identical_graph_criterion==1536
out=dict(status='passed',per_anchor_unique_relation_units=len(by_unit),per_length_checks=len(lengths),aggregate_checks=len(summary),
 unique_base_anchors={d:len(v) for d,v in unique.items()},
 graph_reverse_max_complement_out_raw_wrong_criterion_identical_anchor_count=identical_graph_criterion,
 dependency_proof='Because W commutes with R and C, both raw wrong-formula events are max_out(A)=max_out(W(A)). This dependency does not imply their joint answer/density selectors are identical.',
 inputs={n:sha(HERE/n) for n in ['W_actual_per_anchor.csv','W_actual_per_length.csv','W_actual_summary.csv']},script_sha256=sha(__file__),
 no_models=True,no_forward=True,no_training=True,no_cka=True,no_new_inputs=True)
(HERE/'W_reader_validation.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out))
