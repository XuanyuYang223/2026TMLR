"""Frozen mathematical catalogue and saved-anchor audit; no model/metric imports.

The catalogue is deliberately not an exhaustive algebraic independence claim.
Only the original designated binary identities are asserted. The empty
other_known_exact category is retained rather than populated by finite searches.
"""
from __future__ import annotations
import argparse
import ast
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

BASE = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
PERM_PROTOCOL = BASE / 'diagnostics/permworld_eight_relations_20261010/protocol.json'
GRAPH_PROTOCOL = BASE / 'diagnostics/graphworld_pilot_20261010/protocol.json'
GRAPH_DATA = Path('/mnt/yangx-projects-f/20261010/graphworld_pilot/world/data.npz')
CATEGORIES = ['designated', 'other_known_exact', 'not_designated']
PROOFS = {
 'descent_inverse': 'Recoil count is the descent count of the inverse. Applying inverse twice recovers pi, so recoils(pi^-1)=descents(pi).',
 'peak_complement': 'C(pi)_i=n+1-pi_i reverses every strict comparison. Each interior peak of pi is exactly a valley of C(pi).',
 'fixed_complement': 'pi_i=i iff C(pi)_i=n+1-i; hence fixed_points(pi)=anti_fixed_points(C(pi)).',
 'exceedance_inverse': 'The bijection i -> j=pi_i sends each pi_i>i to pi^-1_j=i<j, and conversely. Exceedances(pi)=deficiencies(pi^-1).',
 'subsequence_complement': 'For each fixed subsequence of indices, complement reverses all value inequalities; increasing subsequences of pi correspond to decreasing subsequences of C(pi), with equal lengths.',
 'run_complement': 'Complement reverses adjacent comparisons on every contiguous interval; maximal increasing-run length of pi equals maximal decreasing-run length of C(pi).',
 'record_complement': 'pi_i exceeds all preceding values iff C(pi)_i is below all preceding values. Left-to-right maxima and minima thus correspond.',
 'decomposition_complement': 'A global descent cut has every prefix value greater than every suffix value. After complement the prefix comprises {1,...,k}, exactly a direct-sum component cut. A nonempty permutation has one more component than component cuts, so components(C(pi))=global_descents(pi)+1.',
 'reverse_max': 'Transposition sends the outdegree of each vertex to its indegree. Taking the maximum gives max_in(R(A))=max_out(A).',
 'reverse_min': 'Transposition sends the outdegree of each vertex to its indegree. Taking the minimum gives min_in(R(A))=min_out(A).',
 'complement_out': 'Off-diagonal complement sends every outdegree d to n-1-d. Therefore min_out(C(A))=n-1-max_out(A).',
 'complement_in': 'Off-diagonal complement sends every indegree d to n-1-d. Therefore min_in(C(A))=n-1-max_in(A).',
}

def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()

def read(path): return json.loads(Path(path).read_text())
def write_json(path,obj): Path(path).write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def write_csv(path,rows):
 assert rows
 with open(path,'w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def file_record(p): return {'path':str(p),'sha256':sha(p)}

def sources():
 paths=[PERM_PROTOCOL,GRAPH_PROTOCOL,
  BASE/'external/neurips/src/neurips_permutations/math_ops.py',
  BASE/'external/neurips/configs/property_task_geometry.toml',
  BASE/'experiments/permworld_eight_relations_20261010/common.py',
  BASE/'experiments/graphworld_pilot_20261010/core.py',
  BASE/'experiments/graphworld_pilot_20261010/evaluation.py']
 return [file_record(p) for p in paths]

def relations(domain,config):
 out=[]
 for r in config['relations']:
  if domain=='permworld':
   out.append(dict(relation=r['pair_id'],left=r['left'],right=r['right'],
    correct=r['input_transform'],wrong='complement' if r['input_transform']=='inverse' else 'inverse',
    offset=int(r['right_label_offset']),mapping='y_minus_offset'))
  else:out.append(dict(r,offset=0))
 return out

def comparison_formula(r,right,domain):
 value=f"{right}({r['correct']}(x))"
 if domain=='permworld' and r['offset']: value+=f" - {r['offset']}"
 elif r['mapping']=='n_minus_1_minus_y':value=f'n - 1 - {value}'
 return f"{r['left']}(x) = {value}"

def classify():
 rows=[];registry={}
 source=BASE/'external/neurips/src/neurips_permutations/math_ops.py'
 for node in ast.parse(source.read_text()).body:
  if isinstance(node,ast.AnnAssign) and isinstance(node.target,ast.Name) and node.target.id=='PROPERTY_FUNCTIONS':
   registry={ast.literal_eval(k):v.id for k,v in zip(node.value.keys,node.value.values)}
 assert registry
 for domain,path in [('permworld',PERM_PROTOCOL),('graphworld',GRAPH_PROTOCOL)]:
  config=read(path)
  for r in relations(domain,config):
   assert r['relation'] in PROOFS
   for task in config['tasks']:
    if task==r['left']: continue
    designated=task==r['right'];formula=comparison_formula(r,task,domain)
    rows.append(dict(domain=domain,relation=r['relation'],left_task=r['left'],right_task=task,
     transform=r['correct'],category='designated' if designated else 'not_designated',
     formula=formula if designated else '',right_label_offset=r['offset'] if domain=='permworld' else '',
     right_label_map=r['mapping'],designated_right_task=r['right'],
     original_relation_formula=comparison_formula(r,r['right'],domain),
     comparison_formula=formula,proof_id=r['relation'] if designated else '',
     designation_source=str(path)))
 assert len(rows)==132 and len({(r['domain'],r['relation'],r['right_task']) for r in rows})==132
 assert Counter(r['domain'] for r in rows)=={'permworld':120,'graphworld':12}
 counts={d:{c:sum(r['domain']==d and r['category']==c for r in rows) for c in CATEGORIES}
  for d in ['permworld','graphworld']}
 assert counts=={'permworld':{'designated':8,'other_known_exact':0,'not_designated':112},
  'graphworld':{'designated':4,'other_known_exact':0,'not_designated':8}}
 write_csv(OUT/'classification_manifest.csv',rows)
 write_json(OUT/'classification_audit.json',dict(status='passed',registered_utc=datetime.now(timezone.utc).isoformat(),
  rows=len(rows),categories=CATEGORIES,category_counts=counts,
  catalogue_scope='A fixed known-exact binary-correspondence catalogue for the original directed f/T rows; not a complete classification of all possible algebraic relations.',
  not_designated_meaning='Not listed in this catalogue; does not assert mathematical independence or absence of any relation.',
  comparison_formula_meaning='For not_designated rows this is the specific candidate equation inherited from the original map, not an asserted identity.',
  search_policy='No new affine-formula enumeration or synthetic input search; no unproved consistent candidate promoted to other_known_exact.',
  possible_not_proven=[],proofs=PROOFS,
  excluded_scope='Self-task candidates, identities using three or more tasks, conditional/nonlinear identities and exhaustive independence claims are outside this catalogue.',
  registry={k:registry[k] for k in read(PERM_PROTOCOL)['tasks']},sources=sources(),
  script=file_record(Path(__file__)),manifest=file_record(OUT/'classification_manifest.csv'),
  no_new_cka_read=True,no_models=True,no_forward=True,no_training=True))
 print(json.dumps({'status':'passed','rows':132,'counts':counts,'manifest_sha256':sha(OUT/'classification_manifest.csv')}))

def stratum_counts(keys):
 _,inverse,count=np.unique(keys,axis=0,return_inverse=True,return_counts=True)
 return count[inverse]

def perm_anchor_ids(permutations,lengths):
 return np.array([hashlib.blake2b(b'['+b','.join(str(int(v)).encode() for v in p[:n])+b']',digest_size=16).hexdigest()
  for p,n in zip(permutations,lengths)])

def audit_w():
 assert (OUT/'classification_audit.json').exists(),'Freeze classification before auditing old inputs'
 classification=read(OUT/'classification_audit.json')
 assert classification['manifest']['sha256']==sha(OUT/'classification_manifest.csv')
 assert classification['script']['sha256']==sha(Path(__file__))
 all_rows=[];summaries=[];datasets={};correct_counts={}
 for domain,protocol_path,split in [('permworld',PERM_PROTOCOL,'test'),('graphworld',GRAPH_PROTOCOL,'analysis')]:
  config=read(protocol_path);path=Path(config['data_path']) if domain=='permworld' else GRAPH_DATA
  expected=config['data_sha256'] if domain=='permworld' else read(path.with_name('data_audit.json'))['data_sha256']
  assert sha(path)==expected
  keys_to_read=[split+'_lengths',split+'_labels']
  keys_to_read+= [split+'_permutations'] if domain=='permworld' else [split+'_anchor_id',split+'_orbit_id',split+'_edges',split+'_graphs']
  with np.load(path,allow_pickle=False) as ar: data={k:ar[k] for k in keys_to_read}
  ns=data[split+'_lengths'].astype(np.int64);ys=data[split+'_labels'].astype(np.int64)
  tasks=config['tasks'];actions=config['actions']
  assert ys.shape==(len(ns),len(actions),len(tasks))
  assert sorted(np.unique(ns).tolist())==config['lengths']
  assert all((ns==n).sum()==512 for n in config['lengths'])
  if domain=='permworld':
   assert actions==['identity','complement','inverse']
   ids=perm_anchor_ids(data[split+'_permutations'],ns);orbits=np.repeat('',len(ns))
  else:
   assert actions==['identity','R','WR','C','WC']
   ids=data[split+'_anchor_id'];orbits=data[split+'_orbit_id']
   # Independent labels from already saved graphs, never generating inputs.
   for n in config['lengths']:
    g=data[split+'_graphs'][ns==n,:,:n,:n]
    out=g.sum(axis=-1);inn=g.sum(axis=-2)
    check=np.stack([out.max(-1),out.min(-1),inn.max(-1),inn.min(-1)],axis=-1)
    assert np.array_equal(check,ys[ns==n])
  assert len(set(ids.tolist()))==len(ids)
  datasets[domain]=dict(file=file_record(path),split=split,actions=actions,tasks=tasks,arrays={k:{'shape':list(v.shape),'dtype':str(v.dtype)} for k,v in data.items()},
   unique_anchor_count=len(ids),rows_per_length=512,unique_orbit_count=len(set(orbits.tolist())) if domain=='graphworld' else None)
  for r in relations(domain,config):
   left=ys[:,0,tasks.index(r['left'])];right=ys[:,:,tasks.index(r['right'])]
   if domain=='permworld':right=right-r['offset']
   elif r['mapping']=='n_minus_1_minus_y':right=ns[:,None]-1-right
   c=actions.index(r['correct']);w=actions.index(r['wrong'])
   assert np.all(left==right[:,c]);correct_counts[r['relation']]=len(left)
   joint=np.column_stack([ns,left,right]) if domain=='permworld' else np.column_stack([ns,left,right[:,[0,c,w]]])
   count=stratum_counts(joint)
   densitycount=stratum_counts(np.column_stack([joint,data[split+'_edges'][:,0]])) if domain=='graphworld' else np.zeros(len(ns),dtype=int)
   wrong=left==right[:,w];identity=left==right[:,0]
   for i in range(len(ns)):
    all_rows.append(dict(domain=domain,relation=r['relation'],split=split,n=int(ns[i]),anchor_row=i,anchor_id=str(ids[i]),orbit_id=str(orbits[i]),
     left_task=r['left'],right_task=r['right'],correct_transform=r['correct'],wrong_transform=r['wrong'],formula=comparison_formula(r,r['right'],domain),
     left_answer=int(left[i]),mapped_right_identity=int(right[i,0]),mapped_right_correct=int(right[i,c]),mapped_right_wrong=int(right[i,w]),
     correct_formula_satisfied=int(left[i]==right[i,c]),wrong_formula_satisfied=int(wrong[i]),identity_formula_satisfied=int(identity[i]),
     original_answer_stratum_count=int(count[i]),original_answer_strata_selected=int(count[i]>=3),
     original_answer_density_stratum_count=int(densitycount[i]) if domain=='graphworld' else '',
     original_answer_density_strata_selected=int(densitycount[i]>=3) if domain=='graphworld' else ''))
   controls={'raw_within_length' if domain=='permworld' else 'raw_within_n':np.ones(len(ns),bool),'answer_strata':count>=3}
   if domain=='graphworld':controls['answer_density_strata']=densitycount>=3
   for control,retained in controls.items():
    for n in config['lengths']:
     raw=ns==n;use=raw&retained;num=int(wrong[use].sum());den=int(use.sum())
     summaries.append(dict(domain=domain,relation=r['relation'],split=split,n=int(n),control=control,
      original_raw_anchor_count=int(raw.sum()),retained_anchor_count=den,excluded_anchor_count=int(raw.sum()-den),
      wrong_formula_satisfied_count=num,wrong_formula_fraction=num/den if den else None,
      correct_formula_satisfied_count=int((left[use]==right[use,c]).sum()),identity_formula_satisfied_count=int(identity[use].sum()),
      original_relation_formula=comparison_formula(r,r['right'],domain),correct_transform=r['correct'],wrong_transform=r['wrong']))
 assert len(all_rows)==92160 and len(summaries)==372
 write_csv(OUT/'W_actual_per_anchor.csv',all_rows);write_csv(OUT/'W_actual_per_length.csv',summaries)
 aggregate=[]
 for domain in datasets:
  for r in relations(domain,read(PERM_PROTOCOL if domain=='permworld' else GRAPH_PROTOCOL)):
   for control in sorted({s['control'] for s in summaries if s['domain']==domain}):
    rows=[s for s in summaries if s['domain']==domain and s['relation']==r['relation'] and s['control']==control]
    numerator=sum(s['wrong_formula_satisfied_count'] for s in rows);denominator=sum(s['retained_anchor_count'] for s in rows)
    fractions=[s['wrong_formula_fraction'] for s in rows]
    assert all(v is not None for v in fractions)
    aggregate.append(dict(domain=domain,relation=r['relation'],control=control,length_count=len(rows),
     fixed_equal_length_wrong_formula_fraction=sum(fractions)/len(fractions),pooled_wrong_formula_numerator=numerator,
     pooled_retained_denominator=denominator,pooled_wrong_formula_fraction=numerator/denominator,
     original_raw_anchor_exposures=sum(s['original_raw_anchor_count'] for s in rows)))
 write_csv(OUT/'W_actual_summary.csv',aggregate)
 # Only sample-count fields are read from old completed tables, not CKA values.
 checks=[]
 for domain,name in [('graphworld','diagnostics/graphworld_pilot_20261010/geometry_per_n.csv'),('permworld','diagnostics/permworld_eight_relations_20261010/per_length_results.csv')]:
  old=BASE/name
  with open(old,newline='') as f:
   for row in csv.DictReader(f):
    expected_branch='prefix_final' if domain=='graphworld' else 'hidden'
    if row['branch']!=expected_branch or row['metric']!='linear_cka':continue
    if domain=='permworld' and (row['architecture']!='transformer' or row['encoding']!='one_line' or row['layer']!='final_norm'):continue
    n=int(row['n'] if domain=='graphworld' else row['length'])
    matches=[s for s in summaries if s['domain']==domain and s['relation']==row['relation'] and s['control']==row['control'] and s['n']==n]
    assert len(matches)==1
    actual=int(row['analysis_count'] if domain=='graphworld' else row['test_samples'])
    assert actual==matches[0]['retained_anchor_count']
    checks.append(dict(domain=domain,relation=row['relation'],n=n,control=row['control'],count=actual))
  datasets[domain]['original_count_table']=file_record(old)
 assert len(checks)==372
 randompath=BASE/'diagnostics/graphworld_pilot_20261010/mathematics_audit.json'
 random_reference=[]
 for record in read(randompath)['records']:
  for r,fraction in record['accidental_relation_fraction'].items():
   random_reference.append(dict(n=record['n'],relation=r,random_math_samples=record['random_samples'],random_math_wrong_formula_fraction=fraction))
 write_csv(OUT/'W_original_random_math_reference.csv',random_reference)
 write_json(OUT/'W_actual_audit.json',dict(status='passed',datasets=datasets,per_anchor_rows=len(all_rows),per_length_control_rows=len(summaries),
  correct_formula_all_true=correct_counts,old_analysis_count_checks=checks,old_analysis_count_check_count=len(checks),
  classification_manifest_sha256=sha(OUT/'classification_manifest.csv'),classification_audit_sha256=sha(OUT/'classification_audit.json'),
  script=file_record(Path(__file__)),outputs={name:sha(OUT/name) for name in ['W_actual_per_anchor.csv','W_actual_per_length.csv','W_actual_summary.csv','W_original_random_math_reference.csv']},
  perm_selector='Joint[n,left(identity),right(identity),right(complement),right(inverse)] after original offset; retain joint count>=3 independently per split.',
  graph_selector='Joint[n,left(identity),mappedright(identity),mappedright(correct),mappedright(wrong)]; retain joint count>=3 independently per split; density adds identity edge count.',
  raw_anchor_policy='Every original 512 anchor row per length retained in the per-anchor table, including accidentally formula-preserving wrong transforms.',
  weights='Fixed equal 1/21 Perm or 1/3 Graph length mean, separately reported from pooled numerator/denominator.',
  dependency_limit='Relations reuse anchors. 92160 relation-anchor exposures are only 10752 Perm and 1536 Graph unique base inputs; architecture/encoding copies are not additional anchors.',
  random_math_reference=file_record(randompath),random_reference_scope='Prior 3000 random mathematics-check graphs per n; not the actual analysis sample and never used in actual numerator/denominator.',
  no_models=True,no_forward=True,no_training=True,no_cka=True,no_new_inputs=True))
 print(json.dumps({'status':'passed','per_anchor_rows':len(all_rows),'count_checks':len(checks),'aggregate':aggregate}))

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['classify','audit-w'],required=True)
 args=parser.parse_args();classify() if args.stage=='classify' else audit_w()
