import argparse,csv,hashlib,json,math,subprocess,time
from statistics import mean as arithmetic_mean
from pathlib import Path
from collections import Counter,defaultdict
parser=argparse.ArgumentParser(description='Read-only original PermWorld checkpoint and saved-table audit')
parser.add_argument('--repo-root',type=Path,default=Path(__file__).resolve().parents[2])
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
ROOT=args.repo_root.resolve()
OUT=args.output.resolve()
OUT.mkdir(parents=True,exist_ok=True)
def read(p):return json.loads(Path(p).read_text())
def rows(p):
 with Path(p).open(newline='') as f:return list(csv.DictReader(f))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def num(v):return float(v) if v not in ('','nan','NaN') else float('nan')
def mean(v):return math.fsum(v)/len(v) if all(math.isfinite(x) for x in v) else float('nan')
def finite(v):return v if math.isfinite(v) else None
p=read(ROOT/'diagnostics/permworld_eight_relations_20261010/protocol.json')
source=read(ROOT/'diagnostics/permworld_layers_20261009/encoding_development/artifacts/protocol.json')
m=rows(ROOT/'diagnostics/permworld_eight_relations_20261010/completed_checkpoint_manifest.csv')
start=time.monotonic()
checks=[]; bindings=[]
for r in m:
 d={k:r[k] for k in ['run_id','architecture','encoding','task','model_seed','data_seed','batch_seed','initial_parameter_sha256','starting_rng_hash']}
 for key,hkey in [('checkpoint_path','checkpoint_sha256'),('completed_path','completed_sha256')]:
  path=Path(r[key]); got=sha(path) if path.is_file() else None
  d[key]={'path':str(path),'exists':path.is_file(),'sha256':got,'matches_manifest':got==r[hkey]}
  checks.append(got==r[hkey])
 completion=read(r['completed_path']);d['completion_step']=completion['step'];d['completion_initial_parameter_sha256']=completion['initial_parameter_sha256'];d['completion_finite_gradient_updates']=completion['finite_gradient_updates']
 checks.extend([completion['step']==20000,completion['finite_gradient_updates']==20000,completion['initial_parameter_sha256']==r['initial_parameter_sha256']])
 bindings.append(d)
initial=[]
for path in sorted({r['initial_checkpoint_path'] for r in m}):
 rr=next(r for r in m if r['initial_checkpoint_path']==path)
 initial.append({'path':path,'architecture':rr['architecture'],'sha256':sha(path),'expected_sha256':rr['initial_checkpoint_sha256'],'parameter_sha256':rr['initial_parameter_sha256']})
initial_streams=[]
for path in sorted({r['starting_rng_path'] for r in m}):
 rr=next(r for r in m if r['starting_rng_path']==path)
 initial_streams.append({'path':path,'sha256':sha(path),'expected_sha256':rr['starting_rng_checkpoint_sha256'],'rng_hash':rr['starting_rng_hash'],'models_using':sum(r['starting_rng_path']==path for r in m)})
groups=defaultdict(list)
with (ROOT/'diagnostics/permworld_eight_relations_20261010/per_length_results.csv').open(newline='') as f:
 for r in csv.DictReader(f):
  if r['layer']=='final_norm' and r['metric']=='linear_cka' and r['branch']=='hidden':groups[r['architecture'],r['encoding'],r['relation'],r['control']].append(r)
A=[]
for k,rr in sorted(groups.items()):
 rr.sort(key=lambda r:int(r['length']));assert [int(r['length']) for r in rr]==list(range(10,31))
 d=dict(zip(['architecture','encoding','relation','control'],k));
 for field in ['initial_correct','initial_wrong','trained_correct','trained_wrong','B0','Bt','dc','dw','Delta']:d[field]=mean([num(r[field]) for r in rr])
 for field,expected in [('B0',d['initial_correct']-d['initial_wrong']),('Bt',d['trained_correct']-d['trained_wrong']),('dc',d['trained_correct']-d['initial_correct']),('dw',d['trained_wrong']-d['initial_wrong']),('Delta',d['Bt']-d['B0'])]:assert abs(d[field]-expected)<=1e-10
 A.append(d)
pgroups=defaultdict(list)
for r in rows(ROOT/'diagnostics/representation_pairing_review_20261011/pair_summary.csv'):
 if r['domain']=='permworld':pgroups[r['architecture'],r['encoding'],r['relation'],r['control']].append(r)
B=[]
for k,rr in sorted(pgroups.items()):
 assert len(rr)==15 and len({r['right_task'] for r in rr})==15
 des=next(r for r in rr if r['category']=='designated');other=[r for r in rr if r['category']=='not_designated'];assert len(other)==14
 d=dict(zip(['architecture','encoding','relation','control'],k))
 for phase in ['initial','trained']:
  score=num(des['fixed_transform_score_'+phase]);values=[num(r['fixed_transform_score_'+phase]) for r in other]
  d[phase+'_designated_score']=score;d[phase+'_other_mean']=mean(values);d[phase+'_contrast']=score-mean(values)
  d[phase+'_rank']=1+sum(v>score for v in values);d[phase+'_ties']=sum(v==score for v in values);d[phase+'_above_mean']=score>mean(values);d[phase+'_unique_first']=d[phase+'_rank']==1 and d[phase+'_ties']==0
 B.append(d)
counts={}
for control in ['raw_within_length','answer_strata']:
 aa=[r for r in A if r['control']==control];bb=[r for r in B if r['control']==control];assert len(aa)==len(bb)==32
 counts[control]={'conditions':32,'A_Bt_positive':sum(r['Bt']>0 for r in aa),'A_Delta_positive':sum(r['Delta']>0 for r in aa),'B_above_mean':sum(r['trained_above_mean'] for r in bb),'B_unique_first':sum(r['trained_unique_first'] for r in bb),'initial_B_contrast_range':[min(r['initial_contrast'] for r in bb),max(r['initial_contrast'] for r in bb)]}
assert counts['raw_within_length']['A_Bt_positive']==27 and counts['answer_strata']['A_Bt_positive']==26
assert counts['raw_within_length']['A_Delta_positive']==25 and counts['answer_strata']['A_Delta_positive']==18
assert counts['raw_within_length']['B_above_mean']==30 and counts['answer_strata']['B_above_mean']==29
assert counts['raw_within_length']['B_unique_first']==25 and counts['answer_strata']['B_unique_first']==19
assert max(abs(x) for x in counts['raw_within_length']['initial_B_contrast_range'])<=1e-15
assert all(max(num(r['fixed_transform_score_initial']) for r in rr)==min(num(r['fixed_transform_score_initial']) for r in rr) for k,rr in pgroups.items() if k[-1]=='raw_within_length')
for filename,data in [('original_comparison_A_recomputed.csv',A),('original_comparison_B_recomputed.csv',B)]:
 with (OUT/filename).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
inputs=['CURRENT_RESEARCH.txt','manuscripts/representation_selectivity_20261012/manuscript.tex','diagnostics/permworld_eight_relations_20261010/protocol.json','diagnostics/permworld_eight_relations_20261010/completed_checkpoint_manifest.csv','diagnostics/permworld_eight_relations_20261010/per_length_results.csv','diagnostics/representation_pairing_review_20261011/protocol.json','diagnostics/representation_pairing_review_20261011/pair_summary.csv','experiments/permworld_encoding_archdev_20261009/training.py','experiments/permworld_eight_relations_20261010/train.py','experiments/permworld_encoding/data.py','experiments/permworld_encoding/core.py']
world=Path(p['data_path']).parent
source_files=[p['data_path'],p['batch_schedule_path'],str(world/'batch_lengths.npy'),str(world.parent/'historical_input_hashes.npy')]
audit={
 'scope':'Read-only original PermWorld unit audit; table recomputation, byte hashes and saved completion records. No model forward, optimizer, training, new anchors or Git modifications.',
 'git_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
 'worktree_status_at_audit_time':subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True).splitlines(),
 'original_independent_units':1,'original_unit_id':'archdev001','model_count':len(m),'models_per_architecture':dict(Counter(r['architecture'] for r in m)),
 'world_sampling':source['data'], 'data_seed':95200001,'batch_seed':95300001,'architecture_initial_seeds':p['architecture_model_seeds'],
 'training_seed_policy':{'old_tasks':['descents','recoils','peaks','valleys'],'old_tasks_formula':'95400000 + 1000 * architecture_index + old_task_index; indexes transformer=0/gru=1 and tasks in the listed order','old_tasks_encoding_pairing':'Identical saved start RNG within architecture and task across encodings','added_12_tasks':'Reuse the exact saved architecture-specific descents starting RNG. All 12 added tasks plus descents share it; no distinct per-added-task seeds.','new_unit_requirement':'Resample data, batch, architecture initial and dropout seed bases independently between units while preserving this within-unit coupling pattern.'},
 'evaluation_randomness':{'separate_evaluation_seed':None,'reason':'No random evaluation sampler; original generator consumes one data_rng sequentially for train, validation and test. Evaluation is deterministic with dropout disabled. Evaluation inputs are resampled by each unit data_seed.','geometry_split':'test','anchors_per_length':512,'lengths':list(range(10,31)),'length_weights':'1/21 fixed; any undefined length preserves undefined aggregate, no reweighting'},
 'comparison_B_zero':{'raw_initial_contrast_recomputed_range':counts['raw_within_length']['initial_B_contrast_range'],'raw_initial_all_15_candidate_scores_identical':True,'arithmetic_roundoff_limit':1e-15,'answer_controlled_initial_contrast_range':counts['answer_strata']['initial_B_contrast_range'],'scope':'Shared numerical parameters within architecture, task-before readout (input end), model.eval(), common raw anchors and common preprocessing. Candidate task is not in the measured prefix. Candidate-specific answer residualization and selected support need not share a zero.','not_claimed':'Training contrast does not isolate pure mathematical relation, and within-unit sharing cannot identify cross-initialization robustness.'},
 'counts_recomputed_from':{'A':'Original per_length_results.csv, exact final_norm/hidden/linear_cka then fixed 21-length averaging','B':'Original pair_summary.csv, all 15 right candidates, designate versus other14; rank strictly greater and exact ties','rank_rule':'Unique first iff no other score > designated and no exact tie. No significance/binomial model.'},
 'counts':counts,'file_verification_passed':all(checks),'file_verification_check_count':len(checks),'models':bindings,'saved_initializations':initial,'saved_start_rngs':initial_streams,
 'code_and_table_sha256':{rel:sha(ROOT/rel) for rel in inputs},'large_source_file_sha256':{path:sha(path) for path in source_files},
 'available_evaluation_cache_index':{'path':p['evaluation_index_path'],'exists':Path(p['evaluation_index_path']).is_file()},
 'audit_limits':['Weights are bound to the saved real initial state by checkpoint/completion metadata and historical execution audit. This no-forward audit verifies bytes and completion bindings; it does not rerun historical training.','Original independent unit is exploratory; four new units are independent repetitions conditional on the fixed universe, not retroactive prospective registration of the first unit.','Models and 32 conditions within a unit share anchors/weights/random streams; they are not independent inferential units.'],
 'elapsed_seconds':time.monotonic()-start}
# Independently verify numerical saved step-0 parameters, not just file names.
import torch
state_checks=[]
for meta in initial:
 saved=torch.load(meta['path'],map_location='cpu',weights_only=False);h=hashlib.sha256()
 for name,value in sorted(saved['model'].items()):
  array=value.detach().cpu().contiguous().numpy()
  for identity in (name,str(array.dtype),str(array.shape)):h.update(identity.encode())
  h.update(array.tobytes())
 state_checks.append(dict(path=meta['path'],recomputed_parameter_sha256=h.hexdigest(),saved_parameter_sha256=saved['parameter_sha256'],matches=h.hexdigest()==saved['parameter_sha256']==meta['parameter_sha256'],tensor_entries_in_state_dict=len(saved['model']),torch_version=saved.get('torch_version'),genuine_initial_stage=saved.get('stage'),GRU_value_preserving_conversion={k:saved[k] for k in ['original_initial_checkpoint_sha256','original_initial_parameter_sha256','canonical_parameter_sha256','constructor_seed_coordinates_match_verified'] if k in saved}))
assert all(row['matches'] for row in state_checks)
audit['independent_saved_state_dict_parameter_hash_verification']=state_checks
costs=defaultdict(list)
for row in m:costs[row['architecture']+'/'+row['encoding']].append(float(row['elapsed_process_wall_seconds']))
audit['historical_process_wall_cost']=dict(all64_models_hours=sum(sum(v) for v in costs.values())/3600,groups={k:dict(models=len(v),sum_hours=sum(v)/3600,mean_minutes=arithmetic_mean(v)/60,range_minutes=[min(v)/60,max(v)/60]) for k,v in costs.items()},limit='Historical process wall, not active GPU time or a guarantee on current resources.')
audit['fixed_history_index_file']=str(world.parent/'historical_input_hashes.npy')
audit['elapsed_seconds']=time.monotonic()-start
(OUT/'original_audit.json').write_text(json.dumps(audit,indent=2,allow_nan=False)+'\n')
print(json.dumps({'output':str(OUT/'original_audit.json'),'checks':len(checks),'passed':all(checks),'counts':counts,'elapsed_seconds':audit['elapsed_seconds']},indent=2))
