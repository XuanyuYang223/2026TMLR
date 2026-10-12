"""Saved actual step0/20k inference; one final-norm input-end endpoint only.

Full 188-class output probabilities and teacher-forced answer/EOS scores are
kept separately. No optimizer is created and no source update occurs.
"""
from __future__ import annotations
import os
from pathlib import Path
import time
import numpy as np
from .common import ROOT, DIAG, CONFIG, read, sha, csv_write, json_write, verify, source_version, gpu_deadline
from .core import TASKS, ACTIONS, TOKEN_TO_ID, TASK_TOKENS, encode, transform, labels, batch, prefix_length

class EvaluationBudgetExhausted(RuntimeError):pass
def guard(deadline,reserve=5.):
 if deadline is not None and time.monotonic()+reserve>=float(deadline):raise EvaluationBudgetExhausted('Fixed overall GPU monotonic deadline reached; no additional forward allowed')

def prepare_evaluation_data(data,cpu_deadline=None):
 """Only after formal freeze/seal; generate fixed action views of saved anchors.

The returned dictionary is new. The original archives are never changed.
Training/source geometry uses identity only. Final labels must be sealed rather
than regenerated here; validation labels may be derived from saved matrices.
"""
 verify()  # Final-label access is forbidden before the formal science freeze.
 out=dict(data)
 assert '_provenance' in data
 provenance=data['_provenance']
 for key,digest in [('data_path','data_sha256'),('analysis_labels_path','analysis_labels_sha256')]:assert sha(provenance[key])==provenance[digest]
 with np.load(provenance['analysis_labels_path'],allow_pickle=False) as sealed_file:
  assert sealed_file['actions'].tolist()==list(ACTIONS) and sealed_file['tasks'].tolist()==list(TASKS)
  assert np.array_equal(sealed_file['labels'],data['analysis_labels'])
  for sealed_key,data_key in [('lengths','analysis_lengths'),('ids','analysis_ids'),('orbit_ids','analysis_orbit_ids')]:
   assert np.array_equal(sealed_file[sealed_key],data[data_key]),('Sealed final-label row identity mismatch',sealed_key)
 for split in ('train','validation','analysis'):
  ns=np.asarray(data[split+'_lengths'],np.int64);N=len(ns)
  per_n={'train':4096,'validation':256,'analysis':512}[split]
  assert all(int((ns==n).sum())==per_n for n in [6,8,10]) and N==3*per_n
  assert data[split+'_input'].shape==(N,128)
  actions=(0,) if split=='train' else range(len(ACTIONS))
  prefixes=np.full((N,len(actions),128),TOKEN_TO_ID['<PAD>'],np.int16)
  ys=np.empty((N,len(actions),len(TASKS)),np.int16)
  if split=='analysis':
   sealed=np.asarray(data['analysis_labels'])
   assert sealed.shape==(N,len(ACTIONS),4);ys[:]=sealed
  for i,n in enumerate(ns):
   if cpu_deadline is not None and time.monotonic()>=float(cpu_deadline):raise EvaluationBudgetExhausted('CPU preparation deadline reached')
   A=data[split+'_matrices'][i,:n,:n]
   for local,ai in enumerate(actions):
    if ai==0:
     prefixes[i,local]=data[split+'_input'][i]
     if split!='analysis':ys[i,local]=data[split+'_labels'][i]
    else:
     B=transform(A,ACTIONS[ai]);ids=encode(B);prefixes[i,local,:len(ids)]=ids
     if split!='analysis':ys[i,local]=labels(B)
  assert np.all((ys>=1)&(ys<=ns[:,None,None]))
  out[split+'_action_input']=prefixes;out[split+'_action_labels']=ys
 return out

def manual_final(net,ids,mask):
 """Independent same-shape embedding->blocks->norm path, without a hook/head."""
 import torch
 positions=torch.arange(ids.shape[1],device=ids.device)
 x=net.token_embedding(ids)+net.position_embedding(positions)[None]
 x=net.embedding_dropout(x)*mask[...,None]
 for block in net.blocks:x=block(x,mask)
 return net.final_norm(x)*mask[...,None]

def bindings(config,job,condition):
 """Read actual saved initial weights; never call an initialization creator."""
 import torch
 if condition=='initial':
  path=Path(job['initial_checkpoint_path']);expected=job['initial_checkpoint_sha256']
  assert path.exists() and sha(path)==expected
  cp=torch.load(path,map_location='cpu',weights_only=False)
  assert cp['actual_step0'] and cp['unit']==job['unit']
  assert cp['parameter_sha256']==job['initial_parameter_sha256']
  assert cp['model_config']==config['model'] and cp['vocabulary']==TOKEN_TO_ID
 else:
  path=Path(job['checkpoint_path']);receipt=read(path.parent/'completed.json')
  expected=sha(path)
  assert receipt['checkpoint_sha256']==expected and receipt['step']==20000
  assert receipt['protocol_sha256']==sha(CONFIG) and receipt['source_version_sha256']==source_version()
  cp=torch.load(path,map_location='cpu',weights_only=False)
  required=dict(step=20000,run_id=job['run_id'],unit=job['unit'],task=job['task'],
   protocol_sha256=sha(CONFIG),source_version_sha256=source_version(),data_sha256=job['data_sha256'],
   batch_schedule_sha256=job['batch_schedule_sha256'],initial_parameter_sha256=job['initial_parameter_sha256'],
   initial_checkpoint_sha256=job['initial_checkpoint_sha256'],starting_rng_hash=job['starting_rng_hash'])
  for k,v in required.items():assert cp[k]==v,(k,cp.get(k),v)
  for k in ['run_id','unit','task','protocol_sha256','source_version_sha256','data_sha256']:
   assert receipt[k]==required[k],('receipt binding',k)
 return path,expected

def extract_cache(config,job,condition,split,data,deadline,device='cuda'):
 import torch
 from torch.nn import functional as F
 from .model_runtime import load_saved_model,parameter_hash
 cp_path,cp_hash=bindings(config,job,condition)
 provenance=data['_provenance']
 assert provenance['data_sha256']==job['data_sha256']
 sig=dict(protocol_sha256=sha(CONFIG),source_version_sha256=source_version(),
  data_sha256=provenance['data_sha256'],analysis_labels_sha256=provenance['analysis_labels_sha256'],
  checkpoint_sha256=cp_hash,unit=job['unit'],task=job['task'],condition=condition,split=split,
  evaluation='FP32/eval/inference_mode/autocast disabled/TF32 disabled',batch_size=int(config['evaluation']['batch_size']))
 directory=ROOT/'evaluation_cache'/job['unit'];directory.mkdir(parents=True,exist_ok=True)
 path=directory/f"{job['task']}_{condition}_{split}.npz"
 if path.exists():
  metadata=read(path.with_suffix('.json'))
  assert metadata['status']=='completed' and metadata['signature']==sig and metadata['sha256']==sha(path)
  return dict(unit=job['unit'],task=job['task'],condition=condition,split=split,path=str(path),metadata=metadata)
 guard(deadline)
 net=load_saved_model(config,cp_path,device);net.eval()
 assert all(p.dtype==torch.float32 for p in net.parameters())
 if condition=='initial':assert parameter_hash(net)==job['initial_parameter_sha256']
 before=parameter_hash(net);captured=[]
 if condition=='trained':assert before==read(cp_path.parent/'completed.json')['parameter_sha256']
 hook=net.final_norm.register_forward_hook(lambda module,inputs,output:captured.append(output))
 ns=np.asarray(data[split+'_lengths']);prefix=data[split+'_action_input'];truth=data[split+'_action_labels'];N=len(ns);A=prefix.shape[1]
 assert A==(1 if split=='train' else len(ACTIONS))
 arrays=dict(prefix_final=np.full((N,A,256),np.nan,np.float32),output_prob=np.full((N,A,188),np.nan,np.float32),
  prediction=np.full((N,A),-1,np.int16),answer_ce=np.full((N,A),np.nan,np.float64),eos_prediction=np.full((N,A),-1,np.int16),
  eos_ce=np.full((N,A),np.nan,np.float64),gold_token_id=np.full((N,A),-1,np.int16))
 checks=[];begun=time.monotonic();done=np.zeros((N,A),bool);batch_size=int(config['evaluation']['batch_size'])
 assert batch_size in (32,64,128)
 try:
  with torch.inference_mode():
   for n in config['lengths']:
    rows_n=np.flatnonzero(ns==n)
    for start in range(0,len(rows_n),batch_size):
     use=rows_n[start:start+batch_size]
     for ai in range(A):
      guard(deadline);gold=truth[use,ai,TASKS.index(job['task'])]
      x,mask,target,end,eq=batch(prefix[use,ai],ns[use],job['task'],answers=gold,device=device)
      rr=torch.arange(len(use),device=device)
      assert torch.all(end==prefix_length(n)-1)
      assert torch.all(x[rr,end+1]==TOKEN_TO_ID[TASK_TOKENS[job['task']]])
      assert torch.all(target[rr,eq+1]==x[rr,eq+1]) and torch.all(target[rr,eq+2]==TOKEN_TO_ID['<EOS>'])
      assert int(target.ne(-100).sum())==2*len(use)
      captures=captured;captures.clear()
      # No autocast context: both the hook and complete logits remain FP32.
      with torch.autocast(device_type=torch.device(device).type,enabled=False):z=net(x,mask)
      assert z.dtype==torch.float32 and captures[-1].dtype==torch.float32
      hidden=captures[-1][rr,end];scores=z[rr,eq];eos=z[rr,eq+1];y=x[rr,eq+1]
      arrays['prefix_final'][use,ai]=hidden.cpu().numpy()
      arrays['output_prob'][use,ai]=scores.softmax(-1).cpu().numpy()
      arrays['prediction'][use,ai]=scores.argmax(-1).cpu().numpy()
      arrays['answer_ce'][use,ai]=F.cross_entropy(scores,y,reduction='none').cpu().double().numpy()
      arrays['eos_prediction'][use,ai]=eos.argmax(-1).cpu().numpy()
      arrays['eos_ce'][use,ai]=F.cross_entropy(eos,torch.full_like(y,TOKEN_TO_ID['<EOS>']),reduction='none').cpu().double().numpy()
      arrays['gold_token_id'][use,ai]=y.cpu().numpy();done[use,ai]=True
      if split=='validation' and start==0:
       guard(deadline)
       explicit=manual_final(net,x,mask)[rr,end]
       error=float((explicit[:8]-hidden[:8]).abs().max())
       assert error<=2e-5,('Fixed same-shape hook/manual tolerance failed',job['run_id'],condition,n,ai,error)
       checks.append(dict(n=n,action=ACTIONS[ai],vectors=min(8,len(use)),same_shape=True,max_absolute_error=error,tolerance=2e-5))
  if torch.device(device).type=='cuda':torch.cuda.synchronize()
  assert done.all() and all(np.isfinite(v).all() for v in arrays.values())
  assert np.allclose(arrays['output_prob'].sum(-1),1,atol=2e-6,rtol=0)
  after=parameter_hash(net);assert before==after and all(p.grad is None for p in net.parameters())
 finally:
  hook.remove();del net
 # Uncompressed arrays avoid long compression outside the fixed wall budget.
 temporary=path.with_name(path.name+'.tmp')
 with temporary.open('wb') as f:np.savez(f,**arrays)
 os.replace(temporary,path)
 metadata=dict(status='completed',signature=sig,sha256=sha(path),checkpoint_path=str(cp_path),
  parameter_sha256_before=before,parameter_sha256_after=after,dtypes={k:str(v.dtype) for k,v in arrays.items()},
  shapes={k:list(v.shape) for k,v in arrays.items()},source_updates=0,endpoint='final_norm input-end before common query',
  answer_probability_vocabulary=188,EOS_evaluated=True,EOS_condition='teacher-forced true answer token, separately scored',
  manual_extraction_checks=checks,wall_seconds=time.monotonic()-begun)
 json_write(path.with_suffix('.json'),metadata)
 return dict(unit=job['unit'],task=job['task'],condition=condition,split=split,path=str(path),metadata=metadata)

def evaluate(config,data_by_unit,deadline=None,device='cuda'):
 import torch
 from .model_runtime import device_setup
 assert verify()==config
 deadline=gpu_deadline() if deadline is None else float(deadline)
 assert np.isfinite(deadline)
 device=device_setup(device);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 assert len(config['training_manifest'])==8 and config['training']['updates']==20000
 assert not (ROOT/'evaluation_cache/index.json').exists(),'Completed extraction is not implicitly rerun'
 records=[];source=[];begun=time.monotonic();initial_hidden={};initial_probs={}
 for job in config['training_manifest']:
  data=data_by_unit[job['unit']]
  for condition in ('initial','trained'):
   for split in ('train','validation','analysis'):
    guard(deadline)
    record=extract_cache(config,job,condition,split,data,deadline,device)
    records.append(record)
    with np.load(record['path'],allow_pickle=False) as f:arrays={k:f[k] for k in f.files}
    if condition=='initial' and split!='train':
     key=job['unit'],split
     if key in initial_hidden:
      assert np.array_equal(initial_hidden[key],arrays['prefix_final']) and np.array_equal(initial_probs[key],arrays['output_prob'])
     else:initial_hidden[key]=arrays['prefix_final'].copy();initial_probs[key]=arrays['output_prob'].copy()
    for n in config['lengths']:
     use=np.asarray(data[split+'_lengths'])==n
     train=np.asarray(data['train_labels'])[np.asarray(data['train_lengths'])==n,TASKS.index(job['task'])]
     mode=int(np.argmax(np.bincount(train,minlength=n+1)));mode_token=TOKEN_TO_ID[f'{mode:02d}']
     for ai in range(arrays['prediction'].shape[1]):
      truth=data[split+'_action_labels'][use,ai,TASKS.index(job['task'])]
      source.append(dict(unit=job['unit'],run_id=job['run_id'],task=job['task'],condition=condition,split=split,n=n,action=ACTIONS[ai],samples=int(use.sum()),
       full_vocab_answer_accuracy=float((arrays['prediction'][use,ai]==arrays['gold_token_id'][use,ai]).mean()),
       answer_ce=float(arrays['answer_ce'][use,ai].mean()),eos_accuracy=float((arrays['eos_prediction'][use,ai]==TOKEN_TO_ID['<EOS>']).mean()),
       eos_ce=float(arrays['eos_ce'][use,ai].mean()),combined_two_token_ce=float((arrays['answer_ce'][use,ai].mean()+arrays['eos_ce'][use,ai].mean())/2),
       training_mode=mode,training_mode_token_id=mode_token,training_mode_accuracy=float((truth==mode).mean()),
       EOS_condition='teacher-forced true answer; separate from answer learning',configuration_selected_using_final_results=False))
  print(dict(extracted=job['run_id'],caches=len(records)),flush=True)
 assert len(records)==48 and len(source)==816
 index=dict(status='completed',protocol_sha256=sha(CONFIG),source_version_sha256=source_version(),records=records,logical_cache_count=48,
  geometry_cache_count=32,source_only_identity_train_cache_count=16,actions=list(ACTIONS),tasks=list(TASKS),
  source_models=8,units=2,initial_hidden_and_answer_probabilities_task_independent_bitwise=True,source_updates=0,
  wall_seconds=time.monotonic()-begun,deadline_monotonic=deadline)
 json_write(ROOT/'evaluation_cache/index.json',index);csv_write(DIAG/'source_learning.csv',source)
 json_write(DIAG/'evaluation_completion.json',dict(status='completed',protocol_sha256=sha(CONFIG),source_version_sha256=source_version(),
  cache_index_sha256=sha(ROOT/'evaluation_cache/index.json'),source_learning_sha256=sha(DIAG/'source_learning.csv'),
  logical_cache_count=48,geometry_cache_count=32,source_learning_rows=816,source_models=8,units=2,source_updates=0,
  manual_extraction_checks=sum(len(r['metadata']['manual_extraction_checks']) for r in records),
  wall_seconds=index['wall_seconds'],GPU_stage_deadline_monotonic=deadline))
 return index
