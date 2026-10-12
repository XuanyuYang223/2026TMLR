"""Read-only extension of the frozen PermWorld checkpoint diagnostic.

All writes are confined to this new directory and its explicitly named F-drive
cache. No training or modification of an inherited artifact is performed.
"""
import argparse
from collections import defaultdict
import csv
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
WS = ROOT.parents[1]
OLD = WS / 'diagnostics/permworld_scope_inventory_20261009'
CACHE = Path('/mnt/yangx-projects-f/20261009/permworld_new_artifacts/layers_cache')
sys.path[:0] = [str(OLD), str(WS), str(WS/'external/neurips/src')]
import numpy as np
import torch
import evaluate as inherited
from inventory import sha, js, csv_read
from metrics import center, scores
from prepare_eval import ACTIONS
from neurips_permutations.passage import TOKEN_TO_ID
from experiments.longrun_attention import accelerate

LAYERS = ('block_01', 'block_02', 'block_03', 'block_04', 'final_norm')
BRANCHES = tuple(f'{layer}_{pool}' for layer in LAYERS for pool in ('end', 'value_mean'))
CONSTANTS = ('embedding_end', 'embedding_value_mean')
FIELDS = ['initial_correct','initial_wrong','initial_identity','trained_correct','trained_wrong','trained_identity',
          'B0','Bt','delta_correct','delta_wrong','delta']
KEYS = ['cohort','relation','family','kind','branch','control','metric']


def save(name, value):
    (ROOT/name).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def write_csv(name, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    target=ROOT/name;temporary=target.with_suffix(target.suffix+'.partial')
    with temporary.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    temporary.replace(target)


def source_rows():
    pairs = js(OLD/'evaluation_pairs.json')
    wanted = {(p['cohort'], p[f'{s}_id']) for p in pairs for s in ('left','right')}
    return [r for r in csv_read(OLD/'training_inventory.csv') if (r['cohort'],r['logical_position']) in wanted]


def freeze():
    if (ROOT/'freeze.json').exists():
        verify();return
    inherited.verify()
    files = [ROOT/'run.py', ROOT/'input_geometry.py', ROOT/'summarize.py',ROOT/'compatible_extraction_freezes.json',
             OLD/'metrics.py', OLD/'evaluate.py', OLD/'inventory.py', OLD/'relations.py',
             OLD/'evaluation_inputs.npz', OLD/'evaluation_pairs.json', OLD/'training_inventory.csv',
             WS/'experiments/longrun_attention.py', WS/'experiments/permworld_combinations.py',
             WS/'external/neurips/src/neurips_permutations/models.py',
             WS/'external/neurips/src/neurips_permutations/training.py']
    protocol = {
        'status':'frozen_before_new_layer_results', 'exploratory':True, 'new_source_training':0,
        'source_checkpoint_count':len(source_rows()), 'source_conditions':164,
        'main':'Inherited final_norm input end before task; exact inherited metric rows retained.',
        'layers':LAYERS, 'new_branches':BRANCHES,
        'pooling':'ONE_END at2n+3; sensitivity is contextual numeric-token mean positions4,6,...,2n+2 only.',
        'extraction_numerics':'Prefix-only by default; if legacy comparison exceeds2e-5, use exact original task suffix and require original2e-5 tolerance. Measured states causally precede task, unchanged mathematically. Log both errors/method; no effect-based method selection.',
        'structural_constants':list(CONSTANTS),
        'constant_rule':'Same length embedding end and unordered mean are mathematically constant; mark all metrics undefined, do not analyze summation noise.',
        'input_baselines':['position_numeric_onehot','position_numeric_embedding'],
        'input_position_rule':'Flatten numeric token embeddings by position. Position and punctuation contributions are sample-constant within length and cancel under centering.',
        'onehot_neighbor_numerics':'Within-length onehot Euclidean distances use their exact integer Gram before centering (translation does not alter distances). This preserves predeclared sample-ID tie handling rather than floating-point tie breaking. Answer-strata use the centered Gram. Hidden-state neighbor implementation is unchanged.',
        'controls':['within_length','answer_strata'],
        'metrics':'Inherited linear CKA, k10 Euclidean neighbor overlap, fitted rank8 heldout Procrustes; debiased CKA sensitivity.',
        'lengths':list(range(10,31)), 'weights':'Each length1/21; any undefined length leaves the overall value undefined.',
        'splits':'Exactly inherited independent fitting/test orbits and input order; no new inputs or source weights.',
        'answers':'Reuse inherited true-answer onehot/scalar and numeric output probability controls; no subtraction of output CKA from hidden CKA.',
        'initialization':'All existing sources grade C constructor rebuild; original step0 identity unverified. No seed-only exact identity claims.',
        'decomposition':'B0=initial correct-wrong;Bt=trained correct-wrong;dc=trained correct-initial correct;dw=trained wrong-initial wrong;Delta=Bt-B0=dc-dw.',
        'summaries':'All available same-seed pairs, each cohort separate; complete length means, relation/source medians, leave-one-relation and shared-family removals. No independent-relation or world CI from seeds.',
        'validation':{'hooks_max_abs':2e-5,'legacy_activation_max_abs':2e-5,'independent_metric_max_abs':1e-8},
        'selection':'No choice of favorable layer, task, length, metric or kernel after results.',
        'storage':str(CACHE), 'F17':'Existing F17 artifacts and all inherited files stay unchanged.',
        'scope':'Complete analysis and authorized engineering preflight only; no formal bulk training or GroupWorld.',
        'git_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=WS,text=True).strip(),
        'torch':torch.__version__, 'numpy':np.__version__,
    }
    save('protocol.json',protocol)
    save('freeze.json', {'protocol_sha256':sha(ROOT/'protocol.json'),
         'files':{str(p):sha(p) for p in files}, 'created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())})
    CACHE.mkdir(parents=True,exist_ok=True)
    save('storage.json',{'activation_cache':str(CACHE),'cache_is_external':True,'source_inputs':str(OLD/'evaluation_inputs.npz')})


def verify():
    frozen=js(ROOT/'freeze.json')
    assert sha(ROOT/'protocol.json')==frozen['protocol_sha256']
    for path,expected in frozen['files'].items():
        assert sha(path)==expected, ('Changed frozen file',path)


@torch.inference_mode()
def forward_layers(model, inputs, ns, device, batch=256, task_suffix=None):
    """Task-preceding states; optional causal suffix matches legacy numerics."""
    model=accelerate(model).to(device).eval()
    out={b:[] for b in BRANCHES}
    for start in range(0,len(ns),batch):
        n=torch.as_tensor(ns[start:start+batch],device=device,dtype=torch.long)
        x=torch.as_tensor(inputs[start:start+batch],device=device,dtype=torch.long)
        end=2*n+3; last=end if task_suffix is None else end+2
        width=int(last.max())+1
        ids=torch.full((len(x),width),TOKEN_TO_ID['<PAD>'],device=device,dtype=torch.long)
        ids[:,:min(width,x.shape[1])]=x[:,:width]
        positions=torch.arange(width,device=device)[None,:]
        valid=positions<=last[:,None]
        ids[~valid]=TOKEN_TO_ID['<PAD>']
        if task_suffix is not None:
            ar=torch.arange(len(ids),device=device)
            ids[ar,end+1]=TOKEN_TO_ID[f'<{task_suffix.upper()}>'];ids[ar,end+2]=TOKEN_TO_ID['=']
        h,_=model._embed_inputs(ids,valid)
        ar=torch.arange(len(ids),device=device)
        numeric=(positions>=4)&(positions<=2*n[:,None]+2)&(positions.remainder(2)==0)
        def collect(layer, state):
            out[layer+'_end'].append(state[ar,end].cpu().numpy())
            pooled=(state*numeric[:,:,None]).sum(1)/n[:,None]
            out[layer+'_value_mean'].append(pooled.cpu().numpy())
        for i,block in enumerate(model.blocks):
            h=block(h,valid);collect(f'block_{i+1:02d}',h)
        collect('final_norm',model.final_norm(h))
    return {b:np.concatenate(xs) for b,xs in out.items()}


def cache_path(row,condition):
    return CACHE/(inherited.name(row,condition)+'.npy')


def extract(limit=None):
    verify();data=np.load(OLD/'evaluation_inputs.npz');rows=source_rows()
    if limit is not None: rows=rows[:limit]
    device='cuda' if torch.cuda.is_available() else 'cpu'
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    began=time.monotonic();audit=[]
    for i,row in enumerate(rows):
        assert sha(row['checkpoint'])==row['checkpoint_sha256']
        for cond in ('initial','trained'):
            target=cache_path(row,cond);meta=target.with_suffix('.json')
            if target.exists() and meta.exists():
                rec=js(meta);assert sha(target)==rec['archive_sha256']
                assert sha(target.with_suffix('.embedding.npy'))==rec['embedding_sha256']
                assert rec['data_sha256']==sha(OLD/'evaluation_inputs.npz')
                assert rec['checkpoint_sha256']==row['checkpoint_sha256']
                assert rec['branch_order']==list(BRANCHES)
                assert rec['shape']==[len(BRANCHES),len(data['lengths']),4,256]
                accepted=[sha(ROOT/'freeze.json')]+js(ROOT/'compatible_extraction_freezes.json')['freeze_sha256']
                assert rec['freeze_sha256'] in accepted
                assert rec['legacy_final_norm_max_abs']<=2e-5
                audit.append(rec);continue
            model,ph=inherited.make_source(row,cond)
            legacy_meta=js(OLD/'features'/(inherited.name(row,cond)+'.json'))
            assert ph==legacy_meta['parameter_sha256'], 'rebuild changed; identity grade remains C'
            embedding=model.token_embedding.weight.detach().cpu().numpy().copy()
            flat=data['input'].reshape(-1,64);ns=np.repeat(data['lengths'],4)
            new=forward_layers(model,flat,ns,device)
            arrays=np.stack([new[b].reshape(len(data['lengths']),4,-1) for b in BRANCHES])
            with np.load(OLD/'features'/(inherited.name(row,cond)+'.npz')) as old:
                err=float(np.abs(arrays[BRANCHES.index('final_norm_end')]-old['prefix_final_norm']).max())
                prefix_err=err;method='prefix_only'
                if err>2e-5:
                    del new,arrays;gc.collect()
                    new=forward_layers(model,flat,ns,device,task_suffix=row['tasks'])
                    arrays=np.stack([new[b].reshape(len(data['lengths']),4,-1) for b in BRANCHES])
                    err=float(np.abs(arrays[BRANCHES.index('final_norm_end')]-old['prefix_final_norm']).max())
                    method='exact_legacy_suffix_for_numeric_tolerance'
            assert err<=2e-5, ('Legacy activation mismatch',row['tasks'],cond,err)
            assert inherited.weight_hash(model.state_dict())==ph, 'source parameters changed'
            temp=target.with_suffix('.partial.npy');np.save(temp,arrays);temp.replace(target)
            np.save(target.with_suffix('.embedding.npy'),embedding)
            rec={'model':inherited.name(row,cond),'archive':str(target),'archive_sha256':sha(target),
                 'embedding_sha256':sha(target.with_suffix('.embedding.npy')),'parameter_sha256':ph,
                 'checkpoint_sha256':row['checkpoint_sha256'],'data_sha256':sha(OLD/'evaluation_inputs.npz'),
                 'initialization_grade':'C','original_initial_identity_verified':False,
                 'legacy_final_norm_max_abs':err,'branch_order':BRANCHES,'shape':list(arrays.shape),'source_updates':0,
                 'prefix_only_legacy_max_abs':prefix_err,'extraction_forward':method,
                 'freeze_sha256':sha(ROOT/'freeze.json')}
            meta.write_text(json.dumps(rec,indent=2)+'\n');audit.append(rec)
            del model,new,arrays;gc.collect();torch.cuda.empty_cache()
        print(json.dumps({'extracted_sources':i+1,'total':len(rows),'task':row['tasks'],'seed':row['seed'],'elapsed':round(time.monotonic()-began,1)}),flush=True)
    save('extraction_audit.json',{'sources':len(rows),'conditions':len(audit),'records':audit,
         'seconds':time.monotonic()-began,'new_source_training':0})


def validate():
    verify()
    from neurips_permutations.cka import _layer_modules
    from input_geometry import validate as validate_input
    data=np.load(OLD/'evaluation_inputs.npz');row=source_rows()[0]
    results=[];device='cuda' if torch.cuda.is_available() else 'cpu';torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    for cond in ('initial','trained'):
        model,ph=inherited.make_source(row,cond)
        for n in (10,20,30):
            selected=np.flatnonzero((data['lengths']==n)&(data['split']==1))[:8]
            for ai in (0,2,3):
                inputs=data['input'][selected,ai];lengths=data['lengths'][selected]
                direct=forward_layers(model,inputs,lengths,device,batch=8)
                captured={};handles=[]
                for layer,module in _layer_modules(model):
                    def hook(_m,_i,value,key=layer):captured[key]=value.detach().cpu()
                    handles.append(module.register_forward_hook(hook))
                with torch.inference_mode():
                    ids=torch.tensor(inputs[:,:2*n+4],device=device)
                    model(input_ids=ids,attention_mask=ids.ne(TOKEN_TO_ID['<PAD>']))
                for h in handles:h.remove()
                for layer in LAYERS:
                    end=captured[layer][:,2*n+3].numpy()
                    pooled=captured[layer][:,4:2*n+3:2].mean(1).numpy()
                    for pool,other in [('end',end),('value_mean',pooled)]:
                        err=float(np.abs(direct[layer+'_'+pool]-other).max())
                        assert err<=2e-5;results.append(err)
                # Suffix changes cannot affect any task-preceding landmark.
                task_ids=torch.full((len(ids),2*n+6),TOKEN_TO_ID['<PAD>'],device=device,dtype=torch.long)
                task_ids[:,:len(ids[0])]=ids
                task_ids[:,2*n+4]=TOKEN_TO_ID['<VALLEYS>'];task_ids[:,2*n+5]=TOKEN_TO_ID['=']
                captured={};handle=model.final_norm.register_forward_hook(lambda _m,_i,v:captured.update(final=v.detach().cpu()))
                with torch.inference_mode():model(input_ids=task_ids,attention_mask=task_ids.ne(TOKEN_TO_ID['<PAD>']))
                handle.remove();err=float(np.abs(direct['final_norm_end']-captured['final'][:,2*n+3].numpy()).max())
                assert err<=2e-5;results.append(err)
        assert inherited.weight_hash(model.state_dict())==ph
    inputs=validate_input()
    save('validation.json',{'hook_and_suffix_checks':len(results),'max_abs':max(results),
         'parameters_unchanged':True,'input_geometry_checks':inputs,'threshold':2e-5})
    print('Independent extraction and input metrics validated',flush=True)


def evaluate_pairs(start=0, stop=None):
    verify()
    from input_geometry import relation_input_rows
    torch.set_num_threads(2)
    data=dict(np.load(OLD/'evaluation_inputs.npz'));tasks=list(data['tasks'])
    pairs=js(OLD/'evaluation_pairs.json');lookup={(r['cohort'],r['logical_position']):r for r in source_rows()}
    (ROOT/'pair_tables').mkdir(exist_ok=True);began=time.monotonic()
    for pi,pair in enumerate(pairs):
        if pi<start or (stop is not None and pi>=stop):continue
        target=ROOT/'pair_tables'/f'{pi:03d}.csv'
        if target.exists():
            rec=js(target.with_suffix('.json'))
            assert rec['freeze_sha256']==sha(ROOT/'freeze.json') and rec['sha256']==sha(target)
            continue
        left=lookup[pair['cohort'],pair['left_id']];right=lookup[pair['cohort'],pair['right_id']]
        actions=[0,ACTIONS.index(pair['correct']),ACTIONS.index(pair['wrong'])]
        arrays={cond:(np.load(cache_path(left,cond),mmap_mode='r'),np.load(cache_path(right,cond),mmap_mode='r')) for cond in ('initial','trained')}
        embeddings={cond:(np.load(cache_path(left,cond).with_suffix('.embedding.npy')),np.load(cache_path(right,cond).with_suffix('.embedding.npy'))) for cond in ('initial','trained')}
        yl=data['labels'][:,0,tasks.index(pair['left'])];yr=data['labels'][:,actions,tasks.index(pair['right'])]
        rows=[]
        for n in range(10,31):
            ids=np.flatnonzero(data['lengths']==n);fit=np.flatnonzero(data['split'][ids]==0);test=np.flatnonzero(data['split'][ids]==1)
            features={branch:{cond:{'left':a[bi,ids,0], 'right':b[bi,ids][:,actions]} for cond,(a,b) in arrays.items()}
                      for bi,branch in enumerate(BRANCHES) if branch!='final_norm_end'}
            meta=dict(domain='PermWorld',cohort=pair['cohort'],relation=pair['id'],family=pair['family'],kind=pair['kind'],
                      left_task=pair['left'],right_task=pair['right'],source_unit=f'seed{pair["seed"]}',seed=pair['seed'],
                      world='one_fixed_cohort_world',length=n,correct_transform=pair['correct'],wrong_transform=pair['wrong'],
                      initialization_grade='C',original_initial_weight_identity_verified=False,
                      interpretation='exploratory; constructor rebuild not verified original initialization')
            labels={'left':yl[ids],'right':yr[ids]}
            rows.extend(inherited.comparison_rows(meta,fit,test,features,labels))
            rows.extend(relation_input_rows(meta,fit,test,data['permutations'][ids,:,:n],actions,embeddings,labels))
            for branch in CONSTANTS:
                for control in ('within_length','answer_strata'):
                    for metric in ('linear_cka','debiased_cka','knn_overlap','procrustes'):
                        row0=inherited.decomposed({c:{role:(None,None) for role in ('correct','wrong','identity')} for c in ('initial','trained')})
                        rows.append(dict(**meta,branch=branch,control=control,metric=metric,undefined_reason='structural constant within length',**row0))
        write_csv(str(target.relative_to(ROOT)),rows)
        target.with_suffix('.json').write_text(json.dumps({'freeze_sha256':sha(ROOT/'freeze.json'),
               'sha256':sha(target),'rows':len(rows),'pair':pair['id'],'seed':pair['seed']})+'\n')
        print(json.dumps({'evaluated_pair':pi+1,'total':len(pairs),'relation':pair['id'],'seed':pair['seed'],'rows':len(rows),'elapsed':round(time.monotonic()-began,1)}),flush=True)
    files=sorted((ROOT/'pair_tables').glob('*.csv'))
    if len(files)==len(pairs):
        allrows=[r for path in files for r in csv_read(path)]
        # Preserve all old final-layer/query/output/answer results bit for bit.
        inherited_rows=csv_read(OLD/'permutation_metric_results.csv')
        for row in inherited_rows:
            row['provenance']='reused_frozen_metric_row'
            if row['branch']=='prefix_final_norm':row['branch']='final_norm_end'
        allrows.extend(inherited_rows);write_csv('layer_metric_results.csv',allrows)
        save('evaluation_audit.json',{'pairs':len(pairs),'rows':len(allrows),'reused_rows':len(inherited_rows),
             'source_updates':0,'code_freeze_sha256':sha(ROOT/'freeze.json'),'seconds_this_invocation':time.monotonic()-began})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['freeze','validate','extract','evaluate'])
    parser.add_argument('--limit',type=int);parser.add_argument('--start',type=int,default=0);parser.add_argument('--stop',type=int)
    args=parser.parse_args()
    {'freeze':freeze,'validate':validate,'extract':lambda:extract(args.limit),
     'evaluate':lambda:evaluate_pairs(args.start,args.stop)}[args.stage]()
