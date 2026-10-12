"""Explicit source fitting and checks; no implicit train cascade."""
import copy
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

from .common import *
from experiments.permworld_encoding.core import fingerprint, orbit, input_key
from experiments.permworld_encoding.data import contains, generate_world


def history_signature(config):
    payload=dict(historical_exclusions=config['historical_exclusions'],
        additional_archive_parser='all-three-development-permutation-splits/full-eight-orbits/v2')
    return hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def prior_development_inputs(path):
    """Read anchors explicitly, rather than silently skip prefixed NPZ arrays."""
    with np.load(path,allow_pickle=False) as archive:
        for split in ['train','validation','test']:
            pk=split+'_permutations';nk=split+'_lengths'
            assert pk in archive.files and nk in archive.files, f'Missing prior development arrays: {pk}/{nk}'
            permutations=archive[pk];lengths=archive[nk]
            assert len(permutations)==len(lengths)
            for row,n in zip(permutations,lengths):
                n=int(n);p=tuple(map(int,row[:n]))
                assert 1<=n<=30 and sorted(p)==list(range(1,n+1))
                yield split,p


def audit_development_exclusion(config,index):
    original=json.loads((WORKSPACE/'results/permworld_encoding_preflight_20261009/protocol.json').read_text())['historical_exclusions']
    anchors=0;members=0;canonical=0;batch_hashes=[];split_counts={}
    for entry in config['historical_exclusions']['archives'][len(original['archives']):]:
        assert sha(entry['path'])==entry['sha256']
        for split,p in prior_development_inputs(entry['path']):
            qs=orbit(p);assert len(qs)==8
            anchors+=1;members+=len(qs);split_counts[split]=split_counts.get(split,0)+1
            batch_hashes.extend(fingerprint(q) for q in qs)
            # Explicitly include the representative for every prior orbit.
            assert contains(index,[fingerprint(min(qs,key=input_key))]).all()
            canonical+=1
            if len(batch_hashes)>=8192:
                assert contains(index,batch_hashes).all();batch_hashes=[]
    if batch_hashes:assert contains(index,batch_hashes).all()
    assert anchors==102144 and members==817152, 'Unexpected fixed prior development pool'
    return dict(prior_development_anchors=anchors,prior_development_orbit_members=members,
        all_prior_orbit_representatives_present=canonical,split_counts=split_counts,
        all_prior_development_orbit_members_present=True)


def prepare():
    config = verify()
    target = Path(config['output']); target.mkdir(parents=True, exist_ok=True)
    history = target/'historical_input_hashes.npy'; meta_path = history.with_suffix('.json')
    if not history.exists():
        previous = WORKSPACE/'results/permworld_encoding_preflight_20261009/development/historical_input_hashes.npy'
        old_meta = json.loads(previous.with_suffix('.json').read_text())
        old_protocol = json.loads((WORKSPACE/'results/permworld_encoding_preflight_20261009/protocol.json').read_text())
        assert sha(previous) == old_meta['sha256']
        original = old_protocol['historical_exclusions']
        assert config['historical_exclusions']['parent_manifest'] == original['parent_manifest']
        assert config['historical_exclusions']['parent_manifest_sha256'] == original['parent_manifest_sha256']
        assert config['historical_exclusions']['archives'][:len(original['archives'])] == original['archives']
        extras = [];split_counts={};anchors=0
        for entry in config['historical_exclusions']['archives'][len(original['archives']):]:
            assert sha(entry['path']) == entry['sha256']
            for split,p in prior_development_inputs(entry['path']):
                extras.extend(fingerprint(q) for q in orbit(p))
                anchors+=1;split_counts[split]=split_counts.get(split,0)+1
        assert anchors==102144 and len(extras)==817152, 'Prior development exclusion was not fully parsed'
        unique = np.unique(np.concatenate([np.load(previous), np.asarray(extras, dtype='V16')]))
        np.save(history, unique)
        signature = history_signature(config)
        json_write(meta_path, dict(signature=signature, sha256=sha(history), unique_inputs=len(unique),
            previous_index_sha256=old_meta['sha256'], appended_prior_development_orbit_members=len(extras),
            appended_prior_development_anchors=anchors,prior_development_split_counts=split_counts,
            archive_parser='all-three-development-permutation-splits/full-eight-orbits/v2',
            policy='Reuse SHA-verified historical index, append all prior development input orbits; no historical inputs admitted.'))
    meta = json.loads(meta_path.read_text()); assert sha(history) == meta['sha256']
    assert meta['signature']==history_signature(config), 'Historical exclusion signature differs; refuse stale index reuse'
    assert meta['appended_prior_development_orbit_members']==817152
    index = np.load(history, mmap_mode='r')
    exclusion_audit=audit_development_exclusion(config,index)
    generate_world(config, config['units'][0], index, meta, world(config))
    audit = json.loads((world(config)/'data_audit.json').read_text())
    assert sha(world(config)/'data.npz') == audit['data_sha256']
    assert sha(world(config)/'batch_indices.npy') == audit['batch_indices_sha256']
    assert audit['historical_index']['signature']==meta['signature'] and audit['historical_index']['sha256']==meta['sha256']
    json_write(ROOT/'data_ready.json', dict(status='passed', data_sha256=audit['data_sha256'],
        history_signature=meta['signature'],history_sha256=meta['sha256'],**exclusion_audit,
        within_world_orbit_disjoint=True, shared_underlying_objects_and_batches=True,
        development_only=True, formal_training_started=False))


def load_data(config):
    path = world(config)
    audit = json.loads((path/'data_audit.json').read_text())
    assert sha(path/'data.npz') == audit['data_sha256']
    assert sha(path/'batch_indices.npy') == audit['batch_indices_sha256']
    history=Path(config['output'])/'historical_input_hashes.npy'
    historical_metadata=json.loads(history.with_suffix('.json').read_text())
    assert sha(history)==historical_metadata['sha256']
    assert historical_metadata['signature']==history_signature(config)
    assert historical_metadata['sha256']==audit['historical_index']['sha256']
    assert historical_metadata['signature']==audit['historical_index']['signature']
    assert historical_metadata['appended_prior_development_orbit_members']==817152
    return dict(np.load(path/'data.npz')), np.load(path/'batch_indices.npy', mmap_mode='r'), audit


def source_version():
    hashes=json.loads((ROOT/'freeze_manifest.json').read_text())['source_sha256']
    return hashlib.sha256(json.dumps(hashes,sort_keys=True,separators=(',',':')).encode()).hexdigest()


@torch.inference_mode()
def accuracy(net, data, encoding, task, device, split='validation', selected=None):
    net.eval(); total=0; answer=0; numeric=0; eos=0
    indexes = np.arange(len(data[split+'_lengths'])) if selected is None else np.asarray(selected)
    for start in range(0, len(indexes), 128):
        use=indexes[start:start+128]; ns=data[split+'_lengths'][use]
        y=data[split+'_labels'][use,0,TASKS.index(task)].astype('int64')
        ids,mask,target,_,eq=batch(data[f'{split}_{encoding}_input'][use,0],ns,task,y,device)
        z=net(ids,mask);rows=torch.arange(len(use),device=device);truth=torch.as_tensor(y,device=device)
        answer+=int((z[rows,eq].argmax(-1)==truth).sum())
        numeric+=int((z[rows,eq,:31].argmax(-1)==truth).sum())
        eos+=int((z[rows,eq+1].argmax(-1)==TOKEN_TO_ID['<EOS>']).sum());total+=len(use)
    return dict(answer_accuracy=answer/total, numeric_accuracy=numeric/total, eos_accuracy=eos/total, samples=total)


def run_job(run_id, stop_at=None, output=None):
    config=verify(); job=next(j for j in config['training_manifest'] if j['run_id']==run_id)
    version=source_version()
    out=ROOT/'models'/run_id if output is None else Path(output)
    assert out.resolve().is_relative_to(ROOT.resolve()), 'Development output must stay in its own root'
    out.mkdir(parents=True,exist_ok=True); completed=out/'completed.json'; checkpoint=out/'checkpoint.pt'
    maximum=config['training']['updates'] if stop_at is None else int(stop_at)
    assert 0 < maximum <= config['training']['updates']
    if completed.exists() and stop_at is None:
        rec=json.loads(completed.read_text());assert rec['protocol_sha256']==sha(CONFIG) and rec['checkpoint_sha256']==sha(checkpoint)
        assert rec['execution_source_version_sha256']==version, 'Completed model source version differs'
        print({'already_complete':run_id},flush=True);return
    data,schedule,audit=load_data(config); saved,meta=initial(config,job['architecture']);device=device_setup()
    net=model(config,job['architecture'],job['model_seed'],device);net.load_state_dict(saved['model'])
    assert parameter_hash(net)==meta['parameter_sha256']
    settings=config['training']; optimizer=torch.optim.AdamW(net.parameters(),lr=settings['learning_rate'],
        weight_decay=settings['weight_decay'],betas=tuple(settings['betas']),eps=settings['epsilon'],fused=device.type=='cuda')
    stochastic_seed=config['units'][0]['dropout_seed_base']+1000*ARCHITECTURES.index(job['architecture'])+DEV_TASKS.index(job['task'])
    torch.manual_seed(stochastic_seed);np.random.seed(stochastic_seed);random.seed(stochastic_seed)
    if device.type=='cuda':torch.cuda.manual_seed_all(stochastic_seed)
    starting_rng=rng_state();rngfile=ROOT/'initializations'/job['architecture']/f'start_rng_{job["task"]}.pt'
    if rngfile.exists():
        prior=torch.load(rngfile,map_location='cpu',weights_only=False)
        assert prior['rng_hash']==rng_hash(starting_rng) and prior['parameter_sha256']==meta['parameter_sha256']
    else:
        atomic_checkpoint(rngfile,dict(rng=starting_rng,rng_hash=rng_hash(starting_rng),parameter_sha256=meta['parameter_sha256']))
    step=0;curve=[];elapsed=0.; gradient_updates=0;min_norm=math.inf;max_norm=0.
    if checkpoint.exists():
        cp=torch.load(checkpoint,map_location='cpu',weights_only=False)
        assert cp['run_id']==run_id and cp['protocol_sha256']==sha(CONFIG)
        assert cp['execution_source_version_sha256']==version, 'Cannot resume across source versions'
        assert cp['data_sha256']==audit['data_sha256'] and cp['initial_parameter_sha256']==meta['parameter_sha256']
        net.load_state_dict(cp['model']);optimizer.load_state_dict(cp['optimizer']);restore_rng(cp['rng'])
        step=cp['step'];curve=cp['curve'];elapsed=cp['elapsed_seconds'];gradient_updates=cp['finite_gradient_updates']
        min_norm=cp['minimum_gradient_norm'];max_norm=cp['maximum_gradient_norm']
    assert step<=maximum, 'Existing checkpoint is beyond requested stage endpoint'
    invocation_start=step
    started=time.monotonic()
    for index in range(step,maximum):
        net.train();optimizer.zero_grad(set_to_none=True); step=index+1
        if step<=settings['warmup_updates']:factor=step/settings['warmup_updates']
        else:factor=settings['minimum_lr_ratio']+(1-settings['minimum_lr_ratio'])*.5*(1+math.cos(math.pi*(step-settings['warmup_updates'])/(settings['updates']-settings['warmup_updates'])))
        for group in optimizer.param_groups:group['lr']=settings['learning_rate']*factor
        value_total=0.
        for k in range(settings['accumulation']):
            chosen=schedule[index,k*settings['micro_batch']:(k+1)*settings['micro_batch']]
            ids,mask,target,_,_=batch(data[f'train_{job["encoding"]}_input'][chosen,0],data['train_lengths'][chosen],
                job['task'],data['train_labels'][chosen,0,TASKS.index(job['task'])],device)
            with torch.autocast(device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'):
                value=loss(net(ids,mask),target)/settings['accumulation']
            assert torch.isfinite(value);value.backward();value_total+=float(value.detach())
        norm=float(torch.nn.utils.clip_grad_norm_(net.parameters(),settings['gradient_clip'],error_if_nonfinite=True))
        gradient_updates+=1;min_norm=min(min_norm,norm);max_norm=max(max_norm,norm);optimizer.step()
        if step%settings['validation_every']==0 or step==maximum:
            record=dict(step=step,train_loss=value_total,**accuracy(net,data,job['encoding'],job['task'],device))
            curve.append(record);print({'run':run_id,**record},flush=True)
        if step%settings['checkpoint_every']==0 or step==maximum:
            atomic_checkpoint(checkpoint,dict(model=net.state_dict(),optimizer=optimizer.state_dict(),rng=rng_state(),
                step=step,curve=curve,run_id=run_id,architecture=job['architecture'],protocol_sha256=sha(CONFIG),
                execution_source_version_sha256=version,invocation_start_step=invocation_start,
                invocation_updates=step-invocation_start,
                data_sha256=audit['data_sha256'],initial_parameter_sha256=meta['parameter_sha256'],
                initial_checkpoint_sha256=meta['checkpoint_sha256'],starting_rng_hash=rng_hash(starting_rng),
                elapsed_seconds=elapsed+time.monotonic()-started,finite_gradient_updates=gradient_updates,
                minimum_gradient_norm=min_norm,maximum_gradient_norm=max_norm))
    if maximum==settings['updates']:
        assert parameter_hash(net)!=meta['parameter_sha256']
        json_write(completed,dict(**job,step=maximum,checkpoint_sha256=sha(checkpoint),protocol_sha256=sha(CONFIG),
            execution_source_version_sha256=version,
            data_sha256=audit['data_sha256'],initial_parameter_sha256=meta['parameter_sha256'],
            actual_initial_identity_verified=True,finite_gradient_updates=gradient_updates,
            elapsed_seconds=elapsed+time.monotonic()-started,curve=curve,parameters=sum(p.numel() for p in net.parameters()),
            exposures=maximum*settings['effective_batch'],supervised_tokens=2*maximum*settings['effective_batch']))


def overfit():
    config=verify();data,_,_=load_data(config);device=device_setup();records=[]
    for architecture in ARCHITECTURES:
        for task in DEV_TASKS:
            candidates=np.flatnonzero(data['train_lengths']==10);y=data['train_labels'][candidates,0,TASKS.index(task)]
            counts=np.bincount(y,minlength=31);classes=sorted(np.argsort(-counts,kind='stable')[:4].tolist())
            assert min(counts[classes])>=12
            selected=np.concatenate([candidates[y==c][:12] for c in classes])
            for encoding in ENCODINGS:
                out=ROOT/'overfit'/f'{architecture}_{encoding}_{task}';out.mkdir(parents=True,exist_ok=True)
                if (out/'result.json').exists():
                    record=json.loads((out/'result.json').read_text());assert record['protocol_sha256']==sha(CONFIG)
                    records.append(record);continue
                net=model(config,architecture,95500001+ARCHITECTURES.index(architecture),device)
                before=parameter_hash(net);optimizer=torch.optim.AdamW(net.parameters(),lr=.0003,weight_decay=.01,fused=device.type=='cuda')
                torch.manual_seed(95600000+DEV_TASKS.index(task))
                if device.type=='cuda':torch.cuda.manual_seed_all(95600000+DEV_TASKS.index(task))
                minimum=math.inf;maximum=0.;curve=[]
                # Independent loss/mask and future answer causal checks.
                ns=data['train_lengths'][selected];answers=data['train_labels'][selected,0,TASKS.index(task)]
                ids,mask,target,_,eq=batch(data[f'train_{encoding}_input'][selected,0],ns,task,answers,device)
                assert torch.all((target!=-100).sum(1)==2)
                rows=torch.arange(48,device=device);gold=torch.as_tensor(answers.astype('int64'),device=device)
                toy=torch.randn(48,ids.shape[1],len(TOKEN_TO_ID),device=device,requires_grad=True)
                explicit=.5*(F.cross_entropy(toy[rows,eq],gold)+F.cross_entropy(toy[rows,eq+1],torch.full_like(gold,TOKEN_TO_ID['<EOS>'])))
                assert abs(float((loss(toy,target)-explicit).detach()))<1e-6
                for step in range(1,config['engineering']['overfit_updates']+1):
                    net.train();optimizer.zero_grad(set_to_none=True)
                    for k in range(3):
                        use=selected[k*16:(k+1)*16]
                        x,m,t,_,_=batch(data[f'train_{encoding}_input'][use,0],data['train_lengths'][use],task,
                            data['train_labels'][use,0,TASKS.index(task)],device)
                        with torch.autocast(device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'):value=loss(net(x,m),t)/3
                        assert torch.isfinite(value);value.backward()
                    norm=float(torch.nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True))
                    minimum=min(minimum,norm);maximum=max(maximum,norm);optimizer.step()
                    if step%100==0:
                        rec=dict(step=step,**accuracy(net,data,encoding,task,device,'train',selected));curve.append(rec)
                        print({'overfit':out.name,**rec},flush=True)
                final=curve[-1];assert before!=parameter_hash(net) and maximum>0
                net.eval()
                with torch.inference_mode():
                    full=net(ids,mask)[rows,eq]
                    x,m,_,_,e=batch(data[f'train_{encoding}_input'][selected,0],ns,task,device=device)
                    prompt=net(x,m)[rows,e];causal=float((full-prompt).abs().max())
                assert causal<2e-5
                passed=final['answer_accuracy']>=config['engineering']['overfit_answer_accuracy_min'] and final['eos_accuracy']>=config['engineering']['overfit_eos_accuracy_min']
                record=dict(architecture=architecture,encoding=encoding,task=task,protocol_sha256=sha(CONFIG),
                    samples=48,classes=classes,parameter_changed=True,finite_gradient_updates=1000,
                    minimum_gradient_norm=minimum,maximum_gradient_norm=maximum,future_answer_logit_difference=causal,
                    loss_mask_independently_checked=True,final=final,passed=passed,curve=curve)
                json_write(out/'result.json',record);records.append(record);del net,optimizer
                if device.type=='cuda':torch.cuda.empty_cache()
    status='passed' if all(r['passed'] for r in records) else 'failed'
    json_write(ROOT/'overfit_summary.json',dict(status=status,records=records,geometry_not_used_for_gate=True))
    assert status=='passed', 'Overfit gate failed; no complete training launched'


def child(run_id,stop,output):
    subprocess.run([sys.executable,'-m','experiments.permworld_encoding_archdev_20261009','--stage','train-one',
        '--run-id',run_id,'--stop-step',str(stop),'--output',str(output)],cwd=WORKSPACE,check=True)


def difference(a,b):
    if torch.is_tensor(a):
        assert a.shape==b.shape and a.dtype==b.dtype
        return float((a.double()-b.double()).abs().max()) if a.numel() else 0.
    if isinstance(a,dict):
        assert a.keys()==b.keys();return max([difference(a[k],b[k]) for k in a]+[0.])
    if isinstance(a,(list,tuple)):
        assert len(a)==len(b);return max([difference(x,y) for x,y in zip(a,b)]+[0.])
    assert a==b;return 0.


def resume():
    config=verify();assert json.loads((ROOT/'overfit_summary.json').read_text())['status']=='passed'
    version=source_version()
    records=[]
    for architecture in ARCHITECTURES:
        for encoding in ENCODINGS:
            job=next(j for j in config['training_manifest'] if j['architecture']==architecture and j['encoding']==encoding and j['task']=='descents')
            directory=ROOT/'resume'/f'{architecture}_{encoding}';directory.mkdir(parents=True,exist_ok=True)
            result=directory/'result.json'
            if result.exists():
                rec=json.loads(result.read_text());assert rec['protocol_sha256']==sha(CONFIG)
                assert rec['execution_source_version_sha256']==version
                records.append(rec);continue
            # A completed reference must contain an uninterrupted single-process
            # 0->2000 run. An interrupted attempt is retained but never resumed
            # as a continuous reference; a fresh attempt gets a new directory.
            receipt=directory/'continuous_reference_complete.json'
            if receipt.exists():
                reference=json.loads(receipt.read_text());continuous=Path(reference['directory'])
                assert reference['execution_source_version_sha256']==version
                assert sha(continuous/'checkpoint.pt')==reference['checkpoint_sha256']
            else:
                attempts=directory/'continuous_attempts';attempts.mkdir(exist_ok=True)
                number=len(list(attempts.glob('attempt*')))+1
                continuous=attempts/f'attempt{number:03d}'
                assert not continuous.exists()
                child(job['run_id'],2000,continuous)
                cp=torch.load(continuous/'checkpoint.pt',map_location='cpu',weights_only=False)
                assert cp['step']==2000 and cp['invocation_start_step']==0 and cp['invocation_updates']==2000
                json_write(receipt,dict(directory=str(continuous),checkpoint_sha256=sha(continuous/'checkpoint.pt'),
                    execution_source_version_sha256=version,uninterrupted_single_process_from_step0=True))
            replayed=directory/'split';restart=directory/'restart1000.pt'
            if not restart.exists():
                child(job['run_id'],1000,replayed)
                cp=torch.load(replayed/'checkpoint.pt',map_location='cpu',weights_only=False)
                assert cp['step']==1000
                shutil.copyfile(replayed/'checkpoint.pt',restart)
            child(job['run_id'],2000,replayed)
            a=torch.load(continuous/'checkpoint.pt',map_location='cpu',weights_only=False)
            b=torch.load(replayed/'checkpoint.pt',map_location='cpu',weights_only=False)
            assert a['invocation_start_step']==0 and a['invocation_updates']==2000
            assert b['invocation_start_step']==1000 and b['invocation_updates']==1000
            assert a['execution_source_version_sha256']==b['execution_source_version_sha256']==version
            weights=difference(a['model'],b['model']);moments=difference(a['optimizer'],b['optimizer'])
            assert weights<=config['engineering']['parameter_tolerance'] and moments<=config['engineering']['optimizer_tolerance']
            assert rng_hash(a['rng'])==rng_hash(b['rng'])
            data,_,_=load_data(config);device=device_setup();logits=[]
            for cp in [a,b]:
                net=model(config,architecture,job['model_seed'],device).eval();net.load_state_dict(cp['model'])
                ids,mask,_,_,eq=batch(data[f'validation_{encoding}_input'][:64,0],data['validation_lengths'][:64],'descents',device=device)
                with torch.inference_mode():logits.append(net(ids,mask)[torch.arange(64,device=device),eq].cpu())
                del net
            logit=float((logits[0]-logits[1]).abs().max());assert logit<=config['engineering']['logit_tolerance']
            rec=dict(status='passed',architecture=architecture,encoding=encoding,protocol_sha256=sha(CONFIG),
                execution_source_version_sha256=version,uninterrupted_reference_from_step0=True,
                reference_directory=str(continuous),split_directory=str(replayed),
                independent_process_resume=True,from_step=1000,until_step=2000,parameter_max_difference=weights,
                optimizer_max_difference=moments,logit_max_difference=logit,rng_identity=True,
                continuous_sha256=sha(continuous/'checkpoint.pt'),replayed_sha256=sha(replayed/'checkpoint.pt'))
            json_write(result,rec);records.append(rec)
    json_write(ROOT/'resume_summary.json',dict(status='passed',records=records,execution_source_version_sha256=version))


def train_all():
    config=verify()
    for filename in ['cpu_check.json','overfit_summary.json','resume_summary.json']:
        assert json.loads((ROOT/filename).read_text())['status']=='passed', filename
    for job in config['training_manifest']:
        subprocess.run([sys.executable,'-m','experiments.permworld_encoding_archdev_20261009',
            '--stage','train-one','--run-id',job['run_id']],cwd=WORKSPACE,check=True)
    json_write(ROOT/'training_complete.json',dict(models=16,updates_per_model=20000,formal_training_started=False,
        execution_source_version_sha256=source_version(),protocol_sha256=sha(CONFIG),
        development_source_training_started=True))
