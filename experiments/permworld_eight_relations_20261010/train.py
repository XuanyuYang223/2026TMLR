"""Only explicitly listed missing sources; exact20k original recipe and RNG."""
import argparse
import copy
import time
from torch.nn import functional as F
from .common import *

@torch.inference_mode()
def accuracy(net,data,encoding,task,device,split='validation',selected=None):
    net.eval(); selected=np.arange(len(data[split+'_lengths'])) if selected is None else np.asarray(selected)
    correct=eos=numeric=0
    for start in range(0,len(selected),128):
        use=selected[start:start+128];ns=data[split+'_lengths'][use]
        y=data[split+'_labels'][use,0,TASKS.index(task)].astype(np.int64)
        ids,mask,target,_,eq=batch(data[f'{split}_{encoding}_input'][use,0],ns,task,y,device)
        z=net(ids,mask); rr=torch.arange(len(use),device=device);gold=torch.as_tensor(y,device=device)
        correct+=int((z[rr,eq].argmax(-1)==gold).sum())
        numeric+=int((z[rr,eq,:31].argmax(-1)==gold).sum())
        eos+=int((z[rr,eq+1].argmax(-1)==TOKEN_TO_ID['<EOS>']).sum())
    return dict(answer_accuracy=correct/len(selected),numeric_accuracy=numeric/len(selected),eos_accuracy=eos/len(selected),samples=len(selected))

def optimizer(net,s,device):
    return torch.optim.AdamW(net.parameters(),lr=s['learning_rate'],weight_decay=s['weight_decay'],
        betas=tuple(s['betas']),eps=s['epsilon'],fused=device.type=='cuda')

def update(net,opt,data,schedule,job,index,s,device):
    net.train();opt.zero_grad(set_to_none=True);step=index+1
    if step<=s['warmup_updates']: factor=step/s['warmup_updates']
    else: factor=s['minimum_lr_ratio']+(1-s['minimum_lr_ratio'])*.5*(1+math.cos(math.pi*(step-s['warmup_updates'])/(s['updates']-s['warmup_updates'])))
    for group in opt.param_groups: group['lr']=s['learning_rate']*factor
    value_total=0.
    for k in range(s['accumulation']):
        use=schedule[index,k*s['micro_batch']:(k+1)*s['micro_batch']]
        ids,mask,target,_,_=batch(data[f'train_{job["encoding"]}_input'][use,0],data['train_lengths'][use],job['task'],
            data['train_labels'][use,0,TASKS.index(job['task'])],device)
        with torch.autocast(device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'):
            value=loss(net(ids,mask),target)/s['accumulation']
        assert torch.isfinite(value);value.backward();value_total+=float(value.detach())
    norm=float(torch.nn.utils.clip_grad_norm_(net.parameters(),s['gradient_clip'],error_if_nonfinite=True))
    opt.step();return value_total,norm

def make(c,arch,device):
    saved,ref=initial(c,arch)
    net=model(c,arch,c['architecture_model_seeds'][arch],device);net.load_state_dict(saved['model'])
    assert parameter_hash(net)==ref['parameter_sha256'];return net,ref

def run_job(run_id,stop_at=None,output=None):
    c=verify();job=next(j for j in c['training_manifest'] if j['run_id']==run_id)
    out=ROOT/'models'/run_id if output is None else Path(output)
    assert out.resolve().is_relative_to(ROOT.resolve())
    assert output is None or out.resolve().is_relative_to((ROOT/'engineering').resolve())
    out.mkdir(parents=True,exist_ok=True);cpfile=out/'checkpoint.pt';done=out/'completed.json'
    version=source_version();maximum=c['training']['updates'] if stop_at is None else int(stop_at)
    assert 0<maximum<=20000
    if done.exists() and stop_at is None:
        meta=js(done); assert sha(cpfile)==meta['checkpoint_sha256'] and meta['execution_source_version_sha256']==version
        assert meta['protocol_sha256']==sha(CONFIG) and meta['step']==20000
        print(dict(already_complete=run_id),flush=True);return
    data,schedule=load_data(c);device=device_setup();net,ref=make(c,job['architecture'],device)
    s=c['training'];opt=optimizer(net,s,device);restore_rng(start_rng(c,job['architecture']))
    starting=rng_state();assert rng_hash(starting)==c['new_task_start_rng'][job['architecture']]['rng_hash']
    step=0;curve=[];elapsed=0.;finite=0;low=math.inf;high=0.;max_memory=0
    if cpfile.exists():
        cp=torch.load(cpfile,map_location='cpu',weights_only=False)
        assert cp['run_id']==run_id and cp['protocol_sha256']==sha(CONFIG) and cp['execution_source_version_sha256']==version
        assert cp['data_sha256']==c['data_sha256'] and cp['initial_parameter_sha256']==ref['parameter_sha256']
        assert cp['starting_rng_hash']==rng_hash(starting)
        net.load_state_dict(cp['model']);opt.load_state_dict(cp['optimizer']);restore_rng(cp['rng'])
        step=cp['step'];curve=cp['curve'];elapsed=cp['elapsed_seconds'];finite=cp['finite_gradient_updates']
        low=cp['minimum_gradient_norm'];high=cp['maximum_gradient_norm'];max_memory=cp.get('peak_cuda_allocated_bytes',0)
    assert step<=maximum
    startstep=step;started=time.monotonic()
    if device.type=='cuda':torch.cuda.reset_peak_memory_stats()
    for index in range(step,maximum):
        value,norm=update(net,opt,data,schedule,job,index,s,device);step=index+1;finite+=1
        low=min(low,norm);high=max(high,norm)
        if step%s['validation_every']==0 or step==maximum:
            rec=dict(step=step,train_loss=value,**accuracy(net,data,job['encoding'],job['task'],device))
            curve.append(rec);print(dict(run=run_id,elapsed_seconds=elapsed+time.monotonic()-started,**rec),flush=True)
        if step%s['checkpoint_every']==0 or step==maximum:
            if device.type=='cuda': max_memory=max(max_memory,torch.cuda.max_memory_allocated())
            atomic_checkpoint(cpfile,dict(model=net.state_dict(),optimizer=opt.state_dict(),rng=rng_state(),step=step,curve=curve,
                run_id=run_id,architecture=job['architecture'],protocol_sha256=sha(CONFIG),execution_source_version_sha256=version,
                invocation_start_step=startstep,invocation_updates=step-startstep,data_sha256=c['data_sha256'],
                batch_schedule_sha256=c['batch_schedule_sha256'],initial_parameter_sha256=ref['parameter_sha256'],
                initial_checkpoint_sha256=ref['sha256'],starting_rng_hash=rng_hash(starting),
                elapsed_seconds=elapsed+time.monotonic()-started,finite_gradient_updates=finite,
                minimum_gradient_norm=low,maximum_gradient_norm=high,peak_cuda_allocated_bytes=max_memory))
    if maximum==20000:
        assert finite==20000 and parameter_hash(net)!=ref['parameter_sha256']
        json_write(done,dict(**job,step=step,checkpoint_sha256=sha(cpfile),parameter_sha256=parameter_hash(net),
            protocol_sha256=sha(CONFIG),execution_source_version_sha256=version,data_sha256=c['data_sha256'],
            batch_schedule_sha256=c['batch_schedule_sha256'],initial_parameter_sha256=ref['parameter_sha256'],
            initial_checkpoint_sha256=ref['sha256'],starting_rng_hash=rng_hash(starting),actual_initial_identity_verified=True,
            finite_gradient_updates=finite,elapsed_seconds=elapsed+time.monotonic()-started,curve=curve,
            parameters=sum(v.numel() for v in net.parameters()),exposures=20000*s['effective_batch'],
            supervised_tokens=40000*s['effective_batch'],peak_cuda_allocated_bytes=max_memory))
    del net,opt;torch.cuda.empty_cache()

def engineering():
    c=verify();data,schedule=load_data(c);device=device_setup();records=[]
    assert not any((ROOT/'models').glob('*/completed.json')) if (ROOT/'models').exists() else True
    for arch in ARCHITECTURES:
        for enc in ENCODINGS:
            job=next(j for j in c['training_manifest'] if j['architecture']==arch and j['encoding']==enc and j['task']=='fixed_points')
            net,ref=make(c,arch,device);opt=optimizer(net,c['training'],device);restore_rng(start_rng(c,arch))
            before=parameter_hash(net);norms=[];losses=[];started=time.monotonic();middle=None
            for idx in range(40):
                value,norm=update(net,opt,data,schedule,job,idx,c['training'],device);losses.append(value);norms.append(norm)
                if idx==19:middle=dict(model=copy.deepcopy(net.state_dict()),optimizer=copy.deepcopy(opt.state_dict()),rng=rng_state())
            continuous_model=copy.deepcopy(net.state_dict());continuous_optimizer=copy.deepcopy(opt.state_dict());continuous_rng=rng_state()
            assert before!=parameter_hash(net) and min(norms)>=0 and max(norms)>0
            samples=np.flatnonzero(data['train_lengths']==10)[:16]; ns=data['train_lengths'][samples]
            y=data['train_labels'][samples,0,TASKS.index(job['task'])]
            ids,mask,target,end,eq=batch(data[f'train_{enc}_input'][samples,0],ns,job['task'],y,device)
            assert torch.all((target!=-100).sum(1)==2)
            toy=torch.arange(ids.numel()*len(TOKEN_TO_ID),device=device,dtype=torch.float32).reshape(*ids.shape,len(TOKEN_TO_ID))/100000
            rr=torch.arange(len(samples),device=device);gold=torch.as_tensor(y.astype(np.int64),device=device)
            explicit=.5*(F.cross_entropy(toy[rr,eq],gold)+F.cross_entropy(toy[rr,eq+1],torch.full_like(gold,TOKEN_TO_ID['<EOS>'])))
            mask_error=abs(float(loss(toy,target)-explicit));assert mask_error<=1e-6
            net.eval()
            with torch.inference_mode():
                pred=net(ids,mask)[rr,eq];changed=ids.clone();changed[rr,eq+1]=(gold+1)%31
                causal=float((pred-net(changed,mask)[rr,eq]).abs().max());assert causal<=1e-5
            disk=ROOT/'engineering'/f'{arch}_{enc}'/'step20.pt';disk.parent.mkdir(parents=True,exist_ok=True);atomic_checkpoint(disk,middle)
            reloaded=torch.load(disk,map_location='cpu',weights_only=False)
            net2,ref2=make(c,arch,device);opt2=optimizer(net2,c['training'],device)
            net2.load_state_dict(reloaded['model']);opt2.load_state_dict(reloaded['optimizer']);restore_rng(reloaded['rng'])
            for idx in range(20,40): update(net2,opt2,data,schedule,job,idx,c['training'],device)
            err=max(float((continuous_model[k].to(device)-v).abs().max()) for k,v in net2.state_dict().items())
            assert err<=1e-5 and rng_hash(rng_state())==rng_hash(continuous_rng)
            oe=0.
            for k,s in continuous_optimizer['state'].items():
                for key,v in s.items():
                    other=opt2.state_dict()['state'][k][key]
                    if torch.is_tensor(v):oe=max(oe,float((v.to(other.device)-other).abs().max()))
                    else:assert v==other
            assert oe<=1e-5
            record=dict(architecture=arch,encoding=enc,task=job['task'],real_recipe_updates=40,optimizer_resume_from=20,
                elapsed_wall_seconds=time.monotonic()-started,finite_gradient_min=min(norms),finite_gradient_max=max(norms),
                parameter_changed=True,mask_error=mask_error,causal_suffix_error=causal,
                resume_parameter_max_difference=err,resume_optimizer_max_difference=oe,
                same_final_rng=True,peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated() if device.type=='cuda' else None,
                no_accuracy_or_geometry_gate=True)
            records.append(record);print(record,flush=True)
            del net,net2,opt,opt2;torch.cuda.empty_cache()
    json_write(DIAG/'engineering_audit.json',dict(status='passed',records=records,engineering_not_formal_models=True,
        real_starting_RNG=True,no_new_initial_parameters=True,no_learning_threshold=True,no_geometry_loaded=True))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['engineering','job'],required=True);p.add_argument('--run-id')
    a=p.parse_args();engineering() if a.stage=='engineering' else run_job(a.run_id)
