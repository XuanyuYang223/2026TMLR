"""The original20k recipe on fixed graph batches; explicit bounded commands only."""
import math
import subprocess
import sys
import time
from torch.nn import functional as F
from .common import *

@torch.inference_mode()
def accuracy(net,data,task,device,split='validation',selected=None):
    net.eval();indexes=np.arange(len(data[split+'_lengths'])) if selected is None else np.asarray(selected)
    totals={n:dict(samples=0,answer=0,numeric=0,eos=0,answer_ce_sum=0.,combined_ce_sum=0.) for n in (6,7,8)}
    for start in range(0,len(indexes),128):
        use=indexes[start:start+128];ns=data[split+'_lengths'][use];y=data[split+'_labels'][use,0,TASKS.index(task)]
        ids,mask,target,_,eq=batch(data[split+'_input'][use,0],ns,task,y,device)
        z=net(ids,mask);rows=torch.arange(len(use),device=device);gold=torch.as_tensor(y.astype(np.int64),device=device)
        az=z[rows,eq];ez=z[rows,eq+1]
        av=(az.argmax(-1)==gold).cpu().numpy();nv=(az[:,:31].argmax(-1)==gold).cpu().numpy()
        ev=(ez.argmax(-1)==TOKEN_TO_ID['<EOS>']).cpu().numpy()
        ce=F.cross_entropy(az,gold,reduction='none').cpu().double().numpy()
        ec=F.cross_entropy(ez,torch.full_like(gold,TOKEN_TO_ID['<EOS>']),reduction='none').cpu().double().numpy()
        for n in np.unique(ns):
            k=ns==n;r=totals[int(n)];r['samples']+=int(k.sum());r['answer']+=int(av[k].sum());r['numeric']+=int(nv[k].sum());r['eos']+=int(ev[k].sum())
            r['answer_ce_sum']+=float(ce[k].sum());r['combined_ce_sum']+=float(((ce[k]+ec[k])/2).sum())
    per=[]
    for n,r in totals.items():
        if r['samples']:
            per.append(dict(n=n,samples=r['samples'],answer_accuracy=r['answer']/r['samples'],numeric_accuracy=r['numeric']/r['samples'],
                       eos_accuracy=r['eos']/r['samples'],answer_ce=r['answer_ce_sum']/r['samples'],combined_ce=r['combined_ce_sum']/r['samples']))
    alln=sum(r['samples'] for r in totals.values())
    return dict(samples=alln,answer_accuracy=sum(r['answer'] for r in totals.values())/alln,
        numeric_accuracy=sum(r['numeric'] for r in totals.values())/alln,eos_accuracy=sum(r['eos'] for r in totals.values())/alln,
        answer_ce=sum(r['answer_ce_sum'] for r in totals.values())/alln,combined_ce=sum(r['combined_ce_sum'] for r in totals.values())/alln,per_n=per)

def run_job(run_id,stop_at=20000,output=None):
    c=verify();job=next(j for j in c['training_manifest'] if j['run_id']==run_id)
    assert 0<stop_at<=20000
    if output is None:
        assert read(DIAG/'gpu_engineering.json')['status']=='passed','Run explicit engineering stage first'
        directory=ROOT/'models'/run_id
    else:
        directory=Path(output).resolve();assert directory.is_relative_to((ROOT/'engineering').resolve())
    directory.mkdir(parents=True,exist_ok=True);checkpoint=directory/'checkpoint.pt';done=directory/'completed.json'
    version=source_version();pd=sha(CONFIG);data,schedule,audit=load_data(c)
    if done.exists():
        r=read(done);assert r['step']==20000 and r['protocol_sha256']==pd and r['source_version_sha256']==version
        assert sha(checkpoint)==r['checkpoint_sha256'] and r['data_sha256']==audit['data_sha256'];return r
    device=device_setup();assert device.type=='cuda','No implicit CPU source training; root must schedule CUDA explicitly'
    saved=initial(c);start_rng=starting_rng(c)
    net=model(c,'transformer',c['model_seed'],device);net.load_state_dict(saved['model'],strict=True)
    assert parameter_hash(net)==c['initial_parameter_sha256']
    s=c['training'];opt=torch.optim.AdamW(net.parameters(),lr=s['learning_rate'],weight_decay=s['weight_decay'],
        betas=tuple(s['betas']),eps=s['epsilon'],fused=device.type=='cuda')
    restore_rng(start_rng);assert rng_hash(rng_state())==c['starting_rng_hash']
    step=0;curve=[];elapsed=0.;minimum=math.inf;maximum=0.;finite=0
    if checkpoint.exists():
        cp=torch.load(checkpoint,map_location='cpu',weights_only=False)
        for k,v in dict(run_id=run_id,protocol_sha256=pd,source_version_sha256=version,data_sha256=audit['data_sha256'],
                       initial_parameter_sha256=c['initial_parameter_sha256'],starting_rng_hash=c['starting_rng_hash']).items():assert cp[k]==v,(k,cp[k],v)
        net.load_state_dict(cp['model'],strict=True);opt.load_state_dict(cp['optimizer']);restore_rng(cp['rng'])
        step=cp['step'];curve=cp['curve'];elapsed=cp['elapsed_seconds'];minimum=cp['minimum_gradient_norm'];maximum=cp['maximum_gradient_norm'];finite=cp['finite_gradient_updates']
    assert step<=stop_at;invocation_start=step
    if device.type=='cuda':torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
    begun=time.monotonic()
    for index in range(step,stop_at):
        step=index+1;net.train();opt.zero_grad(set_to_none=True)
        factor=step/s['warmup_updates'] if step<=s['warmup_updates'] else s['minimum_lr_ratio']+(1-s['minimum_lr_ratio'])*.5*(1+math.cos(math.pi*(step-s['warmup_updates'])/(20000-s['warmup_updates'])))
        for group in opt.param_groups:group['lr']=s['learning_rate']*factor
        value_total=0.
        for k in range(s['accumulation']):
            chosen=schedule[index,k*s['micro_batch']:(k+1)*s['micro_batch']]
            ids,mask,target,_,_=batch(data['train_input'][chosen,0],data['train_lengths'][chosen],job['task'],
                                    data['train_labels'][chosen,0,TASKS.index(job['task'])],device)
            with torch.autocast(device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'):
                value=loss(net(ids,mask),target)/s['accumulation']
            assert torch.isfinite(value);value.backward();value_total+=float(value.detach())
        norm=float(torch.nn.utils.clip_grad_norm_(net.parameters(),s['gradient_clip'],error_if_nonfinite=True))
        assert math.isfinite(norm);finite+=1;minimum=min(minimum,norm);maximum=max(maximum,norm);opt.step()
        if step%s['validation_every']==0 or step==stop_at:
            validation=accuracy(net,data,job['task'],device)
            curve.append(dict(step=step,train_loss=value_total,validation=validation))
            json_write(directory/f'stage_{step}.json',dict(step=step,run_id=run_id,protocol_sha256=pd,source_version_sha256=version,
                       data_sha256=audit['data_sha256'],validation=validation,fixed_endpoint=step==20000))
            print({'run':run_id,'step':step,'loss':value_total,'answer_accuracy':validation['answer_accuracy']},flush=True)
        if step%s['checkpoint_every']==0 or step==stop_at:
            if device.type=='cuda':torch.cuda.synchronize()
            atomic_checkpoint(checkpoint,dict(model=net.state_dict(),optimizer=opt.state_dict(),rng=rng_state(),step=step,curve=curve,
                run_id=run_id,protocol_sha256=pd,source_version_sha256=version,data_sha256=audit['data_sha256'],
                initial_parameter_sha256=c['initial_parameter_sha256'],initial_checkpoint_sha256=c['initial_checkpoint_sha256'],
                starting_rng_hash=c['starting_rng_hash'],elapsed_seconds=elapsed+time.monotonic()-begun,
                invocation_start_step=invocation_start,invocation_updates=step-invocation_start,finite_gradient_updates=finite,
                minimum_gradient_norm=minimum,maximum_gradient_norm=maximum))
    if device.type=='cuda':torch.cuda.synchronize()
    receipt=dict(**job,step=step,protocol_sha256=pd,source_version_sha256=version,data_sha256=audit['data_sha256'],
        checkpoint_sha256=sha(checkpoint),initial_parameter_sha256=c['initial_parameter_sha256'],actual_initial_identity_verified=True,
        initial_checkpoint_sha256=c['initial_checkpoint_sha256'],starting_rng_hash=c['starting_rng_hash'],
        finite_gradient_updates=finite,minimum_gradient_norm=minimum,maximum_gradient_norm=maximum,
        final_parameter_sha256=parameter_hash(net),elapsed_seconds=elapsed+time.monotonic()-begun,
        invocation_start_step=invocation_start,invocation_updates=step-invocation_start,curve=curve,
        peak_cuda_memory_bytes=torch.cuda.max_memory_allocated() if device.type=='cuda' else None,
        exposures=step*64,supervised_tokens=step*128,parameters=sum(p.numel() for p in net.parameters()))
    assert maximum>0 and receipt['final_parameter_sha256']!=c['initial_parameter_sha256']
    if output is None and step==20000:json_write(done,receipt)
    else:json_write(directory/'partial.json',receipt)
    return receipt

def difference(a,b):
    if torch.is_tensor(a):
        assert a.shape==b.shape and a.dtype==b.dtype
        return float((a.double()-b.double()).abs().max()) if a.numel() else 0.
    if isinstance(a,dict):assert a.keys()==b.keys();return max([difference(a[k],b[k]) for k in a]+[0.])
    if isinstance(a,(list,tuple)):assert len(a)==len(b);return max([difference(x,y) for x,y in zip(a,b)]+[0.])
    assert a==b;return 0.

def engineering():
    """Explicit small GPU checks, distinct paths, no formal endpoint produced."""
    c=verify();assert read(DIAG/'cpu_ready.json')['status']=='passed'
    assert not (DIAG/'gpu_engineering.json').exists(),'Existing engineering retained; inspect before rerunning'
    records=[];begun=time.monotonic()
    for job in c['training_manifest']:
        out=ROOT/'engineering'/'gradient'/job['task']
        r=run_job(job['run_id'],32,out);assert r['finite_gradient_updates']==32 and r['maximum_gradient_norm']>0
        records.append(dict(task=job['task'],parameter_changed=True,finite_gradient_updates=32,validation=r['curve'][-1]['validation']))
    job=c['training_manifest'][0];base=ROOT/'engineering'/'resume'
    def child(stop,path):
        subprocess.run([sys.executable,'-m','experiments.graphworld_pilot_20261010','--stage','train-one','--run-id',job['run_id'],
                        '--stop-step',str(stop),'--output',str(path)],cwd=WORKSPACE,check=True)
    child(32,base/'continuous');child(16,base/'split');child(32,base/'split')
    a=torch.load(base/'continuous/checkpoint.pt',map_location='cpu',weights_only=False)
    b=torch.load(base/'split/checkpoint.pt',map_location='cpu',weights_only=False)
    pd=difference(a['model'],b['model']);od=difference(a['optimizer'],b['optimizer']);same=rng_hash(a['rng'])==rng_hash(b['rng'])
    assert a['invocation_start_step']==0 and a['invocation_updates']==32 and a['step']==b['step']==32
    assert pd<=c['engineering']['parameter_tolerance'] and od<=c['engineering']['optimizer_tolerance'] and same
    json_write(DIAG/'gpu_engineering.json',dict(status='passed',protocol_sha256=sha(CONFIG),source_version_sha256=source_version(),
        source_endpoints_produced=0,records=records,resume_parameter_difference=pd,resume_optimizer_difference=od,resume_rng_identical=same,
        wall_seconds=time.monotonic()-begun,accuracy_not_a_gate=True,geometry_not_evaluated=True))

def train_all():
    c=verify();assert read(DIAG/'gpu_engineering.json')['status']=='passed'
    records=[run_job(j['run_id'],20000) for j in c['training_manifest']]
    assert len(records)==4 and all(r['step']==20000 for r in records)
    json_write(ROOT/'training_complete.json',dict(status='completed',protocol_sha256=sha(CONFIG),source_version_sha256=source_version(),
        source_models=4,worlds=1,records=records,no_added_models=True))
