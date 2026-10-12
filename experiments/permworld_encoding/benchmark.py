"""Discarded fixed-parameter F/B+AdamW-kernel cost profile; no source training."""
import argparse
import time
import platform
import numpy as np
import torch
from .core import *
from experiments.longrun_attention import accelerate


def run(config_path,output):
    cfg=load_config(config_path);assert cfg['model']['d_model']==256
    if not torch.cuda.is_available():raise RuntimeError('GPU cost profile requires CUDA')
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model=accelerate(make_model(cfg,73061903)).to('cuda').train();before=parameter_hash(model)
    # lr=0 preserves every parameter; moments allocate and update to time actual AdamW kernels.
    opt=torch.optim.AdamW(model.parameters(),lr=0.,weight_decay=.01,fused=True)
    rng=np.random.default_rng(73061904);micro=cfg['training']['micro_batch'];accum=cfg['training']['accumulation']
    warmups=3;timed=8;rows=[]
    torch.cuda.reset_peak_memory_stats()
    for n in cfg['lengths']:
        perms=[tuple(map(int,rng.permutation(n)+1)) for _ in range(micro*accum)]
        answers=np.array([PROPERTY_FUNCTIONS[TASKS[0]](p) for p in perms]);lengths=np.full(micro,n)
        for enc in ENCODINGS:
            prefixes=np.array([encode(p,enc)+tuple([TOKEN_TO_ID['<PAD>']]*(64-len(encode(p,enc)))) for p in perms])
            durations=[]
            for step in range(warmups+timed):
                torch.cuda.synchronize();start=time.perf_counter();opt.zero_grad(set_to_none=True)
                for k in range(accum):
                    sl=slice(k*micro,(k+1)*micro)
                    ids,mask,target,_,_=batch(prefixes[sl],lengths,TASKS[0],answers[sl],device='cuda')
                    with torch.autocast('cuda',dtype=torch.bfloat16):value=loss(model(ids,mask),target)/accum
                    value.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
                torch.cuda.synchronize();dt=time.perf_counter()-start
                if step>=warmups:durations.append(dt)
            rows.append(dict(encoding=enc,length=n,sequence_tokens=2*n+8,timed_updates=timed,
                update_seconds_mean=float(np.mean(durations)),update_seconds_median=float(np.median(durations)),
                update_seconds_p95=float(np.quantile(durations,.95)),durations=durations))
            if n in [10,20,30]:print({'bench_length':n,'encoding':enc,'seconds_per_update':rows[-1]['update_seconds_mean']},flush=True)
    assert parameter_hash(model)==before,'lr=0 benchmark must preserve parameters exactly'
    timings={e:float(np.mean([r['update_seconds_mean'] for r in rows if r['encoding']==e])) for e in ENCODINGS}
    result=dict(status='complete',rows=rows,uniform_length_mean_update_seconds=timings,
        benchmark_seed=73061903,benchmark_data_seed=73061904,parameters=sum(p.numel() for p in model.parameters()),
        gpu=torch.cuda.get_device_name(),torch_version=torch.__version__,python_version=platform.python_version(),
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),micro_batch=micro,accumulation=accum,
        actual_parameter_updates=0,temporary_adam_moment_steps=len(rows)*(warmups+timed),lr_zero=True,
        parameter_hash_preserved=before,formal_training_started=False,
        scope='Warm-start CUDA wall-time including CPU-to-GPU batch construction, forward/backward, clipping and AdamW state kernels. lr=0; no learned model/checkpoint or accuracy outcome.',
        extrapolation_limits=['No long-run thermal or shared-GPU contention guarantee.','Does not directly time full historical-input exclusion, all evaluation CKAs or checkpoint I/O.','No training speed benefit inferred from unchanged-parameter profile.'])
    json_write(output,result);print({'cost_profile_complete':timings,'peak_GB':result['peak_cuda_allocated_bytes']/2**30},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--output',required=True);a=p.parse_args();run(a.config,a.output)
