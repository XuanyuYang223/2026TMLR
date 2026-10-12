"""Small disposable forward/Gram timing profile, not a scientific evaluation."""
import argparse
import time
import numpy as np
import torch
from .core import *
from .evaluate import extract,center,alignment,u_center
from experiments.specialist_input_adjustment import adjust


def run(config_path,output):
    cfg=load_config(config_path);torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    rng=np.random.default_rng(73061910);N=512;n=20
    data={'test_lengths':np.full(N,n,np.int16)}
    for enc in ENCODINGS:
        data[f'test_{enc}_input']=np.full((N,3,64),TOKEN_TO_ID['<PAD>'],np.uint16)
    for i in range(N):
        p=tuple(map(int,rng.permutation(n)+1))
        for a,action in enumerate(ACTIONS):
            q=transform(p,action)
            for enc in ENCODINGS:
                x=encode(q,enc);data[f'test_{enc}_input'][i,a,:len(x)]=x
    times={}
    for enc in ENCODINGS:
        model=make_model(cfg,73061911)
        extract(model,{k:v[:128] for k,v in data.items()},enc,TASKS[0],'cuda')
        torch.cuda.synchronize();start=time.perf_counter();arrays=extract(model,data,enc,TASKS[0],'cuda');torch.cuda.synchronize()
        times[enc]=(time.perf_counter()-start)/(N*3)
    x=center(rng.normal(size=(N,256)));k=x@x.T;z=center(rng.normal(size=(N,400)));g=z@z.T
    # Fixed CPU eigendecomposition kernel dominates the declared input-adjusted sensitivity.
    durations=[]
    for _ in range(8):
        start=time.perf_counter();adjust(k,g);durations.append(time.perf_counter()-start)
    json_write(output,dict(status='complete',gpu_forward_seconds_per_sequence=times,profile_sequences_per_encoding=N*3,
        cpu_psd_adjust_seconds_mean=float(np.mean(durations)),samples=N,feature_dimension=256,
        formal_training_started=False,no_model_accuracy_or_cka_outcome_recorded=True,
        caveat='Inference timed at n20; CPU PSD uses dense synthetic 512-row kernels. Actual common-stratum sizes and GPU frequency may differ.'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--output',required=True);a=p.parse_args();run(a.config,a.output)
