"""Small CPU fixtures; no source learning and no GPU forward."""
import copy
import numpy as np
import torch
from torch.nn import functional as F
from .common import *
from .training import difference


def cpu_check():
    config=verify();device=device_setup(cpu=True);records=[]
    examples=[(2,3,1,5,4,6),(6,5,4,3,2,1),(1,3,5,2,6,4),(3,1,6,2,5,4)]
    for architecture in ARCHITECTURES:
        seed=config['architecture_model_seeds'][architecture]
        net=model(config,architecture,seed,device).eval()
        initial_hash=parameter_hash(net);source,metadata=initial(config,architecture)
        assert initial_hash==metadata['parameter_sha256']
        paired=model(config,architecture,seed+999,device);paired.load_state_dict(source['model'])
        assert parameter_hash(paired)==initial_hash
        prefix_error=0.;future_error=0.;mask_error=0.
        for encoding in ENCODINGS:
            prefix=np.full((4,64),TOKEN_TO_ID['<PAD>'],dtype='int64')
            for i,p in enumerate(examples):
                tokens=encode(p,encoding);assert decode(tokens,encoding)==p;prefix[i,:len(tokens)]=tokens
            for task in DEV_TASKS:
                from experiments.permworld_encoding.core import labels
                answers=[labels(p)[TASKS.index(task)] for p in examples]
                ids,mask,target,end,eq=batch(prefix,[6]*4,task,answers,device)
                assert torch.all((target!=-100).sum(1)==2)
                rows=torch.arange(4);y=torch.tensor(answers)
                toy=torch.randn(4,ids.shape[1],len(TOKEN_TO_ID),requires_grad=True)
                manual=.5*(F.cross_entropy(toy[rows,eq],y)+F.cross_entropy(toy[rows,eq+1],torch.full_like(y,TOKEN_TO_ID['<EOS>'])))
                mask_error=max(mask_error,abs(float((loss(toy,target)-manual).detach())))
                loss(toy,target).backward();actual=toy.grad.abs().sum(-1)>0
                expected=torch.zeros_like(actual);expected[rows,eq]=True;expected[rows,eq+1]=True
                assert torch.equal(actual,expected)
                captured=[];hook=net.final_norm.register_forward_hook(lambda m,i,o:captured.append(o.detach().clone()))
                with torch.inference_mode():
                    supplied=net(ids,mask);full=captured[-1][rows,end]
                    only=net(ids[:,:16],mask[:,:16]);short=captured[-1][:,-1]
                    prompt_ids,prompt_mask,_,_,prompt_eq=batch(prefix,[6]*4,task,device=device)
                    prompt=net(prompt_ids,prompt_mask)[rows,prompt_eq]
                hook.remove()
                prefix_error=max(prefix_error,float((full-short).abs().max()))
                future_error=max(future_error,float((supplied[rows,eq]-prompt).abs().max()))
        assert prefix_error<2e-5 and future_error<2e-5 and mask_error<1e-6
        # Two CPU steps and a checkpoint-in-memory RNG replay, using temporary
        # fixture models. These are not any published development source model.
        fixture=model(config,architecture,95700001,device).train()
        opt=torch.optim.AdamW(fixture.parameters(),lr=.0003)
        before=parameter_hash(fixture)
        def update():
            fixture.train();opt.zero_grad(set_to_none=True);value=loss(fixture(ids,mask),target)
            assert torch.isfinite(value);value.backward()
            norm=torch.nn.utils.clip_grad_norm_(fixture.parameters(),1.,error_if_nonfinite=True)
            assert norm>0;opt.step();return float(value.detach())
        update();saved=dict(model=copy.deepcopy(fixture.state_dict()),optimizer=copy.deepcopy(opt.state_dict()),rng=rng_state())
        continued=update();expected_model=copy.deepcopy(fixture.state_dict());expected_optimizer=copy.deepcopy(opt.state_dict())
        fixture.load_state_dict(saved['model']);opt.load_state_dict(saved['optimizer']);restore_rng(saved['rng'])
        replayed=update();error=difference(fixture.state_dict(),expected_model);moment_error=difference(opt.state_dict(),expected_optimizer)
        assert error==0 and moment_error==0 and continued==replayed and parameter_hash(fixture)!=before
        records.append(dict(architecture=architecture,parameters=metadata['parameters'],shared_parameter_hash=initial_hash,
            actual_step0_sha256=metadata['checkpoint_sha256'],paired_parameter_identity_verified=True,
            prefix_causal_max_difference=prefix_error,answer_future_causal_max_difference=future_error,
            independent_loss_max_difference=mask_error,loss_mask_gradients_checked=True,
            cpu_fixture_optimizer_updates=3,cpu_fixture_resume_parameter_error=error,cpu_fixture_resume_optimizer_error=moment_error,
            fixture_parameter_changed=True,source_model_optimizer_updates=0))
    json_write(ROOT/'cpu_check.json',dict(status='passed',records=records,metrics_checks=metrics.verify(),
        GPU_forward_started=False,formal_training_started=False,development_source_training_started=False))
    print({'cpu_check':'passed','architectures':records},flush=True)
