"""Actual-timestep, unchanged-h1, and frozen historical replay invariants."""
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from experiments.permworld_local_causal_20261110.recurrent import (
    ROOT, frozen_gru_class, load_model, parameter_hash, run_prefix, resume, sha,
)
from experiments.permworld_local_timing_20261110.replay import replay_upper


@pytest.fixture(autouse=True)
def exact_original_frozen_numerical_context():
    old=(torch.get_num_threads(),torch.backends.cuda.matmul.allow_tf32,
         torch.backends.cudnn.allow_tf32,torch.backends.cudnn.deterministic)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.deterministic=True
    yield
    torch.set_num_threads(old[0])
    torch.backends.cuda.matmul.allow_tf32=old[1]
    torch.backends.cudnn.allow_tf32=old[2]
    torch.backends.cudnn.deterministic=old[3]


def small_model(seed=911):
    torch.manual_seed(seed)
    return frozen_gru_class()(dict(d_model=8,layers=4,heads=2,mlp_ratio=4,
        dropout=.4,tie_embeddings=True,max_seq_len=80,vocab_size=188)).eval()


def same_bits(a, b):
    return torch.equal(a.contiguous().view(torch.int32), b.contiguous().view(torch.int32))


def test_noedit_replay_recovers_native_current_step_and_future():
    model=small_model();ids=torch.randint(1,50,(3,19));cut=12
    before=run_prefix(model,ids,cut-1)
    native=resume(model,ids[:,cut:cut+1],cut,before).states
    replayed=replay_upper(model,before,native[:,0])
    torch.testing.assert_close(replayed,native,atol=2e-6,rtol=0)
    assert same_bits(replayed[:,0],native[:,0])
    a=resume(model,ids[:,cut+1:],cut+1,native)
    b=resume(model,ids[:,cut+1:],cut+1,replayed)
    torch.testing.assert_close(a.logits,b.logits,atol=2e-6,rtol=0)


def test_replay_uses_before_upper_seeds_and_never_reruns_l1_or_norm():
    model=small_model();ids=torch.randint(1,50,(3,19));cut=12
    before=run_prefix(model,ids,cut-1)
    native=resume(model,ids[:,cut:cut+1],cut,before).states
    edited=native[:,0].clone();edited[:,2]+=1.5
    calls=[];seeds=[];hooks=[]
    for name,module in [('embedding',model.token_embedding),('positions',model.position_embedding),
                        ('layer1',model.recurrent_layers[0]),('final_norm',model.final_norm)]:
        hooks.append(module.register_forward_hook(lambda m,a,o,name=name:calls.append(name)))
    for module in model.recurrent_layers[1:]:
        hooks.append(module.register_forward_pre_hook(lambda m,args:seeds.append(args[1].clone())))
    previous_copy=before.clone();edited_copy=edited.clone()
    result=replay_upper(model,before,edited)
    for hook in hooks:hook.remove()
    assert calls==[]
    assert len(seeds)==3
    for layer,seed in enumerate(seeds,1):assert torch.equal(seed.squeeze(0),before[:,layer])
    assert torch.equal(before,previous_copy) and same_bits(edited,edited_copy)
    assert same_bits(result[:,0],edited)
    wrong=replay_upper(model,native,edited)
    assert float((result[:,1:]-wrong[:,1:]).abs().max())>1e-8


def test_early_and_late_have_identical_edited_h1_but_different_upper_updates():
    model=small_model();ids=torch.randint(1,50,(4,16));cut=12
    before=run_prefix(model,ids,cut-1)
    native=resume(model,ids[:,cut:cut+1],cut,before).states
    # A fixed arbitrary edit tests the computational contract, not efficacy.
    edited=(native[:,0].double()+torch.linspace(-1.,1.,8).double()).float()
    late=native.clone();late[:,0]=edited
    early=replay_upper(model,before,edited)
    assert same_bits(early[:,0],late[:,0])
    assert torch.equal(late[:,1:],native[:,1:])
    assert float((early[:,1:]-late[:,1:]).abs().max())>1e-8


def test_donor_h1_replay_is_recipient_upper_replay_not_full_donor_swap():
    model=small_model();recipient=torch.randint(1,50,(4,17));donor=recipient.clone();cut=10
    donor[:,:cut]=torch.randint(51,99,(4,cut))
    recipient_before=run_prefix(model,recipient,cut-1)
    donor_before=run_prefix(model,donor,cut-1)
    donor_after=resume(model,donor[:,cut:cut+1],cut,donor_before).states
    hybrid=replay_upper(model,recipient_before,donor_after[:,0])
    assert same_bits(hybrid[:,0],donor_after[:,0])
    assert float((hybrid[:,1:]-donor_after[:,1:]).abs().max())>1e-8
    donor_replay=replay_upper(model,donor_before,donor_after[:,0])
    torch.testing.assert_close(donor_replay,donor_after,atol=2e-6,rtol=0)
    # The distinct whole-state positive control still recovers the donor with
    # identical suffix; it is not relabeled as the early local intervention.
    swapped=resume(model,recipient[:,cut+1:],cut+1,donor_after)
    with torch.inference_mode():donor_logits=model(donor)
    torch.testing.assert_close(swapped.logits,donor_logits[:,cut+1:],atol=2e-6,rtol=0)


def test_wrong_dtype_shape_or_training_mode_is_rejected():
    model=small_model();previous=torch.zeros(2,4,8);first=torch.zeros(2,8)
    with pytest.raises(ValueError,match='FP32'):replay_upper(model,previous.double(),first)
    with pytest.raises(ValueError,match='shape'):replay_upper(model,previous,torch.zeros(2,1,8))
    model.train()
    with pytest.raises(ValueError,match='eval'):replay_upper(model,previous,first)


@pytest.mark.parametrize('task',['peaks','valleys'])
def test_real_historical_eight_validation_anchors_noedit_replay(task):
    path=ROOT/'diagnostics/permworld_local_causal_20261110/source_audit.json'
    if not path.exists():pytest.skip('Historical source inventory unavailable')
    audit=json.loads(path.read_text());unit=next(u for u in audit['units'] if u['historical'])
    desc=unit['models'][task]
    if not Path(desc['checkpoint_path']).exists() or not Path(unit['data_path']).exists():
        pytest.skip('Historical frozen checkpoint/data unavailable')
    device='cuda' if torch.cuda.is_available() else 'cpu'
    model=load_model(desc,device=device,model_config=audit['model'])
    from experiments.permworld_peaks_probe_20261109.vendor.passage import TOKEN_TO_ID
    data_hash_before=sha(unit['data_path']);checkpoint_hash_before=sha(desc['checkpoint_path'])
    with np.load(unit['data_path'],allow_pickle=False) as data:
        n=10;rows=np.flatnonzero(data['validation_lengths']==n)[:8]
        prefix=data['validation_one_line_input'][rows,0,:2*n+4]
    suffix=np.tile([TOKEN_TO_ID[f'<{task.upper()}>'],TOKEN_TO_ID['=']],(len(rows),1))
    ids=torch.as_tensor(np.concatenate([prefix,suffix],1),device=device,dtype=torch.long)
    parameter_before=parameter_hash(model);cut=12
    previous=run_prefix(model,ids,cut-1)
    native=resume(model,ids[:,cut:cut+1],cut,previous).states
    replayed=replay_upper(model,previous,native[:,0])
    continuation=resume(model,ids[:,cut+1:],cut+1,replayed)
    with torch.inference_mode():original=model(ids)
    state_error=float((replayed-native).abs().max())
    logit_error=float((continuation.logits-original[:,cut+1:]).abs().max())
    assert state_error<=2e-5 and logit_error<=1e-4
    assert same_bits(replayed[:,0],native[:,0])
    assert parameter_hash(model)==parameter_before
    assert sha(unit['data_path'])==data_hash_before==unit['data_sha256']
    assert sha(desc['checkpoint_path'])==checkpoint_hash_before==desc['checkpoint_sha256']
    print(json.dumps(dict(historical_validation_preflight='passed',task=task,anchors=len(rows),
        length=n,device=device,noedit_state_max_error=state_error,
        noedit_query_logits_max_error=logit_error,h1_bit_identical=True,
        matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
        cudnn_tf32=torch.backends.cudnn.allow_tf32,
        cudnn_deterministic=torch.backends.cudnn.deterministic,
        parameters_and_source_files_unchanged=True,no_formal_new_unit_scoring=True)),flush=True)
