"""Causal-engine invariants, including an actual historical validation preflight."""
from pathlib import Path
import json

import numpy as np
import pytest
import torch

from experiments.permworld_local_causal_20261110.recurrent import (
    ROOT, frozen_gru_class, load_model, parameter_hash, sha, full_forward,
    run_prefix, resume, step,
)


def small_model(seed=72):
    torch.manual_seed(seed)
    return frozen_gru_class()(dict(d_model=8,layers=4,heads=2,mlp_ratio=4,
        dropout=.3,tie_embeddings=True,max_seq_len=80,vocab_size=188)).eval()


def test_exact_full_original_forward_and_actual_four_layer_states():
    model = small_model(); ids = torch.randint(1,50,(3,17))
    mask = torch.ones_like(ids,dtype=torch.bool); mask[1,-3:] = False
    captured = []
    hooks = [layer.register_forward_hook(lambda m,a,o:captured.append(o[1].squeeze(0)))
             for layer in model.recurrent_layers]
    with torch.inference_mode():
        original = model(ids,mask)
    for hook in hooks: hook.remove()
    result = full_forward(model,ids,mask)
    torch.testing.assert_close(result.logits,original,atol=2e-6,rtol=0)
    torch.testing.assert_close(result.states,torch.stack(captured,1),atol=2e-6,rtol=0)
    assert result.states.shape == (3,4,8)
    assert not result.logits.requires_grad


@pytest.mark.parametrize('cut',[0,3,9,15])
def test_chunk_resume_keeps_absolute_positions_and_each_layer_state(cut):
    model = small_model(); ids = torch.randint(1,50,(3,17))
    original = full_forward(model,ids)
    state = run_prefix(model,ids,cut)
    before = state.clone()
    result = resume(model,ids[:,cut+1:],cut+1,state)
    torch.testing.assert_close(result.logits,original.logits[:,cut+1:],atol=2e-6,rtol=0)
    torch.testing.assert_close(result.normalized,original.normalized[:,cut+1:],atol=2e-6,rtol=0)
    torch.testing.assert_close(result.states,original.states,atol=2e-6,rtol=0)
    assert torch.equal(state,before)


def test_step_unroll_matches_original_chunk_and_noop_swap():
    model = small_model(); ids = torch.randint(1,50,(2,15))
    result = full_forward(model,ids)
    state = None; outputs = []
    for t in range(ids.shape[1]):
        one = step(model,ids[:,t],t,state); state = one.states; outputs.append(one.logits)
    torch.testing.assert_close(torch.cat(outputs,1),result.logits,atol=2e-6,rtol=0)
    cut = 7; state = run_prefix(model,ids,cut)
    changed = state.clone(); changed[:,2,:] = state[:,2,:]
    a = resume(model,ids[:,cut+1:],cut+1,state)
    b = resume(model,ids[:,cut+1:],cut+1,changed)
    assert torch.equal(a.logits,b.logits)


def test_partial_layer_patch_is_the_recurrence_seed_not_a_readout_patch():
    model = small_model(); ids = torch.randint(1,50,(2,15)); cut = 7
    state = run_prefix(model,ids,cut); altered = state.clone(); altered[:,1,3] += 2.
    seeds = []
    hooks = [layer.register_forward_pre_hook(lambda m,args:seeds.append(args[1].clone()))
             for layer in model.recurrent_layers]
    actual = resume(model,ids[:,cut+1:],cut+1,altered)
    for hook in hooks: hook.remove()
    for layer in range(4):
        assert torch.equal(seeds[layer].squeeze(0),altered[:,layer,:])
    assert torch.equal(altered[:,[0,2,3]],state[:,[0,2,3]])
    original = resume(model,ids[:,cut+1:],cut+1,state)
    assert float((actual.logits-original.logits).abs().max()) > 1e-8


def test_full_donor_recurrent_swap_with_identical_suffix_recovers_donor():
    model = small_model(); source = torch.randint(1,50,(4,17)); cut = 8
    donor = source.clone(); donor[:,:cut] = torch.randint(1,50,(4,cut))
    assert torch.equal(source[:,cut+1:],donor[:,cut+1:])
    donor_state = run_prefix(model,donor,cut)
    transferred = resume(model,source[:,cut+1:],cut+1,donor_state)
    with torch.inference_mode():
        expected = model(donor)
    torch.testing.assert_close(transferred.logits,expected[:,cut+1:],atol=2e-6,rtol=0)


def test_hash_bound_loader_strict_state_and_frozen_parameters(tmp_path):
    original = small_model(); path = tmp_path/'checkpoint.pt'
    config = vars(original.config)
    torch.save(dict(model=original.state_dict(),architecture='gru',step=20000,
                    finite_gradient_updates=20000,model_config=config),path)
    before_rng = torch.random.get_rng_state().clone()
    loaded = load_model(dict(path=str(path),sha256=sha(path),parameter_sha256=parameter_hash(original)))
    assert torch.equal(torch.random.get_rng_state(),before_rng)
    assert parameter_hash(loaded) == parameter_hash(original)
    assert not loaded.training and not any(p.requires_grad for p in loaded.parameters())
    with pytest.raises(ValueError,match='SHA'):
        load_model(dict(path=str(path),sha256='0'*64))
    torch.save(dict(model={k:v for k,v in original.state_dict().items() if k!='final_norm.bias'},
                    architecture='gru',step=20000,finite_gradient_updates=20000,model_config=config),path)
    with pytest.raises(RuntimeError):
        load_model(dict(path=str(path),sha256=sha(path)))


def test_training_mode_rejected_and_empty_suffix_preserves_all_states():
    model = small_model(); ids = torch.randint(1,50,(2,8)); states = run_prefix(model,ids,7)
    empty = resume(model,ids[:,8:],8,states)
    assert empty.logits.shape == (2,0,188) and torch.equal(empty.states,states)
    model.train()
    with pytest.raises(ValueError,match='eval'):
        full_forward(model,ids)


def test_real_historical_eight_validation_anchors_noop_and_full_donor_swap():
    audit_path = ROOT/'diagnostics/permworld_lemma_composition_20261110/cache_audit.json'
    if not audit_path.exists():
        pytest.skip('Historical cache inventory is unavailable')
    unit = next(u for u in json.loads(audit_path.read_text())['units'] if u['unit']=='archdev001')
    checkpoint = next(c for c in unit['trained_checkpoints'] if c['task']=='peaks')
    if not Path(checkpoint['path']).exists() or not Path(unit['data_path']).exists():
        pytest.skip('Historical original-disk checkpoint/data are unavailable')
    config = json.loads((ROOT/'diagnostics/permworld_repetition_20261106/protocol.json').read_text())['model']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_model(dict(checkpoint_path=checkpoint['path'],checkpoint_sha256=checkpoint['sha256'],
                      parameter_sha256=checkpoint['parameter_sha256'],data_sha256=unit['data_sha256'],
                      task='peaks',encoding='one_line'),device=device,model_config=config)
    from experiments.permworld_peaks_probe_20261109.vendor.passage import TOKEN_TO_ID
    with np.load(unit['data_path'],allow_pickle=False) as raw:
        assert sha(unit['data_path']) == unit['data_sha256']
        n = 10; rows = np.flatnonzero(raw['validation_lengths']==n)[:8]
        prefix = raw['validation_one_line_input'][rows,0,:2*n+4]
    suffix = np.tile([TOKEN_TO_ID['<PEAKS>'],TOKEN_TO_ID['=']],(len(rows),1))
    ids = torch.as_tensor(np.concatenate([prefix,suffix],axis=1),device=device,dtype=torch.long)
    parameter_before = parameter_hash(model)
    cut = 12  # Immediately after the fifth value token; development only.
    normalized_reference = []
    hook = model.final_norm.register_forward_hook(lambda m,a,o:normalized_reference.append(o))
    with torch.inference_mode(): original_logits = model(ids)
    hook.remove()
    whole = full_forward(model,ids)
    state = run_prefix(model,ids,cut)
    ordinary = resume(model,ids[:,cut+1:],cut+1,state)
    noop = resume(model,ids[:,cut+1:],cut+1,state.clone())
    donor_ids = ids.clone(); donor_ids[:,[4,6]] = donor_ids[:,[6,4]]
    donor_state = run_prefix(model,donor_ids,cut)
    transfer = resume(model,ids[:,cut+1:],cut+1,donor_state)
    with torch.inference_mode(): donor_logits = model(donor_ids)
    torch.testing.assert_close(whole.logits,original_logits,atol=2e-5,rtol=0)
    torch.testing.assert_close(whole.normalized,normalized_reference[0],atol=2e-5,rtol=0)
    torch.testing.assert_close(ordinary.normalized,normalized_reference[0][:,cut+1:],atol=2e-5,rtol=0)
    torch.testing.assert_close(ordinary.logits,original_logits[:,cut+1:],atol=2e-5,rtol=0)
    assert torch.equal(ordinary.logits,noop.logits)
    torch.testing.assert_close(transfer.logits,donor_logits[:,cut+1:],atol=2e-5,rtol=0)
    torch.testing.assert_close(ordinary.logits[:,-1].softmax(-1),original_logits[:,-1].softmax(-1),atol=2e-6,rtol=0)
    assert parameter_hash(model) == parameter_before
    print(json.dumps(dict(historical_validation_preflight='passed',anchors=len(rows),length=n,device=device,
        whole_logits_max_error=float((whole.logits-original_logits).abs().max()),
        continuation_hidden_max_error=float((ordinary.normalized-normalized_reference[0][:,cut+1:]).abs().max()),
        continuation_logits_max_error=float((ordinary.logits-original_logits[:,cut+1:]).abs().max()),
        query_probability_max_error=float((ordinary.logits[:,-1].softmax(-1)-original_logits[:,-1].softmax(-1)).abs().max()),
        full_donor_swap_logits_max_error=float((transfer.logits-donor_logits[:,cut+1:]).abs().max()),
        same_state_swap_bitwise=True,parameters_unchanged=True)),flush=True)
    # Tests never load a formal test split, fit a probe, or inspect true counts.
