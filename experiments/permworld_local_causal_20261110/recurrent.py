"""Exact frozen-GRU prefix/continuation engine for recurrent interventions.

The intervention state is the actual ``h_n`` of each recurrent layer, not a
LayerNorm readout. A patch after token t becomes that layer's h_0 at token
t+1; the other layers retain their own states at t. Dropout is disabled by
requiring evaluation mode. No optimizer or backward pass is constructed.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
import hashlib
import operator
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
VENDOR = ROOT / 'experiments/permworld_peaks_probe_20261109/vendor'
MODEL_SOURCE_SHA256 = '6942c768e441235ac657949b43514c9ba5d0f2b5900c9bd89743fffce28b700b'
VOCABULARY_SOURCE_SHA256 = 'd200ffe6f5f6cff717c1b6ed6698a5a6af42bb6580fa4e8729a2fd84b79babd6'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def parameter_hash(model):
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        array = tensor.detach().cpu().contiguous().numpy()
        for piece in (name, str(array.dtype), str(array.shape)):
            digest.update(piece.encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def frozen_gru_class():
    """Load only the original hash-verified class AST, never its run workflow."""
    source = VENDOR / 'archdev_common.py'
    passage = VENDOR / 'passage.py'
    if sha(source) != MODEL_SOURCE_SHA256 or sha(passage) != VOCABULARY_SOURCE_SHA256:
        raise ValueError('Frozen model or vocabulary source changed')
    from experiments.permworld_peaks_probe_20261109.vendor.passage import TOKEN_TO_ID
    node = next(item for item in ast.parse(source.read_text()).body
                if isinstance(item, ast.ClassDef) and item.name == 'CausalGRU')
    namespace = dict(torch=torch, nn=nn, SimpleNamespace=SimpleNamespace,
                     TOKEN_TO_ID=TOKEN_TO_ID, __name__=__name__)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), namespace)
    return namespace['CausalGRU']


def load_model(descriptor, device='cpu', model_config=None):
    """Strictly load an existing hash-bound state into the frozen class.

    ``descriptor`` accepts ``checkpoint_path/checkpoint_sha256`` or
    ``path/sha256``. Trained checkpoints omit configuration, so callers must
    pass their frozen model config or attach ``model_config`` to the descriptor.
    A supplied parameter/data/task binding is verified rather than inferred.
    """
    path = Path(descriptor.get('checkpoint_path') or descriptor['path'])
    expected = descriptor.get('checkpoint_sha256') or descriptor.get('sha256')
    if not expected or sha(path) != expected:
        raise ValueError(f'Checkpoint SHA mismatch: {path}')
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    if checkpoint.get('architecture') != 'gru':
        raise ValueError('Only the frozen GRU architecture is supported')
    config = model_config or descriptor.get('model_config') or checkpoint.get('model_config')
    if config is None:
        raise ValueError('An explicit frozen model_config is required for this trained checkpoint')
    if descriptor.get('data_sha256') and checkpoint.get('data_sha256') != descriptor['data_sha256']:
        raise ValueError('Checkpoint data binding differs')
    if descriptor.get('task') and checkpoint.get('run_id'):
        suffix = f'_gru_{descriptor.get("encoding", "one_line")}_{descriptor["task"]}'
        if not checkpoint['run_id'].endswith(suffix):
            raise ValueError('Checkpoint task/encoding binding differs')
    if descriptor.get('phase', 'trained') == 'trained':
        if checkpoint.get('step') != 20000 or checkpoint.get('finite_gradient_updates') != 20000:
            raise ValueError('Expected the actual frozen20k endpoint')
    constructor = frozen_gru_class()
    # Constructor randomness is irrelevant after strict loading and is restored
    # here so caller-owned random controls are not affected by model loading.
    with torch.random.fork_rng(devices=[]):
        model = constructor(dict(config))
    model.load_state_dict(checkpoint['model'], strict=True)
    if descriptor.get('parameter_sha256') and parameter_hash(model) != descriptor['parameter_sha256']:
        raise ValueError('Actual checkpoint parameter identity differs')
    model = model.to(device).eval().requires_grad_(False)
    model.checkpoint_binding = dict(path=str(path), sha256=expected,
        parameter_sha256=parameter_hash(model), model_source_sha256=MODEL_SOURCE_SHA256,
        vocabulary_source_sha256=VOCABULARY_SOURCE_SHA256, model_config=dict(config))
    if sha(path) != expected:
        raise ValueError('Checkpoint file changed during loading')
    return model


@dataclass(frozen=True)
class ResumeResult:
    logits: torch.Tensor       # [batch,time,vocabulary], original lm_head
    states: torch.Tensor       # [batch,layer,width], actual GRU h_n
    normalized: torch.Tensor   # [batch,time,width], original final_norm/mask


def _validate(model, ids, states=None, mask=None):
    if model.training or any(layer.training for layer in model.modules()):
        raise ValueError('Recurrent interventions require model.eval(), including dropout modules')
    parameter = next(model.parameters())
    ids = torch.as_tensor(ids, device=parameter.device)
    if ids.ndim != 2 or ids.dtype not in (torch.int8, torch.uint8, torch.int16, torch.int32, torch.int64):
        raise ValueError('ids must be an integer tensor [batch,time]')
    ids = ids.long()
    if mask is None:
        # Match CausalGRU.forward's default; callers with PAD tokens can pass
        # the exact original explicit mask. Earlier states remain causal.
        mask = torch.ones_like(ids, dtype=torch.bool)
    else:
        mask = torch.as_tensor(mask, dtype=torch.bool, device=parameter.device)
        if mask.shape != ids.shape:
            raise ValueError('mask must have the same shape as ids')
    layers, width = len(model.recurrent_layers), model.config.d_model
    if states is None:
        states = torch.zeros((len(ids), layers, width), device=parameter.device, dtype=parameter.dtype)
    elif states.shape != (len(ids), layers, width) or states.dtype != parameter.dtype or states.device != parameter.device:
        raise ValueError('states must match [batch,layers,width], model dtype and model device')
    return ids, states, mask


@torch.inference_mode()
def resume(model, ids_suffix, start_position, states, mask=None):
    """Consume a suffix seeded by each layer's actual state at the cut.

    ``start_position`` is the suffix's absolute first-token position, an int
    or one integer per row. It is never reset to zero at an intervention.
    ``states`` is not modified. No hidden state is normalized before feedback.
    """
    ids, states, mask = _validate(model, ids_suffix, states, mask)
    positions = torch.as_tensor(start_position, device=ids.device)
    if positions.dtype not in (torch.int8, torch.uint8, torch.int16, torch.int32, torch.int64):
        raise ValueError('start_position must be integer')
    if positions.ndim == 0:
        positions = positions.long() + torch.arange(ids.shape[1], device=ids.device)
    elif positions.shape == (len(ids),):
        positions = positions.long()[:,None] + torch.arange(ids.shape[1], device=ids.device)[None]
    else:
        raise ValueError('start_position must be scalar or one value per row')
    if not ids.shape[1]:
        empty = torch.empty((len(ids), 0, model.config.d_model), dtype=states.dtype, device=ids.device)
        return ResumeResult(model.lm_head(empty), states.clone(), empty)
    if bool((positions < 0).any()) or bool((positions >= model.config.max_seq_len).any()):
        raise ValueError('Absolute token position outside the frozen embedding table')
    positional = model.position_embedding(positions)
    if positions.ndim == 1:
        positional = positional[None]
    hidden = model.embedding_dropout(model.token_embedding(ids) + positional) * mask[...,None]
    final_states = []
    for layer_index, layer in enumerate(model.recurrent_layers):
        initial = states[:,layer_index,:].unsqueeze(0).contiguous()
        hidden, final = layer(hidden, initial)
        final_states.append(final.squeeze(0))
        if layer_index < len(model.interlayer_dropout):
            hidden = model.interlayer_dropout[layer_index](hidden)
    normalized = model.final_norm(hidden) * mask[...,None]
    return ResumeResult(model.lm_head(normalized), torch.stack(final_states, dim=1), normalized)


@torch.inference_mode()
def run_prefix(model, ids, cut, mask=None):
    """Actual recurrent states immediately AFTER zero-based inclusive token cut."""
    ids = torch.as_tensor(ids, device=next(model.parameters()).device)
    if isinstance(cut, bool):
        raise ValueError('cut must be a valid zero-based inclusive token index')
    try:
        cut = operator.index(cut)
    except TypeError as error:
        raise ValueError('cut must be an integer scalar') from error
    if ids.ndim != 2 or not 0 <= cut < ids.shape[1]:
        raise ValueError('cut must be a valid zero-based inclusive token index')
    selected_mask = None if mask is None else torch.as_tensor(mask,device=ids.device)[:,:cut+1]
    return resume(model, ids[:,:cut+1], 0, None, selected_mask).states


@torch.inference_mode()
def full_forward(model, ids, mask=None):
    """Convenience zero-state engine run; compare to original model.forward."""
    return resume(model, ids, 0, None, mask)


@torch.inference_mode()
def step(model, token_ids, absolute_position, states):
    """One-token equivalent, useful for an independent unroll verification."""
    ids = torch.as_tensor(token_ids,device=next(model.parameters()).device)
    if ids.ndim == 1:
        ids = ids[:,None]
    if ids.ndim != 2 or ids.shape[1] != 1:
        raise ValueError('step requires one token per row')
    return resume(model, ids, absolute_position, states)
