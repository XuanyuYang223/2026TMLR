"""Replay only current-token GRU layers2–4 with an explicitly supplied h1.

Both timing conditions receive exactly the same FP32 first-layer state. The
early condition recomputes the current upper-layer update from RECIPIENT
before-token upper states. The late condition keeps the native after-token
upper states. Neither operation is a whole-counterfactual-input replacement.
"""
from __future__ import annotations

import torch


@torch.inference_mode()
def replay_upper(model, previous_states, first_layer_after):
    """Return actual carried h_n[B,4,D] after one upper-layer replay.

    ``previous_states`` is the recipient's state immediately BEFORE the
    current right-value token. ``first_layer_after`` is the same edited raw
    FP32 h1 used in the late condition. Layers2–4 each consume one step from
    their corresponding previous state. Layer1, embedding and normalization
    are never rerun. Inputs are read only and the returned h1 is bit-identical.
    The caller retains the old frozen numerical context: CUDA matmul/cuDNN
    TF32 disabled and deterministic cuDNN enabled; this function does not
    silently change process-wide backend flags.
    """
    if model.training or any(module.training for module in model.modules()):
        raise ValueError('Upper replay requires the complete model in eval mode')
    if len(model.recurrent_layers) != 4 or len(model.interlayer_dropout) != 3:
        raise ValueError('Expected the frozen four-layer causal GRU')
    parameter = next(model.parameters())
    if parameter.dtype != torch.float32:
        raise ValueError('The frozen replay uses FP32 model parameters')
    previous = torch.as_tensor(previous_states, device=parameter.device)
    first = torch.as_tensor(first_layer_after, device=parameter.device)
    width = model.config.d_model
    if previous.ndim != 3 or previous.shape[1:] != (4, width):
        raise ValueError('previous_states must have shape [batch,4,width]')
    if first.shape != (len(previous), width):
        raise ValueError('first_layer_after must have shape [batch,width]')
    if previous.dtype != torch.float32 or first.dtype != torch.float32:
        raise ValueError('Previous and edited carried states must already be FP32')
    if not bool(torch.isfinite(previous).all()) or not bool(torch.isfinite(first).all()):
        raise ValueError('Recurrent states must be finite')
    # Disable ambient autocast explicitly; no conversion or arithmetic is
    # applied to first itself, which is retained exactly as the late edit.
    with torch.autocast(device_type=parameter.device.type, enabled=False):
        carried = [first]
        value = first.unsqueeze(1).contiguous()
        for layer_index in range(1, 4):
            value = model.interlayer_dropout[layer_index-1](value)
            initial = previous[:, layer_index, :].unsqueeze(0).contiguous()
            value, final = model.recurrent_layers[layer_index](value, initial)
            carried.append(final.squeeze(0))
        return torch.stack(carried, dim=1)
