"""Differentiable frozen-GRU early replay; no optimizer and no fitted objects.

Unlike the previous inference-only helper, this graph retains derivatives
from the final numeric logits to the supplied *raw* first-layer state.
"""
from __future__ import annotations

import torch


def early_logits(model, previous_states, first_layer_after, suffix_ids, start_position):
    """Return [batch,vocab] after upper-current-token replay and original suffix.

    Call with CuDNN disabled for eval-mode recurrent backward. All four GRUs
    stay in eval mode; changing to train mode would change the frozen model.
    No normalization is fed back as a recurrent state.
    """
    if torch.is_inference_mode_enabled():
        raise RuntimeError('The gradient path must not run in inference mode')
    if model.training or any(module.training for module in model.modules()):
        raise ValueError('The complete frozen model must remain in eval mode')
    parameter = next(model.parameters())
    h = first_layer_after
    previous = torch.as_tensor(previous_states, device=parameter.device, dtype=parameter.dtype)
    ids = torch.as_tensor(suffix_ids, device=parameter.device, dtype=torch.long)
    if h.shape != (len(ids), model.config.d_model):
        raise ValueError('Unexpected first-layer state shape')
    if h.device != parameter.device or h.dtype != parameter.dtype:
        raise ValueError('First-layer tensor must already use the frozen model dtype/device')
    if previous.shape != (len(ids), 4, model.config.d_model):
        raise ValueError('Expected recipient previous state [batch,4,width]')
    if len(model.recurrent_layers) != 4 or ids.shape[1] < 1:
        raise ValueError('Expected four layers and a nonempty original suffix')
    current = [h]
    value = h[:, None, :]
    for layer in range(1, 4):
        value = model.interlayer_dropout[layer - 1](value)
        value, final = model.recurrent_layers[layer](
            value, previous[:, layer, :].unsqueeze(0).contiguous())
        current.append(final.squeeze(0))
    positions = int(start_position) + torch.arange(ids.shape[1], device=ids.device)
    hidden = model.embedding_dropout(
        model.token_embedding(ids) + model.position_embedding(positions)[None, :, :])
    for layer in range(4):
        hidden, _ = model.recurrent_layers[layer](hidden, current[layer].unsqueeze(0).contiguous())
        if layer < 3:
            hidden = model.interlayer_dropout[layer](hidden)
    return model.lm_head(model.final_norm(hidden))[:, -1, :]


def margin(logits, true_count, sign):
    """Recipient local goal c+s, including cumulative-count-matched donors."""
    c = torch.as_tensor(true_count, device=logits.device, dtype=torch.long)
    s = torch.as_tensor(sign, device=logits.device, dtype=torch.long)
    if bool(((c < 0) | (c > 30) | (c + s < 0) | (c + s > 30)).any()):
        raise ValueError('Local count goal outside fixed vocabulary')
    rows = torch.arange(len(logits), device=logits.device)
    return logits[rows, c + s] - logits[rows, c]
