"""GPU-ready causal sequence models for permutation-language experiments.

The two architectures intentionally expose the same language-model API:

``forward(input_ids, attention_mask=None) -> logits[B, L, vocab_size]``.

``CausalTransformer`` is a decoder-only Transformer with explicit causal
self-attention.  ``CausalMLP`` contains no attention or recurrence: every
block uses a learned, lower-triangular token-mixing MLP followed by a
position-wise channel MLP.  Masking both token-mixing matrices makes the MLP
strictly prefix-causal, including after composing multiple layers.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F


__all__ = [
    "CausalMLP",
    "CausalTransformer",
    "ModelConfig",
    "build_model",
    "count_parameters",
]


_MODEL_TYPE_ALIASES = {
    "transformer": "transformer",
    "causal_transformer": "transformer",
    "decoder_transformer": "transformer",
    "mlp": "mlp",
    "causal_mlp": "mlp",
    "mlp_mixer": "mlp",
}


def _require_positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


@dataclass(frozen=True)
class ModelConfig:
    """Configuration shared by both causal model families."""

    vocab_size: int
    max_seq_len: int
    d_model: int = 256
    layers: int = 4
    dropout: float = 0.0
    model_type: str = "transformer"
    n_heads: int | None = None
    mlp_ratio: float = 4.0
    tie_embeddings: bool = True

    def __post_init__(self) -> None:
        _require_positive_int(self.vocab_size, "vocab_size")
        _require_positive_int(self.max_seq_len, "max_seq_len")
        _require_positive_int(self.d_model, "d_model")
        _require_positive_int(self.layers, "layers")

        if isinstance(self.dropout, bool) or not isinstance(
            self.dropout, (int, float)
        ):
            raise TypeError("dropout must be a real number")
        if not 0.0 <= float(self.dropout) < 1.0:
            raise ValueError("dropout must satisfy 0 <= dropout < 1")
        object.__setattr__(self, "dropout", float(self.dropout))

        if not isinstance(self.model_type, str):
            raise TypeError("model_type must be a string")
        try:
            normalized_type = _MODEL_TYPE_ALIASES[self.model_type.lower()]
        except KeyError:
            choices = ", ".join(sorted(set(_MODEL_TYPE_ALIASES.values())))
            raise ValueError(f"unknown model_type {self.model_type!r}; use {choices}") from None
        object.__setattr__(self, "model_type", normalized_type)

        if self.n_heads is not None:
            _require_positive_int(self.n_heads, "n_heads")
            if self.d_model % self.n_heads:
                raise ValueError("d_model must be divisible by n_heads")

        if isinstance(self.mlp_ratio, bool) or not isinstance(
            self.mlp_ratio, (int, float)
        ):
            raise TypeError("mlp_ratio must be a real number")
        if float(self.mlp_ratio) <= 0:
            raise ValueError("mlp_ratio must be positive")
        object.__setattr__(self, "mlp_ratio", float(self.mlp_ratio))

        if not isinstance(self.tie_embeddings, bool):
            raise TypeError("tie_embeddings must be a boolean")

    @property
    def resolved_n_heads(self) -> int:
        """Return an explicit attention-head count that divides ``d_model``."""

        if self.n_heads is not None:
            return self.n_heads
        for candidate in (8, 4, 2, 1):
            if self.d_model % candidate == 0:
                return candidate
        return 1

    @property
    def hidden_dim(self) -> int:
        return max(1, int(round(self.d_model * self.mlp_ratio)))


def _initialize_standard_module(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


class _CausalLanguageModel(nn.Module):
    """Shared embedding, validation, and output projection implementation."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.max_seq_len, config.d_model)
        self.embedding_dropout = nn.Dropout(config.dropout)
        self.final_norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

    def _finish_initialization(self) -> None:
        self.apply(_initialize_standard_module)
        if self.config.tie_embeddings:
            # Assign after initialization so the two names refer to one
            # Parameter and optimizers/counting utilities do not double count.
            self.lm_head.weight = self.token_embedding.weight

    def _embed_inputs(
        self, input_ids: Tensor, attention_mask: Tensor | None
    ) -> tuple[Tensor, Tensor]:
        if not isinstance(input_ids, Tensor):
            raise TypeError("input_ids must be a torch.Tensor")
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        if input_ids.dtype not in (
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        ):
            raise TypeError("input_ids must use an integer dtype")

        batch_size, sequence_length = input_ids.shape
        if batch_size == 0:
            raise ValueError("input_ids must contain at least one sequence")
        if not 1 <= sequence_length <= self.config.max_seq_len:
            raise ValueError(
                "sequence length must be between 1 and "
                f"max_seq_len={self.config.max_seq_len}"
            )

        if attention_mask is None:
            valid_tokens = torch.ones(
                (batch_size, sequence_length),
                dtype=torch.bool,
                device=input_ids.device,
            )
        else:
            if not isinstance(attention_mask, Tensor):
                raise TypeError("attention_mask must be a torch.Tensor")
            if attention_mask.shape != input_ids.shape:
                raise ValueError("attention_mask must have the same shape as input_ids")
            valid_tokens = attention_mask.to(
                device=input_ids.device, dtype=torch.bool
            )

        positions = torch.arange(sequence_length, device=input_ids.device)
        hidden = self.token_embedding(input_ids.long())
        hidden = hidden + self.position_embedding(positions).unsqueeze(0)
        hidden = self.embedding_dropout(hidden)
        hidden = hidden * valid_tokens.unsqueeze(-1).to(hidden.dtype)
        return hidden, valid_tokens

    def _output_logits(self, hidden: Tensor, valid_tokens: Tensor) -> Tensor:
        hidden = self.final_norm(hidden)
        hidden = hidden * valid_tokens.unsqueeze(-1).to(hidden.dtype)
        # lm_head deliberately has no bias, hence padding positions are exactly
        # zero and remain harmless for losses that inspect the whole tensor.
        return self.lm_head(hidden)


class _ChannelMLP(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.input_projection = nn.Linear(d_model, hidden_dim)
        self.output_projection = nn.Linear(hidden_dim, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden: Tensor) -> Tensor:
        hidden = self.input_projection(hidden)
        hidden = F.gelu(hidden)
        hidden = self.dropout(hidden)
        hidden = self.output_projection(hidden)
        return self.dropout(hidden)


class _CausalSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        max_seq_len: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = 1.0 / sqrt(self.head_dim)
        self.qkv_projection = nn.Linear(d_model, 3 * d_model)
        self.output_projection = nn.Linear(d_model, d_model)
        self.attention_dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(dropout)
        self.register_buffer(
            "causal_mask",
            torch.ones(max_seq_len, max_seq_len, dtype=torch.bool).tril(),
            persistent=False,
        )

    def forward(self, hidden: Tensor, valid_tokens: Tensor) -> Tensor:
        batch_size, sequence_length, d_model = hidden.shape
        qkv = self.qkv_projection(hidden)
        qkv = qkv.reshape(
            batch_size, sequence_length, 3, self.n_heads, self.head_dim
        ).permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(dim=0)

        scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        allowed = self.causal_mask[:sequence_length, :sequence_length]
        allowed = allowed.view(1, 1, sequence_length, sequence_length)
        allowed = allowed & valid_tokens[:, None, None, :]

        # A fully padded row has no allowed key.  Using the finite minimum and
        # zeroing disallowed probabilities after softmax avoids NaNs while
        # yielding an exactly zero attention result for that row.
        scores = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min)
        weights = F.softmax(scores, dim=-1)
        weights = weights.masked_fill(~allowed, 0.0)
        weights = self.attention_dropout(weights)

        attended = torch.matmul(weights, value)
        attended = attended.transpose(1, 2).contiguous().view(
            batch_size, sequence_length, d_model
        )
        attended = self.output_projection(attended)
        attended = self.output_dropout(attended)
        return attended * valid_tokens.unsqueeze(-1).to(attended.dtype)


class _TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(config.d_model)
        self.attention = _CausalSelfAttention(
            config.d_model,
            config.resolved_n_heads,
            config.max_seq_len,
            config.dropout,
        )
        self.channel_norm = nn.LayerNorm(config.d_model)
        self.channel_mlp = _ChannelMLP(
            config.d_model, config.hidden_dim, config.dropout
        )

    def forward(self, hidden: Tensor, valid_tokens: Tensor) -> Tensor:
        token_mask = valid_tokens.unsqueeze(-1).to(hidden.dtype)
        hidden = hidden + self.attention(self.attention_norm(hidden), valid_tokens)
        hidden = hidden * token_mask
        hidden = hidden + self.channel_mlp(self.channel_norm(hidden)) * token_mask
        return hidden * token_mask


class CausalTransformer(_CausalLanguageModel):
    """Decoder-only Transformer with strict causal and padding masks."""

    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int,
        d_model: int = 256,
        layers: int = 4,
        dropout: float = 0.0,
        *,
        n_heads: int | None = None,
        mlp_ratio: float = 4.0,
        tie_embeddings: bool = True,
    ) -> None:
        config = ModelConfig(
            vocab_size=vocab_size,
            max_seq_len=max_seq_len,
            d_model=d_model,
            layers=layers,
            dropout=dropout,
            model_type="transformer",
            n_heads=n_heads,
            mlp_ratio=mlp_ratio,
            tie_embeddings=tie_embeddings,
        )
        super().__init__(config)
        self.blocks = nn.ModuleList(
            [_TransformerBlock(config) for _ in range(config.layers)]
        )
        self._finish_initialization()

    def forward(
        self, input_ids: Tensor, attention_mask: Tensor | None = None
    ) -> Tensor:
        hidden, valid_tokens = self._embed_inputs(input_ids, attention_mask)
        for block in self.blocks:
            hidden = block(hidden, valid_tokens)
        return self._output_logits(hidden, valid_tokens)


class _CausalTokenMixingMLP(nn.Module):
    """Two learned causal linear maps with GELU between them.

    Both square matrices are lower triangular at use time.  They are shared
    over channels, as in an MLP-Mixer token MLP, but unlike a conventional
    Mixer they cannot move information from a later token to an earlier one.
    """

    def __init__(self, max_seq_len: int, dropout: float) -> None:
        super().__init__()
        self.max_seq_len = max_seq_len
        self.input_weight = nn.Parameter(torch.empty(max_seq_len, max_seq_len))
        self.input_bias = nn.Parameter(torch.zeros(max_seq_len))
        self.output_weight = nn.Parameter(torch.empty(max_seq_len, max_seq_len))
        self.output_bias = nn.Parameter(torch.zeros(max_seq_len))
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "lower_triangle",
            torch.ones(max_seq_len, max_seq_len, dtype=torch.bool).tril(),
            persistent=False,
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.input_weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.input_bias)
        nn.init.normal_(self.output_weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.output_bias)

    def forward(self, hidden: Tensor, valid_tokens: Tensor) -> Tensor:
        sequence_length = hidden.shape[1]
        triangle = self.lower_triangle[:sequence_length, :sequence_length]
        input_weight = self.input_weight[:sequence_length, :sequence_length]
        output_weight = self.output_weight[:sequence_length, :sequence_length]
        input_weight = input_weight * triangle.to(input_weight.dtype)
        output_weight = output_weight * triangle.to(output_weight.dtype)

        # F.linear mixes the last dimension, so transpose [B,L,D] to [B,D,L].
        mixed = F.linear(
            hidden.transpose(1, 2),
            input_weight,
            self.input_bias[:sequence_length],
        )
        mixed = F.gelu(mixed)
        source_mask = valid_tokens[:, None, :].to(mixed.dtype)
        mixed = self.dropout(mixed) * source_mask
        mixed = F.linear(
            mixed,
            output_weight,
            self.output_bias[:sequence_length],
        )
        mixed = self.dropout(mixed).transpose(1, 2)
        return mixed * valid_tokens.unsqueeze(-1).to(mixed.dtype)


class _CausalMLPBlock(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.token_norm = nn.LayerNorm(config.d_model)
        self.token_mlp = _CausalTokenMixingMLP(
            config.max_seq_len, config.dropout
        )
        self.channel_norm = nn.LayerNorm(config.d_model)
        self.channel_mlp = _ChannelMLP(
            config.d_model, config.hidden_dim, config.dropout
        )

    def forward(self, hidden: Tensor, valid_tokens: Tensor) -> Tensor:
        token_mask = valid_tokens.unsqueeze(-1).to(hidden.dtype)
        normalized = self.token_norm(hidden) * token_mask
        hidden = hidden + self.token_mlp(normalized, valid_tokens)
        hidden = hidden * token_mask
        hidden = hidden + self.channel_mlp(self.channel_norm(hidden)) * token_mask
        return hidden * token_mask


class CausalMLP(_CausalLanguageModel):
    """Causal sequence model built only from token- and channel-mixing MLPs."""

    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int,
        d_model: int = 256,
        layers: int = 4,
        dropout: float = 0.0,
        *,
        mlp_ratio: float = 4.0,
        tie_embeddings: bool = True,
    ) -> None:
        config = ModelConfig(
            vocab_size=vocab_size,
            max_seq_len=max_seq_len,
            d_model=d_model,
            layers=layers,
            dropout=dropout,
            model_type="mlp",
            mlp_ratio=mlp_ratio,
            tie_embeddings=tie_embeddings,
        )
        super().__init__(config)
        self.blocks = nn.ModuleList(
            [_CausalMLPBlock(config) for _ in range(config.layers)]
        )
        self._finish_initialization()

    def forward(
        self, input_ids: Tensor, attention_mask: Tensor | None = None
    ) -> Tensor:
        hidden, valid_tokens = self._embed_inputs(input_ids, attention_mask)
        for block in self.blocks:
            hidden = block(hidden, valid_tokens)
        return self._output_logits(hidden, valid_tokens)


def build_model(
    config: ModelConfig | str | None = None, **config_values: Any
) -> CausalTransformer | CausalMLP:
    """Build a model from a config, or from a model name plus config keywords.

    Examples::

        build_model(ModelConfig(vocab_size=136, max_seq_len=1024))
        build_model("mlp", vocab_size=136, max_seq_len=1024)
        build_model(vocab_size=136, max_seq_len=1024, model_type="transformer")
    """

    if isinstance(config, ModelConfig):
        if config_values:
            raise TypeError("config keywords cannot accompany a ModelConfig")
        resolved = config
    elif isinstance(config, str):
        if "model_type" in config_values:
            raise TypeError("model_type was provided both positionally and by keyword")
        resolved = ModelConfig(model_type=config, **config_values)
    elif config is None:
        resolved = ModelConfig(**config_values)
    else:
        raise TypeError("config must be a ModelConfig, model-type string, or None")

    common = {
        "vocab_size": resolved.vocab_size,
        "max_seq_len": resolved.max_seq_len,
        "d_model": resolved.d_model,
        "layers": resolved.layers,
        "dropout": resolved.dropout,
        "mlp_ratio": resolved.mlp_ratio,
        "tie_embeddings": resolved.tie_embeddings,
    }
    if resolved.model_type == "transformer":
        return CausalTransformer(
            **common,
            n_heads=resolved.resolved_n_heads,
        )
    return CausalMLP(**common)


def count_parameters(model: nn.Module, *, trainable_only: bool = True) -> int:
    """Count unique scalar parameters, respecting tied embedding weights."""

    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    parameters = model.parameters()
    if trainable_only:
        return sum(parameter.numel() for parameter in parameters if parameter.requires_grad)
    return sum(parameter.numel() for parameter in parameters)
