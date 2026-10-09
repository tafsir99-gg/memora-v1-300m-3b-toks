"""Memora v1 model architecture."""

import math
from typing import cast

import torch
import torch.nn as nn
from torch.nn import functional as F

from memora.config import ModelConfig


class RotaryPositionalEmbedding(nn.Module):
    """RoPE implementation caching cos/sin."""

    def __init__(self, dim: int, max_seq_len: int, base: float = 10000.0) -> None:
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        # Compute theta in fp32
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)

        # Cache as non-persistent buffers (shape: max_seq_len, dim // 2)
        self.register_buffer("cos", freqs.cos(), persistent=False)
        self.register_buffer("sin", freqs.sin(), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RoPE to x of shape (B, T, n_heads, head_dim)."""
        T = x.size(1)
        cos_buf = cast(torch.Tensor, self.cos)
        sin_buf = cast(torch.Tensor, self.sin)
        # cos, sin shape: (T, dim // 2) -> (1, T, 1, dim // 2)
        cos = cos_buf[:T].view(1, T, 1, -1)
        sin = sin_buf[:T].view(1, T, 1, -1)

        # Split into real/imaginary parts
        d = self.dim // 2
        x1, x2 = x[..., :d], x[..., d:]

        # Apply rotation
        rotated = torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)
        return rotated


class Attention(nn.Module):
    def __init__(self, config: ModelConfig, rope: RotaryPositionalEmbedding) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.d_model = config.d_model
        self.head_dim = self.d_model // self.n_heads

        self.qkv = nn.Linear(self.d_model, 3 * self.d_model, bias=config.use_bias)
        self.proj = nn.Linear(self.d_model, self.d_model, bias=config.use_bias)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.rope = rope
        self.dropout_p = config.dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()

        qkv = self.qkv(x)
        q, k, v = qkv.split(self.d_model, dim=2)

        q = q.view(B, T, self.n_heads, self.head_dim)
        k = k.view(B, T, self.n_heads, self.head_dim)
        v = v.view(B, T, self.n_heads, self.head_dim)

        q = self.rope(q)
        k = self.rope(k)

        # Transpose for SDPA: (B, n_heads, T, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Causal SDPA
        y = cast(torch.Tensor, F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=True,
        ))

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return cast(torch.Tensor, self.resid_dropout(self.proj(y)))


class MLP(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.fc1 = nn.Linear(config.d_model, config.d_ff, bias=config.use_bias)
        self.fc2 = nn.Linear(config.d_ff, config.d_model, bias=config.use_bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = cast(torch.Tensor, F.gelu(self.fc1(x)))
        x = self.fc2(x)
        return cast(torch.Tensor, self.dropout(x))


class Block(nn.Module):
    def __init__(self, config: ModelConfig, rope: RotaryPositionalEmbedding) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.d_model, eps=config.norm_eps)
        self.attn = Attention(config, rope)
        self.ln_2 = nn.LayerNorm(config.d_model, eps=config.norm_eps)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class MemoraModel(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.emb_dropout = nn.Dropout(config.dropout)

        # Shared RoPE across layers
        head_dim = config.d_model // config.n_heads
        self.rope = RotaryPositionalEmbedding(head_dim, config.max_seq_len, config.rope_base)

        self.layers = nn.ModuleList([Block(config, self.rope) for _ in range(config.n_layers)])

        self.ln_f = nn.LayerNorm(config.d_model, eps=config.norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        if config.tie_embeddings:
            self.lm_head.weight = self.embedding.weight

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, T = idx.size()
        if self.config.max_seq_len < T:
            raise ValueError(
                f"Sequence length {T} exceeds max_seq_len {self.config.max_seq_len}"
            )

        x = self.embedding(idx)
        x = self.emb_dropout(x)

        for layer in self.layers:
            x = layer(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            # Compute CE loss over all logits and targets in fp32
            logits_flat = logits.view(-1, logits.size(-1)).float()
            targets_flat = targets.view(-1)
            loss = F.cross_entropy(logits_flat, targets_flat)

        return logits, loss

    def init_weights(self, seed: int) -> None:
        """Initialize weights with a specific seed deterministically."""
        # Use a local generator to keep it deterministic and isolated
        gen = torch.Generator(device="cpu")
        gen.manual_seed(seed)

        def _init_module(m: nn.Module) -> None:
            if isinstance(m, nn.Linear):
                # If we are on 'meta' device, skip actual random fills since tensor has no storage
                if m.weight.device.type != "meta":
                    m.weight.data.normal_(mean=0.0, std=self.config.init_std, generator=gen)
                if m.bias is not None and m.bias.device.type != "meta":
                    m.bias.data.zero_()
            elif isinstance(m, nn.Embedding):
                if m.weight.device.type != "meta":
                    m.weight.data.normal_(mean=0.0, std=self.config.init_std, generator=gen)
            elif isinstance(m, nn.LayerNorm):
                if m.weight.device.type != "meta":
                    m.weight.data.fill_(1.0)
                if m.bias is not None and m.bias.device.type != "meta":
                    m.bias.data.zero_()

        self.apply(_init_module)

        # Residual projection scaling
        scale = 1.0 / math.sqrt(2.0 * self.config.n_layers)
        for name, p in self.named_parameters():
            if (name.endswith("attn.proj.weight") or name.endswith("mlp.fc2.weight")) and p.device.type != "meta":
                p.data.mul_(scale)

    def weight_decay_groups(self, weight_decay: float) -> list[dict]:
        """Return parameter groups for optimizer (decay vs no-decay)."""
        decay = set()
        no_decay = set()

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            # No decay for layernorm, embedding, and biases
            if param.ndim < 2 or name.endswith("embedding.weight"):
                no_decay.add(name)
            else:
                decay.add(name)

        param_dict = dict(self.named_parameters())

        # Deduplicate tied weights to prevent them from appearing twice
        # or in multiple groups if logic differs.
        decay_params = []
        no_decay_params = []

        # Keep track of param IDs to avoid duplicate references (e.g. tied weights)
        seen_ids = set()

        for name in sorted(decay):
            p = param_dict[name]
            if id(p) not in seen_ids:
                decay_params.append(p)
                seen_ids.add(id(p))

        for name in sorted(no_decay):
            p = param_dict[name]
            if id(p) not in seen_ids:
                no_decay_params.append(p)
                seen_ids.add(id(p))

        groups = []
        if decay_params:
            groups.append({"params": decay_params, "weight_decay": weight_decay})
        if no_decay_params:
            groups.append({"params": no_decay_params, "weight_decay": 0.0})

        return groups
