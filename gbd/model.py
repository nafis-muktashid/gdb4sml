"""
Causal Decoder Transformer students for Stage 1 GBD experiments.

Presets:
  - "30m":        n_layer=7,  n_head=8,  d_model=512  (~30.5M params at V=16k, course reference scale)
  - "gpt2-small": n_layer=12, n_head=12, d_model=768  (exact GPT-2-small transformer blocks; ~97.6M at V=16k, 124.4M at V=50,257)
  - "tiny":       n_layer=4,  n_head=4,  d_model=256  (~7.3M params at V=16k, fast smoke tests)

Uses PyTorch `F.scaled_dot_product_attention(..., is_causal=True)` for FlashAttention / SDPA on GPU,
supports tied token/output embeddings, static-embedding initialisation (Baseline 8), and
Masked-Input Regularisation (MIR, Baseline 9).
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    vocab_size: int = 16_000
    max_seq_len: int = 512
    n_layer: int = 7
    n_head: int = 8
    d_model: int = 512
    dropout: float = 0.0
    tie_weights: bool = True

    @classmethod
    def from_preset(cls, name: str, vocab_size: int = 16_000, max_seq_len: int = 512, dropout: float = 0.0) -> "ModelConfig":
        key = name.lower().strip()
        if key in ("30m", "small-30m"):
            return cls(vocab_size=vocab_size, max_seq_len=max_seq_len, n_layer=7, n_head=8, d_model=512, dropout=dropout)
        if key in ("gpt2-small", "124m", "gpt2"):
            return cls(vocab_size=vocab_size, max_seq_len=max_seq_len, n_layer=12, n_head=12, d_model=768, dropout=dropout)
        if key == "39m":
            return cls(vocab_size=vocab_size, max_seq_len=max_seq_len, n_layer=10, n_head=8, d_model=512, dropout=dropout)
        if key == "tiny":
            return cls(vocab_size=vocab_size, max_seq_len=max_seq_len, n_layer=4, n_head=4, d_model=256, dropout=dropout)
        raise ValueError(f"Unknown model preset: {name!r}")


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.head_dim = cfg.d_model // cfg.n_head
        self.dropout = cfg.dropout
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.proj._res_scale = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, L, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, L, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, L, self.n_head, self.head_dim).transpose(1, 2)
        y = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        y = y.transpose(1, 2).contiguous().view(B, L, D)
        return self.proj(y)


class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.fc1 = nn.Linear(cfg.d_model, 4 * cfg.d_model, bias=False)
        self.fc2 = nn.Linear(4 * cfg.d_model, cfg.d_model, bias=False)
        self.fc2._res_scale = True
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.fc2(F.gelu(self.fc1(x), approximate="tanh")))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class CausalLM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_weights:
            self.lm_head.weight = self.tok_emb.weight
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            std = 0.02 / math.sqrt(2 * self.cfg.n_layer) if getattr(module, "_res_scale", False) else 0.02
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    @torch.no_grad()
    def init_from_static_embeddings(self, E: np.ndarray, seed: int = 0):
        """Baseline 8: initialise token embeddings from same-corpus static vectors E (V, D_e)."""
        E = np.asarray(E, dtype=np.float32)
        E_norm = E / np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-12)
        V, De = E_norm.shape
        D = self.cfg.d_model
        if De == D:
            proj = E_norm
        elif De < D:
            # random orthogonal / semi-orthogonal isometric lift to d_model
            rng = np.random.default_rng(seed)
            Q, _ = np.linalg.qr(rng.normal(size=(D, De)))
            proj = E_norm @ Q.T
        else:
            U, S, _ = np.linalg.svd(E_norm, full_matrices=False)
            proj = U[:, :D] * S[:D]
        # scale to match standard embedding std=0.02
        proj = proj * (0.02 / max(float(proj.std()), 1e-8))
        self.tok_emb.weight[:V].copy_(torch.from_numpy(proj.astype(np.float32)))

    def num_parameters(self, non_embedding: bool = False) -> int:
        total = sum(p.numel() for p in self.parameters())
        if non_embedding:
            total -= self.pos_emb.weight.numel()
            if self.cfg.tie_weights:
                total -= self.tok_emb.weight.numel()
        return total

    def forward_hidden(self, idx: torch.Tensor, layer: int = -1) -> torch.Tensor:
        """
        Return contextual hidden states (B, L, d_model) without materialising vocab logits.
        `layer`:
          - 0 .. n_layer-1: output of `self.blocks[layer]` (0-indexed; e.g. n_layer // 2 for mid-layer)
          - -1 (default):   final `self.ln_f` output right before `lm_head`
        """
        B, L = idx.shape
        assert L <= self.cfg.max_seq_len, f"Sequence length {L} exceeds max_seq_len {self.cfg.max_seq_len}"
        pos = torch.arange(0, L, device=idx.device, dtype=torch.long)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        if layer == -1 or layer >= self.cfg.n_layer:
            for blk in self.blocks:
                x = blk(x)
            return self.ln_f(x)
        if layer < 0:
            layer = self.cfg.n_layer + layer
        assert 0 <= layer < self.cfg.n_layer, f"Invalid layer index {layer} for n_layer={self.cfg.n_layer}"
        for li in range(layer + 1):
            x = self.blocks[li](x)
        return x

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return self.lm_head(self.forward_hidden(idx, layer=-1))
