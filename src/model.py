"""FORGE model: Llama-style decoder-only Transformer.

Design (from research/architectures.md):
- Pre-norm RMSNorm, RoPE, SwiGLU, Grouped-Query Attention (GQA)
- Tied embeddings below ~1B params
- Config-driven: FORGE-30M (smoke), FORGE-125M, FORGE-350M

This file is import-safe without torch for config/param-count checks.
Training requires PyTorch (CPU wheel for smoke tests, CUDA on Kaggle).
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class ForgeConfig:
    vocab_size: int = 16384
    d_model: int = 768
    n_layers: int = 12
    n_heads: int = 12       # query heads
    n_kv_heads: int = 3     # KV heads (GQA factor = n_heads // n_kv_heads)
    max_seq_len: int = 2048
    dropout: float = 0.0
    rope_theta: float = 10000.0

    @property
    def head_dim(self) -> int:
        assert self.d_model % self.n_heads == 0
        return self.d_model // self.n_heads

    @property
    def intermediate_size(self) -> int:
        # SwiGLU: ~8/3 * d_model, rounded to multiple of 256 (Llama convention)
        raw = int(8 * self.d_model / 3)
        return ((raw + 255) // 256) * 256

    def param_count(self) -> dict:
        """Exact parameter count by arithmetic (no torch needed)."""
        d = self.d_model
        V = self.vocab_size
        L = self.n_layers
        hd = self.head_dim
        kv_dim = self.n_kv_heads * hd
        inter = self.intermediate_size

        emb = V * d  # tied: counted once
        per_layer = (
            d * d            # q proj
            + d * kv_dim     # k proj
            + d * kv_dim     # v proj
            + d * d          # o proj
            + d * inter * 2  # SwiGLU gate + up
            + inter * d      # SwiGLU down
            + 2 * d          # 2x RMSNorm weights
        )
        final_norm = d
        total = emb + L * per_layer + final_norm
        return {
            "embedding": emb,
            "per_layer": per_layer,
            "total": total,
            "total_M": round(total / 1e6, 2),
            "non_embedding_M": round((total - emb) / 1e6, 2),
            "intermediate_size": inter,
            "head_dim": hd,
            "kv_dim": kv_dim,
        }


# Presets from feasibility.md §3 — layer counts tuned so arithmetic
# param counts actually match the labels (verified by __main__ below).
FORGE_30M = ForgeConfig(vocab_size=8192, d_model=512, n_layers=8, n_heads=8, n_kv_heads=2, max_seq_len=1024)
FORGE_125M = ForgeConfig(vocab_size=16384, d_model=768, n_layers=18, n_heads=12, n_kv_heads=3, max_seq_len=2048)
FORGE_350M = ForgeConfig(vocab_size=32768, d_model=1024, n_layers=28, n_heads=16, n_kv_heads=4, max_seq_len=2048)


def gguf_size_mb(n_params: int, bits: float = 4.5) -> float:
    """Rough GGUF file size: params * bits/8 + ~5% overhead. Q4_K_M ~4.5 bpw effective."""
    return n_params * bits / 8 / 1e6 * 1.05


def kv_cache_mb(cfg: ForgeConfig, ctx: int = 2048, bytes_per_elem: int = 2) -> float:
    """KV cache for batch=1: 2 (K+V) * layers * ctx * kv_dim * bytes."""
    return 2 * cfg.n_layers * ctx * (cfg.n_kv_heads * cfg.head_dim) * bytes_per_elem / 1e6


# --- PyTorch implementation (only defined when torch is available) ---
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class RMSNorm(nn.Module):
        def __init__(self, dim: int, eps: float = 1e-6):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(dim))
            self.eps = eps

        def forward(self, x):
            # Numerics: sum-of-squares in fp16 overflows easily (max 65504),
            # so normalize in fp32 and cast back. For fp32 inputs this is a
            # no-op (x.float() is x), so the FP32 baseline is bit-identical.
            xf = x.float()
            var = xf.pow(2).mean(dim=-1, keepdim=True)
            return (xf * torch.rsqrt(var + self.eps) * self.weight).to(x.dtype)

    def _rope_freqs(head_dim: int, max_len: int, theta: float, device):
        inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
        t = torch.arange(max_len, device=device).float()
        freqs = torch.outer(t, inv)
        return torch.cos(freqs), torch.sin(freqs)

    def _apply_rope(x, cos, sin):
        # x: (B, H, T, D)
        x1, x2 = x[..., 0::2], x[..., 1::2]
        cos = cos[: x.shape[2]].unsqueeze(0).unsqueeze(0)
        sin = sin[: x.shape[2]].unsqueeze(0).unsqueeze(0)
        out1 = x1 * cos - x2 * sin
        out2 = x1 * sin + x2 * cos
        out = torch.stack([out1, out2], dim=-1).flatten(-2)
        return out

    class Block(nn.Module):
        def __init__(self, cfg: ForgeConfig):
            super().__init__()
            self.cfg = cfg
            d = cfg.d_model
            self.attn_norm = RMSNorm(d)
            self.ffn_norm = RMSNorm(d)
            self.q = nn.Linear(d, cfg.n_heads * cfg.head_dim, bias=False)
            self.k = nn.Linear(d, cfg.n_kv_heads * cfg.head_dim, bias=False)
            self.v = nn.Linear(d, cfg.n_kv_heads * cfg.head_dim, bias=False)
            self.o = nn.Linear(d, d, bias=False)
            inter = cfg.intermediate_size
            self.gate = nn.Linear(d, inter, bias=False)
            self.up = nn.Linear(d, inter, bias=False)
            self.down = nn.Linear(inter, d, bias=False)
            self.drop = nn.Dropout(cfg.dropout)

        def forward(self, x, cos, sin):
            B, T, _ = x.shape
            h = self.attn_norm(x)
            q = self.q(h).view(B, T, self.cfg.n_heads, self.cfg.head_dim).transpose(1, 2)
            k = self.k(h).view(B, T, self.cfg.n_kv_heads, self.cfg.head_dim).transpose(1, 2)
            v = self.v(h).view(B, T, self.cfg.n_kv_heads, self.cfg.head_dim).transpose(1, 2)
            q = _apply_rope(q, cos, sin)
            k = _apply_rope(k, cos, sin)
            # Expand KV for GQA
            rep = self.cfg.n_heads // self.cfg.n_kv_heads
            if rep > 1:
                k = k.repeat_interleave(rep, dim=1)
                v = v.repeat_interleave(rep, dim=1)
            attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            attn = attn.transpose(1, 2).contiguous().view(B, T, -1)
            x = x + self.drop(self.o(attn))
            h2 = self.ffn_norm(x)
            x = x + self.drop(self.down(F.silu(self.gate(h2)) * self.up(h2)))
            return x

    class ForgeModel(nn.Module):
        def __init__(self, cfg: ForgeConfig):
            super().__init__()
            self.cfg = cfg
            self.emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
            self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
            self.norm = RMSNorm(cfg.d_model)
            self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
            self.lm_head.weight = self.emb.weight  # tied embeddings
            self.register_buffer("rope_cos", torch.empty(1), persistent=False)
            self.register_buffer("rope_sin", torch.empty(1), persistent=False)

        def forward(self, idx, targets=None):
            B, T = idx.shape
            if T > self.cfg.max_seq_len:
                raise ValueError(
                    f"sequence length {T} exceeds model max_seq_len "
                    f"{self.cfg.max_seq_len}; increase max_seq_len or shorten input")
            cos, sin = _rope_freqs(self.cfg.head_dim, self.cfg.max_seq_len, self.cfg.rope_theta, idx.device)
            x = self.emb(idx)
            for blk in self.blocks:
                x = blk(x, cos, sin)
            x = self.norm(x)
            logits = self.lm_head(x)
            loss = None
            if targets is not None:
                # reshape (not view): targets may be a non-contiguous slice
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                       targets.reshape(-1))
            return logits, loss

        @torch.no_grad()
        def generate(self, idx, max_new: int = 64, temperature: float = 0.0):
            self.eval()
            for _ in range(max_new):
                window = idx[:, -self.cfg.max_seq_len:]
                logits, _ = self(window)
                last = logits[:, -1, :]
                if temperature == 0.0:
                    nxt = last.argmax(dim=-1, keepdim=True)
                else:
                    probs = F.softmax(last / temperature, dim=-1)
                    nxt = torch.multinomial(probs, 1)
                idx = torch.cat([idx, nxt], dim=1)
            return idx

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


if __name__ == "__main__":
    for name, cfg in [("FORGE-30M", FORGE_30M), ("FORGE-125M", FORGE_125M), ("FORGE-350M", FORGE_350M)]:
        p = cfg.param_count()
        print(f"{name}: total={p['total_M']}M non-emb={p['non_embedding_M']}M inter={p['intermediate_size']} "
              f"Q4~{gguf_size_mb(p['total']):.0f}MB KV@2k={kv_cache_mb(cfg):.0f}MB")
    print(f"torch available: {HAS_TORCH}")
