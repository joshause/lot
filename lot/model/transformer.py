"""
Task-agnostic transformer trunk.
Choice of attention module injected from config.
"""
import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional


class TransformerBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attn_type: str = "lattice",  # "lattice" | "vanilla"
        **attn_kwargs,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        if attn_type == "lattice":
            from .lattice import LatticeMultiHeadAttention
            self.attn = LatticeMultiHeadAttention(embed_dim, num_heads, dropout=dropout, **attn_kwargs)
        else:
            from .vanilla import VanillaMultiHeadAttention
            self.attn = VanillaMultiHeadAttention(embed_dim, num_heads, dropout=dropout)
        mlp_hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor, mask: Optional[Tensor] = None):
        h, info = self.attn(self.norm1(x), mask=mask)
        x = x + h
        x = x + self.mlp(self.norm2(x))
        return x, info


class Transformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int,
        embed_dim: int,
        num_heads: int,
        num_layers: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attn_type: str = "lattice",
        **attn_kwargs,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.pos = nn.Parameter(torch.empty(1, max_seq_len, embed_dim))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList(
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout, attn_type, **attn_kwargs)
            for _ in range(num_layers)
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, idx: Tensor, mask: Optional[Tensor] = None):
        """
        idx: (B, T)  token indices
        returns: logits (B, T, V)  and dict with attn/phase tensors
        """
        B, T = idx.shape
        if T > self.pos.size(1):
            # enlarge buffer (once per run)
            new_pos = torch.nn.Parameter(
                torch.empty(1, T, self.embed_dim, device=self.pos.device, dtype=self.pos.dtype)
            )
            torch.nn.init.trunc_normal_(new_pos, std=0.02)
            self.pos = new_pos
        x = self.embed(idx) + self.pos[:, :T, :]
        infos = []
        for blk in self.blocks:
            x, info = blk(x, mask=mask)
            infos.append(info)
        return self.head(self.norm(x)), {"layers": infos}