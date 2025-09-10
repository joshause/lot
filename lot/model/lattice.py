import math
from typing import Optional, Tuple, Dict, List
import torch
import torch.nn as nn
from torch import Tensor
from .components import kuramoto_step, make_causal_mask

__all__ = ["LatticeMultiHeadAttention"]

class LatticeMultiHeadAttention(nn.Module):
    """
    Drop-in replacement for nn.MultiheadAttention.
    Extra args:
        lattice_shape: (rows, cols)  must satisfy rows*cols >= n_heads
        init_freq_range: (low, high)  uniform init for intrinsic frequencies
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        lattice_shape: Optional[Tuple[int, int]] = None,
        init_freq_range: Tuple[float, float] = (0.95, 1.05),
        dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        # build lattice
        if lattice_shape is None:
            r = int(math.sqrt(num_heads))
            lattice_shape = (r, r + 1) if r * r < num_heads else (r, r)
        self.lattice_shape = lattice_shape
        self.positions: List[Tuple[int, int]] = []
        for h in range(num_heads):
            row, col = divmod(h, lattice_shape[1])
            self.positions.append((row, col))

        # params
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.dropout = dropout

        # oscillator buffers (registered → included in state_dict)
        self.register_buffer("phase", torch.zeros(num_heads))
        self.register_buffer("intrinsic_freq", torch.empty(num_heads).uniform_(*init_freq_range))
        self.coupling = nn.Parameter(torch.ones(num_heads) * 0.1)  # learnable

        # neighbour topology (fixed)
        self.register_buffer("nbr_idx", torch.full((num_heads, 8), -1, dtype=torch.long))
        self.register_buffer("nbr_w", torch.zeros(num_heads, 8))
        self._build_topology()

    def _build_topology(self):
        for i, (r1, c1) in enumerate(self.positions):
            k = 0
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    if k >= 8:
                        break
                    r2, c2 = r1 + dr, c1 + dc
                    for j, (r, c) in enumerate(self.positions):
                        if (r, c) == (r2, c2):
                            dist = (dr ** 2 + dc ** 2) ** 0.5
                            self.nbr_idx[i, k] = j
                            self.nbr_w[i, k] = math.exp(-dist)
                            k += 1
                            break

    def forward(
        self,
        x: Tensor,  # (B, T, C)
        mask: Optional[Tensor] = None,  # (T, T) additive mask
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, H, T, hd)

        # ----- oscillator update (vectorised) -----
        new_phase = kuramoto_step(
            self.phase, self.intrinsic_freq, self.coupling, self.nbr_idx, self.nbr_w
        )
        with torch.no_grad():
            self.phase.copy_(new_phase)
        # phase bias: (H, 1, 1)  broadcast to (B, H, T, T)
        phase_bias = 0.1 * torch.cos(self.phase).view(self.num_heads, 1, 1)

        # ----- attention -----
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, H, T, T)
        scores = scores + phase_bias
        if mask is not None:
            scores = scores + mask.unsqueeze(0).unsqueeze(0)
        attn = torch.softmax(scores, dim=-1)
        attn = torch.dropout(attn, p=self.dropout, train=self.training)
        out = torch.matmul(attn, v)  # (B, H, T, hd)
        out = out.transpose(1, 2).reshape(B, T, C)
        return self.out_proj(out), {"attn": attn, "phase": self.phase.clone()}