"""
Parameter-matched vanilla attention baseline.
Same qkv projection shapes, same dropout, same out-proj.
Only difference: no oscillator buffers/parameters.
"""
import torch.nn as nn
from torch import Tensor
from typing import Optional, Tuple, Dict


class VanillaMultiHeadAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0, bias: bool = True):
        super().__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True, bias=bias
        )

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tuple[Tensor, Dict[str, Tensor]]:
        # mask is additive (T,T)
        out, attn_weights = self.mha(x, x, x, attn_mask=mask, average_attn_weights=False)
        return out, {"attn": attn_weights}