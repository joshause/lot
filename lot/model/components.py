"""
CUDA-friendly, batch-first, vectorised Kuramoto step.
Single autograd node → no Python loop over heads.
"""
import math
import torch
import torch.nn.functional as F
from torch import Tensor


@torch.jit.script
def kuramoto_step(
    phase: Tensor,               # (H,)
    intrinsic_freq: Tensor,      # (H,)
    coupling: Tensor,            # (H,)
    nbr_idx: Tensor,             # (H, 8)  int64
    nbr_w: Tensor,               # (H, 8)
    dt: float = 0.01,
) -> Tensor:
    """One Euler step for all oscillators in parallel."""
    H = phase.size(0)
    valid = nbr_idx.ge(0)                                      # (H, 8)
    idx = nbr_idx.clamp(min=0)                                 # (H, 8)
    nbr_phase = phase[idx]                                     # (H, 8)
    diff = nbr_phase - phase.unsqueeze(1)                      # (H, 8)
    force = (nbr_w * torch.sin(diff) * valid).sum(1)           # (H,)
    new_phase = phase + dt * (intrinsic_freq + coupling * force)
    return new_phase.remainder(2 * math.pi)


def make_causal_mask(seq_len: int, device: torch.device) -> Tensor:
    return torch.triu(torch.full((seq_len, seq_len), -1e4, device=device), diagonal=1)