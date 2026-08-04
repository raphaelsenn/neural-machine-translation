from typing import Any

import torch
import torch.nn as nn


class RoPE(nn.Module):
    """
    Rotary Positional Embedding
    
    Reference:
    ---------- 
    RoFormer: Enhanced Transformer with Rotary Position Embedding,
    https://arxiv.org/abs/2104.09864
    """
    def __init__(self, d_model: int, base=10000.0, dtype: Any | None = None):
        super().__init__()
        self.d_model = d_model
        self.dtype = dtype

        theta = 1.0 / (base ** (torch.arange(0, d_model, 2).to(self.dtype) / d_model))
        self.register_buffer("theta", theta)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        L, D = x.shape[-2:]

        # NOTE: This implementation is based on the implementation from:
        # https://wangkuiyi.github.io/rope.html

        # Read more about Rotary Positional Embedding here:
        # https://arxiv.org/abs/2104.09864 
        positions = torch.arange(L, device=x.device, dtype=self.dtype) # [0, 1, 2 ... L]

        angles = torch.outer(positions, self.theta)
        angles = torch.cat([angles, angles], dim=-1)

        # Make them broadcastable to [..., L, D]
        angles = angles.view(*([1] * (x.ndim - 2)), L, D)

        sin = torch.sin(angles)   # Consider cache it. 
        cos = torch.cos(angles)

        xu, xd = x.chunk(2, dim=-1)
        hatx = torch.cat([-xd, xu], dim=-1)

        return x * cos + hatx * sin
