from typing import Any

import torch
import torch.nn as nn


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(x, cos, sin):
    return (x * cos) + (rotate_half(x) * sin)


class RoPE(nn.Module):
    """
    Rotary Positional Embedding
    
    Reference:
    ---------- 
    RoFormer: Enhanced Transformer with Rotary Position Embedding,
    https://arxiv.org/abs/2104.09864
    """
    def __init__(self, d_model: int, max_seq_len: int = 768, dtype: Any | None = None) -> None:
        super().__init__()

        # NOTE: This implementation is based on the implementation from:
        # https://machinelearningmastery.com/building-a-transformer-model-for-language-translation/

        # Read more about Rotary Positional Embedding here:
        # https://arxiv.org/abs/2104.09864 

        N = 10000
        inv_freq = 1. / (N ** (torch.arange(0, d_model, 2).to(dtype) / d_model))
        position = torch.arange(max_seq_len).float()
        inv_freq = torch.cat((inv_freq, inv_freq), dim=-1)
        sinusoid_inp = torch.outer(position, inv_freq)
        self.register_buffer("cos", sinusoid_inp.cos())
        self.register_buffer("sin", sinusoid_inp.sin())
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, num_heads, L, dk]
        seq_len = x.size(-2)

        cos = self.cos[:seq_len].view(1, 1, seq_len, -1)
        sin = self.sin[:seq_len].view(1, 1, seq_len, -1)

        return apply_rotary_pos_emb(x, cos, sin)
