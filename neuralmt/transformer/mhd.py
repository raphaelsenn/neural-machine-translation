from typing import Any

import torch
import torch.nn as nn

from neuralmt.transformer.rope import RoPE


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention.

    Reference:
    ----------
    Attention Is All You Need,
    https://arxiv.org/abs/1706.03762 
    """
    def __init__(
            self, 
            d_model: int,
            num_heads: int,
            dk_dim: int,
            dv_dim: int,
            dtype: Any | None = None
    ) -> None:
        super().__init__()

        self.d_model = d_model
        self.dk_dim = dk_dim
        self.dv_dim = dv_dim
        self.num_heads = num_heads

        self.q_layer = nn.Linear(d_model, dk_dim * num_heads, bias=False, dtype=dtype)
        self.k_layer = nn.Linear(d_model, dk_dim * num_heads, bias=False, dtype=dtype)
        self.v_layer = nn.Linear(d_model, dv_dim * num_heads, bias=False, dtype=dtype)
        self.mhd_layer = nn.Linear(num_heads * dv_dim, d_model, bias=False, dtype=dtype)

        self.rope = RoPE(dk_dim, dtype=dtype)

    def forward(
            self, 
            query: torch.Tensor, 
            key: torch.Tensor, 
            value: torch.Tensor,
            mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        # A somewhat "semi-efficient" implementation of multi-head attention :D
        # Let B := batch_size, Lq := seq_len of q and Lc := seq_len of k and v.
        B, Lq, _ = query.shape
        _, Lc, _ = key.shape

        Q = self.q_layer(query)                             # [B, Lq, num_heads * dk]
        K = self.k_layer(key)                               # [B, Lc, num_heads * dk]
        V = self.v_layer(value)                             # [B, Lc, num_heads * dv]

        Q = Q.reshape(B, Lq, self.num_heads, self.dk_dim)   # [B, Lq, num_heads, dk]
        Q = Q.transpose(1, 2)                               # [B, num_heads, Lq, dk]
        Q = self.rope(Q)

        K = K.view(B, Lc, self.num_heads, self.dk_dim)      # [B, Lc, num_heads, dk]
        K = K.transpose(1, 2)                               # [B, num_heads, Lc, dk]
        K = self.rope(K)

        V = V.view(B, Lc, self.num_heads, self.dv_dim)     # [B, Lc, num_heads, dv]
        V = V.transpose(1, 2)                              # [B, num_heads, Lc, dk]

        # Computing attention for every head
        # [B, num_heads, Lq, dk] @ [B, num_heads, dk, Lc]
        logits = Q @ K.transpose(-2, -1)                   # [B, num_heads, Lq, Lc]

        if mask is not None:
            logits = logits.masked_fill(mask == 0, float("-inf"))

        # Scaled dot-product attention
        scale = self.dk_dim**(-0.5)
        probs = torch.softmax(scale * logits, dim=-1)# [B, num_heads, Lq, Lc]
        attention = probs @ V                               # [B, num_heads, Lq, dv]

        attention = attention.transpose(1, 2).contiguous()  # [B, Lq, num_heads, dv]
        attention = attention.view(
            B, Lq, self.num_heads * self.dv_dim
        )                                                   # [B, Lq, num_heads * dv]

        # Computing multihead attention
        mhd = self.mhd_layer(attention)                     # [B, Lq, d_model]
        return mhd
