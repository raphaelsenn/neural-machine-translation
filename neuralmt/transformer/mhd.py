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
            max_seq_len: int,
            dk_dim: int,
            dv_dim: int,
            p_dropout: float,
            rope: bool = False,
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
        self.out_proj = nn.Linear(num_heads * dv_dim, d_model, bias=False, dtype=dtype)

        self.dropout = nn.Dropout(p_dropout)
        self.rope = RoPE(dk_dim, max_seq_len, dtype) if rope is True else nn.Identity()

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

        # mask has shape:                                  # [B, 1, Lq, Lc]
        if mask is not None:
            logits = logits + mask                         # [B, num_heads, Lq, Lc]

        # Scaled dot-product attention
        scale = self.dk_dim**(-0.5)
        attn_weight = torch.softmax(scale * logits, dim=-1)     # [B, num_heads, Lq, Lc]
        attn_weight = self.dropout(attn_weight)
        attn_weight = attn_weight @ V                           # [B, num_heads, Lq, dv]

        attn_weight = attn_weight.transpose(1, 2).contiguous()  # [B, Lq, num_heads, dv]
        attn_weight = attn_weight.view(
            B, Lq, self.num_heads * self.dv_dim
        )                                                       # [B, Lq, num_heads * dv]

        # Computing multihead attention
        mhd = self.out_proj(attn_weight)                        # [B, Lq, d_model]
        return mhd
