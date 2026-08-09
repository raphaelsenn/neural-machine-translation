from typing import Any

import torch
import torch.nn as nn

from neuralmt.transformer.mhd import MultiHeadAttention
from neuralmt.transformer.ffn import FFNSwiGLU


class EncoderLayer(nn.Module):
    def __init__(
            self, 
            num_heads: int, 
            d_model: int,
            max_seq_len: int,
            ff_dim: int,
            dk_dim: int, 
            dv_dim: int,
            p_dropout: float,
            dtype: Any | None = None
    ) -> None:
        super().__init__()

        # I use pre-norm, instead of post-norm, read more here:
        # On Layer Normalization in the Transformer Architecture,
        # https://arxiv.org/abs/2002.04745

        self.norm_1 = nn.RMSNorm(d_model, dtype=dtype)
        self.mhd = MultiHeadAttention(d_model, num_heads, max_seq_len, dk_dim, dv_dim, p_dropout, True, dtype)
        self.dropout_1 = nn.Dropout(p_dropout)

        self.norm_2 = nn.RMSNorm(d_model, dtype=dtype)
        self.ffn = FFNSwiGLU(d_model, ff_dim, dtype)
        self.dropout_2 = nn.Dropout(p_dropout)

    def forward(self, src: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        h1 = self.norm_1(src)
        h1 = self.mhd(h1, h1, h1, mask=mask)
        h1 = self.dropout_1(h1) + src

        h2 = self.norm_2(h1)
        h2 = self.ffn(h2)
        h2 = self.dropout_2(h2) + h1

        return h2


class Encoder(nn.Module):
    def __init__(
            self, 
            num_layers: int,
            num_heads: int,
            d_model: int,
            max_seq_len: int,
            ff_dim: int,
            dk_dim: int, 
            dv_dim: int,
            p_dropout: float,
            dtype: Any | None = None
    ) -> None:
        super().__init__()

        self.layers = nn.ModuleList([
            EncoderLayer(num_heads, d_model, max_seq_len, ff_dim, dk_dim, dv_dim, p_dropout, dtype)
            for _ in range(num_layers)
        ])
        self.norm_out = nn.RMSNorm(d_model, dtype=dtype)


    def forward(self, src_emb: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        # src_emb:                              # [B, L, d_model]
        # mask (padding mask):                  # [B, L, L]
        for layer in self.layers:
            src_emb = layer(src_emb, mask)      # [B, L, d_model]
        return self.norm_out(src_emb)           # [B, L, d_model]
