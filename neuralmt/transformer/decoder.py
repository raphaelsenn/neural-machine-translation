from typing import Any

import torch
import torch.nn as nn

from neuralmt.transformer.mhd import MultiHeadAttention
from neuralmt.transformer.ffn import FFNSwiGLU


class DecoderLayer(nn.Module):
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
        self.mhd_1 = MultiHeadAttention(d_model, num_heads, max_seq_len, dk_dim, dv_dim, p_dropout, True, dtype)
        self.dropout_1 = nn.Dropout(p_dropout)

        self.norm_2 = nn.RMSNorm(d_model, dtype=dtype)
        self.mhd_2 = MultiHeadAttention(d_model, num_heads, max_seq_len, dk_dim, dv_dim, p_dropout, False, dtype)
        self.dropout_2 = nn.Dropout(p_dropout)

        self.norm_3 = nn.RMSNorm(d_model, dtype=dtype)
        self.ffn = FFNSwiGLU(d_model, ff_dim, dtype)
        self.dropout_3 = nn.Dropout(p_dropout)

    def forward(
            self, 
            src: torch.Tensor, 
            tgt: torch.Tensor,
            src_mask: torch.Tensor | None = None,
            tgt_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        h1 = self.norm_1(tgt)
        h1 = self.mhd_1(h1, h1, h1, tgt_mask)
        h1 = self.dropout_1(h1) + tgt

        h2 = self.norm_2(h1)
        h2 = self.mhd_2(h2, src, src, src_mask)
        h2 = self.dropout_2(h2) + h1

        h3 = self.norm_3(h2)
        h3 = self.ffn(h3)
        h3 = self.dropout_3(h3) + h2

        return h3


class Decoder(nn.Module):
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
            DecoderLayer(num_heads, d_model, max_seq_len, ff_dim, dk_dim, dv_dim, p_dropout, dtype)
            for _ in range(num_layers)
        ])
        self.norm_out = nn.RMSNorm(d_model, dtype=dtype)

    def forward(
            self, 
            src_emb: torch.Tensor, 
            tgt_emb: torch.Tensor,
            src_mask: torch.Tensor | None = None,
            tgt_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        for layer in self.layers:
            tgt_emb = layer(src_emb, tgt_emb, src_mask, tgt_mask)         # [B, Lc, d_model]
        return self.norm_out(tgt_emb)
