from typing import Any

import torch
import torch.nn as nn

from neuralmt.transformer.mhd import MultiHeadAttention
from neuralmt.transformer.ffn import FeedForwardNetwork


class DecoderLayer(nn.Module):
    def __init__(
            self, 
            num_heads: int, 
            d_model: int,
            ff_dim: int,
            dk_dim: int, 
            dv_dim: int,
            p_dropout: float,
            dtype: Any | None = None
    ) -> None:
        super().__init__()

        self.mhd_1 = MultiHeadAttention(d_model, num_heads, dk_dim, dv_dim, dtype)
        self.norm_1 = nn.LayerNorm(d_model, dtype=dtype)

        self.mhd_2 = MultiHeadAttention(d_model, num_heads, dk_dim, dv_dim, dtype)
        self.norm_2 = nn.LayerNorm(d_model, dtype=dtype)

        self.ffn = FeedForwardNetwork(d_model, ff_dim, d_model, p_dropout, dtype)
        self.norm_3 = nn.LayerNorm(d_model, dtype=dtype)

    def forward(
            self, 
            src: torch.Tensor, 
            tgt: torch.Tensor,
            src_mask: torch.Tensor | None = None,
            tgt_mask: torch.Tensor | None = None
    ) -> None:
        h1 = self.mhd_1(tgt, tgt, tgt, tgt_mask)
        h1 = self.norm_1(tgt + h1)

        h2 = self.mhd_2(tgt, src, src, src_mask)
        h2 = self.norm_2(h1 + h2)

        h3 = self.ffn(h2)
        h3 = self.norm_3(h2 + h3)

        return h3


class Decoder(nn.Module):
    def __init__(
            self, 
            num_layers: int,
            num_heads: int,
            d_model: int,
            ff_dim: int,
            dk_dim: int, 
            dv_dim: int,
            p_dropout: float,
            dtype: Any | None = None
    ) -> None:
        super().__init__()

        self.layers = nn.ModuleList([
            DecoderLayer(num_heads, d_model, ff_dim, dk_dim, dv_dim, p_dropout, dtype)
            for _ in range(num_layers)
        ])

    def forward(
            self, 
            src_emb: torch.Tensor, 
            tgt_emb: torch.Tensor,
            src_mask: torch.Tensor | None = None,
            tgt_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        for layer in self.layers:
            tgt_emb = layer(src_emb, tgt_emb, src_mask, tgt_mask)         # [B, Lc, d_model]

        return tgt_emb
