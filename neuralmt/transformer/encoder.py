from typing import Any

import torch
import torch.nn as nn

from neuralmt.transformer.mhd import MultiHeadAttention
from neuralmt.transformer.ffn import FeedForwardNetwork


class EncoderLayer(nn.Module):
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

        self.mhd = MultiHeadAttention(d_model, num_heads, dk_dim, dv_dim, dtype)
        self.norm_1 = nn.LayerNorm(d_model, dtype=dtype)

        self.ffn = FeedForwardNetwork(d_model, ff_dim, d_model, p_dropout, dtype)
        self.norm_2 = nn.LayerNorm(d_model, dtype=dtype)

    def forward(self, src: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        h1 = self.mhd(src, src, src, mask=mask)
        h1 = self.norm_1(h1 + src)

        h2 = self.ffn(h1)
        h2 = self.norm_2(h1 + h2)

        return h2


class Encoder(nn.Module):
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
            EncoderLayer(num_heads, d_model, ff_dim, dk_dim, dv_dim, p_dropout, dtype)
            for _ in range(num_layers)
        ])

    def forward(self, src_emb: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        for layer in self.layers:
            src_emb = layer(src_emb, mask)                  # [B, Lq, d_model]
        return src_emb
