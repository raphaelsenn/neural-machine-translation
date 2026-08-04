from typing import Any

import torch
import torch.nn as nn


class FeedForwardNetwork(nn.Module):
    """Feed Forward Network (1-layer MLP with ReLU units)."""
    def __init__(
            self, 
            in_dim: int, 
            ff_dim: int, 
            out_dim: int, 
            p_dropout: float, 
            dtype: Any | None = None
    ) -> None:
        super().__init__()

        self.ffn = nn.Sequential(
            nn.Linear(in_dim, ff_dim, dtype=dtype),
            nn.ReLU(),
            nn.Linear(ff_dim, out_dim, dtype=dtype),
            nn.Dropout(p_dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ffn(x)
