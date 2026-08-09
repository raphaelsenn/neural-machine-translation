from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedForwardNetwork(nn.Module):
    """Feed Forward Network (1-layer MLP with ReLU units)."""
    def __init__(
            self, 
            d_model: int, 
            ff_dim: int, 
            dtype: Any | None = None
    ) -> None:
        super().__init__()

        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_dim, dtype=dtype),
            nn.ReLU(),
            nn.Linear(ff_dim, d_model, dtype=dtype),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ffn(x)


class FFNSwiGLU(nn.Module):
    """SwiGLU Feed Forward Network."""
    def __init__(
            self, 
            d_model: int, 
            ff_dim: int, 
            dtype: Any | None = None
    ) -> None:
        super().__init__()

        # Read more here:
        # Glu variants improve transformer, 
        # https://arxiv.org/pdf/2002.05202

        self.gate = nn.Linear(d_model, ff_dim, dtype=dtype)
        self.down = nn.Linear(d_model, ff_dim, dtype=dtype)
        self.up = nn.Linear(ff_dim, d_model, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.silu(self.gate(x)) * self.down(x)
        x = self.up(x)
        return x
