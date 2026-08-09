from typing import Any
import torch


def create_causal_mask(
        seq_len: int, 
        device: torch.device | None = None,
    ) -> torch.Tensor:
    """Output shape: [1, seq_len, seq_len]"""
    # Read more here:
    # https://machinelearningmastery.com/building-a-transformer-model-for-language-translation/ 
    mask = torch.triu(torch.full((1, seq_len, seq_len), float("-inf"), device=device, dtype=torch.float32), diagonal=1)
    return mask


def create_pad_mask(
        sequences: torch.Tensor,
        pad_id: int,
    ) -> torch.Tensor:
    """Input shape: [batch_size, seq_len], output shape: [batch_size, 1, 1, seq_len]"""
    mask = torch.fill(sequences == pad_id, float("-inf"))   # [B, L]
    mask = mask.unsqueeze(1).unsqueeze(2)                   # [B, 1, 1, L]
    return mask
