from typing import Any

import torch
import torch.nn as nn

from neuralmt.transformer.encoder import Encoder
from neuralmt.transformer.decoder import Decoder


class NeuralMachineTranslator(nn.Module):
    def __init__(
            self, 
            num_layers: int,
            num_heads: int,
            d_model: int,
            vocab_size: int,
            max_seq_len: int,
            ff_dim: int,
            dk_dim: int, 
            dv_dim: int,
            p_dropout: float,
            dtype: Any | None = None 
    ) -> None:
        super().__init__()

        self.max_seq_len = max_seq_len

        self.embedding = nn.Embedding(vocab_size, d_model, dtype=dtype)
        self.embedding_dropout = nn.Dropout(p_dropout)

        self.encoder = Encoder(num_layers, num_heads, d_model, max_seq_len, ff_dim, dk_dim, dv_dim, p_dropout, dtype)
        self.decoder = Decoder(num_layers, num_heads, d_model, max_seq_len, ff_dim, dk_dim, dv_dim, p_dropout, dtype)

        self.output_layer = nn.Linear(d_model, vocab_size, bias=False, dtype=dtype)
        self.output_layer.weight = self.embedding.weight

    def forward(
            self, 
            src: torch.Tensor, 
            tgt: torch.Tensor, 
            src_mask: torch.Tensor | None = None,
            tgt_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        src_emb = self.embedding(src)                                   # [B, L_src, d_model]
        src_emb = self.embedding_dropout(src_emb)                       # [B, L_src, d_model]
        src_emb = self.encoder(src_emb, src_mask)                       # [B, L_src, d_model]

        tgt_emb = self.embedding(tgt)                                   # [B, L_tgt, d_model]
        tgt_emb = self.embedding_dropout(tgt_emb)                       # [B, L_tgt, d_model]
        tgt_emb = self.decoder(src_emb, tgt_emb, src_mask, tgt_mask)    # [B, L_tgt, d_model]

        return self.output_layer(tgt_emb)                               # [B, L_tgt, vocab_size]

    @torch.no_grad()
    def generate(
            self,
            src: torch.Tensor,
            bos_id: int,
            eos_id: int,
            pad_id: int,
    ) -> torch.Tensor:
        # Src padding mask
        src_pad_mask = torch.fill(src == pad_id, float("-inf")) # [B, L]
        src_pad_mask = src_pad_mask.unsqueeze(1)                # [B, 1, L]
        src_pad_mask = src_pad_mask.unsqueeze(2)                # [B, 1, 1, L]

        # Shape of src: [B, L]
        src_emb = self.embedding(src)                   # [B, L, d_model]
        src_emb = self.embedding_dropout(src_emb)       # [B, L, d_model]
        src_emb = self.encoder(src_emb, src_pad_mask)   # [B, L, d_model]

        # Prepare output tensor
        B = src.shape[0]
        tgt = torch.full((B, 1), bos_id, dtype=torch.int64, device=src.device)
        finished = torch.zeros(B, dtype=torch.bool, device=src.device)

        # Autoregressive next-token prediction
        # Inspired from: 
        # https://machinelearningmastery.com/building-a-transformer-model-for-language-translation/
        # https://github.com/karpathy/minGPT/blob/master/mingpt/model.py 
        for _ in range(self.max_seq_len - 1):
            tgt_emb = self.embedding(tgt)                           # [B, T, d_model]
            tgt_emb = self.embedding_dropout(tgt_emb)               # [B, T, d_model]
            tgt_emb = self.decoder(src_emb, tgt_emb, src_pad_mask)  # [B, T, d_model]
            logits = self.output_layer(tgt_emb)                     # [B, T, vocab_size]
            pred_token = torch.argmax(logits[:, -1, :], dim=-1)     # [B]

            # Add padding for sequences that already generated EOS
            pred_token = torch.where(finished, torch.full_like(pred_token, pad_id), pred_token)

            # Update translation sequence
            tgt = torch.cat((tgt, pred_token.unsqueeze(1)), dim=1)

            # Break if every target seq. has element EOS token.
            finished |= pred_token == eos_id
            if finished.all():
                break

        return tgt
