import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from torchmetrics.text import BLEUScore
from torchmetrics.text import Perplexity

from neuralmt.bpe.bpe import BPETokenizer
from neuralmt.training.utils import create_causal_mask, create_pad_mask


@torch.no_grad()
def validate(
        model: nn.Module,
        tokenizer: BPETokenizer,
        dataloader: DataLoader,
        device: torch.device
) -> float:
    model.eval()

    pad_id = tokenizer.pad_id
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)
    perplexity = Perplexity(ignore_index=pad_id).to(device)

    n_tokens = 0
    cross_entropy = 0.0

    for batch in dataloader:
        src = batch["src"].to(device)  # [B, L]
        tgt = batch["tgt"].to(device)  # [B, S]

        # Teacher forcing
        # https://en.wikipedia.org/wiki/Teacher_forcing
        # https://machinelearningmastery.com/teacher-forcing-for-recurrent-neural-networks/
        tgt_input = tgt[:, :-1]             # [B, S - 1], [y1, y2, y3, ..., yS-1]
        tgt_output = tgt[:, 1:]             # [B, S - 1], [y2, y3, y4, ..., yS-1, yS]

        # Masking (valid_value = 0.0, mask_value = -float("inf"))
        # Padding masks for multi-head attention
        src_mask = create_pad_mask(src, pad_id)                                 # [B, 1, 1, L]
        tgt_input_pad_mask = create_pad_mask(tgt_input, pad_id)                 # [B, 1, 1, S - 1]

        # Decoder causal mask for self-attention
        tgt_input_causal_mask = create_causal_mask(tgt_input.size(1), device)   # [B, 1, S - 1, S - 1] 
        # [B, 1, 1, S - 1] + [B, 1, S - 1, S - 1]
        tgt_input_mask = tgt_input_pad_mask + tgt_input_causal_mask             # [B, 1, S - 1, S - 1]

        # Forward propagation
        logits = model(src, tgt_input, src_mask, tgt_input_mask)                # [B, S - 1, C]
        perplexity.update(logits, tgt_output)

        B, S_minus_1, C = logits.shape 
        logits = logits.reshape(B * S_minus_1, C)                               # [B * (S - 1), C]
        tgt_out = tgt_output.reshape(B * S_minus_1)                             # [B * (S - 1)]

        # Compute loss
        loss = criterion(logits, tgt_out)
        valid_tokens = int((tgt_out != pad_id).sum().item())
        cross_entropy += loss * valid_tokens
        n_tokens += valid_tokens

    dataset_cross_entropy = float(cross_entropy / n_tokens)
    dataset_peplexity_score = float(perplexity.compute().item())

    return dataset_cross_entropy, dataset_peplexity_score


@torch.no_grad()
def validate_with_bleu(
        model: nn.Module,
        tokenizer: BPETokenizer,
        dataloader: DataLoader,
        device: torch.device
) -> tuple[float, float]:
    model.eval()

    bos_id = tokenizer.bos_id
    eos_id = tokenizer.eos_id
    pad_id = tokenizer.pad_id

    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)
    bleu = BLEUScore()

    n_tokens = 0
    cross_entropy = 0.0

    for batch in dataloader:
        src = batch["src"].to(device)  # [B, L]
        tgt = batch["tgt"].to(device)  # [B, S]

        # Teacher forcing
        # https://en.wikipedia.org/wiki/Teacher_forcing
        # https://machinelearningmastery.com/teacher-forcing-for-recurrent-neural-networks/
        tgt_input = tgt[:, :-1]             # [B, S - 1], [y1, y2, y3, ..., yS-1]
        tgt_output = tgt[:, 1:]             # [B, S - 1], [y2, y3, y4, ..., yS-1, yS]

        # Masking (valid_value = 0.0, mask_value = -float("inf"))
        # Padding masks for multi-head attention
        src_mask = create_pad_mask(src, pad_id)                                 # [B, 1, 1, L]
        tgt_input_pad_mask = create_pad_mask(tgt_input, pad_id)                 # [B, 1, 1, S - 1]

        # Decoder causal mask for self-attention
        tgt_input_causal_mask = create_causal_mask(tgt_input.size(1), device)   # [B, 1, S - 1, S - 1] 
        # [B, 1, 1, S - 1] + [B, 1, S - 1, S - 1]
        tgt_input_mask = tgt_input_pad_mask + tgt_input_causal_mask             # [B, 1, S - 1, S - 1]

        # Forward propagation
        logits = model(src, tgt_input, src_mask, tgt_input_mask)                # [B, S - 1, C]
        B, S_minus_1, C = logits.shape 
        logits = logits.reshape(B * S_minus_1, C)                               # [B * (S - 1), C]
        tgt_out = tgt_output.reshape(B * S_minus_1)                             # [B * (S - 1)]

        # Compute loss
        loss = criterion(logits, tgt_out)
        valid_tokens = int((tgt_out != pad_id).sum().item())
        cross_entropy += loss * valid_tokens
        n_tokens += valid_tokens

        # BLEU score
        # https://lightning.ai/docs/torchmetrics/stable/text/bleu_score.html
        tokens = model.generate(src, bos_id, eos_id, pad_id)    # [B, generated_length]
        tokens = tokens.cpu().tolist()
        targets = tgt.cpu().tolist()

        pred_texts, target_texts = [], []
        for pred_tokens, target_tokens in zip(tokens, targets):
            pred_texts.append(tokenizer.decode(pred_tokens, skip_special_tokens=True))
            target_texts.append([tokenizer.decode(target_tokens, skip_special_tokens=True)])
        bleu.update(pred_texts, target_texts)

    dataset_cross_entropy = float(cross_entropy / n_tokens)
    bleu_score = 100.0 * float(bleu.compute().item())
    return dataset_cross_entropy, bleu_score
