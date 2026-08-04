import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from torchtext.data.metrics import bleu_score
from tokenizers import Tokenizer

from neuralmt.dataset.wmt14 import (
    BOS_TOKEN,
    EOS_TOKEN,
    PAD_TOKEN,
)


TOKENIZER_FILE = "wmt14_bpe.json"
TOKENIZER = Tokenizer.from_file(TOKENIZER_FILE)


@torch.no_grad()
def validate(
        model: nn.Module,
        criterion: nn.Module,
        dataloader: DataLoader,
        device: torch.device
) -> tuple[float, float]:
    model.eval()

    n_tokens = 0
    cross_entropy = 0.0

    candidate_corpus = []
    reference_corpus = []

    for batch in dataloader:
        src = batch["src"].to(device)                           # [B, Lq]
        tgt = batch["tgt"].to(device)                           # [B, Lc]

        # Padding masks for multi-head attention
        src_pad_mask = batch["src_pad_mask"].to(device)         # [B, Lq]
        tgt_pad_mask = batch["tgt_pad_mask"].to(device)         # [B, Lc]
        src_pad_mask = src_pad_mask.unsqueeze(1).unsqueeze(2)   # [B, 1, 1, Lq]
        tgt_pad_mask = tgt_pad_mask.unsqueeze(1).unsqueeze(2)   # [B, 1, 1, Lc]

        # Teacher forcing
        tgt_input = tgt[:, :-1]                                 # [B, Lc - 1], [y1, y2, y3, ...]
        tgt_output = tgt[:, 1:]                                 # [B, Lc - 1], [y2, y3, y4, ...]
        tgt_input_pad_mask = tgt_pad_mask[:, :, :, :-1]         # [B, 1, 1, Lc - 1]
        logits = model(src, tgt_input, src_pad_mask, tgt_input_pad_mask)

        # Compute loss
        B, L, C = logits.shape
        logits = logits.view(B * L, C).to(torch.float32)
        tgt_output = tgt_output.reshape(B * L)
        ce_loss = criterion(logits, tgt_output)

        # Ignore padding token
        valid_tokens = int((tgt_output != PAD_TOKEN).sum().item())
        cross_entropy += ce_loss.item() * valid_tokens
        n_tokens += valid_tokens

        # Generate complete translations autoregressively.
        # Inspired from:
        # https://machinelearningmastery.com/building-a-transformer-model-for-language-translation/ 
        pred = model.generate(src, BOS_TOKEN, EOS_TOKEN, PAD_TOKEN, tgt.size(1))     # [B, generated_length]
        for pred_tokens, tgt_tokens in zip(pred, tgt):
            pred_text = TOKENIZER.decode(pred_tokens.cpu().tolist(), skip_special_tokens=True)
            tgt_text = TOKENIZER.decode(tgt_tokens.cpu().tolist(), skip_special_tokens=True)
            candidate_corpus.append(pred_text.split())
            reference_corpus.append([tgt_text.split()])

    dataset_cross_entropy = cross_entropy / n_tokens
    dataset_bleu = bleu_score(candidate_corpus, reference_corpus)

    return dataset_cross_entropy, dataset_bleu
