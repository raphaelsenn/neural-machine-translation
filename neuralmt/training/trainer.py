from typing import Any
import time

import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, LambdaLR, CosineAnnealingLR

from torchmetrics.text.perplexity import Perplexity

from neuralmt.bpe.bpe import BPETokenizer
from neuralmt.training.utils import create_pad_mask, create_causal_mask
from neuralmt.evaluation.evaluate import validate


class NMTTrainer:
    def __init__(
            self,
            model: nn.Module,
            tokenizer: BPETokenizer,
            d_model: int,
            n_epochs: int,
            learning_rate: float,
            warmup_steps: float,
            weight_decay: float,
            betas: tuple[float, float],
            epsilon: float,
            label_smoothing: float,
            clip_grad_norm: float,
            device: str = "cpu",
            dtype: Any | None = None,
            verbose: bool = True
    ) -> None:
        self.device = torch.device(device)

        self.model = model.to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), learning_rate, betas, epsilon, weight_decay)
        self.criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id, label_smoothing=label_smoothing)
        self.perplexity = Perplexity(ignore_index=tokenizer.pad_id).to(self.device)

        def lr_lambda(step_num: int) -> float:
            step_num = max(step_num, 1)
            return d_model**(-0.5) * min(step_num**(-0.5), step_num * (warmup_steps**(-1.5)))
        self.scheduler = LambdaLR(self.optimizer, lr_lambda)

        self.tokenizer = tokenizer
        self.n_epochs = n_epochs
        self.verbose = verbose
        self.params = round(sum(p.numel() for p in model.parameters()) / 1_000_000, 1)
        self.clip_grad_norm = clip_grad_norm

        assert dtype in {None, torch.float16, torch.float32}, f"Invalid dtype."
        self.dtype = torch.float32 if dtype is None else dtype

        self.bos_id = tokenizer.bos_id
        self.eos_id = tokenizer.eos_id
        self.pad_id = tokenizer.pad_id

        self.best_val_score = float("inf")
        self.stats_dict = {"train_loss": [], "val_loss": [], "train_perplexity": [], "val_perplexity": []}

    def train_one_epoch(self, dataloader: DataLoader) -> float:
        self.model.train()

        n_tokens = 0
        total_loss = 0.0
        self.perplexity.reset()

        for batch in dataloader:
            src = batch["src"].to(self.device)  # [B, L]
            tgt = batch["tgt"].to(self.device)  # [B, S]

            # Teacher forcing
            # https://en.wikipedia.org/wiki/Teacher_forcing
            # https://machinelearningmastery.com/teacher-forcing-for-recurrent-neural-networks/
            tgt_input = tgt[:, :-1]             # [B, S - 1], [y1, y2, y3, ..., yS-1]
            tgt_output = tgt[:, 1:]             # [B, S - 1], [y2, y3, y4, ..., yS-1, yS]

            # Masking (valid_value = 0.0, mask_value = -float("inf"))
            # Padding masks for multi-head attention
            src_pad_mask = create_pad_mask(src, self.pad_id)                    # [B, 1, 1, L]
            tgt_input_pad_mask = create_pad_mask(tgt_input, self.pad_id)        # [B, 1, 1, S - 1]

            # Decoder causal mask for self-attention
            tgt_input_causal_mask = create_causal_mask(tgt_input.size(1), self.device) # [B, 1, S - 1, S - 1] 
            # [B, 1, 1, S - 1] + [B, 1, S - 1, S - 1]
            tgt_input_mask = tgt_input_pad_mask + tgt_input_causal_mask                # [B, 1, S - 1, S - 1]

            # Forward propagation and computation of loss w.r.t. model params 
            logits = self.model(src, tgt_input, src_pad_mask, tgt_input_mask)          # [B, S - 1, C]
            self.perplexity.update(logits, tgt_output)

            B, S_minus_1, C = logits.shape
            logits = logits.view(B * S_minus_1, C).to(torch.float32)            # [B * (S - 1), C] 
            tgt_output = tgt_output.reshape(B * S_minus_1)                      # [B * (S - 1)]
            ce_loss = self.criterion(logits, tgt_output)                        # [1]

            # Backpropagation
            self.optimizer.zero_grad()
            ce_loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
            self.optimizer.step()
            self.scheduler.step()

            valid_tokens = int((tgt_output != self.pad_id).sum().item())
            n_tokens += valid_tokens
            total_loss += ce_loss.item() * valid_tokens

        mean_loss = total_loss / n_tokens
        perplexity_score = float(self.perplexity.compute().item())

        return mean_loss, perplexity_score

    def train(
            self, 
            train_loader: DataLoader, 
            val_loader: DataLoader,
    ) -> None:
        for epoch in range(self.n_epochs):
            start_time = time.monotonic()
            train_loss, train_perplexity = self.train_one_epoch(train_loader)
            val_loss, val_perplexity = validate(self.model, self.tokenizer, val_loader, self.device)
            duration = time.monotonic() - start_time

            # Save stats and model
            self.stats_dict["train_loss"].append(train_loss)
            self.stats_dict["train_perplexity"].append(train_perplexity)
            self.stats_dict["val_loss"].append(val_loss)
            self.stats_dict["val_perplexity"].append(val_perplexity)

            if val_perplexity < self.best_val_score:
                self.best_val_score = val_perplexity
                self._save_model()

            if self.verbose is True:
                report = (
                    f"Epoch: {epoch} / {self.n_epochs} | "
                    f"Train Loss: {train_loss:.4f} | "
                    f"Train Perplexity: {train_perplexity:.4f} | "
                    f"Val Loss: {val_loss:.4f} | "
                    f"Val Perplexity: {val_perplexity:.4f} | "
                    f"Time: {duration:.2f}s"
                )
                print(report)
            self._save_report()

    def _save_model(self) -> None:
        save_name_model = f"nmt-{self.params}M-params.pt"
        torch.save(self.model.state_dict(), save_name_model)

    def _save_report(self) -> None:
        save_name_report = f"report-{self.params}M-params.csv"
        pd.DataFrame.from_dict(self.stats_dict).to_csv(save_name_report, index=False)
