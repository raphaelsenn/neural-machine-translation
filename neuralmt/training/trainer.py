import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from neuralmt.evaluation.evaluate import validate
from neuralmt.dataset.wmt14 import PAD_TOKEN


class NMTTrainer:
    def __init__(
            self,
            model: nn.Module,
            n_epochs: int,
            learning_rate: float,
            lr_reduction_factor: float,
            betas: tuple[float, float],
            epsilon: float,
            device: str = "cpu",
            verbose: bool = True
    ) -> None:
        self.device = torch.device(device)
        self.n_epochs = n_epochs
        self.verbose = verbose
        self.params = round(sum(p.numel() for p in model.parameters()) / 1_000_000, 1)

        self.model = model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), learning_rate, betas, epsilon)
        self.scheduler = ReduceLROnPlateau(self.optimizer, factor=lr_reduction_factor)
        self.criterion = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)

        self.stats_dict = {
            "train_loss": [],   # Full train losses
            "val_loss": [], "val_bleu": [],
            "proxy_train_loss": [], "proxy_train_bleu": []
        }

    def train_one_epoch(self, dataloader: DataLoader) -> float:
        self.model.train()

        n_tokens = 0
        total_loss = 0.0
        for batch in dataloader:
            src = batch["src"].to(self.device)                      # [B, Lq]
            tgt = batch["tgt"].to(self.device)                      # [B, Lc]

            # Padding masks for multi-head attention
            src_pad_mask = batch["src_pad_mask"].to(self.device)    # [B, Lq]
            tgt_pad_mask = batch["tgt_pad_mask"].to(self.device)    # [B, Lc]
            src_pad_mask = src_pad_mask.unsqueeze(1).unsqueeze(2)   # [B, 1, 1, Lq]
            tgt_pad_mask = tgt_pad_mask.unsqueeze(1).unsqueeze(2)   # [B, 1, 1, Lc]

            # Teacher forcing
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]
            tgt_input_pad_mask = tgt_pad_mask[:, :, :, :-1]
            logits = self.model(src, tgt_input, src_pad_mask, tgt_input_pad_mask)

            # Compute loss w.r.t. model params 
            B, L, C = logits.shape
            logits = logits.view(B * L, C).to(torch.float32)
            tgt_output = tgt_output.reshape(B * L)
            ce_loss = self.criterion(logits, tgt_output)

            # Backpropagation
            self.optimizer.zero_grad()
            ce_loss.backward()
            self.optimizer.step()

            valid_tokens = int((tgt_output != PAD_TOKEN).sum().item())
            n_tokens += valid_tokens
            total_loss += ce_loss.item() * valid_tokens

        return total_loss / n_tokens

    def train(
            self, 
            train_loader: DataLoader, 
            val_loader: DataLoader,
            proxy_train_loader: DataLoader
    ) -> None:
        self.proxy_loader = proxy_train_loader
        for epoch in range(self.n_epochs):
            # Train on full dataset (~4.5M samples)
            train_loss = self.train_one_epoch(train_loader)

            # Proxy train and validation set have 3K samples
            val_loss, val_bleu = validate(self.model, self.criterion, val_loader, self.device)
            prxy_train_loss, prxy_train_bleu = validate(self.model, self.criterion, proxy_train_loader, self.device)
            self.scheduler.step(val_loss)

            self._update_stats("proxy_train", prxy_train_loss.item(), prxy_train_bleu.item())
            self._update_stats("train", train_loss.item())
            self._update_stats("val", val_loss.item(), val_bleu.item())

            # Save model and stats
            self._save()

            if self.verbose is True:
                report = (
                    f"Epoch: {epoch} / {self.n_epochs} | "
                    f"Train Loss (full): {train_loss:.4f} | "
                    f"Proxy Train Loss: {prxy_train_loss:.4f}"
                    f"Proxy BLEU: {prxy_train_bleu:.4f}"
                    f"Val Loss: {val_loss:.4f}"
                    f"Val BLEU: {val_bleu:.4f}"
                )
                print(report)

    def _save(self) -> None:
        save_name_model = f"nmt-{self.params}M-params.pth"
        torch.save(self.model.state_dict(), save_name_model)

        save_name_report = f"report-{self.params}M-params.csv"
        pd.DataFrame.from_dict(self.stats_dict).to_csv(save_name_report, index=False)

    def _update_stats(self, key: str, loss: float, bleu: float | None = None) -> None:
        if key == "train":
            self.stats_dict["train_loss"].append(loss)
        elif key == "proxy_train":
            self.stats_dict["proxy_train_loss"].append(loss)
            self.stats_dict["proxy_train_bleu"].append(bleu)
        else:
            self.stats_dict["val_loss"].append(loss)
            self.stats_dict["val_bleu"].append(bleu)
