from typing import Dict
import os

import pandas as pd

import torch
from torch.utils.data import Dataset
from tokenizers import Tokenizer

# Special tokens (HuggingFace tokenizer)
PAD_TOKEN = 0
BOS_TOKEN = 1
EOS_TOKEN = 2

PAD = "<PAD>"
BOS = "<BOS>"
EOS = "<EOS>"


class WMT14Dataset(Dataset):
    """WMT14 german to englisch (or vice versa) dataset."""
    def __init__(
            self,
            root: str,
            tokenizer_file: str = "wmt14_bpe.json",
            language_pair: tuple[str, str] = ("en", "de")   # "en" -> "de"
    ) -> None:
        super().__init__()

        if not os.path.exists(root):
            raise FileNotFoundError(f"File {root} does not exist.")
        if not os.path.exists(tokenizer_file):
            raise FileNotFoundError(f"File {tokenizer_file} does not exist.")

        self.root = root
        self.tokenizer = Tokenizer.from_file(tokenizer_file)
        self.dataframe = pd.read_csv(root)
        self.language_pair = language_pair
        self._preprocess_data()

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        src, tgt = self.src[index], self.tgt[index]
        return {"src": src, "tgt": tgt} 

    def _preprocess_data(self) -> None:
        self.dataframe = self.dataframe.dropna()

        src, tgt = self.language_pair
        src = BOS + self.dataframe[src].astype(str) + EOS
        tgt = BOS + self.dataframe[tgt].astype(str) + EOS

        src_enc = self.tokenizer.encode_batch(src)
        tgt_enc = self.tokenizer.encode_batch(tgt)

        # 4M * 30 * 64 bit = 4M * 30 * 8 bytes = 960 MB per language
        # German + English = 1.92 GB
        self.src = [torch.as_tensor(enc.ids, dtype=torch.int64) for enc in src_enc]
        self.tgt = [torch.as_tensor(enc.ids, dtype=torch.int64) for enc in tgt_enc]
