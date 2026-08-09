from typing import Dict

import torch
from torch.utils.data import Dataset

from datasets import load_dataset

from neuralmt.bpe.bpe import BPETokenizer


class Multi30kDataset(Dataset):
    """Multi30K german to englisch (or vice versa) dataset."""
    def __init__(
            self,
            tokenizer: BPETokenizer,
            language_pair: tuple[str, str] = ("en", "de"),   # "en" -> "de",
            split: str = "train"
    ) -> None:
        super().__init__()

        src_lang, tgt_lang = language_pair
        assert all(lang in {"de", "en"} for lang in language_pair) and src_lang != tgt_lang, (
            f"Invalid language pair."
        )
        assert split in {"train", "validation", "test"}, f"Invalid split."

        self.language_pair = language_pair
        self.split = split
        self.tokenizer = tokenizer
        self.bos_id = tokenizer.bos_id
        self.eos_id = tokenizer.eos_id

        dataset = load_dataset("bentrevett/multi30k")
        self.ds = dataset[split]
        self._preprocess_data()

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        index = int(index)

        src = self.src_lst[index]
        tgt = self.tgt_lst[index]

        return {"src": src, "tgt": tgt}

    def _preprocess_data(self) -> None:
        src_lang, tgt_lang = self.language_pair

        src_list, tgt_list = [], []
        for src_text, tgt_text in zip(self.ds[src_lang], self.ds[tgt_lang]):
            src_enc = [self.bos_id] + self.tokenizer.encode(src_text) + [self.eos_id]
            tgt_enc = [self.bos_id] + self.tokenizer.encode(tgt_text) + [self.eos_id]

            src_list.append(torch.as_tensor(src_enc, dtype=torch.int64))
            tgt_list.append(torch.as_tensor(tgt_enc, dtype=torch.int64))

        self.src_lst = src_list
        self.tgt_lst = tgt_list
