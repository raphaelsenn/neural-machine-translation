from typing import List, Dict

import torch

from neuralmt.dataset.wmt14 import (
    PAD_TOKEN,
)


def wmt14_collate_fn(data: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    # Inpired from: 
    # https://discuss.pytorch.org/t/how-to-create-batches-of-a-list-of-varying-dimension-tensors/50773/14
    # https://discuss.pytorch.org/t/how-to-use-collate-fn/27181/4 
    batch_size = len(data)
    src_max_length = max(len(sample["src"]) for sample in data)
    tgt_max_length = max(len(sample["tgt"]) for sample in data)

    src_out = torch.ones(size=(batch_size, src_max_length), dtype=torch.int64) * PAD_TOKEN
    tgt_out = torch.ones(size=(batch_size, tgt_max_length), dtype=torch.int64) * PAD_TOKEN

    src_mask = torch.zeros(size=(batch_size, src_max_length), dtype=torch.int64)
    tgt_mask = torch.zeros(size=(batch_size, tgt_max_length), dtype=torch.int64)

    for i in range(len(data)):
        src_len = len(data[i]["src"])
        tgt_len = len(data[i]["tgt"])

        src_out[i, torch.arange(src_len)] = data[i]["src"]
        tgt_out[i, torch.arange(tgt_len)] = data[i]["tgt"]

        src_mask[i, torch.arange(src_len)] = 1 
        tgt_mask[i, torch.arange(tgt_len)] = 1 

    return {"src": src_out, "tgt": tgt_out, "src_pad_mask": src_mask, "tgt_pad_mask": tgt_mask} 
