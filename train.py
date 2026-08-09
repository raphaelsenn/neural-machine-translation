from typing import List, Dict

import torch
from torch.utils.data import DataLoader

from neuralmt import (
    Multi30kDataset,
    BPETokenizer,
    NeuralMachineTranslator,
    NMTTrainer,
    validate_with_bleu
)


# Tokenizers
TOKENIZER_FILE = "tokenizer_de_en_bpe_8k.pickle"
TOKENIZER = BPETokenizer.load(TOKENIZER_FILE)
LANGUAGE_PAIR = ("de", "en")
PAD_TOKEN = TOKENIZER.pad_id     # 7997
BOS_TOKEN = TOKENIZER.bos_id     # 7998
EOS_TOKEN = TOKENIZER.eos_id     # 7999

# Hyperparameters Transformer
# Some transformer hyperparamters from:
# https://arxiv.org/abs/1706.03762
# NOTE: These hyperparameters max out my RTX 3060 TI (8gb VRAM)
VOCAB_SIZE = len(TOKENIZER)     # 8000
MAX_SEQ_LENGTH = 256
D_MODEL = 512
NUM_LAYERS = 6
NUM_HEADS = 8
DK_DIM = 64
DV_DIM = 64
FF_DIM = 1365
P_DROPOUT = 0.5

# Hyperparameters Training
# NOTE: These hyperparameters max out my RTX 3060 TI (8gb VRAM)
BATCH_SIZE = 64
N_EPOCHS = 50
LEARNING_RATE = 0.5
WARMUP_STEPS = 4000
WEIGHT_DECAY = 5e-4
BETAS = (0.9, 0.98)
EPSILON = 1e-9
LABEL_SMOOTHING = 0.1
CLIP_GRAD_NORM = 1.0
SEED = 0
NUM_WORKERS = 4 if torch.cuda.is_available() else 0
DEVICE = "cuda"
DTYPE = torch.float32


def collate_fn(data: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    # https://discuss.pytorch.org/t/how-to-use-collate-fn/27181
    # https://discuss.pytorch.org/t/how-to-create-batches-of-a-list-of-varying-dimension-tensors/50773/14
    batch_size = len(data)
    src_max_len = max(len(sample["src"]) for sample in data)
    tgt_max_len = max(len(sample["tgt"]) for sample in data)

    # Padded sequences
    src_padded = torch.ones((batch_size, src_max_len), dtype=torch.int64) * PAD_TOKEN
    tgt_padded = torch.ones((batch_size, tgt_max_len), dtype=torch.int64) * PAD_TOKEN

    for i in range(len(data)):
        src = data[i]["src"][:src_max_len]
        tgt = data[i]["tgt"][:tgt_max_len]

        src_padded[i, torch.arange(0, len(src))] = src
        tgt_padded[i, torch.arange(0, len(tgt))] = tgt

    return {"src": src_padded, "tgt": tgt_padded}


def create_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """Causal mask for decoder self-attention."""
    mask = torch.triu((seq_len, seq_len), float("-inf"), device=device, dtype=DTYPE)
    return mask


def set_seeds(seed: int) -> None:
    """Set random seed."""
    torch.manual_seed(seed)


def initialize_weights(m: torch.nn.Module) -> None:
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        torch.nn.init.xavier_uniform_(m.weight)


def main() -> None:
    set_seeds(SEED)
    train_set = Multi30kDataset(TOKENIZER, LANGUAGE_PAIR, "train")
    val_set = Multi30kDataset(TOKENIZER, LANGUAGE_PAIR, "validation")
    test_set = Multi30kDataset(TOKENIZER, LANGUAGE_PAIR, "test")

    train_loader = DataLoader(train_set, BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_set, BATCH_SIZE, collate_fn=collate_fn, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_set, BATCH_SIZE, collate_fn=collate_fn, num_workers=NUM_WORKERS)

    model = NeuralMachineTranslator(
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        d_model=D_MODEL,
        vocab_size=VOCAB_SIZE,
        max_seq_len=MAX_SEQ_LENGTH,
        ff_dim=FF_DIM,
        dk_dim=DK_DIM,
        dv_dim=DV_DIM,
        p_dropout=P_DROPOUT,
        dtype=DTYPE
    )
    model.apply(initialize_weights)

    trainer = NMTTrainer(
        model=model, 
        tokenizer=TOKENIZER,
        d_model=D_MODEL,
        n_epochs=N_EPOCHS, 
        learning_rate=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        weight_decay=WEIGHT_DECAY,
        betas=BETAS,
        epsilon=EPSILON,
        label_smoothing=LABEL_SMOOTHING,
        clip_grad_norm=CLIP_GRAD_NORM,
        device=DEVICE
    )

    trainer.train(train_loader, val_loader)
    test_ce, test_bleu = validate_with_bleu(model, TOKENIZER, test_loader, DEVICE)
    print(f"Cross-entropy on test set: {test_ce:.4f} | BLEU score on test set: {test_bleu:.4f}")


if __name__ == "__main__":
    main()
