import torch
from torch.utils.data import Subset, DataLoader

from neuralmt.dataset.wmt14 import WMT14Dataset
from neuralmt.training.trainer import NMTTrainer
from neuralmt.transformer.nmt import NeuralMachineTranslator # Encoder-Decoder Transformer
from neuralmt.dataset.wmt14_collate_fn import wmt14_collate_fn
from neuralmt.evaluation.evaluate import validate


# WMT14 dataset can be downloaded here:
# https://www.kaggle.com/datasets/mohamedlotfy50/wmt-2014-english-german
TRAIN_FILE = "./wmt14/wmt14_translate_de-en_train.csv"
VAL_FILE = "./wmt14/wmt14_translate_de-en_validation.csv"
TEST_FILE = "./wmt14/wmt14_translate_de-en_test.csv"
TOKENIZER_FILE = "wmt14_bpe.json"

# Hyperparameters Transformer 
LANGUAGE_PAIR = ("en", "de")    # English to german
VOCAB_SIZE = 37_000
D_MODEL = 512
NUM_LAYERS = 6
NUM_HEADS = 8
DK_DIM = 64
DV_DIM = 64
FF_DIM = 4096
P_DROPOUT = 0.1

# Hyperparameters Training
BATCH_SIZE = 128
N_EPOCHS = 1000
WARMUP_STEPS = 40
LEARNING_RATE = 1e-5
LR_REDUCTION_FACTOR = 0.5
BETAS = (0.9, 0.98)
EPSILON = 1e-9
SEED = 0
NUM_WORKERS = 4 if torch.cuda.is_available() else 0
DEVICE = "cuda"
DTYPE = torch.float32


def set_seeds(seed: int) -> None:
    torch.manual_seed(seed)


def main() -> None:
    set_seeds(SEED)

    train_set = WMT14Dataset(TRAIN_FILE, TOKENIZER_FILE)
    val_set = WMT14Dataset(TRAIN_FILE, TOKENIZER_FILE)
    test_set = WMT14Dataset(TRAIN_FILE, TOKENIZER_FILE)
    proxy_train_set = Subset(train_set, torch.randperm(len(train_set))[:len(val_set)])

    train_loader = DataLoader(train_set, BATCH_SIZE, collate_fn=wmt14_collate_fn, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_set, BATCH_SIZE, collate_fn=wmt14_collate_fn, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_set, BATCH_SIZE, collate_fn=wmt14_collate_fn, num_workers=NUM_WORKERS)
    proxy_train_loader = DataLoader(proxy_train_set, BATCH_SIZE, collate_fn=wmt14_collate_fn, num_workers=NUM_WORKERS)

    model = NeuralMachineTranslator(
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        d_model=D_MODEL,
        vocab_size=VOCAB_SIZE,
        ff_dim=FF_DIM,
        dk_dim=DK_DIM,
        dv_dim=DV_DIM,
        p_dropout=P_DROPOUT,
        dtype=DTYPE
    )

    trainer = NMTTrainer(
        model=model, 
        n_epochs=N_EPOCHS, 
        learning_rate=LEARNING_RATE,
        lr_reduction_factor=LR_REDUCTION_FACTOR,
        betas=BETAS,
        epsilon=EPSILON,
        device=DEVICE
    )

    trainer.train(train_loader, val_loader, proxy_train_loader)
    test_ce, test_bleu = validate(model, torch.nn.CrossEntropyLoss(), test_loader, DEVICE)
    print(f"Cross-entropy on test set: {test_ce:.4f} | BLEU on test set: {test_bleu:.4f}")

if __name__ == "__main__":
    main()
