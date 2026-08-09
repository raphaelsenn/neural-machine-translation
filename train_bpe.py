"""Train an educational BPE tokenizer on the Multi30k EN to DE dataset."""
from itertools import chain

# Install: pip install datasets
from datasets import load_dataset

# Naive BPE tokenizer :D
from neuralmt.bpe.bpe import BPETokenizer


# Joint vocabulary (DE and EN) of 8000 tokens:
# https://arxiv.org/abs/1706.03762
VOCAB_SIZE = 8_000
VERBOSE = True


def main() -> None:
    """Train and save a shared German to English BPE tokenizer."""
    ds = load_dataset("bentrevett/multi30k")

    # Train the tokenizer
    texts = chain(list(ds["train"]["de"]), list(ds["train"]["en"]))
    bpe = BPETokenizer(VOCAB_SIZE, VERBOSE)
    bpe.train(texts)
    bpe.save("tokenizer_de_en_bpe_8k.pickle")


if __name__ == "__main__":
    main()
