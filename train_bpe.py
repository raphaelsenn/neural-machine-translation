"""Train an educational BPE tokenizer on the WMT14 DE–EN dataset."""

from itertools import chain

import pandas as pd

# Hugging face BPE tokenizer
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel as ByteLevelPreTokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.trainers import BpeTrainer

# Naive BPE tokenizer
from neuralmt.bpe.bpe import BPETokenizer

# WMT14 dataset can be downloaded here:
# https://www.kaggle.com/datasets/mohamedlotfy50/wmt-2014-english-german
DATA_FILE = "./wmt14/wmt14_translate_de-en_train.csv"

# Joint vocabulary (DE and EN) of 37000 tokens, similar to "Attention is All You Need":
# https://arxiv.org/abs/1706.03762
USE_NAIVE_TOKENIZER = False     # Set to True is not recommended :D
VOCAB_SIZE = 37_000
VERBOSE = True


def main() -> None:
    """Train and save a shared German–English BPE tokenizer."""
    df = pd.read_csv(DATA_FILE)
    texts = chain(df["de"].dropna().astype(str), df["en"].dropna().astype(str))

    if USE_NAIVE_TOKENIZER is True:     # Naive BPE tokenizer
        bpe = BPETokenizer(VOCAB_SIZE, VERBOSE)
        bpe.train(texts)
        bpe.save("wmt14_bpe.pickle")
    else:                               # Hugging Face BPE tokenizer
        tokenizer = Tokenizer(BPE())
        tokenizer.pre_tokenizer = ByteLevelPreTokenizer(add_prefix_space=False)
        tokenizer.decoder = ByteLevelDecoder()
        trainer = BpeTrainer(vocab_size=VOCAB_SIZE, special_tokens=["<PAD>", "<BOS>", "<EOS>"])
        tokenizer.train_from_iterator(texts, trainer=trainer)
        tokenizer.save("wmt14_bpe.json")    


if __name__ == "__main__":
    main()
