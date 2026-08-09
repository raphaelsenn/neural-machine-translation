"""
This implementation is based on openAI's gpt2 encoder.py:
https://github.com/openai/gpt-2/blob/master/src/encoder.py,
and Andrej Karpathy's implementation of bpe in minGPT:
https://github.com/karpathy/minGPT/blob/master/mingpt/bpe.py
"""
from typing import Dict, Tuple, List
from collections import defaultdict, Counter
from collections.abc import Iterable

import pickle
import time
import regex as re


# Original GPT2 regex pattern, taken from openAI's gpt2 encoder.py:
# https://github.com/openai/gpt-2/blob/master/src/encoder.py
GPT2_SPLIT_PATTERN = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

# Special tokens (padding, begin of sequence, end of sequence)
SPECIAL_TOKENS = ["<PAD>", "<BOS>", "<EOS>"]

INFINITY = 2**64


def get_stats(ids: list[int]) -> Dict[Tuple[int, int], int]:
    """Count consecutive pairs in a list of integers using a dict."""
    stats = defaultdict(int)
    for i in range(1, len(ids)):
        pair = (ids[i - 1], ids[i])
        stats[pair] += 1
    return stats


class BPETokenizer:
    """Byte Pair Encoding Tokenizer (UTF-8 encoding)."""
    def __init__(self, max_vocab_size: int, verbose: bool = True) -> None:
        if max_vocab_size < 256 + len(SPECIAL_TOKENS):
            raise ValueError(f"Vocabulary size is to small.")
        self.max_vocab_size = max_vocab_size
        self.verbose = verbose

        self.first_special_idx = max_vocab_size - len(SPECIAL_TOKENS)
        self.special_tokens = {token: self.first_special_idx + i for i, token in enumerate(SPECIAL_TOKENS)}

        self.merge_rules = {}               # Learnable parameter of BPE
        self.next_idx = 256

        # Vocabulary; assume max_vocab_size = 37_000:
        # 0 - 255 raw byte tokens
        # 256 - 36,996 learned bpe tokens
        # 36,997 - 36,999 special tokens
        self.vocab = self._build_vocab()    # Learnable parameter of BPE

    def __len__(self) -> int:
        return len(self.vocab)

    def encode(self, text: str) -> List[int]:
        tokens = re.findall(GPT2_SPLIT_PATTERN, text)
        output = []
        for token in tokens:
            token_bytes = token.encode("utf-8")
            token_idx = list(token_bytes)
            while len(token_idx) >= 2:
                stats = get_stats(token_idx)
                pair = min(stats, key = lambda pair: self.merge_rules.get(pair, INFINITY)) 
                if pair not in self.merge_rules:
                    break
                token_idx = self._merge(token_idx, pair, self.merge_rules[pair])
            output.extend(token_idx)
        return output

    def decode(self, tokens: List[int], skip_special_tokens: bool = False) -> str:
        # tokens_bytes = bytes([self.vocab[token] for token in tokens])
        tokens_bytes = b"".join(self.vocab[token] for token in tokens)
        text = tokens_bytes.decode("utf-8", errors="replace")
        if skip_special_tokens is True:
            for special_token in self.special_tokens.keys():
                text = text.replace(special_token, "")
        return text

    def train(self, texts: Iterable[str]) -> None:
        # Flat representation of all byte-chunks (very inefficient - but educational :))
        # E.g. texts = ["Hello world!"]
        # =>   re.findall(GPT2_SPLIT_PATTERN..., texts[0]) = ['Hello', ' world', '!']
        # =>   byte_chunks = [[72, 101, 108, 108, 111], [32, 119, 111, 114, 108, 100], [33]]
        byte_chunks = [
            list(chunk.encode("utf-8")) 
            for text in texts 
            for chunk in re.findall(GPT2_SPLIT_PATTERN, text)
        ]

        while self.next_idx < self.first_special_idx:
            # Track time
            start_time = time.monotonic()
            # Maps byte-pairs to thier frecuency count
            global_chunk_stats: Dict[Tuple[int, int], int] = Counter()

            # Count most frequent byte pairs
            for chunk in byte_chunks:
                global_chunk_stats.update(get_stats(chunk))

            if not global_chunk_stats:
                break

            # Get most common byte-pair
            pair, _ = global_chunk_stats.most_common(1)[0]

            # Apply merge rule
            self.merge_rules[pair] = self.next_idx
            byte_chunks = [self._merge(chunk, pair, self.next_idx) for chunk in byte_chunks]
            self.next_idx += 1

            if self.verbose is True:
                elapsed = time.monotonic() - start_time
                completed_merges = self.next_idx - 256
                total_merges = self.first_special_idx - 256
                progress = 100 * (completed_merges / total_merges)
                report = (
                    f"Merge {completed_merges:,} / {total_merges:,} ({progress:.2f}%) | "
                    f"vocab size {self.next_idx + len(SPECIAL_TOKENS)} | "
                    f"elapsed {elapsed:.1f}s"
                )
                print(report)

        self.vocab = self._build_vocab()

    @property
    def bos_id(self) -> int:
        return self.special_tokens["<BOS>"]

    @property
    def eos_id(self) -> int:
        return self.special_tokens["<EOS>"]

    @property
    def pad_id(self) -> int:
        return self.special_tokens["<PAD>"]

    def save(self, path: str) -> None:
        with open(path, "wb") as file:
            pickle.dump(self, file)

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        with open(path, "rb") as file:
            return pickle.load(file)

    def _merge(self, indices: list[int], pair: Tuple[int, int], index: int) -> List[int]:
        merged_indices = []
        i = 0
        while i < len(indices) - 1:
            if indices[i] == pair[0] and indices[i + 1] == pair[1]:
                merged_indices.append(index)
                i += 2
            else:
                merged_indices.append(indices[i])
                i += 1

        if i < len(indices):
            merged_indices.append(indices[i])
        return merged_indices

    def _build_vocab(self) -> Dict[int, bytes]:
        vocab = {idx : bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in self.merge_rules.items():
            vocab[idx] = vocab[p0] + vocab[p1]
        for special, idx in self.special_tokens.items():
            vocab[idx] = special.encode("utf-8")
        return vocab
