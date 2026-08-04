from neuralmt.bpe import BPETokenizer, get_stats

# WMT14 dataset can be downloaded here:
# https://www.kaggle.com/datasets/mohamedlotfy50/wmt-2014-english-german
DATA_FILE = "./wmt14/wmt14_translate_de-en_val.csv"
VOCAB_SIZE = 512
VERBOSE = False


class TestBPETokenizer:
    def test_encode_decode(self) -> None:
        bpe = BPETokenizer(VOCAB_SIZE, VERBOSE)

        query = "Mein Sternzeichen ist 🦁."
        assert bpe.decode(bpe.encode(query)) == query

        # Taken from: https://en.wikipedia.org/wiki/Computer_science, 04.08.2026
        query = "Computer science is the study of computation, information, and automation."
        assert bpe.decode(bpe.encode(query)) == query

        # Taken from: https://de.wikipedia.org/wiki/Physik, 04.08.2026
        query = (
            "Die Physik ist eine Naturwissenschaft, die grundlegende Phänomene der Natur untersucht. "
            "Um deren Eigenschaften und Verhalten anhand von quantitativen Modellen und Gesetzmäßigkeiten zu erklären, "
            "befasst sie sich insbesondere mit Materie und Energie und deren Wechselwirkungen in Raum und Zeit."
        )
        assert bpe.decode(bpe.encode(query)) == query

    def test_get_stats(self) -> None:
        ids = [1, 2, 1, 2, 1]
        stats = get_stats(ids)
        assert stats == {(1, 2): 2, (2, 1): 2}
    
    def test_merge(self) -> None:
        bpe = BPETokenizer(VOCAB_SIZE, VERBOSE)
        ids = [1, 2, 1, 2, 1]
        merged_ids = bpe._merge(ids, (1, 2), 3)
        assert merged_ids == [3, 3, 1]

    def test_encode(self) -> None:
        bpe = BPETokenizer(VOCAB_SIZE, VERBOSE)

        # Merge "a" and "b" into token 256
        bpe.merge_rules[(ord("a"), ord("b"))] = 256
        bpe.vocab = bpe._build_vocab()

        assert bpe.encode("abc") == [256, ord("c")]

    def test_decode(self) -> None:
        bpe = BPETokenizer(VOCAB_SIZE, VERBOSE)

        # Merge "a" and "b" into token 256
        bpe.merge_rules[(ord("a"), ord("b"))] = 256
        bpe.vocab = bpe._build_vocab()

        assert bpe.decode([256, ord("c")]) == "abc"
