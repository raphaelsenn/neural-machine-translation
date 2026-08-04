import os


# WMT14 dataset can be downloaded here:
# https://www.kaggle.com/datasets/mohamedlotfy50/wmt-2014-english-german
WMT14_ROOT = "./wmt14"
WMT14_TRAIN_FILE = "wmt14_translate_de-en_train.csv"
WMT14_VAL_FILE = "wmt14_translate_de-en_validation.csv"
WMT14_TEST_FILE = "wmt14_translate_de-en_test.csv"


class TestWMT14:
    def test_dir_exists(self) -> None:
        assert os.path.isdir(WMT14_ROOT), "WMT14 folder does not exist."

    def test_train_file_exists(self) -> None:
        path = os.path.join(WMT14_ROOT, WMT14_TRAIN_FILE)
        assert os.path.isfile(path), "WMT14 train file does not exist."

    def test_val_file_exists(self) -> None:
        path = os.path.join(WMT14_ROOT, WMT14_VAL_FILE)
        assert os.path.isfile(path), "WMT14 validation file does not exist."

    def test_test_file_exists(self) -> None:
        path = os.path.join(WMT14_ROOT, WMT14_TEST_FILE)
        assert os.path.isfile(path), "WMT14 test file does not exist."
