# Neural Machine Translation

## Usage

The WMT14 German to Englisch Machine Translation dataset can be downloaded here:

[WMT 2014 Englisch-German](https://www.kaggle.com/datasets/mohamedlotfy50/wmt-2014-english-german?select=wmt14_translate_de-en_train.csv)


### Inference

Download the tokenizer file from: 

Download the weight of the NMT model from:

```python
tokenizer = Tokenizer.from_file(TOKENIZER_FILE)
nmt = NeuralMachineTranslator()

```

### Training

#### 1. Train Byte Pair Encoding Tokenizer

```bash
python3 train_bpe.py
```

#### 2. Train the Neural-Machine-Translator

```bash
python3 train.py
```

## Citations

```bibtex
@misc{su2023roformerenhancedtransformerrotary,
      title={RoFormer: Enhanced Transformer with Rotary Position Embedding}, 
      author={Jianlin Su and Yu Lu and Shengfeng Pan and Ahmed Murtadha and Bo Wen and Yunfeng Liu},
      year={2023},
      eprint={2104.09864},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2104.09864}, 
}

@misc{vaswani2023attentionneed,
      title={Attention Is All You Need}, 
      author={Ashish Vaswani and Noam Shazeer and Niki Parmar and Jakob Uszkoreit and Llion Jones and Aidan N. Gomez and Lukasz Kaiser and Illia Polosukhin},
      year={2023},
      eprint={1706.03762},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/1706.03762}, 
}
```