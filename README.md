# CGG

This is the official PyTorch implementation for the paper "CGG: Category-aware global graph contrastive learning for session-based recommendation".

## Requirements

- Python 3.8
- CUDA 10.2
- PyTorch 1.10.1
- DGL 0.8.0
- NumPy 1.22.1
- Pandas 1.4.1

## Datasets

The preprocessed datasets are included in the folder called `datasets` (e.g. datasets/yoochoose1_64/train.txt).

You can also download the raw datasets and run data pre-processing script to obtain the preprocessed dataset for model training.

- [Tmall](https://tianchi.aliyun.com/dataset/42?t=1694360823083)
- [Yoochoose](https://www.kaggle.com/chadgostopp/recsys-challenge-2015)
- [Cosmetics-purchase](https://www.kaggle.com/mkechinov/ecommerce-events-history-in-cosmetics-shop)

## Usage

```bash
python run.py --model CGG --dataset DATASET
```
