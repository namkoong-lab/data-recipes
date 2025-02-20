# Simulator

## Setup Instructions

1. Install the UV package manager:
   ```bash
   # Follow installation instructions at:
   https://docs.astral.sh/uv/getting-started/installation/
   ```

2. Install project dependencies:
   ```bash
   uv sync
   ```

## Inputs

### Features (X)
9 input features are extracted:
1. Token proportions (5 features): [float], all 5 features are between 0 and 1, and sum to 1
   - RedPajamaWikipedia
   - RedPajamaStackExchange 
   - RedPajamaGithub
   - RedPajamaArXiv
   - RedPajamaBook
2. Model architecture (3 features):
   - Model size (in millions) [float]
   - d_model dimension [int]
   - Number of attention heads [int]
3. Training steps [int]

### Targets (y) 
11 target metrics:
- train/CrossEntropyLoss
- eval/RedPajamaCommonCrawl/CrossEntropyLoss
- eval/RedPajamaC4/CrossEntropyLoss  
- eval/RedPajamaWikipedia/CrossEntropyLoss
- eval/RedPajamaStackExchange/CrossEntropyLoss
- eval/RedPajamaGithub/CrossEntropyLoss
- eval/RedPajamaArXiv/CrossEntropyLoss
- eval/RedPajamaBook/CrossEntropyLoss
- eval/downstream/hellaswag_len_norm
- eval/downstream/piqa_len_norm  
- eval/downstream/arc_easy_acc



## Data Processing
The data processing code is located in `data.ipynb`. The processed data splits are stored in the `data` directory.

There are many types of data splits:

1. **Model Size Split**
   - Splits data based on model size >= 1B parameters
   - Train/Test split stored in `X_train_size.npy` and `X_test_size.npy`

2. **Random Split** 
   - Standard 80/20 random train/test split
   - Train/Test split stored in `X_train_random.npy` and `X_test_random.npy`

3. **Step Split**
   - Splits data at step threshold > 15000
   - Train/Test split stored in `X_train_step.npy` and `X_test_step.npy`

4. **Larger than Size Split**
   - Splits data based on model size >= Size threshold
   - e.g. 150M has <= 150M in Train Set, > 150M in Test Set
   - Train/Test split stored in `X_train_size.npy` and `X_test_size.npy`



## Training

The training code is implemented in `train.py`. There are two ways to run training:

1. **Sweep Mode**
   - Configure hyperparameters in `config.yaml`
   - Run sweeps using `uv run sweep.py`

2. **Single Run Mode** 
   - Edit parameters at bottom of `train.py`
   - Run directly with `uv run train.py`

