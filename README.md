# ASDGC

PyTorch implementation of **Adaptive-Scale Dynamic Graph Learning with Cross-Scale Global Fusion for Non-Stationary Time Series Forecasting**.

This repository contains the model implementation and dataset-specific experiment scripts used in the manuscript submitted to *Data Mining and Knowledge Discovery*.

## Companion repositories

| Research artifact | GitHub repository | Responsibility |
|---|---|---|
| Model code and experiment scripts | [xiuyu88/ASDGC](https://github.com/xiuyu88/ASDGC) | Environment setup, model implementation, training, evaluation, and reproduction commands |
| Benchmark data and source notes | [xiuyu88/ASDGC-Dataset](https://github.com/xiuyu88/ASDGC-Dataset) | Experiment data files, expected directory layout, dataset statistics, and upstream source information |

## Repository structure

```text
ASDGC/
├── README.md
├── layers.py              # RevIN, adaptive scale, attention, graph, and fusion layers
├── net.py                 # ASDGC model definition
├── train.py               # Data loading, training, validation, and testing
├── util.py                # Normalization and evaluation metrics
├── requirements.txt
├── scripts/               # Dataset-specific experiment commands
└── dataset/               # Create locally; obtain files from ASDGC-Dataset
```

Generated checkpoints, logs, and result files are not part of the source repository.

## Environment

The implementation was tested with:

```text
Python 3.9 or later
torch==2.5.1
numpy==1.23.5
```

Create an isolated environment from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate          # Linux/macOS
# .venv\Scripts\Activate.ps1       # Windows PowerShell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

CUDA is used when a valid GPU index is supplied and CUDA is available. Otherwise, the program runs on CPU.

## Dataset preparation

Download and extract the experiment files from the companion data repository:

```text
https://github.com/xiuyu88/ASDGC-Dataset
```

Place the extracted files under the following local structure:

```text
dataset/
├── ETTm1/ETTm1.txt
├── ETTm2/ETTm2.txt
├── exchange_rate/exchange_rate.txt
├── illness/illness.txt
├── weather/weather.txt
├── solar-energy/solar-energy.txt
├── PEMS04/PEMS04.txt
└── PEMS07/PEMS07.txt
```

Each input file is a comma-separated numeric matrix with:

- one time step per row;
- one variable or sensor per column;
- no header row.

The data repository also records the original dataset sources and use notices. Users should cite the original providers in addition to the ASDGC paper.

## Quick start

Run one complete dataset script from the project root:

```bash
bash scripts/exchange_rate.sh
```

The released scripts evaluate forecasting horizons `3`, `6`, `12`, and `24`:

```bash
bash scripts/ETTm1.sh
bash scripts/ETTm2.sh
bash scripts/exchange_rate.sh
bash scripts/illness.sh
bash scripts/weather.sh
bash scripts/solar-energy.sh
bash scripts/PEMS04.sh
bash scripts/PEMS07.sh
```

## Reproduction settings

The common experimental settings are:

- input look-back length: `32`;
- maximum epochs: `100`;
- early-stopping patience: `25`;
- initial learning rate: `1e-3`;
- weight decay: `1e-4`;
- dropout: `0.2`;
- random seed: `2025`;
- chronological train/validation/test split: `60% / 20% / 20%`;
- checkpoint selection criterion: validation Huber loss.

Dataset-specific settings are encoded in the scripts:

| Dataset/run | `norm_type` | Batch size |
|---|---:|---:|
| ETTm1 | `global` | 128 |
| ETTm2 | `revin` | 128 |
| Exchange-Rate | `global_revin` | 128 |
| ILI (`illness`) | `revin` | 32 |
| Weather | `revin` | 128 |
| Solar-Energy | `global` | 32 |
| PEMS07 | `global` | 32 |
| PEMS04, horizons 6/12/24 | `global` | 32 |
| PEMS04, horizon 3 | `revin` | 32 |

Normalization modes:

- `global`: variable-wise standardization fitted on the training range;
- `revin`: reversible instance normalization inside the model;
- `global_revin`: global standardization followed by RevIN.

## Outputs

Each run writes:

```text
save/
├── best_model/            # Best validation checkpoint
└── results/
    ├── *.json             # Configuration, timing, and test metrics
    └── results_summary.txt
```

A timestamped log file is also created in the repository root. Reported metrics are RSE, RAE, and empirical correlation (CORR), consistent with the manuscript.

## Useful arguments

```text
--data DATASET_NAME
--seq_len 32
--pred_len {3,6,12,24}
--num_scales 5
--batch_size 32
--epochs 100
--early_stopping_patience 25
--lr 0.001
--dropout 0.2
--weight_decay 0.0001
--norm_type {global,revin,global_revin}
--gpus 0
--pretrained_model PATH_TO_CHECKPOINT
--resume_training
```

To restore the optimizer, scheduler, and epoch state, pass both `--pretrained_model` and `--resume_training`.

## Citation

Please cite the ASDGC article when its bibliographic information becomes available. When using the benchmark files, also cite the corresponding original dataset papers, websites, or upstream repositories listed in `ASDGC-Dataset`.

## Contact

Use GitHub Issues for reproducibility questions, bug reports, and requests for clarification. Do not upload datasets, credentials, model checkpoints, or other large files to an issue.
