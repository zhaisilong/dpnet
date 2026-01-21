# Drug Property Net (DPNet)

Drug Property Net (DPNet) is a lightweight framework for preprocessing and managing small-molecule property prediction tasks. It focuses on building clean, reproducible datasets with scaffold-aware splitting, and is designed to serve as a data layer for downstream machine learning and deep learning models.

DPNet is model-agnostic and particularly useful for ADMET, toxicity, and general molecular property prediction benchmarks.

- Features
  - SMILES canonicalization and deduplication
  - Murcko scaffold generation
  - Strict scaffold-based train/valid/test splitting
  - Unified task and label metadata (TaskMeta, LabelMeta)
  - Command-line interface and Python API
  - Support for binary, multiclass, and regression tasks

## Installation

```bash
mamba create -n dpnet python=3.12
mamba activate dpnet

pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
  --index-url https://download.pytorch.org/whl/cu126

git clone https://github.com/zhaisilong/dpnet
cd dpnet
pip install -e ".[dev]"
```

## Data Layout

- DPNet expects datasets under the database/ directory:

```bash
database/
└── bbbp/
    ├── raw/
    │   └── bbbp.csv
    ├── task_meta.json
    └── processed/**
```

## Usage

Run preprocessing with a single command:

```bash
dpnet process bbbp
dpnet process avian_tox
```

This performs SMILES standardization, deduplication, scaffold generation, strict scaffold splitting, and exports processed CSV files.

### Python API Example

```bash
from dpnet.core import DPNet

dpnet = DPNet("database/bbbp/processed/bbbp")
train_data = dpnet.datasets["train"]
```

Each sample is returned as a dictionary containing SMILES and labels.

Examples

See `./example.ipynb` for an end-to-end usage example.
