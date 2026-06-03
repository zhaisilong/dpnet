# Drug Property Net (DPNet)

Drug Property Net (DPNet) is a lightweight framework for small-molecule property
prediction tasks. It focuses on reproducible preprocessing, scaffold-aware data
splitting, task metadata, and simple machine learning baselines.

DPNet is designed for ADMET, toxicity, and general molecular property prediction
benchmarks where clean split files and repeatable baselines matter.

## Features

- SMILES canonicalization and deduplication.
- RDKit Murcko scaffold generation.
- Scaffold-aware, random, and perimeter train/valid/test splitting.
- Optional label stratification for classification splits.
- Unified task and label metadata with `TaskMeta` and `LabelMeta`.
- CLI and Python API for processed task data.
- Random Forest baseline on RDKit Morgan fingerprints.
- Support for binary, multiclass, and regression labels.

## Installation

```bash
mamba create -n dpnet python=3.12
mamba activate dpnet

git clone https://github.com/zhaisilong/dpnet
cd dpnet
pip install -e ".[dev]"
```

If your environment needs a specific PyTorch build, install it before installing
DPNet. For example:

```bash
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
  --index-url https://download.pytorch.org/whl/cu126
```

## Quick Start

Preprocess a task:

```bash
dpnet process bbbp
```

Run the Random Forest baseline:

```bash
dpnet run bbbp --model rf
```

Load processed data in Python:

```python
from dpnet import DPNet

dpnet = DPNet("database/distribution/bbbp/processed/bbbp")
train_data = dpnet.datasets["train"]
sample = train_data[0]
```

## Data Layout

DPNet expects task directories under `database/`. Tasks may be nested by
category.

```text
database/<category>/<task>/
|-- raw/
|   `-- <task>.csv
|-- task_meta.json
`-- processed/
    `-- <task>/
        |-- <task>.json
        |-- train.csv
        |-- valid.csv
        `-- test.csv
```

See [docs/data-format.md](docs/data-format.md) for metadata fields and processed
split conventions.

## Splitting

`dpnet process` canonicalizes SMILES, removes duplicates, generates Murcko
scaffolds, and exports processed train/valid/test CSV files. The default split
method is scaffold split, controlled by `strict_test` in each task metadata
file. Use `split_method` in `task_meta.json` or `--split-method` to request
`scaffold`, `random`, or `perimeter`.

Perimeter split selects molecules near the boundary of Morgan fingerprint
space. It is limited to 10000 samples because it computes pairwise molecular
distances. If `dpnet process` requests perimeter split for a larger processed
dataset, DPNet logs a warning and falls back to scaffold split.

For classification tasks, `dpnet process` automatically uses the first
non-regression label as the split stratification column when feasible.

See [docs/splitting.md](docs/splitting.md) for strict scaffold split,
non-strict scaffold split, random split helpers, perimeter split, 5-fold
helpers, and stratification fallback behavior.

## Baseline Outputs

RF baseline runs write artifacts under:

```text
runs/<task>/rf/
|-- config.json
|-- metrics.json
|-- models/<label_id>.joblib
`-- predictions/
    |-- train.csv
    |-- valid.csv
    `-- test.csv
```

Use `--task-dir` to point directly at a processed task directory:

```bash
dpnet run bbbp \
  --task-dir database/distribution/bbbp/processed/bbbp \
  --n-estimators 500 \
  --n-jobs 4
```

## Documentation

- [docs/index.md](docs/index.md)
- [docs/data-format.md](docs/data-format.md)
- [docs/splitting.md](docs/splitting.md)
- [CHANGELOG.md](CHANGELOG.md)

## Development

Run tests:

```bash
python -m pytest -q
```

Example notebook:

```text
example.ipynb
```
