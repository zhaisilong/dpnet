# DPNet Documentation

DPNet is a lightweight data and baseline framework for small-molecule property
prediction tasks. It focuses on reproducible preprocessing, scaffold-aware,
random, and perimeter splitting, and simple machine learning baselines for
ADMET, toxicity, and related molecular property benchmarks.

## Quick Start

Install the project in editable mode:

```bash
git clone https://github.com/zhaisilong/dpnet
cd dpnet
pip install -e ".[dev]"
```

Process a dataset task:

```bash
dpnet process bbbp
```

Run the Random Forest baseline:

```bash
dpnet run bbbp --model rf
```

## Documentation

- [Data Format](data-format.md): task metadata, raw CSV files, and processed
  split layout.
- [Splitting Methods](splitting.md): scaffold, random, and perimeter split
  functions, classification stratification, and split method selection used by
  `dpnet process`.

## Main Outputs

Preprocessing writes processed split files:

```text
database/<category>/<task>/processed/<task>/
|-- <task>.json
|-- train.csv
|-- valid.csv
`-- test.csv
```

Baseline runs write model artifacts:

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
