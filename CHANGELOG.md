# Changelog

All notable changes to DPNet are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and versions follow the package version in `pyproject.toml`.

## [Unreleased]

### Added

- Project documentation under `docs/`.
- Detailed documentation for DPNet data layout and splitting methods.
- Optional `stratify_col` support for scaffold, random, and 5-fold split helpers.
- Perimeter split helpers for small-molecule holdout splitting on Morgan
  fingerprints.
- `split_method` and `split_config` task metadata fields for choosing scaffold,
  random, or perimeter preprocessing splits.
- `dpnet process --split-method` for CLI-level split method overrides.

### Changed

- `dpnet process` automatically uses the first classification label for split
  stratification when feasible.
- Perimeter split requests above 10000 processed samples now warn and fall back
  to scaffold split in the CLI.
- Removed the legacy standalone 8:1:1 splitter path from the tracked code and
  documentation.

## [0.1.0] - 2026-06-02

### Added

- `dpnet run` command for package-native machine learning baselines.
- Random Forest baseline support for binary, multiclass, and regression tasks.
- RDKit Morgan fingerprint featurization for ML baselines.
- Baseline output artifacts: metrics, predictions, serialized models, and run config.
- Unit tests for Morgan fingerprint featurization, RF baseline runner, and CLI usage.

### Changed

- Updated package version to `0.1.0`.
- Exported baseline runner APIs from `dpnet`.
- Kept preprocessing imports lazy so ML baseline usage does not require importing preprocessing code at CLI import time.

### Fixed

- Implemented the previously empty `dpnet run` CLI command.

## [0.0.1] - Initial release

### Added

- Initial DPNet package structure.
- Task metadata models for datasets and labels.
- SMILES canonicalization, deduplication, Murcko scaffold generation, and processed split export.
- Basic dataset loader for processed DPNet task directories.
