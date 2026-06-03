import json

import pandas as pd
import pytest
from click.testing import CliRunner

from dpnet.main import cli
from dpnet.preprocess import (
    perimeter_split_df,
    perimeter_split_df_no_test,
    random_split_df,
    random_split_df_5fold,
    random_split_df_no_test,
    scaffold_split_df,
    scaffold_split_df_5fold,
    scaffold_split_df_no_test,
)


def make_balanced_df(n: int = 40) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "row_id": [f"row_{i}" for i in range(n)],
            "smiles": [f"C{i}" for i in range(n)],
            "scaffold": [f"s{i // 2}" for i in range(n)],
            "label": [i % 2 for i in range(n)],
        }
    )


def make_valid_molecule_df(n: int = 40) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "row_id": [f"row_{i}" for i in range(n)],
            "smiles": ["C" * (i + 1) for i in range(n)],
            "scaffold": [f"s{i // 2}" for i in range(n)],
            "label": [i % 2 for i in range(n)],
        }
    )


def assert_partition(df: pd.DataFrame, *parts: pd.DataFrame) -> None:
    rows = pd.concat(parts, ignore_index=True)
    assert len(rows) == len(df)
    assert set(rows["row_id"]) == set(df["row_id"])
    assert rows["row_id"].is_unique


def assert_has_both_classes(df: pd.DataFrame) -> None:
    assert set(df["label"]) == {0, 1}


def test_random_split_helpers_support_stratify_col():
    df = make_balanced_df()

    splits = random_split_df(
        df,
        valid_size=0.2,
        test_size=0.2,
        stratify_col="label",
        random_state=42,
    )
    assert_partition(df, splits["train"], splits["valid"], splits["test"])
    for split_df in splits.values():
        assert_has_both_classes(split_df)

    no_test = random_split_df_no_test(
        df,
        valid_size=0.2,
        stratify_col="label",
        random_state=42,
    )
    assert_partition(df, no_test["train"], no_test["valid"])
    for split_df in no_test.values():
        assert_has_both_classes(split_df)

    folds = random_split_df_5fold(
        df,
        n_splits=5,
        stratify_col="label",
        random_state=42,
    )
    assert len(folds) == 5
    for fold in folds:
        assert_partition(df, fold["train"], fold["valid"])
        assert_has_both_classes(fold["valid"])


def test_scaffold_split_helpers_accept_stratify_col():
    df = make_balanced_df()

    splits = scaffold_split_df(
        df,
        stratify_col="label",
        valid_size=0.2,
        random_state=42,
        strict=True,
    )
    assert_partition(df, splits["train"], splits["valid"], splits["test"])

    mixed_scaffold_df = df.copy()
    mixed_scaffold_df.loc[0, "scaffold"] = pd.NA
    non_strict = scaffold_split_df(
        mixed_scaffold_df,
        stratify_col="label",
        valid_size=0.2,
        random_state=42,
        strict=False,
    )
    assert_partition(
        mixed_scaffold_df,
        non_strict["train"],
        non_strict["valid"],
        non_strict["test"],
    )

    no_test = scaffold_split_df_no_test(
        df,
        stratify_col="label",
        valid_size=0.2,
        random_state=42,
    )
    assert_partition(df, no_test["train"], no_test["valid"])

    folds = scaffold_split_df_5fold(
        df,
        n_splits=5,
        stratify_col="label",
        random_state=42,
    )
    assert len(folds) == 5
    for fold in folds:
        assert_partition(df, fold["train"], fold["valid"])


def test_scaffold_helpers_handle_all_empty_scaffolds_with_stratify_col():
    df = make_balanced_df()
    df["scaffold"] = pd.NA

    no_test = scaffold_split_df_no_test(
        df,
        stratify_col="label",
        valid_size=0.2,
        random_state=42,
    )
    assert_partition(df, no_test["train"], no_test["valid"])

    folds = scaffold_split_df_5fold(
        df,
        n_splits=5,
        stratify_col="label",
        random_state=42,
    )
    assert len(folds) == 5
    for fold in folds:
        assert_partition(df, fold["train"], fold["valid"])


def test_missing_stratify_col_raises_value_error():
    df = make_balanced_df()

    with pytest.raises(ValueError, match="stratify_col 'missing'"):
        random_split_df_no_test(df, stratify_col="missing")


def test_perimeter_split_helpers_support_stratify_col():
    df = make_valid_molecule_df()

    splits = perimeter_split_df(
        df,
        valid_size=0.2,
        test_size=0.2,
        stratify_col="label",
        random_state=42,
    )
    assert_partition(df, splits["train"], splits["valid"], splits["test"])
    assert_has_both_classes(splits["valid"])
    assert_has_both_classes(splits["test"])

    repeat = perimeter_split_df(
        df,
        valid_size=0.2,
        test_size=0.2,
        stratify_col="label",
        random_state=42,
    )
    assert repeat["test"]["row_id"].tolist() == splits["test"]["row_id"].tolist()

    no_test = perimeter_split_df_no_test(
        df,
        valid_size=0.2,
        stratify_col="label",
        random_state=42,
    )
    assert_partition(df, no_test["train"], no_test["valid"])
    assert_has_both_classes(no_test["valid"])


def test_perimeter_split_enforces_max_samples():
    df = make_valid_molecule_df(12)

    with pytest.raises(ValueError, match="limited to 10 samples"):
        perimeter_split_df(df, max_samples=10)


def test_process_cli_uses_classification_label_for_stratification(tmp_path):
    task = "toy_process"
    task_root = tmp_path / task
    raw_dir = task_root / "raw"
    raw_dir.mkdir(parents=True)

    rows = [{"smiles": "C" * (i + 1), "label": i % 2} for i in range(20)]
    pd.DataFrame(rows).to_csv(raw_dir / f"{task}.csv", index=False)
    (task_root / "task_meta.json").write_text(
        json.dumps(
            {
                "name": task,
                "seed": 42,
                "id_col": None,
                "smiles_col": "smiles",
                "labels": [
                    {
                        "id": task,
                        "label_col": "label",
                        "problem_type": "binary",
                        "num_classes": 2,
                    }
                ],
                "strict_test": True,
                "processed_dir": "processed",
                "extra_cols": [],
            }
        )
    )

    result = CliRunner().invoke(
        cli,
        [
            "process",
            task,
            "task_meta",
            "--root-dir",
            str(task_root),
        ],
    )

    assert result.exit_code == 0, result.output
    processed_dir = task_root / "processed" / task
    assert (processed_dir / "train.csv").is_file()
    assert (processed_dir / "valid.csv").is_file()
    assert (processed_dir / "test.csv").is_file()

    train_df = pd.read_csv(processed_dir / "train.csv")
    assert task in train_df.columns


def test_process_cli_supports_perimeter_split_method(tmp_path):
    task = "toy_perimeter"
    task_root = tmp_path / task
    raw_dir = task_root / "raw"
    raw_dir.mkdir(parents=True)

    rows = [{"smiles": "C" * (i + 1), "label": i % 2} for i in range(24)]
    pd.DataFrame(rows).to_csv(raw_dir / f"{task}.csv", index=False)
    (task_root / "task_meta.json").write_text(
        json.dumps(
            {
                "name": task,
                "seed": 42,
                "id_col": None,
                "smiles_col": "smiles",
                "labels": [
                    {
                        "id": task,
                        "label_col": "label",
                        "problem_type": "binary",
                        "num_classes": 2,
                    }
                ],
                "strict_test": True,
                "processed_dir": "processed",
                "extra_cols": [],
            }
        )
    )

    result = CliRunner().invoke(
        cli,
        [
            "process",
            task,
            "task_meta",
            "--root-dir",
            str(task_root),
            "--split-method",
            "perimeter",
        ],
    )

    assert result.exit_code == 0, result.output
    processed_dir = task_root / "processed" / task
    assert (processed_dir / "train.csv").is_file()
    assert (processed_dir / "valid.csv").is_file()
    assert (processed_dir / "test.csv").is_file()
    processed_meta = json.loads((processed_dir / f"{task}.json").read_text())
    assert processed_meta["split_method"] == "perimeter"


def test_process_cli_falls_back_to_scaffold_when_perimeter_is_too_large(tmp_path):
    task = "toy_perimeter_fallback"
    task_root = tmp_path / task
    raw_dir = task_root / "raw"
    raw_dir.mkdir(parents=True)

    rows = [{"smiles": "C" * (i + 1), "label": i % 2} for i in range(12)]
    pd.DataFrame(rows).to_csv(raw_dir / f"{task}.csv", index=False)
    (task_root / "task_meta.json").write_text(
        json.dumps(
            {
                "name": task,
                "seed": 42,
                "id_col": None,
                "smiles_col": "smiles",
                "labels": [
                    {
                        "id": task,
                        "label_col": "label",
                        "problem_type": "binary",
                        "num_classes": 2,
                    }
                ],
                "strict_test": True,
                "processed_dir": "processed",
                "extra_cols": [],
                "split_method": "perimeter",
                "split_config": {"perimeter_max_samples": 10},
            }
        )
    )

    result = CliRunner().invoke(
        cli,
        [
            "process",
            task,
            "task_meta",
            "--root-dir",
            str(task_root),
        ],
    )

    assert result.exit_code == 0, result.output
    processed_dir = task_root / "processed" / task
    assert (processed_dir / "train.csv").is_file()
    assert (processed_dir / "valid.csv").is_file()
    assert (processed_dir / "test.csv").is_file()
    processed_meta = json.loads((processed_dir / f"{task}.json").read_text())
    assert processed_meta["split_method"] == "scaffold"
