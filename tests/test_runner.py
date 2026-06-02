import json

import pandas as pd
from click.testing import CliRunner

from dpnet.main import cli
from dpnet.runner import BaselineRunConfig, run_baseline


def write_task(task_dir, task_name, problem_type, values):
    task_dir.mkdir(parents=True)
    splits = {
        "train": [
            ("cid_0", "CCO", values[0]),
            ("cid_1", "CCN", values[1]),
            ("cid_2", "CCC", values[2]),
            ("cid_3", "CCCl", values[3]),
            ("cid_4", "COC", values[4]),
            ("cid_5", "CNC", values[5]),
        ],
        "valid": [
            ("cid_6", "CCBr", values[6]),
            ("cid_7", "CCF", values[7]),
        ],
        "test": [
            ("cid_8", "CC=O", values[8]),
            ("cid_9", "CC#N", values[9]),
        ],
    }
    for split, rows in splits.items():
        pd.DataFrame(rows, columns=["cid", "smiles", task_name]).to_csv(
            task_dir / f"{split}.csv",
            index=False,
        )

    num_classes = 2 if problem_type == "binary" else None
    (task_dir / f"{task_name}.json").write_text(
        json.dumps(
            {
                "name": task_name,
                "version": 1,
                "dialect": "dpnet",
                "processed_dir": "processed",
                "id_col": None,
                "smiles_col": "smiles",
                "strict_test": True,
                "labels": [
                    {
                        "id": task_name,
                        "label_col": "label",
                        "problem_type": problem_type,
                        "num_classes": num_classes,
                    }
                ],
                "seed": 42,
                "extra_cols": [],
            }
        )
    )


def test_run_baseline_binary_classification(tmp_path):
    task_name = "toy_class"
    task_dir = tmp_path / task_name
    write_task(task_dir, task_name, "binary", [0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

    result = run_baseline(
        BaselineRunConfig(
            task=task_name,
            task_dir=task_dir,
            output_dir=tmp_path / "runs",
            n_estimators=5,
            n_jobs=1,
        )
    )

    metrics = json.loads(result.metrics_path.read_text())
    assert metrics["task"] == task_name
    assert metrics["labels"][task_name]["problem_type"] == "binary"
    assert (result.run_dir / "models" / f"{task_name}.joblib").is_file()
    predictions = pd.read_csv(result.run_dir / "predictions" / "test.csv")
    assert f"pred_{task_name}" in predictions.columns
    assert f"prob_{task_name}_0" in predictions.columns


def test_run_baseline_regression(tmp_path):
    task_name = "toy_reg"
    task_dir = tmp_path / task_name
    write_task(
        task_dir,
        task_name,
        "regression",
        [0.1, 0.5, 0.9, 1.2, 1.5, 1.8, 0.3, 1.0, 0.7, 1.6],
    )

    result = run_baseline(
        BaselineRunConfig(
            task=task_name,
            task_dir=task_dir,
            output_dir=tmp_path / "runs",
            n_estimators=5,
            n_jobs=1,
        )
    )

    metrics = json.loads(result.metrics_path.read_text())
    assert metrics["labels"][task_name]["problem_type"] == "regression"
    assert "rmse" in metrics["labels"][task_name]["splits"]["test"]
    predictions = pd.read_csv(result.run_dir / "predictions" / "valid.csv")
    assert f"pred_{task_name}" in predictions.columns


def test_run_cli(tmp_path):
    task_name = "toy_cli"
    task_dir = tmp_path / task_name
    write_task(task_dir, task_name, "binary", [0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

    result = CliRunner().invoke(
        cli,
        [
            "run",
            task_name,
            "--task-dir",
            str(task_dir),
            "--output-dir",
            str(tmp_path / "runs"),
            "--n-estimators",
            "5",
            "--n-jobs",
            "1",
        ],
    )

    assert result.exit_code == 0, result.output
    assert (tmp_path / "runs" / task_name / "rf" / "metrics.json").is_file()
