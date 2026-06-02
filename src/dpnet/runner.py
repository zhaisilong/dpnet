import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from loguru import logger

from .features import MorganFeaturizer, MorganFingerprintConfig
from .models import classification_metrics, make_rf_model, regression_metrics
from .parser import LabelMeta, TaskMeta
from .utils import find_task_root


@dataclass(frozen=True)
class BaselineRunConfig:
    task: str
    model: str = "rf"
    root_dir: Optional[Path | str] = None
    task_dir: Optional[Path | str] = None
    output_dir: Path | str = Path("runs")
    seed: Optional[int] = None
    n_estimators: int = 500
    n_jobs: int = 4
    fp_radius: int = 2
    fp_n_bits: int = 2048


@dataclass(frozen=True)
class BaselineRunResult:
    run_dir: Path
    metrics_path: Path
    config_path: Path


def run_baseline(config: BaselineRunConfig) -> BaselineRunResult:
    if config.model != "rf":
        raise ValueError(f"Unsupported baseline model: {config.model}")

    root_dir = Path(config.root_dir) if config.root_dir is not None else None
    task_dir_config = Path(config.task_dir) if config.task_dir is not None else None
    output_dir = Path(config.output_dir)

    task_dir = resolve_processed_task_dir(
        config.task,
        root_dir=root_dir,
        task_dir=task_dir_config,
    )
    task_meta = load_processed_task_meta(task_dir)
    seed = config.seed if config.seed is not None else task_meta.seed or 42
    splits = load_split_frames(task_dir)

    smiles_col = task_meta.smiles_col or "smiles"
    featurizer = MorganFeaturizer(
        MorganFingerprintConfig(radius=config.fp_radius, n_bits=config.fp_n_bits)
    )
    features = {
        split: featurizer.transform(df[smiles_col].tolist())
        for split, df in splits.items()
    }

    run_dir = output_dir / task_meta.name / config.model
    models_dir = run_dir / "models"
    pred_dir = run_dir / "predictions"
    models_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)

    predictions = {split: df.copy() for split, df in splits.items()}
    metrics: dict = {
        "task": task_meta.name,
        "model": config.model,
        "seed": seed,
        "features": featurizer.config.to_dict(),
        "labels": {},
    }

    for label in task_meta.iter_labels():
        logger.info(f"Training {config.model} baseline for label '{label.id}'")
        label_col = resolve_label_column(label, splits["train"])
        model = make_rf_model(
            label,
            seed=seed,
            n_estimators=config.n_estimators,
            n_jobs=config.n_jobs,
        )

        train_df = splits["train"]
        train_mask = train_df[label_col].notna().to_numpy()
        if not train_mask.any():
            raise ValueError(f"Label '{label.id}' has no train labels")

        y_train = train_df.loc[train_mask, label_col].to_numpy()
        model.fit(features["train"][train_mask], y_train)
        joblib.dump(model, models_dir / f"{label.id}.joblib")

        label_metrics = {"problem_type": label.problem_type, "splits": {}}
        for split, df in splits.items():
            split_pred, split_proba = predict_split(model, features[split], label)
            predictions[split][f"pred_{label.id}"] = split_pred

            if split_proba is not None:
                for class_idx, class_name in enumerate(model.classes_):
                    predictions[split][f"prob_{label.id}_{class_name}"] = split_proba[
                        :, class_idx
                    ]

            label_col_for_split = resolve_label_column(label, df)
            label_metrics["splits"][split] = score_split(
                label,
                df[label_col_for_split].to_numpy(),
                split_pred,
                split_proba,
                getattr(model, "classes_", None),
            )
        metrics["labels"][label.id] = label_metrics

    for split, df in predictions.items():
        df.to_csv(pred_dir / f"{split}.csv", index=False)

    metrics_path = run_dir / "metrics.json"
    metrics_path.write_text(json.dumps(to_jsonable(metrics), indent=2))

    config_path = run_dir / "config.json"
    config_path.write_text(
        json.dumps(
            to_jsonable(
                {
                    **asdict(config),
                    "root_dir": str(root_dir) if root_dir else None,
                    "task_dir": str(task_dir_config) if task_dir_config else None,
                    "output_dir": str(output_dir),
                    "resolved_task_dir": str(task_dir),
                }
            ),
            indent=2,
        )
    )

    return BaselineRunResult(
        run_dir=run_dir,
        metrics_path=metrics_path,
        config_path=config_path,
    )


def resolve_processed_task_dir(
    task: str,
    *,
    root_dir: Path | str | None = None,
    task_dir: Path | str | None = None,
) -> Path:
    if task_dir is not None:
        return require_processed_task_dir(Path(task_dir))

    candidates: list[Path] = []
    if root_dir is not None:
        root = Path(root_dir)
        candidates.extend(
            [
                root,
                root / "processed" / task,
                root / task / "processed" / task,
            ]
        )
        candidates.extend(
            path
            for path in root.rglob(task)
            if path.is_dir()
            and (path / "train.csv").exists()
            and (path / "valid.csv").exists()
            and (path / "test.csv").exists()
        )
    else:
        task_root = find_task_root(task)
        candidates.append(task_root / "processed" / task)

    valid = []
    for candidate in candidates:
        try:
            valid.append(require_processed_task_dir(candidate))
        except FileNotFoundError:
            continue

    unique = sorted(set(valid))
    if len(unique) == 1:
        return unique[0]
    if len(unique) > 1:
        choices = "\n".join(str(path) for path in unique)
        raise RuntimeError(f"Multiple processed task dirs found for '{task}':\n{choices}")

    raise RuntimeError(f"Processed task dir not found for '{task}'")


def require_processed_task_dir(path: Path) -> Path:
    required = ["train.csv", "valid.csv", "test.csv"]
    if all((path / name).is_file() for name in required):
        return path
    raise FileNotFoundError(path)


def load_processed_task_meta(task_dir: Path) -> TaskMeta:
    candidates = [
        task_dir / f"{task_dir.name}.json",
        task_dir.parent / f"{task_dir.name}.json",
    ]
    for path in candidates:
        if path.is_file():
            return TaskMeta.load(path)
    raise FileNotFoundError(f"Task metadata not found for processed dir: {task_dir}")


def load_split_frames(task_dir: Path) -> dict[str, pd.DataFrame]:
    return {
        split: pd.read_csv(task_dir / f"{split}.csv")
        for split in ["train", "valid", "test"]
    }


def resolve_label_column(label: LabelMeta, df: pd.DataFrame) -> str:
    if label.id in df.columns:
        return label.id
    if label.label_col in df.columns:
        return label.label_col
    raise KeyError(f"Label column not found for '{label.id}'")


def predict_split(
    model,
    x: np.ndarray,
    label: LabelMeta,
) -> tuple[np.ndarray, np.ndarray | None]:
    if x.shape[0] == 0:
        return np.array([]), None
    pred = model.predict(x)
    if label.problem_type == "regression" or not hasattr(model, "predict_proba"):
        return pred, None
    return pred, model.predict_proba(x)


def score_split(
    label: LabelMeta,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None,
    classes: np.ndarray | None,
) -> dict:
    mask = pd.notna(y_true)
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if label.problem_type == "regression":
        return regression_metrics(y_true.astype(float), y_pred.astype(float))

    y_score = None
    if y_proba is not None and classes is not None and len(classes) == 2:
        y_score = y_proba[mask, 1]
    return classification_metrics(y_true, y_pred, y_score)


def to_jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_jsonable(v) for v in value]
    return value
