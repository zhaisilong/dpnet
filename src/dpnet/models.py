from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)

from .parser import LabelMeta


def make_rf_model(
    label: LabelMeta,
    *,
    seed: int,
    n_estimators: int,
    n_jobs: int,
) -> RandomForestClassifier | RandomForestRegressor:
    kwargs: dict[str, Any] = {
        "random_state": seed,
        "n_estimators": n_estimators,
        "n_jobs": n_jobs,
    }
    if label.problem_type == "regression":
        return RandomForestRegressor(**kwargs)
    return RandomForestClassifier(**kwargs)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    if y_true.size == 0:
        return {"rmse": None, "mae": None, "r2": None}

    out = {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": None,
    }
    if y_true.size >= 2:
        out["r2"] = float(r2_score(y_true, y_pred))
    return out


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray | None = None,
) -> dict:
    if y_true.size == 0:
        return {"accuracy": None, "f1_macro": None, "roc_auc": None}

    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "roc_auc": None,
    }

    if y_score is not None and np.unique(y_true).size == 2:
        try:
            out["roc_auc"] = float(roc_auc_score(y_true, y_score))
        except ValueError:
            out["roc_auc"] = None
    return out
