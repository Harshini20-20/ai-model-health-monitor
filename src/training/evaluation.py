"""Classification metrics, confusion matrices and JSON report writer."""

import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

SUPPORTED_METRICS = ("accuracy", "precision", "recall", "f1", "roc_auc")

# Predictions use scikit-learn's default: the positive class is predicted at p >= 0.5.
DECISION_THRESHOLD = 0.5


def compute_metrics(y_true, y_pred, y_score) -> dict[str, float]:
    """Compute the headline classification metrics for the positive class (1)."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
    }


def build_confusion_matrix(y_true, y_pred) -> dict[str, int]:
    """Return confusion matrix counts for the positive class (1)."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return {
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }


def evaluate_model(model, X: pd.DataFrame, y: pd.Series) -> dict:
    """Score a fitted classifier pipeline on a labelled dataset."""
    y_pred = model.predict(X)
    y_score = model.predict_proba(X)[:, 1]

    return {
        "metrics": compute_metrics(y, y_pred, y_score),
        "confusion_matrix": build_confusion_matrix(y, y_pred),
    }


def write_json(data: dict, path: Path) -> Path:
    """Write a dictionary to a UTF-8 JSON file, creating parent folders if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)

    return path
