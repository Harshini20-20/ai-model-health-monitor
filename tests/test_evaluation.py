import json

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from src.training.evaluation import (
    SUPPORTED_METRICS,
    build_confusion_matrix,
    compute_metrics,
    evaluate_model,
    write_json,
)


def test_compute_metrics_matches_hand_calculated_values():
    # tp=1, fp=0, fn=1, tn=2
    y_true = [1, 0, 1, 0]
    y_pred = [1, 0, 0, 0]
    y_score = [0.9, 0.1, 0.4, 0.2]

    metrics = compute_metrics(y_true, y_pred, y_score)

    assert metrics["accuracy"] == pytest.approx(0.75)
    assert metrics["precision"] == pytest.approx(1.0)
    assert metrics["recall"] == pytest.approx(0.5)
    assert metrics["f1"] == pytest.approx(2 / 3)
    assert metrics["roc_auc"] == pytest.approx(1.0)


def test_perfect_predictions_score_one():
    y = [0, 1, 0, 1]

    metrics = compute_metrics(y, y, [0.1, 0.9, 0.2, 0.8])

    assert all(metrics[name] == pytest.approx(1.0) for name in SUPPORTED_METRICS)


def test_precision_is_zero_when_nothing_is_predicted_positive():
    metrics = compute_metrics([1, 0], [0, 0], [0.1, 0.2])

    assert metrics["precision"] == 0.0


def test_confusion_matrix_layout():
    # Actual: 0,0,1,1. Predicted: 0,1,1,1.
    matrix = build_confusion_matrix([0, 0, 1, 1], [0, 1, 1, 1])

    assert matrix == {
        "true_negative": 1,
        "false_positive": 1,
        "false_negative": 0,
        "true_positive": 2,
    }


def test_confusion_matrix_counts_are_python_ints():
    matrix = build_confusion_matrix([0, 1], [0, 1])

    assert all(type(value) is int for value in matrix.values())


def test_evaluate_model_returns_metrics_and_confusion_matrix():
    rng = np.random.default_rng(1)
    X = pd.DataFrame({"feature": rng.normal(size=100)})
    y = pd.Series((X["feature"] > 0).astype(int))
    model = LogisticRegression().fit(X, y)

    result = evaluate_model(model, X, y)

    assert set(result["metrics"]) == set(SUPPORTED_METRICS)
    assert result["metrics"]["roc_auc"] > 0.9
    assert sum(result["confusion_matrix"].values()) == len(y)


def test_write_json_round_trip(tmp_path):
    payload = {"model": "logistic_regression", "roc_auc": 0.84}

    path = write_json(payload, tmp_path / "nested" / "metrics.json")

    assert path.is_file()
    assert json.loads(path.read_text(encoding="utf-8")) == payload
