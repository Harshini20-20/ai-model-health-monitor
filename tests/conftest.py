import copy

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synthetic_churn_df() -> pd.DataFrame:
    """Deterministic, imbalanced dataset shaped like the Telco churn data."""
    rng = np.random.default_rng(0)
    n = 300

    tenure = rng.integers(0, 72, n)
    monthly = rng.uniform(20, 110, n).round(2)
    contract = rng.choice(["Month-to-month", "One year", "Two year"], n, p=[0.5, 0.3, 0.2])
    payment = rng.choice(["Electronic check", "Mailed check", "Bank transfer"], n)
    senior = rng.integers(0, 2, n)

    churn_probability = np.clip(
        0.05 + 0.45 * (contract == "Month-to-month") + 0.2 * (tenure < 12),
        0,
        0.9,
    )
    churn = np.where(rng.random(n) < churn_probability, "Yes", "No")

    total_charges = pd.Series((monthly * tenure).round(2).astype(str))
    total_charges.iloc[::25] = " "  # blank values, as in the real dataset

    return pd.DataFrame({
        "customerID": [f"ID-{i:04d}" for i in range(n)],
        "SeniorCitizen": senior,
        "tenure": tenure,
        "MonthlyCharges": monthly,
        "TotalCharges": total_charges,
        "Contract": contract,
        "PaymentMethod": payment,
        "Churn": churn,
    })


@pytest.fixture
def test_config() -> dict:
    """Small in-memory configuration with a fast search space for unit tests."""
    return {
        "data": {
            "dataset": {
                "name": "synthetic",
                "source": "local",
                "path": "data/raw/synthetic.csv",
                "format": "csv",
            },
            "target_column": "Churn",
            "positive_class": "Yes",
            "negative_class": "No",
            "id_columns": ["customerID"],
            "numerical_columns": ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"],
            "split": {"test_size": 0.2, "random_state": 42, "stratify": True},
            "validation": {
                "check_missing_values": True,
                "check_duplicates": True,
                "reject_missing_target": True,
            },
            "processed": {
                "train_path": "data/processed/train.csv",
                "test_path": "data/processed/test.csv",
            },
        },
        "training": {
            "random_state": 42,
            "cv_folds": 3,
            "n_jobs": 1,
            "imbalance_strategy": "balanced",
            "scoring": ["accuracy", "precision", "recall", "f1", "roc_auc"],
            "selection_metric": "roc_auc",
            "artifacts": {
                "model_dir": "models",
                "model_version": "test",
                "reports_dir": "reports",
            },
            "models": {
                "logistic_regression": {"params": {"max_iter": 1000}, "param_grid": {"C": [1.0]}},
                "random_forest": {"params": {"n_estimators": 20}, "param_grid": {"max_depth": [3]}},
                "gradient_boosting": {
                    "params": {},
                    "param_grid": {"n_estimators": [20], "max_depth": [2]},
                },
            },
        },
    }


@pytest.fixture
def raw_root(tmp_path, synthetic_churn_df, test_config):
    """Write the synthetic dataset to a temporary project root and return that root."""
    raw_path = tmp_path / test_config["data"]["dataset"]["path"]
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    synthetic_churn_df.to_csv(raw_path, index=False)
    return tmp_path


@pytest.fixture
def config_copy(test_config) -> dict:
    """A deep copy that individual tests can modify safely."""
    return copy.deepcopy(test_config)
