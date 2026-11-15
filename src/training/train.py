"""Phase 2: train, tune, compare and persist churn classifiers.

Run with:  python -m src.training.train

Leakage rules followed here:
- Preprocessing (cleaning, imputation, scaling, encoding) is part of the pipeline,
  so it is fitted inside each cross-validation fold and never sees test data.
- Hyperparameters and the best model are chosen from cross-validation on the
  training set only. The test set is scored once per model, for reporting.
"""

import json
import logging
from datetime import datetime, timezone
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.utils.class_weight import compute_sample_weight

from src.config import load_config
from src.data.cleaning import clean_dataframe
from src.data.loader import PROJECT_ROOT
from src.data.preprocessing import build_preprocessor
from src.logging_config import configure_logging
from src.training.artifacts import save_model_artifact
from src.training.data_preparation import load_processed_datasets
from src.training.evaluation import (
    DECISION_THRESHOLD,
    SUPPORTED_METRICS,
    evaluate_model,
    write_json,
)
from src.training.models import build_candidate_models
from src.tracking.mlflow_setup import validate_tracking_config
from src.tracking.training_runs import log_training_session

logger = logging.getLogger(__name__)


def encode_target(y: pd.Series, positive_class: str, negative_class: str) -> pd.Series:
    """Map the positive class to 1 and the negative class to 0."""
    allowed = {positive_class, negative_class}
    unexpected = set(y.dropna().unique()) - allowed

    if unexpected or y.isna().any():
        raise ValueError(
            f"Target must contain only {sorted(allowed)} without missing values. "
            f"Unexpected: {sorted(unexpected)}"
        )

    return (y == positive_class).astype(int)


def split_features_target(df: pd.DataFrame, data_config: dict) -> tuple[pd.DataFrame, pd.Series]:
    """Separate model inputs from the encoded target, dropping id columns."""
    target_column = data_config["target_column"]
    drop_columns = [target_column, *data_config.get("id_columns", [])]

    X = df.drop(columns=drop_columns)
    y = encode_target(
        df[target_column],
        data_config["positive_class"],
        data_config["negative_class"],
    )

    return X, y


def split_feature_types(X: pd.DataFrame, numerical_columns: list[str]) -> tuple[list[str], list[str]]:
    """Return (numerical, categorical) feature names; every other column is categorical."""
    missing = [column for column in numerical_columns if column not in X.columns]
    if missing:
        raise ValueError(f"Numerical columns missing from features: {missing}")

    categorical_columns = [column for column in X.columns if column not in numerical_columns]
    return list(numerical_columns), categorical_columns


def build_pipeline(
    numerical_features: list[str],
    categorical_features: list[str],
    estimator,
) -> Pipeline:
    """Build the full raw-data-to-prediction pipeline.

    The cleaning step is stateless, so it is safe to apply to any input,
    including single raw rows at serving time.
    """
    cleaning = FunctionTransformer(
        partial(clean_dataframe, numerical_columns=numerical_features)
    )

    return Pipeline(
        steps=[
            ("cleaning", cleaning),
            ("preprocessor", build_preprocessor(numerical_features, categorical_features)),
            ("model", estimator),
        ]
    )


def compute_training_weights(y: pd.Series, strategy: str) -> np.ndarray:
    """Return per-row training weights that counter class imbalance."""
    if strategy == "balanced":
        return compute_sample_weight(class_weight="balanced", y=y)

    if strategy == "none":
        return np.ones(len(y))

    raise ValueError(f"Unsupported imbalance_strategy: '{strategy}'. Use 'balanced' or 'none'.")


def tune_model(
    candidate: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    sample_weight: np.ndarray,
    numerical_features: list[str],
    categorical_features: list[str],
    cv: StratifiedKFold,
    scoring: list[str],
    refit_metric: str,
    n_jobs: int,
) -> dict:
    """Grid-search one candidate with stratified CV on the training data."""
    pipeline = build_pipeline(numerical_features, categorical_features, candidate["estimator"])

    param_grid = {
        f"model__{key}": values for key, values in candidate["param_grid"].items()
    }

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring=scoring,
        refit=refit_metric,
        cv=cv,
        n_jobs=n_jobs,
    )
    search.fit(X_train, y_train, model__sample_weight=sample_weight)

    best = search.best_index_
    results = search.cv_results_

    return {
        "best_params": {
            key.removeprefix("model__"): value for key, value in search.best_params_.items()
        },
        "cv_mean": {metric: float(results[f"mean_test_{metric}"][best]) for metric in scoring},
        "cv_std": {metric: float(results[f"std_test_{metric}"][best]) for metric in scoring},
        "estimator": search.best_estimator_,
    }


def select_best_model(tuned: dict[str, dict], metric: str) -> str:
    """Return the model with the highest mean CV score. Ties go to the first model in config order."""
    return max(tuned, key=lambda name: tuned[name]["cv_mean"][metric])


def build_comparison_table(
    tuned: dict[str, dict],
    test_results: dict[str, dict],
    best_name: str,
    scoring: list[str],
) -> pd.DataFrame:
    """One row per model with CV mean and std, plus test metrics."""
    rows = []

    for name, result in tuned.items():
        row = {
            "model": name,
            "selected": name == best_name,
            "best_params": json.dumps(result["best_params"], sort_keys=True),
        }

        for metric in scoring:
            row[f"cv_{metric}_mean"] = result["cv_mean"][metric]
            row[f"cv_{metric}_std"] = result["cv_std"][metric]

        for metric in scoring:
            row[f"test_{metric}"] = test_results[name]["metrics"][metric]

        rows.append(row)

    return pd.DataFrame(rows)


def _validate_training_config(scoring: list[str], selection_metric: str) -> None:
    unsupported = set(scoring) - set(SUPPORTED_METRICS)
    if unsupported:
        raise ValueError(f"Unsupported scoring metrics: {sorted(unsupported)}")

    if selection_metric not in scoring:
        raise ValueError(f"selection_metric '{selection_metric}' must be listed in scoring.")


def run_training(config: dict, root_dir: Path | None = None) -> dict:
    """Run the full Phase 2 training workflow and save all outputs.

    Outputs:
      models/churn_pipeline_<version>.joblib           fitted pipeline (preprocessing + model)
      models/churn_pipeline_<version>.metadata.json    input schema and selection details
      reports/model_comparison.csv                     one row per model
      reports/evaluation_metrics.json                  full metrics and confusion matrices
    """
    data_config = config["data"]
    training_config = config["training"]

    seed = training_config["random_state"]
    scoring = list(training_config["scoring"])
    selection_metric = training_config["selection_metric"]
    _validate_training_config(scoring, selection_metric)

    tracking_config = config.get("tracking") or {}
    tracking_enabled = bool(tracking_config.get("enabled", False))
    if tracking_enabled:
        validate_tracking_config(tracking_config)

    base_dir = root_dir or PROJECT_ROOT

    train_df, test_df = load_processed_datasets(config, root_dir=base_dir)
    X_train, y_train = split_features_target(train_df, data_config)
    X_test, y_test = split_features_target(test_df, data_config)
    numerical, categorical = split_feature_types(X_train, data_config["numerical_columns"])

    sample_weight = compute_training_weights(y_train, training_config["imbalance_strategy"])

    cv = StratifiedKFold(
        n_splits=training_config["cv_folds"],
        shuffle=True,
        random_state=seed,
    )

    tuned = {}
    for name, candidate in build_candidate_models(training_config).items():
        logger.info("Tuning %s with %d-fold stratified CV", name, cv.get_n_splits())
        tuned[name] = tune_model(
            candidate,
            X_train,
            y_train,
            sample_weight,
            numerical,
            categorical,
            cv,
            scoring,
            selection_metric,
            training_config["n_jobs"],
        )

    best_name = select_best_model(tuned, selection_metric)
    logger.info("Selected %s by mean CV %s", best_name, selection_metric)

    # Test set is scored once per model for reporting only. It played no part in selection.
    test_results = {
        name: evaluate_model(result["estimator"], X_test, y_test)
        for name, result in tuned.items()
    }

    comparison = build_comparison_table(tuned, test_results, best_name, scoring)

    artifacts_config = training_config["artifacts"]
    version = artifacts_config["model_version"]
    reports_dir = base_dir / artifacts_config["reports_dir"]
    model_dir = base_dir / artifacts_config["model_dir"]

    reports_dir.mkdir(parents=True, exist_ok=True)
    comparison_path = reports_dir / "model_comparison.csv"
    comparison.to_csv(comparison_path, index=False)

    best_result = tuned[best_name]
    model_path = save_model_artifact(
        best_result["estimator"],
        metadata={
            "model_name": best_name,
            "model_version": version,
            "trained_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "sklearn_version": sklearn.__version__,
            "random_state": seed,
            "target_column": data_config["target_column"],
            "positive_class": data_config["positive_class"],
            "negative_class": data_config["negative_class"],
            "decision_threshold": DECISION_THRESHOLD,
            "input_columns": list(X_train.columns),
            "numerical_columns": numerical,
            "categorical_columns": categorical,
            "best_params": best_result["best_params"],
            "selection_metric": selection_metric,
            "cv_mean": best_result["cv_mean"],
            "test_metrics": test_results[best_name]["metrics"],
        },
        model_path=model_dir / f"churn_pipeline_{version}.joblib",
    )

    metrics_path = write_json(
        {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "selected_model": best_name,
            "selection": {
                "metric": selection_metric,
                "basis": (
                    f"Highest mean {training_config['cv_folds']}-fold stratified CV "
                    f"{selection_metric} on the training set. The test set was not used."
                ),
                "tie_break": "first model in config order",
            },
            "random_state": seed,
            "imbalance_strategy": training_config["imbalance_strategy"],
            "data": {
                "train_rows": len(X_train),
                "test_rows": len(X_test),
                "train_positive_rate": float(y_train.mean()),
                "test_positive_rate": float(y_test.mean()),
            },
            "models": {
                name: {
                    "best_params": result["best_params"],
                    "cv_mean": result["cv_mean"],
                    "cv_std": result["cv_std"],
                    "test_metrics": test_results[name]["metrics"],
                    "test_confusion_matrix": test_results[name]["confusion_matrix"],
                }
                for name, result in tuned.items()
            },
            "artifact": {
                "model_path": str(model_path.relative_to(base_dir)),
                "metadata_path": str(model_path.with_suffix(".metadata.json").relative_to(base_dir)),
            },
        },
        reports_dir / "evaluation_metrics.json",
    )

    logger.info("Saved model to %s", model_path)
    logger.info("Saved comparison to %s", comparison_path)
    logger.info("Saved metrics to %s", metrics_path)

    summary = {
        "selected_model": best_name,
        "model_path": str(model_path),
        "comparison_path": str(comparison_path),
        "metrics_path": str(metrics_path),
    }

    if tracking_enabled:
        tracking_summary = log_training_session(
            tracking_config,
            base_dir,
            model_version=version,
            run_params={
                "selected_model": best_name,
                "selection_metric": selection_metric,
                "cv_folds": training_config["cv_folds"],
                "random_state": seed,
                "imbalance_strategy": training_config["imbalance_strategy"],
                "decision_threshold": DECISION_THRESHOLD,
                "train_rows": len(X_train),
                "test_rows": len(X_test),
                "train_positive_rate": float(y_train.mean()),
                "test_positive_rate": float(y_test.mean()),
            },
            tuned=tuned,
            test_results=test_results,
            selected_model=best_name,
            scoring=scoring,
            selection_metric=selection_metric,
            best_pipeline=best_result["estimator"],
            reports=[comparison_path, metrics_path],
            metadata_path=model_path.with_suffix(".metadata.json"),
        )
        summary.update(tracking_summary)
        logger.info(
            "Logged MLflow run %s and registered %s version %s",
            tracking_summary["mlflow_run_id"],
            tracking_summary["registered_model_name"],
            tracking_summary["registry_version"],
        )

    return summary


def main() -> None:
    config = load_config()
    configure_logging(config["log_level"])

    summary = run_training(config)
    logger.info("Training complete. Selected model: %s", summary["selected_model"])


if __name__ == "__main__":
    main()
