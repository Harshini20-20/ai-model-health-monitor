import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from src.config import load_config
from src.data.loader import PROJECT_ROOT
from src.training.data_preparation import (
    load_processed_datasets,
    prepare_datasets,
    save_processed_datasets,
)
from src.training.models import build_candidate_models
from src.training.train import (
    build_pipeline,
    compute_training_weights,
    encode_target,
    run_training,
    select_best_model,
    split_feature_types,
    split_features_target,
)

REAL_RAW_PATH = PROJECT_ROOT / "data/raw/telco_customer_churn.csv"


# ---------- Data preparation ----------

def test_prepare_datasets_is_stratified_and_reproducible(raw_root, test_config):
    train_1, test_1 = prepare_datasets(test_config, root_dir=raw_root)
    train_2, test_2 = prepare_datasets(test_config, root_dir=raw_root)

    assert len(test_1) == 60
    assert len(train_1) == 240
    pd.testing.assert_frame_equal(train_1, train_2)
    pd.testing.assert_frame_equal(test_1, test_2)

    overall_rate = pd.concat([train_1, test_1])["Churn"].eq("Yes").mean()
    assert abs(test_1["Churn"].eq("Yes").mean() - overall_rate) < 0.05


def test_prepare_datasets_rejects_invalid_dataset(raw_root, config_copy):
    config_copy["data"]["target_column"] = "Missing"

    with pytest.raises(ValueError, match="validation failed"):
        prepare_datasets(config_copy, root_dir=raw_root)


def test_processed_datasets_round_trip(raw_root, test_config):
    train_df, test_df = prepare_datasets(test_config, root_dir=raw_root)
    save_processed_datasets(train_df, test_df, test_config, root_dir=raw_root)

    loaded_train, loaded_test = load_processed_datasets(test_config, root_dir=raw_root)

    assert list(loaded_train.columns) == list(train_df.columns)
    assert len(loaded_train) == len(train_df)
    assert len(loaded_test) == len(test_df)


def test_load_processed_datasets_missing_files_gives_clear_error(tmp_path, test_config):
    with pytest.raises(FileNotFoundError, match="prepare_data"):
        load_processed_datasets(test_config, root_dir=tmp_path)


# ---------- Target and feature handling ----------

def test_encode_target_maps_positive_class_to_one():
    y = pd.Series(["Yes", "No", "Yes"])

    assert encode_target(y, "Yes", "No").tolist() == [1, 0, 1]


def test_encode_target_rejects_unexpected_labels():
    with pytest.raises(ValueError, match="Unexpected"):
        encode_target(pd.Series(["Yes", "Maybe"]), "Yes", "No")


def test_split_features_target_drops_id_and_target(synthetic_churn_df, test_config):
    X, y = split_features_target(synthetic_churn_df, test_config["data"])

    assert "customerID" not in X.columns
    assert "Churn" not in X.columns
    assert len(X) == len(y) == len(synthetic_churn_df)
    assert set(y.unique()) <= {0, 1}


def test_split_feature_types_rejects_missing_numerical_column(synthetic_churn_df, test_config):
    X, _ = split_features_target(synthetic_churn_df, test_config["data"])

    with pytest.raises(ValueError, match="missing"):
        split_feature_types(X, ["not_a_column"])


# ---------- Leakage and imbalance ----------

def test_preprocessing_is_fitted_on_training_data_only(synthetic_churn_df, test_config):
    data_config = test_config["data"]
    X, y = split_features_target(synthetic_churn_df, data_config)
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)

    numerical, categorical = split_feature_types(X_train, data_config["numerical_columns"])
    pipeline = build_pipeline(numerical, categorical, LogisticRegression(max_iter=500))
    pipeline.fit(X_train, y_train)

    scaler = pipeline.named_steps["preprocessor"].named_transformers_["numerical"].named_steps["scaler"]
    tenure_index = numerical.index("tenure")

    # The scaler must reflect the training mean only, not the mean of all rows.
    assert np.isclose(scaler.mean_[tenure_index], X_train["tenure"].mean())
    assert not np.isclose(scaler.mean_[tenure_index], X["tenure"].mean())


def test_balanced_weights_give_equal_total_weight_per_class():
    y = pd.Series([0] * 8 + [1] * 2)

    weights = compute_training_weights(y, "balanced")

    assert np.isclose(weights[y.values == 0].sum(), weights[y.values == 1].sum())


def test_no_imbalance_strategy_gives_unit_weights():
    y = pd.Series([0, 0, 0, 1])

    assert np.all(compute_training_weights(y, "none") == 1.0)


def test_unsupported_imbalance_strategy_raises():
    with pytest.raises(ValueError, match="imbalance_strategy"):
        compute_training_weights(pd.Series([0, 1]), "oversample")


# ---------- Model registry and selection ----------

def test_build_candidate_models_returns_three_seeded_models(test_config):
    candidates = build_candidate_models(test_config["training"])

    assert set(candidates) == {"logistic_regression", "random_forest", "gradient_boosting"}
    assert all(candidate["estimator"].random_state == 42 for candidate in candidates.values())
    assert candidates["random_forest"]["param_grid"] == {"max_depth": [3]}


def test_build_candidate_models_rejects_unknown_model(test_config):
    test_config["training"]["models"]["xgboost"] = {"params": {}, "param_grid": {}}

    with pytest.raises(ValueError, match="Unsupported model"):
        build_candidate_models(test_config["training"])


def test_select_best_model_uses_cv_score_of_chosen_metric():
    tuned = {
        "first": {"cv_mean": {"roc_auc": 0.70, "f1": 0.90}},
        "second": {"cv_mean": {"roc_auc": 0.80, "f1": 0.50}},
    }

    assert select_best_model(tuned, "roc_auc") == "second"
    assert select_best_model(tuned, "f1") == "first"


def test_run_training_rejects_selection_metric_outside_scoring(config_copy, tmp_path):
    config_copy["training"]["scoring"] = ["accuracy"]
    config_copy["training"]["selection_metric"] = "roc_auc"

    with pytest.raises(ValueError, match="must be listed in scoring"):
        run_training(config_copy, root_dir=tmp_path)


# ---------- Full workflow ----------

def test_run_training_writes_artifacts_and_comparison(raw_root, test_config):
    train_df, test_df = prepare_datasets(test_config, root_dir=raw_root)
    save_processed_datasets(train_df, test_df, test_config, root_dir=raw_root)

    summary = run_training(test_config, root_dir=raw_root)

    comparison = pd.read_csv(summary["comparison_path"])
    assert len(comparison) == 3
    assert comparison["selected"].sum() == 1
    assert comparison.loc[comparison["selected"], "model"].item() == summary["selected_model"]

    assert (raw_root / "models" / "churn_pipeline_test.joblib").is_file()
    assert (raw_root / "models" / "churn_pipeline_test.metadata.json").is_file()
    assert (raw_root / "reports" / "evaluation_metrics.json").is_file()


@pytest.mark.skipif(not REAL_RAW_PATH.is_file(), reason="Raw Telco CSV not present in data/raw/")
def test_full_training_on_real_dataset(tmp_path):
    config = load_config()

    train_df, test_df = prepare_datasets(config)
    save_processed_datasets(train_df, test_df, config, root_dir=tmp_path)

    summary = run_training(config, root_dir=tmp_path)

    assert len(test_df) == 1409
    comparison = pd.read_csv(summary["comparison_path"])
    assert len(comparison) == 3
    selected_row = comparison.loc[comparison["selected"]].iloc[0]
    assert selected_row["test_roc_auc"] > 0.7
