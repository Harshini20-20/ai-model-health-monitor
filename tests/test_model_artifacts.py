import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from src.training.artifacts import load_model_artifact, metadata_path_for, save_model_artifact
from src.training.train import build_pipeline, split_feature_types, split_features_target


@pytest.fixture
def fitted_pipeline(synthetic_churn_df, test_config):
    """A fitted raw-data pipeline built from the synthetic dataset."""
    data_config = test_config["data"]
    X, y = split_features_target(synthetic_churn_df, data_config)
    numerical, categorical = split_feature_types(X, data_config["numerical_columns"])

    pipeline = build_pipeline(numerical, categorical, LogisticRegression(max_iter=1000))
    pipeline.fit(X, y)

    return pipeline, X, numerical, categorical


def test_save_and_load_gives_identical_predictions(tmp_path, fitted_pipeline):
    pipeline, X, _, _ = fitted_pipeline
    model_path = tmp_path / "models" / "churn_pipeline_test.joblib"

    save_model_artifact(pipeline, {"model_name": "logistic_regression"}, model_path)
    loaded_pipeline, _ = load_model_artifact(model_path)

    np.testing.assert_array_equal(pipeline.predict(X), loaded_pipeline.predict(X))
    np.testing.assert_allclose(
        pipeline.predict_proba(X),
        loaded_pipeline.predict_proba(X),
    )


def test_metadata_sidecar_is_written_and_returned(tmp_path, fitted_pipeline):
    pipeline, _, _, _ = fitted_pipeline
    metadata = {"model_name": "logistic_regression", "input_columns": ["tenure"]}
    model_path = tmp_path / "churn_pipeline_test.joblib"

    save_model_artifact(pipeline, metadata, model_path)
    _, loaded_metadata = load_model_artifact(model_path)

    assert metadata_path_for(model_path) == tmp_path / "churn_pipeline_test.metadata.json"
    assert loaded_metadata == metadata


def test_loaded_pipeline_scores_raw_serving_rows(tmp_path, fitted_pipeline):
    """Cleaning lives inside the pipeline, so a raw row with a blank TotalCharges works."""
    pipeline, X, _, _ = fitted_pipeline
    model_path = tmp_path / "churn_pipeline_test.joblib"
    save_model_artifact(pipeline, {}, model_path)
    loaded_pipeline, _ = load_model_artifact(model_path)

    raw_row = X.iloc[[0]].copy()
    raw_row["TotalCharges"] = " "

    probabilities = loaded_pipeline.predict_proba(raw_row)

    assert probabilities.shape == (1, 2)
    assert np.isfinite(probabilities).all()


def test_load_missing_artifact_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="Model artifact"):
        load_model_artifact(tmp_path / "does_not_exist.joblib")


def test_load_fails_when_metadata_is_missing(tmp_path, fitted_pipeline):
    pipeline, _, _, _ = fitted_pipeline
    model_path = tmp_path / "churn_pipeline_test.joblib"
    save_model_artifact(pipeline, {}, model_path)
    metadata_path_for(model_path).unlink()

    with pytest.raises(FileNotFoundError):
        load_model_artifact(model_path)
