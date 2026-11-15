import mlflow.sklearn
import numpy as np
import pytest
from mlflow import MlflowClient

from src.training.artifacts import load_model_artifact
from src.training.data_preparation import (
    load_processed_datasets,
    prepare_datasets,
    save_processed_datasets,
)
from src.training.train import run_training, split_features_target
from src.tracking.mlflow_setup import (
    TrackingError,
    ensure_experiment,
    resolve_artifact_location,
    resolve_tracking_uri,
    tracking_uri_scope,
    validate_tracking_config,
)
from src.tracking.registry import (
    assign_alias,
    get_version_by_alias,
    list_model_versions,
    remove_alias,
    tag_model_version,
)

MODEL_NAME = "churn_pipeline_test"
EXPERIMENT_NAME = "churn_training_test"


@pytest.fixture
def tracking_config() -> dict:
    return {
        "enabled": True,
        "tracking_uri": "sqlite:///mlflow.db",
        "artifact_root": "mlruns",
        "experiment_name": EXPERIMENT_NAME,
        "registry": {"model_name": MODEL_NAME, "candidate_alias": "candidate"},
    }


@pytest.fixture
def prepared_root(raw_root, test_config):
    """Project root with processed train and test CSVs, ready for training."""
    train_df, test_df = prepare_datasets(test_config, root_dir=raw_root)
    save_processed_datasets(train_df, test_df, test_config, root_dir=raw_root)
    return raw_root


@pytest.fixture
def tracked_config(test_config, tracking_config) -> dict:
    test_config["tracking"] = tracking_config
    return test_config


def _client_for(root) -> MlflowClient:
    uri = resolve_tracking_uri({"tracking_uri": "sqlite:///mlflow.db"}, root)
    return MlflowClient(tracking_uri=uri)


# ---------- Configuration and URI handling ----------

def test_resolve_tracking_uri_anchors_sqlite_path_to_base_dir(tmp_path, tracking_config):
    uri = resolve_tracking_uri(tracking_config, tmp_path)

    assert uri == f"sqlite:///{(tmp_path / 'mlflow.db').resolve().as_posix()}"


def test_resolve_tracking_uri_passes_server_url_through(tmp_path, tracking_config):
    tracking_config["tracking_uri"] = "http://localhost:5000"

    assert resolve_tracking_uri(tracking_config, tmp_path) == "http://localhost:5000"


def test_resolve_artifact_location_creates_folder_and_returns_file_uri(tmp_path, tracking_config):
    location = resolve_artifact_location(tracking_config, tmp_path)

    assert location.startswith("file:")
    assert (tmp_path / "mlruns").is_dir()


def test_validate_accepts_complete_config(tracking_config):
    validate_tracking_config(tracking_config)


def test_validate_reports_missing_keys(tracking_config):
    del tracking_config["experiment_name"]
    del tracking_config["registry"]["model_name"]

    with pytest.raises(ValueError, match="experiment_name.*registry.model_name"):
        validate_tracking_config(tracking_config)


def test_validate_rejects_file_store_because_registry_needs_database(tracking_config):
    tracking_config["tracking_uri"] = "file:./mlruns"

    with pytest.raises(ValueError, match="database"):
        validate_tracking_config(tracking_config)


def test_validate_rejects_numeric_alias(tracking_config):
    tracking_config["registry"]["candidate_alias"] = "2"

    with pytest.raises(ValueError, match="numeric"):
        validate_tracking_config(tracking_config)


# ---------- Experiment creation ----------

def test_experiment_creation_is_idempotent(tmp_path, tracking_config):
    uri = resolve_tracking_uri(tracking_config, tmp_path)
    artifact_location = resolve_artifact_location(tracking_config, tmp_path)

    with tracking_uri_scope(uri):
        client = MlflowClient(tracking_uri=uri)
        first_id = ensure_experiment(client, EXPERIMENT_NAME, artifact_location)
        second_id = ensure_experiment(client, EXPERIMENT_NAME, artifact_location)

        experiment = client.get_experiment(first_id)

    assert first_id == second_id
    assert experiment.name == EXPERIMENT_NAME
    assert experiment.artifact_location.startswith("file:")


def test_ensure_experiment_rejects_deleted_experiment(tmp_path, tracking_config):
    uri = resolve_tracking_uri(tracking_config, tmp_path)
    artifact_location = resolve_artifact_location(tracking_config, tmp_path)

    with tracking_uri_scope(uri):
        client = MlflowClient(tracking_uri=uri)
        experiment_id = client.create_experiment(EXPERIMENT_NAME, artifact_location=artifact_location)
        client.delete_experiment(experiment_id)

        with pytest.raises(TrackingError, match="deleted"):
            ensure_experiment(client, EXPERIMENT_NAME, artifact_location)


# ---------- Registry lifecycle ----------

@pytest.fixture
def registry_client(tmp_path, tracking_config):
    """Client on a fresh SQLite registry with two model versions of one model."""
    uri = resolve_tracking_uri(tracking_config, tmp_path)
    client = MlflowClient(tracking_uri=uri)
    client.create_registered_model(MODEL_NAME)
    client.create_model_version(MODEL_NAME, source=str(tmp_path / "model_v1"))
    client.create_model_version(MODEL_NAME, source=str(tmp_path / "model_v2"))
    return client


def test_alias_assignment_and_lookup(registry_client):
    assign_alias(registry_client, MODEL_NAME, "candidate", 2)

    version = get_version_by_alias(registry_client, MODEL_NAME, "candidate")

    assert version is not None
    assert int(version.version) == 2


def test_alias_can_be_moved_to_another_version(registry_client):
    assign_alias(registry_client, MODEL_NAME, "candidate", 1)
    assign_alias(registry_client, MODEL_NAME, "candidate", 2)

    assert int(get_version_by_alias(registry_client, MODEL_NAME, "candidate").version) == 2


def test_missing_alias_returns_none(registry_client):
    assert get_version_by_alias(registry_client, MODEL_NAME, "champion") is None


def test_removed_alias_returns_none(registry_client):
    assign_alias(registry_client, MODEL_NAME, "candidate", 1)
    remove_alias(registry_client, MODEL_NAME, "candidate")

    assert get_version_by_alias(registry_client, MODEL_NAME, "candidate") is None


def test_list_model_versions_reports_aliases_and_tags(registry_client):
    tag_model_version(registry_client, MODEL_NAME, 1, {"model_version": "v1"})
    assign_alias(registry_client, MODEL_NAME, "candidate", 2)

    versions = list_model_versions(registry_client, MODEL_NAME)

    assert [item["version"] for item in versions] == [1, 2]
    assert versions[0]["tags"]["model_version"] == "v1"
    assert versions[0]["aliases"] == []
    assert versions[1]["aliases"] == ["candidate"]


# ---------- Logging a training session ----------

def test_training_session_logs_parent_run_params_and_metrics(prepared_root, tracked_config):
    summary = run_training(tracked_config, root_dir=prepared_root)
    client = _client_for(prepared_root)

    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    run = client.get_run(summary["mlflow_run_id"])

    assert experiment is not None
    assert run.info.experiment_id == experiment.experiment_id
    assert run.data.params["selected_model"] == summary["selected_model"]
    assert run.data.params["selection_metric"] == "roc_auc"
    assert run.data.params["random_state"] == "42"
    assert run.data.tags["phase"] == "3"

    selected_test_roc_auc = run.data.metrics["test_roc_auc"]
    assert 0.0 <= selected_test_roc_auc <= 1.0
    assert "cv_roc_auc_mean" in run.data.metrics
    assert "cv_roc_auc_std" in run.data.metrics


def test_each_candidate_gets_a_child_run_with_its_params_and_metrics(prepared_root, tracked_config):
    summary = run_training(tracked_config, root_dir=prepared_root)
    client = _client_for(prepared_root)
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)

    children = client.search_runs(
        [experiment.experiment_id],
        filter_string=f"tags.`mlflow.parentRunId` = '{summary['mlflow_run_id']}'",
    )

    assert sorted(child.info.run_name for child in children) == sorted(
        ["logistic_regression", "random_forest", "gradient_boosting"]
    )
    for child in children:
        assert any(key.startswith("best_") for key in child.data.params)
        assert "cv_roc_auc_mean" in child.data.metrics
        assert "test_roc_auc" in child.data.metrics


def test_reports_and_metadata_are_logged_as_artifacts(prepared_root, tracked_config):
    summary = run_training(tracked_config, root_dir=prepared_root)
    client = _client_for(prepared_root)

    report_files = {item.path for item in client.list_artifacts(summary["mlflow_run_id"], "reports")}
    metadata_files = {
        item.path for item in client.list_artifacts(summary["mlflow_run_id"], "model_metadata")
    }

    assert {"reports/model_comparison.csv", "reports/evaluation_metrics.json"} <= report_files
    assert "model_metadata/churn_pipeline_test.metadata.json" in metadata_files


# ---------- Registration and versioning ----------

def test_training_registers_new_version_and_moves_candidate_alias(prepared_root, tracked_config):
    first = run_training(tracked_config, root_dir=prepared_root)
    second = run_training(tracked_config, root_dir=prepared_root)
    client = _client_for(prepared_root)

    assert first["registry_version"] == 1
    assert second["registry_version"] == 2

    versions = list_model_versions(client, MODEL_NAME)
    assert [item["version"] for item in versions] == [1, 2]

    candidate = get_version_by_alias(client, MODEL_NAME, "candidate")
    assert int(candidate.version) == 2
    assert versions[0]["aliases"] == []
    assert versions[1]["aliases"] == ["candidate"]


def test_registry_version_records_phase2_label_and_training_run(prepared_root, tracked_config):
    summary = run_training(tracked_config, root_dir=prepared_root)
    client = _client_for(prepared_root)

    version = list_model_versions(client, MODEL_NAME)[0]

    assert version["tags"]["model_version"] == "test"
    assert version["tags"]["selected_model"] == summary["selected_model"]
    assert version["tags"]["training_run_id"] == summary["mlflow_run_id"]
    assert version["run_id"] == summary["mlflow_run_id"]


def test_registered_model_matches_phase2_joblib_artifact(prepared_root, tracked_config):
    """The registry copy of the model must predict exactly like the Phase 2 file."""
    run_training(tracked_config, root_dir=prepared_root)
    _, test_df = load_processed_datasets(tracked_config, root_dir=prepared_root)
    X_test, _ = split_features_target(test_df, tracked_config["data"])

    phase2_pipeline, _ = load_model_artifact(prepared_root / "models" / "churn_pipeline_test.joblib")
    uri = resolve_tracking_uri({"tracking_uri": "sqlite:///mlflow.db"}, prepared_root)
    with tracking_uri_scope(uri):
        registry_pipeline = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@candidate")

    np.testing.assert_allclose(
        phase2_pipeline.predict_proba(X_test),
        registry_pipeline.predict_proba(X_test),
    )


# ---------- Integration with Phase 2 behaviour ----------

def test_tracking_disabled_leaves_phase2_untouched(prepared_root, test_config):
    test_config["tracking"] = {"enabled": False, "tracking_uri": "file:./invalid-on-purpose"}

    summary = run_training(test_config, root_dir=prepared_root)

    assert "mlflow_run_id" not in summary
    assert (prepared_root / "models" / "churn_pipeline_test.joblib").is_file()
    assert not (prepared_root / "mlflow.db").exists()


def test_tracking_is_skipped_when_section_absent(prepared_root, test_config):
    summary = run_training(test_config, root_dir=prepared_root)

    assert set(summary) == {"selected_model", "model_path", "comparison_path", "metrics_path"}
    assert not (prepared_root / "mlflow.db").exists()


def test_invalid_tracking_config_fails_before_training_starts(prepared_root, test_config):
    test_config["tracking"] = {"enabled": True}

    with pytest.raises(ValueError, match="missing required keys"):
        run_training(test_config, root_dir=prepared_root)

    assert not (prepared_root / "models").exists()
