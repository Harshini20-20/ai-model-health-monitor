"""Log a Phase 2 training session to MLflow and register the selected model.

Run structure, all inside one experiment:
- parent run: run parameters, the selected model's CV and test metrics, the
  comparison and evaluation reports, and the registered model version
- child run per candidate: its best hyperparameters, CV mean and std, test
  metrics and test confusion matrix

Every session registers a new model version and moves the candidate alias to it.
"""

import logging
from datetime import datetime, timezone
from pathlib import Path

import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from sklearn.pipeline import Pipeline

from src.tracking.mlflow_setup import (
    TrackingError,
    ensure_experiment,
    resolve_artifact_location,
    resolve_tracking_uri,
    tracking_uri_scope,
)
from src.tracking.registry import assign_alias, tag_model_version

logger = logging.getLogger(__name__)


def log_training_session(
    tracking_config: dict,
    base_dir: Path,
    *,
    model_version: str,
    run_params: dict,
    tuned: dict[str, dict],
    test_results: dict[str, dict],
    selected_model: str,
    scoring: list[str],
    selection_metric: str,
    best_pipeline: Pipeline,
    reports: list[Path],
    metadata_path: Path,
) -> dict:
    """Log one training session and register its selected pipeline.

    model_version is the Phase 2 label from config (for example "v1"). The registry
    assigns its own integer version, which is returned as registry_version.
    """
    tracking_uri = resolve_tracking_uri(tracking_config, base_dir)
    artifact_location = resolve_artifact_location(tracking_config, base_dir)
    experiment_name = tracking_config["experiment_name"]
    model_name = tracking_config["registry"]["model_name"]
    alias = tracking_config["registry"]["candidate_alias"]
    run_name = f"training-{model_version}-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"

    try:
        with tracking_uri_scope(tracking_uri):
            client = MlflowClient(tracking_uri=tracking_uri)
            experiment_id = ensure_experiment(client, experiment_name, artifact_location)

            with mlflow.start_run(experiment_id=experiment_id, run_name=run_name) as parent:
                mlflow.set_tags({
                    "phase": "3",
                    "selected_model": selected_model,
                    "selection_metric": selection_metric,
                })
                mlflow.log_params(run_params)
                mlflow.log_metrics(
                    _metric_values(tuned[selected_model], test_results[selected_model], scoring)
                )

                _log_candidate_runs(experiment_id, tuned, test_results, scoring)

                for report in reports:
                    mlflow.log_artifact(str(report), artifact_path="reports")
                mlflow.log_artifact(str(metadata_path), artifact_path="model_metadata")

                # cloudpickle, not MLflow's default skops: the pipeline contains a
                # functools.partial and the project's clean_dataframe function, which
                # skops refuses. Phase 2 already stores this pipeline with joblib (pickle),
                # so the trust level is unchanged.
                #
                # No signature or input example: MLflow would enforce them as a strict
                # schema, which rejects raw serving rows such as a blank TotalCharges.
                # The Phase 2 cleaning step exists to accept exactly those rows.
                model_info = mlflow.sklearn.log_model(
                    best_pipeline,
                    name="model",
                    registered_model_name=model_name,
                    serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
                )
                parent_run_id = parent.info.run_id

            registry_version = str(model_info.registered_model_version)
            assign_alias(client, model_name, alias, registry_version)
            tag_model_version(
                client,
                model_name,
                registry_version,
                {
                    "model_version": model_version,
                    "selected_model": selected_model,
                    "selection_metric": selection_metric,
                    "test_roc_auc": test_results[selected_model]["metrics"]["roc_auc"],
                    "training_run_id": parent_run_id,
                },
            )
    except (MlflowException, OSError) as exc:
        raise TrackingError(f"MLflow tracking failed for model '{model_name}': {exc}") from exc

    logger.info("Registered %s version %s as '%s'", model_name, registry_version, alias)

    return {
        "mlflow_tracking_uri": tracking_uri,
        "mlflow_experiment_id": experiment_id,
        "mlflow_run_id": parent_run_id,
        "registered_model_name": model_name,
        "registry_version": int(registry_version),
        "candidate_alias": alias,
        "model_uri": model_info.model_uri,
    }


def _log_candidate_runs(
    experiment_id: str,
    tuned: dict[str, dict],
    test_results: dict[str, dict],
    scoring: list[str],
) -> None:
    """Create one nested child run per candidate model."""
    for name, result in tuned.items():
        with mlflow.start_run(experiment_id=experiment_id, run_name=name, nested=True):
            mlflow.set_tag("candidate_model", name)
            mlflow.log_params({f"best_{key}": value for key, value in result["best_params"].items()})
            mlflow.log_metrics(_metric_values(result, test_results[name], scoring))
            mlflow.log_dict(
                test_results[name]["confusion_matrix"],
                "evaluation/test_confusion_matrix.json",
            )


def _metric_values(cv_result: dict, test_result: dict, scoring: list[str]) -> dict[str, float]:
    """Flatten CV and test metrics into MLflow metric names, e.g. cv_roc_auc_mean."""
    values = {}

    for metric in scoring:
        values[f"cv_{metric}_mean"] = cv_result["cv_mean"][metric]
        values[f"cv_{metric}_std"] = cv_result["cv_std"][metric]
        values[f"test_{metric}"] = test_result["metrics"][metric]

    return values
