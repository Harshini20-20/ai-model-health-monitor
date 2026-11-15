"""MLflow connection setup: configuration validation, URI resolution and experiments.

Tracking is configured under the `tracking` key in config/config.yaml. Relative
paths resolve against the project root (or the root_dir passed in tests), so runs
never write outside the project.

The model registry needs a database backend, so tracking_uri must be a SQLite
database (sqlite:///...) or an MLflow tracking server (http:// or https://).
"""

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

SQLITE_PREFIX = "sqlite:///"
SERVER_PREFIXES = ("http://", "https://")
REQUIRED_KEYS = ("tracking_uri", "artifact_root", "experiment_name")
REQUIRED_REGISTRY_KEYS = ("model_name", "candidate_alias")


class TrackingError(RuntimeError):
    """Raised when MLflow rejects a tracking, experiment or registry operation."""


def validate_tracking_config(tracking_config: dict) -> None:
    """Fail early with a clear message when the tracking configuration is unusable."""
    missing = [key for key in REQUIRED_KEYS if not tracking_config.get(key)]

    registry = tracking_config.get("registry") or {}
    missing += [
        f"registry.{key}" for key in REQUIRED_REGISTRY_KEYS if not registry.get(key)
    ]
    if missing:
        raise ValueError(f"Tracking configuration is missing required keys: {missing}")

    uri = tracking_config["tracking_uri"]
    if not uri.startswith((SQLITE_PREFIX, *SERVER_PREFIXES)):
        raise ValueError(
            "tracking_uri must be a SQLite database (sqlite:///...) or a tracking server "
            "URL (http:// or https://), because the model registry needs a database backend. "
            f"Got: '{uri}'"
        )

    alias = registry["candidate_alias"]
    if alias.isdigit():
        raise ValueError(f"Registry alias must not be numeric: '{alias}'")


def resolve_tracking_uri(tracking_config: dict, base_dir: Path) -> str:
    """Return the tracking URI with any relative SQLite path anchored to base_dir."""
    uri = tracking_config["tracking_uri"]
    if not uri.startswith(SQLITE_PREFIX):
        return uri

    db_path = Path(uri[len(SQLITE_PREFIX):])
    if not db_path.is_absolute():
        db_path = base_dir / db_path

    db_path = db_path.resolve()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return f"{SQLITE_PREFIX}{db_path.as_posix()}"


def resolve_artifact_location(tracking_config: dict, base_dir: Path) -> str:
    """Return a file:// URI for the artifact folder, creating it if needed."""
    artifact_dir = (base_dir / tracking_config["artifact_root"]).resolve()
    artifact_dir.mkdir(parents=True, exist_ok=True)
    return artifact_dir.as_uri()


@contextmanager
def tracking_uri_scope(tracking_uri: str) -> Iterator[None]:
    """Point the global MLflow tracking URI at tracking_uri, then restore the previous value."""
    previous = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(tracking_uri)
    try:
        yield
    finally:
        mlflow.set_tracking_uri(previous)


def ensure_experiment(client: MlflowClient, name: str, artifact_location: str) -> str:
    """Return the id of the named experiment, creating it on first use."""
    try:
        experiment = client.get_experiment_by_name(name)
        if experiment is None:
            return client.create_experiment(name, artifact_location=artifact_location)

        if experiment.lifecycle_stage == "deleted":
            raise TrackingError(
                f"MLflow experiment '{name}' exists but is deleted. "
                "Restore it in MLflow or change tracking.experiment_name."
            )

        return experiment.experiment_id
    except MlflowException as exc:
        raise TrackingError(f"Could not create or open MLflow experiment '{name}': {exc}") from exc
