
"""Load the registered candidate model from MLflow."""

import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from src.config import ROOT_DIR, load_config
from src.tracking.mlflow_setup import (
    resolve_tracking_uri,
    validate_tracking_config,
)


class ModelLoadError(RuntimeError):
    """Raised when the registered serving model cannot be loaded."""


def load_registered_model():
    """Load the registered Scikit-learn model using its configured alias."""

    config = load_config()
    tracking_config = config.get("tracking") or {}

    try:
        validate_tracking_config(tracking_config)

        tracking_uri = resolve_tracking_uri(tracking_config, ROOT_DIR)

        model_name = tracking_config["registry"]["model_name"]
        alias = tracking_config["registry"]["candidate_alias"]

        mlflow.set_tracking_uri(tracking_uri)

        client = MlflowClient(tracking_uri=tracking_uri)

        model_version = client.get_model_version_by_alias(
            model_name,
            alias,
        )

        model_uri = f"models:/{model_name}@{alias}"

        # Load using the Scikit-learn flavor so predict_proba is available.
        model = mlflow.sklearn.load_model(model_uri)

        metadata = {
            "model_name": model_name,
            "model_version": str(model_version.version),
            "alias": alias,
            "run_id": model_version.run_id,
        }

        return model, metadata

    except (MlflowException, OSError, ValueError, KeyError) as exc:
        raise ModelLoadError(
            f"Unable to load MLflow model "
            f"'{tracking_config.get('registry', {}).get('model_name', 'unknown')}' "
            f"using alias '{tracking_config.get('registry', {}).get('candidate_alias', 'unknown')}': {exc}"
        ) from exc
