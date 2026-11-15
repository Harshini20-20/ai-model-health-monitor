"""Save and load the complete fitted pipeline together with its metadata."""

import json
from pathlib import Path

import joblib
from sklearn.pipeline import Pipeline


def save_model_artifact(pipeline: Pipeline, metadata: dict, model_path: Path) -> Path:
    """Persist a fitted pipeline with joblib and a JSON metadata sidecar file."""
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(pipeline, model_path)

    with metadata_path_for(model_path).open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    return model_path


def load_model_artifact(model_path: Path) -> tuple[Pipeline, dict]:
    """Load a fitted pipeline and its metadata saved by save_model_artifact."""
    model_path = Path(model_path)
    metadata_path = metadata_path_for(model_path)

    for path in (model_path, metadata_path):
        if not path.is_file():
            raise FileNotFoundError(f"Model artifact file not found: {path}")

    pipeline = joblib.load(model_path)

    with metadata_path.open("r", encoding="utf-8") as file:
        metadata = json.load(file)

    return pipeline, metadata


def metadata_path_for(model_path: Path) -> Path:
    """Return the metadata path that sits beside a model file."""
    return Path(model_path).with_suffix(".metadata.json")
