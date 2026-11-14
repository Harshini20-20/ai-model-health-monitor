
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_dataset(
    config: dict,
    root_dir: Path | None = None,
) -> pd.DataFrame:
    """Load a dataset using the project configuration."""

    dataset_config = config.get("data", {}).get("dataset", {})

    dataset_path = dataset_config.get("path")
    file_format = dataset_config.get("format", "csv").lower()
    source = dataset_config.get("source", "local")

    if not dataset_path:
        raise ValueError("Dataset path is missing from configuration.")

    if source != "local":
        raise NotImplementedError(
            f"Dataset source '{source}' is not supported yet."
        )

    if file_format != "csv":
        raise ValueError(
            f"Unsupported dataset format: {file_format}"
        )

    path = Path(dataset_path)

    if not path.is_absolute():
        base_dir = root_dir or PROJECT_ROOT
        path = base_dir / path

    if not path.is_file():
        raise FileNotFoundError(
            f"Dataset file not found: {path}"
        )

    return pd.read_csv(path)
