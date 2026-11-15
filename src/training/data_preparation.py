"""Create processed train and test datasets using the Phase 1 pipeline."""

import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.cleaning import clean_dataframe
from src.data.loader import PROJECT_ROOT, load_dataset
from src.data.validation import validate_dataframe

logger = logging.getLogger(__name__)


def prepare_datasets(
    config: dict,
    root_dir: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load, clean, validate and split the raw dataset.

    The split is stratified on the target and seeded from configuration, so the
    same config always produces the same train and test customers. Both returned
    frames keep the id and target columns for traceability.
    """
    data_config = config["data"]
    target_column = data_config["target_column"]

    raw_df = load_dataset(config, root_dir=root_dir)

    cleaned_df = clean_dataframe(
        raw_df,
        numerical_columns=data_config["numerical_columns"],
    )

    report = validate_dataframe(
        cleaned_df,
        target_column=target_column,
        **data_config["validation"],
    )

    for warning in report["warnings"]:
        logger.warning(warning)

    if not report["is_valid"]:
        raise ValueError(f"Dataset validation failed: {report['errors']}")

    split_config = data_config["split"]

    train_df, test_df = train_test_split(
        cleaned_df,
        test_size=split_config["test_size"],
        random_state=split_config["random_state"],
        stratify=cleaned_df[target_column] if split_config["stratify"] else None,
    )

    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def save_processed_datasets(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: dict,
    root_dir: Path | None = None,
) -> tuple[Path, Path]:
    """Write the train and test datasets to the configured processed paths."""
    train_path, test_path = _processed_paths(config, root_dir)

    train_path.parent.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    logger.info("Saved %d training rows to %s", len(train_df), train_path)
    logger.info("Saved %d test rows to %s", len(test_df), test_path)

    return train_path, test_path


def load_processed_datasets(
    config: dict,
    root_dir: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the processed train and test datasets written by prepare_datasets."""
    train_path, test_path = _processed_paths(config, root_dir)

    for path in (train_path, test_path):
        if not path.is_file():
            raise FileNotFoundError(
                f"Processed dataset not found: {path}. "
                "Run `python -m src.training.prepare_data` first."
            )

    return pd.read_csv(train_path), pd.read_csv(test_path)


def _processed_paths(config: dict, root_dir: Path | None) -> tuple[Path, Path]:
    base_dir = root_dir or PROJECT_ROOT
    processed = config["data"]["processed"]

    return (
        base_dir / processed["train_path"],
        base_dir / processed["test_path"],
    )
