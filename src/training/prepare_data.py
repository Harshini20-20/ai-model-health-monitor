"""Command-line entry point: create processed train and test datasets.

Run with:  python -m src.training.prepare_data
"""

import logging

from src.config import load_config
from src.logging_config import configure_logging
from src.training.data_preparation import prepare_datasets, save_processed_datasets

logger = logging.getLogger(__name__)


def main() -> None:
    config = load_config()
    configure_logging(config["log_level"])

    train_df, test_df = prepare_datasets(config)
    save_processed_datasets(train_df, test_df, config)

    logger.info(
        "Preparation complete: %d train rows, %d test rows",
        len(train_df),
        len(test_df),
    )


if __name__ == "__main__":
    main()
