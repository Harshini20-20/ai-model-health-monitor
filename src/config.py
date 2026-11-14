from pathlib import Path
import os

import yaml
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[1]


def load_config() -> dict:
    """Load project YAML configuration and local environment variables."""
    load_dotenv(ROOT_DIR / ".env")
    config_path = ROOT_DIR / "config" / "config.yaml"

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}

    config["environment"] = os.getenv("APP_ENV", config.get("project", {}).get("environment", "local"))
    config["log_level"] = os.getenv("LOG_LEVEL", config.get("logging", {}).get("level", "INFO"))
    return config
