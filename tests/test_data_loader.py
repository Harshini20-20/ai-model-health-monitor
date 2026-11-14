
import pandas as pd
import pytest

from src.data.loader import load_dataset


def make_config(path="data/test.csv", source="local", fmt="csv"):
    return {
        "data": {
            "dataset": {
                "path": path,
                "source": source,
                "format": fmt,
            }
        }
    }


def test_load_valid_csv(tmp_path):
    df = pd.DataFrame({
        "tenure": [12, 24, 36],
        "Churn": ["Yes", "No", "Yes"],
    })

    csv_path = tmp_path / "test.csv"
    df.to_csv(csv_path, index=False)

    config = make_config("test.csv")
    result = load_dataset(config, root_dir=tmp_path)

    assert len(result) == 3
    assert list(result.columns) == ["tenure", "Churn"]


def test_missing_dataset_file(tmp_path):
    config = make_config("missing.csv")

    with pytest.raises(FileNotFoundError):
        load_dataset(config, root_dir=tmp_path)


def test_missing_dataset_path():
    config = {"data": {"dataset": {"source": "local"}}}

    with pytest.raises(ValueError, match="Dataset path"):
        load_dataset(config)


def test_unsupported_format():
    config = make_config(fmt="json")

    with pytest.raises(ValueError, match="Unsupported dataset format"):
        load_dataset(config)


def test_unsupported_source():
    config = make_config(source="remote")

    with pytest.raises(NotImplementedError):
        load_dataset(config)
