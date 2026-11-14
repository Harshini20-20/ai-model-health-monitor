
import pandas as pd
import pytest

from src.data.validation import validate_dataframe


def test_valid_dataframe():
    df = pd.DataFrame({
        "tenure": [12, 24, 36],
        "MonthlyCharges": [50.0, 60.0, 70.0],
        "Churn": ["Yes", "No", "Yes"],
    })

    report = validate_dataframe(df, target_column="Churn")

    assert report["is_valid"] is True
    assert report["errors"] == []


def test_missing_target_column():
    df = pd.DataFrame({
        "tenure": [12, 24],
        "MonthlyCharges": [50.0, 60.0],
    })

    report = validate_dataframe(df, target_column="Churn")

    assert report["is_valid"] is False
    assert "Target column 'Churn' is missing." in report["errors"]


def test_missing_target_values():
    df = pd.DataFrame({
        "tenure": [12, 24, 36],
        "Churn": ["Yes", None, "No"],
    })

    report = validate_dataframe(df, target_column="Churn")

    assert report["is_valid"] is False
    assert report["missing_target"] == 1


def test_duplicate_rows():
    df = pd.DataFrame({
        "tenure": [12, 12, 36],
        "Churn": ["Yes", "Yes", "No"],
    })

    report = validate_dataframe(df, target_column="Churn")

    assert report["duplicate_rows"] == 1
    assert report["is_valid"] is True


def test_empty_dataframe():
    df = pd.DataFrame(columns=["tenure", "Churn"])

    report = validate_dataframe(df, target_column="Churn")

    assert report["is_valid"] is False
    assert "Dataset is empty." in report["errors"]


def test_non_dataframe_input():
    with pytest.raises(TypeError):
        validate_dataframe(
            [{"tenure": 12, "Churn": "Yes"}],
            target_column="Churn",
        )
