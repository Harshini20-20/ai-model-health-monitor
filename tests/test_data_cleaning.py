
import pandas as pd
import pytest

from src.data.cleaning import clean_dataframe


def test_numeric_conversion():
    df = pd.DataFrame({
        "TotalCharges": ["100.5", "200.0", "300.5"]
    })

    result = clean_dataframe(df, ["TotalCharges"])

    assert pd.api.types.is_numeric_dtype(result["TotalCharges"])
    assert result["TotalCharges"].iloc[0] == 100.5


def test_blank_values_become_nan():
    df = pd.DataFrame({
        "TotalCharges": ["100.5", " ", "300.5"]
    })

    result = clean_dataframe(df, ["TotalCharges"])

    assert result["TotalCharges"].isna().sum() == 1


def test_whitespace_removal():
    df = pd.DataFrame({
        "Contract": [" Month-to-month ", " Two year "]
    })

    result = clean_dataframe(df)

    assert result["Contract"].tolist() == [
        "Month-to-month",
        "Two year",
    ]


def test_original_dataframe_is_preserved():
    df = pd.DataFrame({
        "TotalCharges": ["100.5", "200.0"]
    })

    original = df.copy()
    clean_dataframe(df, ["TotalCharges"])

    pd.testing.assert_frame_equal(df, original)


def test_invalid_numerical_column():
    df = pd.DataFrame({
        "tenure": [12, 24]
    })

    with pytest.raises(ValueError, match="does not exist"):
        clean_dataframe(df, ["TotalCharges"])


def test_non_dataframe_input():
    with pytest.raises(TypeError):
        clean_dataframe([{"tenure": 12}])
