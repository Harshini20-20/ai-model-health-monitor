
import numpy as np
import pandas as pd

from src.data.preprocessing import build_preprocessor


def test_preprocessor_fits_and_transforms():
    df = pd.DataFrame({
        "tenure": [12, 24, 36, 48],
        "MonthlyCharges": [50.0, 60.0, 70.0, 80.0],
        "Contract": [
            "Month-to-month",
            "One year",
            "Two year",
            "Month-to-month",
        ],
    })

    preprocessor = build_preprocessor(
        numerical_features=["tenure", "MonthlyCharges"],
        categorical_features=["Contract"],
    )

    transformed = preprocessor.fit_transform(df)

    assert transformed.shape[0] == 4
    assert transformed.shape[1] >= 4
    assert np.isfinite(transformed.toarray()).all() if hasattr(
        transformed, "toarray"
    ) else np.isfinite(transformed).all()


def test_preprocessor_handles_missing_values():
    df = pd.DataFrame({
        "tenure": [12, np.nan, 36],
        "Contract": ["Month-to-month", None, "Two year"],
    })

    preprocessor = build_preprocessor(
        numerical_features=["tenure"],
        categorical_features=["Contract"],
    )

    transformed = preprocessor.fit_transform(df)

    assert transformed.shape[0] == 3
    assert np.isfinite(transformed.toarray()).all() if hasattr(
        transformed, "toarray"
    ) else np.isfinite(transformed).all()


def test_preprocessor_handles_unknown_categories():
    train = pd.DataFrame({
        "tenure": [12, 24, 36],
        "Contract": ["Month-to-month", "One year", "Two year"],
    })

    new_data = pd.DataFrame({
        "tenure": [18],
        "Contract": ["Three year"],
    })

    preprocessor = build_preprocessor(
        numerical_features=["tenure"],
        categorical_features=["Contract"],
    )

    preprocessor.fit(train)
    transformed = preprocessor.transform(new_data)

    assert transformed.shape[0] == 1


def test_preprocessor_uses_expected_transformers():
    preprocessor = build_preprocessor(
        numerical_features=["tenure"],
        categorical_features=["Contract"],
    )

    assert {name for name, _, _ in preprocessor.transformers} == {        "numerical",
        "categorical",
    }
