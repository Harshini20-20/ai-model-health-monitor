
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import load_config
from src.data.cleaning import clean_dataframe
from src.data.loader import load_dataset
from src.data.preprocessing import build_preprocessor
from src.data.validation import validate_dataframe


def test_end_to_end_data_pipeline():
    # 1. Load configuration and dataset
    config = load_config()
    raw_df = load_dataset(config)

    assert len(raw_df) == 7043
    assert "Churn" in raw_df.columns

    # 2. Clean the dataset
    cleaned_df = clean_dataframe(
        raw_df,
        numerical_columns=["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"],
    )

    assert pd.api.types.is_numeric_dtype(cleaned_df["TotalCharges"])
    assert cleaned_df["TotalCharges"].isna().sum() == 11

    # 3. Validate the cleaned dataset
    validation_config = config["data"]["validation"]

    report = validate_dataframe(
        cleaned_df,
        target_column=config["data"]["target_column"],
        **validation_config,
    )

    assert report["is_valid"] is True
    assert report["missing_target"] == 0
    assert report["rows"] == 7043

    # 4. Separate features and target
    target_column = config["data"]["target_column"]
    id_columns = config["data"]["id_columns"]

    X = cleaned_df.drop(columns=[target_column, *id_columns])
    y = cleaned_df[target_column]

    assert "customerID" not in X.columns
    assert "Churn" not in X.columns

    # 5. Split data before fitting preprocessing
    split_config = config["data"]["split"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=split_config["test_size"],
        random_state=split_config["random_state"],
        stratify=y if split_config["stratify"] else None,
    )

    # 6. Build and fit preprocessing on training data only
    numerical_features = X_train.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_features = X_train.select_dtypes(
        include=["object", "string"]
    ).columns.tolist()

    preprocessor = build_preprocessor(
        numerical_features,
        categorical_features,
    )

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # 7. Verify final output
    assert X_train_processed.shape[0] == len(y_train)
    assert X_test_processed.shape[0] == len(y_test)
    assert X_train_processed.shape[1] == X_test_processed.shape[1]
    assert X_train_processed.shape[1] > 0
