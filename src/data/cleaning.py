
import pandas as pd

def clean_dataframe(
    df: pd.DataFrame,
    numerical_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Clean data types and whitespace without modifying raw data."""

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    cleaned = df.copy()

    # Remove leading and trailing whitespace from text columns.
    text_columns = cleaned.select_dtypes(include=["object", "string"]).columns

    for column in text_columns:
        cleaned[column] = cleaned[column].str.strip()

    # Convert configured numerical columns.
    for column in numerical_columns or []:
        if column not in cleaned.columns:
            raise ValueError(
                f"Numerical column '{column}' does not exist."
            )

        cleaned[column] = pd.to_numeric(
            cleaned[column],
            errors="coerce",
        )

    return cleaned