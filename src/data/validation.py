
import pandas as pd


def validate_dataframe(
    df: pd.DataFrame,
    target_column: str,
    check_missing_values: bool = True,
    check_duplicates: bool = True,
    reject_missing_target: bool = True,
) -> dict:
    """Validate a dataset and return a structured quality report."""

    errors = []
    warnings = []

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    if df.empty:
        errors.append("Dataset is empty.")

    if target_column not in df.columns:
        errors.append(f"Target column '{target_column}' is missing.")

        return {
            "is_valid": False,
            "rows": len(df),
            "columns": len(df.columns),
            "errors": errors,
            "warnings": warnings,
        }

    missing_target = int(df[target_column].isna().sum())

    if missing_target > 0:
        message = (
            f"Target column contains {missing_target} missing values."
        )

        if reject_missing_target:
            errors.append(message)
        else:
            warnings.append(message)

    missing_values = {}

    if check_missing_values:
        missing_values = {
            column: int(count)
            for column, count in df.isna().sum().items()
            if count > 0
        }

        if missing_values:
            warnings.append(
                "Missing values detected. Preprocessing may be required."
            )

    duplicate_rows = 0

    if check_duplicates:
        duplicate_rows = int(df.duplicated().sum())

        if duplicate_rows > 0:
            warnings.append(
                f"Dataset contains {duplicate_rows} duplicate rows."
            )

    return {
        "is_valid": len(errors) == 0,
        "rows": len(df),
        "columns": len(df.columns),
        "target_column": target_column,
        "missing_target": missing_target,
        "missing_values": missing_values,
        "duplicate_rows": duplicate_rows,
        "errors": errors,
        "warnings": warnings,
    }
