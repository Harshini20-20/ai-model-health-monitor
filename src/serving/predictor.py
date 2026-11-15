
"""Generate churn predictions using the loaded MLflow model."""

import pandas as pd

from src.serving.model_loader import ModelLoadError


class PredictionError(RuntimeError):
    """Raised when the model cannot generate a valid prediction."""


def predict_customer(model, customer_data: dict) -> dict:
    """Generate a churn prediction from one raw customer record."""

    try:
        input_df = pd.DataFrame([customer_data])

        predictions = model.predict(input_df)

        prediction_value = int(predictions[0])

        if prediction_value not in (0, 1):
            raise ValueError(
                f"Unexpected prediction value: {prediction_value}"
            )

        probabilities = model.predict_proba(input_df)
        churn_probability = float(probabilities[0][1])

        return {
            "prediction": "Yes" if prediction_value == 1 else "No",
            "churn_probability": churn_probability,
        }

    except Exception as exc:
        raise PredictionError(
            f"Unable to generate customer churn prediction: {exc}"
        ) from exc
