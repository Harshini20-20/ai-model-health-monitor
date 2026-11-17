
"""Request and response schemas for model serving."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class PredictionRequest(BaseModel):
    """Raw customer features accepted by the churn model."""

    model_config = ConfigDict(extra="forbid")

    gender: Literal["Female", "Male"]
    SeniorCitizen: int = Field(ge=0, le=1)
    Partner: Literal["Yes", "No"]
    Dependents: Literal["Yes", "No"]
    tenure: int = Field(ge=0, le=72)
    PhoneService: Literal["Yes", "No"]
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Literal["Yes", "No", "No internet service"]
    OnlineBackup: Literal["Yes", "No", "No internet service"]
    DeviceProtection: Literal["Yes", "No", "No internet service"]
    TechSupport: Literal["Yes", "No", "No internet service"]
    StreamingTV: Literal["Yes", "No", "No internet service"]
    StreamingMovies: Literal["Yes", "No", "No internet service"]
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: Literal["Yes", "No"]
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: str | float


class PredictionResponse(BaseModel):
    """Prediction returned by the serving API."""

    prediction: Literal["Yes", "No"]
    churn_probability: float = Field(ge=0, le=1)
    model_name: str
    model_version: str


class HealthResponse(BaseModel):
    """API health status."""

    status: str
    model_loaded: bool


class ModelInfoResponse(BaseModel):
    """Information about the currently loaded model."""

    model_name: str
    model_version: str
    alias: str
    run_id: str | None = None
