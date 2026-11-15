
"""FastAPI routes for model serving."""

from fastapi import APIRouter, HTTPException, Request

from src.serving.predictor import PredictionError, predict_customer
from src.serving.schemas import (
    HealthResponse,
    ModelInfoResponse,
    PredictionRequest,
    PredictionResponse,
)

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health_check(request: Request):
    """Check API and model availability."""

    model = getattr(request.app.state, "model", None)

    return HealthResponse(
        status="healthy" if model is not None else "unavailable",
        model_loaded=model is not None,
    )


@router.get("/model-info", response_model=ModelInfoResponse)
def model_info(request: Request):
    """Return metadata for the loaded model."""

    metadata = getattr(request.app.state, "model_metadata", None)

    if metadata is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded.",
        )

    return ModelInfoResponse(**metadata)


@router.post("/predict", response_model=PredictionResponse)
def predict(request: Request, customer: PredictionRequest):
    """Predict customer churn."""

    model = getattr(request.app.state, "model", None)
    metadata = getattr(request.app.state, "model_metadata", None)

    if model is None or metadata is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not available.",
        )

    try:
        result = predict_customer(
            model,
            customer.model_dump(),
        )

        return PredictionResponse(
            **result,
            model_name=metadata["model_name"],
            model_version=metadata["model_version"],
        )

    except PredictionError as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc
