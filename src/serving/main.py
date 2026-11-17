
"""FastAPI application entry point for model serving."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.serving.model_loader import ModelLoadError, load_registered_model
from src.serving.routes import router

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the registered model once when the application starts."""

    app.state.model = None
    app.state.model_metadata = None

    try:
        model, metadata = load_registered_model()

        app.state.model = model
        app.state.model_metadata = metadata

        logger.info(
            "Loaded model '%s', version '%s'",
            metadata["model_name"],
            metadata["model_version"],
        )

    except ModelLoadError:
        logger.exception(
            "Model loading failed. Prediction endpoints will be unavailable."
        )

    yield

    app.state.model = None
    app.state.model_metadata = None
    logger.info("Model serving application shut down.")


app = FastAPI(
    title="AI Model Health Monitor",
    description="Customer churn prediction API using an MLflow-registered model.",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(router)
