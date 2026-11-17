
"""Automated tests for the FastAPI model-serving application."""

import pytest
from fastapi.testclient import TestClient

from src.serving.main import app
from src.serving.model_loader import ModelLoadError


class DummyModel:
    """Mock model for testing API behavior without loading MLflow."""

    def predict(self, data):
        return [1]

    def predict_proba(self, data):
        return [[0.38, 0.62]]


@pytest.fixture
def client(monkeypatch):
    """Create a test client with a mocked registered model."""

    monkeypatch.setattr(
        "src.serving.main.load_registered_model",
        lambda: (
            DummyModel(),
            {
                "model_name": "churn_pipeline",
                "model_version": "1",
                "alias": "candidate",
                "run_id": "test-run-id",
            },
        ),
    )

    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def customer_data():
    """Return a valid customer record with all 19 model features."""

    return {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 70.35,
        "TotalCharges": "844.2",
    }


def test_health_endpoint(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "healthy",
        "model_loaded": True,
    }


def test_model_info_endpoint(client):
    response = client.get("/model-info")

    assert response.status_code == 200
    assert response.json()["model_name"] == "churn_pipeline"
    assert response.json()["model_version"] == "1"
    assert response.json()["alias"] == "candidate"


def test_prediction_endpoint(client, customer_data):
    response = client.post("/predict", json=customer_data)

    assert response.status_code == 200

    result = response.json()

    assert result["prediction"] == "Yes"
    assert result["churn_probability"] == pytest.approx(0.62)
    assert result["model_name"] == "churn_pipeline"
    assert result["model_version"] == "1"


def test_prediction_rejects_invalid_input(client, customer_data):
    customer_data["tenure"] = 100

    response = client.post("/predict", json=customer_data)

    assert response.status_code == 422


def test_model_unavailable_returns_service_error(monkeypatch):
    def fail_model_loading():
        raise ModelLoadError("Test model unavailable")

    monkeypatch.setattr(
        "src.serving.main.load_registered_model",
        fail_model_loading,
    )

    with TestClient(app) as test_client:
        response = test_client.get("/health")

        assert response.status_code == 200
        assert response.json() == {
            "status": "unavailable",
            "model_loaded": False,
        }

        prediction_response = test_client.post(
            "/predict",
            json={
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 12,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "DSL",
                "OnlineSecurity": "No",
                "OnlineBackup": "Yes",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "Yes",
                "StreamingMovies": "No",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 70.35,
                "TotalCharges": "844.2",
            },
        )

        assert prediction_response.status_code == 503
