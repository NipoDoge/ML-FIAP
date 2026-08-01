"""
Smoke Tests - Quick integration tests covering main application flows.
These tests verify that critical endpoints work end-to-end.
"""

from typing import ClassVar

import pytest
from fastapi.testclient import TestClient

from api.v1.endpoints import authorize
from core.deps import get_current_user, get_session
from main import app
from platform_ring.schemas.contracts import (
    ComparisonPredict,
    InferenceReport,
    ServedModelPredict,
    TrainingSelectionSummaryPredict,
)
from services.auth import auth_service


class DummyUser:
    """Mock user for testing"""

    def __init__(self, id=1, role_id=1):
        self.id = id
        self.role_id = role_id


class MockRegisteredUser:
    """Mock registered user for auth tests"""

    def __init__(self, id, name, email, active=True, role_id=1):
        self.id = id
        self.name = name
        self.email = email
        self.active = active
        self.role_id = role_id


class DummyPrediction:
    id = 1
    pipeline_run_id = 10
    prediction = 1
    probability = 0.8
    input_data: ClassVar[dict[str, str]] = {"gender": "Female"}


def _fake_inference_report() -> InferenceReport:
    return InferenceReport(
        served_model=ServedModelPredict(
            inference_backend="sklearn",
            predict_model_key="sklearn_pipeline",
            name="Logistic Regression",
            origin="fe_holdout",
        ),
        training_selection_summary=TrainingSelectionSummaryPredict(),
        comparison=ComparisonPredict(),
    )


@pytest.fixture(autouse=True)
def app_overrides(monkeypatch):
    """Override app dependencies for all smoke tests"""
    app.dependency_overrides[get_current_user] = lambda: DummyUser()
    app.dependency_overrides[get_session] = lambda: None
    app.dependency_overrides[authorize.get_session] = lambda: None

    async def fake_register_user(user, db):
        return MockRegisteredUser(id=1, name=user.name, email=user.email)

    async def fake_predict_for_domain(*args, **kwargs):
        return DummyPrediction(), _fake_inference_report(), None

    monkeypatch.setattr(
        "platform_ring.domains.common.processor_service.predict_for_domain",
        fake_predict_for_domain,
    )
    monkeypatch.setattr(auth_service, "register_user", fake_register_user)
    yield
    app.dependency_overrides.clear()


@pytest.fixture
def client():
    return TestClient(app)


class TestSmokeAuthFlow:
    """Smoke tests for authentication flow"""

    def test_signup_endpoint_works(self, client):
        """Test that user signup endpoint is working"""
        payload = {
            "name": "Test User",
            "email": "test@example.com",
            "password": "testpass123",
        }

        response = client.post("/v1/auth/signup", json=payload)

        assert response.status_code == 201
        assert response.json()["id"] == 1
        assert response.json()["email"] == "test@example.com"


class TestSmokePredictorFlow:
    """Smoke tests for ML predictor flow"""

    def test_churn_predict_endpoint_works(self, client):
        """Test that domain churn prediction endpoint is working"""
        payload = {
            "gender": "Female",
            "seniorcitizen": 0,
            "partner": 1,
            "dependents": 0,
            "tenure": 12,
            "phoneservice": 1,
            "multiplelines": 0,
            "internetservice": "DSL",
            "onlinesecurity": 0,
            "onlinebackup": 0,
            "deviceprotection": 0,
            "techsupport": 0,
            "streamingtv": 0,
            "streamingmovies": 0,
            "contract": "Month-to-month",
            "paperlessbilling": 1,
            "paymentmethod": "Electronic check",
            "monthlycharges": 70.5,
            "totalcharges": 845.0,
        }

        response = client.post("/v1/domains/churn/predict", json=payload)

        assert response.status_code == 200
        assert response.json()["prediction"] == 1
        assert response.json()["probability"] == 80.0
        body = response.json()
        assert "inference_report" in body
        assert body["inference_report"]["served_model"]["inference_backend"] == "sklearn"
