import pytest
from fastapi.testclient import TestClient

from api.v1.endpoints import authorize
from main import app
from services.auth import auth_service


class MockRegisteredUser:
    def __init__(self, id, name, email, active=True, created_at=None, role_id=1):
        self.id = id
        self.name = name
        self.email = email
        self.active = active
        self.created_at = created_at
        self.role_id = role_id


@pytest.fixture(autouse=True)
def app_overrides(monkeypatch):
    app.dependency_overrides[authorize.get_session] = lambda: None

    async def fake_register_user(user, db):
        return MockRegisteredUser(id=1, name=user.name, email=user.email)

    monkeypatch.setattr(auth_service, "register_user", fake_register_user)
    yield
    app.dependency_overrides.clear()


def test_signup_endpoint_creates_user(client):
    payload = {
        "name": "Gabriel",
        "email": "gabriel@example.com",
        "password": "secret123",
    }

    response = client.post("/v1/auth/signup", json=payload)

    assert response.status_code == 201
    json_response = response.json()
    assert json_response["id"] == 1
    assert json_response["name"] == "Gabriel"
    assert json_response["email"] == "gabriel@example.com"
    assert json_response["active"] is True


def test_signup_endpoint_rejects_invalid_payload(client):
    payload = {
        "name": "Gabriel",
        "email": "not-an-email",
        "password": "secret123",
    }

    response = client.post("/v1/auth/signup", json=payload)

    assert response.status_code == 422


@pytest.fixture
def client():
    return TestClient(app)
