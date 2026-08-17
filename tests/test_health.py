"""Tests for the health endpoint."""

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_returns_200() -> None:
    """Health endpoint should respond with HTTP 200."""
    response = client.get("/health")
    assert response.status_code == 200


def test_health_response_body() -> None:
    """Health endpoint should return the expected JSON payload."""
    response = client.get("/health")
    data = response.json()
    assert data["status"] == "ok"
    assert data["service"] == "TenderLens"


def test_health_content_type() -> None:
    """Health endpoint should return JSON content-type."""
    response = client.get("/health")
    assert "application/json" in response.headers["content-type"]
