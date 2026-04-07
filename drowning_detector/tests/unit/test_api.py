"""Tests for FastAPI endpoints."""

import pytest
from fastapi.testclient import TestClient

from drowning_detector.api.main import app

client = TestClient(app)


class TestHealthEndpoints:
    """Test system health endpoints."""

    def test_health_check(self) -> None:
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["error"] is None

    def test_readiness_check(self) -> None:
        response = client.get("/ready")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
