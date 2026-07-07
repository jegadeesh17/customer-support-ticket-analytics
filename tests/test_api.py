"""API and inference tests."""

import os
import sys
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


@pytest.fixture
def client():
    with patch("api.main.predict_classification", return_value="High"):
        with patch("api.main.predict_regression", return_value=18.5):
            with patch("api.main.os.path.exists", return_value=True):
                from api.main import app

                yield TestClient(app)


class TestHealth:
    def test_health_ok(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"


class TestPredictPriority:
    def test_predict_priority_schema(self, client):
        response = client.post(
            "/predict_priority",
            json={"issue_description": "Payment failed twice, need urgent help."},
        )
        assert response.status_code == 200
        assert response.json()["predicted_priority"] == "High"

    def test_predict_priority_rejects_short_text(self, client):
        response = client.post("/predict_priority", json={"issue_description": "hi"})
        assert response.status_code == 422


class TestPredictResolution:
    def test_predict_resolution_schema(self, client):
        response = client.post(
            "/predict_resolution_hours",
            json={"issue_description": "App crashes after latest update."},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["predicted_resolution_hours"] == 18.5
