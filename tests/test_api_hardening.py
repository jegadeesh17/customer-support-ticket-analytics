"""Tests for M2 hardening: readiness, /config, rate limit, LLM budget, CORS, errors, confidence."""

import os
import sys
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import api.main as main
from configs.settings import settings
from src import agent_triage, triage_gate

TICKET = {"issue_description": "Payment failed twice, need urgent help."}


@pytest.fixture
def client():
    return TestClient(main.app)


@pytest.fixture
def mocked_tier1():
    with patch("api.main.predict_classification_with_confidence", return_value=("High", 0.5)):
        with patch("api.main.predict_regression", return_value=20.0):
            yield


def test_health_models_ready_false_until_loaded(client):
    main._models_ready.clear()
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["models_ready"] is False


def test_health_models_ready_true_after_preload(client):
    main._models_ready.clear()
    with patch("api.main.load_model_bundle", return_value={"model": object()}):
        main._preload_models()
    try:
        assert client.get("/health").json()["models_ready"] is True
    finally:
        main._models_ready.clear()


def test_preload_missing_model_stays_not_ready():
    main._models_ready.clear()
    with patch("api.main.load_model_bundle", return_value=None):
        main._preload_models()
    assert not main._models_ready.is_set()


def test_config_matches_gate_constants(client):
    body = client.get("/config").json()
    assert body["confidence_threshold"] == triage_gate.CONFIDENCE_THRESHOLD
    assert body["resolution_hours_threshold"] == triage_gate.RESOLUTION_HOURS_THRESHOLD
    assert body["high_risk_segment"] == triage_gate.HIGH_RISK_SEGMENT
    assert body["high_risk_complexity_threshold"] == triage_gate.HIGH_RISK_COMPLEXITY_THRESHOLD


def test_priority_includes_confidence(client):
    with patch("api.main.predict_classification", return_value="High"):
        with patch("api.main.predict_classification_with_confidence", return_value=("High", 0.91)):
            body = client.post("/predict_priority", json=TICKET).json()
    assert body["predicted_priority"] == "High"
    assert body["confidence"] == 0.91


def test_priority_confidence_is_null_when_helper_fails(client):
    with patch("api.main.predict_classification", return_value="High"):
        with patch("api.main.predict_classification_with_confidence", side_effect=RuntimeError("boom")):
            response = client.post("/predict_priority", json=TICKET)
    assert response.status_code == 200
    assert response.json()["confidence"] is None


def test_triage_rate_limit_returns_429(client, mocked_tier1):
    with patch.object(settings, "TRIAGE_RATE_LIMIT_PER_MIN", 2):
        assert client.post("/triage_agent", json=TICKET).status_code == 200
        assert client.post("/triage_agent", json=TICKET).status_code == 200
        response = client.post("/triage_agent", json=TICKET)
    assert response.status_code == 429
    assert "detail" in response.json()
    assert int(response.headers["retry-after"]) >= 1


def test_triage_rate_limit_is_per_ip_and_uses_first_forwarded_hop(client, mocked_tier1):
    with patch.object(settings, "TRIAGE_RATE_LIMIT_PER_MIN", 1):
        a = {"X-Forwarded-For": "1.1.1.1, 9.9.9.9"}
        b = {"X-Forwarded-For": "2.2.2.2, 9.9.9.9"}
        assert client.post("/triage_agent", json=TICKET, headers=a).status_code == 200
        assert client.post("/triage_agent", json=TICKET, headers=b).status_code == 200
        assert client.post("/triage_agent", json=TICKET, headers=a).status_code == 429


def test_rate_limit_does_not_affect_other_endpoints(client):
    with patch.object(settings, "TRIAGE_RATE_LIMIT_PER_MIN", 1):
        for _ in range(3):
            assert client.get("/config").status_code == 200


def test_llm_budget_exhausted_uses_heuristic_with_reason():
    ticket = {"issue_description": "Server outage, urgent", "subscription_type": "Enterprise"}
    with patch.object(settings, "GROQ_API_KEY", "x"), patch.object(settings, "TRIAGE_DAILY_LLM_CALLS", 0):
        with patch("urllib.request.urlopen") as urlopen:
            result = agent_triage.run_agent_triage(ticket)
    urlopen.assert_not_called()
    assert result.triage_source == "heuristic_fallback"
    assert "budget exhausted" in result.urgency_reasoning


def test_llm_budget_counts_and_resets_each_utc_day():
    with patch.object(settings, "TRIAGE_DAILY_LLM_CALLS", 2):
        assert agent_triage._consume_llm_budget() is True
        assert agent_triage._consume_llm_budget() is True
        assert agent_triage._consume_llm_budget() is False
        agent_triage._budget_day = "1999-01-01"
        assert agent_triage._consume_llm_budget() is True


def test_no_provider_does_not_consume_budget():
    with patch.object(settings, "GROQ_API_KEY", None), patch.object(settings, "OPENROUTER_API_KEY", None), \
            patch.object(settings, "OPENAI_API_KEY", None):
        agent_triage.run_agent_triage({"issue_description": "Server outage"})
    assert agent_triage._budget_used == 0


def test_no_cross_origin_access_by_default(client):
    response = client.get("/config", headers={"Origin": "https://evil.example"})
    assert "access-control-allow-origin" not in response.headers


def test_cors_wildcard_is_ignored():
    assert "*" not in main._cors_origins


def test_unhandled_error_returns_generic_500():
    @main.app.get("/_boom_test")
    def boom():
        raise RuntimeError("secret internal detail")

    try:
        response = TestClient(main.app, raise_server_exceptions=False).get("/_boom_test")
    finally:
        main.app.router.routes[:] = [r for r in main.app.router.routes if getattr(r, "path", "") != "/_boom_test"]
    assert response.status_code == 500
    assert response.json() == {"detail": main.INTERNAL_ERROR_DETAIL}
    assert "secret" not in response.text
