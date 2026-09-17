"""Tests for the Agentic Triage Engine and API endpoint."""

import json
import os
import sys
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from api.main import app
from configs.settings import settings
from src.agent_triage import AgentTriageResult, run_agent_triage


@pytest.fixture
def client():
    return TestClient(app)


def test_heuristic_triage_critical_ticket():
    ticket = {
        "issue_description": "Our production database has crashed and customers cannot log in. This is critical outage!",
        "subscription_type": "Enterprise",
        "issue_complexity_score": 9,
        "product": "Cloud Storage",
        "category": "Performance Issue",
    }
    result = run_agent_triage(ticket, api_key=None)

    assert isinstance(result, AgentTriageResult)
    assert result.escalate_to_tier2 is True
    assert result.customer_frustration_score >= 7
    assert "Identity" in result.root_cause_category or "Application" in result.root_cause_category or "Fault" in result.root_cause_category
    assert result.triage_source == "heuristic_fallback"
    assert len(result.auto_drafted_response) > 50


def test_heuristic_triage_routine_inquiry():
    ticket = {
        "issue_description": "Can someone help me update my invoice email address for next month?",
        "subscription_type": "Basic",
        "issue_complexity_score": 2,
        "product": "Billing System",
        "category": "Feature Request",
    }
    result = run_agent_triage(ticket, api_key=None)

    assert isinstance(result, AgentTriageResult)
    assert result.escalate_to_tier2 is False
    assert result.customer_frustration_score <= 6
    assert result.triage_source == "heuristic_fallback"
    assert "Billing" in result.root_cause_category


def test_api_triage_agent_endpoint(client):
    payload = {
        "issue_description": "Emergency: all payment transactions are failing with code 500.",
        "product": "Payment Gateway",
        "category": "Payment Problem",
        "subscription_type": "Enterprise",
        "issue_complexity_score": 9,
    }
    response = client.post("/triage_agent", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "escalate_to_tier2" in data
    assert "customer_frustration_score" in data
    assert "root_cause_category" in data
    assert "recommended_action" in data
    assert "auto_drafted_response" in data
    assert data["customer_frustration_score"] >= 1


def test_agent_triage_llm_mock():
    mock_payload = {
        "ticket_id": "TCK-9901",
        "escalate_to_tier2": True,
        "customer_frustration_score": 9,
        "root_cause_category": "Database Lockout",
        "urgency_reasoning": "Outage reported by enterprise client.",
        "recommended_action": "Route to L2 on-call.",
        "auto_drafted_response": "We are currently investigating the issue...",
        "confidence": 0.95,
    }

    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps({
        "choices": [{"message": {"content": json.dumps(mock_payload)}}]
    }).encode("utf-8")
    mock_resp.__enter__.return_value = mock_resp

    with patch("urllib.request.urlopen", return_value=mock_resp):
        res = run_agent_triage({"issue_description": "Critical error"}, api_key="sk-mock-key")
        assert res.triage_source == "agent_llm"
        assert res.escalate_to_tier2 is True
        assert res.confidence == 0.95


def test_groq_provider_used_when_configured(monkeypatch):
    monkeypatch.setattr(settings, "GROQ_API_KEY", "gsk-mock-key")
    monkeypatch.setattr(settings, "OPENROUTER_API_KEY", None)
    monkeypatch.setattr(settings, "OPENAI_API_KEY", None)

    mock_payload = {
        "ticket_id": None,
        "escalate_to_tier2": True,
        "customer_frustration_score": 8,
        "root_cause_category": "Service Outage",
        "urgency_reasoning": "Enterprise outage.",
        "recommended_action": "Escalate to on-call.",
        "auto_drafted_response": "We are investigating.",
        "confidence": 0.9,
    }
    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps({
        "choices": [{"message": {"content": json.dumps(mock_payload)}}]
    }).encode("utf-8")
    mock_resp.__enter__.return_value = mock_resp

    captured_requests = []

    def fake_urlopen(req, timeout=5):
        captured_requests.append(req)
        return mock_resp

    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        result = run_agent_triage({"issue_description": "Outage"})

    assert result.triage_source == "agent_llm"
    assert captured_requests[0].full_url == "https://api.groq.com/openai/v1/chat/completions"


def test_groq_failure_falls_back_to_heuristic(monkeypatch):
    monkeypatch.setattr(settings, "GROQ_API_KEY", "gsk-mock-key")

    with patch("urllib.request.urlopen", side_effect=TimeoutError("no response")):
        result = run_agent_triage({
            "issue_description": "App is broken and I am furious",
            "issue_complexity_score": 9,
        })

    assert result.triage_source == "heuristic_fallback"
