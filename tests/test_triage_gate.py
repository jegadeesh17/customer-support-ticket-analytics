"""Tests for the Tier-1 -> Tier-2 escalation gate (docs/SPEC.md Section 5.1)."""

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.triage_gate import should_escalate


def test_low_confidence_triggers_escalation():
    result = should_escalate(subscription_type="Basic", issue_complexity_score=3, confidence=0.55, resolution_hours=5.0)
    assert result == "low_confidence"


def test_severe_resolution_triggers_escalation():
    result = should_escalate(subscription_type="Basic", issue_complexity_score=3, confidence=0.95, resolution_hours=200.0)
    assert result == "severe_resolution"


def test_high_risk_segment_triggers_escalation():
    result = should_escalate(subscription_type="Enterprise", issue_complexity_score=9, confidence=0.95, resolution_hours=5.0)
    assert result == "high_risk_segment"


def test_routine_ticket_does_not_escalate():
    result = should_escalate(subscription_type="Basic", issue_complexity_score=3, confidence=0.95, resolution_hours=5.0)
    assert result is None


def test_confidence_exactly_at_threshold_does_not_escalate():
    result = should_escalate(subscription_type="Basic", issue_complexity_score=3, confidence=0.67, resolution_hours=5.0)
    assert result is None


def test_resolution_exactly_at_threshold_does_not_escalate():
    result = should_escalate(subscription_type="Basic", issue_complexity_score=3, confidence=0.95, resolution_hours=185.0)
    assert result is None
