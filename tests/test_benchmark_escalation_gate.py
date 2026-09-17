"""Tests for the escalation-rate aggregation logic used by the benchmark script."""

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts.benchmark_escalation_gate import compute_escalation_rate


def test_compute_escalation_rate_basic():
    triggers = [None, None, None, "low_confidence"]
    pct, breakdown = compute_escalation_rate(triggers)
    assert pct == 75.0
    assert breakdown == {"low_confidence": 1}


def test_compute_escalation_rate_all_escalated():
    triggers = ["severe_resolution", "high_risk_segment"]
    pct, breakdown = compute_escalation_rate(triggers)
    assert pct == 0.0
    assert breakdown == {"severe_resolution": 1, "high_risk_segment": 1}


def test_compute_escalation_rate_empty():
    pct, breakdown = compute_escalation_rate([])
    assert pct == 0.0
    assert breakdown == {}
