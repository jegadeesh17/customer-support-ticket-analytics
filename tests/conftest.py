"""Shared fixtures: reset in-memory rate-limit and LLM-budget state between tests."""

import os
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


@pytest.fixture(autouse=True)
def _reset_in_memory_limits():
    import api.main as main
    import src.agent_triage as agent

    main._rate_hits.clear()
    agent._budget_day, agent._budget_used = None, 0
    yield
    main._rate_hits.clear()
    agent._budget_day, agent._budget_used = None, 0
