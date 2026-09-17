"""Pure routing logic deciding when a ticket needs Tier-2 agentic triage.

Thresholds match docs/SPEC.md Section 5.1 (Agentic Escalation Tier Specification).
"""

from __future__ import annotations

from typing import Optional

CONFIDENCE_THRESHOLD = 0.80
RESOLUTION_HOURS_THRESHOLD = 48.0
HIGH_RISK_COMPLEXITY_THRESHOLD = 8


def should_escalate(
    subscription_type: str,
    issue_complexity_score: int,
    confidence: float,
    resolution_hours: float,
) -> Optional[str]:
    """Return the trigger reason if Tier 2 should run, else None.

    Trigger reasons: "low_confidence", "severe_resolution", "high_risk_segment".
    """
    if confidence < CONFIDENCE_THRESHOLD:
        return "low_confidence"
    if resolution_hours > RESOLUTION_HOURS_THRESHOLD:
        return "severe_resolution"
    if subscription_type == "Enterprise" and issue_complexity_score > HIGH_RISK_COMPLEXITY_THRESHOLD:
        return "high_risk_segment"
    return None
