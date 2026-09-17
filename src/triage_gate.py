"""Pure routing logic deciding when a ticket needs Tier-2 agentic triage.

Thresholds match docs/SPEC.md Section 5.1 (Agentic Escalation Tier Specification).
"""

from __future__ import annotations

from typing import Optional

# Recalibrated against data/customer_support_ticket_sample.csv (2026-09-17):
# the original fixed 0.80/48.0 values were never checked against this
# dataset's real distribution and fired on the majority of tickets, not a
# minority. These are now percentile-derived: flag the least-confident ~15%
# of Tier-1 predictions (P15 of confidence) and the longest-predicted ~15%
# of resolution estimates (P85 of resolution hours), measured on a
# 500-ticket sample of real data run through the real trained models.
CONFIDENCE_THRESHOLD = 0.67
RESOLUTION_HOURS_THRESHOLD = 185.0
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
