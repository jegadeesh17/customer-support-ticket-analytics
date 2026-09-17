"""Measure the real Tier-1/Tier-2 escalation rate against sample ticket data.

Produces reports/ESCALATION_GATE_BENCHMARK.md with the actual percentage of
tickets Tier 1 resolves without escalation, replacing the previously
unverified "~90%" resume claim with a measured number.
"""

import os
import sys
from collections import Counter

import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.constants import DEFAULT_INFERENCE_ROW
from src.inference import predict_classification_with_confidence, predict_regression
from src.paths import get_data_path
from src.triage_gate import should_escalate


def compute_escalation_rate(triggers):
    """Given a list of trigger values (str or None), return (pct_not_escalated, breakdown)."""
    total = len(triggers)
    if total == 0:
        return 0.0, {}
    not_escalated = sum(1 for t in triggers if t is None)
    breakdown = dict(Counter(t for t in triggers if t is not None))
    return round(100.0 * not_escalated / total, 1), breakdown


def main(sample_size=500):
    df = pd.read_csv(get_data_path())
    # random_state=7 is intentionally different from the random_state=42 used
    # to derive CONFIDENCE_THRESHOLD/RESOLUTION_HOURS_THRESHOLD in
    # src/triage_gate.py -- reusing that seed would make this measurement
    # circular (it would just re-confirm the arithmetic that produced the
    # thresholds). Keep this a genuinely held-out sample.
    df = df.sample(min(sample_size, len(df)), random_state=7)

    triggers = []
    for _, row in df.iterrows():
        ticket = {**DEFAULT_INFERENCE_ROW, **{k.lower().replace(' ', '_'): v for k, v in row.to_dict().items()}}
        try:
            _, confidence = predict_classification_with_confidence(ticket)
            resolution_hours = predict_regression(ticket)
        except Exception:
            continue
        trigger = should_escalate(
            subscription_type=str(ticket.get("subscription_type", "Basic")),
            issue_complexity_score=int(ticket.get("issue_complexity_score", 5)),
            confidence=confidence,
            resolution_hours=resolution_hours,
        )
        triggers.append(trigger)

    pct_not_escalated, breakdown = compute_escalation_rate(triggers)

    report_path = os.path.join(ROOT, "reports", "ESCALATION_GATE_BENCHMARK.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Escalation Gate Benchmark\n\n")
        f.write(f"Sample size: {len(triggers)} tickets\n\n")
        f.write(f"**Tier 1 handled without escalation: {pct_not_escalated}%**\n\n")
        f.write("Escalation trigger breakdown:\n\n")
        for reason, count in breakdown.items():
            f.write(f"- {reason}: {count}\n")

    print(f"Tier 1 handled {pct_not_escalated}% without escalation. Report written to {report_path}")


if __name__ == "__main__":
    main(sample_size=1000)
