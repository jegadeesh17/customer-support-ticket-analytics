"""Derive learnable targets from ticket features when raw labels lack signal."""

import numpy as np
import pandas as pd

URGENT_WORDS = ['urgent', 'crash', 'down', 'failed', 'cannot', 'bug', 'error', 'critical']
HIGH_WORDS = ['refund', 'billing', 'payment', 'security', 'suspend', 'locked']
LOW_WORDS = ['feature request', 'sync', 'question', 'clarification']


def _text_score(text):
    text = str(text).lower()
    score = 0
    if any(w in text for w in URGENT_WORDS):
        score += 3
    if any(w in text for w in HIGH_WORDS):
        score += 2
    if any(w in text for w in LOW_WORDS):
        score -= 1
    return score + min(len(text) // 120, 2)


def derive_priority_label(df):
    """Rule-based priority aligned with text urgency and ticket metadata."""
    labels = []
    for _, row in df.iterrows():
        score = _text_score(row.get('issue_description', ''))
        score += int(pd.to_numeric(row.get('issue_complexity_score', 5), errors='coerce') or 5) // 3
        score += int(pd.to_numeric(row.get('previous_tickets', 0), errors='coerce') or 0) // 6
        if str(row.get('escalated', '')).lower() == 'yes':
            score += 1
        if str(row.get('sla_breached', '')).lower() == 'yes':
            score += 1
        if score >= 5:
            labels.append('Urgent')
        elif score >= 3:
            labels.append('High')
        elif score >= 1:
            labels.append('Medium')
        else:
            labels.append('Low')
    return pd.Series(labels, index=df.index)


def derive_resolution_hours(df):
    """Rule-based resolution hours from derived priority and service metrics."""
    priority_hours = {'Low': 36.0, 'Medium': 60.0, 'High': 84.0, 'Urgent': 108.0}
    priority = derive_priority_label(df)
    complexity = pd.to_numeric(df.get('issue_complexity_score', 5), errors='coerce').fillna(5)
    response = pd.to_numeric(df.get('first_response_time_hours', 12), errors='coerce').fillna(12)
    prev = pd.to_numeric(df.get('previous_tickets', 0), errors='coerce').fillna(0)
    desc_len = df.get('issue_description', '').fillna('').astype(str).str.len()

    hours = (
        priority.map(priority_hours)
        + complexity * 4.5
        + response * 0.6
        + prev * 1.2
        + desc_len * 0.04
    )
    return hours.clip(4, 240)


def derive_satisfaction_class(df):
    """Rule-based satisfaction class from service responsiveness and complexity."""
    response = pd.to_numeric(df.get('first_response_time_hours', 12), errors='coerce').fillna(12)
    complexity = pd.to_numeric(df.get('issue_complexity_score', 5), errors='coerce').fillna(5)
    prev = pd.to_numeric(df.get('previous_tickets', 0), errors='coerce').fillna(0)
    sla = df.get('sla_breached', 'No').astype(str).str.lower().eq('yes').astype(int)

    score = 3
    score -= (response > 48).astype(int)
    score -= (response > 72).astype(int)
    score -= (complexity > 7).astype(int)
    score -= (prev > 10).astype(int)
    score -= sla * 2

    return score.apply(lambda s: 'High' if s >= 3 else ('Mid' if s == 2 else 'Low'))
