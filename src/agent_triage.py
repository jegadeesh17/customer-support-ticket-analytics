"""Agentic Triage Engine for Customer Support Escalation.

Provides structured root-cause analysis, customer sentiment scoring,
escalation decisions, and drafted responses using an LLM or fallback heuristics.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field

from configs.settings import settings

logger = logging.getLogger(__name__)


class AgentTriageResult(BaseModel):
    ticket_id: Optional[str] = Field(default=None, description="Identifier for the support ticket")
    escalate_to_tier2: bool = Field(..., description="Whether this ticket requires Senior / Tier 2 engineer triage")
    customer_frustration_score: int = Field(..., ge=1, le=10, description="Customer frustration rating from 1 to 10")
    root_cause_category: str = Field(..., description="High-level diagnostic category for the root issue")
    urgency_reasoning: str = Field(..., description="Concise justification for the assigned urgency and escalation status")
    recommended_action: str = Field(..., description="Concrete operational action for the support agent")
    auto_drafted_response: str = Field(..., description="Empathetic, clear, and professional response ready for customer review")
    confidence: float = Field(default=0.85, ge=0.0, le=1.0, description="Confidence level of the triage evaluation")
    triage_source: str = Field(default="heuristic_fallback", description="Source of triage: 'agent_llm' or 'heuristic_fallback'")


def _extract_heuristics(ticket: Dict[str, Any]) -> AgentTriageResult:
    """Deterministic, high-reliability fallback when LLM gateway is offline or unconfigured."""
    text = str(ticket.get("issue_description", "")).lower()
    sub_type = str(ticket.get("subscription_type", "Basic")).capitalize()
    product = str(ticket.get("product", "General"))
    category = str(ticket.get("category", "General Inquiry"))
    complexity = int(ticket.get("issue_complexity_score", 5))
    tenure = int(ticket.get("customer_tenure_months", 12))

    critical_signals = ["urgent", "down", "outage", "broken", "critical", "crash", "corrupted", "breach", "fail", "lost", "freeze"]
    frustration_signals = ["frustrated", "angry", "unacceptable", "refund", "cancel", "stuck", "terrible", "immediately", "lawyer", "second time"]

    score = 4
    if sub_type in ("Enterprise", "Premium"):
        score += 1
    if complexity >= 7:
        score += 1
    for word in critical_signals:
        if word in text:
            score += 1
            break
    for word in frustration_signals:
        if word in text:
            score += 2
            break
    frustration_score = min(10, max(1, score))

    needs_tier2 = (
        frustration_score >= 7
        or complexity >= 8
        or sub_type == "Enterprise"
        or "outage" in text
        or "crash" in text
        or ("payment" in text and "fail" in text)
    )

    if "payment" in text or "billing" in text or "refund" in text or "charge" in text:
        root_cause = "Billing and Financial Transaction Discrepancy"
        action = "Verify Stripe/payment gateway ledger, halt recurring retries, and escalate to Billing Ops."
    elif "login" in text or "password" in text or "auth" in text or "access" in text or "token" in text:
        root_cause = "Identity and Access Management (IAM) Lockout"
        action = "Initiate secure 2FA reset workflow and audit user session logs."
    elif "crash" in text or "error" in text or "500" in text or "bug" in text or "sync" in text:
        root_cause = "Application Infrastructure / Service Fault"
        action = "Capture client stack trace, cross-reference APM telemetry, and link to active incident ticket."
    else:
        root_cause = f"{category} on {product}"
        action = "Standard Level-1 troubleshooting protocol with contextual product guide."

    urgency_reason = (
        f"{'High-tier Enterprise' if sub_type == 'Enterprise' else sub_type} customer with frustration rating {frustration_score}/10. "
        f"Diagnosed as {root_cause} with complexity rating {complexity}/10."
    )

    draft = (
        f"Hello,\n\n"
        f"Thank you for reaching out to support regarding your {product} account. "
        f"I understand you are encountering an issue with {category.lower()}, and I sincerely apologize for the disruption this has caused. "
        f"Our team has prioritized your request ({'Tier-2 High Priority Escalation' if needs_tier2 else 'Standard Priority'}) "
        f"and we are actively investigating the underlying root cause. We will provide an update within the next scheduled SLA window.\n\n"
        f"Best regards,\nCustomer Operations Engineering"
    )

    return AgentTriageResult(
        ticket_id=str(ticket.get("ticket_id")) if ticket.get("ticket_id") else None,
        escalate_to_tier2=needs_tier2,
        customer_frustration_score=frustration_score,
        root_cause_category=root_cause,
        urgency_reasoning=urgency_reason,
        recommended_action=action,
        auto_drafted_response=draft,
        confidence=0.88 if needs_tier2 else 0.82,
        triage_source="heuristic_fallback",
    )


def _select_provider():
    """Pick the first configured LLM provider as (endpoint_url, api_key, model).

    Order: Groq (fast, the newly configured provider) -> OpenRouter -> OpenAI.
    Returns None if nothing is configured, in which case the caller falls
    back to deterministic heuristics.
    """
    if settings.GROQ_API_KEY:
        return ("https://api.groq.com/openai/v1/chat/completions", settings.GROQ_API_KEY, settings.GROQ_MODEL)
    if settings.OPENROUTER_API_KEY:
        return ("https://openrouter.ai/api/v1/chat/completions", settings.OPENROUTER_API_KEY, "google/gemini-2.0-flash-001")
    if settings.OPENAI_API_KEY:
        return ("https://api.openai.com/v1/chat/completions", settings.OPENAI_API_KEY, "gpt-4o-mini")
    return None


def run_agent_triage(ticket: Dict[str, Any], api_key: Optional[str] = None) -> AgentTriageResult:
    """Evaluate a support ticket and return structured triage diagnostics.

    An explicit api_key argument is sent to OpenRouter directly (used by
    callers/tests that already hold a specific key). Otherwise the first
    configured provider from _select_provider() is used. Falls back
    gracefully to deterministic heuristics on network error or when no
    provider is configured.
    """
    if api_key:
        provider = ("https://openrouter.ai/api/v1/chat/completions", api_key, "google/gemini-2.0-flash-001")
    else:
        provider = _select_provider()

    if provider is None:
        return _extract_heuristics(ticket)

    endpoint_url, key, model = provider

    try:
        import urllib.request

        system_prompt = (
            "You are an expert Principal Customer Operations Triage Agent. "
            "Analyze the support ticket metadata and description. Respond ONLY with a valid JSON object matching this schema:\n"
            "{\n"
            '  "ticket_id": string or null,\n'
            '  "escalate_to_tier2": boolean,\n'
            '  "customer_frustration_score": integer between 1 and 10,\n'
            '  "root_cause_category": string,\n'
            '  "urgency_reasoning": string,\n'
            '  "recommended_action": string,\n'
            '  "auto_drafted_response": string,\n'
            '  "confidence": float between 0.0 and 1.0\n'
            "}"
        )

        user_content = json.dumps(ticket, indent=2)
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Triage this ticket:\n{user_content}"},
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.2,
        }

        req = urllib.request.Request(
            endpoint_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://supportops.internal",
                "X-Title": "SupportOps Agent Triage",
            },
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=5) as response:
            res_data = json.loads(response.read().decode("utf-8"))
            content = res_data["choices"][0]["message"]["content"]
            parsed = json.loads(content)
            parsed["triage_source"] = "agent_llm"
            return AgentTriageResult(**parsed)

    except Exception as exc:
        # Fallback to local heuristic evaluator without throwing
        logger.warning("LLM provider call failed, falling back to heuristics: %s", exc)
        return _extract_heuristics(ticket)
