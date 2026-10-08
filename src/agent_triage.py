"""Agentic Triage Engine for Customer Support Escalation.

Provides structured root-cause analysis, customer sentiment scoring,
escalation decisions, and drafted responses using an LLM or fallback heuristics.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
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


_budget_lock = threading.Lock()
_budget_day: Optional[str] = None
_budget_used = 0


def _consume_llm_budget() -> bool:
    """Count one Tier-2 LLM call against today's (UTC) in-memory budget.

    Returns False when TRIAGE_DAILY_LLM_CALLS is exhausted. State is per
    process and resets on a UTC day change or restart.
    """
    global _budget_day, _budget_used
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    with _budget_lock:
        if _budget_day != today:
            _budget_day, _budget_used = today, 0
        if _budget_used >= settings.TRIAGE_DAILY_LLM_CALLS:
            return False
        _budget_used += 1
        return True


def _describe_llm_failure(exc: Exception) -> str:
    """Short, secret-free reason for a failed LLM call (safe to show in the response)."""
    import urllib.error

    if isinstance(exc, urllib.error.HTTPError):
        detail = ""
        try:
            raw = exc.read()
            if hasattr(exc, "fp") and hasattr(exc.fp, "seek"):
                try:
                    exc.fp.seek(0)
                except Exception:
                    pass
            elif hasattr(exc, "seek"):
                try:
                    exc.seek(0)
                except Exception:
                    pass
            error = json.loads(raw.decode("utf-8")).get("error", {})
            detail = str(error.get("code") or error.get("message") or error.get("type") or "")[:60]
        except Exception:
            pass
        return f"provider returned HTTP {exc.code}" + (f" ({detail})" if detail else "")
    if isinstance(exc, TimeoutError) or "timed out" in str(exc).lower():
        return "provider timed out"
    if isinstance(exc, urllib.error.URLError):
        return "provider unreachable"
    if isinstance(exc, (json.JSONDecodeError, KeyError, IndexError, ValueError)):
        return "provider reply was not valid triage JSON"
    return f"unexpected {type(exc).__name__}"


_circuit_lock = threading.Lock()
_circuit_state: Dict[str, Dict[str, Any]] = {}
CIRCUIT_FAILURE_THRESHOLD = 3
CIRCUIT_COOLDOWN_SECONDS = 30.0


def _reset_circuit_breaker() -> None:
    """Reset circuit breaker state (used in testing)."""
    with _circuit_lock:
        _circuit_state.clear()


def _is_circuit_open(provider_name: str) -> bool:
    now = time.monotonic()
    with _circuit_lock:
        state = _circuit_state.get(provider_name)
        if not state:
            return False
        return state.get("open_until", 0.0) > now


def _record_provider_success(provider_name: str) -> None:
    with _circuit_lock:
        if provider_name in _circuit_state:
            _circuit_state[provider_name] = {"failures": 0, "open_until": 0.0}


def _record_provider_failure(provider_name: str) -> None:
    now = time.monotonic()
    with _circuit_lock:
        state = _circuit_state.setdefault(provider_name, {"failures": 0, "open_until": 0.0})
        state["failures"] += 1
        if state["failures"] >= CIRCUIT_FAILURE_THRESHOLD:
            state["open_until"] = now + CIRCUIT_COOLDOWN_SECONDS
            logger.warning(
                "Circuit breaker tripped for LLM provider '%s' after %d consecutive failures. Cooling down for %.0fs.",
                provider_name,
                state["failures"],
                CIRCUIT_COOLDOWN_SECONDS,
            )


def _get_configured_providers() -> List[Tuple[str, str, str, str]]:
    """Return all configured LLM providers as a list of (name, endpoint_url, api_key, model).

    Priority order: Groq (fast, primary) -> OpenRouter -> OpenAI.
    """
    providers = []
    if settings.GROQ_API_KEY:
        providers.append(("Groq", "https://api.groq.com/openai/v1/chat/completions", settings.GROQ_API_KEY, settings.GROQ_MODEL))
        # Fallback to secondary model if primary model is unavailable or mistyped
        fallback_model = "openai/gpt-oss-120b" if settings.GROQ_MODEL != "openai/gpt-oss-120b" else "qwen/qwen3.8-27b"
        if fallback_model != settings.GROQ_MODEL:
            providers.append(("Groq-Fallback", "https://api.groq.com/openai/v1/chat/completions", settings.GROQ_API_KEY, fallback_model))
    if settings.OPENROUTER_API_KEY:
        providers.append(("OpenRouter", "https://openrouter.ai/api/v1/chat/completions", settings.OPENROUTER_API_KEY, "google/gemini-2.0-flash-001"))
    if settings.OPENAI_API_KEY:
        providers.append(("OpenAI", "https://api.openai.com/v1/chat/completions", settings.OPENAI_API_KEY, "gpt-4o-mini"))
    return providers


def _select_provider():
    """Pick the first configured LLM provider as (endpoint_url, api_key, model).

    Preserved for backwards compatibility.
    """
    providers = _get_configured_providers()
    if providers:
        _, endpoint_url, key, model = providers[0]
        return (endpoint_url, key, model)
    return None


def run_agent_triage(ticket: Dict[str, Any], api_key: Optional[str] = None) -> AgentTriageResult:
    """Evaluate a support ticket and return structured triage diagnostics.

    An explicit api_key argument is sent to OpenRouter directly (used by callers/tests).
    Otherwise configured providers are attempted in priority order (Groq -> OpenRouter -> OpenAI).
    Falls back to secondary configured providers on transient/HTTP errors, and finally
    to deterministic heuristics if all providers fail or their circuit breaker is open.
    """
    if api_key:
        providers = [("Custom", "https://openrouter.ai/api/v1/chat/completions", api_key, "google/gemini-2.0-flash-001")]
    else:
        providers = _get_configured_providers()

    if not providers:
        return _extract_heuristics(ticket)

    if not _consume_llm_budget():
        logger.warning("Daily Tier-2 LLM budget exhausted; using heuristic fallback")
        result = _extract_heuristics(ticket)
        result.urgency_reasoning += " (Daily LLM budget exhausted; heuristic fallback used.)"
        return result

    available_providers = [p for p in providers if not _is_circuit_open(p[0])]
    if not available_providers:
        logger.warning("All LLM providers currently in circuit-breaker cooldown; using heuristic fallback")
        result = _extract_heuristics(ticket)
        result.urgency_reasoning += " (LLM unavailable: circuit breaker open; heuristic fallback used.)"
        return result

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

    import urllib.error
    import urllib.request

    user_content = json.dumps(ticket, indent=2)
    last_reason = "provider unavailable"

    for name, endpoint_url, key, model in available_providers:
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
                "User-Agent": "SupportOpsAnalytics/1.0 (CustomerSupportTicketAnalytics)",
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://supportops.internal",
                "X-Title": "SupportOps Agent Triage",
            },
            method="POST",
        )

        try:
            try:
                with urllib.request.urlopen(req, timeout=5) as response:
                    res_data = json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                # If provider rejects response_format with 400, retry once without it
                if exc.code == 400 and "response_format" in str(exc.read().decode("utf-8", errors="ignore")):
                    payload_no_fmt = {k: v for k, v in payload.items() if k != "response_format"}
                    req_retry = urllib.request.Request(
                        endpoint_url,
                        data=json.dumps(payload_no_fmt).encode("utf-8"),
                        headers=req.headers,
                        method="POST",
                    )
                    with urllib.request.urlopen(req_retry, timeout=5) as retry_resp:
                        res_data = json.loads(retry_resp.read().decode("utf-8"))
                else:
                    raise

            content = res_data["choices"][0]["message"]["content"]
            clean_content = content.strip()
            if "```" in clean_content:
                clean_content = re.sub(r"^```(?:json)?\s*", "", clean_content)
                clean_content = re.sub(r"\s*```$", "", clean_content)
            json_match = re.search(r"\{.*\}", clean_content, re.DOTALL)
            if json_match:
                clean_content = json_match.group(0)

            parsed = json.loads(clean_content)
            parsed["triage_source"] = "agent_llm"
            _record_provider_success(name)
            return AgentTriageResult(**parsed)
        except Exception as exc:
            _record_provider_failure(name)
            reason = _describe_llm_failure(exc)
            last_reason = reason
            logger.warning("LLM provider '%s' call failed (%s): %s", name, reason, exc)
            continue

    # Fallback to local heuristic evaluator if all attempted providers failed
    result = _extract_heuristics(ticket)
    result.urgency_reasoning += f" (LLM unavailable: {last_reason}; heuristic fallback used.)"
    return result

