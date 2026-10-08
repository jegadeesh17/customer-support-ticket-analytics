"""FastAPI inference service for support ticket analytics."""

from __future__ import annotations

import logging
import os
import sys
import threading
import time
from collections import deque
from contextlib import asynccontextmanager
from typing import Deque, Dict, Optional

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from pydantic import BaseModel, Field

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from configs.settings import settings
from src.constants import DEFAULT_INFERENCE_ROW
from src.inference import load_model_bundle, predict_classification, predict_regression, predict_satisfaction, predict_classification_with_confidence
from src import triage_gate
from src.triage_gate import should_escalate

logger = logging.getLogger(__name__)

INTERNAL_ERROR_DETAIL = "An internal error occurred while processing the request."
MODEL_UNAVAILABLE_DETAIL = "Service temporarily unavailable: required model file is missing."

MODEL_FILES = ("classification_model.pkl", "regression_model.pkl", "satisfaction_model.pkl")
_models_ready = threading.Event()


def _preload_models() -> None:
    """Unpickle all three bundles once so /health.models_ready reflects memory, not disk."""
    try:
        if all(load_model_bundle(name) is not None for name in MODEL_FILES):
            _models_ready.set()
        else:
            logger.warning("Model preload incomplete: a model file is missing")
    except Exception:
        logger.exception("Model preload failed")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    threading.Thread(target=_preload_models, name="model-preload", daemon=True).start()
    yield


app = FastAPI(
    title="Support Ops Intelligence API",
    description="Priority and resolution-time predictions for support tickets.",
    version="1.0.0",
    lifespan=lifespan,
)

# No cross-origin access unless CORS_ALLOW_ORIGINS lists origins (comma-separated); the UI is same-origin.
_cors_origins = [o.strip() for o in settings.CORS_ALLOW_ORIGINS.split(",") if o.strip() and o.strip() != "*"]
if _cors_origins:
    app.add_middleware(CORSMiddleware, allow_origins=_cors_origins, allow_methods=["GET", "POST"], allow_headers=["Content-Type"])


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.error("Unhandled error on %s %s", request.method, request.url.path, exc_info=exc)
    return JSONResponse(status_code=500, content={"detail": INTERNAL_ERROR_DETAIL})


_RATE_WINDOW_SECONDS = 60.0
_RATE_MAX_TRACKED_CLIENTS = 10000
_rate_lock = threading.Lock()
_rate_hits: Dict[str, Deque[float]] = {}


def _client_ip(request: Request) -> str:
    # Cloud Run appends the real client IP to the end of any client-supplied X-Forwarded-For,
    # so only the last entry can be trusted; earlier entries are attacker-controlled.
    forwarded = request.headers.get("x-forwarded-for", "")
    last = forwarded.split(",")[-1].strip()
    if last:
        return last
    return request.client.host if request.client else "unknown"


def triage_rate_limit(request: Request) -> None:
    """In-memory, per-instance sliding-window limit per client IP (see ADR-08)."""
    limit = settings.TRIAGE_RATE_LIMIT_PER_MIN
    if limit <= 0:
        return
    now = time.monotonic()
    ip = _client_ip(request)
    with _rate_lock:
        if len(_rate_hits) >= _RATE_MAX_TRACKED_CLIENTS and ip not in _rate_hits:
            for key in [k for k, q in _rate_hits.items() if not q or now - q[-1] >= _RATE_WINDOW_SECONDS]:
                del _rate_hits[key]
            if len(_rate_hits) >= _RATE_MAX_TRACKED_CLIENTS:
                _rate_hits.clear()
        hits = _rate_hits.setdefault(ip, deque())
        while hits and now - hits[0] >= _RATE_WINDOW_SECONDS:
            hits.popleft()
        if len(hits) >= limit:
            retry_after = max(1, int(_RATE_WINDOW_SECONDS - (now - hits[0])) + 1)
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded for /triage_agent. Please wait and try again.",
                headers={"Retry-After": str(retry_after)},
            )
        hits.append(now)


class TicketInput(BaseModel):
    issue_description: str = Field(..., min_length=5)
    product: str = "Web Portal"
    category: str = "Login Issue"
    channel: str = "Email"
    region: str = "North America"
    subscription_type: str = "Premium"
    customer_age: int = 35
    customer_gender: str = "Male"
    customer_tenure_months: int = 24
    previous_tickets: int = 3
    issue_complexity_score: int = 5
    priority: str = "Medium"
    first_response_time_hours: float = 12.0


class PriorityResponse(BaseModel):
    predicted_priority: str
    confidence: Optional[float] = None


class ResolutionResponse(BaseModel):
    predicted_resolution_hours: float


class SatisfactionResponse(BaseModel):
    predicted_satisfaction_band: str


@app.get("/health")
def health() -> dict:
    from src.paths import get_models_dir

    models_dir = get_models_dir()
    return {
        "status": "ok",
        "classification_model": os.path.exists(os.path.join(models_dir, "classification_model.pkl")),
        "regression_model": os.path.exists(os.path.join(models_dir, "regression_model.pkl")),
        "satisfaction_model": os.path.exists(os.path.join(models_dir, "satisfaction_model.pkl")),
        "models_ready": _models_ready.is_set(),
    }


@app.get("/config")
def get_config() -> dict:
    """Read-only escalation thresholds, taken from src/triage_gate.py so they cannot drift."""
    return {
        "confidence_threshold": triage_gate.CONFIDENCE_THRESHOLD,
        "resolution_hours_threshold": triage_gate.RESOLUTION_HOURS_THRESHOLD,
        "high_risk_segment": triage_gate.HIGH_RISK_SEGMENT,
        "high_risk_complexity_threshold": triage_gate.HIGH_RISK_COMPLEXITY_THRESHOLD,
    }




from fastapi.responses import FileResponse

@app.get("/app", response_class=FileResponse, include_in_schema=False)
@app.get("/app/", response_class=FileResponse, include_in_schema=False)
def serve_app_ui():
    html_path = os.path.join(os.path.dirname(__file__), "index.html")
    if not os.path.exists(html_path):
        raise HTTPException(status_code=404, detail="UI file not found")
    return FileResponse(html_path)


@app.post("/predict_priority", response_model=PriorityResponse)
def predict_priority(ticket: TicketInput) -> PriorityResponse:
    payload = {**DEFAULT_INFERENCE_ROW, **ticket.model_dump()}
    try:
        priority, confidence = predict_classification_with_confidence(payload)
        return PriorityResponse(predicted_priority=str(priority), confidence=confidence)
    except FileNotFoundError as exc:
        logger.exception("Missing model file in predict_priority")
        raise HTTPException(status_code=503, detail=MODEL_UNAVAILABLE_DETAIL) from exc
    except Exception as exc:
        logger.warning("Could not compute priority confidence, attempting fallback: %s", exc)
        try:
            priority = predict_classification(payload)
            return PriorityResponse(predicted_priority=str(priority), confidence=None)
        except FileNotFoundError as fnf_exc:
            raise HTTPException(status_code=503, detail=MODEL_UNAVAILABLE_DETAIL) from fnf_exc
        except Exception as inner_exc:
            logger.exception("Unhandled error in predict_priority")
            raise HTTPException(status_code=500, detail=INTERNAL_ERROR_DETAIL) from inner_exc


@app.post("/predict_resolution_hours", response_model=ResolutionResponse)
def predict_resolution_hours(ticket: TicketInput) -> ResolutionResponse:
    payload = {**DEFAULT_INFERENCE_ROW, **ticket.model_dump()}
    try:
        hours = predict_regression(payload)
    except FileNotFoundError as exc:
        logger.exception("Missing model file in predict_resolution_hours")
        raise HTTPException(status_code=503, detail=MODEL_UNAVAILABLE_DETAIL) from exc
    except Exception as exc:
        logger.exception("Unhandled error in predict_resolution_hours")
        raise HTTPException(status_code=500, detail=INTERNAL_ERROR_DETAIL) from exc
    return ResolutionResponse(predicted_resolution_hours=float(hours))


@app.post("/predict_satisfaction", response_model=SatisfactionResponse)
def predict_satisfaction_band(ticket: TicketInput) -> SatisfactionResponse:
    payload = {**DEFAULT_INFERENCE_ROW, **ticket.model_dump()}
    try:
        band = predict_satisfaction(payload)
    except FileNotFoundError as exc:
        logger.exception("Missing model file in predict_satisfaction_band")
        raise HTTPException(status_code=503, detail=MODEL_UNAVAILABLE_DETAIL) from exc
    except Exception as exc:
        logger.exception("Unhandled error in predict_satisfaction_band")
        raise HTTPException(status_code=500, detail=INTERNAL_ERROR_DETAIL) from exc
    return SatisfactionResponse(predicted_satisfaction_band=str(band))


from src.agent_triage import run_agent_triage


class TwoTierTriageResponse(BaseModel):
    ticket_id: Optional[str] = None
    tier1_priority: str
    tier1_confidence: float
    tier1_resolution_hours: float
    escalate_to_tier2: bool
    escalation_trigger: Optional[str] = None
    tier2_recommends_escalation: Optional[bool] = None
    customer_frustration_score: Optional[int] = None
    root_cause_category: Optional[str] = None
    urgency_reasoning: Optional[str] = None
    recommended_action: Optional[str] = None
    auto_drafted_response: Optional[str] = None
    triage_source: Optional[str] = None


@app.post("/triage_agent", response_model=TwoTierTriageResponse, dependencies=[Depends(triage_rate_limit)])
def triage_agent_endpoint(ticket: TicketInput, force: bool = False) -> TwoTierTriageResponse:
    """Two-tier triage: Tier 1 always runs; Tier 2 (agentic diagnosis) only
    runs when Tier 1 signals low confidence, a severe resolution estimate,
    or a high-risk Enterprise segment -- or when force=True is passed."""
    payload = {**DEFAULT_INFERENCE_ROW, **ticket.model_dump()}
    try:
        priority, confidence = predict_classification_with_confidence(payload)
        resolution_hours = predict_regression(payload)
    except FileNotFoundError as exc:
        logger.exception("Missing model file in triage_agent_endpoint")
        raise HTTPException(status_code=503, detail=MODEL_UNAVAILABLE_DETAIL) from exc
    except Exception as exc:
        logger.exception("Unhandled error in triage_agent_endpoint")
        raise HTTPException(status_code=500, detail=INTERNAL_ERROR_DETAIL) from exc

    trigger = should_escalate(
        subscription_type=ticket.subscription_type,
        issue_complexity_score=ticket.issue_complexity_score,
        confidence=confidence,
        resolution_hours=resolution_hours,
    )
    if force and trigger is None:
        trigger = "forced"

    base_fields = dict(
        ticket_id=None,
        tier1_priority=str(priority),
        tier1_confidence=confidence,
        tier1_resolution_hours=resolution_hours,
        escalate_to_tier2=trigger is not None,
        escalation_trigger=trigger,
    )

    if trigger is None:
        return TwoTierTriageResponse(**base_fields)

    try:
        diagnosis = run_agent_triage(payload)
    except Exception as exc:
        logger.exception("Unhandled error in Tier 2 agent triage")
        raise HTTPException(status_code=500, detail=INTERNAL_ERROR_DETAIL) from exc

    return TwoTierTriageResponse(
        **base_fields,
        tier2_recommends_escalation=diagnosis.escalate_to_tier2,
        customer_frustration_score=diagnosis.customer_frustration_score,
        root_cause_category=diagnosis.root_cause_category,
        urgency_reasoning=diagnosis.urgency_reasoning,
        recommended_action=diagnosis.recommended_action,
        auto_drafted_response=diagnosis.auto_drafted_response,
        triage_source=diagnosis.triage_source,
    )
