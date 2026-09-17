"""FastAPI inference service for support ticket analytics."""

from __future__ import annotations

import logging
import os
import sys

from fastapi import FastAPI, HTTPException
from typing import Optional

from pydantic import BaseModel, Field

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.constants import DEFAULT_INFERENCE_ROW
from src.inference import predict_classification, predict_regression, predict_satisfaction, predict_classification_with_confidence
from src.triage_gate import should_escalate

logger = logging.getLogger(__name__)

INTERNAL_ERROR_DETAIL = "An internal error occurred while processing the request."
MODEL_UNAVAILABLE_DETAIL = "Service temporarily unavailable: required model file is missing."

app = FastAPI(
    title="Support Ops Intelligence API",
    description="Priority and resolution-time predictions for support tickets.",
    version="1.0.0",
)


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
        priority = predict_classification(payload)
    except FileNotFoundError as exc:
        logger.exception("Missing model file in predict_priority")
        raise HTTPException(status_code=503, detail=MODEL_UNAVAILABLE_DETAIL) from exc
    except Exception as exc:
        logger.exception("Unhandled error in predict_priority")
        raise HTTPException(status_code=500, detail=INTERNAL_ERROR_DETAIL) from exc
    return PriorityResponse(predicted_priority=str(priority))


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


@app.post("/triage_agent", response_model=TwoTierTriageResponse)
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
