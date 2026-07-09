"""FastAPI inference service for support ticket analytics."""

from __future__ import annotations

import os
import sys

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.constants import DEFAULT_INFERENCE_ROW
from src.inference import predict_classification, predict_regression, predict_satisfaction

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


@app.post("/predict_priority", response_model=PriorityResponse)
def predict_priority(ticket: TicketInput) -> PriorityResponse:
    payload = {**DEFAULT_INFERENCE_ROW, **ticket.model_dump()}
    try:
        priority = predict_classification(payload)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=repr(exc)) from exc
    return PriorityResponse(predicted_priority=str(priority))


@app.post("/predict_resolution_hours", response_model=ResolutionResponse)
def predict_resolution_hours(ticket: TicketInput) -> ResolutionResponse:
    payload = {**DEFAULT_INFERENCE_ROW, **ticket.model_dump()}
    try:
        hours = predict_regression(payload)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=repr(exc)) from exc
    return ResolutionResponse(predicted_resolution_hours=float(hours))


@app.post("/predict_satisfaction", response_model=SatisfactionResponse)
def predict_satisfaction_band(ticket: TicketInput) -> SatisfactionResponse:
    payload = {**DEFAULT_INFERENCE_ROW, **ticket.model_dump()}
    try:
        band = predict_satisfaction(payload)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=repr(exc)) from exc
    return SatisfactionResponse(predicted_satisfaction_band=str(band))
