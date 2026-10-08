"""Smoke tests for pipeline construction, label derivation, and data loading."""

import os
import sys
import pandas as pd
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.model_trainer import build_feature_transformer, get_classification_pipelines
from src.label_engineering import (
    derive_priority_label,
    derive_resolution_hours,
    derive_satisfaction_class,
)
from src.data_loader import load_tickets


@pytest.fixture
def sample_ticket_df():
    return pd.DataFrame([
        {
            "ticket_id": "T-1",
            "issue_description": "Cannot log in, urgent authentication failed",
            "product": "Web Portal",
            "category": "Login Issue",
            "channel": "Email",
            "region": "North America",
            "subscription_type": "Premium",
            "customer_age": 30,
            "customer_tenure_months": 12,
            "previous_tickets": 2,
            "issue_complexity_score": 8,
            "escalated": "No",
            "sla_breached": "No",
            "first_response_time_hours": 4.0,
        },
        {
            "ticket_id": "T-2",
            "issue_description": "General question about billing refund",
            "product": "Mobile App",
            "category": "Billing Inquiry",
            "channel": "Chat",
            "region": "Europe",
            "subscription_type": "Basic",
            "customer_age": 45,
            "customer_tenure_months": 6,
            "previous_tickets": 0,
            "issue_complexity_score": 3,
            "escalated": "No",
            "sla_breached": "No",
            "first_response_time_hours": 1.0,
        },
        {
            "ticket_id": "T-3",
            "issue_description": "Database crash, service completely down",
            "product": "API Gateway",
            "category": "Outage",
            "channel": "Phone",
            "region": "Asia",
            "subscription_type": "Enterprise",
            "customer_age": 40,
            "customer_tenure_months": 36,
            "previous_tickets": 5,
            "issue_complexity_score": 10,
            "escalated": "Yes",
            "sla_breached": "Yes",
            "first_response_time_hours": 24.0,
        },
        {
            "ticket_id": "T-4",
            "issue_description": "Feature request for dark mode theme",
            "product": "Web Portal",
            "category": "Feedback",
            "channel": "Email",
            "region": "North America",
            "subscription_type": "Basic",
            "customer_age": 25,
            "customer_tenure_months": 2,
            "previous_tickets": 1,
            "issue_complexity_score": 1,
            "escalated": "No",
            "sla_breached": "No",
            "first_response_time_hours": 8.0,
        },
    ])


def test_label_engineering_derives_valid_labels(sample_ticket_df):
    priorities = derive_priority_label(sample_ticket_df)
    assert len(priorities) == 4
    assert set(priorities).issubset({"Low", "Medium", "High", "Urgent"})

    hours = derive_resolution_hours(sample_ticket_df)
    assert len(hours) == 4
    assert (hours >= 4.0).all() and (hours <= 240.0).all()

    satisfaction = derive_satisfaction_class(sample_ticket_df)
    assert len(satisfaction) == 4
    assert set(satisfaction).issubset({"Low", "Mid", "High"})


def test_build_feature_transformer_on_sample(sample_ticket_df):
    transformer = build_feature_transformer(sample_ticket_df, include_text=True)
    transformed = transformer.fit_transform(sample_ticket_df)
    assert transformed.shape[0] == 4
    assert transformed.shape[1] > 0


def test_classification_pipelines_train_tiny_slice(sample_ticket_df):
    df = sample_ticket_df.copy()
    y = derive_priority_label(df)
    X = df.drop(columns=["ticket_id"])
    pipelines = get_classification_pipelines(X)
    lr_pipe = pipelines["Logistic Regression"]
    lr_pipe.fit(X, y)
    preds = lr_pipe.predict(X)
    assert len(preds) == 4


def test_data_loader_csv_fallback():
    df = load_tickets(use_db=False)
    assert not df.empty
    assert "issue_description" in [c.lower().replace(" ", "_") for c in df.columns]
