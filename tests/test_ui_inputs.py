"""Guard against UI controls that the fitted pipelines silently ignore.

Each Streamlit page sends a payload of user inputs. If a key in that payload is not a
column the fitted ColumnTransformer consumes, the control is dead: the user changes it,
nothing happens, and no error is raised (remainder='drop' discards the extra column).
This is exactly how the 'Current Priority (optional hint)' and regression-page
'First Response Time' controls shipped broken.

Skipped when the .pkl bundles are absent so CI stays green without them.
"""

import os
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

# Keys each page puts in its prediction payload — keep in sync with app/pages/*.py.
PAGE_PAYLOAD_KEYS = {
    "classification_model.pkl": {
        "product", "category", "issue_description", "subscription_type",
        "channel", "issue_complexity_score", "previous_tickets",
    },
    "regression_model.pkl": {
        "category", "issue_description", "priority",
        "issue_complexity_score", "previous_tickets",
    },
    "satisfaction_model.pkl": {
        "issue_description", "category", "priority", "channel", "subscription_type",
        "first_response_time_hours", "previous_tickets", "issue_complexity_score",
        "sla_breached", "escalated",
    },
}


def _consumed_columns(model):
    """Raw input columns the fitted preprocessor actually reads."""
    preprocessor = model.named_steps["preprocessor"]
    columns = set()
    for name, _transformer, cols in preprocessor.transformers:
        if name == "remainder":
            continue
        if isinstance(cols, str):
            columns.add(cols)
        else:
            columns.update(cols)
    return columns


@pytest.mark.parametrize("filename,payload_keys", sorted(PAGE_PAYLOAD_KEYS.items()))
def test_every_ui_input_reaches_the_model(filename, payload_keys):
    from src.inference import load_model_bundle

    bundle = load_model_bundle(filename)
    if bundle is None:
        pytest.skip(f"{filename} not available locally")

    consumed = _consumed_columns(bundle["model"])
    dead = payload_keys - consumed
    assert not dead, (
        f"{filename}: UI sends {sorted(dead)}, which the fitted pipeline never reads. "
        "Either remove the control or retrain with those features."
    )
