"""Tests for Tier-1 confidence scoring used by the escalation gate."""

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.inference import predict_classification_with_confidence, load_model_bundle, _prepare_features
from src.preprocessor import build_inference_row
from src.constants import DEFAULT_INFERENCE_ROW, PRIORITY_LEVELS


def test_confidence_is_a_valid_probability():
    label, confidence = predict_classification_with_confidence(DEFAULT_INFERENCE_ROW)
    assert label in PRIORITY_LEVELS
    assert 0.0 <= confidence <= 1.0


def test_confidence_matches_manual_predict_proba_max():
    import numpy as np

    bundle = load_model_bundle('classification_model.pkl')
    model = bundle['model']
    processed = _prepare_features(build_inference_row(DEFAULT_INFERENCE_ROW), 'classification', model)
    expected_confidence = float(np.max(model.predict_proba(processed)[0]))

    _, confidence = predict_classification_with_confidence(DEFAULT_INFERENCE_ROW)
    assert confidence == expected_confidence
