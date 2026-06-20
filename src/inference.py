"""Shared helpers for Streamlit inference pages."""

import os

import joblib
import numpy as np
import pandas as pd

from src.constants import DEFAULT_INFERENCE_ROW
from src.paths import get_data_path, get_models_dir
from src.preprocessor import build_inference_row, preprocess_for_inference


def load_sample_ticket():
    """Return a random ticket row from CSV merged with defaults."""
    csv_path = get_data_path()
    if not os.path.exists(csv_path):
        return dict(DEFAULT_INFERENCE_ROW)
    df = pd.read_csv(csv_path)
    row = df.sample(1).iloc[0].to_dict()
    normalized = {k.lower().replace(' ', '_'): v for k, v in row.items()}
    return {**DEFAULT_INFERENCE_ROW, **normalized}


def load_model_bundle(filename):
    path = os.path.join(get_models_dir(), filename)
    if not os.path.exists(path):
        return None
    return joblib.load(path)


def _align_to_training_columns(processed, model):
    """Ensure inference DataFrame has all columns expected by the fitted pipeline."""
    preprocessor = model.named_steps.get('preprocessor')
    if preprocessor is None:
        return processed

    expected = set()
    for name, trans, cols in preprocessor.transformers:
        if name == 'remainder':
            continue
        if isinstance(cols, str):
            expected.add(cols)
        else:
            expected.update(cols)

    aligned = processed.copy()
    for col in expected:
        if col not in aligned.columns:
            if col == 'issue_description':
                aligned[col] = ''
            elif col in DEFAULT_INFERENCE_ROW:
                aligned[col] = DEFAULT_INFERENCE_ROW[col]
            else:
                aligned[col] = 0
    return aligned


def _prepare_features(raw_df, task_type, model):
    processed = preprocess_for_inference(raw_df, task_type=task_type)
    for col in processed.select_dtypes(include=['object']).columns:
        processed[col] = processed[col].fillna('Unknown')
    return _align_to_training_columns(processed, model)


def predict_classification(user_inputs):
    bundle = load_model_bundle('classification_model.pkl')
    if bundle is None:
        raise FileNotFoundError('classification_model.pkl not found. Run: python src/train_models.py')
    model = bundle['model']
    processed = _prepare_features(build_inference_row(user_inputs), 'classification', model)
    return model.predict(processed)[0]


def predict_regression(user_inputs):
    bundle = load_model_bundle('regression_model.pkl')
    if bundle is None:
        raise FileNotFoundError('regression_model.pkl not found. Run: python src/train_models.py')
    model = bundle['model']
    processed = _prepare_features(build_inference_row(user_inputs), 'regression', model)
    pred = model.predict(processed)[0]
    if bundle.get('log_target'):
        pred = float(np.expm1(pred))
    return pred


def predict_satisfaction(user_inputs):
    bundle = load_model_bundle('satisfaction_model.pkl')
    if bundle is None:
        raise FileNotFoundError('satisfaction_model.pkl not found. Run: python src/train_models.py')
    model = bundle['model']
    processed = _prepare_features(build_inference_row(user_inputs), 'satisfaction', model)
    return model.predict(processed)[0]
