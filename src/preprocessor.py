import pandas as pd

from src.constants import ID_COLS, LEAKAGE_COLS, TEXT_COL, DEFAULT_INFERENCE_ROW
from src.label_engineering import _text_score


def standardize_columns(df):
    df = df.copy()
    df.columns = [col.lower().replace(' ', '_') for col in df.columns]
    return df


def clean_data(df, task_type='classification', drop_leakage=True):
    """Standardize columns and remove non-predictive IDs and leakage columns."""
    df_clean = standardize_columns(df)
    cols_to_drop = list(ID_COLS)

    if drop_leakage:
        if task_type == 'regression':
            leakage = [c for c in LEAKAGE_COLS if c != 'resolution_time_hours']
        elif task_type == 'satisfaction':
            leakage = [c for c in LEAKAGE_COLS if c not in ('customer_satisfaction_score', 'first_response_time_hours', 'escalated', 'sla_breached')]
        else:
            leakage = list(LEAKAGE_COLS)
        cols_to_drop.extend(leakage)

    df_clean = df_clean.drop(columns=[c for c in cols_to_drop if c in df_clean.columns], errors='ignore')
    return df_clean


def engineer_features(df, task_type='classification'):
    """Create derived features from raw ticket data."""
    df_fe = df.copy()

    if TEXT_COL in df_fe.columns:
        df_fe['desc_length'] = df_fe[TEXT_COL].fillna('').astype(str).str.len()
        df_fe['desc_word_count'] = df_fe[TEXT_COL].fillna('').astype(str).str.split().str.len()
        df_fe['text_urgency_score'] = df_fe[TEXT_COL].fillna('').apply(_text_score)

    if 'category' in df_fe.columns:
        df_fe['category_length'] = df_fe['category'].fillna('').astype(str).str.len()

    if 'ticket_created_date' in df_fe.columns:
        dates = pd.to_datetime(df_fe['ticket_created_date'], errors='coerce')
        df_fe['created_month'] = dates.dt.month
        df_fe['created_dow'] = dates.dt.dayofweek
        df_fe['created_is_weekend'] = (dates.dt.dayofweek >= 5).astype(int)
        df_fe = df_fe.drop(columns=['ticket_created_date'])

    for col in ['resolution_time_hours', 'first_response_time_hours', 'customer_age',
                'customer_tenure_months', 'previous_tickets', 'issue_complexity_score']:
        if col in df_fe.columns:
            df_fe[col] = pd.to_numeric(df_fe[col], errors='coerce')

    if task_type == 'regression' and 'resolution_time_hours' in df_fe.columns:
        df_fe['resolution_time_hours'] = pd.to_numeric(df_fe['resolution_time_hours'], errors='coerce')

    if task_type == 'satisfaction' and 'customer_satisfaction_score' in df_fe.columns:
        df_fe['customer_satisfaction_score'] = pd.to_numeric(
            df_fe['customer_satisfaction_score'], errors='coerce'
        )

    return df_fe


def build_inference_row(user_inputs=None):
    """Merge user inputs with defaults for Streamlit prediction."""
    row = dict(DEFAULT_INFERENCE_ROW)
    if user_inputs:
        row.update(user_inputs)
    return pd.DataFrame([row])


def preprocess_for_inference(df, task_type='classification'):
    """Prepare raw inference data (keeps text column for pipeline TF-IDF)."""
    df_clean = clean_data(df, task_type=task_type, drop_leakage=False)
    if task_type == 'regression':
        df_clean = df_clean.drop(columns=['resolution_time_hours'], errors='ignore')
    if task_type == 'satisfaction':
        df_clean = df_clean.drop(columns=['customer_satisfaction_score'], errors='ignore')
    return engineer_features(df_clean, task_type=task_type)
