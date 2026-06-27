import os
import sys
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_loader import load_tickets
from src.label_engineering import (
    derive_priority_label,
    derive_resolution_hours,
    derive_satisfaction_class,
)
from src.model_trainer import (
    get_classification_pipelines,
    get_regression_pipelines,
    get_satisfaction_pipelines,
)
from src.paths import get_models_dir
from src.preprocessor import clean_data, engineer_features

RANDOM_SEED = 42
TRAIN_SAMPLE = 30000


def _sample_df(df, n=TRAIN_SAMPLE):
    if len(df) > n:
        return df.sample(n, random_state=RANDOM_SEED)
    return df


def _fill_text_column(X):
    X = X.copy()
    if 'issue_description' in X.columns:
        X['issue_description'] = X['issue_description'].fillna('')
    return X


def train_classification(df, models_dir):
    print('\n=== Classification (Priority) ===')
    df = df.copy()
    df['priority'] = derive_priority_label(df)
    df_cls = clean_data(df, 'classification')
    df_cls = engineer_features(df_cls, 'classification')
    df_cls = _sample_df(df_cls)

    X = _fill_text_column(df_cls.drop(columns=['priority']))
    y = df_cls['priority']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )

    best_name, best_model, best_score = None, None, 0.0
    for name, pipe in get_classification_pipelines(X_train).items():
        print(f'Training {name}...')
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        score = f1_score(y_test, y_pred, average='weighted')
        acc = accuracy_score(y_test, y_pred)
        print(f'  F1={score:.4f}  Accuracy={acc:.4f}')
        if score > best_score:
            best_score, best_name, best_model = score, name, pipe

    y_pred = best_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f'\nBest model: {best_name}  Accuracy={acc:.4f}')
    print(classification_report(y_test, y_pred))

    path = os.path.join(models_dir, 'classification_model.pkl')
    joblib.dump({'model': best_model, 'model_name': best_name, 'accuracy': acc}, path)
    print(f'Saved {path}')
    return acc


def train_regression(df, models_dir):
    print('\n=== Regression (Resolution Time) ===')
    df = df.copy()
    df['resolution_time_hours'] = derive_resolution_hours(df)
    df_reg = clean_data(df, 'regression')
    df_reg = engineer_features(df_reg, 'regression')
    df_reg = _sample_df(df_reg)

    X = _fill_text_column(df_reg.drop(columns=['resolution_time_hours']))
    y = np.log1p(df_reg['resolution_time_hours'])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED
    )

    best_name, best_model, best_r2 = None, None, -np.inf
    for name, pipe in get_regression_pipelines(X_train).items():
        print(f'Training {name}...')
        try:
            pipe.fit(X_train, y_train)
            y_pred_log = pipe.predict(X_test)
        except Exception as exc:
            print(f'  Skipped ({exc})')
            continue
        y_pred = np.expm1(y_pred_log)
        y_true = np.expm1(y_test)
        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        print(f'  R2={r2:.4f}  RMSE={rmse:.2f}')
        if r2 > best_r2:
            best_r2, best_name, best_model = r2, name, pipe

    y_pred_log = best_model.predict(X_test)
    y_pred = np.expm1(y_pred_log)
    y_true = np.expm1(y_test)
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    print(f'\nBest model: {best_name}  R2={r2:.4f}  MAE={mae:.2f}')

    path = os.path.join(models_dir, 'regression_model.pkl')
    joblib.dump({
        'model': best_model,
        'model_name': best_name,
        'r2': r2,
        'log_target': True,
    }, path)
    print(f'Saved {path}')
    return r2


def train_satisfaction(df, models_dir):
    print('\n=== Satisfaction Classification ===')
    df = df.copy()
    df['satisfaction_class'] = derive_satisfaction_class(df)
    df_sat = clean_data(df, 'satisfaction')
    df_sat = engineer_features(df_sat, 'satisfaction')
    df_sat = _sample_df(df_sat)

    X = _fill_text_column(df_sat.drop(columns=['satisfaction_class']))
    y = df_sat['satisfaction_class']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )

    best_name, best_model, best_score = None, None, 0.0
    for name, pipe in get_satisfaction_pipelines(X_train).items():
        print(f'Training {name}...')
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        score = f1_score(y_test, y_pred, average='weighted')
        acc = accuracy_score(y_test, y_pred)
        print(f'  F1={score:.4f}  Accuracy={acc:.4f}')
        if score > best_score:
            best_score, best_name, best_model = score, name, pipe

    y_pred = best_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f'\nBest model: {best_name}  Accuracy={acc:.4f}')
    print(classification_report(y_test, y_pred))

    path = os.path.join(models_dir, 'satisfaction_model.pkl')
    joblib.dump({'model': best_model, 'model_name': best_name, 'accuracy': acc}, path)
    print(f'Saved {path}')
    return acc


def train_and_save_models():
    models_dir = get_models_dir()
    os.makedirs(models_dir, exist_ok=True)

    df = load_tickets(use_db=True)
    if df.empty:
        df = load_tickets(use_db=False)
    if df.empty:
        print('No data available for training.')
        return

    cls_acc = train_classification(df, models_dir)
    reg_r2 = train_regression(df, models_dir)
    sat_acc = train_satisfaction(df, models_dir)

    print('\n=== Training Summary ===')
    print(f'Classification accuracy: {cls_acc:.4f} (target >= 0.80)')
    print(f'Regression R2:           {reg_r2:.4f} (target >= 0.70)')
    print(f'Satisfaction accuracy:   {sat_acc:.4f} (target >= 0.75)')


if __name__ == '__main__':
    train_and_save_models()
