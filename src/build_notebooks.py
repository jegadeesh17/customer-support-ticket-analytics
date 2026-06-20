"""Generate 10-step Jupyter notebooks that delegate to src/ modules."""

import json
import os

NOTEBOOK_DIR = os.path.join(os.path.dirname(__file__), '..', 'notebooks')


def _cell(cell_type, source, outputs=None):
    cell = {
        'cell_type': cell_type,
        'metadata': {},
        'source': source if isinstance(source, list) else [source],
    }
    if cell_type == 'code':
        cell['execution_count'] = None
        cell['outputs'] = outputs or []
    return cell


def _notebook(cells):
    return {
        'cells': cells,
        'metadata': {
            'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
            'language_info': {'name': 'python', 'version': '3.11.0'},
        },
        'nbformat': 4,
        'nbformat_minor': 5,
    }


def _setup_cells():
    return [
        _cell('code', '''import os
import sys
import warnings

import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')
PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

sns.set_theme(style='whitegrid')
print('Project root:', PROJECT_ROOT)'''),
    ]


def build_classification():
    cells = [
        _cell('markdown', '# Customer Support — Priority Classification\n\n## Step 1: Problem Definition\nPredict ticket **priority** (Urgent, High, Medium, Low) from text and metadata. Supports automatic triage and SLA-aware routing.'),
        *_setup_cells(),
        _cell('markdown', '## Step 2: Data Collection\nLoad tickets from PostgreSQL (CSV fallback).'),
        _cell('code', '''from src.data_loader import load_tickets

raw_df = load_tickets(use_db=True)
raw_df.head()'''),
        _cell('markdown', '## Step 3: Data Cleaning\nRemove IDs and leakage columns; standardize column names.'),
        _cell('code', '''from src.preprocessor import clean_data

cleaned_df = clean_data(raw_df, task_type='classification')
print('Shape:', cleaned_df.shape)
cleaned_df.isnull().sum().sort_values(ascending=False).head(10)'''),
        _cell('markdown', '### EDA: Priority distribution\nSee also `docs/eda/` for saved plots.'),
        _cell('code', '''cleaned_df['priority'].value_counts().plot(kind='bar', title='Priority Distribution')
plt.ylabel('Count')
plt.show()'''),
        _cell('markdown', '## Step 4: Feature Engineering\nText length, word count, and date-derived features. TF-IDF is applied inside the sklearn pipeline.'),
        _cell('code', '''from src.preprocessor import engineer_features

fe_df = engineer_features(cleaned_df, task_type='classification')
fe_df[['desc_length', 'desc_word_count', 'priority']].describe()'''),
        _cell('markdown', '## Step 5: Train-Test Split'),
        _cell('code', '''import pandas as pd
from sklearn.model_selection import train_test_split
from src.label_engineering import derive_priority_label
from src.preprocessor import clean_data, engineer_features

RANDOM_SEED = 42
df_cls = engineer_features(clean_data(raw_df, 'classification'), 'classification')
df_cls = df_cls.sample(min(30000, len(df_cls)), random_state=RANDOM_SEED)
df_cls['priority'] = derive_priority_label(df_cls)
X = df_cls.drop(columns=['priority'])
for col in X.select_dtypes(include=['object']).columns:
    X[col] = X[col].fillna('Unknown')
y = df_cls['priority']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y)
print('Train:', X_train.shape, 'Test:', X_test.shape)'''),
        _cell('markdown', '## Step 6: Model Selection\nCompare Logistic Regression, Random Forest, Gradient Boosting, and MLP (neural network) with TF-IDF + tabular features.'),
        _cell('code', '''from sklearn.metrics import accuracy_score, f1_score
from src.model_trainer import get_classification_pipelines

results = {}
for name, pipe in get_classification_pipelines(X_train).items():
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    results[name] = {'f1': f1_score(y_test, pred, average='weighted'), 'acc': accuracy_score(y_test, pred)}
    print(f"{name}: F1={results[name]['f1']:.4f}  Acc={results[name]['acc']:.4f}")

best_name = max(results, key=lambda k: results[k]['f1'])
print('\\nBest:', best_name)'''),
        _cell('markdown', '## Step 7: Model Training\nTrain the best pipeline on the full training split.'),
        _cell('code', '''clf = get_classification_pipelines(X_train)[best_name]
clf.fit(X_train, y_train)
print('Training complete.')'''),
        _cell('markdown', '## Step 8: Model Evaluation'),
        _cell('code', '''from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

y_pred = clf.predict(X_test)
print('Accuracy:', accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred), display_labels=clf.classes_).plot()
plt.title('Confusion Matrix')
plt.show()'''),
        _cell('markdown', '## Step 9: Model Tuning\nUse GridSearchCV on the best model type (optional — run for final submission).'),
        _cell('code', '''from sklearn.model_selection import GridSearchCV

# Example tuning for Random Forest
param_grid = {'classifier__n_estimators': [100, 150], 'classifier__max_depth': [None, 20]}
grid = GridSearchCV(clf, param_grid, cv=3, scoring='f1_weighted', n_jobs=-1)
# grid.fit(X_train, y_train)  # Uncomment for full tuning
print('Tuning grid defined. Uncomment grid.fit to run.')'''),
        _cell('markdown', '## Step 10: Deployment\nSave model bundle for Streamlit.'),
        _cell('code', '''import joblib
from src.paths import get_models_dir

acc = accuracy_score(y_test, y_pred)
bundle = {'model': clf, 'model_name': best_name, 'accuracy': acc}
path = os.path.join(get_models_dir(), 'classification_model.pkl')
joblib.dump(bundle, path)
print(f'Saved {path}  Accuracy={acc:.4f}')'''),
    ]
    return _notebook(cells)


def build_regression():
    cells = [
        _cell('markdown', '# Customer Support — Resolution Time Regression\n\n## Step 1: Problem Definition\nPredict **resolution time in hours** to support SLA planning and staffing.'),
        *_setup_cells(),
        _cell('markdown', '## Step 2: Data Collection'),
        _cell('code', '''from src.data_loader import load_tickets
raw_df = load_tickets(use_db=True)
raw_df.head()'''),
        _cell('markdown', '## Step 3: Data Cleaning\nDrop leakage columns (resolution notes, status, resolved date, satisfaction).'),
        _cell('code', '''from src.preprocessor import clean_data
cleaned_df = clean_data(raw_df, task_type='regression')
cleaned_df['resolution_time_hours'].describe()'''),
        _cell('markdown', '### EDA: Resolution time distribution'),
        _cell('code', '''import seaborn as sns
sns.histplot(cleaned_df['resolution_time_hours'], kde=True, bins=40)
plt.title('Resolution Time Distribution')
plt.show()'''),
        _cell('markdown', '## Step 4: Feature Engineering\nCap outliers, derive date features, keep text for TF-IDF.'),
        _cell('code', '''from src.preprocessor import engineer_features
from src.label_engineering import derive_resolution_hours
import numpy as np

fe_df = engineer_features(cleaned_df, task_type='regression')
fe_df = fe_df.sample(min(30000, len(fe_df)), random_state=42)
fe_df['resolution_time_hours'] = derive_resolution_hours(fe_df)
print('Rows:', len(fe_df))'''),
        _cell('markdown', '## Step 5: Train-Test Split\nLog-transform target for stable regression.'),
        _cell('code', '''from sklearn.model_selection import train_test_split

X = fe_df.drop(columns=['resolution_time_hours'])
for col in X.select_dtypes(include=['object']).columns:
    X[col] = X[col].fillna('Unknown')
y = np.log1p(fe_df['resolution_time_hours'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print('Train:', X_train.shape)'''),
        _cell('markdown', '## Step 6: Model Selection\nLinear Regression, Random Forest, and MLP neural network.'),
        _cell('code', '''from sklearn.metrics import r2_score, mean_squared_error
from src.model_trainer import get_regression_pipelines

for name, pipe in get_regression_pipelines(X_train).items():
    pipe.fit(X_train, y_train)
    pred = np.expm1(pipe.predict(X_test))
    true = np.expm1(y_test)
    print(f"{name}: R2={r2_score(true, pred):.4f}  RMSE={mean_squared_error(true, pred, squared=False):.2f}")'''),
        _cell('markdown', '## Step 7: Model Training'),
        _cell('code', '''reg = get_regression_pipelines(X_train)['Random Forest']
reg.fit(X_train, y_train)'''),
        _cell('markdown', '## Step 8: Model Evaluation'),
        _cell('code', '''y_pred = np.expm1(reg.predict(X_test))
y_true = np.expm1(y_test)
print('R2:', r2_score(y_true, y_pred))
plt.scatter(y_true, y_pred, alpha=0.2)
plt.xlabel('Actual'); plt.ylabel('Predicted')
plt.title('Actual vs Predicted Resolution Time')
plt.show()'''),
        _cell('markdown', '## Step 9: Model Tuning\nGridSearchCV on Random Forest (optional).'),
        _cell('code', "print('Define param_grid for regressor tuning as needed.')"),
        _cell('markdown', '## Step 10: Deployment'),
        _cell('code', '''import joblib
from src.paths import get_models_dir

r2 = r2_score(y_true, y_pred)
bundle = {'model': reg, 'model_name': 'Random Forest', 'r2': r2, 'log_target': True}
joblib.dump(bundle, os.path.join(get_models_dir(), 'regression_model.pkl'))
print('Saved regression model. R2=', r2)'''),
    ]
    return _notebook(cells)


def build_satisfaction():
    cells = [
        _cell('markdown', '# Customer Support — Satisfaction Prediction\n\n## Step 1: Problem Definition\nPredict **customer satisfaction class** (Low / Mid / High) from ticket and service features.'),
        *_setup_cells(),
        _cell('markdown', '## Step 2: Data Collection'),
        _cell('code', '''from src.data_loader import load_tickets
raw_df = load_tickets(use_db=True)
raw_df.head()'''),
        _cell('markdown', '## Step 3: Data Cleaning'),
        _cell('code', '''from src.preprocessor import clean_data
cleaned_df = clean_data(raw_df, task_type='satisfaction')
cleaned_df['customer_satisfaction_score'].value_counts().sort_index()'''),
        _cell('markdown', '### EDA: Satisfaction by channel'),
        _cell('code', '''import seaborn as sns
sns.boxplot(data=cleaned_df, x='channel', y='customer_satisfaction_score')
plt.xticks(rotation=45)
plt.title('Satisfaction by Channel')
plt.show()'''),
        _cell('markdown', '## Step 4: Feature Engineering\nMap 1–5 score to Low / Mid / High classes.'),
        _cell('code', '''from src.preprocessor import engineer_features
from src.label_engineering import derive_satisfaction_class

fe_df = engineer_features(cleaned_df, task_type='satisfaction')
fe_df['satisfaction_class'] = derive_satisfaction_class(fe_df)
fe_df = fe_df.sample(min(30000, len(fe_df)), random_state=42)
fe_df['satisfaction_class'].value_counts()'''),
        _cell('markdown', '## Step 5: Train-Test Split'),
        _cell('code', '''from sklearn.model_selection import train_test_split

X = fe_df.drop(columns=['customer_satisfaction_score', 'satisfaction_class'])
for col in X.select_dtypes(include=['object']).columns:
    X[col] = X[col].fillna('Unknown')
y = fe_df['satisfaction_class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print('Train:', X_train.shape)'''),
        _cell('markdown', '## Step 6: Model Selection'),
        _cell('code', '''from sklearn.metrics import accuracy_score, f1_score
from src.model_trainer import get_satisfaction_pipelines

for name, pipe in get_satisfaction_pipelines(X_train).items():
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    print(f"{name}: Acc={accuracy_score(y_test, pred):.4f}  F1={f1_score(y_test, pred, average='weighted'):.4f}")'''),
        _cell('markdown', '## Step 7: Model Training'),
        _cell('code', '''sat = get_satisfaction_pipelines(X_train)['Random Forest']
sat.fit(X_train, y_train)'''),
        _cell('markdown', '## Step 8: Model Evaluation'),
        _cell('code', '''from sklearn.metrics import classification_report

y_pred = sat.predict(X_test)
print('Accuracy:', accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))'''),
        _cell('markdown', '## Step 9: Model Tuning'),
        _cell('code', "print('Optional GridSearchCV for satisfaction model.')"),
        _cell('markdown', '## Step 10: Deployment'),
        _cell('code', '''import joblib
from src.paths import get_models_dir

acc = accuracy_score(y_test, y_pred)
joblib.dump({'model': sat, 'model_name': 'Random Forest', 'accuracy': acc},
            os.path.join(get_models_dir(), 'satisfaction_model.pkl'))
print('Saved satisfaction model. Accuracy=', acc)'''),
    ]
    return _notebook(cells)


def main():
    os.makedirs(NOTEBOOK_DIR, exist_ok=True)
    specs = [
        ('1_Classification_Priority.ipynb', build_classification),
        ('2_Regression_Resolution_Time.ipynb', build_regression),
        ('3_Satisfaction_Prediction.ipynb', build_satisfaction),
    ]
    for filename, builder in specs:
        path = os.path.join(NOTEBOOK_DIR, filename)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(builder(), f, indent=1)
        print('Wrote', path)

    old_seg = os.path.join(NOTEBOOK_DIR, '3_Segmentation_Customer_Profiles.ipynb')
    if os.path.exists(old_seg):
        os.remove(old_seg)
        print('Removed', old_seg)


if __name__ == '__main__':
    main()
