# Customer Support Analytics — Technical Specification

---

## Document Control

| Field | Value |
|-------|-------|
| **Document** | PROJECT_SPEC.md |
| **Version** | 1.0 |
| **Status** | Active |
| **Last updated** | 2026-07-08 |
| **Repository** | [github.com/jegadeesh17/customer-support-ticket-analytics](https://github.com/jegadeesh17/customer-support-ticket-analytics) |
| **Related docs** | [README.md](../README.md), [DEMO.md](./DEMO.md), [reports/evaluation.md](../reports/evaluation.md) |

---

## 1. Executive Summary

Customer Support Analytics is a **multi-task machine learning platform** for support operations intelligence. It predicts ticket **priority** (4-class), **resolution hours** (regression), and **customer satisfaction** (3-band classification) from ~200K tickets combining TF-IDF text features with tabular metadata. Deployed via PostgreSQL, Streamlit multi-page dashboard, and FastAPI inference APIs.

**Interview pitch:**

> *"I built a multi-task ML platform on ~200K support tickets — priority at 82% accuracy, resolution R² 0.72, satisfaction at 93% — with sklearn ColumnTransformer pipelines, FastAPI endpoints, and honest documentation about engineered labels."*

---

## 2. Scope

### 2.1 In Scope

| # | Capability |
|---|------------|
| 1 | CSV → PostgreSQL ingestion |
| 2 | Rule-based label engineering for three targets |
| 3 | TF-IDF + tabular ColumnTransformer pipelines |
| 4 | Three separate trained models (priority, resolution, satisfaction) |
| 5 | Multi-page Streamlit dashboard |
| 6 | FastAPI `/predict_priority`, `/predict_resolution_hours` |
| 7 | pytest API and inference tests |
| 8 | EDA artifact generation |

### 2.2 Out of Scope

- Human-annotated ground truth labels
- Real-time streaming inference at scale
- BERT/transformer text models (future improvement)
- Model drift monitoring

---

## 3. Requirements

### 3.1 Functional Requirements

| ID | Requirement | Module | Status |
|----|-------------|--------|--------|
| FR-01 | Load CSV to PostgreSQL | `src/load_data_to_db.py` | ✅ |
| FR-02 | Engineer training labels | `src/label_engineering.py` | ✅ |
| FR-03 | Train three task models | `src/train_models.py` | ✅ |
| FR-04 | Export evaluation report | `scripts/export_evaluation.py` | ✅ |
| FR-05 | Inference with schema lock | `src/inference.py` | ✅ |
| FR-06 | Streamlit multi-page UI | `app/app.py`, `app/pages/*` | ✅ |
| FR-07 | REST prediction API | `api/main.py` | ✅ |

### 3.2 Non-Functional Requirements

| ID | Requirement | Target |
|----|-------------|--------|
| NFR-01 | Laptop-friendly training sample | ≤30K rows per task |
| NFR-02 | Stratified 80/20 split | Classification tasks |
| NFR-03 | Leakage columns dropped | Regression pipeline |
| NFR-04 | Metrics exported to markdown | `reports/evaluation.md` |

---

## 4. Architecture

```text
customer_support_ticket.csv
        │
        ▼
PostgreSQL ◀── load_data_to_db.py
        │
        ▼
label_engineering.py ──▶ targets (priority, hours, satisfaction)
        │
        ▼
ColumnTransformer (TF-IDF + tabular)
        │
        ├── Gradient Boosting → priority
        ├── Random Forest → resolution hours
        └── Random Forest → satisfaction
        │
        ▼
models/*.pkl ──▶ inference.py ──▶ FastAPI + Streamlit
```

---

## 5. Data Specification

| Field | Detail |
|-------|--------|
| Source file | `data/customer_support_ticket.csv` |
| Rows | ~200,000 |
| Columns | 30 (product, category, issue_description, channel, region, SLA flags, etc.) |
| Text field | `issue_description` (TF-IDF) |
| Labels | Engineered — see Section 8 |

---

## 6. Models & Metrics

| Task | Algorithm | Holdout Metric | Target |
|------|-----------|----------------|--------|
| Priority | Gradient Boosting | Accuracy 0.821 | ≥0.80 |
| Resolution hours | Random Forest | R² 0.719 | ≥0.70 |
| Satisfaction | Random Forest | Accuracy 0.931 | ≥0.75 |

Regenerate: `python src/train_models.py && python scripts/export_evaluation.py`

---

## 7. API Specification

### `GET /health`

Service and model artifact status.

### `POST /predict_priority`

**Input:** `TicketInput` — issue text + tabular metadata fields.  
**Output:** Predicted priority class + confidence.

### `POST /predict_resolution_hours`

**Input:** `TicketInput` (leakage-safe feature set).  
**Output:** Predicted resolution hours.

---

## 8. Label Engineering (Critical)

Targets are derived in `src/label_engineering.py` using business heuristics (urgency keywords, complexity signals, controlled noise). **This is not human-labeled ground truth.**

Interview framing: metrics validate pipeline engineering and train/serve consistency — not production KPI guarantees.

### Leakage Defenses

- Drop `resolution_time_hours`, `ticket_id` from regression features
- Inference builds full schema with defaults matching training
- Document shared-signal caveat between features and engineered targets

---

## 9. Deployment

```powershell
pip install -r requirements.txt
python src/load_data_to_db.py
python src/train_models.py
streamlit run app/app.py
uvicorn api.main:app --port 8002
pytest -q
```

---

## 10. Testing

`tests/test_api.py` — health, priority, resolution endpoints with mocked/sample payloads.

---

## 11. Module Index

| Path | Purpose |
|------|---------|
| `src/label_engineering.py` | Target generation rules |
| `src/model_trainer.py` | Pipeline builders per task |
| `src/train_models.py` | Training orchestration |
| `src/inference.py` | Load models + predict |
| `api/main.py` | FastAPI app |
| `notebooks/1_Classification_Priority.ipynb` | Notebook source of truth |

---

## 12. Future Improvements

- Human-labeled validation subset
- SHAP explainability in Streamlit
- BERT text encoder comparison
- Batch prediction export API
