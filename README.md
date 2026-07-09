# Customer Support Tickets Analytics — Support Ops Intelligence
---
### **Project Overview**
Machine learning platform for customer support ticket analysis. The project covers three tasks aligned with the course requirements:

1. **Priority Classification** — Multi-class prediction of ticket priority (Urgent, High, Medium, Low) using TF-IDF text features and tabular metadata with traditional ML and neural network (MLP) models.
2. **Resolution Time Regression** — Predict resolution time in hours with leakage-safe features, log-transformed target, and neural network comparison.
3. **Customer Satisfaction** — Classify satisfaction into Low / Mid / High bands from ticket and service attributes.

Data is stored in PostgreSQL; Jupyter notebooks in `notebooks/` follow the standard 10-step ML process; production training and inference live in `src/`.

**Repository:** [github.com/jegadeesh17/customer-support-ticket-analytics](https://github.com/jegadeesh17/customer-support-ticket-analytics)  
**Full specification:** [docs/PROJECT_SPEC.md](docs/PROJECT_SPEC.md)

---
### **Key Features**
- PostgreSQL data warehouse integration
- TF-IDF NLP on `issue_description` combined with structured ticket features
- Class imbalance handling via `class_weight='balanced'`
- Multi-model comparison (Logistic Regression, Random Forest, Gradient Boosting, MLP)
- EDA generation supported via `python src/generate_eda.py`
- Multi-page Streamlit dashboard with sample-ticket loading

---
### **Dataset**
- **Source:** `data/customer_support_ticket.csv` (~200,000 rows, 30 columns)
- **In repo:** `data/customer_support_ticket_sample.csv` for clone-friendly demos
- **Full data:** Place `customer_support_ticket.csv` in `data/` for training — see [data/DATA_SETUP.md](data/DATA_SETUP.md)
- **Key fields:** product, category, issue_description, priority, channel, region, subscription_type, resolution_time_hours, customer_satisfaction_score, SLA flags, and customer metadata

---
### **Project Structure**
```
CustomerSupportAnalytics/
├── app/                  # Streamlit application
│   ├── app.py
│   └── pages/
├── data/                 # Raw CSV / Excel
├── docs/                 # Project docs and EDA plots
├── models/               # Saved .pkl model bundles
├── notebooks/            # 10-step Jupyter notebooks
├── src/                  # Training, EDA, DB, inference
├── requirements.txt
├── .env                  # Database credentials (not committed)
└── README.md
```

---
### **How It Works**
1. Load CSV into PostgreSQL: `python src/load_data_to_db.py`
2. Generate EDA plots: `python src/generate_eda.py`
3. Train models: `python src/train_models.py`
4. Explore workflows in `notebooks/` (same logic as `src/`)
5. Launch Streamlit: `streamlit run app/app.py`

Inference builds a full feature row from user inputs plus defaults so sklearn pipelines receive the same schema as training.

**Note:** The bundled CSV has weak label signal in raw columns. Training uses rule-based label engineering (`src/label_engineering.py`) so models learn meaningful patterns from text urgency, complexity, and service metrics — suitable for end-to-end pipeline demonstration.

### Data Caveats and Mitigation
- This project is built as an **applied pipeline demonstration** where targets are engineered from business-style heuristics.
- Priority, resolution hours, and satisfaction labels are derived using deterministic rules plus controlled noise in `src/label_engineering.py`.
- Because targets are engineered, very high scores can occur when model features strongly align with generation rules.
- To avoid over-claiming, treat reported metrics as **pipeline validity indicators**, not production KPI guarantees from human-labeled ground truth.

### Leakage Defense Notes
- Regression training intentionally drops direct leakage columns (`resolution_time_hours`, `ticket_id`) and only keeps usable predictors.
- Inference path builds full feature schema consistently so train/serve mismatch is minimized.
- Remaining caveat: some engineered targets and feature inputs share related signals by design; this is disclosed and should be discussed transparently in interviews.

---
### **Model Performance**
Run `python src/train_models.py` to print current metrics. Targets from the project brief:

| Task | Target |
|------|--------|
| Priority classification | ≥ 80% accuracy |
| Resolution time regression | R² ≥ 0.70 |
| Satisfaction classification | ≥ 75% accuracy |

Interpretation guidance:
- If observed metrics are near-perfect, validate against caveat notes above before claiming real-world generalization.
- Prefer reporting this project as an end-to-end ML system design and modeling workflow demonstration.

---
### **Interactive Application Deployment**
```bash
streamlit run app/app.py
uvicorn api.main:app --port 8002
```
Use **Load sample ticket** on each page to populate the form from the dataset, then submit to get predictions.

**Evaluation report:** `reports/evaluation.md` (generated after training)

#### **Sample Input / Output**
- Input ticket: "Payment failed twice after renewal, account locked, need urgent access."
- Typical output:
  - Priority page -> `High/Urgent`
  - Regression page -> predicted resolution hours
  - Satisfaction page -> risk band (`Low/Mid/High`)

#### **Metrics + Limitations Block**
- Targets: Priority >=0.80 acc, Regression R2 >=0.70, Satisfaction >=0.75 acc
- Strength: end-to-end multi-task ML pipeline with DB + UI integration
- Limitation: labels are engineered (not human-annotated ground truth), so near-perfect metrics should be interpreted cautiously

---
### **Technology Stack**
- Python, pandas, scikit-learn, SQLAlchemy, PostgreSQL
- Streamlit, matplotlib, seaborn, joblib
- Jupyter notebooks

---
### **Getting Started**
### **1. Clone Repository**
```bash
git clone https://github.com/jegadeesh17/customer-support-ticket-analytics.git
cd customer-support-ticket-analytics
```

### **2. Install Dependencies**
```bash
pip install -r requirements.txt
```

### **3. Configure Database**
Create `.env` in the project root:
```
DB_HOST=localhost
DB_USER=postgres
DB_PASSWORD=your_password
DB_NAME=customer_support_ticket
DB_PORT=5432
```

### **4. Load Data & Train**
```bash
python src/load_data_to_db.py
python src/generate_eda.py
python src/train_models.py
```
Model files (`.pkl`) are generated locally and excluded from git due to size. Run `train_models.py` after cloning.

### **5. Launch Dashboard**
```bash
streamlit run app/app.py
```

---
### **Example Use Case**
A support agent enters a ticket description and channel. The **Priority** page predicts urgency for routing; the **Regression** page estimates SLA hours; the **Satisfaction** page flags at-risk customers for proactive follow-up.

---
### **Future Improvements**
- BERT / LSTM text models for priority classification
- SHAP explanations in the Streamlit UI
- Batch prediction and export
- Streamlit Cloud deployment

---
### **Contributors**
- jegadeesh17

---
### **License**
MIT (or as required by your institution)
