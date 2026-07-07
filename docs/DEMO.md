# Support Ops Intelligence — 5-Minute Demo

## Setup
```powershell
cd CustomerSupportAnalytics
pip install -r requirements.txt
python src/train_models.py
python scripts/export_evaluation.py
streamlit run app/app.py
```

## API Demo
```powershell
uvicorn api.main:app --port 8002
```
```powershell
curl -X POST http://localhost:8002/predict_priority -H "Content-Type: application/json" -d "{\"issue_description\": \"Payment failed twice after renewal, account locked.\"}"
```

## Talking Points
- Multi-task ML: priority + resolution hours + satisfaction
- TF-IDF + tabular features with sklearn pipelines
- Honest caveat: engineered labels for pipeline demonstration
