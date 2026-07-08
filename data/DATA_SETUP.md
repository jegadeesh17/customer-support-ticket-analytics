# Data Setup

## Included in Git (demo / dashboard)

| File | Purpose |
|------|---------|
| `customer_support_ticket_sample.csv` | ~2K-row sample for notebooks, training smoke tests, and Streamlit |

## Full dataset (local only)

| File | Purpose |
|------|---------|
| `customer_support_ticket.csv` | Full ~200K-row dataset for production training |

**How to obtain:** Download from your original source (Kaggle or internal export) and place at `data/customer_support_ticket.csv`.

**Resolution order:** `src/paths.py` prefers the full file when present; otherwise falls back to the sample.

**Retrain:** `python src/train_models.py`
