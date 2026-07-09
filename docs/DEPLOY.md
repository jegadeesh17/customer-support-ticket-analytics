# Deploy Customer Support Analytics (Free Tier)

Zero-cost stack:

| Layer | Service |
|-------|---------|
| UI | Streamlit Community Cloud |
| API | GCP Cloud Run (always-free tier) |
| Models | Hugging Face Hub (public repo) |
| Database | Neon PostgreSQL (free tier) |
| CI/CD | GitHub Actions |

## Prerequisites

1. Public GitHub repo: [github.com/jegadeesh17/customer-support-ticket-analytics](https://github.com/jegadeesh17/customer-support-ticket-analytics)
2. [Hugging Face](https://huggingface.co) account
3. [Neon](https://neon.tech) account (free Postgres)
4. [Google Cloud](https://cloud.google.com) account for Cloud Run API
5. Trained models in `models/` (`python src/train_models.py`)

## Step 1 — Upload models to Hugging Face

```bash
pip install huggingface_hub
hf auth login

python scripts/upload_models_to_hf.py --repo-id jegadeesh17/support-ops-models
```

This uploads three `.pkl` bundles (~280 MB total). Use a **public** repo.

## Step 2 — Set up Neon PostgreSQL

1. Create a project at [neon.tech](https://neon.tech)
2. Copy the connection string (includes `?sslmode=require`)
3. Load ticket data locally, then push to Neon:

```bash
# In .env locally:
DATABASE_URL=postgresql://user:pass@ep-xxx.region.aws.neon.tech/neondb?sslmode=require

python src/load_data_to_db.py
```

For Streamlit Cloud, you can skip the DB and use the bundled sample CSV — the app falls back automatically.

## Step 3 — Deploy Streamlit UI (free)

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. New app → this repo → main file: `app/app.py`
3. Add secrets:

```toml
HF_MODEL_REPO = "jegadeesh17/support-ops-models"

# Optional — use Neon for live DB; omit to use sample CSV
DATABASE_URL = "postgresql://user:pass@ep-xxx.region.aws.neon.tech/neondb?sslmode=require"
```

4. Deploy. First load downloads model bundles from Hugging Face.

## Step 4 — GCP setup for Cloud Run API

Same as RiceLeafDetection — see [RiceLeafDetection/docs/DEPLOY.md](../RiceLeafDetection/docs/DEPLOY.md) Step 3, or run:

```bash
gcloud services enable run.googleapis.com artifactregistry.googleapis.com
gcloud artifacts repositories create ml-apis --repository-format=docker --location=asia-south1
```

Create the `github-deployer` service account and download `gcp-key.json` (same roles as RiceLeaf project).

## Step 5 — GitHub secrets

| Secret | Value |
|--------|-------|
| `GCP_PROJECT_ID` | Your GCP project ID |
| `GCP_SA_KEY` | Contents of `gcp-key.json` |
| `HF_MODEL_REPO` | `jegadeesh17/support-ops-models` |

## Step 6 — Deploy API to Cloud Run

1. **Actions** → **Deploy API to Cloud Run** → **Run workflow**
2. Test endpoints:

```bash
curl https://YOUR-SERVICE-xxx.run.app/health

curl -X POST https://YOUR-SERVICE-xxx.run.app/predict_priority \
  -H "Content-Type: application/json" \
  -d '{"issue_description": "Payment failed twice, need urgent help."}'

curl -X POST https://YOUR-SERVICE-xxx.run.app/predict_resolution_hours \
  -H "Content-Type: application/json" \
  -d '{"issue_description": "App crashes after latest update."}'

curl -X POST https://YOUR-SERVICE-xxx.run.app/predict_satisfaction \
  -H "Content-Type: application/json" \
  -d '{"issue_description": "Support was slow but issue got fixed."}'
```

## Local Docker test (optional)

```bash
docker build -t support-ops-api .
docker run -p 8080:8080 -e HF_MODEL_REPO=jegadeesh17/support-ops-models support-ops-api
```

## Interview talking points

- Multi-task ML API with three prediction endpoints
- Models served from Hugging Face; containers stay small
- Neon Postgres for cloud data layer with SSL
- Streamlit dashboard with CSV fallback for zero-DB demos

## Cost notes

- Neon free tier: 0.5 GB storage — enough for ticket data
- Hugging Face Hub: free for public models
- Cloud Run + Streamlit Cloud: free within demo limits
