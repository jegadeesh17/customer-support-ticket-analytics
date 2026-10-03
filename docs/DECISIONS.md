# Architecture Decisions — Customer Support Analytics

> Lightweight architecture decision records (ADRs). The technical specification is [SPEC.md](./SPEC.md).

Rationale here is taken only from recorded sources (code comments, commit messages, tracked reports). Where no reason was recorded, the entry says so and lists only the observable trade-off.

---

## ADR-01: Two-tier triage with a rule-based escalation gate

**Context:** Tier 1 (classical sklearn models) is fast and cheap; Tier 2 (LLM diagnosis) is slower and costs money. Running Tier 2 on every ticket would waste both.
**Decision:** Tier 1 always runs. A pure function `should_escalate` (`src/triage_gate.py:22`) sends a ticket to Tier 2 only when confidence is below 0.67, predicted resolution exceeds 185.0 hours, or an Enterprise ticket has complexity above 8. `POST /triage_agent?force=true` bypasses the gate (`api/main.py:169-170`). Commit `a224197` added the Tier-1 confidence score the gate needs.
**Alternatives rejected:** The original fixed thresholds of 0.80 confidence and 48.0 hours were dropped because they "fired on the majority of tickets, not a minority" (comment at `src/triage_gate.py:10-16`, commit `af6d5e0`). Always running Tier 2 is not recorded as considered.
**Consequences:** Tier 2 runs on roughly 30 percent of tickets: the benchmark in `reports/ESCALATION_GATE_BENCHMARK.md` reports 69.5% handled by Tier 1 over 1,000 tickets (147 low confidence, 140 severe resolution, 18 high-risk segment). Evidence: `src/triage_gate.py:17-19`, commits `a224197`, `af6d5e0`, `153e549`.

---

## ADR-02: Percentile-derived thresholds, measured with a different seed

**Context:** The first gate thresholds were never checked against the real data distribution (see ADR-01).
**Decision:** Thresholds are percentiles measured on a 500-ticket sample run through the real trained models: P15 of confidence (0.67) and P85 of predicted resolution hours (185.0). The benchmark script samples with `random_state=7`, deliberately different from the seed 42 used to derive the thresholds, so the reported rate is not circular (comment in `scripts/benchmark_escalation_gate.py`, commit `153e549`).
**Alternatives rejected:** None recorded.
**Consequences:** Both samples come from the same CSV. The models may also have been trained on it (`src/paths.py:get_data_path` prefers the full dataset and falls back to the sample), so the 69.5% figure is a gate-rate measurement, not an accuracy claim. Evidence: `src/triage_gate.py:10-16`, commits `af6d5e0`, `901177b`, `153e549`.

---

## ADR-03: Deterministic heuristic fallback for Tier 2

**Context:** The LLM call can fail (network error, bad key, malformed JSON) or no provider may be configured.
**Decision:** `run_agent_triage` catches any exception from the provider call and returns `_extract_heuristics`, a keyword and metadata scoring routine, with `triage_source="heuristic_fallback"`. The provider request has a 5 second timeout. With no provider configured it goes straight to the heuristics (`src/agent_triage.py:32`, `:184`, `:191-194`, commit `bca4f0c`).
**Alternatives rejected:** None recorded.
**Consequences:** `/triage_agent` does not fail when the LLM is down, but the response can come from heuristics without any signal other than `triage_source`. Tier 2 is not retried; there is no circuit breaker. A retired or misspelled model name degrades silently to heuristics (see ADR-05).

---

## ADR-04: Groq first, then OpenRouter, then OpenAI

**Context:** Tier 2 needs a low-latency LLM endpoint with an OpenAI-compatible API.
**Decision:** `_select_provider` (`src/agent_triage.py:108`) picks the first configured key in the order Groq (`GROQ_API_KEY`, model `GROQ_MODEL`), OpenRouter (`google/gemini-2.0-flash-001`), OpenAI (`gpt-4o-mini`). Commit `26bca59` added Groq as the primary provider; the docstring describes it as "fast".
**Alternatives rejected:** No comparison between providers is recorded.
**Consequences:** Only one provider is tried per request; there is no cross-provider failover, only the heuristic fallback of ADR-03. Cloud Run receives `GROQ_API_KEY` as an env var from `.github/workflows/deploy.yml`.

---

## ADR-05: Groq model default `llama-3.3-70b-versatile` (status UNVERIFIED)

**Context:** `GROQ_MODEL` needs a default.
**Decision:** The default is `llama-3.3-70b-versatile` (`configs/settings.py`).
**Alternatives rejected:** None recorded.
**Consequences:** UNVERIFIED: in the sibling repo SuperKalamProject, commit `db271f3` records that `llama-3.3-70b-versatile` was retired from Groq's catalog (a 404 was confirmed there), so this default may be retired for this repo too; that has not been checked against this repo's live service. If it is, every Tier 2 call fails and silently falls back to heuristics (ADR-03). This repo contains no test that calls Groq.

---

## ADR-06: Model bundles hosted on Hugging Face Hub, downloaded at runtime

**Context:** The three `.pkl` bundles are about 280 MB together (`regression_model.pkl` alone is 218 MB, per `src/inference.py`) and are excluded from git (`.gitignore`: `models/*.pkl`).
**Decision:** Bundles are uploaded to a public Hugging Face repo with `scripts/upload_models_to_hf.py`. `ensure_models` (`src/model_assets.py:25`) downloads any missing bundle when `HF_MODEL_REPO` is set. The Docker image contains an empty `models/` directory, so Cloud Run fetches the bundles at runtime. Commit `1b8d33d` introduced this, together with Cloud Run, Neon and Streamlit Cloud deployment.
**Alternatives rejected:** Baking the models into the image is not recorded as considered; `docs/DEPLOY.md` states the aim as keeping containers small.
**Consequences:** The first request on a cold instance waits for the download and unpickling. The service needs network access to Hugging Face. Evidence: `Dockerfile`, `src/model_assets.py`, commit `1b8d33d`.

---

## ADR-07: PostgreSQL for training and data loading only, with CSV fallback

**Context:** The project started as a PostgreSQL-backed pipeline (commit `a4b77bf`).
**Decision:** `load_tickets` reads the `tickets` table when a database is reachable and otherwise reads the local CSV (`src/data_loader.py`). The FastAPI service never imports the database layer, so inference needs no database. Evidence: `src/data_loader.py`, `src/load_data_to_db.py`, `api/main.py` imports.
**Alternatives rejected:** None recorded.
**Consequences:** `docker-compose.yml` still starts a Postgres container for the API service, which the API does not use. Neon is optional (`docs/DEPLOY.md`).
