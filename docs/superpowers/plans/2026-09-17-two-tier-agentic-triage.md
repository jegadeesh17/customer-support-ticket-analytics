# Two-Tier Agentic Triage Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the resume's claim — "Implemented Tier 2 agentic escalation activating only for low confidence (<0.80) or severe resolution (>48h)" — literally true in the deployed `CustomerSupportAnalytics` service, instead of it describing an unwired design intent.

**Architecture:** `POST /triage_agent` currently always runs the full LLM/heuristic diagnostic unconditionally, with no automatic gate. This plan (1) adds a confidence score to the Tier-1 classifier, (2) extracts the three trigger conditions already documented in `docs/SPEC.md` §5.1 into a pure, testable gate function, (3) rewrites `/triage_agent` to call Tier 1 first and only invoke Tier 2 when the gate fires (or when explicitly forced, for demo purposes), (4) wires the newly-added `GROQ_API_KEY` secret through to the deployed Cloud Run service as the primary LLM provider, (5) surfaces the feature in the actual live demo UI (`api/index.html` — the Streamlit pages under `app/` are dev-only and are never copied into the Docker image, so they are out of scope), (6) measures the real "% handled without escalation" instead of assuming 90%, and (7) syncs the resume and portfolio to whatever is actually true once built.

**Tech Stack:** Python 3.11, FastAPI, scikit-learn (RandomForest/LogisticRegression/GradientBoosting/MLP — all already `predict_proba`-capable), pytest, Groq's OpenAI-compatible chat completions API via `urllib.request` (no new dependency), GitHub Actions + Cloud Run.

**Spec:** `docs/SPEC.md` §5 ("Agentic Escalation Tier Specification") — already documents these exact three trigger conditions; this plan is the first thing that actually implements them.

## Global Constraints

- Gate thresholds must exactly match `docs/SPEC.md` §5.1: confidence `< 0.80`, resolution hours `> 48.0`, Enterprise subscription with `issue_complexity_score > 8`. Do not invent different numbers.
- No new third-party dependency. Groq's API is OpenAI-compatible chat completions — reuse the existing `urllib.request` pattern already in `src/agent_triage.py`, don't add an SDK.
- Follow existing test-mocking conventions exactly: endpoint-level mocks patch `api.main.<function_name>` (see `tests/test_api.py`), not the underlying module.
- Every task must leave the full test suite green (`pytest -q` from the repo root) before moving to the next task.
- The live Cloud Run image only contains `api/`, `src/`, `configs/`, and the sample CSV (see `Dockerfile`) — `app/` (Streamlit) is never deployed. Any UI change for this feature goes in `api/index.html`, not `app/pages/`.
- `configs/settings.py` is the single place new env vars are declared (pydantic-settings, matches existing `OPENROUTER_API_KEY`/`OPENAI_API_KEY` pattern).
- Cloud Run env vars are forwarded via plaintext `--set-env-vars` in `.github/workflows/deploy.yml` (existing pattern for `HF_MODEL_REPO`) — this repo does not use Secret Manager elsewhere, so don't introduce it here.
- The resume (`ZGeneral/resume/Jegadeesh_D_AI_Systems_Resume.tex`) is the source of truth for what's claimed; the final task reconciles it and the portfolio (`PersonalPortfolioFolder`) to the numbers actually measured after implementation — never guessed.

---

### Task 1: Tier-1 confidence scoring

**Files:**
- Modify: `src/inference.py`
- Test: `tests/test_inference_confidence.py`

**Interfaces:**
- Produces: `predict_classification_with_confidence(user_inputs: dict) -> tuple[str, float]` in `src/inference.py` — returns `(predicted_label, confidence)` where confidence is the winning class's predicted probability, in `[0.0, 1.0]`. Consumed by Task 4.

- [ ] **Step 1: Write the failing test**

Create `tests/test_inference_confidence.py`:

```python
"""Tests for Tier-1 confidence scoring used by the escalation gate."""

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.inference import predict_classification_with_confidence, load_model_bundle, _prepare_features
from src.preprocessor import build_inference_row
from src.constants import DEFAULT_INFERENCE_ROW, PRIORITY_LEVELS


def test_confidence_is_a_valid_probability():
    label, confidence = predict_classification_with_confidence(DEFAULT_INFERENCE_ROW)
    assert label in PRIORITY_LEVELS
    assert 0.0 <= confidence <= 1.0


def test_confidence_matches_manual_predict_proba_max():
    import numpy as np

    bundle = load_model_bundle('classification_model.pkl')
    model = bundle['model']
    processed = _prepare_features(build_inference_row(DEFAULT_INFERENCE_ROW), 'classification', model)
    expected_confidence = float(np.max(model.predict_proba(processed)[0]))

    _, confidence = predict_classification_with_confidence(DEFAULT_INFERENCE_ROW)
    assert confidence == expected_confidence
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_inference_confidence.py -v`
Expected: FAIL with `ImportError: cannot import name 'predict_classification_with_confidence'`

- [ ] **Step 3: Implement `predict_classification_with_confidence`**

In `src/inference.py`, add this function directly after the existing `predict_classification` function (after line 97):

```python
def predict_classification_with_confidence(user_inputs):
    """Like predict_classification, but also returns the winning class's
    predicted probability, used by the Tier-2 escalation gate."""
    bundle = load_model_bundle('classification_model.pkl')
    if bundle is None:
        raise FileNotFoundError('classification_model.pkl not found. Run: python src/train_models.py')
    model = bundle['model']
    processed = _prepare_features(build_inference_row(user_inputs), 'classification', model)
    proba = model.predict_proba(processed)[0]
    idx = int(np.argmax(proba))
    label = model.classes_[idx]
    confidence = float(proba[idx])
    return label, confidence
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_inference_confidence.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add src/inference.py tests/test_inference_confidence.py
git commit -m "feat: add Tier-1 classifier confidence scoring"
```

---

### Task 2: Escalation gate logic

**Files:**
- Create: `src/triage_gate.py`
- Test: `tests/test_triage_gate.py`

**Interfaces:**
- Consumes: nothing from Task 1 (pure function, independent).
- Produces: `should_escalate(subscription_type: str, issue_complexity_score: int, confidence: float, resolution_hours: float) -> Optional[str]` — returns one of `"low_confidence"`, `"severe_resolution"`, `"high_risk_segment"`, or `None`. Consumed by Task 4.

- [ ] **Step 1: Write the failing test**

Create `tests/test_triage_gate.py`:

```python
"""Tests for the Tier-1 -> Tier-2 escalation gate (docs/SPEC.md Section 5.1)."""

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.triage_gate import should_escalate


def test_low_confidence_triggers_escalation():
    result = should_escalate(subscription_type="Basic", issue_complexity_score=3, confidence=0.55, resolution_hours=5.0)
    assert result == "low_confidence"


def test_severe_resolution_triggers_escalation():
    result = should_escalate(subscription_type="Basic", issue_complexity_score=3, confidence=0.95, resolution_hours=72.0)
    assert result == "severe_resolution"


def test_high_risk_segment_triggers_escalation():
    result = should_escalate(subscription_type="Enterprise", issue_complexity_score=9, confidence=0.95, resolution_hours=5.0)
    assert result == "high_risk_segment"


def test_routine_ticket_does_not_escalate():
    result = should_escalate(subscription_type="Basic", issue_complexity_score=3, confidence=0.95, resolution_hours=5.0)
    assert result is None


def test_confidence_exactly_at_threshold_does_not_escalate():
    result = should_escalate(subscription_type="Basic", issue_complexity_score=3, confidence=0.80, resolution_hours=5.0)
    assert result is None


def test_resolution_exactly_at_threshold_does_not_escalate():
    result = should_escalate(subscription_type="Basic", issue_complexity_score=3, confidence=0.95, resolution_hours=48.0)
    assert result is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_triage_gate.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.triage_gate'`

- [ ] **Step 3: Implement the gate**

Create `src/triage_gate.py`:

```python
"""Pure routing logic deciding when a ticket needs Tier-2 agentic triage.

Thresholds match docs/SPEC.md Section 5.1 (Agentic Escalation Tier Specification).
"""

from __future__ import annotations

from typing import Optional

CONFIDENCE_THRESHOLD = 0.80
RESOLUTION_HOURS_THRESHOLD = 48.0
HIGH_RISK_COMPLEXITY_THRESHOLD = 8


def should_escalate(
    subscription_type: str,
    issue_complexity_score: int,
    confidence: float,
    resolution_hours: float,
) -> Optional[str]:
    """Return the trigger reason if Tier 2 should run, else None.

    Trigger reasons: "low_confidence", "severe_resolution", "high_risk_segment".
    """
    if confidence < CONFIDENCE_THRESHOLD:
        return "low_confidence"
    if resolution_hours > RESOLUTION_HOURS_THRESHOLD:
        return "severe_resolution"
    if subscription_type == "Enterprise" and issue_complexity_score > HIGH_RISK_COMPLEXITY_THRESHOLD:
        return "high_risk_segment"
    return None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_triage_gate.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add src/triage_gate.py tests/test_triage_gate.py
git commit -m "feat: add pure Tier-1/Tier-2 escalation gate per SPEC.md 5.1"
```

---

### Task 3: Groq LLM provider integration

**Files:**
- Modify: `configs/settings.py`
- Modify: `src/agent_triage.py`
- Modify: `.github/workflows/deploy.yml`
- Test: `tests/test_agent_triage.py` (append)

**Interfaces:**
- Consumes: nothing from Tasks 1-2.
- Produces: `run_agent_triage(ticket, api_key=None)` now prefers Groq when `settings.GROQ_API_KEY` is set, falling back to OpenRouter/OpenAI/heuristic exactly as before. Signature and return type (`AgentTriageResult`) are unchanged, so Task 4 and existing callers need no changes.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_agent_triage.py` (add `from configs.settings import settings` near the top imports, alongside the existing `from api.main import app` / `from src.agent_triage import ...` lines):

```python
def test_groq_provider_used_when_configured(monkeypatch):
    monkeypatch.setattr(settings, "GROQ_API_KEY", "gsk-mock-key")
    monkeypatch.setattr(settings, "OPENROUTER_API_KEY", None)
    monkeypatch.setattr(settings, "OPENAI_API_KEY", None)

    mock_payload = {
        "ticket_id": None,
        "escalate_to_tier2": True,
        "customer_frustration_score": 8,
        "root_cause_category": "Service Outage",
        "urgency_reasoning": "Enterprise outage.",
        "recommended_action": "Escalate to on-call.",
        "auto_drafted_response": "We are investigating.",
        "confidence": 0.9,
    }
    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps({
        "choices": [{"message": {"content": json.dumps(mock_payload)}}]
    }).encode("utf-8")
    mock_resp.__enter__.return_value = mock_resp

    captured_requests = []

    def fake_urlopen(req, timeout=5):
        captured_requests.append(req)
        return mock_resp

    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        result = run_agent_triage({"issue_description": "Outage"})

    assert result.triage_source == "agent_llm"
    assert captured_requests[0].full_url == "https://api.groq.com/openai/v1/chat/completions"


def test_groq_failure_falls_back_to_heuristic(monkeypatch):
    monkeypatch.setattr(settings, "GROQ_API_KEY", "gsk-mock-key")

    with patch("urllib.request.urlopen", side_effect=TimeoutError("no response")):
        result = run_agent_triage({
            "issue_description": "App is broken and I am furious",
            "issue_complexity_score": 9,
        })

    assert result.triage_source == "heuristic_fallback"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_agent_triage.py -v -k groq`
Expected: FAIL — `test_groq_provider_used_when_configured` fails because the request goes to `openrouter.ai`, not `api.groq.com` (no Groq branch exists yet).

- [ ] **Step 3: Add Groq settings**

In `configs/settings.py`, change:

```python
    # src/agent_triage.py
    OPENROUTER_API_KEY: Optional[str] = None
    OPENAI_API_KEY: Optional[str] = None
```

to:

```python
    # src/agent_triage.py
    OPENROUTER_API_KEY: Optional[str] = None
    OPENAI_API_KEY: Optional[str] = None
    GROQ_API_KEY: Optional[str] = None
    GROQ_MODEL: str = "llama-3.3-70b-versatile"
```

- [ ] **Step 4: Add provider selection to `src/agent_triage.py`**

Replace the `run_agent_triage` function (currently lines 105-165) with:

```python
def _select_provider():
    """Pick the first configured LLM provider as (endpoint_url, api_key, model).

    Order: Groq (fast, the newly configured provider) -> OpenRouter -> OpenAI.
    Returns None if nothing is configured, in which case the caller falls
    back to deterministic heuristics.
    """
    if settings.GROQ_API_KEY:
        return ("https://api.groq.com/openai/v1/chat/completions", settings.GROQ_API_KEY, settings.GROQ_MODEL)
    if settings.OPENROUTER_API_KEY:
        return ("https://openrouter.ai/api/v1/chat/completions", settings.OPENROUTER_API_KEY, "google/gemini-2.0-flash-001")
    if settings.OPENAI_API_KEY:
        return ("https://api.openai.com/v1/chat/completions", settings.OPENAI_API_KEY, "gpt-4o-mini")
    return None


def run_agent_triage(ticket: Dict[str, Any], api_key: Optional[str] = None) -> AgentTriageResult:
    """Evaluate a support ticket and return structured triage diagnostics.

    An explicit api_key argument is sent to OpenRouter directly (used by
    callers/tests that already hold a specific key). Otherwise the first
    configured provider from _select_provider() is used. Falls back
    gracefully to deterministic heuristics on network error or when no
    provider is configured.
    """
    if api_key:
        provider = ("https://openrouter.ai/api/v1/chat/completions", api_key, "google/gemini-2.0-flash-001")
    else:
        provider = _select_provider()

    if provider is None:
        return _extract_heuristics(ticket)

    endpoint_url, key, model = provider

    try:
        import urllib.request

        system_prompt = (
            "You are an expert Principal Customer Operations Triage Agent. "
            "Analyze the support ticket metadata and description. Respond ONLY with a valid JSON object matching this schema:\n"
            "{\n"
            '  "ticket_id": string or null,\n'
            '  "escalate_to_tier2": boolean,\n'
            '  "customer_frustration_score": integer between 1 and 10,\n'
            '  "root_cause_category": string,\n'
            '  "urgency_reasoning": string,\n'
            '  "recommended_action": string,\n'
            '  "auto_drafted_response": string,\n'
            '  "confidence": float between 0.0 and 1.0\n'
            "}"
        )

        user_content = json.dumps(ticket, indent=2)
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Triage this ticket:\n{user_content}"},
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.2,
        }

        req = urllib.request.Request(
            endpoint_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://supportops.internal",
                "X-Title": "SupportOps Agent Triage",
            },
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=5) as response:
            res_data = json.loads(response.read().decode("utf-8"))
            content = res_data["choices"][0]["message"]["content"]
            parsed = json.loads(content)
            parsed["triage_source"] = "agent_llm"
            return AgentTriageResult(**parsed)

    except Exception:
        # Fallback to local heuristic evaluator without throwing
        return _extract_heuristics(ticket)
```

- [ ] **Step 5: Forward the secret to Cloud Run**

In `.github/workflows/deploy.yml`, change the last line of the `Deploy to Cloud Run` step from:

```yaml
            --set-env-vars "HF_MODEL_REPO=${{ secrets.HF_MODEL_REPO }}"
```

to:

```yaml
            --set-env-vars "HF_MODEL_REPO=${{ secrets.HF_MODEL_REPO }},GROQ_API_KEY=${{ secrets.GROQ_API_KEY }}"
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `pytest tests/test_agent_triage.py -v`
Expected: 6 passed (the 4 pre-existing tests plus the 2 new Groq tests)

- [ ] **Step 7: Commit**

```bash
git add configs/settings.py src/agent_triage.py .github/workflows/deploy.yml tests/test_agent_triage.py
git commit -m "feat: add Groq as the primary Tier-2 LLM provider"
```

---

### Task 4: Two-tier `/triage_agent` endpoint

**Files:**
- Modify: `api/main.py`
- Test: `tests/test_agent_triage.py` (append)

**Interfaces:**
- Consumes: `predict_classification_with_confidence` (Task 1), `should_escalate` (Task 2), `run_agent_triage` (Task 3, unchanged signature).
- Produces: `TwoTierTriageResponse` Pydantic model in `api/main.py` with fields `ticket_id`, `tier1_priority`, `tier1_confidence`, `tier1_resolution_hours`, `escalate_to_tier2`, `escalation_trigger`, `customer_frustration_score`, `root_cause_category`, `urgency_reasoning`, `recommended_action`, `auto_drafted_response`, `triage_source` (last 5 are `Optional`, `None` when not escalated). Consumed by Task 5 (UI).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_agent_triage.py`:

```python
def test_two_tier_triage_routine_ticket_no_escalation():
    with patch("api.main.predict_classification_with_confidence", return_value=("Low", 0.95)):
        with patch("api.main.predict_regression", return_value=6.0):
            from api.main import app
            client = TestClient(app)
            response = client.post("/triage_agent", json={
                "issue_description": "Can you update my billing email address please?",
                "subscription_type": "Basic",
                "issue_complexity_score": 2,
            })
    assert response.status_code == 200
    data = response.json()
    assert data["escalate_to_tier2"] is False
    assert data["escalation_trigger"] is None
    assert data["customer_frustration_score"] is None
    assert data["auto_drafted_response"] is None
    assert data["tier1_priority"] == "Low"
    assert data["tier1_confidence"] == 0.95


def test_two_tier_triage_low_confidence_escalates():
    with patch("api.main.predict_classification_with_confidence", return_value=("Medium", 0.5)):
        with patch("api.main.predict_regression", return_value=6.0):
            from api.main import app
            client = TestClient(app)
            response = client.post("/triage_agent", json={
                "issue_description": "Not sure what's wrong but something feels off.",
                "subscription_type": "Basic",
                "issue_complexity_score": 2,
            })
    assert response.status_code == 200
    data = response.json()
    assert data["escalate_to_tier2"] is True
    assert data["escalation_trigger"] == "low_confidence"
    assert data["auto_drafted_response"] is not None


def test_two_tier_triage_severe_resolution_escalates():
    with patch("api.main.predict_classification_with_confidence", return_value=("Medium", 0.95)):
        with patch("api.main.predict_regression", return_value=72.0):
            from api.main import app
            client = TestClient(app)
            response = client.post("/triage_agent", json={
                "issue_description": "This migration is taking forever to complete.",
                "subscription_type": "Basic",
                "issue_complexity_score": 4,
            })
    assert response.status_code == 200
    data = response.json()
    assert data["escalate_to_tier2"] is True
    assert data["escalation_trigger"] == "severe_resolution"


def test_two_tier_triage_force_override_escalates_routine_ticket():
    with patch("api.main.predict_classification_with_confidence", return_value=("Low", 0.95)):
        with patch("api.main.predict_regression", return_value=6.0):
            from api.main import app
            client = TestClient(app)
            response = client.post(
                "/triage_agent",
                params={"force": True},
                json={
                    "issue_description": "Just double-checking my subscription renewal date.",
                    "subscription_type": "Basic",
                    "issue_complexity_score": 1,
                },
            )
    assert response.status_code == 200
    data = response.json()
    assert data["escalate_to_tier2"] is True
    assert data["escalation_trigger"] == "forced"
```

Leave the pre-existing `test_api_triage_agent_endpoint` untouched — its ticket (`subscription_type: "Enterprise"`, `issue_complexity_score: 9`) triggers the `high_risk_segment` gate unconditionally, so it stays valid against the real (unmocked) model regardless of what confidence the real classifier produces.

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_agent_triage.py -v -k two_tier`
Expected: FAIL — current endpoint always returns the flat `AgentTriageResult` shape and has no `escalate_to_tier2`-gating behavior; response will be missing `tier1_priority`/`tier1_confidence`/`escalation_trigger` keys.

- [ ] **Step 3: Rewrite the endpoint**

In `api/main.py`, change the import on line 10 from:

```python
from pydantic import BaseModel, Field
```

to:

```python
from typing import Optional

from pydantic import BaseModel, Field
```

Change line 17 from:

```python
from src.inference import predict_classification, predict_regression, predict_satisfaction
```

to:

```python
from src.inference import predict_classification, predict_regression, predict_satisfaction, predict_classification_with_confidence
from src.triage_gate import should_escalate
```

Replace everything from the `from src.agent_triage import AgentTriageResult, run_agent_triage` line to the end of the file (lines 125-137) with:

```python
from src.agent_triage import run_agent_triage


class TwoTierTriageResponse(BaseModel):
    ticket_id: Optional[str] = None
    tier1_priority: str
    tier1_confidence: float
    tier1_resolution_hours: float
    escalate_to_tier2: bool
    escalation_trigger: Optional[str] = None
    customer_frustration_score: Optional[int] = None
    root_cause_category: Optional[str] = None
    urgency_reasoning: Optional[str] = None
    recommended_action: Optional[str] = None
    auto_drafted_response: Optional[str] = None
    triage_source: Optional[str] = None


@app.post("/triage_agent", response_model=TwoTierTriageResponse)
def triage_agent_endpoint(ticket: TicketInput, force: bool = False) -> TwoTierTriageResponse:
    """Two-tier triage: Tier 1 always runs; Tier 2 (agentic diagnosis) only
    runs when Tier 1 signals low confidence, a severe resolution estimate,
    or a high-risk Enterprise segment -- or when force=True is passed."""
    payload = {**DEFAULT_INFERENCE_ROW, **ticket.model_dump()}
    try:
        priority, confidence = predict_classification_with_confidence(payload)
        resolution_hours = predict_regression(payload)
    except FileNotFoundError as exc:
        logger.exception("Missing model file in triage_agent_endpoint")
        raise HTTPException(status_code=503, detail=MODEL_UNAVAILABLE_DETAIL) from exc
    except Exception as exc:
        logger.exception("Unhandled error in triage_agent_endpoint")
        raise HTTPException(status_code=500, detail=INTERNAL_ERROR_DETAIL) from exc

    trigger = should_escalate(
        subscription_type=ticket.subscription_type,
        issue_complexity_score=ticket.issue_complexity_score,
        confidence=confidence,
        resolution_hours=resolution_hours,
    )
    if force and trigger is None:
        trigger = "forced"

    base_fields = dict(
        ticket_id=None,
        tier1_priority=str(priority),
        tier1_confidence=confidence,
        tier1_resolution_hours=resolution_hours,
        escalate_to_tier2=trigger is not None,
        escalation_trigger=trigger,
    )

    if trigger is None:
        return TwoTierTriageResponse(**base_fields)

    try:
        diagnosis = run_agent_triage(payload)
    except Exception as exc:
        logger.exception("Unhandled error in Tier 2 agent triage")
        raise HTTPException(status_code=500, detail=INTERNAL_ERROR_DETAIL) from exc

    return TwoTierTriageResponse(
        **base_fields,
        customer_frustration_score=diagnosis.customer_frustration_score,
        root_cause_category=diagnosis.root_cause_category,
        urgency_reasoning=diagnosis.urgency_reasoning,
        recommended_action=diagnosis.recommended_action,
        auto_drafted_response=diagnosis.auto_drafted_response,
        triage_source=diagnosis.triage_source,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_agent_triage.py -v`
Expected: 10 passed (6 from Task 3 plus the 4 new two-tier endpoint tests)

Then run the full suite to make sure nothing else broke:

Run: `pytest -q`
Expected: all tests pass

- [ ] **Step 5: Commit**

```bash
git add api/main.py tests/test_agent_triage.py
git commit -m "feat: rewire /triage_agent into a real two-tier gated pipeline"
```

---

### Task 5: Escalation Triage tab in the live demo UI

**Files:**
- Modify: `api/index.html`
- Test: `tests/test_ui_html.py`

**Interfaces:**
- Consumes: `TwoTierTriageResponse` field names from Task 4 (`tier1_priority`, `tier1_confidence`, `tier1_resolution_hours`, `escalate_to_tier2`, `escalation_trigger`, `customer_frustration_score`, `root_cause_category`, `urgency_reasoning`, `auto_drafted_response`, `triage_source`).
- Produces: nothing consumed by later tasks.

- [ ] **Step 1: Write the failing test**

Create `tests/test_ui_html.py`:

```python
"""Sanity checks for the live demo UI (api/index.html) -- the file actually
served at /app on Cloud Run. The Streamlit pages under app/ are dev-only and
are never copied into the Docker image (see Dockerfile), so they are not
covered here."""

import os

HTML_PATH = os.path.join(os.path.dirname(__file__), "..", "api", "index.html")


def _read_html():
    with open(HTML_PATH, "r", encoding="utf-8") as f:
        return f.read()


def test_escalation_tab_present():
    html = _read_html()
    assert "tab-escalation" in html
    assert "Escalation Triage" in html
    assert "handleTriage" in html


def test_escalation_tab_posts_to_triage_agent_endpoint():
    html = _read_html()
    assert "/triage_agent?force=" in html
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_ui_html.py -v`
Expected: FAIL — `tab-escalation` not found in `api/index.html`

- [ ] **Step 3: Add the tab button**

In `api/index.html`, change:

```html
            <button class="tab-btn active" type="button" role="tab" onclick="showTab('priority', this)">Priority Routing</button>
            <button class="tab-btn" type="button" role="tab" onclick="showTab('sla', this)">SLA Prediction</button>
            <button class="tab-btn" type="button" role="tab" onclick="showTab('satisfaction', this)">Satisfaction Risk</button>
```

to:

```html
            <button class="tab-btn active" type="button" role="tab" onclick="showTab('priority', this)">Priority Routing</button>
            <button class="tab-btn" type="button" role="tab" onclick="showTab('sla', this)">SLA Prediction</button>
            <button class="tab-btn" type="button" role="tab" onclick="showTab('satisfaction', this)">Satisfaction Risk</button>
            <button class="tab-btn" type="button" role="tab" onclick="showTab('escalation', this)">Escalation Triage</button>
```

- [ ] **Step 4: Add the tab content**

Immediately after the closing `</div>` of `<div id="tab-satisfaction" ...>` (right before `</main>`), insert:

```html
        <div id="tab-escalation" class="tab-content">
            <div class="model-strip">
                <span>Two-tier triage: Tier 1 routes instantly; Tier 2 (Groq LLM) only runs when Tier 1 is uncertain</span>
            </div>
            <form id="form-escalation" onsubmit="handleTriage(event)">
                <div class="section-label">Ticket</div>
                <div class="form-group">
                    <label for="e-desc">Issue description</label>
                    <textarea id="e-desc" name="issue_description" required minlength="5">I am unable to access my account after entering the correct credentials.</textarea>
                </div>

                <div class="section-label">Context</div>
                <div class="grid-2">
                    <div class="form-group">
                        <label for="e-product">Product</label>
                        <select id="e-product" name="product" data-vocab="products"></select>
                    </div>
                    <div class="form-group">
                        <label for="e-category">Category</label>
                        <select id="e-category" name="category" data-vocab="categories"></select>
                    </div>
                    <div class="form-group">
                        <label for="e-sub">Subscription</label>
                        <select id="e-sub" name="subscription_type" data-vocab="subscriptions"></select>
                    </div>
                    <div class="form-group">
                        <label for="e-complexity">Issue complexity score (1-10)</label>
                        <input type="number" id="e-complexity" name="issue_complexity_score" value="5" min="1" max="10" required>
                    </div>
                </div>

                <div class="form-group" style="display:flex; align-items:center; gap:.5rem; margin-top:.5rem;">
                    <input type="checkbox" id="e-force" name="force" style="width:auto;">
                    <label for="e-force" style="margin:0;">Force Tier 2 escalation (bypass thresholds, for demo purposes)</label>
                </div>

                <button type="submit" class="submit-btn"><span>Run two-tier triage</span><div class="spinner"></div></button>
                <div class="result-card" id="res-escalation"></div>
            </form>
        </div>
```

- [ ] **Step 5: Add the `handleTriage` JS function**

Immediately before the closing `</script>` tag, insert:

```javascript
    async function handleTriage(e) {
        e.preventDefault();
        const form = e.target;
        const btn = form.querySelector('.submit-btn');
        const spinner = form.querySelector('.spinner');
        const box = document.getElementById('res-escalation');

        btn.disabled = true;
        spinner.style.display = 'inline-block';
        box.className = 'result-card';

        const slowNotice = setTimeout(() => {
            renderResult(box, 'info', 'Working', 'Waking the service…',
                'The instance had scaled to zero and is loading model bundles. First request can take up to a minute.');
        }, 3500);

        const formData = Object.fromEntries(new FormData(form).entries());
        const force = formData.force === 'on';
        delete formData.force;
        formData.issue_complexity_score = parseInt(formData.issue_complexity_score, 10);

        try {
            const response = await fetch('/triage_agent?force=' + force, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(formData),
            });
            clearTimeout(slowNotice);

            if (!response.ok) {
                let detail = `HTTP ${response.status}`;
                try {
                    const err = await response.json();
                    if (err.detail) detail = typeof err.detail === 'string' ? err.detail : JSON.stringify(err.detail);
                } catch (_) { /* non-JSON error body */ }
                renderResult(box, 'danger', 'Request failed', 'Could not triage this ticket', detail);
                return;
            }

            const data = await response.json();
            box.className = 'result-card show ' + (data.escalate_to_tier2 ? 'r-warning' : 'r-success');
            box.innerHTML = '';

            const mk = (cls, txt) => { const d = document.createElement('div'); d.className = cls; d.textContent = txt; return d; };
            box.appendChild(mk('result-label', 'Tier 1'));
            box.appendChild(mk('result-value', `${data.tier1_priority} · ${(data.tier1_confidence * 100).toFixed(0)}% confidence · ${data.tier1_resolution_hours.toFixed(1)}h est. resolution`));

            if (data.escalate_to_tier2) {
                box.appendChild(mk('result-note', `Escalated to Tier 2 — trigger: ${data.escalation_trigger}`));
                box.appendChild(mk('result-label', 'Frustration score'));
                box.appendChild(mk('result-value', `${data.customer_frustration_score} / 10`));
                box.appendChild(mk('result-note', data.root_cause_category));
                box.appendChild(mk('result-note', data.urgency_reasoning));
                box.appendChild(mk('result-meta', `Source: ${data.triage_source}`));
                const draft = document.createElement('div');
                draft.className = 'result-note';
                draft.style.whiteSpace = 'pre-wrap';
                draft.style.marginTop = '.6rem';
                draft.textContent = data.auto_drafted_response;
                box.appendChild(draft);
            } else {
                box.appendChild(mk('result-note', 'Handled by Tier 1 — no agentic escalation needed.'));
            }
        } catch (err) {
            clearTimeout(slowNotice);
            renderResult(box, 'danger', 'Request failed', 'Could not reach the service', 'Check your connection and try again.');
        } finally {
            btn.disabled = false;
            spinner.style.display = 'none';
        }
    }
```

- [ ] **Step 6: Run test to verify it passes**

Run: `pytest tests/test_ui_html.py -v`
Expected: 2 passed

- [ ] **Step 7: Commit**

```bash
git add api/index.html tests/test_ui_html.py
git commit -m "feat: surface the two-tier escalation triage in the live demo UI"
```

---

### Task 6: Escalation-rate benchmark

**Files:**
- Create: `scripts/benchmark_escalation_gate.py`
- Create: `reports/ESCALATION_GATE_BENCHMARK.md` (generated by the script, not hand-written)
- Test: `tests/test_benchmark_escalation_gate.py`

**Interfaces:**
- Consumes: `predict_classification_with_confidence` (Task 1), `predict_regression` (existing), `should_escalate` (Task 2).
- Produces: `compute_escalation_rate(triggers: list) -> tuple[float, dict]` — pure aggregation function, consumed only by its own test and by `main()` in the same file. Consumed by Task 7 (reads the generated report, not the function).

- [ ] **Step 1: Write the failing test**

Create `tests/test_benchmark_escalation_gate.py`:

```python
"""Tests for the escalation-rate aggregation logic used by the benchmark script."""

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts.benchmark_escalation_gate import compute_escalation_rate


def test_compute_escalation_rate_basic():
    triggers = [None, None, None, "low_confidence"]
    pct, breakdown = compute_escalation_rate(triggers)
    assert pct == 75.0
    assert breakdown == {"low_confidence": 1}


def test_compute_escalation_rate_all_escalated():
    triggers = ["severe_resolution", "high_risk_segment"]
    pct, breakdown = compute_escalation_rate(triggers)
    assert pct == 0.0
    assert breakdown == {"severe_resolution": 1, "high_risk_segment": 1}


def test_compute_escalation_rate_empty():
    pct, breakdown = compute_escalation_rate([])
    assert pct == 0.0
    assert breakdown == {}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_benchmark_escalation_gate.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.benchmark_escalation_gate'`

- [ ] **Step 3: Implement the script**

Create `scripts/benchmark_escalation_gate.py`:

```python
"""Measure the real Tier-1/Tier-2 escalation rate against sample ticket data.

Produces reports/ESCALATION_GATE_BENCHMARK.md with the actual percentage of
tickets Tier 1 resolves without escalation, replacing the previously
unverified "~90%" resume claim with a measured number.
"""

import os
import sys
from collections import Counter

import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.constants import DEFAULT_INFERENCE_ROW
from src.inference import predict_classification_with_confidence, predict_regression
from src.paths import get_data_path
from src.triage_gate import should_escalate


def compute_escalation_rate(triggers):
    """Given a list of trigger values (str or None), return (pct_not_escalated, breakdown)."""
    total = len(triggers)
    if total == 0:
        return 0.0, {}
    not_escalated = sum(1 for t in triggers if t is None)
    breakdown = dict(Counter(t for t in triggers if t is not None))
    return round(100.0 * not_escalated / total, 1), breakdown


def main(sample_size=500):
    df = pd.read_csv(get_data_path())
    df = df.sample(min(sample_size, len(df)), random_state=42)

    triggers = []
    for _, row in df.iterrows():
        ticket = {**DEFAULT_INFERENCE_ROW, **{k.lower().replace(' ', '_'): v for k, v in row.to_dict().items()}}
        try:
            _, confidence = predict_classification_with_confidence(ticket)
            resolution_hours = predict_regression(ticket)
        except Exception:
            continue
        trigger = should_escalate(
            subscription_type=str(ticket.get("subscription_type", "Basic")),
            issue_complexity_score=int(ticket.get("issue_complexity_score", 5)),
            confidence=confidence,
            resolution_hours=resolution_hours,
        )
        triggers.append(trigger)

    pct_not_escalated, breakdown = compute_escalation_rate(triggers)

    report_path = os.path.join(ROOT, "reports", "ESCALATION_GATE_BENCHMARK.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Escalation Gate Benchmark\n\n")
        f.write(f"Sample size: {len(triggers)} tickets\n\n")
        f.write(f"**Tier 1 handled without escalation: {pct_not_escalated}%**\n\n")
        f.write("Escalation trigger breakdown:\n\n")
        for reason, count in breakdown.items():
            f.write(f"- {reason}: {count}\n")

    print(f"Tier 1 handled {pct_not_escalated}% without escalation. Report written to {report_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_benchmark_escalation_gate.py -v`
Expected: 3 passed

- [ ] **Step 5: Actually run the benchmark against real data**

Run: `python scripts/benchmark_escalation_gate.py`

This writes `reports/ESCALATION_GATE_BENCHMARK.md` with the real measured percentage. Read the printed output and the generated file — this number (not "90%") is what Task 7 uses to reconcile the resume.

- [ ] **Step 6: Commit**

```bash
git add scripts/benchmark_escalation_gate.py tests/test_benchmark_escalation_gate.py reports/ESCALATION_GATE_BENCHMARK.md
git commit -m "feat: measure the real Tier-1 no-escalation rate instead of assuming 90%"
```

---

### Task 7: Reconcile resume and portfolio with verified reality

**Files:**
- Modify: `ZGeneral/resume/Jegadeesh_D_AI_Systems_Resume.tex`
- Modify: `PersonalPortfolioFolder/src/components/Projects.jsx`
- Modify: `PersonalPortfolioFolder/public/resume.pdf` (only if you can recompile — see note below)

**Interfaces:**
- Consumes: the exact test count from this task's own Step 1, and the exact escalation percentage from `reports/ESCALATION_GATE_BENCHMARK.md` (Task 6).
- Produces: nothing (terminal task).

- [ ] **Step 1: Get the exact final test count**

Run: `pytest --collect-only -q` from the `CustomerSupportAnalytics` repo root, and read the last line (`N tests collected`). Do not guess this number — use exactly what the command prints. (Before this plan, it was 17; Tasks 1-6 added 2+6+2+4+2+3 = 19 new tests, so it should read 36, but confirm against the actual command output.)

- [ ] **Step 2: Read the measured escalation rate**

Open `reports/ESCALATION_GATE_BENCHMARK.md` (written in Task 6) and note the "Tier 1 handled without escalation: X%" figure.

- [ ] **Step 3: Update the resume**

In `ZGeneral/resume/Jegadeesh_D_AI_Systems_Resume.tex`, the Customer Support Analytics bullets currently read:

```latex
    \item {\small Architected a two-tier triage system over ~200K tickets: Tier 1 classical models handle 90\% of routine traffic without escalation at sub-50ms median latency (\$0 API cost; 82.1\% priority-classification accuracy, R-squared = 0.719 SLA resolution-time regression, zero data leakage).}
    \item {\small Implemented Tier 2 agentic escalation activating only for low confidence ($<0.80$) or severe resolution ($>48$\,h), generating structured JSON with customer frustration diagnostics and auto-drafted replies.}
    \item {\small Deployed FastAPI REST endpoints (\texttt{POST /predict\_*}, \texttt{POST /triage\_agent}) with PostgreSQL orchestration; validated by 17 automated pytest tests. \href{https://support-ops-api-242711953247.asia-south1.run.app/app}{[Live]} \href{https://github.com/jegadeesh17/customer-support-ticket-analytics}{[GitHub]}}
```

- If the Step 2 percentage is within 85-95%, leave "90%" as-is (still a fair approximation, and now it's a measured one). If it falls outside that band, replace `90\%` with the measured value (e.g. `82.4\%`).
- Replace `17 automated pytest tests` with `<N> automated pytest tests`, using the exact number from Step 1.
- The second bullet's claim ("activating only for low confidence (<0.80) or severe resolution (>48h)") is now literally true — no wording change needed there.

- [ ] **Step 4: Update the portfolio**

In `PersonalPortfolioFolder/src/components/Projects.jsx`, find the `customer-support-analytics` project entry. Update:

The `description` field — append a clause noting the auto-gated two-tier system is real, e.g. change:

```
"Production multi-task customer support intelligence platform with automated agentic escalation. Deployed a two-tier triage system achieving 82.1% priority accuracy and R² = 0.719 resolution-time regression with zero data leakage, coupled with an autonomous agentic escalation tier with Pydantic contracts and FastAPI microservice."
```

to:

```
"Production multi-task customer support intelligence platform with automated agentic escalation. Deployed a two-tier triage system over ~200K tickets achieving 82.1% priority accuracy and R² = 0.719 resolution-time regression with zero data leakage; Tier 2 automatically activates only for low-confidence (<80%) or severe-resolution (>48h) predictions, powered by Groq's LLM API with a deterministic heuristic fallback."
```

Add a new metric to the `metrics` array for this project (currently `Triage Accuracy`, `Resolution Reg.`, `Pytest Suite`) — add a fourth entry using the Step 2 percentage:

```javascript
      { label: "Auto-Handled (No Escalation)", value: "<X>%" },
```

using the exact percentage from Step 2, not a guess.

- [ ] **Step 5: Recompile and update the resume PDF (manual step, note only)**

`pdflatex` is not available in this environment (noted in an earlier session). If you have LaTeX installed locally, recompile `Jegadeesh_D_AI_Systems_Resume.tex` to `Jegadeesh_D_AI_Systems_Resume.pdf`, then copy it to `PersonalPortfolioFolder/public/resume.pdf` and commit that binary. If not available now, skip this step and do it before the next portfolio deploy — the `.tex` file remains the source of truth per Global Constraints.

- [ ] **Step 6: Commit**

```bash
cd ZGeneral && git add resume/Jegadeesh_D_AI_Systems_Resume.tex && git commit -m "docs: sync resume with the now-real two-tier auto-escalation numbers"
cd ../PersonalPortfolioFolder && git add src/components/Projects.jsx && git commit -m "feat: reflect the real automatic two-tier escalation gate in the portfolio"
```

(If `ZGeneral` is not itself a git repo, skip the first commit and just leave the `.tex` change staged for the user to review.)

---

## Addendum (added mid-execution, after Task 6): Threshold recalibration

Task 6's benchmark measured 0.0% handled without escalation (0/500 real
tickets), not ~90%. Root cause, verified directly against the real data
(`data/customer_support_ticket_sample.csv`, 5000 rows) and the real trained
models: `resolution_time_hours` has a median of ~119h (80.2% of real tickets
already exceed 48h), and the trained regression model's predictions across a
500-ticket sample never fell below ~72h — so the SPEC.md-documented ">48h =
severe" threshold was miscalibrated against this dataset's actual scale from
the start, never having been checked against real data before this plan.
Separately, the classifier's per-ticket confidence has a median of only
~0.75 on the same sample (58.6% of tickets already fall under 0.80) — a
4-class classifier at ~82% accuracy routinely produces sub-0.80 max-class
probabilities even when correct, so a fixed 0.80 bar also flags the
majority, not outliers.

**Ruling (user-approved):** recalibrate both thresholds to be
percentile-derived from the real historical distribution instead of
arbitrary fixed numbers — flag the least-confident ~15% of predictions
(P15 of confidence) and the longest-predicted ~15% of resolution estimates
(P85 of resolution hours), computed once from the 500-ticket sample already
drawn in Task 6 and hardcoded as named constants (same pattern as the
original thresholds, just recalibrated values): `CONFIDENCE_THRESHOLD = 0.67`,
`RESOLUTION_HOURS_THRESHOLD = 185.0`. The `high_risk_segment` rule
(Enterprise + complexity > 8) is untouched — it's a business rule, not a
statistical calibration, and wasn't implicated in the 0% finding.

**What it costs if wrong:** the exact percentile cutoff (P15/P85 vs. some
other split) is a judgment call with no single correct answer; a different
split would produce a different real escalation rate. This was explicitly
not tuned to hit any target percentage (e.g. 90%) — the resulting rate is
reported honestly in Task 7 whatever it turns out to be, to avoid
reverse-engineering a number to match the old marketing claim.

### Task 6b: Recalibrate escalation thresholds

**Files:**
- Modify: `src/triage_gate.py` (already-reviewed Task 2 code)
- Modify: `docs/SPEC.md` §5.1
- Modify: `tests/test_triage_gate.py` (already-reviewed Task 2 tests)
- Regenerate: `reports/ESCALATION_GATE_BENCHMARK.md` (re-run Task 6's script)

**Interfaces:** `should_escalate`'s signature and return values are
unchanged — only the two constant values change. No other task's code
needs to change.

- [ ] **Step 1: Update the constants in `src/triage_gate.py`**

Change:
```python
CONFIDENCE_THRESHOLD = 0.80
RESOLUTION_HOURS_THRESHOLD = 48.0
HIGH_RISK_COMPLEXITY_THRESHOLD = 8
```
to:
```python
# Recalibrated against data/customer_support_ticket_sample.csv (2026-09-17):
# the original fixed 0.80/48.0 values were never checked against this
# dataset's real distribution and fired on the majority of tickets, not a
# minority. These are now percentile-derived: flag the least-confident ~15%
# of Tier-1 predictions (P15 of confidence) and the longest-predicted ~15%
# of resolution estimates (P85 of resolution hours), measured on a
# 500-ticket sample of real data run through the real trained models.
CONFIDENCE_THRESHOLD = 0.67
RESOLUTION_HOURS_THRESHOLD = 185.0
HIGH_RISK_COMPLEXITY_THRESHOLD = 8
```

- [ ] **Step 2: Update `tests/test_triage_gate.py` boundary/trigger values for the new thresholds**

Change `test_severe_resolution_triggers_escalation`'s `resolution_hours=72.0` to `resolution_hours=200.0` (72.0 no longer exceeds the new 185.0 threshold, so the old value would silently stop testing what its name claims).

Change `test_confidence_exactly_at_threshold_does_not_escalate`'s `confidence=0.80` to `confidence=0.67`.

Change `test_resolution_exactly_at_threshold_does_not_escalate`'s `resolution_hours=48.0` to `resolution_hours=185.0`.

Leave the other three tests unchanged (their values already sit clearly on the correct side of the new thresholds).

- [ ] **Step 3: Update `docs/SPEC.md` §5.1**

Change:
```
1. **Uncertain Priority**: Classifier max class probability < 0.80.
2. **Critical Outlier**: Regression resolution hours > 48.0 hrs.
```
to:
```
1. **Uncertain Priority**: Classifier max class probability < 0.67 (P15 of Tier-1 confidence on real ticket data).
2. **Critical Outlier**: Regression resolution hours > 185.0 hrs (P85 of Tier-1 resolution estimates on real ticket data).
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_triage_gate.py -v` (expect 6 passed) then `pytest -q` (expect 36 passed).

- [ ] **Step 5: Regenerate the benchmark report**

Run: `python scripts/benchmark_escalation_gate.py` — this overwrites `reports/ESCALATION_GATE_BENCHMARK.md` with the real measured rate under the corrected thresholds. Report the new percentage.

- [ ] **Step 6: Commit**

```bash
git add src/triage_gate.py docs/SPEC.md tests/test_triage_gate.py reports/ESCALATION_GATE_BENCHMARK.md
git commit -m "fix: recalibrate escalation thresholds against real data distribution"
```

### Task 6c: Clean up remaining stale threshold references in SPEC.md

Task 6b's reviewer flagged (as a minor, unconfirmed risk) that only §5.1 was
updated. A direct grep confirmed two more stale `0.80`/`48.0` references
still in `docs/SPEC.md` outside §5.1: an overview line and a Mermaid
flowchart. Fix both so the spec doesn't contradict itself.

**Files:** Modify `docs/SPEC.md` only. No tests (documentation-only, no
logic change).

- [ ] Change line 23 from:
  `   - Activates conditionally when classical model confidence is low (<0.80) or when predicted resolution time is abnormally severe (>48.0 hours).`
  to:
  `   - Activates conditionally when classical model confidence is low (<0.67) or when predicted resolution time is abnormally severe (>185.0 hours).`
- [ ] Change line 46 from:
  `    G -->|Confidence >= 0.80 & Resolution <= 48h| H[Return Instant ML Prediction < 15ms]`
  to:
  `    G -->|Confidence >= 0.67 & Resolution <= 185h| H[Return Instant ML Prediction < 15ms]`
- [ ] Change line 47 from:
  `    G -->|Confidence < 0.80 OR Resolution > 48h| I[Agentic Triage Engine]`
  to:
  `    G -->|Confidence < 0.67 OR Resolution > 185h| I[Agentic Triage Engine]`
- [ ] Run `grep -n "0\.80\|48\.0" docs/SPEC.md` to confirm zero remaining matches, then commit: `git add docs/SPEC.md && git commit -m "docs: finish threshold recalibration cleanup in SPEC.md"`

Task 7 (below) is then executed using this task's regenerated benchmark
report and the new threshold values (0.67 / 185.0) instead of the plan's
original 0.80 / 48.0 — Task 7's Step 3 instructions about the resume's
second bullet ("no wording change needed there") are superseded: that
bullet's stated numbers must now be updated to match the recalibrated
thresholds too.

## Post-Implementation Verification

After all 7 tasks: run `pytest -q` from `CustomerSupportAnalytics` (must be fully green), then manually hit the deployed Cloud Run `/triage_agent` endpoint with a low-complexity/high-confidence ticket (expect `escalate_to_tier2: false`, `triage_source: null`) and a low-confidence or Enterprise+complexity>8 ticket (expect `escalate_to_tier2: true`, `triage_source: "agent_llm"` if `GROQ_API_KEY` deployed correctly, else `"heuristic_fallback"`) — the same way testing was done earlier in this conversation via curl/PowerShell against `https://support-ops-api-242711953247.asia-south1.run.app/triage_agent`.
