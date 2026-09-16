# Support Ops Intelligence & Agentic Triage Platform — Technical Specification

---

## 1. Document Control & System Overview

| Field | Specification |
| :--- | :--- |
| **System** | Support Ops Intelligence & Agentic Escalation Platform |
| **Document** | docs/SPEC.md |
| **Version** | 2.0.0 (Production Grade) |
| **Service Tier** | High-throughput Multi-task ML + Fallback Agentic Escalation |
| **Maintainer** | ML & Platform Engineering |

### 1.1 Problem Definition & Business Context
Modern SaaS customer support desks receive hundreds of thousands of incoming tickets across heterogeneous channels (email, web portal, chat, phone). Manual triage introduces variable latency, SLA breaches for critical enterprise accounts, and suboptimal routing.
This platform provides:
1. **Low-Latency Classical Multi-Task ML Inference** (target <15ms per prediction, see §4) for every incoming ticket:
   - **Priority Classification**: 4-class (Urgent, High, Medium, Low)
   - **Resolution Time Regression**: Log-transformed hours to resolution
   - **Customer Satisfaction Band**: 3-class (High, Mid, Low)
2. **Autonomous Agentic Triage Tier**:
   - Activates conditionally when classical model confidence is low (<0.80) or when predicted resolution time is abnormally severe (>48.0 hours).
   - Generates structured diagnostic JSON: customer frustration score (1-10), root cause category, empathetic response draft, and escalation routing instructions.

---

## 2. System Architecture

`mermaid
flowchart TD
    A[Incoming Support Ticket Payload] --> B[FastAPI Gateway /predict_* /triage_agent]
    B --> C{Pydantic Schema Validation}
    C -->|Invalid| D[422 Unprocessable Entity]
    C -->|Valid| E[Preprocessing & ColumnTransformer]
    
    subgraph Classical Multi-Task Pipeline
        E --> F1[GradientBoosting Priority Classifier]
        E --> F2[RandomForest Resolution Regressor]
        E --> F3[RandomForest Satisfaction Classifier]
    end

    F1 --> G{Escalation Gate}
    F2 --> G
    
    G -->|Confidence >= 0.80 & Resolution <= 48h| H[Return Instant ML Prediction < 15ms]
    G -->|Confidence < 0.80 OR Resolution > 48h| I[Agentic Triage Engine]
    
    subgraph Autonomous Escalation Tier
        I --> J[Structured LLM Prompting via OpenRouter/OpenAI]
        J --> K[Pydantic JSON Contract Validation]
        K --> L[AgentTriageResult: Frustration Score, Root Cause, Auto-Draft, Action]
    end
    
    L --> M[Enriched Response Payload with Human-in-the-Loop Handoff]
`

---

## 3. Data Flow & Leakage Defenses

### 3.1 Strict Pre-Resolution Feature Boundary
In support ticket analytics, post-creation attributes routinely contaminate training sets in naive implementations. We enforce strict data boundary protections:

| Column Name | Category | Production Policy |
| :--- | :--- | :--- |
| esolution_notes | Post-Resolution | **Strictly Excluded** (Future text leak) |
| 	icket_resolved_date | Post-Resolution | **Strictly Excluded** (Direct duration target leak) |
| status | Lifecycle | **Strictly Excluded** (Indicates completion) |
| customer_satisfaction_score | Post-Resolution | **Strictly Excluded** from priority/resolution |
| esolution_time_hours | Target Variable | **Strictly Excluded** from feature transformers |
| escalated / sla_breached | Post-Triage | **Excluded at ticket creation** |

### 3.2 Imputation & Unknown Encoding Safety
- **Categoricals**: OneHotEncoder(handle_unknown='ignore') maps previously unseen products, channels, or regions to zero-vectors rather than crashing runtime inference.
- **Numerics**: SimpleImputer(strategy='median') ensures missing tenure or complexity values are neutrally filled using training-distribution medians.

---

## 4. Latency, Concurrency & Cost Budgets

> **Status: targets, not yet benchmarked in production.** The figures below were originally
> asserted design goals with no benchmark code backing them. `scripts/benchmark_inference.py`
> now exists to measure the Classical ML Tier (loads each model bundle once per process, matching
> the `@lru_cache` warm-process pattern in `src/inference.py`, then times N repeated single-row
> predictions). A local run on a warm dev-machine process (200 iterations/model, CPU inference,
> Windows laptop — not the production Cloud Run instance/CPU class) measured:
>
> | Task | p50 | p95 | p99 |
> | :--- | ---: | ---: | ---: |
> | classification | ~24 ms | ~43 ms | ~64 ms |
> | regression | ~50 ms | ~78 ms | ~95 ms |
> | satisfaction | ~50 ms | ~86 ms | ~125 ms |
>
> These are roughly 2-3x the table's p50/p95 targets and up to ~3x at p99, and they exclude request/response
> (de)serialization and network overhead, so real API latency will be somewhat higher still. The gap is likely
> explained by unpickling/prediction cost of the larger regression/satisfaction bundles (regression_model.pkl
> is 218 MB, see `src/inference.py`) plus this being a laptop rather than the deployed Cloud Run CPU class.
> The table below is kept as the aspirational **target**; treat it as directional until it is re-measured
> against the actual Cloud Run service (e.g. with `scripts/benchmark_inference.py` run inside the deployed
> container, or an end-to-end load test against `/predict_*`). The Agentic Escalation Tier numbers remain
> entirely unverified — no benchmark exists for that tier yet.

| Metric | Classical ML Tier (target) | Agentic Escalation Tier (target) |
| :--- | :--- | :--- |
| **p50 Latency** | <= 8 ms | <= 800 ms |
| **p95 Latency** | <= 25 ms | <= 2,200 ms |
| **p99 Latency** | <= 40 ms | <= 3,500 ms |
| **Throughput Target**| >= 250 req/sec (single worker) | >= 15 req/sec (concurrent LLM tasks) |
| **Max Cost Per Ticket** | .0000 (Local CPU) | <= .0012 (Quantized / Cost-efficient LLM) |
| **Timeout Budget** | 100 ms | 5,000 ms (fallback to heuristic rules) |

---

## 5. Agentic Escalation Tier Specification

### 5.1 Trigger Conditions
The agentic escalation tier is invoked either explicitly via POST /triage_agent or when downstream rules detect an edge case:
1. **Uncertain Priority**: Classifier max class probability < 0.80.
2. **Critical Outlier**: Regression resolution hours > 48.0 hrs.
3. **High-Risk Segment**: Enterprise subscription with complexity score > 8.

### 5.2 Structured Output Contract
`json
{
  ticket_id: string,
  escalate_to_tier2: true,
  customer_frustration_score: 8,
  root_cause_category: Database Connectivity Timeout,
  urgency_reasoning: Enterprise customer experiencing production outage with high tenure.,
  recommended_action: Route immediately to Database SRE on-call.,
  auto_drafted_response: Dear Alex, we have flagged this as critical and our core engineering team is actively investigating...,
  confidence: 0.92
}
`

---

## 6. Failure Modes, Timeouts & Fallback Strategies

| Failure Mode | Impact | Mitigation Strategy |
| :--- | :--- | :--- |
| **Model Bundle Unpickling Error** | Service startup failure | Dynamic version compatibility shims for Cython loss functions. |
| **LLM Gateway Timeout / Outage** | Triage latency spike | Circuit breaker with deterministic keyword-based heuristics fallback within 50ms. |
| **Missing Input Features** | Inference ValueError | uild_inference_row automatically hydrates dataset medians and empty text fallbacks. |
| **PostgreSQL Connection Drops** | Historical retrieval failure | Graceful degradation to CSV cold cache without failing healthcheck. |

---

## 7. Containerization & Security Specifications

- **Base Image**: python:3.11-slim with multi-stage build.
- **Least Privilege Execution**: Non-root ppuser (UID 10001, GID 10001).
- **Health Probes**: GET /health responding within 200ms with model presence status.
- **Port Allocations**:
  - FastAPI Service: 8002
  - Streamlit Multi-Page UI: 8502
  - PostgreSQL Storage: 5432
