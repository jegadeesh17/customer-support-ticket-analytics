# Multi-Stage Production Dockerfile for CustomerSupportAnalytics Inference API

# Stage 1: Dependency builder
FROM python:3.11-slim AS builder

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends build-essential curl && rm -rf /var/lib/apt/lists/*

COPY requirements-api.txt ./
RUN python -m venv /opt/venv && \
    /opt/venv/bin/pip install --no-cache-dir --upgrade pip && \
    /opt/venv/bin/pip install --no-cache-dir -r requirements-api.txt huggingface_hub

# Stage 2: Minimal non-root runner
FROM python:3.11-slim AS runner

WORKDIR /app
ENV PATH=/opt/venv/bin: \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8002

RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

RUN groupadd -g 10001 appgroup && \
    useradd -u 10001 -g appgroup -s /bin/bash -m appuser

COPY --from=builder /opt/venv /opt/venv

COPY --chown=appuser:appgroup api/ api/
COPY --chown=appuser:appgroup src/ src/
COPY --chown=appuser:appgroup data/customer_support_ticket_sample.csv data/
RUN mkdir -p models && chown -R appuser:appgroup /app

USER appuser

EXPOSE 8002

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:/health || exit 1

CMD [sh, -c, uvicorn api.main:app --host 0.0.0.0 --port ]
