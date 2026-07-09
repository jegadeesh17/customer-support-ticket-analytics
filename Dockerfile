FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

COPY requirements-api.txt ./
RUN pip install --no-cache-dir -r requirements-api.txt huggingface_hub

COPY api/ api/
COPY src/ src/
COPY data/customer_support_ticket_sample.csv data/
RUN mkdir -p models

EXPOSE 8080
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8080}"]
