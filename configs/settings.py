"""Centralized, typed application settings.

Replaces scattered `os.getenv(...)` calls in `src/db_connector.py`,
`src/agent_triage.py`, and `src/model_assets.py` with a single
`pydantic-settings`-backed object. Every field keeps the same default
that the corresponding `os.getenv(name, default)` call used previously,
so behavior is unchanged when `.env` is absent — this is purely additive.
"""

from __future__ import annotations

from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # src/db_connector.py
    DATABASE_URL: Optional[str] = None
    DB_HOST: str = "localhost"
    DB_USER: str = "postgres"
    DB_PASSWORD: str = "postgres"
    DB_NAME: str = "customer_support_ticket"
    DB_PORT: str = "5432"
    DB_SSLMODE: str = ""

    # src/agent_triage.py
    OPENROUTER_API_KEY: Optional[str] = None
    OPENAI_API_KEY: Optional[str] = None

    # src/model_assets.py
    HF_MODEL_REPO: Optional[str] = None


settings = Settings()
