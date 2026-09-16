"""Tests for src/db_connector.py engine construction and connect timeout."""

import os
import sys
from unittest.mock import MagicMock, patch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src import db_connector


def test_get_engine_uses_connect_timeout_with_database_url():
    """When DATABASE_URL is set, get_engine() should pass it straight through
    to create_engine along with the connect_timeout safeguard."""
    mock_engine = MagicMock()
    with patch.object(
        db_connector.settings, "DATABASE_URL", "postgresql://user:pass@host:5432/db"
    ):
        with patch(
            "src.db_connector.create_engine", return_value=mock_engine
        ) as mock_create_engine:
            engine = db_connector.get_engine()

    assert engine is mock_engine
    mock_create_engine.assert_called_once_with(
        "postgresql://user:pass@host:5432/db",
        connect_args={"connect_timeout": 5},
    )


def test_get_engine_uses_connect_timeout_with_discrete_vars():
    """When DATABASE_URL is unset, get_engine() builds the URL from the
    discrete DB_* settings but must still apply the connect_timeout safeguard."""
    mock_engine = MagicMock()
    with patch.object(db_connector.settings, "DATABASE_URL", None), patch.object(
        db_connector.settings, "DB_HOST", "dbhost"
    ), patch.object(db_connector.settings, "DB_USER", "dbuser"), patch.object(
        db_connector.settings, "DB_PASSWORD", "dbpass"
    ), patch.object(
        db_connector.settings, "DB_NAME", "dbname"
    ), patch.object(
        db_connector.settings, "DB_PORT", "5432"
    ), patch.object(
        db_connector.settings, "DB_SSLMODE", ""
    ):
        with patch(
            "src.db_connector.create_engine", return_value=mock_engine
        ) as mock_create_engine:
            engine = db_connector.get_engine()

    expected_url = "postgresql://dbuser:dbpass@dbhost:5432/dbname"
    assert engine is mock_engine
    mock_create_engine.assert_called_once_with(
        expected_url,
        connect_args={"connect_timeout": 5},
    )
