import pandas as pd
from sqlalchemy import create_engine
from urllib.parse import quote_plus

from configs.settings import settings

# Explicit connect timeout (seconds) so an unreachable Postgres host fails fast
# instead of hanging on OS/driver defaults. This is the psycopg2 connect kwarg;
# both engines below use the psycopg2 driver (see requirements-api.txt).
DB_CONNECT_TIMEOUT_SECONDS = 5


def get_engine():
    """Create and return a SQLAlchemy engine based on environment variables."""
    database_url = settings.DATABASE_URL
    if database_url:
        try:
            return create_engine(
                database_url,
                connect_args={"connect_timeout": DB_CONNECT_TIMEOUT_SECONDS},
            )
        except Exception as e:
            print(f"Error creating database engine from DATABASE_URL: {e}")
            return None

    db_host = settings.DB_HOST
    db_user = settings.DB_USER
    db_password = settings.DB_PASSWORD
    db_name = settings.DB_NAME
    db_port = settings.DB_PORT
    db_sslmode = settings.DB_SSLMODE

    encoded_password = quote_plus(db_password)
    connection_url = f"postgresql://{db_user}:{encoded_password}@{db_host}:{db_port}/{db_name}"
    if db_sslmode:
        connection_url = f"{connection_url}?sslmode={db_sslmode}"

    try:
        engine = create_engine(
            connection_url,
            connect_args={"connect_timeout": DB_CONNECT_TIMEOUT_SECONDS},
        )
        return engine
    except Exception as e:
        print(f"Error creating database engine: {e}")
        return None

def get_connection():
    """Return a raw connection from the SQLAlchemy engine."""
    engine = get_engine()
    if engine:
        try:
            return engine.connect()
        except Exception as e:
            print(f"Error connecting to database: {e}")
            return None
    return None
