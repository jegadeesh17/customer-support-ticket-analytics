import os
import pandas as pd
from sqlalchemy import create_engine
from urllib.parse import quote_plus
from dotenv import load_dotenv

def get_engine():
    """Create and return a SQLAlchemy engine based on environment variables."""
    load_dotenv()
    
    db_host = os.getenv("DB_HOST", "localhost")
    db_user = os.getenv("DB_USER", "postgres")
    db_password = os.getenv("DB_PASSWORD", "postgres")
    db_name = os.getenv("DB_NAME", "customer_support_ticket")
    db_port = os.getenv("DB_PORT", "5432")
    
    encoded_password = quote_plus(db_password)
    connection_url = f"postgresql://{db_user}:{encoded_password}@{db_host}:{db_port}/{db_name}"
    
    try:
        engine = create_engine(connection_url)
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
