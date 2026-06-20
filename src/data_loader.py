import os
import pandas as pd

from src.paths import get_data_path
from src.db_connector import get_engine


def load_tickets(use_db=True):
    """Load tickets from PostgreSQL, falling back to local CSV."""
    if use_db:
        engine = get_engine()
        if engine:
            try:
                df = pd.read_sql('SELECT * FROM tickets', engine)
                if not df.empty:
                    print(f'Loaded {len(df)} rows from database.')
                    return df
            except Exception as exc:
                print(f'Database query failed: {exc}')

    csv_path = get_data_path()
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        print(f'Loaded {len(df)} rows from {csv_path}.')
        return df

    print('No data source available.')
    return pd.DataFrame()
