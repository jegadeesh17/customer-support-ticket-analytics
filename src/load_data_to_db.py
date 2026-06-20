import os
import sys

import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.db_connector import get_engine
from src.paths import get_data_path


def load_data():
    csv_path = get_data_path()
    print(f'Loading data from {csv_path}...')
    try:
        df = pd.read_csv(csv_path)
        print(f'Data loaded successfully. Shape: {df.shape}')

        engine = get_engine()
        if engine is None:
            print('Failed to get database engine. Check your .env file and PostgreSQL connection.')
            return

        print('Writing data to PostgreSQL. This may take a moment depending on data size...')
        df.to_sql('tickets', con=engine, if_exists='replace', index=False)
        print("Data successfully loaded into the 'tickets' table in the database.")

    except Exception as e:
        print(f'An error occurred: {e}')


if __name__ == '__main__':
    load_data()
