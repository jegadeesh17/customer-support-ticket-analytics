import os

def get_project_root():
    """Return the CustomerSupportAnalytics project root directory."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

def get_data_path(filename='customer_support_ticket.csv'):
    return os.path.join(get_project_root(), 'data', filename)

def get_models_dir():
    return os.path.join(get_project_root(), 'models')

def get_eda_dir():
    eda_dir = os.path.join(get_project_root(), 'docs', 'eda')
    os.makedirs(eda_dir, exist_ok=True)
    return eda_dir
