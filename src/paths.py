import os

def get_project_root():
    """Return the CustomerSupportAnalytics project root directory."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

def get_data_path(filename='customer_support_ticket.csv'):
    data_dir = os.path.join(get_project_root(), 'data')
    preferred = os.path.join(data_dir, filename)
    if os.path.exists(preferred):
        return preferred
    sample = os.path.join(data_dir, 'customer_support_ticket_sample.csv')
    if os.path.exists(sample):
        return sample
    return preferred

def get_models_dir():
    models_dir = os.path.join(get_project_root(), 'models')
    from src.model_assets import ensure_models

    return ensure_models(models_dir)

def get_eda_dir():
    eda_dir = os.path.join(get_project_root(), 'docs', 'eda')
    os.makedirs(eda_dir, exist_ok=True)
    return eda_dir
