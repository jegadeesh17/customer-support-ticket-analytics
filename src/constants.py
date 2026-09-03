"""Shared column definitions for training and Streamlit inference."""

ID_COLS = ['ticket_id', 'customer_name', 'customer_email']

# Columns that leak future information (exclude from features at ticket creation time)
LEAKAGE_COLS = [
    'resolution_notes',
    'ticket_resolved_date',
    'status',
    'customer_satisfaction_score',
    'resolution_time_hours',
    'first_response_time_hours',
    'escalated',
    'sla_breached',
]

TEXT_COL = 'issue_description'

PRIORITY_LEVELS = ['Urgent', 'High', 'Medium', 'Low']

# Categorical vocabularies as they appear in the training data. The fitted OneHotEncoder
# uses handle_unknown='ignore', so any value outside these lists is silently encoded as
# all-zeros — keep the UI option lists sourced from here rather than hardcoded per page.
PRODUCTS = [
    'API Service', 'Analytics Dashboard', 'Billing System', 'CRM Platform',
    'Cloud Storage', 'E-commerce Store', 'Mobile App', 'Payment Gateway',
    'Subscription Service', 'Web Portal',
]
CATEGORIES = [
    'Account Suspension', 'Bug Report', 'Data Sync Issue', 'Feature Request',
    'Login Issue', 'Payment Problem', 'Performance Issue', 'Refund Request',
    'Security Concern', 'Subscription Cancellation',
]
CHANNELS = ['Chat', 'Email', 'Phone', 'Social Media', 'Web Form']
SUBSCRIPTIONS = ['Basic', 'Enterprise', 'Free', 'Premium']
REGIONS = ['Africa', 'Asia', 'Australia', 'Europe', 'North America', 'South America']
YES_NO = ['No', 'Yes']

# Observed ranges for the numeric inputs the models weight most heavily.
PREVIOUS_TICKETS_RANGE = (0, 20)
COMPLEXITY_RANGE = (1, 10)

DEFAULT_INFERENCE_ROW = {
    'product': 'Web Portal',
    'category': 'Login Issue',
    'issue_description': 'I am unable to access my account after entering the correct credentials.',
    'priority': 'Medium',
    'channel': 'Email',
    'region': 'North America',
    'customer_age': 35,
    'customer_gender': 'Male',
    'subscription_type': 'Premium',
    'customer_tenure_months': 24,
    # Dataset median. This feature carries ~22-24% of the priority and resolution-time
    # models, so an unrepresentative default visibly skews predictions.
    'previous_tickets': 10,
    'first_response_time_hours': 12.0,
    'ticket_created_date': '2024-01-15',
    'escalated': 'No',
    'sla_breached': 'No',
    'operating_system': 'Windows',
    'browser': 'Chrome',
    'payment_method': 'Credit Card',
    'language': 'English',
    'preferred_contact_time': 'Morning',
    'issue_complexity_score': 5,
    'customer_segment': 'Individual',
}
