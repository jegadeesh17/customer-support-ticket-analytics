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
    'previous_tickets': 3,
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
