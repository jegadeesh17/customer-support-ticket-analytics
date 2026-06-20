import os
import sys

import streamlit as st

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.constants import PRIORITY_LEVELS
from src.inference import load_sample_ticket, predict_satisfaction

st.set_page_config(page_title='Satisfaction Prediction', page_icon='⭐')

CHANNELS = ['Email', 'Chat', 'Phone', 'Social Media', 'Web Form']
SUBSCRIPTIONS = ['Free', 'Basic', 'Premium', 'Enterprise']

st.title('Customer Satisfaction Prediction ⭐')
st.markdown('Predict **satisfaction level** (Low / Mid / High) from ticket characteristics.')

if 'sat_sample' not in st.session_state:
    st.session_state.sat_sample = load_sample_ticket()

if st.button('Load sample ticket'):
    st.session_state.sat_sample = load_sample_ticket()

sample = st.session_state.sat_sample

with st.form('satisfaction_form'):
    st.subheader('Ticket & Service Details')
    issue_description = st.text_area('Issue Description', value=str(sample.get('issue_description', '')))
    category = st.text_input('Category', value=str(sample.get('category', 'Login Issue')))
    priority = st.selectbox(
        'Priority', PRIORITY_LEVELS,
        index=PRIORITY_LEVELS.index(sample['priority']) if sample.get('priority') in PRIORITY_LEVELS else 2,
    )
    channel = st.selectbox(
        'Channel', CHANNELS,
        index=CHANNELS.index(sample['channel']) if sample.get('channel') in CHANNELS else 0,
    )
    subscription_type = st.selectbox(
        'Subscription Type', SUBSCRIPTIONS,
        index=SUBSCRIPTIONS.index(sample['subscription_type']) if sample.get('subscription_type') in SUBSCRIPTIONS else 2,
    )
    first_response = st.number_input('First Response Time (hours)', min_value=0.0, value=float(sample.get('first_response_time_hours', 12.0)))
    submit_button = st.form_submit_button('Predict Satisfaction')

if submit_button:
    if not issue_description.strip():
        st.warning('Please provide an issue description.')
    else:
        with st.spinner('Predicting...'):
            try:
                prediction = predict_satisfaction({
                    'issue_description': issue_description,
                    'category': category,
                    'priority': priority,
                    'channel': channel,
                    'subscription_type': subscription_type,
                    'first_response_time_hours': first_response,
                })
                if prediction == 'High':
                    st.success(f'**Predicted Satisfaction:** {prediction} — Strong experience expected')
                elif prediction == 'Mid':
                    st.warning(f'**Predicted Satisfaction:** {prediction} — Monitor follow-up')
                else:
                    st.error(f'**Predicted Satisfaction:** {prediction} — At-risk customer')
            except Exception as exc:
                st.error(f'Error during prediction: {exc}')
