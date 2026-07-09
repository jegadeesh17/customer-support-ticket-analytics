import os
import sys
import time
import requests

import streamlit as st

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.constants import PRIORITY_LEVELS
from src.inference import load_sample_ticket, predict_regression

# Ensure app is in path so we can import nav
APP_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)
from app import render_nav

st.set_page_config(page_title='SLA Prediction', page_icon='⏱️', layout='wide', initial_sidebar_state='collapsed')

st.title('SLA Prediction ⏱️')
st.markdown('Estimate **resolution time in hours** for incoming support tickets.')
render_nav()

if 'reg_sample' not in st.session_state:
    st.session_state.reg_sample = load_sample_ticket()

if st.button('Load sample ticket'):
    st.session_state.reg_sample = load_sample_ticket()

sample = st.session_state.reg_sample

with st.form('regression_form'):
    st.subheader('Ticket Details')
    category = st.text_input('Category', value=str(sample.get('category', 'Login Issue')))
    issue_description = st.text_area('Issue Description', value=str(sample.get('issue_description', '')))
    priority = st.selectbox(
        'Priority', PRIORITY_LEVELS,
        index=PRIORITY_LEVELS.index(sample['priority']) if sample.get('priority') in PRIORITY_LEVELS else 2,
    )
    first_response = st.number_input('First Response Time (hours)', min_value=0.0, value=float(sample.get('first_response_time_hours', 12.0)))
    issue_complexity = st.slider('Issue Complexity Score', 1, 10, int(sample.get('issue_complexity_score', 5)))
    submit_button = st.form_submit_button('Estimate Resolution Time')

if submit_button:
    if not issue_description.strip():
        st.warning('Please provide an issue description.')
    else:
        with st.spinner('Estimating...'):
            try:
                mode = st.session_state.get('inference_mode', 'Local Model')
                start_time = time.time()
                
                payload = {
                    'category': category,
                    'issue_description': issue_description,
                    'priority': priority,
                    'first_response_time_hours': first_response,
                    'issue_complexity_score': issue_complexity,
                }
                
                if mode == 'Local Model':
                    hours = predict_regression(payload)
                else:
                    api_url = st.session_state.get('api_url', 'http://localhost:8080')
                    response = requests.post(f"{api_url}/predict_resolution_hours", json=payload)
                    response.raise_for_status()
                    hours = response.json().get('resolution_time_hours', 0.0)
                
                latency = time.time() - start_time
                st.success(f'**Estimated Resolution Time:** {hours:.2f} hours (Mode: {mode}, Latency: {latency:.2f}s)')
            except Exception as exc:
                st.error(f'Error during prediction: {exc}')
