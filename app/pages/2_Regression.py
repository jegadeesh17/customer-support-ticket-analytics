import os
import sys
import time

import streamlit as st

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.constants import (
    CATEGORIES,
    COMPLEXITY_RANGE,
    DEFAULT_INFERENCE_ROW,
    PREVIOUS_TICKETS_RANGE,
    PRIORITY_LEVELS,
)
from src.inference import load_sample_ticket, predict_regression

# Ensure app is in path so we can import the shared chrome
APP_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)
from app import apply_theme, page_header, render_footer, render_nav, result_card

st.set_page_config(page_title='SLA Prediction · SupportLens', page_icon='🎯',
                   layout='wide', initial_sidebar_state='collapsed')
apply_theme()
render_nav(active=2)

page_header('SLA Prediction',
            'Forecast how long an incoming ticket will take to resolve.')


def _index_of(options, value, fallback=0):
    """Position of a sample value within an option list, or a fallback."""
    return options.index(value) if value in options else fallback


def _format_duration(hours):
    """Hours as a human-readable span alongside the raw number."""
    if hours < 24:
        return f'{hours:.0f} hours'
    days = hours / 24
    return f'{days:.1f} days'


if 'reg_sample' not in st.session_state:
    st.session_state.reg_sample = load_sample_ticket()

if st.button('Load sample ticket'):
    st.session_state.reg_sample = load_sample_ticket()

sample = st.session_state.reg_sample

with st.form('regression_form'):
    st.markdown('##### Ticket')
    issue_description = st.text_area(
        'Issue description',
        value=str(sample.get('issue_description', '')),
        height=110,
        help='Urgency wording carries ~18% of the model.',
    )

    c1, c2 = st.columns(2)
    with c1:
        issue_complexity = st.slider(
            'Issue complexity score', *COMPLEXITY_RANGE,
            int(sample.get('issue_complexity_score', 5)),
            help='Largest single driver — ~36% of the model.',
        )
        priority = st.selectbox(
            'Priority', PRIORITY_LEVELS,
            index=_index_of(PRIORITY_LEVELS, sample.get('priority'), PRIORITY_LEVELS.index('Medium')),
        )
    with c2:
        previous_tickets = st.number_input(
            'Previous tickets from this customer',
            min_value=PREVIOUS_TICKETS_RANGE[0], max_value=PREVIOUS_TICKETS_RANGE[1],
            value=int(sample.get('previous_tickets', DEFAULT_INFERENCE_ROW['previous_tickets'])),
            help='~22% of the model.',
        )
        category = st.selectbox(
            'Category', CATEGORIES,
            index=_index_of(CATEGORIES, sample.get('category'), CATEGORIES.index('Login Issue')),
        )

    submit_button = st.form_submit_button('Estimate resolution time')

if submit_button:
    if not issue_description.strip():
        st.warning('Please provide an issue description.')
    else:
        with st.spinner('Estimating...'):
            try:
                start_time = time.time()
                payload = {
                    'category': category,
                    'issue_description': issue_description,
                    'priority': priority,
                    'issue_complexity_score': issue_complexity,
                    'previous_tickets': previous_tickets,
                }
                hours = float(predict_regression(payload))
                latency = time.time() - start_time

                tone = 'good' if hours < 48 else 'warn' if hours < 120 else 'bad'
                result_card('Estimated resolution time', f'{hours:.1f} hours',
                            f'About {_format_duration(hours)} from ticket creation.', tone)
                st.caption(f'Estimated in {latency:.2f}s.')
            except Exception as exc:
                st.error('Could not produce an estimate. The model may still be downloading — try again in a moment.')
                with st.expander('Details'):
                    st.code(repr(exc))

st.caption(
    'First-response time is deliberately absent: it is excluded from this model as leakage, '
    'so it cannot influence the estimate. Attributes not shown use dataset defaults.'
)
render_footer()
