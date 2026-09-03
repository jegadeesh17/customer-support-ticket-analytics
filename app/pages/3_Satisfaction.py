import os
import sys
import time

import streamlit as st

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.constants import (
    CATEGORIES,
    CHANNELS,
    COMPLEXITY_RANGE,
    DEFAULT_INFERENCE_ROW,
    PREVIOUS_TICKETS_RANGE,
    PRIORITY_LEVELS,
    SUBSCRIPTIONS,
    YES_NO,
)
from src.inference import load_sample_ticket, predict_satisfaction

# Ensure app is in path so we can import the shared chrome
APP_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)
from app import apply_theme, page_header, render_footer, render_nav, result_card

st.set_page_config(page_title='Satisfaction · SupportLens', page_icon='🎯',
                   layout='wide', initial_sidebar_state='collapsed')
apply_theme()
render_nav(active=3)

page_header('Satisfaction Risk',
            'Score an in-flight ticket for satisfaction risk while intervention is still possible.')

TONE_BY_BAND = {'High': 'good', 'Mid': 'warn', 'Low': 'bad'}
NOTE_BY_BAND = {
    'High': 'On track — no intervention needed.',
    'Mid': 'Monitor follow-up and confirm resolution quality.',
    'Low': 'At-risk customer — consider proactive outreach.',
}


def _index_of(options, value, fallback=0):
    """Position of a sample value within an option list, or a fallback."""
    return options.index(value) if value in options else fallback


if 'sat_sample' not in st.session_state:
    st.session_state.sat_sample = load_sample_ticket()

if st.button('Load sample ticket'):
    st.session_state.sat_sample = load_sample_ticket()

sample = st.session_state.sat_sample

with st.form('satisfaction_form'):
    st.markdown('##### Ticket')
    issue_description = st.text_area(
        'Issue description', value=str(sample.get('issue_description', '')), height=100,
    )

    st.markdown('##### Service outcome so far')
    c1, c2, c3 = st.columns(3)
    with c1:
        sla_breached = st.selectbox(
            'SLA breached', YES_NO, index=_index_of(YES_NO, sample.get('sla_breached')),
            help='Largest single driver — ~29% of the model.',
        )
    with c2:
        escalated = st.selectbox('Escalated', YES_NO, index=_index_of(YES_NO, sample.get('escalated')))
    with c3:
        first_response = st.number_input(
            'First response time (hours)', min_value=0.0,
            value=float(sample.get('first_response_time_hours', 12.0)),
            help='~10% of the model.',
        )

    st.markdown('##### Ticket & customer')
    c4, c5, c6 = st.columns(3)
    with c4:
        previous_tickets = st.number_input(
            'Previous tickets from this customer',
            min_value=PREVIOUS_TICKETS_RANGE[0], max_value=PREVIOUS_TICKETS_RANGE[1],
            value=int(sample.get('previous_tickets', DEFAULT_INFERENCE_ROW['previous_tickets'])),
            help='~12% of the model.',
        )
        category = st.selectbox(
            'Category', CATEGORIES,
            index=_index_of(CATEGORIES, sample.get('category'), CATEGORIES.index('Login Issue')),
        )
    with c5:
        issue_complexity = st.slider(
            'Issue complexity score', *COMPLEXITY_RANGE,
            int(sample.get('issue_complexity_score', 5)),
            help='~8% of the model.',
        )
        priority = st.selectbox(
            'Priority', PRIORITY_LEVELS,
            index=_index_of(PRIORITY_LEVELS, sample.get('priority'), PRIORITY_LEVELS.index('Medium')),
        )
    with c6:
        channel = st.selectbox('Channel', CHANNELS, index=_index_of(CHANNELS, sample.get('channel')))
        subscription_type = st.selectbox(
            'Subscription type', SUBSCRIPTIONS,
            index=_index_of(SUBSCRIPTIONS, sample.get('subscription_type'), SUBSCRIPTIONS.index('Premium')),
        )

    submit_button = st.form_submit_button('Predict satisfaction')

if submit_button:
    if not issue_description.strip():
        st.warning('Please provide an issue description.')
    else:
        with st.spinner('Scoring...'):
            try:
                start_time = time.time()
                payload = {
                    'issue_description': issue_description,
                    'category': category,
                    'priority': priority,
                    'channel': channel,
                    'subscription_type': subscription_type,
                    'first_response_time_hours': first_response,
                    'previous_tickets': previous_tickets,
                    'issue_complexity_score': issue_complexity,
                    'sla_breached': sla_breached,
                    'escalated': escalated,
                }
                prediction = str(predict_satisfaction(payload))
                latency = time.time() - start_time

                result_card('Predicted satisfaction', prediction,
                            NOTE_BY_BAND.get(prediction, ''),
                            TONE_BY_BAND.get(prediction, 'info'))
                st.caption(f'Scored in {latency:.2f}s.')
            except Exception as exc:
                st.error('Could not score this ticket. The model may still be downloading — try again in a moment.')
                with st.expander('Details'):
                    st.code(repr(exc))

st.caption(
    'SLA breach, escalation and first-response time are outcomes rather than ticket-creation '
    'inputs, so this reads as an in-flight risk check on an open ticket, not a pre-triage forecast.'
)
render_footer()
