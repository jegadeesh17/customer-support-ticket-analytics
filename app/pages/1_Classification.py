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
    PRODUCTS,
    SUBSCRIPTIONS,
)
from src.inference import load_sample_ticket, predict_classification

# Ensure app is in path so we can import the shared chrome
APP_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)
from app import apply_theme, page_header, render_footer, render_nav, result_card

st.set_page_config(page_title='Priority Routing · SupportLens', page_icon='🎯',
                   layout='wide', initial_sidebar_state='collapsed')
apply_theme()
render_nav(active=1)

page_header('Priority Routing',
            'Classify an incoming ticket into Urgent / High / Medium / Low.')

TONE_BY_PRIORITY = {'Urgent': 'bad', 'High': 'warn', 'Medium': 'info', 'Low': 'good'}
NOTE_BY_PRIORITY = {
    'Urgent': 'Route immediately to a senior agent.',
    'High': 'Queue ahead of standard tickets.',
    'Medium': 'Handle in normal rotation.',
    'Low': 'Safe to batch with routine work.',
}


def _index_of(options, value, fallback=0):
    """Position of a sample value within an option list, or a fallback."""
    return options.index(value) if value in options else fallback


if 'cls_sample' not in st.session_state:
    st.session_state.cls_sample = load_sample_ticket()

if st.button('Load sample ticket'):
    st.session_state.cls_sample = load_sample_ticket()

sample = st.session_state.cls_sample

with st.form('classification_form'):
    st.markdown('##### Ticket')
    issue_description = st.text_area(
        'Issue description',
        value=str(sample.get('issue_description', '')),
        height=110,
        help='Strongest single driver of the prediction — urgency wording carries ~50% of the model.',
    )

    c1, c2 = st.columns(2)
    with c1:
        issue_complexity = st.slider(
            'Issue complexity score', *COMPLEXITY_RANGE,
            int(sample.get('issue_complexity_score', 5)),
            help='~25% of the model.',
        )
    with c2:
        previous_tickets = st.number_input(
            'Previous tickets from this customer',
            min_value=PREVIOUS_TICKETS_RANGE[0], max_value=PREVIOUS_TICKETS_RANGE[1],
            value=int(sample.get('previous_tickets', DEFAULT_INFERENCE_ROW['previous_tickets'])),
            help='~24% of the model.',
        )

    st.markdown('##### Context')
    c3, c4 = st.columns(2)
    with c3:
        product = st.selectbox('Product', PRODUCTS, index=_index_of(PRODUCTS, sample.get('product')))
        subscription_type = st.selectbox(
            'Subscription type', SUBSCRIPTIONS,
            index=_index_of(SUBSCRIPTIONS, sample.get('subscription_type'), SUBSCRIPTIONS.index('Premium')),
        )
    with c4:
        category = st.selectbox(
            'Category', CATEGORIES,
            index=_index_of(CATEGORIES, sample.get('category'), CATEGORIES.index('Login Issue')),
        )
        channel = st.selectbox('Channel', CHANNELS, index=_index_of(CHANNELS, sample.get('channel')))

    submit_button = st.form_submit_button('Predict priority')

if submit_button:
    if not issue_description.strip():
        st.warning('Please provide an issue description.')
    else:
        with st.spinner('Scoring ticket...'):
            try:
                start_time = time.time()
                payload = {
                    'product': product,
                    'category': category,
                    'issue_description': issue_description,
                    'subscription_type': subscription_type,
                    'channel': channel,
                    'issue_complexity_score': issue_complexity,
                    'previous_tickets': previous_tickets,
                }
                prediction = str(predict_classification(payload))
                latency = time.time() - start_time

                result_card('Predicted priority', prediction,
                            NOTE_BY_PRIORITY.get(prediction, ''),
                            TONE_BY_PRIORITY.get(prediction, 'info'))
                st.caption(f'Scored in {latency:.2f}s.')
            except Exception as exc:
                st.error('Could not score this ticket. The model may still be downloading — try again in a moment.')
                with st.expander('Details'):
                    st.code(repr(exc))

st.caption(
    'Product, category, subscription and channel are model inputs but carry little weight here. '
    'Attributes not shown (region, tenure, device, language) use dataset defaults.'
)
render_footer()
