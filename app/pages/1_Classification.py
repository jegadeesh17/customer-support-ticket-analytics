import os
import sys

import streamlit as st

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.constants import PRIORITY_LEVELS
from src.inference import load_sample_ticket, predict_classification

st.set_page_config(page_title='Classification: Ticket Priority', page_icon='🔴')

PRODUCTS = [
    'Web Portal', 'Mobile App', 'Payment Gateway', 'E-commerce Store',
    'API Service', 'CRM Platform', 'Cloud Storage', 'Billing System',
]
SUBSCRIPTIONS = ['Free', 'Basic', 'Premium', 'Enterprise']
CHANNELS = ['Email', 'Chat', 'Phone', 'Social Media', 'Web Form']

st.title('Classification: Ticket Priority Prediction 🔴')
st.markdown('Predict ticket **priority** from description and customer metadata.')

if 'cls_sample' not in st.session_state:
    st.session_state.cls_sample = load_sample_ticket()

if st.button('Load sample ticket'):
    st.session_state.cls_sample = load_sample_ticket()

sample = st.session_state.cls_sample

with st.form('classification_form'):
    st.subheader('Ticket Information')
    product = st.selectbox('Product', PRODUCTS, index=PRODUCTS.index(sample['product']) if sample.get('product') in PRODUCTS else 0)
    category = st.text_input('Category', value=str(sample.get('category', 'Login Issue')))
    issue_description = st.text_area('Issue Description', value=str(sample.get('issue_description', '')))
    subscription_type = st.selectbox(
        'Subscription Type', SUBSCRIPTIONS,
        index=SUBSCRIPTIONS.index(sample['subscription_type']) if sample.get('subscription_type') in SUBSCRIPTIONS else 2,
    )
    channel = st.selectbox(
        'Channel', CHANNELS,
        index=CHANNELS.index(sample['channel']) if sample.get('channel') in CHANNELS else 0,
    )
    priority_hint = st.selectbox('Current Priority (optional hint)', PRIORITY_LEVELS, index=2)
    submit_button = st.form_submit_button('Predict Priority')

if submit_button:
    if not issue_description.strip():
        st.warning('Please provide an issue description.')
    else:
        with st.spinner('Predicting...'):
            try:
                prediction = predict_classification({
                    'product': product,
                    'category': category,
                    'issue_description': issue_description,
                    'subscription_type': subscription_type,
                    'channel': channel,
                    'priority': priority_hint,
                })
                st.success(f'**Predicted Priority:** {prediction}')
            except Exception as exc:
                st.error(f'Error during prediction: {exc}')
