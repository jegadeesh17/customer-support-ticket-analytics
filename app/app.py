import streamlit as st

st.set_page_config(
    page_title='Customer Support Tickets Analytics',
    page_icon='🎫',
    layout='wide',
    initial_sidebar_state='expanded',
)


def main():
    st.title('Customer Support Tickets Analytics 🎫')
    st.markdown("""
    Welcome to the Customer Support Tickets Analytics platform.
    This application uses machine learning to classify ticket priority, predict resolution time,
    and estimate customer satisfaction.

    ### Available Modules

    1. **Priority Classification** — Predict ticket priority from ticket content and metadata.
    2. **Resolution Time Regression** — Estimate hours to resolve a ticket for SLA planning.
    3. **Customer Satisfaction** — Predict satisfaction level (Low / Mid / High).

    👈 **Select a module from the sidebar to get started.**
    """)

    st.info(
        'Before first use, load data with `python src/load_data_to_db.py` '
        'and train models with `python src/train_models.py`.'
    )

    with st.sidebar:
        st.markdown('### Setup')
        st.code('python src/load_data_to_db.py\npython src/train_models.py', language='bash')
        st.markdown('### Launch')
        st.code('streamlit run app/app.py', language='bash')


if __name__ == '__main__':
    main()
