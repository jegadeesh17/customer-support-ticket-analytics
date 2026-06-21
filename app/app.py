import streamlit as st
import os
import sys

def render_nav():
    # Hide the default sidebar completely
    st.markdown("""
        <style>
            [data-testid="stSidebar"] {display: none;}
            [data-testid="collapsedControl"] {display: none;}
            .stApp, [data-testid="stAppViewContainer"] {
                background: linear-gradient(135deg, #FFFAF5 0%, #F5E6D3 100%);
            }
            h1, .main-header, .sub-header {
                text-align: center !important;
            }
        </style>
    """, unsafe_allow_html=True)
    
    # 4 verticals navigation at the top
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.page_link("app.py", label="Dashboard", icon="🏠")
    with c2:
        st.page_link("pages/1_Classification.py", label="Priority Routing", icon="⚡")
    with c3:
        st.page_link("pages/2_Regression.py", label="SLA Prediction", icon="⏱️")
    with c4:
        st.page_link("pages/3_Satisfaction.py", label="Satisfaction Insights", icon="😊")
    st.markdown("---")

def main():
    st.set_page_config(
        page_title='SupportLens AI | Intelligent Analytics',
        page_icon='🎯',
        layout='wide',
        initial_sidebar_state='collapsed',
    )

    # Custom CSS for better aesthetics
    st.markdown("""
    <style>
        .main-header {
            font-size: 2.5rem !important;
            font-weight: 700 !important;
            color: #B45309;
            margin-bottom: 0 !important;
            padding-bottom: 0 !important;
        }
        .sub-header {
            font-size: 1.2rem !important;
            color: #78350F;
            margin-top: 0 !important;
            padding-bottom: 1rem !important;
        }
    </style>
    """, unsafe_allow_html=True)
    # Hero Section
    st.markdown('<p class="main-header">SupportLens AI 🎯</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Intelligent Customer Support Analytics Platform</p>', unsafe_allow_html=True)
    st.markdown("---")

    render_nav()

    # Features Section
    st.subheader("🚀 Core Capabilities")
    
    c1, c2, c3 = st.columns(3)
    
    with c1:
        with st.container(border=True):
            st.markdown("### ⚡ Priority Routing")
            st.markdown("""
            Automatically classify incoming tickets and assign appropriate priority levels to ensure critical issues are handled immediately without manual triage.
            """)
            
    with c2:
        with st.container(border=True):
            st.markdown("### ⏱️ SLA Prediction")
            st.markdown("""
            Estimate the exact resolution time for every ticket. Help your team plan resources dynamically and guarantee Service Level Agreements.
            """)

    with c3:
        with st.container(border=True):
            st.markdown("### 😊 Satisfaction Insights")
            st.markdown("""
            Proactively gauge customer sentiment and predict satisfaction scores before the ticket is even closed, allowing preemptive action.
            """)
            
    st.markdown("<br>", unsafe_allow_html=True)
    st.info("💡 **Navigation:** Use the menu at the top of the page to access specific analytical modules.")



if __name__ == '__main__':
    main()
