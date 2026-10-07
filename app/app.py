import json
import os

import streamlit as st

REPORTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'reports'))

NAV_ITEMS = [
    ('app.py', 'Dashboard'),
    ('pages/1_Classification.py', 'Priority Routing'),
    ('pages/2_Regression.py', 'SLA Prediction'),
    ('pages/3_Satisfaction.py', 'Satisfaction'),
]


def apply_theme():
    """Global chrome. Call immediately after set_page_config, before any content,
    so Streamlit's default sidebar and toolbar never flash into view."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700&display=swap');

        /* --- strip Streamlit's own chrome so this reads as an app, not a notebook --- */
        [data-testid="stSidebar"], [data-testid="collapsedControl"],
        [data-testid="stToolbar"], [data-testid="stDecoration"],
        #MainMenu, header, footer {display: none !important;}

        .stApp, [data-testid="stAppViewContainer"] {
            background: #f4f6fa;
        }
        html, body, [class*="css"], .stApp {
            font-family: 'Outfit', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        }
        .block-container {
            padding-top: 1.6rem !important;
            padding-bottom: 3rem !important;
            max-width: 1180px;
        }

        /* --- top bar --- */
        .sl-topbar {
            display: flex; align-items: baseline; gap: .6rem;
            padding-bottom: .5rem;
        }
        .sl-brand {
            font-size: 1.05rem; font-weight: 700; color: #1d4fd1; letter-spacing: -.01em;
        }
        .sl-brand-sub {
            font-size: .78rem; color: #536278; font-weight: 500;
        }

        /* nav links rendered by st.page_link */
        [data-testid="stPageLink"] a, a[data-testid="stPageLink-NavLink"] {
            border-radius: 8px; padding: .38rem .55rem !important;
            border: 1px solid transparent; font-size: .88rem !important;
            font-weight: 500; justify-content: center;
        }
        [data-testid="stPageLink"] a:hover, a[data-testid="stPageLink-NavLink"]:hover {
            background: #eef2f7 !important;
        }

        .sl-navitem-active {
            border-radius: 8px; padding: .38rem .55rem; text-align: center;
            font-size: .88rem; font-weight: 600; color: #1d4fd1;
            background: #eef2f7; border: 1px solid #cfd7e3;
        }

        /* --- headings --- */
        h1 {
            font-size: 1.65rem !important; font-weight: 700 !important;
            color: #14202e !important; letter-spacing: -.02em;
            padding-top: .2rem !important; padding-bottom: .1rem !important;
        }
        h2 {font-size: 1.15rem !important; font-weight: 600 !important; color: #14202e !important;}
        h3 {font-size: .95rem !important; font-weight: 600 !important; color: #14202e !important;}
        .sl-page-sub {
            color: #435166; font-size: .92rem; margin: -.2rem 0 1.1rem 0;
        }

        /* --- cards --- */
        [data-testid="stForm"], .sl-card {
            background: #FFFFFF; border: 1px solid #cfd7e3 !important;
            border-radius: 14px; padding: 1.15rem 1.25rem !important;
            box-shadow: 0 1px 2px rgba(20,32,46,.06);
        }

        /* --- KPI tiles --- */
        [data-testid="stMetric"] {
            background: #FFFFFF; border: 1px solid #cfd7e3; border-radius: 12px;
            padding: .85rem 1rem;
        }
        [data-testid="stMetricLabel"] p {
            font-size: .76rem !important; font-weight: 600 !important;
            color: #536278 !important; text-transform: uppercase; letter-spacing: .05em;
        }
        [data-testid="stMetricValue"] {
            font-size: 1.7rem !important; font-weight: 700 !important; color: #14202e !important;
        }

        /* --- inputs --- */
        [data-testid="stWidgetLabel"] p {
            font-size: .82rem !important; font-weight: 600 !important; color: #14202e !important;
        }
        .stTextArea textarea, .stTextInput input {border-radius: 9px !important;}
        .stButton button, .stFormSubmitButton button {
            border-radius: 9px !important; font-weight: 600 !important;
            padding: .48rem 1.05rem !important; border: 1px solid #cfd7e3 !important;
        }
        .stFormSubmitButton button {
            background: #2563eb !important; color: #fff !important; border-color: #2563eb !important;
        }
        .stFormSubmitButton button:hover {background: #1d4fd1 !important;}

        /* --- result panel --- */
        .sl-result {
            border-radius: 14px; padding: 1.15rem 1.3rem; margin-top: .3rem;
            border: 1px solid; display: flex; flex-direction: column; gap: .18rem;
        }
        .sl-result-label {
            font-size: .74rem; font-weight: 700; text-transform: uppercase;
            letter-spacing: .07em; opacity: .72;
        }
        .sl-result-value {font-size: 2.05rem; font-weight: 700; line-height: 1.15;}
        .sl-result-note {font-size: .86rem; opacity: .82;}
        .sl-good {background: #e3f5ec; border-color: #a8d9c1; color: #0b6b4a;}
        .sl-warn {background: #fdf0d6; border-color: #ecc98a; color: #7a4a00;}
        .sl-bad  {background: #fde8e7; border-color: #efb3b0; color: #a3201f;}
        .sl-info {background: #FFFFFF; border-color: #cfd7e3; color: #14202e;}

        /* --- footer --- */
        .sl-footer {
            margin-top: 2.2rem; padding-top: .9rem; border-top: 1px solid #cfd7e3;
            color: #536278; font-size: .76rem; display: flex;
            justify-content: space-between; flex-wrap: wrap; gap: .4rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_nav(active=0):
    """Top bar. The active entry renders as a static pill rather than a link, which
    reads as a current-page indicator and avoids depending on Streamlit's DOM."""
    st.markdown(
        '<div class="sl-topbar"><span class="sl-brand">SupportLens</span>'
        '<span class="sl-brand-sub">Support Ops Intelligence</span></div>',
        unsafe_allow_html=True,
    )
    for i, (col, (path, label)) in enumerate(zip(st.columns(len(NAV_ITEMS)), NAV_ITEMS)):
        with col:
            if i == active:
                st.markdown(f'<div class="sl-navitem-active">{label}</div>',
                            unsafe_allow_html=True)
            else:
                st.page_link(path, label=label)
    st.markdown('<div style="height:.55rem"></div>', unsafe_allow_html=True)


def page_header(title, subtitle):
    st.markdown(f'# {title}')
    st.markdown(f'<p class="sl-page-sub">{subtitle}</p>', unsafe_allow_html=True)


def result_card(label, value, note='', tone='info'):
    """Consistent prediction output. tone: good | warn | bad | info."""
    note_html = f'<div class="sl-result-note">{note}</div>' if note else ''
    st.markdown(
        f'<div class="sl-result sl-{tone}">'
        f'<div class="sl-result-label">{label}</div>'
        f'<div class="sl-result-value">{value}</div>'
        f'{note_html}</div>',
        unsafe_allow_html=True,
    )


def load_metrics():
    """Held-out metrics exported by scripts/export_evaluation.py (small JSON, no model load)."""
    path = os.path.join(REPORTS_DIR, 'metrics.json')
    if not os.path.exists(path):
        return {}
    with open(path, encoding='utf-8') as fh:
        return json.load(fh).get('tasks', {})


def render_footer(note=''):
    left = note or 'Predictions are model estimates, not guarantees.'
    st.markdown(
        f'<div class="sl-footer"><span>{left}</span>'
        f'<span>Models: scikit-learn pipelines · evaluated on a held-out 20% split</span></div>',
        unsafe_allow_html=True,
    )


def main():
    st.set_page_config(
        page_title='SupportLens · Support Ops Intelligence',
        page_icon='🎯',
        layout='wide',
        initial_sidebar_state='collapsed',
    )
    apply_theme()
    render_nav(active=0)

    page_header(
        'Support Ops Intelligence',
        'Triage incoming tickets, forecast resolution time, and flag at-risk customers.',
    )

    metrics = load_metrics()
    if metrics:
        st.markdown('##### Model performance (held-out test set)')
        c1, c2, c3 = st.columns(3)
        cls = metrics.get('priority_classification', {})
        reg = metrics.get('resolution_regression', {})
        sat = metrics.get('satisfaction_classification', {})
        with c1:
            st.metric('Priority accuracy', f"{cls.get('accuracy', 0):.1%}",
                      help=f"{cls.get('model', '—')} · target {cls.get('target', '—')}")
        with c2:
            st.metric('Resolution R²', f"{reg.get('r2', 0):.3f}",
                      help=f"{reg.get('model', '—')} · target {reg.get('target', '—')}")
        with c3:
            st.metric('Satisfaction accuracy', f"{sat.get('accuracy', 0):.1%}",
                      help=f"{sat.get('model', '—')} · target {sat.get('target', '—')}")
        st.markdown('<div style="height:.9rem"></div>', unsafe_allow_html=True)

    st.markdown('##### Modules')
    c1, c2, c3 = st.columns(3)
    modules = [
        (c1, 'Priority Routing',
         'Classify an incoming ticket into Urgent / High / Medium / Low so critical issues '
         'skip manual triage.', 'pages/1_Classification.py'),
        (c2, 'SLA Prediction',
         'Forecast resolution time in hours to plan capacity and protect service-level '
         'commitments.', 'pages/2_Regression.py'),
        (c3, 'Satisfaction',
         'Score an in-flight ticket for satisfaction risk while there is still time to '
         'intervene.', 'pages/3_Satisfaction.py'),
    ]
    for col, title, body, target in modules:
        with col:
            with st.container(border=True):
                st.markdown(f'**{title}**')
                st.markdown(
                    f'<span style="font-size:.86rem;color:#435166">{body}</span>',
                    unsafe_allow_html=True,
                )
                st.page_link(target, label='Open')

    render_footer(
        'Labels are engineered from business rules — treat metrics as pipeline validity, '
        'not production KPIs.'
    )


if __name__ == '__main__':
    main()
