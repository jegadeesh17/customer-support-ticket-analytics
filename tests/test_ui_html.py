"""Sanity checks for the live demo UI (api/index.html) -- the file actually
served at /app on Cloud Run. The Streamlit pages under app/ are dev-only and
are never copied into the Docker image (see Dockerfile), so they are not
covered here."""

import os

HTML_PATH = os.path.join(os.path.dirname(__file__), "..", "api", "index.html")


def _read_html():
    with open(HTML_PATH, "r", encoding="utf-8") as f:
        return f.read()


def test_escalation_tab_present():
    html = _read_html()
    assert "tab-escalation" in html
    assert "Escalation Triage" in html
    assert "handleTriage" in html


def test_escalation_tab_posts_to_triage_agent_endpoint():
    html = _read_html()
    assert "/triage_agent?force=" in html
