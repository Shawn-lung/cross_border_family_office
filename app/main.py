"""Main Streamlit entry point with institutional dashboard navigation."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# Ensure `app` package imports resolve when running: streamlit run app/main.py
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.modules import m1_alm_tax, m2_factors, m3_retirement, m4_sandbox


def _apply_terminal_theme() -> None:
    """Apply a Bloomberg-like dark styling layer for the full app."""
    st.markdown(
        """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;600;700&family=IBM+Plex+Mono:wght@400;600&display=swap');

:root {
    --bg-main: #0a111a;
    --bg-pane: #111c2a;
    --bg-sidebar: #0d1622;
    --text-main: #e7edf5;
    --text-muted: #97a7bb;
    --accent: #f2a900;
    --good: #2dd4bf;
    --bad: #f87171;
}

.stApp {
    background: radial-gradient(1200px 500px at 10% -10%, #1b2d44 0%, var(--bg-main) 50%);
    color: var(--text-main);
    font-family: "IBM Plex Sans", sans-serif;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #121f2f 0%, var(--bg-sidebar) 100%);
    border-right: 1px solid #1e3147;
}

h1, h2, h3, h4 {
    letter-spacing: 0.02em;
    color: var(--text-main);
}

[data-testid="stMetricValue"] {
    font-family: "IBM Plex Mono", monospace;
}

div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div,
div[data-baseweb="textarea"] > div {
    background-color: var(--bg-pane);
}

[data-testid="stRadio"] label p {
    font-size: 0.95rem;
    color: var(--text-main);
}

section.main > div {
    padding-top: 1rem;
}
</style>
""",
        unsafe_allow_html=True,
    )


def _render_sandbox_placeholder() -> None:
    """Render Module 4 placeholder content until research engine is implemented."""
    st.subheader("Module 4: Alpha Sandbox (Placeholder)")
    st.info("WRDS SQL ingestion, Factor Drift Analysis, and Backtesting Engine will be added here.")
    st.markdown("#### Planned Blocks")
    st.markdown("- WRDS data connector and query execution")
    st.markdown("- Factor drift analysis workbench")
    st.markdown("- Backtesting engine placeholder")
    st.markdown("- Future ML research workflow")


def main() -> None:
    """Run the dashboard shell and route to module screens."""
    st.set_page_config(
        page_title="Cross-Border Family Office",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _apply_terminal_theme()

    st.title("Cross-Border Family Office Dashboard")
    st.caption("Institutional Wealth Intelligence Console")

    options = (
        "Module 1: Tax Simulator",
        "Module 2: Factor Engine",
        "Module 3: Monte Carlo",
        "Module 4: Sandbox Placeholder",
    )
    selected = st.sidebar.radio("Navigation", options=options, index=0)

    if selected == "Module 1: Tax Simulator":
        m1_alm_tax.render()
    elif selected == "Module 2: Factor Engine":
        m2_factors.render()
    elif selected == "Module 3: Monte Carlo":
        m3_retirement.render()
    else:
        if hasattr(m4_sandbox, "render"):
            m4_sandbox.render()
        else:
            _render_sandbox_placeholder()


if __name__ == "__main__":
    main()
