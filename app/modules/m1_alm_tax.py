"""Module 1 ALM & Tax Simulator UI for capital drawdown and Box 3 threshold tracking."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from datetime import datetime

from app.utils.forex_engine import ForexAPIError, fetch_live_fx_rates
from app.utils.tax_logic import BOX3_EXEMPTION_LIMIT_EUR, calculate_box3_tax

DEFAULT_FALLBACK_EUR_TWD = 35.0

def _build_timeline_series(
    starting_capital_eur: float,
    total_tuition_eur: float,
    monthly_living_eur: float,
    risk_free_yield_pct: float,
    start_date: str = "2026-08-01",
    months: int = 18,  # 1 year program + 6 month buffer
) -> pd.DataFrame:
    """Build timeline with a ONE-TIME tuition cliff and monthly living expenses."""
    monthly_yield = (risk_free_yield_pct / 100.0) / 12.0
    nav = starting_capital_eur
    
    dates = pd.date_range(start=start_date, periods=months, freq='MS')
    rows = []
    
    tuition_paid = False

    for current_date in dates:
        # Apply yield
        nav = nav * (1.0 + monthly_yield)
        
        # ONE-TIME August Cliff: Deduct massive tuition only once
        if current_date.month == 8 and not tuition_paid:
            nav -= total_tuition_eur
            tuition_paid = True
            
        # Monthly living expense deduction
        nav -= monthly_living_eur
        
        # Ensure NAV doesn't drop below 0 for the chart
        nav = max(0, nav)
        
        rows.append({
            "Date": current_date,
            "NAV_EUR": nav,
            "Is_Jan_1st": current_date.month == 1
        })

    return pd.DataFrame(rows)


def render() -> None:
    """Render the ALM & Tax Simulator screen."""
    st.subheader("Asset-Liability Matching & Box 3 Tax Shield")
    st.markdown("Visualizing the 'Self-Clearing' tax strategy for a 1-Year Master's program.")

    col1, col2 = st.columns(2)
    with col1:
        starting_capital_twd = st.slider("Starting Capital (TWD)", 500_000, 5_000_000, 2_700_000, 50_000)
        total_tuition_eur = st.slider("Total Tuition (EUR) - Paid Once in Aug", 0, 50_000, 24_900, 100)
    with col2:
        monthly_living_eur = st.slider("Monthly Living Expenses (EUR)", 0, 3_000, 1_500, 50)
        risk_free_yield_pct = st.slider("Risk-Free Yield (%) e.g., BOXX", 0.0, 10.0, 3.5, 0.1)

    try:
        rates = fetch_live_fx_rates()
        eur_twd = rates.eur_twd
        st.caption(f"Live EUR/TWD Rate: {eur_twd:.4f}")
    except ForexAPIError as exc:
        eur_twd = DEFAULT_FALLBACK_EUR_TWD
        st.caption(f"Live FX offline. Using baseline EUR/TWD: {DEFAULT_FALLBACK_EUR_TWD:.2f}")

    starting_capital_eur = starting_capital_twd / eur_twd
    
    df = _build_timeline_series(
        starting_capital_eur=starting_capital_eur,
        total_tuition_eur=float(total_tuition_eur),
        monthly_living_eur=float(monthly_living_eur),
        risk_free_yield_pct=float(risk_free_yield_pct),
        months=18
    )

    # Build Institutional Chart
    fig = go.Figure()

    # The NAV Area Chart
    fig.add_trace(go.Scatter(
        x=df["Date"], 
        y=df["NAV_EUR"],
        fill='tozeroy',
        mode='lines',
        line=dict(color='#00F0FF', width=3, shape='spline'),
        name="Portfolio NAV (EUR)"
    ))

    # The Tax Exemption Boundary
    fig.add_hline(
        y=BOX3_EXEMPTION_LIMIT_EUR,
        line_dash="dash",
        line_color="#FF3366",
        line_width=2,
        annotation_text=f"Tax-Free Limit: €{BOX3_EXEMPTION_LIMIT_EUR:,.0f}",
        annotation_position="top left",
        annotation_font=dict(color="#FF3366", size=14)
    )

    # Mark the January 1st Snapshots
    jan_dates = df[df["Is_Jan_1st"]]["Date"]
    for idx, jan_date in enumerate(jan_dates):
        nav_on_jan1 = df[df["Date"] == jan_date]["NAV_EUR"].iloc[0]
        tax_owed = calculate_box3_tax(nav_on_jan1)
        
        fig.add_vline(x=jan_date.timestamp() * 1000, line_width=1, line_color="rgba(255,255,255,0.3)")
        
        # Add a marker point on Jan 1st
        fig.add_trace(go.Scatter(
            x=[jan_date], y=[nav_on_jan1],
            mode='markers+text',
            marker=dict(color='yellow', size=10, symbol='diamond'),
            text=[f"Snapshot {idx+1}<br>Tax: €{tax_owed:,.0f}"],
            textposition="top right",
            name="Jan 1st Tax Snapshot"
        ))

    fig.update_layout(
        title="Household Cashflow vs Dutch Box 3 Tax Boundary (2026-2028)",
        template="plotly_dark",
        height=500,
        margin=dict(l=20, r=20, t=50, b=20),
        yaxis=dict(title="Net Asset Value (€)", tickformat="€,.0f"),
        xaxis=dict(title="")
    )

    st.plotly_chart(fig, use_container_width=True)

    # Metrics
    nav_jan_2027 = df[df["Date"] == "2027-01-01"]["NAV_EUR"].iloc[0] if len(df[df["Date"] == "2027-01-01"]) > 0 else 0
    tax_2027 = calculate_box3_tax(nav_jan_2027)

    c1, c2, c3 = st.columns(3)
    c1.metric("Initial Capital", f"€{starting_capital_eur:,.0f}")
    c2.metric("NAV on Jan 1, 2027", f"€{nav_jan_2027:,.0f}")
    c3.metric("Box 3 Tax Owed (2027)", f"€{tax_2027:,.0f}")