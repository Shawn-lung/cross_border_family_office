"""Module 3 dual-pool goal-based wealth planner."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path

from app.utils.monte_carlo import run_wealth_planning_mcs

N_PATHS = 10_000
ASSET_NAMES = [
    "VT Proxy",
    "QQQ Proxy",
    "PRF Proxy",
    "IJS Proxy",
    "BRK.B",
    "Synthetic ZPRX",
]

# Strategic baseline assumptions (annualized)
DEFAULT_TARGET_WEIGHTS = np.array([0.30, 0.20, 0.15, 0.15, 0.10, 0.10], dtype=float)
DEFAULT_MU = np.array([0.075, 0.110, 0.085, 0.090, 0.100, 0.085], dtype=float)
DEFAULT_VOL = np.array([0.170, 0.280, 0.190, 0.240, 0.200, 0.230], dtype=float)
DEFAULT_CORR = np.array(
    [
        [1.00, 0.80, 0.78, 0.75, 0.65, 0.72],
        [0.80, 1.00, 0.70, 0.72, 0.60, 0.68],
        [0.78, 0.70, 1.00, 0.66, 0.58, 0.64],
        [0.75, 0.72, 0.66, 1.00, 0.60, 0.70],
        [0.65, 0.60, 0.58, 0.60, 1.00, 0.55],
        [0.72, 0.68, 0.64, 0.70, 0.55, 1.00],
    ],
    dtype=float,
)
DEFAULT_SIGMA = np.outer(DEFAULT_VOL, DEFAULT_VOL) * DEFAULT_CORR
MODULE4_BUNDLE_PATH = Path("data") / "wrds_bundle" / "module4_returns_20y_v3.pkl"


def _resolve_mu_sigma_from_module4_or_default() -> tuple[np.ndarray, np.ndarray, str]:
    """Load annualized mu/sigma from Module 4 bundle when available, otherwise fallback."""
    if MODULE4_BUNDLE_PATH.exists():
        try:
            returns_df = pd.read_pickle(MODULE4_BUNDLE_PATH).astype(float)
            if returns_df.shape[1] == len(DEFAULT_TARGET_WEIGHTS) and len(returns_df) >= 24:
                mu = (returns_df.mean() * 12.0).to_numpy(dtype=float)
                sigma = (returns_df.cov() * 12.0).to_numpy(dtype=float)
                return mu, sigma, f"Module 4 bundle ({MODULE4_BUNDLE_PATH.name})"
        except Exception:
            pass
    return DEFAULT_MU, DEFAULT_SIGMA, "Module 3 baseline assumptions"


def render() -> None:
    """Render Module 3 wealth planner UI."""
    st.subheader("Module 3: Goal-Based Monte Carlo Wealth Planner")

    with st.expander("Investment Pools (TWD)", expanded=True):
        c1, c2, c3 = st.columns(3)
        taxable_initial = c1.number_input(
            "NL Taxable Initial Capital (TWD)",
            min_value=0.0,
            value=2_500_000.0,
            step=100_000.0,
            format="%.0f",
        )
        exempt_initial = c2.number_input(
            "Taiwan Tax-Exempt Initial Capital (TWD)",
            min_value=0.0,
            value=5_200_000.0,
            step=100_000.0,
            format="%.0f",
        )
        monthly_contribution = c3.number_input(
            "Monthly Contribution (to NL pool, TWD)",
            value=30_000.0,
            step=1_000.0,
            format="%.0f",
        )

    with st.expander("FIRE Target & Horizon", expanded=True):
        c3, c4 = st.columns(2)
        target_wealth = c3.number_input(
            "Target Wealth / FIRE Number (TWD)",
            min_value=0.0,
            value=30_000_000.0,
            step=100_000.0,
            format="%.0f",
        )
        years = c4.slider("Investment Horizon (Years)", min_value=5, max_value=40, value=15, step=1)

    with st.expander("NL Box 3 Tax (2028+)", expanded=True):
        t1, t2 = st.columns(2)
        box3_tax_rate = t1.slider("Tax Rate (%)", min_value=0.0, max_value=50.0, value=36.0, step=0.5)
        box3_allowance = t2.number_input(
            "Tax-free Allowance (TWD)",
            min_value=0.0,
            value=133_200.0,
            step=100_000.0,
            format="%.0f",
        )

    with st.expander("House Model (Box 3 Exempt)", expanded=True):
        h1, h2, h3 = st.columns(3)
        house_purchase_price = h1.number_input(
            "House Purchase Price (TWD)",
            min_value=0.0,
            value=20_350_000.0,
            step=100_000.0,
            format="%.0f",
        )
        house_initial_payment = h2.number_input(
            "House Downpayment / Initial Payment (TWD)",
            min_value=0.0,
            value=550_000.0,
            step=50_000.0,
            format="%.0f",
        )
        house_appreciation_rate = h3.slider(
            "House Appreciation Rate (%)",
            min_value=0.0,
            max_value=20.0,
            value=5.0,
            step=0.1,
        )
        h4, h5, h6 = st.columns(3)
        mortgage_interest_rate = h4.slider(
            "Mortgage Rate (%)",
            min_value=0.0,
            max_value=12.0,
            value=4.0,
            step=0.1,
        )
        mortgage_term_years = h5.slider(
            "Mortgage Term (Years)",
            min_value=1,
            max_value=40,
            value=20,
            step=1,
        )
        house_purchase_year = h6.number_input(
            "House Purchase Year",
            min_value=2000,
            max_value=2100,
            value=2027,
            step=1,
            format="%d",
        )
        h7, h8 = st.columns(2)
        house_purchase_month = h7.selectbox(
            "House Purchase Month",
            options=list(range(1, 13)),
            index=6,
            format_func=lambda m: f"{m:02d}",
        )
        simulation_start_year = h8.number_input(
            "Simulation Start Year",
            min_value=2000,
            max_value=2100,
            value=2026,
            step=1,
            format="%d",
        )

    with st.expander("Friction Parameters", expanded=True):
        rebalance_friction_bps = st.slider(
            "IBKR Rebalance Friction (bps)",
            min_value=0.0,
            max_value=100.0,
            value=10.0,
            step=0.5,
        )

    mu_vec, sigma_mat, source_label = _resolve_mu_sigma_from_module4_or_default()
    vol_vec = np.sqrt(np.diag(sigma_mat))
    weighted_mu = float(np.dot(DEFAULT_TARGET_WEIGHTS, mu_vec))
    weighted_vol = float(np.sqrt(DEFAULT_TARGET_WEIGHTS @ sigma_mat @ DEFAULT_TARGET_WEIGHTS))

    assumptions_df = pd.DataFrame(
        {
            "Asset": ASSET_NAMES,
            "Target Weight (%)": DEFAULT_TARGET_WEIGHTS * 100.0,
            "Expected Return (%)": mu_vec * 100.0,
            "Volatility (%)": vol_vec * 100.0,
        }
    )
    st.caption(
        "Strategic assumptions used for the Monte Carlo engine."
        f" Source: {source_label}. Portfolio Mu: {weighted_mu * 100:.2f}%, "
        f"Portfolio Vol: {weighted_vol * 100:.2f}%."
    )
    st.dataframe(assumptions_df, use_container_width=True, hide_index=True)

    if house_initial_payment > house_purchase_price:
        st.error("House downpayment cannot exceed house purchase price.")
        return

    final_values, percentile_df = run_wealth_planning_mcs(
        mu=mu_vec,
        sigma=sigma_mat,
        target_weights=DEFAULT_TARGET_WEIGHTS,
        taxable_initial=float(taxable_initial),
        exempt_initial=float(exempt_initial),
        monthly_contribution=float(monthly_contribution),
        years=int(years),
        num_paths=N_PATHS,
        rebalance_friction_bps=float(rebalance_friction_bps),
        box3_tax_rate=float(box3_tax_rate) / 100.0,
        box3_allowance=float(box3_allowance),
        house_purchase_price=float(house_purchase_price),
        house_initial_payment=float(house_initial_payment),
        house_appreciation_rate=float(house_appreciation_rate) / 100.0,
        mortgage_interest_rate=float(mortgage_interest_rate) / 100.0,
        mortgage_term_years=int(mortgage_term_years),
        house_purchase_year=int(house_purchase_year),
        house_purchase_month=int(house_purchase_month),
        simulation_start_year=int(simulation_start_year),
        simulation_start_month=1,
        random_seed=42,
    )

    probability_of_success = float(np.mean(final_values >= float(target_wealth)) * 100.0)
    median_final_wealth = float(np.median(final_values))

    m1, m2 = st.columns(2)
    m1.metric("Probability of Success", f"{probability_of_success:.2f}%")
    m2.metric("Median Final Wealth", f"{median_final_wealth:,.0f} TWD")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=percentile_df["month"],
            y=percentile_df["p10"],
            mode="lines",
            name="P10",
            line=dict(color="#8ecae6", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=percentile_df["month"],
            y=percentile_df["p50"],
            mode="lines",
            name="Median (P50)",
            line=dict(color="#219ebc", width=3),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=percentile_df["month"],
            y=percentile_df["p90"],
            mode="lines",
            name="P90",
            line=dict(color="#023047", width=2),
        )
    )
    fig.add_hline(
        y=float(target_wealth),
        line=dict(color="red", dash="dash", width=2),
        annotation_text="Target Wealth",
        annotation_position="top left",
    )
    fig.update_layout(
        title="Total Wealth Path Percentiles (P10 / P50 / P90)",
        xaxis_title="Month",
        yaxis_title="Total Wealth (TWD)",
        legend_title="Series",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        f"Simulation: {N_PATHS:,} paths, dual-pool monthly GBM dynamics, independent Swedroe 5/25 "
        "rebalancing per pool, annual Box 3 tax on the NL taxable pool only, plus a Box 3-exempt house "
        "sidecar with appreciation and mortgage amortization (payments are treated outside the portfolio model)."
    )
