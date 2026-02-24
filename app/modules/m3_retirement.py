"""Module 3 Monte Carlo retirement simulator with risk-free outperformance probability."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from app.utils.monte_carlo import build_risk_free_benchmark, simulate_gbm_paths

RISK_FREE_BENCHMARK_RATE = 0.035
N_PATHS = 10_000
CHART_PATH_SAMPLE = 250


def _sample_paths_for_chart(paths: np.ndarray, max_lines: int = CHART_PATH_SAMPLE) -> np.ndarray:
    """Sample a subset of simulated paths to keep chart rendering responsive."""
    n_paths = paths.shape[0]
    if n_paths <= max_lines:
        return paths
    rng = np.random.default_rng(7)
    indices = rng.choice(n_paths, size=max_lines, replace=False)
    return paths[indices]


def render() -> None:
    """Render Module 3 retirement simulation dashboard."""
    st.subheader("Module 3: Monte Carlo Retirement Simulator")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### Grandparents")
        gp_initial = st.number_input(
            "Grandparents Initial Savings (TWD)",
            min_value=0.0,
            value=3_000_000.0,
            step=50_000.0,
            format="%.0f",
        )
        gp_monthly = st.number_input(
            "Grandparents Monthly Addition (TWD)",
            value=20_000.0,
            step=1_000.0,
            format="%.0f",
        )
    with c2:
        st.markdown("### Parents")
        p_initial = st.number_input(
            "Parents Initial Savings (TWD)",
            min_value=0.0,
            value=2_000_000.0,
            step=50_000.0,
            format="%.0f",
        )
        p_monthly = st.number_input(
            "Parents Monthly Addition (TWD)",
            value=15_000.0,
            step=1_000.0,
            format="%.0f",
        )

    inflation_pct = st.slider(
        "Expected Inflation (%)",
        min_value=0.0,
        max_value=10.0,
        value=2.0,
        step=0.1,
    )
    sim_years = st.slider("Simulation Years", min_value=5, max_value=40, value=30, step=1)

    st.markdown("### Market Assumptions")
    annual_return_pct = st.slider("Expected Annual Return (%)", 0.0, 15.0, 7.0, 0.1)
    annual_volatility_pct = st.slider("Annual Volatility (%)", 0.0, 40.0, 15.0, 0.1)

    initial_total = float(gp_initial + p_initial)
    monthly_total = float(gp_monthly + p_monthly)

    paths = simulate_gbm_paths(
        initial_value=initial_total,
        monthly_cash_flow=monthly_total,
        years=sim_years,
        n_paths=N_PATHS,
        annual_return=annual_return_pct / 100.0,
        annual_volatility=annual_volatility_pct / 100.0,
        random_seed=42,
    )
    benchmark = build_risk_free_benchmark(
        initial_value=initial_total,
        monthly_cash_flow=monthly_total,
        years=sim_years,
        annual_rate=RISK_FREE_BENCHMARK_RATE,
    )

    months = sim_years * 12
    inflation_factor = (1.0 + inflation_pct / 100.0) ** sim_years
    ending_real = paths[:, -1] / inflation_factor
    benchmark_ending_real = benchmark[-1] / inflation_factor
    outperformance_probability = float(np.mean(ending_real > benchmark_ending_real) * 100.0)

    st.metric(
        "Probability of Outperformance",
        f"{outperformance_probability:.2f}%",
        delta=f"vs {RISK_FREE_BENCHMARK_RATE * 100:.1f}% risk-free baseline",
    )

    sampled = _sample_paths_for_chart(paths, max_lines=CHART_PATH_SAMPLE)
    sampled_df = pd.DataFrame(sampled.T)
    sampled_df["Month"] = np.arange(1, months + 1)
    long_df = sampled_df.melt(id_vars="Month", var_name="Path", value_name="Portfolio Value (TWD)")

    fig = px.line(
        long_df,
        x="Month",
        y="Portfolio Value (TWD)",
        color="Path",
        title="Monte Carlo Fan Chart (Sampled Paths)",
    )
    fig.update_traces(line=dict(width=1), opacity=0.08)
    fig.update_layout(showlegend=False)
    fig.add_scatter(
        x=np.arange(1, months + 1),
        y=benchmark,
        mode="lines",
        name="3.5% Risk-Free Benchmark",
        line=dict(color="black", width=3),
        opacity=0.9,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        f"Simulation uses {N_PATHS:,} total paths over {sim_years} years with monthly compounding and net monthly cash flow."
    )
