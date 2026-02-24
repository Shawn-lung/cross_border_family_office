"""Monte Carlo simulation helpers for long-horizon retirement planning."""

from __future__ import annotations

import numpy as np


def simulate_gbm_paths(
    initial_value: float,
    monthly_cash_flow: float,
    years: int = 30,
    n_paths: int = 10_000,
    annual_return: float = 0.07,
    annual_volatility: float = 0.15,
    random_seed: int | None = 42,
) -> np.ndarray:
    """Simulate monthly portfolio paths using Geometric Brownian Motion.

    Args:
        initial_value: Starting portfolio value.
        monthly_cash_flow: Net monthly cash flow added after monthly growth.
            Positive values represent contributions, negative values withdrawals.
        years: Simulation horizon in years.
        n_paths: Number of Monte Carlo paths.
        annual_return: Expected annual return (decimal form, e.g. 0.07).
        annual_volatility: Annualized volatility (decimal form, e.g. 0.15).
        random_seed: RNG seed for deterministic testing.

    Returns:
        Numpy array with shape (n_paths, years * 12). Each column is month-end value.
    """
    if initial_value < 0:
        raise ValueError("initial_value must be non-negative.")
    if years <= 0:
        raise ValueError("years must be positive.")
    if n_paths <= 0:
        raise ValueError("n_paths must be positive.")
    if annual_volatility < 0:
        raise ValueError("annual_volatility must be non-negative.")

    months = years * 12
    dt = 1.0 / 12.0
    drift = (annual_return - 0.5 * annual_volatility**2) * dt
    diffusion = annual_volatility * np.sqrt(dt)

    rng = np.random.default_rng(random_seed)
    shocks = rng.standard_normal((n_paths, months))

    paths = np.empty((n_paths, months), dtype=float)
    previous_values = np.full(n_paths, initial_value, dtype=float)

    for month in range(months):
        growth_factor = np.exp(drift + diffusion * shocks[:, month])
        next_values = previous_values * growth_factor + monthly_cash_flow
        # Prevent mathematically invalid negative balances after heavy withdrawals.
        next_values = np.maximum(next_values, 0.0)
        paths[:, month] = next_values
        previous_values = next_values

    return paths


def build_risk_free_benchmark(
    initial_value: float,
    monthly_cash_flow: float,
    years: int = 30,
    annual_rate: float = 0.035,
) -> np.ndarray:
    """Build deterministic month-end benchmark balances at a fixed annual rate."""
    if initial_value < 0:
        raise ValueError("initial_value must be non-negative.")
    if years <= 0:
        raise ValueError("years must be positive.")

    months = years * 12
    monthly_rate = annual_rate / 12.0
    balances = np.empty(months, dtype=float)
    value = float(initial_value)

    for month in range(months):
        value = value * (1.0 + monthly_rate) + monthly_cash_flow
        value = max(value, 0.0)
        balances[month] = value

    return balances

