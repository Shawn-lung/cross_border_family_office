"""Monte Carlo simulation helpers for long-horizon retirement planning."""

from __future__ import annotations

import numpy as np
import pandas as pd


def run_wealth_planning_mcs(
    mu: np.ndarray | list[float],
    sigma: np.ndarray | list[list[float]],
    target_weights: np.ndarray | list[float],
    taxable_initial: float = 2_500_000.0,
    exempt_initial: float = 5_200_000.0,
    monthly_contribution: float = 30_000.0,
    years: int = 15,
    num_paths: int = 10_000,
    rebalance_friction_bps: float = 10.0,
    box3_tax_rate: float = 0.36,
    box3_allowance: float = 133_200.0,
    box3_start_year: int = 2028,
    house_purchase_price: float = 20_350_000.0,
    house_initial_payment: float = 550_000.0,
    house_appreciation_rate: float = 0.05,
    mortgage_interest_rate: float = 0.04,
    mortgage_term_years: int = 20,
    house_purchase_year: int = 2027,
    house_purchase_month: int = 7,
    simulation_start_year: int = 2026,
    simulation_start_month: int = 1,
    random_seed: int | None = 42,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Run a fully vectorized dual-pool Monte Carlo wealth simulation.

    Pool definitions:
    - Taxable pool: receives monthly contributions and pays annual Box 3 tax.
    - Exempt pool: compounds identically but is not taxed.
    - House sidecar: follows user-defined appreciation and mortgage amortization,
      excluded from Box 3 tax by design.

    Both pools apply Swedroe 5/25 rebalancing independently.

    Note:
    - Mortgage payments are treated as out-of-model cash flows and are not deducted
      from the simulated investment pools. Adjust `monthly_contribution` manually
      if you want to reflect mortgage cash drag in portfolio contributions.
    """
    mu_vec = np.asarray(mu, dtype=float).reshape(-1)
    sigma_mat = np.asarray(sigma, dtype=float)
    weights = np.asarray(target_weights, dtype=float).reshape(-1)

    if mu_vec.ndim != 1 or mu_vec.size == 0:
        raise ValueError("mu must be a non-empty 1D vector.")
    if sigma_mat.ndim != 2 or sigma_mat.shape[0] != sigma_mat.shape[1]:
        raise ValueError("sigma must be a square 2D covariance matrix.")
    if sigma_mat.shape[0] != mu_vec.size:
        raise ValueError("sigma dimension must match mu length.")
    if weights.size != mu_vec.size:
        raise ValueError("target_weights length must match mu length.")
    if taxable_initial < 0:
        raise ValueError("taxable_initial must be non-negative.")
    if exempt_initial < 0:
        raise ValueError("exempt_initial must be non-negative.")
    if years <= 0:
        raise ValueError("years must be positive.")
    if num_paths <= 0:
        raise ValueError("num_paths must be positive.")
    if monthly_contribution < 0:
        raise ValueError("monthly_contribution must be non-negative.")
    if rebalance_friction_bps < 0:
        raise ValueError("rebalance_friction_bps must be non-negative.")
    if not 0.0 <= box3_tax_rate <= 1.0:
        raise ValueError("box3_tax_rate must be between 0 and 1.")
    if box3_allowance < 0:
        raise ValueError("box3_allowance must be non-negative.")
    if box3_start_year < 1900:
        raise ValueError("box3_start_year must be a valid calendar year.")
    if house_purchase_price < 0:
        raise ValueError("house_purchase_price must be non-negative.")
    if house_initial_payment < 0:
        raise ValueError("house_initial_payment must be non-negative.")
    if house_initial_payment > house_purchase_price:
        raise ValueError("house_initial_payment cannot exceed house_purchase_price.")
    if house_appreciation_rate < -1.0:
        raise ValueError("house_appreciation_rate must be greater than -100%.")
    if mortgage_interest_rate < 0:
        raise ValueError("mortgage_interest_rate must be non-negative.")
    if mortgage_term_years <= 0:
        raise ValueError("mortgage_term_years must be positive.")
    if not 1 <= house_purchase_month <= 12:
        raise ValueError("house_purchase_month must be in 1..12.")
    if not 1 <= simulation_start_month <= 12:
        raise ValueError("simulation_start_month must be in 1..12.")

    weight_sum = float(weights.sum())
    if weight_sum <= 0:
        raise ValueError("target_weights must sum to a positive value.")
    weights = weights / weight_sum

    months = int(years * 12)
    dt = 1.0 / 12.0
    sqrt_dt = np.sqrt(dt)
    house_monthly_growth = (1.0 + house_appreciation_rate) ** (1.0 / 12.0)
    mortgage_monthly_rate = mortgage_interest_rate / 12.0
    mortgage_total_months = mortgage_term_years * 12
    house_purchase_idx = (
        (house_purchase_year - simulation_start_year) * 12
        + (house_purchase_month - simulation_start_month)
    )

    mortgage_principal = max(0.0, house_purchase_price - house_initial_payment)
    if mortgage_principal > 0:
        if mortgage_monthly_rate > 0:
            mortgage_payment = mortgage_principal * mortgage_monthly_rate / (
                1.0 - (1.0 + mortgage_monthly_rate) ** (-mortgage_total_months)
            )
        else:
            mortgage_payment = mortgage_principal / mortgage_total_months
    else:
        mortgage_payment = 0.0

    if np.allclose(sigma_mat, 0.0):
        chol = np.zeros_like(sigma_mat)
    else:
        try:
            chol = np.linalg.cholesky(sigma_mat)
        except np.linalg.LinAlgError:
            # Small jitter guards against near-semi-definite covariance input.
            jitter = np.eye(mu_vec.size) * 1e-12
            chol = np.linalg.cholesky(sigma_mat + jitter)
    drift = (mu_vec - 0.5 * np.diag(sigma_mat)) * dt

    rng = np.random.default_rng(random_seed)

    taxable_assets = np.tile(taxable_initial * weights, (num_paths, 1)).astype(float)
    exempt_assets = np.tile(exempt_initial * weights, (num_paths, 1)).astype(float)
    house_values = np.zeros(num_paths, dtype=float)
    outstanding_mortgage = 0.0
    mortgage_payments_done = 0
    taxable_adjusted_basis = np.full(num_paths, taxable_initial, dtype=float)
    box3_basis_stepped_up = simulation_start_year >= box3_start_year
    monthly_contrib_alloc = monthly_contribution * weights
    rebalance_friction_rate = rebalance_friction_bps / 10_000.0

    p10 = np.empty(months, dtype=float)
    p50 = np.empty(months, dtype=float)
    p90 = np.empty(months, dtype=float)

    for month_idx in range(months):
        shocks = rng.standard_normal((num_paths, mu_vec.size))
        monthly_returns = np.exp(drift + (shocks @ chol.T) * sqrt_dt) - 1.0
        growth = 1.0 + monthly_returns

        taxable_assets *= growth
        exempt_assets *= growth

        taxable_assets += monthly_contrib_alloc
        taxable_adjusted_basis += monthly_contribution

        taxable_assets = np.maximum(taxable_assets, 0.0)
        exempt_assets = np.maximum(exempt_assets, 0.0)

        taxable_assets = _apply_swedroe_rebalance(
            taxable_assets,
            target_weights=weights,
            friction_rate=rebalance_friction_rate,
        )
        exempt_assets = _apply_swedroe_rebalance(
            exempt_assets,
            target_weights=weights,
            friction_rate=rebalance_friction_rate,
        )

        current_month = month_idx + 1
        calendar_month_index = (simulation_start_month - 1) + month_idx
        calendar_year = simulation_start_year + (calendar_month_index // 12)
        calendar_month = (calendar_month_index % 12) + 1

        # Step-up basis at the start of the first taxable year so pre-2028 gains are not taxed.
        if (not box3_basis_stepped_up) and calendar_year >= box3_start_year and calendar_month == 1:
            taxable_adjusted_basis = taxable_assets.sum(axis=1)
            box3_basis_stepped_up = True

        if calendar_year >= box3_start_year and calendar_month == 12:
            taxable_value = taxable_assets.sum(axis=1)
            yearly_gains = taxable_value - taxable_adjusted_basis
            taxable_gains = np.maximum(0.0, yearly_gains - box3_allowance)
            tax_amount = taxable_gains * box3_tax_rate

            taxed = (tax_amount > 0) & (taxable_value > 0)
            if np.any(taxed):
                tax_scale = np.divide(
                    taxable_value[taxed] - tax_amount[taxed],
                    taxable_value[taxed],
                    out=np.zeros_like(taxable_value[taxed]),
                    where=taxable_value[taxed] > 0,
                )
                taxable_assets[taxed] *= tax_scale[:, None]

            taxable_assets = np.maximum(taxable_assets, 0.0)
            taxable_adjusted_basis = taxable_assets.sum(axis=1)

        if house_purchase_price > 0:
            if month_idx == house_purchase_idx:
                house_values[:] = house_purchase_price
                outstanding_mortgage = mortgage_principal
                mortgage_payments_done = 0
            elif month_idx > house_purchase_idx and np.any(house_values > 0):
                house_values *= house_monthly_growth
                if outstanding_mortgage > 0 and mortgage_payments_done < mortgage_total_months:
                    interest_due = outstanding_mortgage * mortgage_monthly_rate
                    principal_paid = max(0.0, mortgage_payment - interest_due)
                    outstanding_mortgage = max(0.0, outstanding_mortgage - principal_paid)
                    mortgage_payments_done += 1

        net_house_equity = np.maximum(0.0, house_values - outstanding_mortgage)
        total_wealth = taxable_assets.sum(axis=1) + exempt_assets.sum(axis=1) + net_house_equity
        p10[month_idx], p50[month_idx], p90[month_idx] = np.quantile(
            total_wealth,
            [0.10, 0.50, 0.90],
        )

    percentile_df = pd.DataFrame(
        {
            "month": np.arange(1, months + 1, dtype=int),
            "p10": p10,
            "p50": p50,
            "p90": p90,
        }
    )
    net_house_equity = np.maximum(0.0, house_values - outstanding_mortgage)
    final_total_wealth = taxable_assets.sum(axis=1) + exempt_assets.sum(axis=1) + net_house_equity
    return final_total_wealth, percentile_df


def _apply_swedroe_rebalance(
    asset_values: np.ndarray,
    *,
    target_weights: np.ndarray,
    friction_rate: float,
) -> np.ndarray:
    """Apply Swedroe 5/25 rebalance in-place for rows that breach drift thresholds."""
    pool_values = asset_values.sum(axis=1)
    valid_rows = pool_values > 0
    if not np.any(valid_rows):
        return asset_values

    current_weights = np.divide(
        asset_values,
        pool_values[:, None],
        out=np.zeros_like(asset_values),
        where=pool_values[:, None] > 0,
    )
    abs_drift = np.abs(current_weights - target_weights)
    rel_drift = np.divide(
        abs_drift,
        target_weights,
        out=np.zeros_like(abs_drift),
        where=target_weights > 0,
    )
    needs_rebalance = valid_rows & np.any((abs_drift >= 0.05) | (rel_drift >= 0.25), axis=1)
    if not np.any(needs_rebalance):
        return asset_values

    totals_before = pool_values[needs_rebalance]
    totals_after = np.maximum(totals_before * (1.0 - friction_rate), 0.0)
    asset_values[needs_rebalance] = totals_after[:, None] * target_weights
    return asset_values


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
