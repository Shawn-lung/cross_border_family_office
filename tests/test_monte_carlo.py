"""Tests for Monte Carlo simulation utilities and dual-pool wealth planner engine."""

import numpy as np
import pytest

from app.utils.monte_carlo import run_wealth_planning_mcs, simulate_gbm_paths


def test_simulation_output_shape_is_10000_by_120_for_10_years():
    """10 years of monthly points must produce exactly 120 columns."""
    paths = simulate_gbm_paths(
        initial_value=1_000_000.0,
        monthly_cash_flow=5_000.0,
        years=10,
        n_paths=10_000,
        annual_return=0.07,
        annual_volatility=0.15,
        random_seed=123,
    )
    assert paths.shape == (10_000, 120)


def test_run_wealth_planning_mcs_output_shape_for_10000_paths_10y():
    """Dual-pool engine should output terminal wealth vector and monthly percentile frame."""
    mu = np.array([0.08, 0.10, 0.07], dtype=float)
    vol = np.array([0.18, 0.25, 0.16], dtype=float)
    corr = np.array(
        [
            [1.0, 0.7, 0.6],
            [0.7, 1.0, 0.65],
            [0.6, 0.65, 1.0],
        ],
        dtype=float,
    )
    sigma = np.outer(vol, vol) * corr
    weights = np.array([0.5, 0.3, 0.2], dtype=float)

    final_total_wealth, percentile_df = run_wealth_planning_mcs(
        mu=mu,
        sigma=sigma,
        target_weights=weights,
        taxable_initial=2_500_000.0,
        exempt_initial=5_200_000.0,
        monthly_contribution=30_000.0,
        years=10,
        num_paths=10_000,
        rebalance_friction_bps=10.0,
        box3_tax_rate=0.36,
        box3_allowance=133_200.0,
        house_purchase_price=0.0,
        house_initial_payment=0.0,
        random_seed=123,
    )

    assert final_total_wealth.shape == (10_000,)
    assert percentile_df.shape == (120, 4)
    assert list(percentile_df.columns) == ["month", "p10", "p50", "p90"]


def test_monthly_contribution_is_applied_to_taxable_pool_only():
    """With zero returns and no tax/friction, final wealth should add only taxable contributions."""
    mu = np.array([0.0, 0.0], dtype=float)
    sigma = np.zeros((2, 2), dtype=float)
    weights = np.array([0.6, 0.4], dtype=float)

    final_total_wealth, percentile_df = run_wealth_planning_mcs(
        mu=mu,
        sigma=sigma,
        target_weights=weights,
        taxable_initial=2_500_000.0,
        exempt_initial=5_200_000.0,
        monthly_contribution=30_000.0,
        years=1,
        num_paths=1_000,
        rebalance_friction_bps=0.0,
        box3_tax_rate=0.0,
        box3_allowance=0.0,
        house_purchase_price=0.0,
        house_initial_payment=0.0,
        random_seed=7,
    )

    expected_final = 2_500_000.0 + 5_200_000.0 + (30_000.0 * 12.0)
    assert float(np.median(final_total_wealth)) == pytest.approx(expected_final, rel=0, abs=1e-6)
    assert float(percentile_df.iloc[-1]["p50"]) == pytest.approx(expected_final, rel=0, abs=1e-6)


def test_box3_tax_applies_only_to_taxable_pool():
    """Tax should not change outcomes when taxable pool is zero and should reduce taxable outcomes."""
    mu = np.array([0.12, 0.10], dtype=float)
    vol = np.array([0.15, 0.20], dtype=float)
    corr = np.array([[1.0, 0.75], [0.75, 1.0]], dtype=float)
    sigma = np.outer(vol, vol) * corr
    weights = np.array([0.55, 0.45], dtype=float)

    exempt_only_no_tax, _ = run_wealth_planning_mcs(
        mu=mu,
        sigma=sigma,
        target_weights=weights,
        taxable_initial=0.0,
        exempt_initial=5_200_000.0,
        monthly_contribution=0.0,
        years=10,
        num_paths=2_000,
        rebalance_friction_bps=10.0,
        box3_tax_rate=0.0,
        box3_allowance=0.0,
        house_purchase_price=0.0,
        house_initial_payment=0.0,
        random_seed=42,
    )
    exempt_only_with_tax, _ = run_wealth_planning_mcs(
        mu=mu,
        sigma=sigma,
        target_weights=weights,
        taxable_initial=0.0,
        exempt_initial=5_200_000.0,
        monthly_contribution=0.0,
        years=10,
        num_paths=2_000,
        rebalance_friction_bps=10.0,
        box3_tax_rate=0.36,
        box3_allowance=0.0,
        house_purchase_price=0.0,
        house_initial_payment=0.0,
        random_seed=42,
    )
    assert float(np.median(exempt_only_with_tax)) == pytest.approx(float(np.median(exempt_only_no_tax)))

    taxable_only_no_tax, _ = run_wealth_planning_mcs(
        mu=mu,
        sigma=sigma,
        target_weights=weights,
        taxable_initial=2_500_000.0,
        exempt_initial=0.0,
        monthly_contribution=30_000.0,
        years=10,
        num_paths=2_000,
        rebalance_friction_bps=10.0,
        box3_tax_rate=0.0,
        box3_allowance=0.0,
        house_purchase_price=0.0,
        house_initial_payment=0.0,
        random_seed=42,
    )
    taxable_only_with_tax, _ = run_wealth_planning_mcs(
        mu=mu,
        sigma=sigma,
        target_weights=weights,
        taxable_initial=2_500_000.0,
        exempt_initial=0.0,
        monthly_contribution=30_000.0,
        years=10,
        num_paths=2_000,
        rebalance_friction_bps=10.0,
        box3_tax_rate=0.36,
        box3_allowance=0.0,
        house_purchase_price=0.0,
        house_initial_payment=0.0,
        random_seed=42,
    )
    assert float(np.median(taxable_only_with_tax)) < float(np.median(taxable_only_no_tax))


def test_house_model_purchase_and_appreciation_are_included_and_tax_exempt():
    """House sidecar should be added on purchase month, appreciate, and remain outside Box 3 tax."""
    mu = np.array([0.0, 0.0], dtype=float)
    sigma = np.zeros((2, 2), dtype=float)
    weights = np.array([0.5, 0.5], dtype=float)

    final_no_tax, _ = run_wealth_planning_mcs(
        mu=mu,
        sigma=sigma,
        target_weights=weights,
        taxable_initial=0.0,
        exempt_initial=0.0,
        monthly_contribution=0.0,
        years=3,
        num_paths=1_000,
        rebalance_friction_bps=0.0,
        box3_tax_rate=0.0,
        box3_allowance=0.0,
        house_purchase_price=550_000.0,
        house_initial_payment=550_000.0,
        house_appreciation_rate=0.05,
        mortgage_interest_rate=0.04,
        mortgage_term_years=20,
        house_purchase_year=2027,
        house_purchase_month=7,
        simulation_start_year=2026,
        simulation_start_month=1,
        random_seed=42,
    )
    final_with_tax, _ = run_wealth_planning_mcs(
        mu=mu,
        sigma=sigma,
        target_weights=weights,
        taxable_initial=0.0,
        exempt_initial=0.0,
        monthly_contribution=0.0,
        years=3,
        num_paths=1_000,
        rebalance_friction_bps=0.0,
        box3_tax_rate=0.36,
        box3_allowance=0.0,
        house_purchase_price=550_000.0,
        house_initial_payment=550_000.0,
        house_appreciation_rate=0.05,
        mortgage_interest_rate=0.04,
        mortgage_term_years=20,
        house_purchase_year=2027,
        house_purchase_month=7,
        simulation_start_year=2026,
        simulation_start_month=1,
        random_seed=42,
    )

    # Purchase month index from Jan-2026 to Jul-2027 is 18 (0-based).
    # Growth starts the month after purchase in current implementation.
    growth_months = 36 - 18 - 1
    expected = 550_000.0 * ((1.0 + 0.05) ** (1.0 / 12.0)) ** growth_months

    assert float(np.median(final_no_tax)) == pytest.approx(expected, rel=1e-10)
    # Box 3 tax should have no effect with zero taxable pool.
    assert float(np.median(final_with_tax)) == pytest.approx(float(np.median(final_no_tax)))


def test_box3_starts_in_2028_with_basis_step_up():
    """Tax should not apply before 2028 and should ignore pre-2028 gains via basis step-up."""
    mu = np.array([0.12], dtype=float)
    sigma = np.zeros((1, 1), dtype=float)
    weights = np.array([1.0], dtype=float)

    final_pre_2028, _ = run_wealth_planning_mcs(
        mu=mu,
        sigma=sigma,
        target_weights=weights,
        taxable_initial=2_500_000.0,
        exempt_initial=0.0,
        monthly_contribution=0.0,
        years=2,  # 2026-2027
        num_paths=100,
        rebalance_friction_bps=0.0,
        box3_tax_rate=0.36,
        box3_allowance=0.0,
        box3_start_year=2028,
        house_purchase_price=0.0,
        house_initial_payment=0.0,
        simulation_start_year=2026,
        simulation_start_month=1,
        random_seed=42,
    )
    final_pre_2028_no_tax, _ = run_wealth_planning_mcs(
        mu=mu,
        sigma=sigma,
        target_weights=weights,
        taxable_initial=2_500_000.0,
        exempt_initial=0.0,
        monthly_contribution=0.0,
        years=2,
        num_paths=100,
        rebalance_friction_bps=0.0,
        box3_tax_rate=0.0,
        box3_allowance=0.0,
        box3_start_year=2028,
        house_purchase_price=0.0,
        house_initial_payment=0.0,
        simulation_start_year=2026,
        simulation_start_month=1,
        random_seed=42,
    )
    assert float(np.median(final_pre_2028)) == pytest.approx(float(np.median(final_pre_2028_no_tax)))

    final_through_2028, _ = run_wealth_planning_mcs(
        mu=mu,
        sigma=sigma,
        target_weights=weights,
        taxable_initial=2_500_000.0,
        exempt_initial=0.0,
        monthly_contribution=0.0,
        years=3,  # includes 2028
        num_paths=100,
        rebalance_friction_bps=0.0,
        box3_tax_rate=0.36,
        box3_allowance=0.0,
        box3_start_year=2028,
        house_purchase_price=0.0,
        house_initial_payment=0.0,
        simulation_start_year=2026,
        simulation_start_month=1,
        random_seed=42,
    )
    final_through_2028_no_tax, _ = run_wealth_planning_mcs(
        mu=mu,
        sigma=sigma,
        target_weights=weights,
        taxable_initial=2_500_000.0,
        exempt_initial=0.0,
        monthly_contribution=0.0,
        years=3,
        num_paths=100,
        rebalance_friction_bps=0.0,
        box3_tax_rate=0.0,
        box3_allowance=0.0,
        box3_start_year=2028,
        house_purchase_price=0.0,
        house_initial_payment=0.0,
        simulation_start_year=2026,
        simulation_start_month=1,
        random_seed=42,
    )
    assert float(np.median(final_through_2028)) < float(np.median(final_through_2028_no_tax))


def test_house_mortgage_equity_tracks_purchase_price_minus_outstanding_loan():
    """With zero appreciation, terminal house equity should equal purchase price minus remaining loan."""
    mu = np.array([0.0], dtype=float)
    sigma = np.zeros((1, 1), dtype=float)
    weights = np.array([1.0], dtype=float)

    final_values, _ = run_wealth_planning_mcs(
        mu=mu,
        sigma=sigma,
        target_weights=weights,
        taxable_initial=0.0,
        exempt_initial=0.0,
        monthly_contribution=0.0,
        years=3,
        num_paths=100,
        rebalance_friction_bps=0.0,
        box3_tax_rate=0.0,
        box3_allowance=0.0,
        house_purchase_price=20_350_000.0,
        house_initial_payment=550_000.0,
        house_appreciation_rate=0.0,
        mortgage_interest_rate=0.04,
        mortgage_term_years=20,
        house_purchase_year=2027,
        house_purchase_month=7,
        simulation_start_year=2026,
        simulation_start_month=1,
        random_seed=42,
    )

    principal = 20_350_000.0 - 550_000.0
    r = 0.04 / 12.0
    n = 20 * 12
    payment = principal * r / (1.0 - (1.0 + r) ** (-n))
    # Purchase index 18 (0-based) in a 36-month run, so 17 payments occur.
    k = 17
    remaining = principal * (1.0 + r) ** k - payment * (((1.0 + r) ** k - 1.0) / r)
    expected_equity = 20_350_000.0 - remaining

    assert float(np.median(final_values)) == pytest.approx(expected_equity, rel=1e-10)
