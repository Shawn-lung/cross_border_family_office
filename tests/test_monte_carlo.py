"""Tests for Monte Carlo GBM simulation output shape and baseline sanity."""

from app.utils.monte_carlo import simulate_gbm_paths


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
