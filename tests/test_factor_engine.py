"""Tests for Module 4 factor construction and portfolio matrix math."""

import numpy as np
import pandas as pd
import pytest

from app.utils.factor_engine import (
    build_brkb_returns_from_secm,
    build_named_returns_from_crsp,
    build_named_returns_from_secm,
    build_synthetic_factor_return_series,
    build_wrds_approx_proxy_returns,
    calculate_annualized_mu_sigma,
    calculate_portfolio_metrics,
    compute_msci_fundamental_weights,
    normalize_weights,
    select_small_value_universe,
)


def test_select_small_value_universe_keeps_bottom_me_and_top_bm():
    """Double sort should keep the small-cap sleeve, then the value sleeve within it."""
    df = pd.DataFrame(
        {
            "date": ["2024-01-31"] * 10,
            "gvkey": [f"{i:03d}" for i in range(10)],
            "me": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "bm": [0.1, 0.2, 0.9, 0.05, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12],
        }
    )

    selected = select_small_value_universe(df, date_col="date", me_col="me", bm_col="bm")
    assert len(selected) == 1
    assert selected.iloc[0]["gvkey"] == "002"


def test_msci_fundamental_weights_clip_negative_flows_and_sum_to_one():
    """Negative flows should be treated as zero before fundamental ratio weighting."""
    df = pd.DataFrame(
        {
            "date": ["2024-01-31", "2024-01-31"],
            "gvkey": ["A", "B"],
            "sale": [100.0, 100.0],
            "earnings": [100.0, -100.0],
            "cash_earnings": [100.0, -20.0],
            "book_value": [100.0, 300.0],
        }
    )
    weighted = compute_msci_fundamental_weights(
        df,
        date_col="date",
        sales_col="sale",
        earnings_col="earnings",
        cash_earnings_col="cash_earnings",
        book_value_col="book_value",
    )

    weight_a = float(weighted.loc[weighted["gvkey"] == "A", "weight"].iloc[0])
    weight_b = float(weighted.loc[weighted["gvkey"] == "B", "weight"].iloc[0])

    assert weight_a == pytest.approx(0.6875)
    assert weight_b == pytest.approx(0.3125)
    assert weighted["weight"].sum() == pytest.approx(1.0)


def test_build_synthetic_factor_return_series_aggregates_weighted_monthly_returns():
    """End-to-end build should rebalance in May and carry those weights into June."""
    merged = pd.DataFrame(
        {
            "date": [
                "2024-05-31",
                "2024-05-31",
                "2024-05-31",
                "2024-06-30",
                "2024-06-30",
                "2024-06-30",
            ],
            "gvkey": ["A", "B", "C", "A", "B", "C"],
            "price": [10.0, 10.0, 10.0, 10.5, 10.2, 9.9],
            "shares": [10.0, 20.0, 30.0, 10.0, 20.0, 30.0],
            "book_value": [200.0, 100.0, 90.0, 205.0, 98.0, 88.0],
            "sale": [100.0, 80.0, 70.0, 102.0, 81.0, 69.0],
            "earnings": [20.0, 10.0, 8.0, 21.0, 9.5, 7.5],
            "cash_earnings": [22.0, 11.0, 9.0, 23.0, 10.5, 8.5],
            "ret": [0.05, 0.02, -0.01, 0.03, 0.015, -0.005],
        }
    )

    series = build_synthetic_factor_return_series(merged, output_name="Synthetic_ZPRX")
    assert isinstance(series, pd.Series)
    assert series.name == "Synthetic_ZPRX"
    assert len(series) == 2
    assert set(pd.to_datetime(series.index).strftime("%Y-%m-%d")) == {"2024-05-31", "2024-06-30"}
    assert series.loc[pd.Timestamp("2024-05-31")] == pytest.approx(0.05)
    assert series.loc[pd.Timestamp("2024-06-30")] == pytest.approx(0.03)


def test_portfolio_matrix_math_matches_dot_product_formulas():
    """Expected return and risk should match w^T*mu and sqrt(w^T*Sigma*w)."""
    returns = pd.DataFrame(
        {
            "VWCE": [0.01, 0.02, 0.00, 0.03],
            "SXRV": [0.02, -0.01, 0.01, 0.04],
            "Synthetic_ZPRX": [0.03, 0.01, -0.02, 0.05],
        }
    )
    mu, sigma = calculate_annualized_mu_sigma(returns)
    weights = normalize_weights({"VWCE": 0.4, "SXRV": 0.3, "Synthetic_ZPRX": 0.3})

    metrics = calculate_portfolio_metrics(mu, sigma, weights, risk_free_rate=0.035)

    w = weights.reindex(mu.index).to_numpy()
    mu_values = mu.to_numpy()
    sigma_values = sigma.reindex(index=mu.index, columns=mu.index).to_numpy()
    expected = float(np.dot(w, mu_values))
    volatility = float(np.sqrt(np.dot(w, np.dot(sigma_values, w))))
    sharpe = (expected - 0.035) / volatility

    assert metrics["expected_return"] == pytest.approx(expected)
    assert metrics["volatility"] == pytest.approx(volatility)
    assert metrics["sharpe_ratio"] == pytest.approx(sharpe)


def test_build_wrds_approx_proxy_returns_outputs_expected_columns():
    """WRDS approximate proxy builder should output VWCE/SXRV/JPGL monthly returns."""
    df = pd.DataFrame(
        {
            "date": ["2024-01-31"] * 6 + ["2024-02-29"] * 6,
            "gvkey": [f"G{i}" for i in range(12)],
            "price": [10, 12, 14, 16, 18, 20, 11, 13, 15, 17, 19, 21],
            "shares": [100, 110, 120, 130, 140, 150, 100, 110, 120, 130, 140, 150],
            "ret": [0.01, 0.02, -0.01, 0.03, 0.00, 0.015, 0.005, 0.01, -0.02, 0.025, 0.01, 0.0],
            "fic_pricing": ["USA", "USA", "DEU", "FRA", "USA", "ITA", "USA", "USA", "DEU", "FRA", "USA", "ITA"],
            "sale": [100] * 12,
            "earnings": [10] * 12,
            "cash_earnings": [12] * 12,
            "book_value": [80] * 12,
        }
    )

    out = build_wrds_approx_proxy_returns(df)
    assert list(out.columns) == ["VWCE", "SXRV", "JPGL"]
    assert len(out) == 2
    assert out.notna().all().all()


def test_build_brkb_returns_from_secm_prefers_brk_dot_b():
    """BRKB series should be derived from preferred BRK.B ticker when present."""
    secm = pd.DataFrame(
        {
            "date": ["2024-01-31", "2024-02-29", "2024-03-31", "2024-01-31", "2024-02-29"],
            "tic": ["BRK.B", "BRK.B", "BRK.B", "BRKB", "BRKB"],
            "price": [100.0, 110.0, 121.0, 50.0, 55.0],
        }
    )
    series = build_brkb_returns_from_secm(secm)
    assert series.name == "BRKB"
    assert len(series) == 2
    assert series.iloc[0] == pytest.approx(0.10)
    assert series.iloc[1] == pytest.approx(0.10)


def test_build_named_returns_from_secm_uses_mapping_priority():
    """Generic secm builder should pick first available ticker per asset."""
    secm = pd.DataFrame(
        {
            "date": [
                "2024-01-31",
                "2024-02-29",
                "2024-03-31",
                "2024-01-31",
                "2024-02-29",
                "2024-03-31",
            ],
            "tic": ["VT", "VT", "VT", "QQQ", "QQQ", "QQQ"],
            "price": [100.0, 110.0, 121.0, 200.0, 210.0, 220.5],
        }
    )
    mapping = {"VWCE": ["VT"], "SXRV": ["QQQ"]}
    out = build_named_returns_from_secm(secm, mapping=mapping)

    assert list(out.columns) == ["VWCE", "SXRV"]
    assert len(out) == 2
    assert out["VWCE"].iloc[0] == pytest.approx(0.10)
    assert out["VWCE"].iloc[1] == pytest.approx(0.10)


def test_build_named_returns_from_crsp_uses_mapping_priority():
    """CRSP builder should map sleeves to their preferred ticker with monthly returns."""
    crsp = pd.DataFrame(
        {
            "date": [
                "2024-01-31",
                "2024-02-29",
                "2024-03-31",
                "2024-01-31",
                "2024-02-29",
                "2024-03-31",
                "2024-01-31",
                "2024-02-29",
                "2024-03-31",
            ],
            "ticker": ["VT", "VT", "VT", "PRF", "PRF", "PRF", "IJS", "IJS", "IJS"],
            "ret": [0.01, 0.02, 0.03, 0.005, 0.006, 0.007, -0.01, 0.012, 0.004],
        }
    )
    mapping = {
        "VT (Proxy for VWCE - Spliced with SPY pre-2008)": ["VT"],
        "PRF (Proxy for JPGL)": ["PRF"],
        "IJS (Proxy for ZPRV)": ["IJS"],
    }
    out = build_named_returns_from_crsp(crsp, mapping=mapping)

    assert list(out.columns) == list(mapping.keys())
    assert len(out) == 3
    assert out.loc[
        pd.Timestamp("2024-02-29"), "VT (Proxy for VWCE - Spliced with SPY pre-2008)"
    ] == pytest.approx(0.02)
    assert out.loc[pd.Timestamp("2024-03-31"), "IJS (Proxy for ZPRV)"] == pytest.approx(0.004)


def test_build_named_returns_from_crsp_deduplicates_same_month_share_classes():
    """Duplicate same-date rows must not be compounded together."""
    crsp = pd.DataFrame(
        {
            "date": ["2024-01-31", "2024-01-31", "2024-02-29"],
            "ticker": ["BRK.B", "BRK.B", "BRK.B"],
            "permno": [14541, 83443, 83443],
            "ret": [0.10, 0.20, 0.05],
        }
    )
    mapping = {"BRK.B": ["BRK.B"]}
    out = build_named_returns_from_crsp(crsp, mapping=mapping)

    # Keep last row per date after sorting (permno 83443 on 2024-01-31), not compounded 1.1*1.2-1.
    assert out.loc[pd.Timestamp("2024-01-31"), "BRK.B"] == pytest.approx(0.20)
    assert out.loc[pd.Timestamp("2024-02-29"), "BRK.B"] == pytest.approx(0.05)


def test_build_named_returns_from_crsp_splices_vt_with_spy_pre_launch():
    """VT sleeve should be backfilled with SPY when VT is unavailable."""
    crsp = pd.DataFrame(
        {
            "date": [
                "2007-12-31",
                "2008-01-31",
                "2008-02-29",
                "2008-01-31",
                "2008-02-29",
            ],
            "ticker": ["SPY", "SPY", "SPY", "VT", "VT"],
            "permno": [84398, 84398, 84398, 12345, 12345],
            "ret": [0.01, -0.02, 0.03, 0.04, 0.05],
        }
    )
    mapping = {"VT (Proxy for VWCE - Spliced with SPY pre-2008)": ["VT", "SPY"]}
    out = build_named_returns_from_crsp(crsp, mapping=mapping)

    col = "VT (Proxy for VWCE - Spliced with SPY pre-2008)"
    assert list(out.columns) == [col]
    assert pd.Timestamp("2007-12-31") in out.index
    assert out.loc[pd.Timestamp("2007-12-31"), col] == pytest.approx(0.01)
    # VT takes precedence where available.
    assert out.loc[pd.Timestamp("2008-01-31"), col] == pytest.approx(0.04)


def test_build_named_returns_from_crsp_qqq_falls_back_to_qqqq():
    """QQQ sleeve should use QQQQ history before QQQ rows exist."""
    crsp = pd.DataFrame(
        {
            "date": ["2010-12-31", "2011-01-31", "2011-02-28", "2011-02-28"],
            "ticker": ["QQQQ", "QQQQ", "QQQQ", "QQQ"],
            "permno": [86755, 86755, 86755, 86755],
            "ret": [0.01, 0.02, 0.03, 0.04],
        }
    )
    mapping = {"QQQ (Proxy for SXRV)": ["QQQ", "QQQQ"]}
    out = build_named_returns_from_crsp(crsp, mapping=mapping)

    col = "QQQ (Proxy for SXRV)"
    assert out.loc[pd.Timestamp("2010-12-31"), col] == pytest.approx(0.01)
    assert out.loc[pd.Timestamp("2011-01-31"), col] == pytest.approx(0.02)
    # Preferred ticker QQQ should win once present.
    assert out.loc[pd.Timestamp("2011-02-28"), col] == pytest.approx(0.04)
