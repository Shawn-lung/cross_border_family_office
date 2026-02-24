"""Tests for Module 4 WRDS SQL builder utilities."""

from app.utils.data_fetcher import (
    build_compustat_eu_fundamentals_query,
    build_compustat_eu_pricing_query,
    build_crsp_proxy_query,
)


def test_build_crsp_proxy_query_includes_required_proxy_tickers():
    """CRSP proxy SQL should include non-BRK tickers plus explicit BRK.B PERMNO logic."""
    sql = build_crsp_proxy_query(years=15, row_limit=200_000)

    assert "FROM crsp.msf" in sql
    assert "JOIN crsp.msenames" in sql
    for ticker in ["'VT'", "'QQQ'", "'PRF'", "'IJS'"]:
        assert ticker in sql
    assert "OR m.permno = 83443" in sql
    assert "WHEN m.permno = 83443 THEN 'BRK.B'" in sql


def test_build_compustat_eu_pricing_query_uses_adjusted_fields_and_fx():
    """Pricing SQL should include ajexm/trt1m and EUR conversion via exrt_mth."""
    sql = build_compustat_eu_pricing_query(years=15, row_limit=200_000)

    assert "FROM comp.secm" in sql
    assert "s.ajexm AS adj_factor" in sql
    assert "s.trt1m" in sql
    assert "FROM comp.exrt_mth" in sql
    assert "fromcurm = 'GBP'" in sql
    assert "fx_eur.exratm / fx_local.exratm" in sql
    assert "AS price_adj_eur" in sql


def test_build_compustat_eu_fundamentals_query_converts_core_fields_to_eur():
    """Fundamentals SQL should map IB/OANCF/CEQ and convert them to EUR."""
    sql = build_compustat_eu_fundamentals_query(years=15, row_limit=200_000)

    assert "FROM comp.g_funda" in sql
    assert "f.ib AS earnings" in sql
    assert "f.oancf AS cash_earnings" in sql
    assert "f.ceq AS book_value" in sql
    assert "FROM comp.exrt_mth" in sql
    assert "fx_eur.exratm / fx_local.exratm" in sql
