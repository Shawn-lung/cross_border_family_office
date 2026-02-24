"""Tests for Module 4 data fetcher lag controls and WRDS query execution."""

from unittest.mock import MagicMock

import pandas as pd
import pytest

from app.utils.data_fetcher import (
    LookAheadBiasError,
    apply_fundamental_lag,
    execute_wrds_query,
    execute_wrds_query_with_cache,
    merge_pricing_with_fundamentals,
)


def test_merge_requires_fundamental_lag_column():
    """Merging without lagged availability date should raise look-ahead guard error."""
    pricing_df = pd.DataFrame({"gvkey": ["001"], "date": ["2025-05-31"], "ret": [0.01]})
    fundamentals_df = pd.DataFrame({"gvkey": ["001"], "datadate": ["2025-04-30"], "book_value": [10.0]})

    with pytest.raises(LookAheadBiasError):
        merge_pricing_with_fundamentals(
            pricing_df=pricing_df,
            fundamentals_lagged_df=fundamentals_df,
            id_col="gvkey",
            pricing_date_col="date",
            available_date_col="fundamental_available_date",
        )


def test_apply_lag_then_merge_prevents_lookahead_assignment():
    """Fundamentals dated one month before price date should not be assigned before 4-month lag."""
    pricing_df = pd.DataFrame(
        {
            "gvkey": ["001", "001"],
            "date": ["2025-05-31", "2025-11-30"],
            "ret": [0.01, 0.02],
        }
    )
    fundamentals_df = pd.DataFrame(
        {
            "gvkey": ["001"],
            "datadate": ["2025-04-30"],
            "book_value": [123.0],
        }
    )

    lagged = apply_fundamental_lag(fundamentals_df, fundamental_date_col="datadate", lag_months=4)
    merged = merge_pricing_with_fundamentals(
        pricing_df=pricing_df,
        fundamentals_lagged_df=lagged,
        id_col="gvkey",
        pricing_date_col="date",
        available_date_col="fundamental_available_date",
    )

    may_row = merged.loc[merged["date"] == pd.Timestamp("2025-05-31")].iloc[0]
    nov_row = merged.loc[merged["date"] == pd.Timestamp("2025-11-30")].iloc[0]

    assert pd.isna(may_row["book_value"])
    assert nov_row["book_value"] == pytest.approx(123.0)


def test_execute_wrds_query_with_mocked_connection_returns_dataframe():
    """WRDS query helper should call raw_sql on provided mocked connection."""
    expected = pd.DataFrame({"gvkey": ["001"], "ret": [0.03]})
    mock_connection = MagicMock()
    mock_connection.raw_sql.return_value = expected

    result = execute_wrds_query("SELECT gvkey, ret FROM mock_table", connection=mock_connection)

    mock_connection.raw_sql.assert_called_once_with("SELECT gvkey, ret FROM mock_table")
    pd.testing.assert_frame_equal(result, expected)


def test_execute_wrds_query_allows_with_cte_queries():
    """CTE-style read queries should be accepted by executor guardrails."""
    expected = pd.DataFrame({"x": [1]})
    mock_connection = MagicMock()
    mock_connection.raw_sql.return_value = expected

    result = execute_wrds_query(
        """
        WITH base AS (
            SELECT 1 AS x
        )
        SELECT x FROM base
        """,
        connection=mock_connection,
    )

    assert mock_connection.raw_sql.call_count == 1
    pd.testing.assert_frame_equal(result, expected)


def test_execute_wrds_query_sanitizes_unicode_operators_and_comp_global_schema():
    """Executor should normalize pasted unicode operators and old schema name."""
    expected = pd.DataFrame({"gvkey": ["001"], "datadate": [pd.Timestamp("2025-01-31")]})
    mock_connection = MagicMock()
    mock_connection.raw_sql.return_value = expected

    sql = """
    SELECT gvkey, datadate
    FROM comp_global.g_funda
    WHERE datadate \u2265 CURRENT_DATE - INTERVAL '15 years'
    LIMIT 1;
    """
    result = execute_wrds_query(sql, connection=mock_connection)

    called_sql = mock_connection.raw_sql.call_args[0][0]
    assert "comp.g_funda" in called_sql
    assert "\u2265" not in called_sql
    assert ">=" in called_sql
    pd.testing.assert_frame_equal(result, expected)


def test_execute_wrds_query_with_cache_reuses_saved_snapshot(tmp_path):
    """Second call with identical SQL should load from local cache, not WRDS."""
    expected = pd.DataFrame({"gvkey": ["001"], "ret": [0.03]})
    mock_connection = MagicMock()
    mock_connection.raw_sql.return_value = expected

    first_df, first_source, cache_path = execute_wrds_query_with_cache(
        "SELECT gvkey, ret FROM mock_table",
        username="demo_user",
        use_cache=True,
        force_refresh=False,
        cache_dir=tmp_path,
        connection=mock_connection,
    )
    assert first_source == "wrds"
    assert cache_path.exists()
    pd.testing.assert_frame_equal(first_df, expected)

    mock_connection.raw_sql.reset_mock()
    mock_connection.raw_sql.side_effect = RuntimeError("Should not hit WRDS on cache hit")

    second_df, second_source, second_path = execute_wrds_query_with_cache(
        "SELECT gvkey, ret FROM mock_table",
        username="demo_user",
        use_cache=True,
        force_refresh=False,
        cache_dir=tmp_path,
        connection=mock_connection,
    )
    assert second_source == "cache"
    assert second_path == cache_path
    mock_connection.raw_sql.assert_not_called()
    pd.testing.assert_frame_equal(second_df, expected)


def test_execute_wrds_query_with_cache_force_refresh_hits_wrds_again(tmp_path):
    """Force-refresh should bypass existing cache and fetch from WRDS."""
    initial = pd.DataFrame({"gvkey": ["001"], "ret": [0.03]})
    refreshed = pd.DataFrame({"gvkey": ["001"], "ret": [0.05]})
    mock_connection = MagicMock()
    mock_connection.raw_sql.return_value = initial

    execute_wrds_query_with_cache(
        "SELECT gvkey, ret FROM mock_table",
        username="demo_user",
        use_cache=True,
        force_refresh=False,
        cache_dir=tmp_path,
        connection=mock_connection,
    )

    mock_connection.raw_sql.reset_mock()
    mock_connection.raw_sql.return_value = refreshed

    refreshed_df, source, _ = execute_wrds_query_with_cache(
        "SELECT gvkey, ret FROM mock_table",
        username="demo_user",
        use_cache=True,
        force_refresh=True,
        cache_dir=tmp_path,
        connection=mock_connection,
    )
    assert source == "wrds"
    mock_connection.raw_sql.assert_called_once()
    pd.testing.assert_frame_equal(refreshed_df, refreshed)
