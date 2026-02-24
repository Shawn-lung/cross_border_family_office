"""Tests for forex triangulation precision and conversion consistency."""

from unittest.mock import MagicMock, patch

import pandas as pd

from app.utils.forex_engine import (
    convert_eur_to_twd,
    convert_twd_to_eur,
    fetch_live_fx_rates,
    relative_error_percent,
)


def _mock_ticker_factory(symbol_to_rate: dict[str, float]):
    """Build a yfinance.Ticker mock returning per-symbol close data."""

    def _factory(symbol: str) -> MagicMock:
        ticker = MagicMock()
        ticker.history.return_value = pd.DataFrame({"Close": [symbol_to_rate[symbol]]})
        return ticker

    return _factory


@patch("app.utils.forex_engine.yf.Ticker")
def test_eur_twd_round_trip_precision_under_point_zero_one_percent(mock_ticker):
    """1,000 EUR converted to TWD and back must stay within 0.01% error."""
    symbol_to_rate = {
        "EURUSD=X": 1.10,
        "USDTWD=X": 31.20,
        "EURTWD=X": 34.32,
    }
    mock_ticker.side_effect = _mock_ticker_factory(symbol_to_rate)

    rates = fetch_live_fx_rates()
    starting_eur = 1000.0
    twd_amount = convert_eur_to_twd(starting_eur, rates, use_triangulated=True)
    eur_round_trip = convert_twd_to_eur(twd_amount, rates, use_triangulated=True)

    error_pct = relative_error_percent(starting_eur, eur_round_trip)

    assert error_pct < 0.01
