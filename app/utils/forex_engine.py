"""Forex utilities for fetching live EUR/USD, USD/TWD, EUR/TWD rates and converting amounts."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlopen

import yfinance as yf

try:
    import certifi
except Exception:  # pragma: no cover - optional import guard
    certifi = None


class ForexAPIError(RuntimeError):
    """Raised when live forex data cannot be fetched or validated."""


@dataclass(frozen=True)
class ForexRates:
    """Container for required live rates used by the dashboard baseline."""

    eur_usd: float
    usd_twd: float
    eur_twd: float

    @property
    def triangulated_eur_twd(self) -> float:
        """Cross rate derived from EUR/USD * USD/TWD."""
        return self.eur_usd * self.usd_twd


_CA_BUNDLE_CONFIGURED = False


def _ensure_curl_ca_bundle() -> None:
    """Force an ASCII CA bundle path for curl-based clients on Windows."""
    global _CA_BUNDLE_CONFIGURED
    if _CA_BUNDLE_CONFIGURED or certifi is None:
        return

    try:
        source_ca = Path(certifi.where())
        if not source_ca.exists():
            _CA_BUNDLE_CONFIGURED = True
            return

        source_path = str(source_ca)
        if source_path.isascii():
            os.environ.setdefault("CURL_CA_BUNDLE", source_path)
            os.environ.setdefault("SSL_CERT_FILE", source_path)
            os.environ.setdefault("REQUESTS_CA_BUNDLE", source_path)
            _CA_BUNDLE_CONFIGURED = True
            return

        target_ca = Path(tempfile.gettempdir()) / "cross_border_family_office_cacert.pem"
        shutil.copyfile(source_ca, target_ca)
        target_path = str(target_ca)
        os.environ["CURL_CA_BUNDLE"] = target_path
        os.environ["SSL_CERT_FILE"] = target_path
        os.environ["REQUESTS_CA_BUNDLE"] = target_path
    finally:
        _CA_BUNDLE_CONFIGURED = True


def _fetch_pair_rate_via_yahoo_chart(symbol: str) -> float:
    """Fallback: pull the latest close from Yahoo Chart endpoint via stdlib HTTP."""
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?range=5d&interval=1d"
    try:
        with urlopen(url, timeout=10) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception as exc:  # pragma: no cover - network failure branch
        raise ForexAPIError(f"Fallback fetch failed for {symbol}: {exc}") from exc

    try:
        closes = payload["chart"]["result"][0]["indicators"]["quote"][0]["close"]
    except Exception as exc:
        raise ForexAPIError(f"Fallback returned malformed data for {symbol}.") from exc

    valid_closes = [float(v) for v in closes if v is not None]
    if not valid_closes:
        raise ForexAPIError(f"Fallback returned no valid close prices for {symbol}.")

    rate = valid_closes[-1]
    if rate <= 0:
        raise ForexAPIError(f"Fallback returned invalid non-positive rate for {symbol}: {rate}")
    return rate


def _fetch_pair_rate(symbol: str) -> float:
    """Fetch the latest close for a yfinance forex symbol."""
    _ensure_curl_ca_bundle()

    try:
        history = yf.Ticker(symbol).history(period="5d")
    except Exception:
        return _fetch_pair_rate_via_yahoo_chart(symbol)

    if history.empty or "Close" not in history.columns:
        return _fetch_pair_rate_via_yahoo_chart(symbol)

    closes = history["Close"].dropna()
    if closes.empty:
        return _fetch_pair_rate_via_yahoo_chart(symbol)

    rate = float(closes.iloc[-1])
    if rate <= 0:
        return _fetch_pair_rate_via_yahoo_chart(symbol)

    return rate


def fetch_live_fx_rates() -> ForexRates:
    """Fetch EUR/USD, USD/TWD, EUR/TWD live rates from yfinance."""
    eur_usd = _fetch_pair_rate("EURUSD=X")
    usd_twd = _fetch_pair_rate("USDTWD=X")
    eur_twd = _fetch_pair_rate("EURTWD=X")

    return ForexRates(eur_usd=eur_usd, usd_twd=usd_twd, eur_twd=eur_twd)


def convert_eur_to_twd(amount_eur: float, rates: ForexRates, use_triangulated: bool = True) -> float:
    """Convert EUR to TWD using triangulated or direct EUR/TWD rate."""
    if amount_eur < 0:
        raise ValueError("amount_eur must be non-negative.")
    rate = rates.triangulated_eur_twd if use_triangulated else rates.eur_twd
    return amount_eur * rate


def convert_twd_to_eur(amount_twd: float, rates: ForexRates, use_triangulated: bool = True) -> float:
    """Convert TWD to EUR using inverse of triangulated or direct EUR/TWD rate."""
    if amount_twd < 0:
        raise ValueError("amount_twd must be non-negative.")
    rate = rates.triangulated_eur_twd if use_triangulated else rates.eur_twd
    return amount_twd / rate


def relative_error_percent(expected: float, actual: float) -> float:
    """Return absolute relative error percentage between expected and actual."""
    if expected == 0:
        raise ValueError("expected must not be zero.")
    return abs((actual - expected) / expected) * 100.0
