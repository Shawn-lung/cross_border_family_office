"""Module 2 Fama-French factor dashboard with Swedroe 5/25 rebalance signaling."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st
import yfinance as yf

from app.utils.forex_engine import ForexAPIError, ForexRates, fetch_live_fx_rates
from app.utils.portfolio_math import swedroe_5_25_rebalance_required

TOTAL_PORTFOLIO_TWD = 7_000_000.0
TARGET_WEIGHTS = {
    "VWCE": 40.0,
    "SXRV": 15.0,
    "ZPRV": 15.0,
    "ZPRX": 5.0,
    "BRKB": 15.0,
    "JPGL": 10.0,
}
ASSET_SYMBOL_CANDIDATES: dict[str, list[str]] = {
    "VWCE": ["VWCE.DE", "VWCE.AS", "VWCE.L", "VWCE"],
    "SXRV": ["SXRV.DE", "SXRV.AS", "SXRV"],
    "ZPRV": ["ZPRV.AS", "ZPRV.DE", "ZPRV"],
    "ZPRX": ["ZPRX.AS", "ZPRX.DE", "ZPRX"],
    "BRKB": ["BRK-B", "BRK.B"],
    "JPGL": ["JPGL.AS", "JPGL.DE", "JPGL.L", "JPGL"],
}
DEFAULT_FX_RATES = ForexRates(eur_usd=1.10, usd_twd=31.0, eur_twd=34.1)


def _weights_from_amounts(amounts_twd: dict[str, float]) -> dict[str, float]:
    """Convert per-asset TWD amounts to percentage weights."""
    total = sum(amounts_twd.values())
    if total <= 0:
        raise ValueError("Total current portfolio value must be greater than zero.")
    return {asset: (amount / total) * 100.0 for asset, amount in amounts_twd.items()}


def _extract_last_price(ticker: yf.Ticker) -> float | None:
    """Get latest valid close/last price from yfinance ticker."""
    try:
        history = ticker.history(period="5d")
        if not history.empty and "Close" in history.columns:
            closes = history["Close"].dropna()
            if not closes.empty:
                price = float(closes.iloc[-1])
                if price > 0:
                    return price
    except Exception:
        pass

    try:
        fast_info = getattr(ticker, "fast_info", {}) or {}
        last_price = fast_info.get("lastPrice") or fast_info.get("regularMarketPrice")
        if last_price is not None and float(last_price) > 0:
            return float(last_price)
    except Exception:
        pass

    try:
        info = ticker.info or {}
        maybe = info.get("regularMarketPrice") or info.get("currentPrice") or info.get("previousClose")
        if maybe is not None and float(maybe) > 0:
            return float(maybe)
    except Exception:
        pass

    return None


def _extract_currency(ticker: yf.Ticker) -> str:
    """Get ticker trading currency with safe defaults."""
    try:
        fast_info = getattr(ticker, "fast_info", {}) or {}
        currency = fast_info.get("currency")
        if currency:
            return str(currency).upper()
    except Exception:
        pass

    try:
        info = ticker.info or {}
        currency = info.get("currency")
        if currency:
            return str(currency).upper()
    except Exception:
        pass

    return "USD"


@st.cache_data(ttl=300)
def _fetch_direct_fx_to_twd(currency: str) -> float:
    """Fetch direct currency-to-TWD rate from Yahoo, e.g., CADTWD=X."""
    symbol = f"{currency}TWD=X"
    ticker = yf.Ticker(symbol)
    price = _extract_last_price(ticker)
    if price is None:
        raise RuntimeError(f"Unable to fetch FX rate for {currency}/TWD.")
    return float(price)


def _price_to_twd(price: float, currency: str, fx_rates: ForexRates) -> float:
    """Convert a quote price to TWD using fetched FX rates when needed."""
    ccy = currency.upper()
    if ccy == "TWD":
        return price
    if ccy == "USD":
        return price * fx_rates.usd_twd
    if ccy == "EUR":
        return price * fx_rates.eur_twd
    if ccy in {"GBP", "GBX", "GBP", "GBP."}:
        gbp_twd = _fetch_direct_fx_to_twd("GBP")
        if ccy == "GBX":
            return (price / 100.0) * gbp_twd
        return price * gbp_twd
    # Last-resort path for other currencies.
    return price * _fetch_direct_fx_to_twd(ccy)


@st.cache_data(ttl=120)
def _fetch_live_quote(asset: str) -> dict[str, str | float]:
    """Resolve live quote for an asset using symbol fallbacks."""
    if asset not in ASSET_SYMBOL_CANDIDATES:
        raise RuntimeError(f"No symbol mapping configured for {asset}.")

    errors: list[str] = []
    for symbol in ASSET_SYMBOL_CANDIDATES[asset]:
        try:
            ticker = yf.Ticker(symbol)
            price = _extract_last_price(ticker)
            if price is None:
                errors.append(f"{symbol}: no valid price")
                continue

            currency = _extract_currency(ticker)
            return {"asset": asset, "symbol": symbol, "price": float(price), "currency": currency}
        except Exception as exc:
            errors.append(f"{symbol}: {exc}")

    raise RuntimeError(f"Live quote unavailable for {asset}. Tried {', '.join(errors)}")


def render() -> None:
    """Render Module 2 factor dashboard."""
    st.subheader("Module 2: Fama-French Factor Engine")
    st.caption(f"Strategic baseline portfolio size: TWD {TOTAL_PORTFOLIO_TWD:,.0f}")

    try:
        fx_rates = fetch_live_fx_rates()
    except ForexAPIError as exc:
        fx_rates = DEFAULT_FX_RATES
        st.warning(f"Live FX fetch failed ({exc}). Using fallback FX assumptions.")

    st.markdown("### Current Holdings Inputs (Shares)")
    st.caption("Enter share counts. Live prices are fetched on each app refresh and converted to TWD.")

    quotes: dict[str, dict[str, str | float]] = {}
    quote_errors: list[str] = []
    with st.spinner("Fetching live prices..."):
        for asset in TARGET_WEIGHTS:
            try:
                quotes[asset] = _fetch_live_quote(asset)
            except Exception as exc:
                quote_errors.append(f"{asset}: {exc}")

    if quote_errors:
        st.error("Unable to fetch all live prices. Please retry.")
        for msg in quote_errors:
            st.caption(msg)
        return

    cols = st.columns(2)
    current_amounts: dict[str, float] = {}
    current_shares: dict[str, int] = {}
    price_rows: list[dict[str, str | float]] = []
    for idx, (asset, target_weight) in enumerate(TARGET_WEIGHTS.items()):
        quote = quotes[asset]
        symbol = str(quote["symbol"])
        price = float(quote["price"])
        currency = str(quote["currency"]).upper()
        price_twd = _price_to_twd(price, currency, fx_rates)
        default_value = TOTAL_PORTFOLIO_TWD * (target_weight / 100.0)
        default_shares = int(round(default_value / price_twd)) if price_twd > 0 else 0

        with cols[idx % 2]:
            shares = st.number_input(
                f"{asset} Shares",
                min_value=0,
                value=int(default_shares),
                step=1,
                format="%d",
                key=f"shares_{asset}",
            )
            st.caption(f"{symbol}: {price:,.4f} {currency} ({price_twd:,.2f} TWD/share)")

        current_shares[asset] = int(shares)
        current_amounts[asset] = float(int(shares)) * price_twd
        price_rows.append(
            {
                "Asset": asset,
                "Symbol": symbol,
                "Price": price,
                "Currency": currency,
                "Price (TWD)": price_twd,
            }
        )

    total_current_twd = float(sum(current_amounts.values()))
    if total_current_twd <= 0:
        st.error("Total current holdings value must be greater than zero.")
        return

    st.metric(
        "Live Portfolio Size (TWD)",
        f"{total_current_twd:,.0f}",
        delta=f"{total_current_twd - TOTAL_PORTFOLIO_TWD:,.0f} vs baseline",
    )

    current_weights = _weights_from_amounts(current_amounts)
    rebalance_required = swedroe_5_25_rebalance_required(
        target_weights=TARGET_WEIGHTS,
        current_weights=current_weights,
    )

    st.markdown(
        """
<style>
div[data-testid="stMetricValue"] {
    font-size: 2.8rem;
    font-weight: 800;
}
div[data-testid="stMetricDelta"] {
    font-size: 1.2rem;
    font-weight: 700;
}
</style>
""",
        unsafe_allow_html=True,
    )

    if rebalance_required:
        status_value = "REBALANCE REQUIRED"
        delta = "+1 trigger breached"
    else:
        status_value = "WITHIN SWEDROE BANDS"
        delta = "-1 no trigger"

    st.metric("Swedroe 5/25 Status", status_value, delta=delta, delta_color="inverse")

    comparison_rows: list[dict[str, str | float]] = []
    drift_rows: list[dict[str, str | float]] = []
    for asset, target in TARGET_WEIGHTS.items():
        actual = current_weights[asset]
        comparison_rows.append({"Asset": asset, "Type": "Target", "Weight (%)": target})
        comparison_rows.append({"Asset": asset, "Type": "Actual", "Weight (%)": actual})
        drift_rows.append(
            {
                "Asset": asset,
                "Target (%)": target,
                "Actual (%)": actual,
                "Drift (pp)": actual - target,
            }
        )

    chart_df = pd.DataFrame(comparison_rows)
    fig = px.bar(
        chart_df,
        x="Asset",
        y="Weight (%)",
        color="Type",
        barmode="group",
        title="Target vs. Actual Weights",
    )
    fig.update_layout(yaxis_title="Weight (%)", xaxis_title="")
    st.plotly_chart(fig, use_container_width=True)

    drift_df = pd.DataFrame(drift_rows)
    price_df = pd.DataFrame(price_rows)
    drift_df["Shares"] = drift_df["Asset"].map(current_shares)
    drift_df["Live Symbol"] = drift_df["Asset"].map(price_df.set_index("Asset")["Symbol"])
    drift_df["Live Price (TWD/share)"] = drift_df["Asset"].map(price_df.set_index("Asset")["Price (TWD)"])
    drift_df["Current Value (TWD)"] = drift_df["Asset"].map(current_amounts)
    drift_df["Target Value (TWD)"] = drift_df["Target (%)"] / 100.0 * total_current_twd
    drift_df["Rebalance Trade (TWD)"] = drift_df["Target Value (TWD)"] - drift_df["Current Value (TWD)"]
    st.dataframe(drift_df, use_container_width=True, hide_index=True)
