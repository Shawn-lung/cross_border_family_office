"""Module 4 Alpha Sandbox UI with WRDS offline bundle and portfolio optimizer."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from app.utils.data_fetcher import (
    DEFAULT_WRDS_USERNAME,
    apply_fundamental_lag,
    build_compustat_eu_fundamentals_query,
    build_compustat_eu_pricing_query,
    build_crsp_proxy_query,
    execute_wrds_query_with_cache,
    merge_pricing_with_fundamentals,
)
from app.utils.factor_engine import (
    build_named_returns_from_crsp,
    build_synthetic_factor_return_series,
    calculate_annualized_mu_sigma,
    calculate_portfolio_metrics,
    normalize_weights,
)

RISK_FREE_RATE = 0.035
BUNDLE_DIR = Path("data") / "wrds_bundle"
BUNDLE_SCHEMA_VERSION = "v2"
CRSP_PROXY_TICKER_MAPPING: dict[str, list[str]] = {
    "VT (Proxy for VWCE)": ["VT"],
    "QQQ (Proxy for SXRV)": ["QQQ"],
    "PRF (Proxy for JPGL)": ["PRF"],
    "IJS (Proxy for ZPRV)": ["IJS"],
    "BRK.B": ["BRK.B"],
}
SYNTHETIC_ZPRX_LABEL = "Synthetic ZPRX (Europe SCV - Compustat)"


@dataclass(frozen=True)
class PipelineResult:
    """Container for factor-pipeline output used by the optimizer UI."""

    returns_df: pd.DataFrame
    source_label: str
    diagnostic_note: str


def _bundle_paths(years: int) -> tuple[Path, Path]:
    """Return data/meta file paths for offline bundle persistence."""
    data_path = BUNDLE_DIR / f"module4_returns_{years}y_{BUNDLE_SCHEMA_VERSION}.pkl"
    meta_path = BUNDLE_DIR / f"module4_returns_{years}y_{BUNDLE_SCHEMA_VERSION}.json"
    return data_path, meta_path


def _save_offline_bundle(years: int, returns_df: pd.DataFrame, metadata: dict[str, object]) -> Path:
    """Persist WRDS-built return bundle for offline runtime reuse."""
    data_path, meta_path = _bundle_paths(years)
    data_path.parent.mkdir(parents=True, exist_ok=True)
    returns_df.to_pickle(data_path)
    meta_path.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    return data_path


def _load_offline_bundle(years: int) -> tuple[pd.DataFrame, dict[str, object], Path] | None:
    """Load persisted WRDS return bundle when available."""
    data_path, meta_path = _bundle_paths(years)
    if not data_path.exists():
        return None
    try:
        returns_df = pd.read_pickle(data_path)
        metadata: dict[str, object] = {}
        if meta_path.exists():
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        return returns_df, metadata, data_path
    except Exception:
        return None


def _pricing_sql(years: int, row_limit: int) -> str:
    """Build monthly EU pricing SQL with adjusted-price and FX normalization fields."""
    return build_compustat_eu_pricing_query(years=years, row_limit=row_limit)


def _fundamentals_sql(years: int, row_limit: int) -> str:
    """Build annual EU fundamentals SQL with EUR-converted accounting fields."""
    return build_compustat_eu_fundamentals_query(years=years, row_limit=row_limit)


def _crsp_proxy_sql(years: int, row_limit: int) -> str:
    """Build CRSP monthly ticker return SQL for public proxy sleeves."""
    tickers = [t for tickers in CRSP_PROXY_TICKER_MAPPING.values() for t in tickers]
    return build_crsp_proxy_query(years=years, row_limit=row_limit, tickers=tickers)


def _prepare_pricing_df(raw_pricing: pd.DataFrame) -> pd.DataFrame:
    """Clean monthly pricing and derive adjusted returns for synthetic construction."""
    required = {"gvkey", "date", "shares", "trt1m"}
    if not required.issubset(raw_pricing.columns):
        missing = required - set(raw_pricing.columns)
        raise ValueError(f"Pricing query missing required columns: {sorted(missing)}")
    if not {"price_adj_eur", "price_eur", "price_local"}.intersection(raw_pricing.columns):
        raise ValueError(
            "Pricing query must include one of: price_adj_eur, price_eur, or price_local."
        )

    pricing = raw_pricing.copy()
    pricing["date"] = pd.to_datetime(pricing["date"], errors="coerce")
    pricing["shares"] = pd.to_numeric(pricing["shares"], errors="coerce")
    pricing["price_adj_eur"] = pd.to_numeric(pricing.get("price_adj_eur"), errors="coerce")
    pricing["price_eur"] = pd.to_numeric(pricing.get("price_eur"), errors="coerce")
    pricing["price_local"] = pd.to_numeric(pricing.get("price_local"), errors="coerce")
    pricing["trt1m"] = pd.to_numeric(pricing["trt1m"], errors="coerce")

    pricing["price"] = pricing["price_adj_eur"].where(
        pricing["price_adj_eur"].notna(), pricing["price_eur"]
    )
    pricing["price"] = pricing["price"].where(pricing["price"].notna(), pricing["price_local"])

    pricing = pricing.dropna(subset=["gvkey", "date", "shares", "price"])
    pricing = pricing[(pricing["price"] > 0) & (pricing["shares"] > 0)]
    pricing["me_raw"] = pricing["price"] * pricing["shares"]
    pricing = pricing[pricing["me_raw"] > 0]

    # One representative line per company per date (largest market-equity line).
    pricing = pricing.sort_values(["gvkey", "date", "me_raw"])
    pricing = pricing.drop_duplicates(subset=["gvkey", "date"], keep="last")

    # Compustat secm is monthly; still enforce one line per month for safety.
    pricing["month"] = pricing["date"].dt.to_period("M")
    pricing = pricing.sort_values(["gvkey", "month", "date", "me_raw"])
    pricing = pricing.drop_duplicates(subset=["gvkey", "month"], keep="last")
    pricing = pricing.sort_values(["gvkey", "date"])
    pricing["ret_from_trt1m"] = pricing["trt1m"] / 100.0
    pricing["ret_from_price"] = pricing.groupby("gvkey")["price"].pct_change()
    pricing["ret"] = pricing["ret_from_trt1m"].where(
        pricing["ret_from_trt1m"].between(-0.95, 5.0), pricing["ret_from_price"]
    )
    pricing = pricing.dropna(subset=["ret"])
    pricing = pricing[np.isfinite(pricing["ret"])]
    pricing = pricing[(pricing["ret"] > -0.95) & (pricing["ret"] < 5.0)]
    pricing = pricing.drop(columns=["month"], errors="ignore")
    return pricing


def _prepare_fundamentals_df(raw_fundamentals: pd.DataFrame) -> pd.DataFrame:
    """Clean fundamentals data for lagging and merge."""
    required = {"gvkey", "datadate", "sale", "earnings", "cash_earnings", "book_value"}
    if not required.issubset(raw_fundamentals.columns):
        missing = required - set(raw_fundamentals.columns)
        raise ValueError(f"Fundamentals query missing required columns: {sorted(missing)}")

    fund = raw_fundamentals.copy()
    fund["datadate"] = pd.to_datetime(fund["datadate"], errors="coerce")
    for col in ["sale", "earnings", "cash_earnings", "book_value"]:
        fund[col] = pd.to_numeric(fund[col], errors="coerce")
    fund = fund.dropna(subset=["gvkey", "datadate"])
    return fund


def _enforce_monthly_returns(returns_df: pd.DataFrame) -> pd.DataFrame:
    """Ensure month-end return index and aggregate duplicate rows by total monthly return."""
    if returns_df.empty:
        return returns_df

    monthly = returns_df.copy()
    monthly.index = pd.to_datetime(monthly.index, errors="coerce")
    monthly = monthly[monthly.index.notna()].sort_index()
    monthly = monthly.apply(pd.to_numeric, errors="coerce").dropna(how="all")
    monthly = (1.0 + monthly).groupby(monthly.index.to_period("M")).prod() - 1.0
    monthly.index = monthly.index.to_timestamp("M")
    monthly = monthly.replace([np.inf, -np.inf], np.nan).dropna(how="any")
    return monthly


def _generate_dummy_returns(years: int) -> pd.DataFrame:
    """Generate realistic fallback monthly returns for UI resilience."""
    months = years * 12
    dates = pd.date_range(end=pd.Timestamp.today().normalize(), periods=months, freq="ME")
    assets = list(CRSP_PROXY_TICKER_MAPPING.keys()) + [SYNTHETIC_ZPRX_LABEL]
    rng = np.random.default_rng(42)

    mean_monthly = np.array([0.0060, 0.0070, 0.0055, 0.0062, 0.0064, 0.0075])
    vol_monthly = np.array([0.045, 0.060, 0.040, 0.058, 0.050, 0.070])
    corr = np.array(
        [
            [1.00, 0.72, 0.68, 0.69, 0.64, 0.65],
            [0.72, 1.00, 0.60, 0.66, 0.58, 0.70],
            [0.68, 0.60, 1.00, 0.61, 0.55, 0.58],
            [0.69, 0.66, 0.61, 1.00, 0.57, 0.66],
            [0.64, 0.58, 0.55, 0.57, 1.00, 0.60],
            [0.65, 0.70, 0.58, 0.66, 0.60, 1.00],
        ]
    )
    cov = np.outer(vol_monthly, vol_monthly) * corr
    draws = rng.multivariate_normal(mean_monthly, cov, size=months)
    return pd.DataFrame(draws, index=dates, columns=assets)


def _build_wrds_return_bundle(
    *,
    username: str,
    years: int,
    row_limit: int,
    use_query_cache: bool,
    force_refresh: bool,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Build full WRDS-only return frame for optimizer sleeves."""
    pricing_raw, pricing_source, _ = execute_wrds_query_with_cache(
        _pricing_sql(years=years, row_limit=row_limit),
        username=username,
        use_cache=use_query_cache,
        force_refresh=force_refresh,
    )
    fund_raw, fund_source, _ = execute_wrds_query_with_cache(
        _fundamentals_sql(years=years, row_limit=row_limit),
        username=username,
        use_cache=use_query_cache,
        force_refresh=force_refresh,
    )
    crsp_raw, crsp_source, _ = execute_wrds_query_with_cache(
        _crsp_proxy_sql(years=years, row_limit=row_limit),
        username=username,
        use_cache=use_query_cache,
        force_refresh=force_refresh,
    )

    pricing_df = _prepare_pricing_df(pricing_raw)
    fund_df = _prepare_fundamentals_df(fund_raw)
    lagged_fund = apply_fundamental_lag(fund_df, fundamental_date_col="datadate", lag_months=4)
    merged = merge_pricing_with_fundamentals(
        pricing_df=pricing_df,
        fundamentals_lagged_df=lagged_fund,
        id_col="gvkey",
        pricing_date_col="date",
        available_date_col="fundamental_available_date",
    )

    synthetic = build_synthetic_factor_return_series(
        merged,
        id_col="gvkey",
        date_col="date",
        price_col="price",
        shares_col="shares",
        book_value_col="book_value",
        sales_col="sale",
        earnings_col="earnings",
        cash_earnings_col="cash_earnings",
        return_col="ret",
        rebalance_months=(5, 11),
        output_name=SYNTHETIC_ZPRX_LABEL,
    )
    proxies = build_named_returns_from_crsp(
        crsp_raw,
        mapping=CRSP_PROXY_TICKER_MAPPING,
        date_col="date",
        ticker_col="ticker",
        return_col="ret",
    )

    returns_df = pd.concat([proxies, synthetic], axis=1)
    returns_df = returns_df.replace([np.inf, -np.inf], np.nan).dropna()
    returns_df = _enforce_monthly_returns(returns_df)
    if len(returns_df) < 24:
        raise RuntimeError("WRDS bundle build produced fewer than 24 monthly observations.")

    metadata: dict[str, object] = {
        "rows_pricing_raw": int(len(pricing_raw)),
        "rows_fund_raw": int(len(fund_raw)),
        "rows_crsp_raw": int(len(crsp_raw)),
        "rows_merged": int(len(merged)),
        "rows_returns": int(len(returns_df)),
        "pricing_source": pricing_source,
        "fund_source": fund_source,
        "crsp_source": crsp_source,
        "proxy_mapping": CRSP_PROXY_TICKER_MAPPING,
        "bundle_schema_version": BUNDLE_SCHEMA_VERSION,
        "years": years,
        "row_limit": row_limit,
    }
    return returns_df, metadata


def _run_factor_pipeline(
    *,
    username: str,
    years: int,
    row_limit: int,
    use_query_cache: bool,
    use_offline_bundle: bool,
    force_refresh: bool,
) -> PipelineResult:
    """Load from offline bundle or rebuild bundle from WRDS."""
    try:
        if use_offline_bundle and not force_refresh:
            cached_bundle = _load_offline_bundle(years=years)
            if cached_bundle is not None:
                returns_df, metadata, path = cached_bundle
                note = (
                    f"Loaded local bundle `{path}` with {len(returns_df):,} monthly rows."
                    f" Meta: {metadata}"
                )
                return PipelineResult(
                    returns_df=returns_df,
                    source_label="WRDS offline bundle",
                    diagnostic_note=note,
                )

        returns_df, metadata = _build_wrds_return_bundle(
            username=username,
            years=years,
            row_limit=row_limit,
            use_query_cache=use_query_cache,
            force_refresh=force_refresh,
        )

        if use_offline_bundle:
            bundle_path = _save_offline_bundle(years=years, returns_df=returns_df, metadata=metadata)
            note = f"Built WRDS bundle and saved to `{bundle_path}`. Meta: {metadata}"
        else:
            note = f"Built directly from WRDS (no offline bundle save). Meta: {metadata}"

        return PipelineResult(
            returns_df=returns_df,
            source_label="WRDS live build",
            diagnostic_note=note,
        )
    except Exception as exc:
        dummy = _generate_dummy_returns(years=years)
        return PipelineResult(
            returns_df=dummy,
            source_label="Fallback synthetic dataset",
            diagnostic_note=f"Fallback reason: {exc}",
        )


def _render_matrix(returns_df: pd.DataFrame, matrix_type: str) -> None:
    """Render covariance or correlation matrix heatmap."""
    if matrix_type == "Covariance (Annualized)":
        matrix = returns_df.cov() * 12.0
        title = "Annualized Covariance Matrix"
        fmt = ".4f"
    else:
        matrix = returns_df.corr()
        title = "Correlation Matrix"
        fmt = ".2f"

    fig = px.imshow(
        matrix,
        text_auto=fmt,
        color_continuous_scale="RdBu",
        zmin=float(matrix.values.min()),
        zmax=float(matrix.values.max()),
        aspect="auto",
        title=title,
    )
    fig.update_layout(coloraxis_colorbar_title="")
    st.plotly_chart(fig, use_container_width=True)


def _render_portfolio_optimizer(returns_df: pd.DataFrame) -> None:
    """Render weight sliders and live MPT metrics."""
    mu, sigma = calculate_annualized_mu_sigma(returns_df)
    assets = list(mu.index)

    st.markdown("### Portfolio Optimizer")
    st.caption("Raw slider weights are normalized to 100% before matrix calculations.")

    cols = st.columns(min(len(assets), 5))
    raw_weights: dict[str, float] = {}
    default_weight = 100.0 / len(assets)
    for idx, asset in enumerate(assets):
        with cols[idx % len(cols)]:
            raw_weights[asset] = st.slider(
                asset,
                min_value=0.0,
                max_value=100.0,
                value=float(default_weight),
                step=1.0,
                key=f"m4_weight_{asset}",
            )

    normalized = normalize_weights(pd.Series(raw_weights, dtype=float))
    metrics = calculate_portfolio_metrics(mu, sigma, normalized, risk_free_rate=RISK_FREE_RATE)

    c1, c2, c3 = st.columns(3)
    c1.metric("Expected Return", f"{metrics['expected_return'] * 100:.2f}%")
    c2.metric("Volatility (Risk)", f"{metrics['volatility'] * 100:.2f}%")
    c3.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.3f}")

    weights_view = pd.DataFrame(
        {
            "Asset": normalized.index,
            "Normalized Weight (%)": normalized.values * 100.0,
            "Annualized Mu (%)": mu.loc[normalized.index].values * 100.0,
        }
    )
    st.dataframe(weights_view, use_container_width=True, hide_index=True)

    cumulative = (1.0 + returns_df).cumprod()
    cumulative = cumulative / cumulative.iloc[0]
    cumulative.index.name = "Date"
    curve_df = cumulative.reset_index().melt(id_vars="Date", var_name="Asset", value_name="Growth")
    fig = px.line(curve_df, x="Date", y="Growth", color="Asset", title="Growth of 1 Unit")
    st.plotly_chart(fig, use_container_width=True)


def render() -> None:
    """Render Module 4 WRDS offline bundle and optimizer UI."""
    st.subheader("Module 4: Alpha Sandbox (WRDS Offline Bundle)")
    st.caption(
        "WRDS-only runtime: CRSP proxies (VT/QQQ/PRF/IJS/BRK) + synthetic ZPRX from Compustat Europe."
    )
    st.caption("Synthetic ZPRX uses EUR-normalized Compustat Global data with May/November rebalancing.")

    c1, c2, c3 = st.columns(3)
    username = c1.text_input("WRDS Username", value=DEFAULT_WRDS_USERNAME)
    years = c2.slider("History (Years)", min_value=5, max_value=20, value=15, step=1)
    row_limit = c3.slider("WRDS Row Limit", min_value=50_000, max_value=500_000, value=200_000, step=10_000)

    c4, c5, c6 = st.columns(3)
    use_query_cache = c4.checkbox("Use WRDS Query Cache (Recommended)", value=True)
    use_offline_bundle = c5.checkbox("Use Offline Bundle (Recommended)", value=True)
    force_refresh = c6.checkbox("Force WRDS Rebuild", value=False)
    matrix_type = st.selectbox("Matrix View", ["Covariance (Annualized)", "Correlation"], index=0)

    run_pipeline = st.button("Load / Refresh WRDS Bundle", type="primary")
    if run_pipeline:
        with st.spinner("Building WRDS-only return bundle..."):
            result = _run_factor_pipeline(
                username=username,
                years=years,
                row_limit=row_limit,
                use_query_cache=use_query_cache,
                use_offline_bundle=use_offline_bundle,
                force_refresh=force_refresh,
            )
        st.session_state["m4_returns_df"] = result.returns_df
        st.session_state["m4_source_label"] = result.source_label
        st.session_state["m4_note"] = result.diagnostic_note

    returns_df = st.session_state.get("m4_returns_df")
    if returns_df is None:
        st.info("Click `Load / Refresh WRDS Bundle` to prepare returns and run the optimizer.")
        return

    st.success(f"Data Source: {st.session_state.get('m4_source_label', 'Unknown')}")
    st.caption(st.session_state.get("m4_note", ""))
    st.dataframe(returns_df.tail(24), use_container_width=True)

    _render_matrix(returns_df, matrix_type=matrix_type)
    _render_portfolio_optimizer(returns_df)
