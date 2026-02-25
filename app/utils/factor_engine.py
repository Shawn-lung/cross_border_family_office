"""Factor-construction and portfolio-math utilities for Module 4."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd


def calculate_core_metrics(
    merged_df: pd.DataFrame,
    *,
    price_col: str = "price",
    shares_col: str = "shares",
    book_value_col: str = "book_value",
    me_col: str = "me",
    bm_col: str = "bm",
) -> pd.DataFrame:
    """Add Market Equity (ME) and Book-to-Market (B/M) metrics."""
    required = {price_col, shares_col, book_value_col}
    if not required.issubset(merged_df.columns):
        missing = required - set(merged_df.columns)
        raise KeyError(f"Missing columns for core metrics: {sorted(missing)}")

    df = merged_df.copy()
    df[price_col] = pd.to_numeric(df[price_col], errors="coerce")
    df[shares_col] = pd.to_numeric(df[shares_col], errors="coerce")
    df[book_value_col] = pd.to_numeric(df[book_value_col], errors="coerce")

    df[me_col] = df[price_col] * df[shares_col]
    df[bm_col] = np.where(df[me_col] > 0, df[book_value_col] / df[me_col], np.nan)
    return df


def select_small_value_universe(
    enriched_df: pd.DataFrame,
    *,
    date_col: str = "date",
    me_col: str = "me",
    bm_col: str = "bm",
    small_cap_quantile: float = 0.30,
    value_quantile: float = 0.70,
) -> pd.DataFrame:
    """Run independent double sort and keep small-cap value names by period."""
    if not 0.0 < small_cap_quantile < 1.0:
        raise ValueError("small_cap_quantile must be between 0 and 1.")
    if not 0.0 < value_quantile < 1.0:
        raise ValueError("value_quantile must be between 0 and 1.")

    required = {date_col, me_col, bm_col}
    if not required.issubset(enriched_df.columns):
        missing = required - set(enriched_df.columns)
        raise KeyError(f"Missing columns for double sort: {sorted(missing)}")

    df = enriched_df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col, me_col, bm_col])
    if df.empty:
        return df

    me_cut = df.groupby(date_col)[me_col].transform(lambda x: x.quantile(small_cap_quantile))
    small_cap = df[df[me_col] <= me_cut].copy()
    if small_cap.empty:
        return small_cap

    bm_cut = small_cap.groupby(date_col)[bm_col].transform(lambda x: x.quantile(value_quantile))
    small_value = small_cap[small_cap[bm_col] >= bm_cut].copy()
    return small_value


def compute_msci_fundamental_weights(
    small_value_df: pd.DataFrame,
    *,
    date_col: str = "date",
    sales_col: str = "sale",
    earnings_col: str = "earnings",
    cash_earnings_col: str = "cash_earnings",
    book_value_col: str = "book_value",
    weight_col: str = "weight",
    max_weight_cap: float | None = 0.05,
) -> pd.DataFrame:
    """Compute MSCI-style fundamental weights by cross-section and period.

    A strict max-weight cap can be applied after initial normalization.
    If the cap is infeasible for a cross-section (e.g., 2 names with 5% cap),
    normalized uncapped weights are retained for that period.
    """
    required = {date_col, sales_col, earnings_col, cash_earnings_col, book_value_col}
    if not required.issubset(small_value_df.columns):
        missing = required - set(small_value_df.columns)
        raise KeyError(f"Missing columns for fundamental weights: {sorted(missing)}")
    if max_weight_cap is not None and max_weight_cap <= 0:
        raise ValueError("max_weight_cap must be positive when provided.")

    df = small_value_df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])
    if df.empty:
        raise ValueError("Cannot compute fundamental weights on an empty universe.")

    metrics = [sales_col, earnings_col, cash_earnings_col, book_value_col]
    ratios: list[np.ndarray] = []
    for col in metrics:
        values = pd.to_numeric(df[col], errors="coerce").fillna(0.0).clip(lower=0.0)
        group_sum = values.groupby(df[date_col]).transform("sum")
        ratio = np.where(group_sum > 0, values / group_sum, 0.0)
        ratios.append(ratio)

    weight_raw = np.mean(np.column_stack(ratios), axis=1)
    df[weight_col] = weight_raw

    group_weight_sum = df.groupby(date_col)[weight_col].transform("sum")
    group_size = df.groupby(date_col)[weight_col].transform("size")
    df[weight_col] = np.where(group_weight_sum > 0, df[weight_col] / group_weight_sum, 1.0 / group_size)

    # Apply strict concentration control by period.
    if max_weight_cap is not None and max_weight_cap < 1.0:
        df[weight_col] = (
            df.groupby(date_col, group_keys=False)[weight_col]
            .apply(
                lambda w: _apply_strict_weight_cap(
                    pd.to_numeric(w, errors="coerce").fillna(0.0),
                    cap=max_weight_cap,
                )
            )
            .astype(float)
        )
    return df


def _apply_strict_weight_cap(weights: pd.Series, *, cap: float) -> pd.Series:
    """Apply iterative cap redistribution while preserving sum-to-one."""
    if cap <= 0:
        raise ValueError("cap must be positive.")

    w = pd.to_numeric(weights, errors="coerce").fillna(0.0).clip(lower=0.0).astype(float)
    n = len(w)
    if n == 0:
        return w

    total = float(w.sum())
    if total <= 0:
        return pd.Series(np.repeat(1.0 / n, n), index=w.index, dtype=float)
    w = w / total

    # If cap is mathematically infeasible for this cross-section, keep normalized weights.
    # Example: 2 names with a 5% cap cannot sum to 100%.
    if n * cap < 1.0 - 1e-12:
        return w

    remaining = w.copy()
    final = pd.Series(0.0, index=w.index, dtype=float)
    remaining_total = 1.0
    eps = 1e-12

    while not remaining.empty and remaining_total > eps:
        rem_sum = float(remaining.sum())
        if rem_sum <= eps:
            final.loc[remaining.index] += remaining_total / len(remaining)
            remaining_total = 0.0
            break

        scaled = remaining / rem_sum * remaining_total
        over = scaled > cap + eps
        if not bool(over.any()):
            final.loc[remaining.index] = scaled
            remaining = remaining.iloc[0:0]
            remaining_total = 0.0
            break

        capped_idx = scaled.index[over]
        final.loc[capped_idx] = cap
        remaining_total -= cap * len(capped_idx)
        remaining = remaining.drop(index=capped_idx)

    if remaining_total > eps:
        remaining_idx = final[final < cap - eps].index
        if len(remaining_idx) > 0:
            final.loc[remaining_idx] += remaining_total / len(remaining_idx)
        else:
            final = final / float(final.sum())

    final = final.clip(lower=0.0)
    final_sum = float(final.sum())
    if final_sum <= 0:
        return pd.Series(np.repeat(1.0 / n, n), index=w.index, dtype=float)
    return final / final_sum


def compute_synthetic_portfolio_returns(
    weighted_df: pd.DataFrame,
    *,
    date_col: str = "date",
    return_col: str = "ret",
    weight_col: str = "weight",
    output_name: str = "Synthetic_ZPRX",
) -> pd.Series:
    """Calculate monthly synthetic portfolio returns from per-name weights."""
    required = {date_col, return_col, weight_col}
    if not required.issubset(weighted_df.columns):
        missing = required - set(weighted_df.columns)
        raise KeyError(f"Missing columns for synthetic return generation: {sorted(missing)}")

    df = weighted_df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df[return_col] = pd.to_numeric(df[return_col], errors="coerce")
    df[weight_col] = pd.to_numeric(df[weight_col], errors="coerce")
    df = df.dropna(subset=[date_col, return_col, weight_col])
    if df.empty:
        raise ValueError("Cannot compute synthetic returns from empty weighted data.")

    df["_weighted_return"] = df[return_col] * df[weight_col]
    series = df.groupby(date_col)["_weighted_return"].sum().sort_index()
    series.name = output_name
    return series


def compute_synthetic_portfolio_returns_with_rebalance_schedule(
    returns_df: pd.DataFrame,
    rebalance_weights_df: pd.DataFrame,
    *,
    id_col: str = "gvkey",
    date_col: str = "date",
    rebalance_date_col: str = "rebalance_date",
    return_col: str = "ret",
    weight_col: str = "weight",
    annual_friction_bps: float = 80.0,
    missing_return_penalty: float = -0.30,
    output_name: str = "Synthetic_ZPRX",
) -> pd.Series:
    """Calculate synthetic returns with lagged weights and survivorship-bias control.

    Rules:
    - Rebalance weights are applied with a one-period lag.
    - If an active name's return is missing for the first time in a rebalance cycle,
      apply `missing_return_penalty` (default -30%).
    - After first missing observation, name enters `dead_stocks` and contributes 0%
      for the remainder of that rebalance cycle (cash parking).
    """
    required_returns = {id_col, date_col, return_col}
    required_weights = {id_col, rebalance_date_col, weight_col}
    if not required_returns.issubset(returns_df.columns):
        missing = required_returns - set(returns_df.columns)
        raise KeyError(f"Missing columns in returns_df: {sorted(missing)}")
    if not required_weights.issubset(rebalance_weights_df.columns):
        missing = required_weights - set(rebalance_weights_df.columns)
        raise KeyError(f"Missing columns in rebalance_weights_df: {sorted(missing)}")
    if annual_friction_bps < 0:
        raise ValueError("annual_friction_bps must be non-negative.")
    if missing_return_penalty < -1.0 or missing_return_penalty > 0.0:
        raise ValueError("missing_return_penalty must be between -1.0 and 0.0.")

    data = returns_df.copy()
    data[date_col] = pd.to_datetime(data[date_col], errors="coerce")
    data[return_col] = pd.to_numeric(data[return_col], errors="coerce")
    data = data.dropna(subset=[id_col, date_col, return_col])
    data = data[np.isfinite(data[return_col])]
    if data.empty:
        raise ValueError("No valid return rows available for synthetic construction.")

    weights = rebalance_weights_df.copy()
    weights[rebalance_date_col] = pd.to_datetime(weights[rebalance_date_col], errors="coerce")
    weights[weight_col] = pd.to_numeric(weights[weight_col], errors="coerce")
    weights = weights.dropna(subset=[id_col, rebalance_date_col, weight_col])
    weights = weights[(weights[weight_col] > 0) & np.isfinite(weights[weight_col])]
    if weights.empty:
        raise ValueError("No valid rebalance weights available for synthetic construction.")

    weight_map: dict[pd.Timestamp, pd.Series] = {}
    for rebal_date, cross in weights.groupby(rebalance_date_col):
        w = cross.groupby(id_col)[weight_col].sum()
        w = w[w > 0]
        total = float(w.sum())
        if total <= 0:
            continue
        weight_map[pd.Timestamp(rebal_date)] = w / total

    if not weight_map:
        raise ValueError("No usable rebalance snapshots after cleaning.")

    available_rebalances = np.array(sorted(weight_map.keys()), dtype="datetime64[ns]")
    monthly_friction = (annual_friction_bps / 10000.0) / 12.0
    results: list[tuple[pd.Timestamp, float]] = []
    last_rebal_idx: int | None = None
    dead_stocks: set[str] = set()
    for dt, cross in data.groupby(date_col):
        # Use strictly prior rebalance snapshot to prevent same-month look-ahead bias.
        rebal_idx = np.searchsorted(available_rebalances, np.datetime64(dt), side="left") - 1
        if rebal_idx < 0:
            continue

        if last_rebal_idx is None or rebal_idx != last_rebal_idx:
            dead_stocks = set()
            last_rebal_idx = rebal_idx

        active_date = pd.Timestamp(available_rebalances[rebal_idx])
        active_weights = weight_map.get(active_date)
        if active_weights is None or active_weights.empty:
            continue

        current_returns = cross.groupby(id_col)[return_col].mean()
        aligned_returns: dict[str, float] = {}
        for asset_id in active_weights.index:
            asset_key = str(asset_id)
            if asset_key in dead_stocks:
                aligned_returns[asset_id] = 0.0
                continue

            observed = current_returns.get(asset_id)
            if observed is None or not math.isfinite(float(observed)):
                aligned_returns[asset_id] = float(missing_return_penalty)
                dead_stocks.add(asset_key)
            else:
                aligned_returns[asset_id] = float(observed)

        r = pd.Series(aligned_returns, index=active_weights.index, dtype=float)
        w = active_weights
        gross_return = float((w * r).sum())
        net_return_raw = gross_return - monthly_friction
        net_return = max(-1.0, net_return_raw)
        if math.isfinite(net_return):
            results.append((pd.Timestamp(dt), net_return))

    if not results:
        raise ValueError("Synthetic return schedule produced no monthly observations.")

    series = pd.Series({d: r for d, r in results}).sort_index()
    series.name = output_name
    return series


def build_synthetic_factor_return_series(
    merged_df: pd.DataFrame,
    *,
    id_col: str = "gvkey",
    date_col: str = "date",
    price_col: str = "price",
    shares_col: str = "shares",
    book_value_col: str = "book_value",
    sales_col: str = "sale",
    earnings_col: str = "earnings",
    cash_earnings_col: str = "cash_earnings",
    return_col: str = "ret",
    rebalance_months: tuple[int, ...] = (5, 11),
    max_weight_cap: float | None = 0.05,
    annual_friction_bps: float = 80.0,
    missing_return_penalty: float = -0.30,
    min_me_floor: float = 250_000_000.0,
    min_price_floor: float = 1.0,
    output_name: str = "Synthetic_ZPRX",
) -> pd.Series:
    """Run end-to-end synthetic factor pipeline with semi-annual rebalance carry-forward.

    `min_me_floor` is interpreted in absolute EUR market-cap units.
    """
    invalid_months = [m for m in rebalance_months if m < 1 or m > 12]
    if invalid_months:
        raise ValueError(f"rebalance_months must be in 1..12, got: {sorted(invalid_months)}")
    if min_me_floor < 0:
        raise ValueError("min_me_floor must be non-negative.")
    if min_price_floor < 0:
        raise ValueError("min_price_floor must be non-negative.")

    if id_col not in merged_df.columns:
        raise KeyError(f"Missing id column for synthetic build: {id_col}")

    enriched = calculate_core_metrics(
        merged_df,
        price_col=price_col,
        shares_col=shares_col,
        book_value_col=book_value_col,
    )
    enriched[date_col] = pd.to_datetime(enriched[date_col], errors="coerce")
    enriched = enriched.dropna(subset=[id_col, date_col, return_col, "me", "bm"])
    if enriched.empty:
        raise ValueError("Merged dataset is empty after metric cleanup.")

    rebalance_universe = enriched[enriched[date_col].dt.month.isin(rebalance_months)].copy()
    # Apply investability screens at rebalance snapshots only.
    rebalance_universe = rebalance_universe[
        (pd.to_numeric(rebalance_universe["me"], errors="coerce") >= min_me_floor)
        & (pd.to_numeric(rebalance_universe[price_col], errors="coerce") >= min_price_floor)
    ]
    if rebalance_universe.empty:
        raise ValueError("No rows found for configured rebalance months.")

    small_value_rebalance = select_small_value_universe(
        rebalance_universe,
        date_col=date_col,
        me_col="me",
        bm_col="bm",
    )
    if small_value_rebalance.empty:
        raise ValueError("No small-value universe rows found at rebalance dates.")

    weighted_rebalance = compute_msci_fundamental_weights(
        small_value_rebalance,
        date_col=date_col,
        sales_col=sales_col,
        earnings_col=earnings_col,
        cash_earnings_col=cash_earnings_col,
        book_value_col=book_value_col,
        max_weight_cap=max_weight_cap,
    )
    weighted_rebalance = weighted_rebalance[[id_col, date_col, "weight"]].rename(
        columns={date_col: "rebalance_date"}
    )

    return compute_synthetic_portfolio_returns_with_rebalance_schedule(
        enriched[[id_col, date_col, return_col]],
        weighted_rebalance,
        id_col=id_col,
        date_col=date_col,
        rebalance_date_col="rebalance_date",
        return_col=return_col,
        weight_col="weight",
        annual_friction_bps=annual_friction_bps,
        missing_return_penalty=missing_return_penalty,
        output_name=output_name,
    )


def calculate_annualized_mu_sigma(returns_df: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    """Return annualized mean vector and covariance matrix from monthly returns."""
    if returns_df.empty:
        raise ValueError("returns_df is empty.")

    numeric_df = returns_df.apply(pd.to_numeric, errors="coerce").dropna(how="all")
    if numeric_df.empty:
        raise ValueError("returns_df contains no numeric data.")

    mu = numeric_df.mean() * 12.0
    sigma = numeric_df.cov() * 12.0
    return mu, sigma


def normalize_weights(
    weights: dict[str, float] | pd.Series | np.ndarray | Iterable[float],
    *,
    asset_names: list[str] | None = None,
) -> pd.Series:
    """Normalize user-provided weights so they sum to 1.0."""
    if isinstance(weights, pd.Series):
        w = weights.astype(float).copy()
    elif isinstance(weights, dict):
        w = pd.Series(weights, dtype=float)
    else:
        values = np.asarray(list(weights), dtype=float)
        if asset_names is None:
            asset_names = [f"asset_{idx}" for idx in range(len(values))]
        if len(values) != len(asset_names):
            raise ValueError("Length mismatch between weights and asset_names.")
        w = pd.Series(values, index=asset_names, dtype=float)

    if (w < 0).any():
        raise ValueError("Weights must be non-negative.")

    total = float(w.sum())
    if total <= 0:
        n = len(w)
        if n == 0:
            raise ValueError("Cannot normalize an empty weight vector.")
        return pd.Series(np.repeat(1.0 / n, n), index=w.index, dtype=float)

    return w / total


def calculate_portfolio_metrics(
    mu: pd.Series,
    sigma: pd.DataFrame,
    weights: dict[str, float] | pd.Series | np.ndarray | Iterable[float],
    *,
    risk_free_rate: float = 0.035,
) -> dict[str, float]:
    """Calculate expected return, volatility, and Sharpe ratio via matrix algebra."""
    if mu.empty:
        raise ValueError("mu vector is empty.")
    if sigma.empty:
        raise ValueError("sigma matrix is empty.")

    if isinstance(weights, np.ndarray) or not isinstance(weights, (dict, pd.Series)):
        w_series = normalize_weights(weights, asset_names=list(mu.index))
    else:
        w_series = normalize_weights(weights).reindex(mu.index).fillna(0.0)
        w_series = normalize_weights(w_series)

    mu_values = mu.to_numpy(dtype=float)
    sigma_aligned = sigma.reindex(index=mu.index, columns=mu.index).fillna(0.0)
    sigma_values = sigma_aligned.to_numpy(dtype=float)
    w = w_series.to_numpy(dtype=float)

    expected_return = float(np.dot(w, mu_values))
    variance = float(np.dot(w, np.dot(sigma_values, w)))
    volatility = float(np.sqrt(max(variance, 0.0)))
    sharpe = float((expected_return - risk_free_rate) / volatility) if volatility > 0 else float("nan")

    return {
        "expected_return": expected_return,
        "volatility": volatility,
        "sharpe_ratio": sharpe,
    }


def _cap_weighted_return(
    df: pd.DataFrame,
    *,
    ret_col: str,
    me_col: str,
) -> float:
    """Compute cap-weighted return for one cross-section."""
    data = df.copy()
    data[ret_col] = pd.to_numeric(data[ret_col], errors="coerce")
    data[me_col] = pd.to_numeric(data[me_col], errors="coerce")
    data = data.dropna(subset=[ret_col, me_col])
    data = data[(data[me_col] > 0) & np.isfinite(data[ret_col])]
    if data.empty:
        return float("nan")

    total_me = float(data[me_col].sum())
    if total_me <= 0:
        return float("nan")
    w = data[me_col] / total_me
    return float((w * data[ret_col]).sum())


def build_wrds_approx_proxy_returns(
    merged_df: pd.DataFrame,
    *,
    date_col: str = "date",
    ret_col: str = "ret",
    price_col: str = "price",
    shares_col: str = "shares",
    fic_col: str = "fic_pricing",
    sales_col: str = "sale",
    earnings_col: str = "earnings",
    cash_earnings_col: str = "cash_earnings",
    book_value_col: str = "book_value",
) -> pd.DataFrame:
    """Build WRDS-only approximate monthly proxies for VWCE, SXRV, JPGL.

    Definitions:
    - VWCE: all-region cap-weighted market return.
    - SXRV: US large-cap cap-weighted return (top 30% by market equity each month).
    - JPGL: quality/value-tilted cap-weighted return (top 30% composite score each month).
    """
    required = {date_col, ret_col, price_col, shares_col}
    if not required.issubset(merged_df.columns):
        missing = required - set(merged_df.columns)
        raise KeyError(f"Missing columns for WRDS proxy construction: {sorted(missing)}")

    df = merged_df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df[ret_col] = pd.to_numeric(df[ret_col], errors="coerce")
    df[price_col] = pd.to_numeric(df[price_col], errors="coerce")
    df[shares_col] = pd.to_numeric(df[shares_col], errors="coerce")
    df = df.dropna(subset=[date_col, ret_col, price_col, shares_col])
    if df.empty:
        raise ValueError("No valid rows available for WRDS proxy construction.")

    df["me_proxy"] = df[price_col] * df[shares_col]
    df = df[(df["me_proxy"] > 0) & np.isfinite(df[ret_col])]
    if df.empty:
        raise ValueError("No positive-ME rows available for WRDS proxy construction.")

    # Inputs for quality/value composite used by JPGL approximation.
    for col in [sales_col, earnings_col, cash_earnings_col, book_value_col]:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["sales_ratio"] = np.where(df["me_proxy"] > 0, df[sales_col].clip(lower=0.0) / df["me_proxy"], np.nan)
    df["earnings_ratio"] = np.where(
        df["me_proxy"] > 0, df[earnings_col].clip(lower=0.0) / df["me_proxy"], np.nan
    )
    df["cash_ratio"] = np.where(
        df["me_proxy"] > 0, df[cash_earnings_col].clip(lower=0.0) / df["me_proxy"], np.nan
    )
    df["book_ratio"] = np.where(
        df["me_proxy"] > 0, df[book_value_col].clip(lower=0.0) / df["me_proxy"], np.nan
    )
    df["quality_score"] = (
        df[["sales_ratio", "earnings_ratio", "cash_ratio", "book_ratio"]]
        .rank(pct=True, method="average")
        .mean(axis=1)
    )

    out_rows: list[dict[str, float | pd.Timestamp]] = []
    for dt, cross in df.groupby(date_col):
        vwce = _cap_weighted_return(cross, ret_col=ret_col, me_col="me_proxy")

        if fic_col in cross.columns:
            us_cross = cross[cross[fic_col].astype(str).str.upper() == "USA"]
        else:
            us_cross = pd.DataFrame()
        if us_cross.empty:
            sxrv_universe = cross
        else:
            cutoff = us_cross["me_proxy"].quantile(0.70)
            sxrv_universe = us_cross[us_cross["me_proxy"] >= cutoff]
            if sxrv_universe.empty:
                sxrv_universe = us_cross
        sxrv = _cap_weighted_return(sxrv_universe, ret_col=ret_col, me_col="me_proxy")

        jpgl_base = cross.dropna(subset=["quality_score"])
        if jpgl_base.empty:
            jpgl_universe = cross
        else:
            cutoff = jpgl_base["quality_score"].quantile(0.70)
            jpgl_universe = jpgl_base[jpgl_base["quality_score"] >= cutoff]
            if jpgl_universe.empty:
                jpgl_universe = jpgl_base
        jpgl = _cap_weighted_return(jpgl_universe, ret_col=ret_col, me_col="me_proxy")

        out_rows.append({"date": dt, "VWCE": vwce, "SXRV": sxrv, "JPGL": jpgl})

    out = pd.DataFrame(out_rows).set_index("date").sort_index()
    out = out.replace([np.inf, -np.inf], np.nan).dropna(how="any")
    if out.empty:
        raise ValueError("Proxy construction produced an empty return frame.")
    return out


def build_brkb_returns_from_secm(
    secm_df: pd.DataFrame,
    *,
    date_col: str = "date",
    ticker_col: str = "tic",
    price_col: str = "price",
) -> pd.Series:
    """Build BRKB monthly return series from Compustat `secm` style data."""
    required = {date_col, ticker_col, price_col}
    if not required.issubset(secm_df.columns):
        missing = required - set(secm_df.columns)
        raise KeyError(f"Missing columns for BRKB construction: {sorted(missing)}")

    df = secm_df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df[ticker_col] = df[ticker_col].astype(str).str.upper()
    df[price_col] = pd.to_numeric(df[price_col], errors="coerce")
    df = df.dropna(subset=[date_col, ticker_col, price_col])
    df = df[df[price_col] > 0]

    if df.empty:
        raise ValueError("No valid rows for BRKB return construction.")

    preferred = ["BRK.B", "BRK-B", "BRKB"]
    chosen = None
    for t in preferred:
        subset = df[df[ticker_col] == t]
        if not subset.empty:
            chosen = subset
            break
    if chosen is None:
        # fallback: highest row count ticker
        counts = df.groupby(ticker_col).size().sort_values(ascending=False)
        chosen = df[df[ticker_col] == counts.index[0]]

    chosen = chosen.sort_values(date_col).drop_duplicates(subset=[date_col], keep="last")
    series = chosen.set_index(date_col)[price_col].pct_change().dropna()
    series.name = "BRKB"
    if series.empty or any(not math.isfinite(v) for v in series.values):
        raise ValueError("BRKB return series is empty or invalid.")
    return series


def build_named_returns_from_secm(
    secm_df: pd.DataFrame,
    *,
    mapping: dict[str, list[str]],
    date_col: str = "date",
    ticker_col: str = "tic",
    price_col: str = "price",
) -> pd.DataFrame:
    """Build monthly return frame from `secm` using asset->ticker preference mapping."""
    required = {date_col, ticker_col, price_col}
    if not required.issubset(secm_df.columns):
        missing = required - set(secm_df.columns)
        raise KeyError(f"Missing columns for ticker return construction: {sorted(missing)}")

    base = secm_df.copy()
    base[date_col] = pd.to_datetime(base[date_col], errors="coerce")
    base[ticker_col] = base[ticker_col].astype(str).str.upper()
    base[price_col] = pd.to_numeric(base[price_col], errors="coerce")
    base = base.dropna(subset=[date_col, ticker_col, price_col])
    base = base[base[price_col] > 0]
    if base.empty:
        raise ValueError("No valid rows in secm data.")

    out: dict[str, pd.Series] = {}
    for asset, preferred_tickers in mapping.items():
        selected: pd.DataFrame | None = None
        for t in preferred_tickers:
            subset = base[base[ticker_col] == t.upper()]
            if not subset.empty:
                selected = subset
                break

        if selected is None:
            raise ValueError(
                f"No rows found for asset {asset} with tickers {preferred_tickers}."
            )

        selected = selected.sort_values(date_col).drop_duplicates(subset=[date_col], keep="last")
        series = selected.set_index(date_col)[price_col].pct_change().dropna()
        if series.empty or any(not math.isfinite(v) for v in series.values):
            raise ValueError(f"Series for asset {asset} is empty or invalid.")
        series.name = asset
        out[asset] = series

    frame = pd.concat(out, axis=1).sort_index()
    frame.columns = [str(c) for c in frame.columns]
    return frame


def build_named_returns_from_crsp(
    crsp_df: pd.DataFrame,
    *,
    mapping: dict[str, list[str]],
    date_col: str = "date",
    ticker_col: str = "ticker",
    return_col: str = "ret",
) -> pd.DataFrame:
    """Build monthly return frame from CRSP using asset->ticker preference mapping."""
    required = {date_col, ticker_col, return_col}
    if not required.issubset(crsp_df.columns):
        missing = required - set(crsp_df.columns)
        raise KeyError(f"Missing columns for CRSP return construction: {sorted(missing)}")

    base = crsp_df.copy()
    base[date_col] = pd.to_datetime(base[date_col], errors="coerce")
    base[ticker_col] = base[ticker_col].astype(str).str.upper()
    base[return_col] = pd.to_numeric(base[return_col], errors="coerce")
    base = base.dropna(subset=[date_col, ticker_col, return_col])
    base = base[np.isfinite(base[return_col])]
    if base.empty:
        raise ValueError("No valid rows in CRSP proxy data.")

    out: dict[str, pd.Series] = {}
    for asset, preferred_tickers in mapping.items():
        stitched: pd.Series | None = None
        for t in preferred_tickers:
            subset = base[base[ticker_col] == t.upper()]
            if subset.empty:
                continue

            # Force a single return per month to prevent multi-share-class double compounding.
            sort_cols = [date_col]
            if "permno" in subset.columns:
                sort_cols.append("permno")
            subset = subset.sort_values(sort_cols).drop_duplicates(subset=[date_col], keep="last")

            monthly_index = subset[date_col].dt.to_period("M")
            ticker_series = (1.0 + subset[return_col]).groupby(monthly_index).prod() - 1.0
            ticker_series.index = ticker_series.index.to_timestamp("M")
            ticker_series = ticker_series.sort_index()
            ticker_series = ticker_series[(ticker_series > -0.95) & (ticker_series < 5.0)]
            if ticker_series.empty:
                continue

            # Preferred ticker order controls splice priority (e.g., VT first, then SPY fallback).
            if stitched is None:
                stitched = ticker_series
            else:
                stitched = stitched.combine_first(ticker_series)

        if stitched is None:
            raise ValueError(
                f"No rows found for asset {asset} with tickers {preferred_tickers}."
            )

        stitched = stitched.sort_index()
        if stitched.empty or any(not math.isfinite(v) for v in stitched.values):
            raise ValueError(f"Series for asset {asset} is empty or invalid.")
        stitched.name = asset
        out[asset] = stitched

    frame = pd.concat(out, axis=1).sort_index()
    frame.columns = [str(c) for c in frame.columns]
    frame = frame.dropna(how="any")
    return frame
