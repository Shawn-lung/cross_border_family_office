"""WRDS data-ingestion and anti-look-ahead merge utilities for Module 4."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

try:
    import wrds
except Exception:  # pragma: no cover - optional dependency at runtime
    wrds = None

DEFAULT_WRDS_USERNAME = "shawnlung0429"
REQUIRED_FUNDAMENTAL_LAG_MONTHS = 4
WRDS_CACHE_DIR = Path("data") / "wrds_cache"
DEFAULT_CRSP_PROXY_TICKERS: tuple[str, ...] = (
    "VT",
    "QQQ",
    "QQQQ",
    "PRF",
    "IJS",
    "SPY",
    "BRK.B",
)
DEFAULT_EU_FIC_CODES: tuple[str, ...] = (
    "AUT",
    "BEL",
    "CHE",
    "DEU",
    "DNK",
    "ESP",
    "FIN",
    "FRA",
    "GBR",
    "IRL",
    "ITA",
    "LUX",
    "NLD",
    "NOR",
    "PRT",
    "SWE",
)


class DataFetcherError(RuntimeError):
    """Raised when WRDS connectivity or query execution fails."""


class LookAheadBiasError(ValueError):
    """Raised when fundamentals are merged without the required lag discipline."""


def _normalize_sql(sql: str) -> str:
    """Normalize SQL text for stable cache keys."""
    return " ".join(_sanitize_sql_for_wrds(sql).split()).strip()


def _sanitize_sql_for_wrds(sql: str) -> str:
    """Normalize common copy/paste SQL issues before execution."""
    cleaned = sql
    replacements = {
        "\u2265": ">=",
        "\u2264": "<=",
        "\u2260": "!=",
        "\uff1d": "=",
    }
    for bad, good in replacements.items():
        cleaned = cleaned.replace(bad, good)

    # Architecture draft used comp_global.g_funda; WRDS uses comp.g_funda.
    cleaned = re.sub(r"\bcomp_global\.", "comp.", cleaned, flags=re.IGNORECASE)
    return cleaned


def _wrds_query_cache_key(sql: str, username: str) -> str:
    """Build a deterministic cache key from username + normalized SQL."""
    normalized = _normalize_sql(sql)
    payload = f"{username}::{normalized}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _wrds_cache_paths(
    sql: str,
    username: str,
    cache_dir: str | Path = WRDS_CACHE_DIR,
) -> tuple[Path, Path]:
    """Return `(data_path, metadata_path)` for a cached WRDS query result."""
    cache_root = Path(cache_dir)
    key = _wrds_query_cache_key(sql, username)
    return cache_root / f"{key}.pkl", cache_root / f"{key}.json"


def load_cached_wrds_query_result(
    sql: str,
    username: str = DEFAULT_WRDS_USERNAME,
    cache_dir: str | Path = WRDS_CACHE_DIR,
) -> tuple[pd.DataFrame, dict[str, Any], Path] | None:
    """Load cached query result if present and readable."""
    data_path, meta_path = _wrds_cache_paths(sql, username, cache_dir=cache_dir)
    if not data_path.exists():
        return None

    try:
        df = pd.read_pickle(data_path)
        meta: dict[str, Any] = {}
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        return df, meta, data_path
    except Exception:
        return None


def save_wrds_query_result_to_cache(
    sql: str,
    result_df: pd.DataFrame,
    username: str = DEFAULT_WRDS_USERNAME,
    cache_dir: str | Path = WRDS_CACHE_DIR,
) -> Path:
    """Persist query result locally so future runs can reuse it without re-downloading."""
    data_path, meta_path = _wrds_cache_paths(sql, username, cache_dir=cache_dir)
    data_path.parent.mkdir(parents=True, exist_ok=True)

    result_df.to_pickle(data_path)
    metadata = {
        "username": username,
        "sql_normalized": _normalize_sql(sql),
        "rows": int(len(result_df)),
        "saved_at_utc": datetime.now(UTC).isoformat(),
        "data_file": data_path.name,
    }
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return data_path


def get_wrds_connection(username: str = DEFAULT_WRDS_USERNAME) -> Any:
    """Create and return a WRDS connection using `.pgpass` credentials."""
    if wrds is None:
        raise DataFetcherError("wrds package is not installed in this environment.")

    try:
        return wrds.Connection(wrds_username=username)
    except Exception as exc:  # pragma: no cover - network/credential branch
        raise DataFetcherError(f"Failed to connect to WRDS as {username}: {exc}") from exc


def execute_wrds_query(
    sql: str,
    username: str = DEFAULT_WRDS_USERNAME,
    connection: Any | None = None,
) -> pd.DataFrame:
    """Execute a read-only WRDS SQL query and return a DataFrame."""
    if not isinstance(sql, str) or not sql.strip():
        raise ValueError("SQL query must be a non-empty string.")

    sanitized_sql = _sanitize_sql_for_wrds(sql)

    sql_head = sanitized_sql.lstrip().lower()
    if not (sql_head.startswith("select") or sql_head.startswith("with")):
        raise ValueError("Only read-only SELECT/CTE queries are allowed in execute_wrds_query.")

    owns_connection = connection is None
    conn = connection if connection is not None else get_wrds_connection(username=username)

    try:
        result = conn.raw_sql(sanitized_sql)
        if not isinstance(result, pd.DataFrame):
            raise DataFetcherError("WRDS raw_sql did not return a pandas DataFrame.")
        return result
    except Exception as exc:
        raise DataFetcherError(f"WRDS query failed: {exc}") from exc
    finally:
        if owns_connection and hasattr(conn, "close"):
            try:
                conn.close()
            except Exception:
                pass


def execute_wrds_query_with_cache(
    sql: str,
    username: str = DEFAULT_WRDS_USERNAME,
    *,
    use_cache: bool = True,
    force_refresh: bool = False,
    cache_dir: str | Path = WRDS_CACHE_DIR,
    connection: Any | None = None,
) -> tuple[pd.DataFrame, str, Path]:
    """Execute WRDS SQL with optional local persistent cache.

    Returns:
        `(result_df, source, cache_path)` where source is `"cache"` or `"wrds"`.
    """
    if use_cache and not force_refresh:
        cached = load_cached_wrds_query_result(sql, username=username, cache_dir=cache_dir)
        if cached is not None:
            cached_df, _, cached_path = cached
            return cached_df, "cache", cached_path

    result_df = execute_wrds_query(sql, username=username, connection=connection)

    if use_cache:
        cache_path = save_wrds_query_result_to_cache(
            sql,
            result_df=result_df,
            username=username,
            cache_dir=cache_dir,
        )
        return result_df, "wrds", cache_path

    return result_df, "wrds", Path("")


def apply_fundamental_lag(
    fundamentals_df: pd.DataFrame,
    *,
    fundamental_date_col: str = "datadate",
    lag_months: int = REQUIRED_FUNDAMENTAL_LAG_MONTHS,
) -> pd.DataFrame:
    """Apply publication lag to fundamentals and add `fundamental_available_date`."""
    if lag_months <= 0:
        raise ValueError("lag_months must be positive.")
    if fundamental_date_col not in fundamentals_df.columns:
        raise KeyError(f"Missing required column: {fundamental_date_col}")

    lagged = fundamentals_df.copy()
    lagged[fundamental_date_col] = pd.to_datetime(lagged[fundamental_date_col], errors="coerce")
    if lagged[fundamental_date_col].isna().any():
        raise ValueError("Fundamental dates contain invalid/null values.")

    lagged["fundamental_available_date"] = lagged[fundamental_date_col] + pd.DateOffset(
        months=lag_months
    )
    return lagged


def merge_pricing_with_fundamentals(
    pricing_df: pd.DataFrame,
    fundamentals_lagged_df: pd.DataFrame,
    *,
    id_col: str = "gvkey",
    pricing_date_col: str = "date",
    available_date_col: str = "fundamental_available_date",
) -> pd.DataFrame:
    """Merge pricing with lagged fundamentals using backward as-of matching.

    Raises:
        LookAheadBiasError: if lag column is missing or if any merged row violates the lag rule.
    """
    required_pricing = {id_col, pricing_date_col}
    required_fund = {id_col, available_date_col}

    if not required_pricing.issubset(pricing_df.columns):
        missing = required_pricing - set(pricing_df.columns)
        raise KeyError(f"Missing pricing columns: {sorted(missing)}")
    if available_date_col not in fundamentals_lagged_df.columns:
        raise LookAheadBiasError(
            "Fundamentals must include `fundamental_available_date`. "
            "Run apply_fundamental_lag() before merging."
        )
    if id_col not in fundamentals_lagged_df.columns:
        raise KeyError(f"Missing fundamentals id column: {id_col}")

    pricing = pricing_df.copy()
    fundamentals = fundamentals_lagged_df.copy()
    pricing[pricing_date_col] = pd.to_datetime(pricing[pricing_date_col], errors="coerce")
    fundamentals[available_date_col] = pd.to_datetime(
        fundamentals[available_date_col], errors="coerce"
    )

    if pricing[pricing_date_col].isna().any():
        raise ValueError("Pricing dates contain invalid/null values.")
    if fundamentals[available_date_col].isna().any():
        raise ValueError("Fundamental available dates contain invalid/null values.")

    merged_parts: list[pd.DataFrame] = []
    for asset_id, pricing_group in pricing.groupby(id_col, sort=False):
        p_group = pricing_group.sort_values(pricing_date_col)
        f_group = fundamentals[fundamentals[id_col] == asset_id].sort_values(available_date_col)

        if f_group.empty:
            merged = p_group.copy()
            merged[id_col] = asset_id
            merged_parts.append(merged)
            continue

        f_group = f_group.drop(columns=[id_col], errors="ignore")
        merged = pd.merge_asof(
            p_group,
            f_group,
            left_on=pricing_date_col,
            right_on=available_date_col,
            direction="backward",
            allow_exact_matches=True,
            suffixes=("_pricing", "_fund"),
        )
        merged[id_col] = asset_id
        merged_parts.append(merged)

    merged_df = pd.concat(merged_parts, ignore_index=True)

    if available_date_col in merged_df.columns:
        violations = merged_df[
            merged_df[available_date_col].notna()
            & (merged_df[available_date_col] > merged_df[pricing_date_col])
        ]
        if not violations.empty:
            raise LookAheadBiasError(
                f"Detected {len(violations)} rows with look-ahead bias in merged data."
            )

    return merged_df


def _sql_list_literal(values: Iterable[str]) -> str:
    """Render an iterable of strings as SQL string literals."""
    cleaned = [str(v).strip().upper() for v in values if str(v).strip()]
    unique = sorted(set(cleaned))
    if not unique:
        raise ValueError("At least one SQL literal value is required.")
    return ", ".join(f"'{v}'" for v in unique)


def build_crsp_proxy_query(
    years: int,
    row_limit: int,
    *,
    tickers: Iterable[str] = DEFAULT_CRSP_PROXY_TICKERS,
    brkb_permno: int = 83443,
) -> str:
    """Build CRSP monthly proxy query with BRK.B PERMNO and VT/SPY splice support."""
    if years < 1:
        raise ValueError("years must be >= 1.")
    if row_limit < 1:
        raise ValueError("row_limit must be >= 1.")
    if brkb_permno <= 0:
        raise ValueError("brkb_permno must be positive.")

    cleaned = [str(t).strip().upper() for t in tickers if str(t).strip()]
    brk_ticker_aliases = {"BRK", "BRK.B", "BRK-B", "BRKB"}
    non_brk_tickers = sorted({t for t in cleaned if t not in brk_ticker_aliases})
    ticker_predicate = "FALSE"
    if non_brk_tickers:
        ticker_sql = _sql_list_literal(non_brk_tickers)
        ticker_predicate = f"UPPER(n.ticker) IN ({ticker_sql})"

    return f"""
SELECT m.date,
       CASE
           WHEN m.permno = {int(brkb_permno)} THEN 'BRK.B'
           ELSE UPPER(n.ticker)
       END AS ticker,
       m.permno,
       m.ret
FROM crsp.msf AS m
JOIN crsp.msenames AS n
  ON m.permno = n.permno
 AND m.date BETWEEN n.namedt AND COALESCE(n.nameendt, DATE '9999-12-31')
WHERE m.date >= CURRENT_DATE - INTERVAL '{years + 1} years'
  AND ({ticker_predicate} OR m.permno = {int(brkb_permno)})
  AND m.ret IS NOT NULL
ORDER BY m.date, n.ticker, m.permno
LIMIT {int(row_limit)};
"""


def build_compustat_eu_pricing_query(
    years: int,
    row_limit: int,
    *,
    fic_codes: Iterable[str] = DEFAULT_EU_FIC_CODES,
) -> str:
    """Build monthly Compustat pricing query with EUR conversion and adjusted fields.

    Notes:
    - Uses `comp.secm` monthly fields with `ajexm` + `trt1m`.
    - Converts local-currency price metrics to EUR via `comp.exrt_mth`.
    - `comp.exrt_mth` uses GBP as base (`fromcurm='GBP'`), so conversion is:
      local->EUR = (GBP->EUR) / (GBP->local).
    """
    if years < 1:
        raise ValueError("years must be >= 1.")
    if row_limit < 1:
        raise ValueError("row_limit must be >= 1.")

    fic_sql = _sql_list_literal(fic_codes)
    return f"""
WITH fx AS (
    SELECT datadate, tocurm, exratm
    FROM comp.exrt_mth
    WHERE fromcurm = 'GBP'
),
base AS (
    SELECT s.gvkey,
           s.iid,
           s.datadate AS date,
           UPPER(s.fic) AS fic,
           UPPER(s.curcdm) AS curcd,
           s.prccm AS price_local,
           s.ajexm AS adj_factor,
           s.trt1m,
           COALESCE(s.cshom, s.cshoq) AS shares
    FROM comp.secm AS s
    WHERE s.datadate >= CURRENT_DATE - INTERVAL '{years + 1} years'
      AND UPPER(s.fic) IN ({fic_sql})
      AND s.prccm IS NOT NULL
      AND COALESCE(s.cshom, s.cshoq) IS NOT NULL
)
SELECT b.gvkey,
       b.iid,
       b.date,
       b.fic,
       b.curcd,
       b.price_local,
       b.adj_factor,
       b.trt1m,
       b.shares,
       CASE
           WHEN b.curcd = 'EUR' THEN 1.0
           WHEN fx_local.exratm IS NOT NULL
             AND fx_local.exratm <> 0
             AND fx_eur.exratm IS NOT NULL
               THEN fx_eur.exratm / fx_local.exratm
           ELSE NULL
       END AS fx_to_eur,
       CASE
           WHEN b.curcd = 'EUR' THEN b.price_local
           WHEN fx_local.exratm IS NOT NULL
             AND fx_local.exratm <> 0
             AND fx_eur.exratm IS NOT NULL
               THEN b.price_local * (fx_eur.exratm / fx_local.exratm)
           ELSE NULL
       END AS price_eur,
       CASE
           WHEN b.adj_factor IS NOT NULL AND b.adj_factor <> 0
               THEN b.price_local / b.adj_factor
           ELSE NULL
       END AS price_adj_local,
       CASE
           WHEN b.adj_factor IS NOT NULL
             AND b.adj_factor <> 0
             AND b.curcd = 'EUR'
               THEN (b.price_local / b.adj_factor)
           WHEN b.adj_factor IS NOT NULL
             AND b.adj_factor <> 0
             AND fx_local.exratm IS NOT NULL
             AND fx_local.exratm <> 0
             AND fx_eur.exratm IS NOT NULL
               THEN (b.price_local / b.adj_factor) * (fx_eur.exratm / fx_local.exratm)
           ELSE NULL
       END AS price_adj_eur
FROM base AS b
LEFT JOIN fx AS fx_local
       ON fx_local.datadate = b.date
      AND fx_local.tocurm = b.curcd
LEFT JOIN fx AS fx_eur
       ON fx_eur.datadate = b.date
      AND fx_eur.tocurm = 'EUR'
ORDER BY b.gvkey, b.date
LIMIT {int(row_limit)};
"""


def build_compustat_eu_fundamentals_query(
    years: int,
    row_limit: int,
    *,
    fic_codes: Iterable[str] = DEFAULT_EU_FIC_CODES,
) -> str:
    """Build annual Compustat fundamentals query with EUR-normalized metrics."""
    if years < 1:
        raise ValueError("years must be >= 1.")
    if row_limit < 1:
        raise ValueError("row_limit must be >= 1.")

    fic_sql = _sql_list_literal(fic_codes)
    return f"""
WITH fx AS (
    SELECT datadate, tocurm, exratm
    FROM comp.exrt_mth
    WHERE fromcurm = 'GBP'
),
base AS (
    SELECT f.gvkey,
           f.datadate,
           UPPER(f.fic) AS fic,
           UPPER(f.curcd) AS curcd,
           f.sale,
           f.ib AS earnings,
           f.oancf AS cash_earnings,
           f.ceq AS book_value
    FROM comp.g_funda AS f
    WHERE f.datadate >= CURRENT_DATE - INTERVAL '{years + 2} years'
      AND UPPER(f.fic) IN ({fic_sql})
)
SELECT b.gvkey,
       b.datadate,
       b.fic,
       b.curcd,
       CASE
           WHEN b.curcd = 'EUR' THEN 1.0
           WHEN fx_local.exratm IS NOT NULL
             AND fx_local.exratm <> 0
             AND fx_eur.exratm IS NOT NULL
               THEN fx_eur.exratm / fx_local.exratm
           ELSE NULL
       END AS fx_to_eur,
       CASE
           WHEN b.curcd = 'EUR' THEN b.sale
           WHEN fx_local.exratm IS NOT NULL
             AND fx_local.exratm <> 0
             AND fx_eur.exratm IS NOT NULL
               THEN b.sale * (fx_eur.exratm / fx_local.exratm)
           ELSE NULL
       END AS sale,
       CASE
           WHEN b.curcd = 'EUR' THEN b.earnings
           WHEN fx_local.exratm IS NOT NULL
             AND fx_local.exratm <> 0
             AND fx_eur.exratm IS NOT NULL
               THEN b.earnings * (fx_eur.exratm / fx_local.exratm)
           ELSE NULL
       END AS earnings,
       CASE
           WHEN b.curcd = 'EUR' THEN b.cash_earnings
           WHEN fx_local.exratm IS NOT NULL
             AND fx_local.exratm <> 0
             AND fx_eur.exratm IS NOT NULL
               THEN b.cash_earnings * (fx_eur.exratm / fx_local.exratm)
           ELSE NULL
       END AS cash_earnings,
       CASE
           WHEN b.curcd = 'EUR' THEN b.book_value
           WHEN fx_local.exratm IS NOT NULL
             AND fx_local.exratm <> 0
             AND fx_eur.exratm IS NOT NULL
               THEN b.book_value * (fx_eur.exratm / fx_local.exratm)
           ELSE NULL
       END AS book_value
FROM base AS b
LEFT JOIN fx AS fx_local
       ON fx_local.datadate = b.datadate
      AND fx_local.tocurm = b.curcd
LEFT JOIN fx AS fx_eur
       ON fx_eur.datadate = b.datadate
      AND fx_eur.tocurm = 'EUR'
ORDER BY b.gvkey, b.datadate
LIMIT {int(row_limit)};
"""
