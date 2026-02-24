# Module 4 Implementation Guide: Factor Engine & Portfolio Optimizer

## Role & Context
You are an expert Quantitative Developer and Financial Engineer. 
We have already built the WRDS data ingestion and anti-look-ahead bias guardrails in `app/utils/data_fetcher.py`. Now, you must complete the core of Module 4 (`app/modules/m4_sandbox.py`) and create a new mathematical utility (`app/utils/factor_engine.py`) to handle Factor Construction and Modern Portfolio Theory (MPT) calculations.

## Objective
Implement the Fama-French independent double sort, the MSCI fundamental weighting algorithm, and an interactive Streamlit UI that calculates real-time Portfolio Expected Return, Volatility, and the Sharpe Ratio using matrix algebra.

## Task 1: Create `app/utils/factor_engine.py`
Write a robust class or set of functions to process the merged DataFrame (pricing data + lagged fundamentals):

1. **Calculate Core Metrics:** * Market Equity: `ME = Price * Shares`
   * Book-to-Market: `B/M = Book Value / ME`
2. **Double Sort (Size & Value):** * Filter the universe to keep only the bottom 30% of stocks by `ME` (Small-Cap).
   * Within that specific sub-universe, keep only the top 30% of stocks by `B/M` ratio (Value).
3. **Fundamental Weighting (MSCI Logic):** * For this resulting "Small-Value" universe, calculate cross-sectional weights for each period.
   * Treat any negative fundamental values (earnings, cash flows) as `0`.
   * Calculate allocation ratios for Sales, Earnings, Cash Earnings, and Book Value (e.g., `W_sales = Sales_i / Sum(Sales)`).
   * The final stock weight `w_i` is the equal-weighted average of these four fundamental ratios.
4. **Synthetic Return Generation:** * Calculate the monthly return of this synthetic factor portfolio: `R_p = Sum(w_i * R_i)`.
   * Return this as a Pandas Series representing the historical monthly returns of our synthetic ZPRX index.

## Task 2: Portfolio Math & Covariance Matrix
In `factor_engine.py`, write a function that takes a DataFrame of historical monthly returns for multiple assets (e.g., our Synthetic ZPRX, plus proxy data for VWCE, SXRV, etc.) and calculates:
* **Annualized Mean Return Vector (mu):** `returns.mean() * 12`
* **Annualized Covariance Matrix (Sigma):** `returns.cov() * 12`

## Task 3: Upgrade `app/modules/m4_sandbox.py` (The Interactive UI)
Remove the current placeholder text and replace it with a fully interactive dashboard:

1. **Data Execution Block:** * Create a button to execute the factor pipeline. 
   * *Fallback:* If WRDS is not actively connected or data is missing, generate a dummy DataFrame with realistic monthly returns so the UI and matrix math can still be developed and tested visually.
2. **The Portfolio Optimizer UI:** * Display a clean correlation heatmap or covariance matrix using Plotly.
   * Create interactive Streamlit sliders for portfolio weights (`w`) for the different assets (e.g., VWCE, Synthetic ZPRX, SXRV). Ensure the UI forces these weights to normalize to 1.0 (100%).
   * Display large, institutional-grade `st.metric` components showing the instantaneous matrix calculation results:
     * **Expected Return:** `w^T * mu`
     * **Volatility (Risk):** `sqrt(w^T * Sigma * w)`
     * **Sharpe Ratio:** `(Expected Return - Risk_Free_Rate) / Volatility` (Assume the Risk-Free Rate is 0.035 for the BOXX ETF yield).

## Constraints & Code Style
* Use strict Python Type Hinting.
* Use `numpy` dot products for all portfolio matrix math (do not use slow loops).
* Use `plotly` for rendering charts and matrices.
* Ensure the code is highly modular, separating the Streamlit UI components from the underlying math engines.