# Project Cross-Border Family Office - System Architecture

## 1. Role & Objective
You are an elite Quantitative Python Developer and Data Architect building an institutional-grade "Family Office Wealth Management Dashboard." The goal is to build a highly modular, visually clean, interactive web application using **Streamlit**. This dashboard will be used in live presentations to family members to visually prove financial safety, tax optimization, and portfolio growth. 

**CRITICAL DIRECTIVE:** The project is not considered "done" until the core mathematical and logical functions pass a comprehensive suite of automated unit tests. Test-Driven Development (TDD) is mandatory.

## 2. Tech Stack
* **Frontend:** `streamlit`, `plotly` (for interactive, institutional-grade presentation charts)
* **Data Manipulation:** `pandas`, `numpy`, `scipy.stats`
* **Data Ingestion:** `wrds` (Wharton Research Data Services for CRSP/Compustat), `yfinance` (for live price and forex fallback)
* **Storage:** `sqlite3` (local state management for portfolio weights and cash flows)
* **Testing:** `pytest`, `unittest.mock`

## 3. Core System Rules: The Forex Baseline
* **First Execution Step:** The system must fetch live exchange rates (EUR/USD, USD/TWD, EUR/TWD) via `yfinance`. 
* All user inputs (TWD/EUR) must be dynamically translated using these live rates so the entire dashboard shares a synchronized, mathematically precise currency baseline before any module executes its logic.

## 4. Core Application Modules

### Module 1: The ALM & Tax Simulator (The Trust Engine)
**Purpose:** Asset-Liability Matching and Dutch Box 3 Tax projection for student cash flows.
* **Inputs:** Starting Capital (Default: 2.7M TWD), One-Time Tuition Cliff (Default: €24,900 for the 1-year master's program), Monthly Living Expenses (€1,500), Risk-Free Yield Rate (Default: `BOXX` at 3.5%), Monthly Additional Contribution (TWD/EUR toggle).
* **Logic:** Calculate the month-by-month drawdown of capital. The tuition must be deducted as a single, massive drop in the designated month, rather than smoothed out.
* **Tax Boundary Line:** Plot a hard horizontal line at the Dutch Box 3 individual exemption limit (€57,684). Add backend logic to increase this limit by a projected inflation rate (+1.5% per year) to reflect moving tax brackets.
* **Output:** A Plotly step-chart visually proving the Net Asset Value (NAV) drops below the Box 3 threshold strictly before January 1st of the assessment year, resulting in a calculated "€0.00 Tax Owed."

### Module 2: Fama-French Factor Engine (The Core)
**Purpose:** Live tracking of a Fama-French 5-Factor equity portfolio.
* **Target Weights:** 40% `VWCE`, 15% `SXRV`, 15% `ZPRV`, 5% `ZPRX`, 15% `BRKB`, 10% `JPGL`.
* **Inputs:** Initial Lump Sum AND Monthly Contribution amount.
* **Live X-Ray:** Ingest live prices, calculate current portfolio weights, and display the "Drift" (Target Weight minus Current Weight).
* **Alert System (The Swedroe 5/25 Rule):** Trigger a visual "REBALANCE REQUIRED" alert ONLY IF:
    1. Absolute Drift > 5% (e.g., a 15% target becomes 20.1%).
    2. Relative Drift > 25% of its specific target weight (e.g., the 5% `ZPRX` target drops below 3.75% or exceeds 6.25%).

### Module 3: Monte Carlo Retirement Simulator (Extended Family)
**Purpose:** Interactive anxiety-reduction tool calculating survival probabilities.
* **Inputs:** Initial Portfolio Value, Monthly Contribution/Withdrawal Amount, Expected Inflation Rate (%), Target Simulation Years (e.g., 20-30 years).
* **Math:** Use Geometric Brownian Motion (GBM) to simulate 10,000 possible market paths. 
* **Output:** A visual distribution fan chart of the portfolio's future value.
* **Key Metric (The Benchmark Test):** Calculate and display the **"Probability of Outperformance."** Define success as: The percentage of the 10,000 simulations that result in an ending balance strictly greater than if the initial capital + contributions were left in a standard Taiwan bank account (1.5% yield) or `BOXX` (3.5% yield).

### Module 4: The Alpha Sandbox (Quantitative Research)
**Purpose:** A dedicated research environment for building and backtesting synthetic factor portfolios using institutional databases.
* **Phase 1: WRDS Database Connection**
  * Initialize the WRDS connection using the username: `shawnlung0429`.
  * Ensure the script handles the `.pgpass` credential file seamlessly.
* **Phase 2: Data Extraction (Compustat Global)**
  * Write the SQL queries via the `db.raw_sql()` method to pull data for European equities over the last 15 years.
  * *Pricing Data (Market Data):* Extract monthly prices, shares outstanding, and monthly returns to calculate Market Equity (ME). 
  * *Fundamental Data:* Extract annual/quarterly accounting data: Book Value, Sales, Earnings, and Cash Earnings (Operating Cash Flow).
* **Phase 3: Data Cleaning & Look-Ahead Bias Prevention**
  * Merge the pricing and fundamental datasets.
  * **Crucial Constraint:** Implement a strict 4-month lag (using `shift()` or date offset) on all fundamental data before merging with pricing data to absolutely prevent look-ahead bias. 
* **Phase 4: Factor Construction & Fundamental Weighting Strategy**
  * Implement the cross-sectional independent double sort and fundamental weighting logic:
    1.  **Universe Filter (Size):** Filter the cross-section to keep only the bottom 30% of companies by Market Equity (Small-Cap).
    2.  **Value Filter (B/M):** Calculate Book-to-Market ($B/M = \frac{Book\ Equity}{Market\ Equity}$). Keep the top 30% of stocks with the highest B/M ratio (Value).
    3.  **Fundamental Weighting Calculation:** For the remaining "Small-Value" universe, calculate the allocation weight $w_i$ for each stock $i$ using the MSCI fundamental weighting methodology. Calculate the cross-sectional ratio of each fundamental metric, substituting $0$ for any negative flow values:
        * $W_{i, sales} = \frac{Sales_i}{\sum Sales}$ (using 3-year trailing average)
        * $W_{i, earnings} = \frac{Earnings_i}{\sum Earnings}$ (using 3-year trailing average)
        * $W_{i, cashearnings} = \frac{CashEarnings_i}{\sum CashEarnings}$ (using 3-year trailing average)
        * $W_{i, bookvalue} = \frac{BookValue_i}{\sum BookValue}$ (using latest available)
        * **Final Weight:** $w_i = \frac{W_{i, sales} + W_{i, earnings} + W_{i, cashearnings} + W_{i, bookvalue}}{4}$
* **Phase 5: Rebalancing & Output**
  * Implement a semi-annual rebalancing schedule strictly in **May** and **November**.
  * Calculate the synthetic portfolio's monthly return time series: $R_{portfolio} = \sum w_i R_i$.
  * Output the annualized Expected Return ($\mu$) and annualized Standard Deviation ($\sigma$) of the portfolio, formatting it cleanly so it can be passed into a covariance matrix for the Streamlit UI.

### Module 5: Testing & Validation
**Purpose:** Mathematically prove the system is reliable before deployment.
* Generate a `tests/` directory utilizing `pytest`.
* **Mocking:** Use `unittest.mock` to mock `yfinance` and `wrds` API calls so tests run instantaneously without network dependencies.
* **Required Test Cases:**
    1. *Test the 5/25 Rule:* Input a dummy portfolio with 4.9% absolute drift (should return False) and 5.1% absolute drift (should return True). Test the 25% relative drift boundary identically.
    2. *Test the Tax Boundary:* Assert that a January 1st NAV of €57,683 triggers €0 tax, and €57,685 triggers the 36% Box 3 wealth tax calculation on the overage.
    3. *Test Forex Triangulation:* Ensure 1000 EUR converts to TWD and back to EUR with less than a 0.01% floating-point precision error.
    4. *Test Look-Ahead Bias:* Ensure Module 4 raises an error or fails if fundamental data is merged with pricing data without the required 4-month lag.

## 5. Design & UI Rules
* The UI must look like an institutional dashboard: clean, data-dense, minimalist, dark mode.
* It must be highly interactive (sliders for inputs) for live "What-If" scenario planning.
* Write modular, heavily commented Python code with strict type hinting.

## 6. Directory Structure
Generate the following boilerplate folder structure, `requirements.txt`, and empty Python files before writing the core logic:

cross_border_family_office/
│
├── ARCHITECTURE.md          # This system prompt
├── requirements.txt         # streamlit, pandas, yfinance, wrds, pytest, plotly, scipy
├── .gitignore               # Ignore .db files, .pgpass, __pycache__, and venv
│
├── app/                     # The main Streamlit application
│   ├── main.py              # The entry point (sidebar navigation menu)
│   │
│   ├── modules/             # The 4 core UI screens
│   │   ├── __init__.py
│   │   ├── m1_alm_tax.py    # The Trust Engine (Cashflow/Box 3)
│   │   ├── m2_factors.py    # The Core (Fama-French & Swedroe 5/25 rule)
│   │   ├── m3_retirement.py # Extended Family Monte Carlo
│   │   └── m4_sandbox.py    # The Alpha Sandbox (WRDS integration)
│   │
│   └── utils/               # The underlying mathematical engines
│       ├── __init__.py
│       ├── forex_engine.py  # yfinance triangulation (EUR/TWD/USD)
│       ├── tax_logic.py     # Hardcoded Dutch Box 3 rules & exemption limits
│       └── data_fetcher.py  # WRDS SQL connections and caching
│
├── data/                    # Local storage
│   └── portfolio_state.db   # SQLite database for weights and cashflows
│
└── tests/                   # TDD Validation Suite
    ├── __init__.py
    ├── test_forex.py        # Proves currency triangulation is accurate
    ├── test_tax_logic.py    # Proves the €57,684 threshold triggers correctly
    ├── test_swedroe.py      # Proves the 5/25 rebalancing alert works
    └── test_monte_carlo.py  # Proves the GBM math outputs logical distributions