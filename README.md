# s-p-500--gold-monthly-comparison
Data pipeline with yfinance + pandas + Plotly: downloads real Yahoo Finance data and builds per-month S&amp;P 500 vs Gold interactive reports.
Sample Python Project: S&P 500 (^GSPC) vs Gold (GC=F) — Monthly Comparison


Key Features
Automated Data Ingestion – Fetches real historical data from Yahoo Finance (yfinance).
Preprocessing Pipeline – Aligns series, groups by month, and normalises to an index (100 = first trading day).
Interactive Visualisations – Monthly line charts with Plotly (Red = S&P 500, Gold = Gold Futures).
Summary Tables – For each month: start, end, absolute change, and % change.
Single Reproducible Artefact – A consolidated HTML report with embedded charts and navigation.
Each monthly chart normalises both series to the Index (100 = first trading day) to make relative movement directly comparable. A concise summary table (start, end, absolute change, % change) accompanies each month.

Methodology
 Download adjusted closing prices for both tickers.
 Align both series on common trading days.
Group by calendar month.
 Normalise each month to 100 on the first trading day.
Compute changes (absolute and %).
 Render interactive Plotly charts plus tables into a single HTML file.

What’s Included
 : spx_gold_monthly_report.py → main script.
output/spx_gold_monthly_report.html → generated HTML report.
output/ folder → stores artefacts.

Installation & Usage
Requirements: Python 3.9+

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install yfinance pandas plotly

Run:

python spx_gold_monthly_report.py   Output: output/spx_gold_monthly_report.html



Assumptions & Limitations
Gold proxy: defaults to GC=F futures; alternatives include GLD (ETF) or XAUUSD=X (spot).
Adjusted Close is used when available.
Only common trading days are kept (inner join).
Normalisation conveys relative changes, not absolute volatility.

Possible Extensions
Matrix heatmap of monthly returns.
Robustness checks across different gold proxies.
Export CSV/Excel appendices with raw computations.
Add statistical tests (rolling correlations, regression diagnostics).

Licensing & Ethics
Data: Yahoo Finance via yfinance, for educational use only.
Code: provided as an academic demonstration


Closing note
This sample is not meant to be a full research project, but a demonstration of method and mindset. My aim is to show how I approach operational questions with structured, transparent, and reproducible Python code.

Citation
Please cite as:
Dev Panu. “S&P 500 vs Gold — Monthly Comparison (Real Yahoo Finance Data).” 2025. Code sample for PhD application.




