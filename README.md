Sample Python Project: S&P 500 (^GSPC) vs Gold (GC=F) — Monthly Comparison


Overview

This project is an interactive Streamlit app that compares the monthly performance of the S&P 500 Index and Gold Futures over the past two years.
It provides:
Side-by-side monthly performance charts
 Dynamic comparison with colour-coded trends
 Monthly returns heatmap for quick insights
Uses live financial market data (via Yahoo Finance)
The goal is to analyze how equities (S&P 500) and commodities (Gold) behave across different months, helping researchers, analysts, and students study market co-movements.

Installation
To run this project locally, follow the steps below:
Clone the repository
git clone https://github.com/your-username/spx-gold-monthly-report.git
cd spx-gold-monthly-report

2. Create a virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate # On macOS/Linux
.venv\Scripts\activate # On Windows

3. Install the dependencies
pip install -r requirements.txt

4. Run the Streamlit app
streamlit run spx_gold_monthly_report.py

5. Open the link provided in the terminal (usually http://localhost:8501) to view the app in your browser.
This will set up the project and launch the interactive dashboard for comparing the S&P 500 and Gold monthly returns.




Key Features
Interactive Dashboard – select months from a dropdown to compare trends
Performance Indexing (100 = first trading day) – makes S&P 500 and Gold directly comparable
Monthly Returns Heatmap – highlights which months performed better or worse
Live Data – automatically fetches the latest available S&P 500 and Gold prices

Methodology
Data Collection
Fetched 2 years of daily price data for S&P 500 (^GSPC) and Gold Futures (GC=F) using yfinance

Preprocessing
Cleaned and aligned data, converted to monthly returns (% change).
                                                               R_{t} = \frac{P_{t} - P_{t-1}}{P_{t-1}} \times 100
Normalized both series for fair comparison.

Analysis
Built a monthly returns heatmap (years × months).
Created comparative line and bar charts for performance trends.

Visualisation
Implemented with Streamlit (interactive app).
Plots made using Plotly with hover insights and filters.

Outcome
Allows quick comparison of seasonal patterns and correlation between stocks and gold.


Example Output
Monthly comparison chart
Shows S&P 500 (red) vs Gold (golden line).

 Heatmap of returns
Colour-coded to highlight strong and weak months.

Licensing & Ethics
Data: Yahoo Finance via yfinance, for educational use only.
Code: provided as an academic demonstration
This project is licensed under the MIT License; feel free to use, modify, and share with attribution.


Closing note
This sample is not meant to be a full research project, but a demonstration of method and mindset. My aim is to show how I approach operational questions with structured, transparent, and reproducible Python code.

Citation
Please cite as:
Dev Panu. “S&P 500 vs Gold — Monthly Comparison (Real Yahoo Finance Data).” 2025. Code sample for PhD application.
