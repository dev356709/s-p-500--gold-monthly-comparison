#!/usr/bin/env python3
"""
File: spx_gold_monthly_report.py

Title
-----
S&P 500 (^GSPC) vs Gold (GC=F) — Monthly Comparison HTML Report (Real Data)

What this does
--------------
• Downloads **real market data** from Yahoo Finance via `yfinance`.
• Builds a **single HTML file** that contains:
  - A **table of contents** with links for every month in the last N years (default 2).
  - For each month: an **interactive chart** (S&P in red, Gold in golden) showing daily movement
    **normalized to 100 on the first trading day** of that month, and a summary table showing
    start price, end price, and % change for both assets.
• All charts are Plotly. The HTML can be created in **offline mode** (embed JS) or using the CDN.

Quick Start
-----------
1) Install requirements (preferably in a virtualenv):
   pip install --upgrade yfinance pandas plotly

2) Run (defaults: ^GSPC and GC=F, last 2 years):
   python spx_gold_monthly_report.py

3) Open the generated file:
   output/spx_gold_monthly_report.html

Command-line options
--------------------
--spx   Ticker for S&P (default: ^GSPC)
--gold  Ticker for Gold (default: GC=F)
--years Lookback years (default: 2)
--out   Output HTML path (default: output/spx_gold_monthly_report.html)
--offline  If set, embed Plotly JS inside the HTML (bigger file but fully offline).

Pseudocode (Plan)
-----------------
- Parse CLI arguments.
- Compute [start, end] date range from `years`.
- Download Adjusted Close (fallback to Close) for both tickers via yfinance.
- Combine into a single DataFrame of daily prices.
- Build a list of months present in the data.
- For each month:
  * Slice that month’s rows; normalize each series so first trading day = 100.
  * Create an interactive Plotly line chart (S&P red, Gold golden) with markers.
  * Compute a small summary table (Start, End, % Change) for both series.
  * Convert figure to HTML (no <html> wrapper) and store block + a small HTML table.
- Assemble the final HTML document with a TOC linking to each month’s anchor.
- Write the HTML to disk (optionally embed plotly JS for offline viewing).
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import yfinance as yf


#  Configuration 
RED = "red"         # S&P color (as requested)
GOLD = "#DAA520"    # Gold color (goldenrod)


@dataclass
class Config:
    spx_ticker: str = "^GSPC"
    gold_ticker: str = "GC=F"
    years: int = 2
    out_html: Path = Path("output/spx_gold_monthly_report.html")
    offline: bool = False  # if True, embed plotly.js inside HTML

    @property
    def display_names(self) -> Dict[str, str]:
        return {
            self.spx_ticker: "^GSPC → The S&P 500 Index",
            self.gold_ticker: "GC=F → Gold Futures",
        }


#  CLI parsing

def parse_args() -> Config:
    p = argparse.ArgumentParser(description="Create a single HTML report comparing ^GSPC vs GC=F per month.")
    p.add_argument("--spx", default="^GSPC", help="S&P 500 ticker (default ^GSPC)")
    p.add_argument("--gold", default="GC=F", help="Gold ticker: GC=F (futures), GLD (ETF), XAUUSD=X (spot)")
    p.add_argument("--years", type=int, default=2, help="Lookback years (default 2)")
    p.add_argument("--out", default="output/spx_gold_monthly_report.html", help="Output HTML path")
    p.add_argument("--offline", action="store_true", help="Embed plotly.js for fully offline HTML (larger file)")
    ns = p.parse_args()
    return Config(
        spx_ticker=ns.spx,
        gold_ticker=ns.gold,
        years=int(ns.years),
        out_html=Path(ns.out),
        offline=bool(ns.offline),
    )


#  Date & Download 

def compute_date_range(years: int) -> Tuple[pd.Timestamp, pd.Timestamp]:
    today = pd.Timestamp.today().normalize()
    start = today - pd.DateOffset(years=years)
    return start, today


def fetch_adj_close(ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    """Download Adjusted Close; fallback to Close when needed.

    Why: Some instruments (futures/spot) may not expose an Adjusted series.
    """
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False, threads=True)
    if df.empty:
        raise ValueError(f"No data returned for {ticker}")
    col = "Adj Close" if "Adj Close" in df.columns else "Close"
    s = df[col].copy().dropna()
    s.name = ticker
    return s


# Transforms 

def months_in_period(daily: pd.DataFrame) -> List[str]:
    months = daily.index.to_period("M").astype(str).unique().tolist()
    months.sort()
    return months


def normalize_to_100(df_month: pd.DataFrame) -> pd.DataFrame:
    base = df_month.iloc[0]
    return (df_month / base) * 100.0


def month_summary(df_month: pd.DataFrame) -> pd.DataFrame:
    """Build a tidy summary table: Start, End, Abs Change, % Change for each series."""
    first = df_month.iloc[0]
    last = df_month.iloc[-1]
    abs_change = last - first
    pct_change = (last / first - 1.0) * 100.0
    table = pd.DataFrame({
        "Start": first.round(4),
        "End": last.round(4),
        "Abs Change": abs_change.round(4),
        "% Change": pct_change.round(2),
    })
    return table


# Plotly Figures 

def monthly_plot_figure(df_month_norm: pd.DataFrame, cfg: Config) -> go.Figure:
    left_tkr, right_tkr = df_month_norm.columns.tolist()
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df_month_norm.index,
            y=df_month_norm[left_tkr],
            mode="lines+markers",
            name=cfg.display_names.get(left_tkr, left_tkr),
            line=dict(color=RED, width=2),
            marker=dict(size=6),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df_month_norm.index,
            y=df_month_norm[right_tkr],
            mode="lines+markers",
            name=cfg.display_names.get(right_tkr, right_tkr),
            line=dict(color=GOLD, width=2),
            marker=dict(size=6),
        )
    )
    # Titles & axes
    month_label = df_month_norm.index[0].strftime("%Y-%m") if len(df_month_norm) else "(empty)"
    fig.update_layout(
        title=f"{month_label}: Daily Movement — Index (100 = first trading day)",
        xaxis_title="Date",
        yaxis_title="Index (100 = first trading day)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=20, t=60, b=40),
        hovermode="x unified",
    )
    return fig


# HTML Assembly

def build_html_report(cfg: Config, daily: pd.DataFrame) -> str:
    """Return the full HTML for the report as a string."""
    months = months_in_period(daily)

    # TOC entries
    toc_items = "\n".join(
        f'<li><a href="#m-{m}">{m}</a></li>' for m in months
    )

    # Build blocks for each month
    blocks: List[str] = []
    for m in months:
        mask = (daily.index.to_period("M").astype(str) == m)
        dfm = daily.loc[mask]
        if dfm.empty:
            continue
        # Summary table (raw prices)
        summary = month_summary(dfm)
        # Normalized chart (index=100 at first trading day)
        dfm_norm = normalize_to_100(dfm)
        fig = monthly_plot_figure(dfm_norm, cfg)

        # Convert figure to HTML fragment
        include_js = "inline" if cfg.offline else "cdn"
        fig_html = pio.to_html(fig, full_html=False, include_plotlyjs=include_js)

        # Pretty table
        # Rename index to display names
        summary_disp = summary.rename(index=cfg.display_names)
        table_html = summary_disp.to_html(classes="summary-table", border=0)

        block = f"""
        <section id="m-{m}" class="month-block">
          <h2>{m}</h2>
          <div class="chart">{fig_html}</div>
          <div class="table-wrap">
            <h3>Summary</h3>
            {table_html}
          </div>
          <p class="backtotop"><a href="#top">Back to top</a></p>
        </section>
        """
        blocks.append(block)

    # CSS styles (simple, print-friendly)
    css = """
    <style>
      :root { --fg:#111; --muted:#555; --accent:#b00; --gold:#DAA520; }
      body { font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; color: var(--fg); margin: 0 auto; max-width: 1024px; padding: 24px; }
      h1 { margin-top: 0; }
      .meta { color: var(--muted); margin-bottom: 12px; }
      nav.toc { background: #f7f7f7; border: 1px solid #eee; padding: 12px 16px; border-radius: 8px; }
      nav.toc ul { columns: 3; -webkit-columns: 3; -moz-columns: 3; margin: 8px 0 0 20px; }
      nav.toc li { line-height: 1.6; }
      section.month-block { border-top: 1px solid #eee; padding-top: 16px; margin-top: 24px; }
      section.month-block h2 { margin-bottom: 4px; }
      .table-wrap { margin: 6px 0 16px; }
      table.summary-table { border-collapse: collapse; font-size: 14px; }
      table.summary-table th, table.summary-table td { border: 1px solid #ddd; padding: 6px 8px; text-align: right; }
      table.summary-table th:first-child, table.summary-table td:first-child { text-align: left; }
      .legend-note { margin-top: 6px; font-size: 14px; color: var(--muted); }
      .backtotop { margin: 8px 0 0; }
      .badge { display:inline-block; padding:2px 6px; border-radius:6px; font-size:12px; border:1px solid #ddd; }
      .badge.red { color:#b00; border-color:#b00; }
      .badge.gold { color:#B8860B; border-color:#B8860B; }
    </style>
    """

    header = f"""
    <header id="top">
      <h1>S&P 500 vs Gold — Monthly Comparison (Real Yahoo Finance Data)</h1>
      <div class="meta">
        Tickers: <span class="badge red">^GSPC → The S&P 500 Index</span>
        &nbsp;|&nbsp;
        <span class="badge gold">GC=F → Gold Futures</span>
        &nbsp;|&nbsp; Lookback: last {cfg.years} years
      </div>
      <p class="legend-note">Lines are normalized within each month: <strong>Index (100 = first trading day)</strong> for comparability.</p>
      <nav class="toc">
        <strong>Months</strong>
        <ul>
          {toc_items}
        </ul>
      </nav>
    </header>
    """

    html = f"""
    <!doctype html>
    <html lang="en">
    <head>
      <meta charset="utf-8" />
      <meta name="viewport" content="width=device-width, initial-scale=1" />
      <title>S&P 500 vs Gold — Monthly Comparison</title>
      {css}
    </head>
    <body>
      {header}
      {'\n'.join(blocks)}
    </body>
    </html>
    """
    return html


#  Orchestration 

def run(cfg: Config) -> Path:
    start, end = compute_date_range(cfg.years)

    # Download daily prices (real data)
    spx = fetch_adj_close(cfg.spx_ticker, start, end)
    gold = fetch_adj_close(cfg.gold_ticker, start, end)

    # Align by inner join on dates
    daily = pd.concat([spx, gold], axis=1, join="inner").sort_index()

    # Build the HTML document
    cfg.out_html.parent.mkdir(parents=True, exist_ok=True)
    html = build_html_report(cfg, daily)

    # Write to disk
    cfg.out_html.write_text(html, encoding="utf-8")
    return cfg.out_html


def main() -> None:
    cfg = parse_args()
    out_path = run(cfg)
    print(f"\nHTML report created → {out_path.resolve()}\n")


if __name__ == "__main__":
    main()
