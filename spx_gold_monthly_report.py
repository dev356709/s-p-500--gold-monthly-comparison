#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Streamlit App — S&P 500 (^GSPC) vs Gold (GC=F)
================================================
Real Yahoo Finance data with an interactive monthly comparison.

Features
- Month buttons → one chart per month (S&P red, Gold golden)
- Toggle view: Normalized (100 = first trading day) or Raw prices
- Top‑level monthly returns heatmap (+ CSV download)
- Download current chart (HTML) or export all months (ZIP)
- New polished landing page with instructions + environment check

Run locally
    pip install -U streamlit yfinance pandas plotly numpy
    streamlit run spx_gold_monthly_report.py

Deploy on Streamlit Cloud
    Push this file + requirements.txt (streamlit, yfinance, pandas, plotly, numpy) to GitHub
    Create app on Streamlit Cloud → Main file path: spx_gold_monthly_report.py
"""
from __future__ import annotations

# Plan (pseudocode)
# 1) Sidebar: tickers (^GSPC, GC=F), lookback years, view mode (Normalized/Raw)
# 2) Fetch daily prices (Adj Close → Close fallback) with caching
# 3) Align daily series; build month list (YYYY‑MM)
# 4) Tabs: Overview (instructions + environment check), Heatmap (monthly % returns), Monthly Charts (per-month plots)
# 5) In Monthly Charts: buttons; slice month; normalize if selected; plot S&P red vs Gold golden; summary + downloads

import io
import zipfile
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

# Constants
SNP_DEFAULT = "^GSPC"
GOLD_DEFAULT = "GC=F"
COLOR_SNP = "red"
COLOR_GOLD = "#DAA520"

@dataclass
class AppConfig:
    spx: str = SNP_DEFAULT
    gold: str = GOLD_DEFAULT
    years: int = 2
    view_mode: str = "Normalized"  # or "Raw"

# Cached downloads
@st.cache_data(show_spinner=True)
def fetch_series(ticker: str, period_years: int) -> pd.Series:
    period = f"{period_years}y"
    df = yf.download(ticker, period=period, auto_adjust=False, progress=False, threads=True)
    if df.empty:
        raise ValueError(f"No data for {ticker}")
    col = "Adj Close" if "Adj Close" in df.columns else "Close"
    s = df[col].copy().dropna()
    s.name = ticker
    return s

# Helpers
def align_daily(spx: pd.Series, gold: pd.Series) -> pd.DataFrame:
    return pd.concat([spx, gold], axis=1, join="inner").sort_index()

def list_months(daily: pd.DataFrame) -> List[str]:
    months = daily.index.to_period("M").astype(str).unique().tolist()
    months.sort()
    return months

def normalize_100(df_month: pd.DataFrame) -> pd.DataFrame:
    base = df_month.iloc[0]
    return (df_month / base) * 100.0

def month_summary(df_month_raw: pd.DataFrame) -> pd.DataFrame:
    first = df_month_raw.iloc[0]
    last = df_month_raw.iloc[-1]
    abs_change = last - first
    pct_change = (last / first - 1.0) * 100.0
    return pd.DataFrame({"Start": first.round(4), "End": last.round(4), "Abs Change": abs_change.round(4), "% Change": pct_change.round(2)})

# Heatmap data
def month_end_levels(daily: pd.DataFrame) -> pd.DataFrame:
    return daily.resample("ME").last()

def monthly_returns(levels: pd.DataFrame) -> pd.DataFrame:
    return levels.pct_change().dropna(how="all")

# Plotting
def plot_month(df_month: pd.DataFrame, display_names: Dict[str, str], view_mode: str) -> go.Figure:
    left, right = df_month.columns.tolist()
    y_title = "Index (100 = first trading day)" if view_mode == "Normalized" else "Price"
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_month.index, y=df_month[left], mode="lines+markers", name=display_names.get(left, left), line=dict(color=COLOR_SNP, width=2), marker=dict(size=6)))
    fig.add_trace(go.Scatter(x=df_month.index, y=df_month[right], mode="lines+markers", name=display_names.get(right, right), line=dict(color=COLOR_GOLD, width=2), marker=dict(size=6)))
    month_label = df_month.index[0].strftime("%Y-%m") if len(df_month) else ""
    title_suffix = "Index (100 = first trading day)" if view_mode == "Normalized" else "Raw Prices"
    fig.update_layout(title=f"{month_label}: Daily Movement — {title_suffix}", xaxis_title="Date", yaxis_title=y_title, hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), margin=dict(l=40, r=20, t=60, b=40))
    return fig

def figure_to_png_bytes(fig: go.Figure) -> bytes:
    try:
        import plotly.io as pio
        return pio.to_image(fig, format="png", scale=2)
    except Exception:
        return fig.to_html(full_html=False, include_plotlyjs="cdn").encode("utf-8")

def export_all_months_zip(daily_raw: pd.DataFrame, display_names: Dict[str, str], view_mode: str) -> bytes:
    months = list_months(daily_raw)
    mem = io.BytesIO()
    with zipfile.ZipFile(mem, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for m in months:
            mask = daily_raw.index.to_period("M").astype(str) == m
            dfm_raw = daily_raw.loc[mask]
            if dfm_raw.empty:
                continue
            dfm = normalize_100(dfm_raw) if view_mode == "Normalized" else dfm_raw
            fig = plot_month(dfm, display_names, view_mode)
            payload = figure_to_png_bytes(fig)
            ext = "png" if payload[:4] == b"\x89PNG" else "html"
            zf.writestr(f"monthly_compare_{m}_{view_mode.lower()}.{ext}", payload)
    mem.seek(0)
    return mem.read()

# UI
st.set_page_config(page_title="S&P 500 vs Gold — Monthly", layout="wide")
st.title("Monthly Comparison: ^GSPC → The S&P 500 Index (red) vs GC=F → Gold Futures (gold)")

with st.sidebar:
    st.header("Settings")
    spx = st.text_input("S&P 500 ticker", value=SNP_DEFAULT)
    gold = st.text_input("Gold ticker", value=GOLD_DEFAULT)
    years = st.slider("Lookback years", 1, 10, 2)
    view_mode = st.radio("View mode", ["Normalized", "Raw"], index=0)

# Fetch and align
try:
    s_spx = fetch_series(spx.strip() or SNP_DEFAULT, years)
    s_gold = fetch_series(gold.strip() or GOLD_DEFAULT, years)
except Exception as e:
    st.error(f"Data download failed: {e}")
    st.stop()

daily_raw = align_daily(s_spx, s_gold)
months = list_months(daily_raw)
if not months:
    st.info("No overlapping trading days in the selected period.")
    st.stop()

# Tabs
tab_overview, tab_heatmap, tab_monthly = st.tabs(["Overview & Instructions", "Heatmap", "Monthly Charts"])

# === Overview Tab ===
with tab_overview:
    st.markdown("""
### How to use
1. Pick **tickers**, **lookback years**, and **view mode** in the left sidebar.
2. Go to **Monthly Charts** to click a month and view daily movements.
3. Use **Heatmap** for a quick scan of month‑over‑month returns.

**Legend:** S&P 500 in **red**; Gold in **golden**. Normalized view sets the first trading day of the month to **100** for both series.
""")
    # Environment check
    start_date = daily_raw.index.min().date()
    end_date = daily_raw.index.max().date()
    rows = len(daily_raw)
    col1, col2, col3 = st.columns(3)
    col1.metric("Data Range", f"{start_date} → {end_date}")
    col2.metric("Rows (overlap)", f"{rows:,}")
    col3.metric("Months", f"{len(months)}")

    with st.expander("Notes & Disclaimers"):
        st.write("""
- Data source: Yahoo Finance via `yfinance`. Availability may vary for futures/indices.
- \"GC=F\" is **COMEX Gold Futures**. Consider "GLD" (ETF) or "XAUUSD=X" (spot) if needed.
- Normalized view is for comparability only; use **Raw** to see actual price levels.
- Exports: current chart (HTML) and all months (ZIP of PNG/HTML) are available in **Monthly Charts**.
""")

# === Heatmap Tab ===
with tab_heatmap:
    st.subheader("Monthly Returns Heatmap (MoM, %)")
    levels = month_end_levels(daily_raw)
    rets = monthly_returns(levels) * 100.0
    rets = rets.round(2)
    heat_df = rets.copy()
    heat_df.index = heat_df.index.strftime("%Y-%m")
    heat_df = heat_df.T  # assets as rows
    if heat_df.empty:
        st.info("Not enough month-end data to compute returns.")
    else:
        max_abs = np.nanmax(np.abs(heat_df.to_numpy())) if heat_df.size else 0
        zmin, zmax = -max_abs, max_abs
        fig_heat = px.imshow(
            heat_df,
            color_continuous_scale="RdBu_r",
            zmin=zmin,
            zmax=zmax,
            aspect="auto",
            labels=dict(color="Return %"),
        )
        fig_heat.update_layout(margin=dict(l=40, r=20, t=40, b=20))
        fig_heat.update_yaxes(
            tickmode="array",
            tickvals=list(range(len(heat_df.index))),
            ticktext=[{"^GSPC": "^GSPC → The S&P 500 Index", "GC=F": "GC=F → Gold Futures"}.get(t, t) for t in heat_df.index],
        )
        st.plotly_chart(fig_heat, use_container_width=True)
        csv_bytes = rets.to_csv(index=True).encode("utf-8")
        st.download_button("Download monthly returns CSV", data=csv_bytes, file_name="monthly_returns_percent.csv", mime="text/csv")

# === Monthly Charts Tab ===
with tab_monthly:
    st.subheader("Select a month")
    # Display names for legend
    display_names = {
        s_spx.name: "^GSPC → The S&P 500 Index" if s_spx.name == SNP_DEFAULT else s_spx.name,
        s_gold.name: "GC=F → Gold Futures" if s_gold.name == GOLD_DEFAULT else s_gold.name,
    }

    cols_per_row = 6
    selected = st.session_state.get("selected_month", months[-1])
    for i in range(0, len(months), cols_per_row):
        row = st.columns(cols_per_row)
        for j, m in enumerate(months[i:i+cols_per_row]):
            if row[j].button(m, key=f"btn_{m}"):
                selected = m
                st.session_state["selected_month"] = m

    sel_period = pd.Period(selected, freq="M")
    df_month_raw = daily_raw[daily_raw.index.to_period("M") == sel_period]
    if df_month_raw.empty:
        st.warning(f"No trading days in {selected} for both series.")
        st.stop()

    df_to_plot = normalize_100(df_month_raw) if view_mode == "Normalized" else df_month_raw
    fig = plot_month(df_to_plot, display_names, view_mode)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Monthly Summary (raw prices)**")
    st.dataframe(month_summary(df_month_raw).rename(index=display_names), use_container_width=True)

    col_a, col_b, col_c = st.columns([1, 1, 1])
    with col_a:
        fig_html_bytes = fig.to_html(full_html=False, include_plotlyjs="cdn").encode("utf-8")
        st.download_button(
            label=f"Download {selected} chart ({view_mode}, HTML)",
            data=fig_html_bytes,
            file_name=f"monthly_compare_{selected}_{view_mode.lower()}.html",
            mime="text/html",
        )
    with col_b:
        zip_bytes = export_all_months_zip(daily_raw, display_names, view_mode)
        st.download_button(
            label=f"Export ALL months as ZIP ({view_mode})",
            data=zip_bytes,
            file_name=f"monthly_charts_{view_mode.lower()}.zip",
            mime="application/zip",
        )
    with col_c:
        if st.button("Reset month selection"):
            st.session_state.pop("selected_month", None)
            st.experimental_rerun()

st.caption("Built with Streamlit • Data: Yahoo Finance via yfinance • This app displays public market data for educational use.")
