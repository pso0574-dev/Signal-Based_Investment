# app.py
import os
import math
import requests
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.express as px
from datetime import datetime, timedelta

# =========================================================
# Page config
# =========================================================
st.set_page_config(
    page_title="Noise vs Signal Dashboard",
    page_icon="📊",
    layout="wide"
)

# =========================================================
# Constants
# =========================================================
FRED_API_KEY = os.getenv("FRED_API_KEY", "").strip()
FRED_BASE_URL = "https://api.stlouisfed.org/fred/series/observations"

# Market assets (Yahoo Finance)
MARKET_ASSETS = [
    {"group": "Equity", "name": "S&P 500 ETF", "symbol": "SPY", "type": "price"},
    {"group": "Equity", "name": "Nasdaq 100 ETF", "symbol": "QQQ", "type": "price"},
    {"group": "Equity", "name": "Dow Jones ETF", "symbol": "DIA", "type": "price"},
    {"group": "Rates Proxy", "name": "20Y Treasury ETF", "symbol": "TLT", "type": "price"},
    {"group": "Rates Proxy", "name": "7-10Y Treasury ETF", "symbol": "IEF", "type": "price"},
    {"group": "Inflation / Commodity", "name": "Gold ETF", "symbol": "GLD", "type": "price"},
    {"group": "Inflation / Commodity", "name": "Oil ETF", "symbol": "USO", "type": "price"},
    {"group": "Inflation / Commodity", "name": "Broad Commodity ETF", "symbol": "DBC", "type": "price"},
    {"group": "FX / Dollar", "name": "US Dollar Index ETF", "symbol": "UUP", "type": "price"},
    {"group": "Risk Asset", "name": "Bitcoin USD", "symbol": "BTC-USD", "type": "price"},
]

# FRED series
FRED_SERIES = [
    {"group": "Rates", "name": "US 10Y Treasury Yield", "symbol": "DGS10", "type": "yield"},
    {"group": "Rates", "name": "US 2Y Treasury Yield", "symbol": "DGS2", "type": "yield"},
    {"group": "Rates", "name": "10Y-2Y Spread", "symbol": "T10Y2Y", "type": "spread"},
    {"group": "Rates", "name": "10Y-3M Spread", "symbol": "T10Y3M", "type": "spread"},
    {"group": "Real Yield", "name": "10Y Real Yield", "symbol": "DFII10", "type": "yield"},
    {"group": "Inflation", "name": "Breakeven Inflation 10Y", "symbol": "T10YIE", "type": "yield"},
    {"group": "Labor", "name": "Unemployment Rate", "symbol": "UNRATE", "type": "macro"},
    {"group": "Consumption", "name": "Retail Sales", "symbol": "RSAFS", "type": "macro"},
    {"group": "Liquidity", "name": "M2 Money Supply", "symbol": "M2SL", "type": "macro"},
    {"group": "Liquidity", "name": "Fed Balance Sheet", "symbol": "WALCL", "type": "macro"},
]

# Dashboard groups for first tab
GROUP_ORDER = [
    "Equity",
    "Rates",
    "Rates Proxy",
    "Real Yield",
    "Inflation",
    "Inflation / Commodity",
    "Liquidity",
    "FX / Dollar",
    "Risk Asset",
    "Labor",
    "Consumption",
]

# =========================================================
# Helpers
# =========================================================
def fmt_pct(x):
    if pd.isna(x):
        return "N/A"
    return f"{x:+.2f}%"

def fmt_abs(x):
    if pd.isna(x):
        return "N/A"
    return f"{x:,.2f}"

def fmt_bp_like(x):
    if pd.isna(x):
        return "N/A"
    return f"{x:+.2f}"

def color_signal(val: str):
    if not isinstance(val, str):
        return ""
    low = val.lower()
    if any(k in low for k in ["risk-on", "positive", "strong", "bull", "improving", "easing", "healthy"]):
        return "background-color: rgba(0, 180, 0, 0.15); color: #116611;"
    if any(k in low for k in ["risk-off", "negative", "weak", "bear", "stress", "tightening", "deteriorating"]):
        return "background-color: rgba(220, 0, 0, 0.15); color: #991111;"
    return "background-color: rgba(180, 180, 0, 0.12); color: #7a5f00;"

def add_change_columns(series: pd.Series):
    """
    Returns latest level and changes for 1D / 1W / 1M / 1Y.
    For daily market prices: percentage change.
    For yields/macro levels: absolute change.
    """
    s = series.dropna().copy()
    if s.empty or len(s) < 3:
        return {
            "latest": np.nan,
            "1D": np.nan,
            "1W": np.nan,
            "1M": np.nan,
            "1Y": np.nan
        }

    latest = s.iloc[-1]

    def safe_shift(periods):
        if len(s) <= periods:
            return np.nan
        return s.iloc[-(periods + 1)]

    return {
        "latest": latest,
        "1D_base": safe_shift(1),
        "1W_base": safe_shift(5),
        "1M_base": safe_shift(21),
        "1Y_base": safe_shift(252),
    }

def calc_pct_change(latest, base):
    if pd.isna(latest) or pd.isna(base) or base == 0:
        return np.nan
    return (latest / base - 1.0) * 100.0

def calc_abs_change(latest, base):
    if pd.isna(latest) or pd.isna(base):
        return np.nan
    return latest - base

def infer_signal(row):
    name = row["Asset"]
    group = row["Group"]
    c1m = row["1M_num"]
    c1y = row["1Y_num"]
    latest = row["Latest_num"]

    if group == "Equity":
        if pd.notna(c1m) and pd.notna(c1y):
            if c1m > 0 and c1y > 0:
                return "Risk-On"
            if c1m < 0 and c1y < 0:
                return "Risk-Off"
        return "Neutral"

    if group == "Risk Asset":
        if pd.notna(c1m) and c1m > 0:
            return "Risk-On"
        if pd.notna(c1m) and c1m < 0:
            return "Risk-Off"
        return "Neutral"

    if group == "FX / Dollar":
        if pd.notna(c1m) and c1m > 0:
            return "USD Strong"
        if pd.notna(c1m) and c1m < 0:
            return "USD Weak"
        return "Neutral"

    if group in ["Rates", "Real Yield"]:
        if "Spread" in name:
            if pd.notna(latest):
                if latest < 0:
                    return "Curve Inverted"
                if latest > 0.5:
                    return "Curve Normal"
            return "Neutral"
        else:
            if pd.notna(c1m):
                if c1m > 0:
                    return "Tightening"
                if c1m < 0:
                    return "Easing"
            return "Neutral"

    if group in ["Inflation", "Inflation / Commodity"]:
        if pd.notna(c1m):
            if c1m > 0:
                return "Inflation Rising"
            if c1m < 0:
                return "Inflation Cooling"
        return "Neutral"

    if group == "Liquidity":
        if pd.notna(c1y):
            if c1y > 0:
                return "Liquidity Positive"
            if c1y < 0:
                return "Liquidity Negative"
        return "Neutral"

    if group == "Labor":
        if pd.notna(c1y):
            if c1y < 0:
                return "Labor Healthy"
            if c1y > 0:
                return "Labor Weakening"
        return "Neutral"

    if group == "Consumption":
        if pd.notna(c1y):
            if c1y > 0:
                return "Demand Positive"
            if c1y < 0:
                return "Demand Weakening"
        return "Neutral"

    return "Neutral"

def build_group_summary(df):
    rows = []
    for grp in GROUP_ORDER:
        g = df[df["Group"] == grp].copy()
        if g.empty:
            continue

        signals = g["Signal"].dropna().tolist()
        positive_cnt = sum(any(k in s.lower() for k in [
            "risk-on", "positive", "strong", "normal", "healthy", "easing", "cooling"
        ]) for s in signals)
        negative_cnt = sum(any(k in s.lower() for k in [
            "risk-off", "negative", "weak", "inverted", "tightening", "rising", "weakening"
        ]) for s in signals)

        if positive_cnt > negative_cnt:
            summary = "Positive"
        elif negative_cnt > positive_cnt:
            summary = "Negative"
        else:
            summary = "Neutral"

        rows.append({
            "Group": grp,
            "Assets": len(g),
            "1D Avg": g["1D_num"].mean(skipna=True),
            "1W Avg": g["1W_num"].mean(skipna=True),
            "1M Avg": g["1M_num"].mean(skipna=True),
            "1Y Avg": g["1Y_num"].mean(skipna=True),
            "Group Signal": summary
        })
    return pd.DataFrame(rows)

# =========================================================
# Data loaders
# =========================================================
@st.cache_data(ttl=60 * 30)
def get_yf_history(symbols, period="2y"):
    if not symbols:
        return pd.DataFrame()

    df = yf.download(
        tickers=symbols,
        period=period,
        interval="1d",
        auto_adjust=True,
        progress=False,
        group_by="ticker",
        threads=True
    )

    if df.empty:
        return pd.DataFrame()

    closes = {}

    # Single ticker case
    if isinstance(df.columns, pd.Index):
        closes[symbols[0]] = df["Close"]
        result = pd.DataFrame(closes)
        result.index = pd.to_datetime(result.index)
        return result.sort_index()

    # Multi-ticker case
    for sym in symbols:
        try:
            closes[sym] = df[sym]["Close"]
        except Exception:
            continue

    result = pd.DataFrame(closes)
    result.index = pd.to_datetime(result.index)
    return result.sort_index()

@st.cache_data(ttl=60 * 60)
def get_fred_series(series_id, observation_start="2020-01-01"):
    if not FRED_API_KEY:
        raise ValueError("FRED_API_KEY is missing. Please set it as an environment variable.")

    params = {
        "series_id": series_id,
        "api_key": FRED_API_KEY,
        "file_type": "json",
        "observation_start": observation_start,
    }
    resp = requests.get(FRED_BASE_URL, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    observations = data.get("observations", [])
    rows = []
    for item in observations:
        value = item.get("value")
        if value in (None, ".", ""):
            val = np.nan
        else:
            try:
                val = float(value)
            except Exception:
                val = np.nan
        rows.append((pd.to_datetime(item["date"]), val))

    s = pd.Series(
        data=[v for _, v in rows],
        index=[d for d, _ in rows],
        name=series_id
    ).sort_index()

    return s

@st.cache_data(ttl=60 * 60)
def load_all_fred(series_meta):
    out = {}
    for item in series_meta:
        sid = item["symbol"]
        try:
            out[sid] = get_fred_series(sid, observation_start="2020-01-01")
        except Exception:
            out[sid] = pd.Series(dtype=float, name=sid)
    return out

# =========================================================
# Transform
# =========================================================
def build_market_table(yf_hist, fred_hist, selected_groups):
    rows = []

    # Market assets
    for item in MARKET_ASSETS:
        if selected_groups and item["group"] not in selected_groups:
            continue

        sym = item["symbol"]
        if sym not in yf_hist.columns:
            continue

        s = yf_hist[sym].dropna()
        stat = add_change_columns(s)

        latest = stat["latest"]
        row = {
            "Group": item["group"],
            "Asset": item["name"],
            "Ticker": sym,
            "Latest_num": latest,
            "1D_num": calc_pct_change(latest, stat["1D_base"]),
            "1W_num": calc_pct_change(latest, stat["1W_base"]),
            "1M_num": calc_pct_change(latest, stat["1M_base"]),
            "1Y_num": calc_pct_change(latest, stat["1Y_base"]),
            "Unit": "pct"
        }
        rows.append(row)

    # FRED assets
    fred_meta_map = {x["symbol"]: x for x in FRED_SERIES}

    for sid, s in fred_hist.items():
        meta = fred_meta_map[sid]
        if selected_groups and meta["group"] not in selected_groups:
            continue

        s = s.dropna()
        if s.empty:
            continue

        # Convert lower-frequency macro series to daily forward-filled
        daily = s.resample("D").ffill()

        stat = add_change_columns(daily)
        latest = stat["latest"]

        row = {
            "Group": meta["group"],
            "Asset": meta["name"],
            "Ticker": sid,
            "Latest_num": latest,
            "1D_num": calc_abs_change(latest, stat["1D_base"]),
            "1W_num": calc_abs_change(latest, stat["1W_base"]),
            "1M_num": calc_abs_change(latest, stat["1M_base"]),
            "1Y_num": calc_abs_change(latest, stat["1Y_base"]),
            "Unit": "abs"
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df["Signal"] = df.apply(infer_signal, axis=1)

    # Display formatting
    def fmt_latest(row):
        if row["Unit"] == "pct":
            return fmt_abs(row["Latest_num"])
        return fmt_abs(row["Latest_num"])

    def fmt_change(row, col):
        if row["Unit"] == "pct":
            return fmt_pct(row[col])
        return fmt_bp_like(row[col])

    df["Latest"] = df.apply(fmt_latest, axis=1)
    df["1D"] = df.apply(lambda r: fmt_change(r, "1D_num"), axis=1)
    df["1W"] = df.apply(lambda r: fmt_change(r, "1W_num"), axis=1)
    df["1M"] = df.apply(lambda r: fmt_change(r, "1M_num"), axis=1)
    df["1Y"] = df.apply(lambda r: fmt_change(r, "1Y_num"), axis=1)

    display_cols = ["Group", "Asset", "Ticker", "Latest", "1D", "1W", "1M", "1Y", "Signal"]
    df = df[display_cols + ["Latest_num", "1D_num", "1W_num", "1M_num", "1Y_num"]]

    return df

def style_market_table(df):
    display_df = df[["Group", "Asset", "Ticker", "Latest", "1D", "1W", "1M", "1Y", "Signal"]].copy()
    return (
        display_df.style
        .map(color_signal, subset=["Signal"])
    )

# =========================================================
# UI
# =========================================================
st.title("📊 Noise vs Signal Dashboard")
st.caption("Long-term investing dashboard focused on signal, not daily noise.")

with st.sidebar:
    st.header("Settings")

    if not FRED_API_KEY:
        st.error("FRED_API_KEY is not set.")
    else:
        st.success("FRED API key loaded.")

    selected_groups = st.multiselect(
        "Select groups",
        options=GROUP_ORDER,
        default=["Equity", "Rates", "Inflation / Commodity", "Liquidity", "FX / Dollar", "Risk Asset"]
    )

    chart_options = [x["symbol"] for x in MARKET_ASSETS] + [x["symbol"] for x in FRED_SERIES]
    default_chart = ["SPY", "QQQ", "DGS10", "T10Y2Y", "DFII10", "WALCL"]
    selected_chart_symbols = st.multiselect(
        "Chart series",
        options=chart_options,
        default=default_chart
    )

    chart_lookback = st.selectbox(
        "Chart lookback",
        options=["6M", "1Y", "2Y"],
        index=1
    )

# =========================================================
# Load data
# =========================================================
yf_symbols = [x["symbol"] for x in MARKET_ASSETS]
fred_data = {}
yf_hist = pd.DataFrame()

try:
    yf_hist = get_yf_history(yf_symbols, period="2y")
except Exception as e:
    st.error(f"Failed to load Yahoo Finance data: {e}")

try:
    fred_data = load_all_fred(FRED_SERIES)
except Exception as e:
    st.error(f"Failed to load FRED data: {e}")
    fred_data = {}

market_df = build_market_table(yf_hist, fred_data, selected_groups)

# =========================================================
# Tabs
# =========================================================
tab1, tab2, tab3 = st.tabs([
    "1) Market Snapshot",
    "2) Signal Charts",
    "3) Raw Data / Notes"
])

# =========================================================
# TAB 1
# =========================================================
with tab1:
    st.subheader("Global Snapshot — 1D / 1W / 1M / 1Y Change")

    if market_df.empty:
        st.warning("No data available.")
    else:
        c1, c2, c3, c4 = st.columns(4)

        eq_count = int((market_df["Group"] == "Equity").sum())
        rates_count = int((market_df["Group"] == "Rates").sum())
        infl_count = int((market_df["Group"].isin(["Inflation", "Inflation / Commodity"])).sum())
        liq_count = int((market_df["Group"] == "Liquidity").sum())

        c1.metric("Equity Series", eq_count)
        c2.metric("Rates Series", rates_count)
        c3.metric("Inflation Series", infl_count)
        c4.metric("Liquidity Series", liq_count)

        st.dataframe(
            style_market_table(market_df),
            use_container_width=True,
            hide_index=True
        )

        st.markdown("---")
        st.subheader("Group Summary")

        group_summary = build_group_summary(market_df)

        if not group_summary.empty:
            # format group summary
            for col in ["1D Avg", "1W Avg", "1M Avg", "1Y Avg"]:
                group_summary[col] = group_summary[col].map(lambda x: "N/A" if pd.isna(x) else f"{x:+.2f}")

            st.dataframe(
                group_summary.style.map(color_signal, subset=["Group Signal"]),
                use_container_width=True,
                hide_index=True
            )

        st.markdown("---")
        st.subheader("Grouped Tables")

        for grp in GROUP_ORDER:
            sub = market_df[market_df["Group"] == grp].copy()
            if sub.empty:
                continue

            st.markdown(f"### {grp}")
            st.dataframe(
                style_market_table(sub),
                use_container_width=True,
                hide_index=True
            )

# =========================================================
# TAB 2
# =========================================================
with tab2:
    st.subheader("Signal Charts")

    if not selected_chart_symbols:
        st.info("Please select at least one chart series in the sidebar.")
    else:
        lookback_days = {"6M": 180, "1Y": 365, "2Y": 730}[chart_lookback]
        min_date = pd.Timestamp.today().normalize() - pd.Timedelta(days=lookback_days)

        combined = []

        # Yahoo series
        for sym in selected_chart_symbols:
            if sym in yf_hist.columns:
                s = yf_hist[sym].dropna()
                s = s[s.index >= min_date]
                if not s.empty:
                    base = s.iloc[0]
                    norm = (s / base) * 100.0
                    temp = pd.DataFrame({
                        "Date": norm.index,
                        "Value": norm.values,
                        "Series": sym
                    })
                    combined.append(temp)

        # FRED series
        fred_map = {x["symbol"]: x for x in FRED_SERIES}
        for sym in selected_chart_symbols:
            if sym in fred_data:
                s = fred_data[sym].dropna()
                if s.empty:
                    continue
                s = s.resample("D").ffill()
                s = s[s.index >= min_date]
                if not s.empty:
                    # Normalize most series for comparison
                    if fred_map[sym]["group"] in ["Liquidity", "Consumption", "Labor"]:
                        base = s.iloc[0]
                        if base != 0 and pd.notna(base):
                            plot_s = (s / base) * 100.0
                        else:
                            plot_s = s.copy()
                    else:
                        plot_s = s.copy()

                    temp = pd.DataFrame({
                        "Date": plot_s.index,
                        "Value": plot_s.values,
                        "Series": sym
                    })
                    combined.append(temp)

        if combined:
            plot_df = pd.concat(combined, ignore_index=True)
            fig = px.line(
                plot_df,
                x="Date",
                y="Value",
                color="Series",
                title="Selected Market / Macro Series"
            )
            fig.update_layout(height=650, legend_title_text="")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No chartable data available for the current selection.")

# =========================================================
# TAB 3
# =========================================================
with tab3:
    st.subheader("Raw Data / Notes")

    st.markdown("### What this dashboard does")
    st.markdown(
        """
- The first tab starts with the **simplest possible market snapshot table**.
- It shows **1D / 1W / 1M / 1Y changes** together.
- It also organizes the same information by **group** so you can separate **noise** from **signal**.
- Market prices use **percentage change**.
- FRED macro / rate series use **absolute change**.
        """
    )

    st.markdown("### Series map")
    series_map = pd.DataFrame(MARKET_ASSETS + FRED_SERIES)
    st.dataframe(series_map, use_container_width=True, hide_index=True)

    st.markdown("### Latest processed dataset")
    if not market_df.empty:
        st.dataframe(market_df, use_container_width=True, hide_index=True)

    st.markdown("### Interpretation guide")
    guide = pd.DataFrame([
        {"Signal": "Risk-On", "Meaning": "Equities / risk assets trending positively"},
        {"Signal": "Risk-Off", "Meaning": "Equities / risk assets weakening"},
        {"Signal": "Tightening", "Meaning": "Rates or real yields moving higher"},
        {"Signal": "Easing", "Meaning": "Rates moving lower"},
        {"Signal": "Curve Inverted", "Meaning": "Yield curve below zero"},
        {"Signal": "Liquidity Positive", "Meaning": "M2 / Fed balance sheet expanding"},
        {"Signal": "Inflation Rising", "Meaning": "Commodity / breakeven inflation rising"},
        {"Signal": "Labor Weakening", "Meaning": "Unemployment trend worsening"},
    ])
    st.dataframe(guide, use_container_width=True, hide_index=True)
