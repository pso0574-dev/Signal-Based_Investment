# signal_dashboard_fred_api.py
# Run:
#   streamlit run signal_dashboard_fred_api.py
#
# Install:
#   pip install streamlit pandas numpy requests plotly yfinance beautifulsoup4 lxml

import os
import re
import io
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go
import yfinance as yf
from bs4 import BeautifulSoup

# --------------------------------------------------
# App Config
# --------------------------------------------------
st.set_page_config(
    page_title="Signal-Based Investment Dashboard (FRED API)",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------
# Constants
# --------------------------------------------------
FRED_BASE_URL = "https://api.stlouisfed.org/fred/series/observations"

FRED_SERIES = {
    # Liquidity / financial conditions
    "WALCL": "Fed Balance Sheet",
    "NFCI": "Chicago Fed NFCI",
    "ANFCI": "Chicago Fed ANFCI",

    # Growth
    "UNRATE": "Unemployment Rate",
    "PAYEMS": "Nonfarm Payrolls",

    # Inflation
    "CPIAUCSL": "CPI Index",
    "T10YIE": "10Y Breakeven Inflation",

    # Rates / curve / real yield
    "DGS10": "US 10Y Yield",
    "DGS2": "US 2Y Yield",
    "DFII10": "US 10Y Real Yield",
    "FEDFUNDS": "Fed Funds Rate",

    # Stress / credit
    "BAMLH0A0HYM2": "US High Yield OAS",
}

ETF_TICKERS = {
    "QQQ": "Nasdaq 100",
    "TLT": "20+Y Treasury",
    "TIP": "TIPS ETF",
    "GLD": "Gold ETF",
    "DBC": "Commodities ETF",
    "SPY": "S&P 500",
}

DEFAULT_WEIGHTS = {
    "QQQ": 0.25,
    "TLT": 0.20,
    "TIP": 0.20,
    "GLD": 0.20,
    "DBC": 0.15,
}

RANGE_OPTIONS = {
    "6M": 180,
    "1Y": 365,
    "3Y": 365 * 3,
    "5Y": 365 * 5,
    "10Y": 365 * 10,
}

# --------------------------------------------------
# Helper Functions
# --------------------------------------------------
def safe_pct_change(series: pd.Series, periods: int = 1) -> pd.Series:
    return series.pct_change(periods=periods) * 100


def latest_valid(series: pd.Series):
    s = series.dropna()
    return s.iloc[-1] if len(s) else np.nan


def classify_score(score: int) -> str:
    if score >= 3:
        return "Risk-On"
    elif score >= 0:
        return "Balanced"
    elif score >= -3:
        return "Defensive"
    return "Risk-Off"


def normalize_weights(weights: dict) -> dict:
    total = sum(weights.values())
    if total == 0:
        return weights
    return {k: v / total for k, v in weights.items()}


def portfolio_bias_from_score(score: int, inflation_signal: int) -> dict:
    w = DEFAULT_WEIGHTS.copy()

    if score >= 3:
        w["QQQ"] = 0.40
        w["TLT"] = 0.15
        w["TIP"] = 0.15
        w["GLD"] = 0.15
        w["DBC"] = 0.15
    elif score >= 0:
        w["QQQ"] = 0.28
        w["TLT"] = 0.18
        w["TIP"] = 0.20
        w["GLD"] = 0.19
        w["DBC"] = 0.15
    elif score >= -3:
        w["QQQ"] = 0.18
        w["TLT"] = 0.22
        w["TIP"] = 0.24
        w["GLD"] = 0.22
        w["DBC"] = 0.14
    else:
        w["QQQ"] = 0.10
        w["TLT"] = 0.28
        w["TIP"] = 0.24
        w["GLD"] = 0.24
        w["DBC"] = 0.14

    if inflation_signal < 0:
        w["TIP"] += 0.03
        w["GLD"] += 0.02
        w["QQQ"] -= 0.03
        w["TLT"] -= 0.02

    return normalize_weights(w)


# --------------------------------------------------
# FRED API Loader
# --------------------------------------------------
@st.cache_data(ttl=60 * 60)
def load_fred_series_api(
    series_id: str,
    api_key: str,
    observation_start: str = "2000-01-01"
) -> pd.Series:
    """
    Load a FRED series using the official FRED API.
    """
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "observation_start": observation_start,
        "sort_order": "asc",
    }

    r = requests.get(FRED_BASE_URL, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()

    if "observations" not in data:
        raise RuntimeError(f"Invalid FRED response for {series_id}: {data}")

    df = pd.DataFrame(data["observations"])
    if df.empty:
        return pd.Series(dtype=float, name=series_id)

    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    s = df.set_index("date")["value"].sort_index()
    s.name = series_id
    return s


@st.cache_data(ttl=60 * 60)
def load_all_fred_api(api_key: str, observation_start: str = "2000-01-01") -> pd.DataFrame:
    frames = []
    errors = []

    for sid in FRED_SERIES.keys():
        try:
            s = load_fred_series_api(sid, api_key=api_key, observation_start=observation_start)
            frames.append(s.rename(sid))
        except Exception as e:
            errors.append((sid, str(e)))

    if errors:
        st.warning("Some FRED series failed to load:")
        for sid, msg in errors:
            st.write(f"- {sid}: {msg}")

    if not frames:
        raise RuntimeError("No FRED series loaded. Check your API key and internet connection.")

    return pd.concat(frames, axis=1).sort_index()


# --------------------------------------------------
# GDPNow Scraper
# --------------------------------------------------
@st.cache_data(ttl=60 * 60)
def fetch_gdpnow():
    url = "https://www.atlantafed.org/research-and-data/data/gdpnow"
    try:
        r = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "lxml")
        text = soup.get_text(" ", strip=True)

        m_value = re.search(r"([+-]?\d+(?:\.\d+)?)\s*%\s*Latest GDPNow Estimate", text, re.I)
        m_date = re.search(r"Updated:\s*([A-Za-z]+\s+\d{1,2},\s+\d{4})", text, re.I)
        m_quarter = re.search(r"Latest GDPNow Estimate for\s*([0-9]{4}:Q[1-4])", text, re.I)

        return {
            "value": float(m_value.group(1)) if m_value else np.nan,
            "updated": pd.to_datetime(m_date.group(1)) if m_date else pd.NaT,
            "quarter": m_quarter.group(1) if m_quarter else "",
        }
    except Exception:
        return {
            "value": np.nan,
            "updated": pd.NaT,
            "quarter": "",
        }


# --------------------------------------------------
# Market Data Loader
# --------------------------------------------------
@st.cache_data(ttl=60 * 60)
def load_market_data(tickers: list[str], period: str = "10y") -> pd.DataFrame:
    data = yf.download(tickers, period=period, auto_adjust=True, progress=False)

    if isinstance(data.columns, pd.MultiIndex):
        if "Close" in data.columns.get_level_values(0):
            close = data["Close"].copy()
        else:
            close = data.xs("Close", axis=1, level=0)
    else:
        close = data.copy()

    return close.dropna(how="all")


# --------------------------------------------------
# Feature Engineering
# --------------------------------------------------
def prepare_macro_features(fred: pd.DataFrame) -> pd.DataFrame:
    df = fred.copy()

    df["YC_10Y_2Y"] = df["DGS10"] - df["DGS2"]

    df["WALCL_13W_PCT"] = safe_pct_change(df["WALCL"], periods=13)
    df["PAYEMS_6M_PCT"] = safe_pct_change(df["PAYEMS"], periods=6)

    df["CPI_YOY"] = safe_pct_change(df["CPIAUCSL"], periods=12)
    df["CPI_3M_ANN"] = ((df["CPIAUCSL"] / df["CPIAUCSL"].shift(3)) ** 4 - 1) * 100

    df["UNRATE_6M_DELTA"] = df["UNRATE"] - df["UNRATE"].shift(6)

    df["NFCI_13W_DELTA"] = df["NFCI"] - df["NFCI"].shift(13)
    df["HY_OAS_13W_DELTA"] = df["BAMLH0A0HYM2"] - df["BAMLH0A0HYM2"].shift(13)
    df["DFII10_13W_DELTA"] = df["DFII10"] - df["DFII10"].shift(13)
    df["T10YIE_13W_DELTA"] = df["T10YIE"] - df["T10YIE"].shift(13)
    df["FEDFUNDS_26W_DELTA"] = df["FEDFUNDS"] - df["FEDFUNDS"].shift(26)

    return df


# --------------------------------------------------
# Signal Engine
# --------------------------------------------------
def compute_category_scores(df: pd.DataFrame, gdpnow_value: float | None = None) -> dict:
    latest = df.ffill().iloc[-1]

    # Liquidity
    liq = 0
    if pd.notna(latest["WALCL_13W_PCT"]):
        liq += 1 if latest["WALCL_13W_PCT"] > 1.0 else -1
    if pd.notna(latest["NFCI"]):
        liq += 1 if latest["NFCI"] < 0 else -1
    liquidity_score = 1 if liq >= 1 else (-1 if liq <= -1 else 0)

    # Growth
    growth = 0
    if pd.notna(latest["UNRATE_6M_DELTA"]):
        growth += 1 if latest["UNRATE_6M_DELTA"] <= 0.0 else -1
    if pd.notna(latest["PAYEMS_6M_PCT"]):
        growth += 1 if latest["PAYEMS_6M_PCT"] > 0.5 else -1
    if gdpnow_value is not None and not np.isnan(gdpnow_value):
        growth += 1 if gdpnow_value > 1.5 else (-1 if gdpnow_value < 0.5 else 0)
    growth_score = 1 if growth >= 1 else (-1 if growth <= -1 else 0)

    # Inflation
    inflation = 0
    if pd.notna(latest["CPI_YOY"]):
        inflation += 1 if latest["CPI_YOY"] < 3.0 else -1
    if pd.notna(latest["CPI_3M_ANN"]):
        inflation += 1 if latest["CPI_3M_ANN"] < 3.0 else -1
    if pd.notna(latest["T10YIE_13W_DELTA"]):
        inflation += 1 if latest["T10YIE_13W_DELTA"] <= 0 else -1
    inflation_score = 1 if inflation >= 1 else (-1 if inflation <= -1 else 0)

    # Rates
    rates = 0
    if pd.notna(latest["YC_10Y_2Y"]):
        rates += 1 if latest["YC_10Y_2Y"] > 0 else -1
    if pd.notna(latest["DFII10_13W_DELTA"]):
        rates += 1 if latest["DFII10_13W_DELTA"] <= 0 else -1
    if pd.notna(latest["FEDFUNDS_26W_DELTA"]):
        rates += 1 if latest["FEDFUNDS_26W_DELTA"] <= 0 else -1
    rates_score = 1 if rates >= 1 else (-1 if rates <= -1 else 0)

    # Stress
    stress = 0
    if pd.notna(latest["NFCI"]):
        stress += 1 if latest["NFCI"] < 0 else -1
    if pd.notna(latest["HY_OAS_13W_DELTA"]):
        stress += 1 if latest["HY_OAS_13W_DELTA"] <= 0 else -1
    stress_score = 1 if stress >= 1 else (-1 if stress <= -1 else 0)

    scores = {
        "Liquidity": liquidity_score,
        "Growth": growth_score,
        "Inflation": inflation_score,
        "Rates": rates_score,
        "Stress": stress_score,
    }
    scores["Total"] = int(sum(scores.values()))
    return scores


def scores_to_df(scores: dict) -> pd.DataFrame:
    rows = []
    for k, v in scores.items():
        if k == "Total":
            continue
        rows.append({
            "Category": k,
            "Score": v,
            "Status": "Bullish" if v > 0 else ("Defensive" if v < 0 else "Neutral")
        })
    return pd.DataFrame(rows)


def build_history_scores(df: pd.DataFrame) -> pd.DataFrame:
    monthly = df.ffill().resample("M").last().copy()
    rows = []

    for idx in monthly.index:
        sub = monthly.loc[:idx].copy()
        if len(sub) < 14:
            continue

        latest = sub.iloc[-1]
        scores = {}

        liq_raw = 0
        liq_raw += 1 if latest["WALCL_13W_PCT"] > 1.0 else -1
        liq_raw += 1 if latest["NFCI"] < 0 else -1
        scores["Liquidity"] = 1 if liq_raw >= 1 else (-1 if liq_raw <= -1 else 0)

        growth_raw = 0
        growth_raw += 1 if latest["UNRATE_6M_DELTA"] <= 0 else -1
        growth_raw += 1 if latest["PAYEMS_6M_PCT"] > 0.5 else -1
        scores["Growth"] = 1 if growth_raw >= 1 else (-1 if growth_raw <= -1 else 0)

        infl_raw = 0
        infl_raw += 1 if latest["CPI_YOY"] < 3.0 else -1
        infl_raw += 1 if latest["CPI_3M_ANN"] < 3.0 else -1
        infl_raw += 1 if latest["T10YIE_13W_DELTA"] <= 0 else -1
        scores["Inflation"] = 1 if infl_raw >= 1 else (-1 if infl_raw <= -1 else 0)

        rates_raw = 0
        rates_raw += 1 if latest["YC_10Y_2Y"] > 0 else -1
        rates_raw += 1 if latest["DFII10_13W_DELTA"] <= 0 else -1
        rates_raw += 1 if latest["FEDFUNDS_26W_DELTA"] <= 0 else -1
        scores["Rates"] = 1 if rates_raw >= 1 else (-1 if rates_raw <= -1 else 0)

        stress_raw = 0
        stress_raw += 1 if latest["NFCI"] < 0 else -1
        stress_raw += 1 if latest["HY_OAS_13W_DELTA"] <= 0 else -1
        scores["Stress"] = 1 if stress_raw >= 1 else (-1 if stress_raw <= -1 else 0)

        total = sum(scores.values())
        rows.append({
            "Date": idx,
            **scores,
            "Total": total,
            "Regime": classify_score(total),
        })

    return pd.DataFrame(rows).set_index("Date")


# --------------------------------------------------
# Plot Functions
# --------------------------------------------------
def make_line_chart(df: pd.DataFrame, title: str, yaxis_title: str = ""):
    fig = go.Figure()
    for col in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df[col], mode="lines", name=col))

    fig.update_layout(
        title=title,
        height=420,
        template="plotly_white",
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        yaxis_title=yaxis_title,
    )
    return fig


def make_bar_chart(df: pd.DataFrame, title: str):
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df["Category"], y=df["Score"], text=df["Status"], textposition="outside"))
    fig.update_layout(
        title=title,
        height=360,
        template="plotly_white",
        margin=dict(l=20, r=20, t=50, b=20),
        yaxis=dict(range=[-1.5, 1.5], dtick=1),
    )
    return fig


def make_pie(weights: dict, title: str):
    fig = go.Figure(data=[go.Pie(labels=list(weights.keys()), values=list(weights.values()), hole=0.4)])
    fig.update_layout(
        title=title,
        height=420,
        template="plotly_white",
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


def make_regime_history_chart(hist: pd.DataFrame, price: pd.Series | None = None):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist.index, y=hist["Total"], mode="lines", name="Signal Score"))

    if price is not None and len(price.dropna()) > 0:
        p = price.reindex(hist.index).ffill()
        p = p / p.iloc[0] * 100
        fig.add_trace(go.Scatter(
            x=p.index, y=p, mode="lines", name="Benchmark (rebased=100)", yaxis="y2"
        ))

    fig.update_layout(
        title="Historical Signal Score vs Benchmark",
        height=460,
        template="plotly_white",
        margin=dict(l=20, r=20, t=50, b=20),
        yaxis=dict(title="Signal Score"),
        yaxis2=dict(title="Benchmark rebased", overlaying="y", side="right"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig


# --------------------------------------------------
# Sidebar
# --------------------------------------------------
st.sidebar.title("FRED API Dashboard Settings")

env_api_key = os.getenv("FRED_API_KEY", "")
api_key_input = st.sidebar.text_input("FRED API Key", value=env_api_key, type="password")

selected_range_label = st.sidebar.selectbox("Lookback Range", list(RANGE_OPTIONS.keys()), index=2)
lookback_days = RANGE_OPTIONS[selected_range_label]

observation_start = st.sidebar.text_input("Observation Start", value="2000-01-01")
show_gdpnow = st.sidebar.checkbox("Use GDPNow in Growth Signal", value=True)
show_market = st.sidebar.checkbox("Load ETF Market Data", value=True)
history_benchmark = st.sidebar.selectbox("History Benchmark", ["QQQ", "SPY", "TLT", "TIP", "GLD"], index=0)

st.sidebar.markdown("---")
st.sidebar.caption("Use your own FRED API key.")

if not api_key_input:
    st.error("Please enter your FRED API key in the sidebar or set FRED_API_KEY.")
    st.stop()

# --------------------------------------------------
# Load Data
# --------------------------------------------------
with st.spinner("Loading macro data from FRED API..."):
    fred = load_all_fred_api(api_key=api_key_input, observation_start=observation_start)
    macro = prepare_macro_features(fred)

gdpnow = fetch_gdpnow() if show_gdpnow else {"value": np.nan, "updated": pd.NaT, "quarter": ""}

market = None
if show_market:
    with st.spinner("Loading ETF market data..."):
        market = load_market_data(list(ETF_TICKERS.keys()), period="10y")

start_date = pd.Timestamp.today().normalize() - pd.Timedelta(days=lookback_days)
macro_view = macro.loc[macro.index >= start_date].copy()
market_view = market.loc[market.index >= start_date].copy() if market is not None else None

scores = compute_category_scores(macro, gdpnow_value=gdpnow["value"])
score_df = scores_to_df(scores)
regime = classify_score(scores["Total"])
weights = portfolio_bias_from_score(scores["Total"], scores["Inflation"])
history_scores = build_history_scores(macro)

# --------------------------------------------------
# Header
# --------------------------------------------------
st.title("Signal-Based Investment Dashboard")
st.caption("FRED API version — designed to focus on structural signals, not daily noise.")

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Liquidity", "Bullish" if scores["Liquidity"] > 0 else ("Defensive" if scores["Liquidity"] < 0 else "Neutral"), scores["Liquidity"])
c2.metric("Growth", "Bullish" if scores["Growth"] > 0 else ("Defensive" if scores["Growth"] < 0 else "Neutral"), scores["Growth"])
c3.metric("Inflation", "Bullish" if scores["Inflation"] > 0 else ("Defensive" if scores["Inflation"] < 0 else "Neutral"), scores["Inflation"])
c4.metric("Rates", "Bullish" if scores["Rates"] > 0 else ("Defensive" if scores["Rates"] < 0 else "Neutral"), scores["Rates"])
c5.metric("Stress", "Bullish" if scores["Stress"] > 0 else ("Defensive" if scores["Stress"] < 0 else "Neutral"), scores["Stress"])
c6.metric("Total Regime", regime, scores["Total"])

if show_gdpnow:
    g1, g2, g3 = st.columns([1, 1, 2])
    g1.metric("GDPNow", f"{gdpnow['value']:.1f}%" if pd.notna(gdpnow["value"]) else "N/A")
    g2.metric("Quarter", gdpnow["quarter"] if gdpnow["quarter"] else "N/A")
    g3.write(f"Updated: {gdpnow['updated'].date() if pd.notna(gdpnow['updated']) else 'N/A'}")

st.markdown("---")

# --------------------------------------------------
# Tabs
# --------------------------------------------------
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
    ["Overview", "Liquidity", "Growth", "Inflation", "Rates & Stress", "Allocation", "History"]
)

# Overview
with tab1:
    left, right = st.columns([1.2, 1])

    with left:
        st.subheader("Signal Scorecard")
        st.dataframe(score_df, use_container_width=True, hide_index=True)
        st.plotly_chart(make_bar_chart(score_df, "Category Scores"), use_container_width=True)

    with right:
        st.subheader("Suggested Portfolio Bias")
        st.plotly_chart(make_pie(weights, f"{regime} Allocation Bias"), use_container_width=True)

        action_text = {
            "Risk-On": "Overweight growth assets. Keep defense exposure but lean into QQQ/equities.",
            "Balanced": "Keep core diversification. Avoid large shifts.",
            "Defensive": "Emphasize TIP, gold, and some duration. Stay selective on risk assets.",
            "Risk-Off": "Prioritize capital preservation. Hold more duration, inflation defense, and lower growth exposure."
        }
        st.info(action_text[regime])

        latest = macro.ffill().iloc[-1]
        key_df = pd.DataFrame([
            ["Fed Balance Sheet 13W %", latest["WALCL_13W_PCT"]],
            ["NFCI", latest["NFCI"]],
            ["Unemployment Rate", latest["UNRATE"]],
            ["CPI YoY", latest["CPI_YOY"]],
            ["10Y-2Y Curve", latest["YC_10Y_2Y"]],
            ["10Y Breakeven", latest["T10YIE"]],
            ["10Y Real Yield", latest["DFII10"]],
            ["High Yield OAS", latest["BAMLH0A0HYM2"]],
        ], columns=["Metric", "Value"])
        key_df["Value"] = key_df["Value"].map(lambda x: round(float(x), 2) if pd.notna(x) else np.nan)
        st.dataframe(key_df, use_container_width=True, hide_index=True)

# Liquidity
with tab2:
    st.subheader("Liquidity Signal")
    l1, l2 = st.columns(2)

    walcl_df = macro_view[["WALCL"]].copy()
    walcl_df["WALCL (Trillions)"] = walcl_df["WALCL"] / 1000
    walcl_df = walcl_df[["WALCL (Trillions)"]]

    nfci_df = macro_view[["NFCI", "ANFCI"]].copy()

    with l1:
        st.plotly_chart(make_line_chart(walcl_df, "Fed Balance Sheet (WALCL)", "USD Trillions"), use_container_width=True)
    with l2:
        st.plotly_chart(make_line_chart(nfci_df, "Financial Conditions (NFCI / ANFCI)", "Index"), use_container_width=True)

# Growth
with tab3:
    st.subheader("Growth Signal")
    gcol1, gcol2 = st.columns(2)

    growth_chart_df = macro_view[["UNRATE"]].copy()
    payroll_chart_df = macro_view[["PAYEMS"]].copy()
    payroll_chart_df["Payrolls (M)"] = payroll_chart_df["PAYEMS"] / 1000
    payroll_chart_df = payroll_chart_df[["Payrolls (M)"]]

    with gcol1:
        st.plotly_chart(make_line_chart(growth_chart_df, "Unemployment Rate", "%"), use_container_width=True)
    with gcol2:
        st.plotly_chart(make_line_chart(payroll_chart_df, "Nonfarm Payrolls", "Millions"), use_container_width=True)

# Inflation
with tab4:
    st.subheader("Inflation Signal")
    i1, i2 = st.columns(2)

    inflation_df = macro_view[["CPI_YOY", "CPI_3M_ANN"]].copy()
    breakeven_df = macro_view[["T10YIE"]].copy()

    with i1:
        st.plotly_chart(make_line_chart(inflation_df, "Inflation Trend", "%"), use_container_width=True)
    with i2:
        st.plotly_chart(make_line_chart(breakeven_df, "10Y Breakeven Inflation", "%"), use_container_width=True)

# Rates & Stress
with tab5:
    st.subheader("Rates, Curve, and Financial Stress")
    r1, r2 = st.columns(2)

    curve_df = macro_view[["DGS10", "DGS2", "YC_10Y_2Y"]].copy()
    stress_df = macro_view[["BAMLH0A0HYM2", "DFII10", "FEDFUNDS"]].copy()

    with r1:
        st.plotly_chart(make_line_chart(curve_df, "Rates and Yield Curve", "%"), use_container_width=True)
    with r2:
        st.plotly_chart(make_line_chart(stress_df, "Credit / Real Yield / Policy Rate", "%"), use_container_width=True)

# Allocation
with tab6:
    st.subheader("Signal-to-Allocation Map")

    alloc_df = pd.DataFrame({
        "Ticker": list(weights.keys()),
        "Asset": [ETF_TICKERS.get(k, k) for k in weights.keys()],
        "Suggested Weight (%)": [round(v * 100, 1) for v in weights.values()],
    })
    st.dataframe(alloc_df, use_container_width=True, hide_index=True)

    if market_view is not None:
        rows = []
        for t in list(weights.keys()):
            if t not in market_view.columns:
                continue
            s = market_view[t].dropna()
            if len(s) < 60:
                continue
            ma50 = s.rolling(50).mean().iloc[-1]
            ma200 = s.rolling(200).mean().iloc[-1] if len(s) >= 200 else np.nan

            rows.append({
                "Ticker": t,
                "Price": round(float(s.iloc[-1]), 2),
                "1M %": round(float(s.pct_change(21).iloc[-1] * 100), 2) if len(s) >= 21 else np.nan,
                "3M %": round(float(s.pct_change(63).iloc[-1] * 100), 2) if len(s) >= 63 else np.nan,
                "Above 50D MA": bool(s.iloc[-1] > ma50) if pd.notna(ma50) else None,
                "Above 200D MA": bool(s.iloc[-1] > ma200) if pd.notna(ma200) else None,
            })

        trend_df = pd.DataFrame(rows)
        st.subheader("ETF Trend Snapshot")
        st.dataframe(trend_df, use_container_width=True, hide_index=True)

# History
with tab7:
    st.subheader("Historical Regime")
    h1, h2 = st.columns([2, 1])

    with h1:
        bench_series = None
        if market is not None and history_benchmark in market.columns:
            bench_series = market[history_benchmark]
        st.plotly_chart(make_regime_history_chart(history_scores, bench_series), use_container_width=True)

    with h2:
        regime_counts = history_scores["Regime"].value_counts().rename_axis("Regime").reset_index(name="Months")
        st.dataframe(regime_counts, use_container_width=True, hide_index=True)

        recent_hist = history_scores.tail(12).reset_index()
        recent_hist["Date"] = recent_hist["Date"].dt.strftime("%Y-%m")
        st.dataframe(recent_hist, use_container_width=True, hide_index=True)

# Footer
st.markdown("---")
st.markdown("""
### Notes
- This version uses the official FRED API with `file_type=json`.
- Review frequency should stay low: weekly review, monthly allocation changes.
- GDPNow scraping can break if the Atlanta Fed page structure changes.
- The allocation logic is heuristic and should be customized to your own rules.
""")
