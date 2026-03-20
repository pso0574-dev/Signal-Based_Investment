# app.py
import os
import requests
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.express as px

# =========================================================
# Page config
# =========================================================
st.set_page_config(
    page_title="Grouped Signal Charts",
    page_icon="📊",
    layout="wide"
)

# =========================================================
# Constants
# =========================================================
FRED_API_KEY = os.getenv("FRED_API_KEY", "").strip()
FRED_BASE_URL = "https://api.stlouisfed.org/fred/series/observations"

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
# Yahoo Finance helpers
# =========================================================
def extract_close_series(downloaded_df: pd.DataFrame, symbol: str) -> pd.Series:
    if downloaded_df is None or downloaded_df.empty:
        return pd.Series(dtype=float, name=symbol)

    cols = downloaded_df.columns

    if isinstance(cols, pd.MultiIndex):
        for price_col in ["Adj Close", "Close"]:
            key1 = (price_col, symbol)
            key2 = (symbol, price_col)
            if key1 in cols:
                s = downloaded_df[key1].copy()
                s.name = symbol
                return s
            if key2 in cols:
                s = downloaded_df[key2].copy()
                s.name = symbol
                return s

        try:
            if symbol in cols.get_level_values(0):
                sub = downloaded_df[symbol]
                for candidate in ["Adj Close", "Close"]:
                    if candidate in sub.columns:
                        s = sub[candidate].copy()
                        s.name = symbol
                        return s
            if symbol in cols.get_level_values(1):
                sub = downloaded_df.xs(symbol, axis=1, level=1)
                for candidate in ["Adj Close", "Close"]:
                    if candidate in sub.columns:
                        s = sub[candidate].copy()
                        s.name = symbol
                        return s
        except Exception:
            pass

    for candidate in ["Adj Close", "Close"]:
        if candidate in downloaded_df.columns:
            s = downloaded_df[candidate].copy()
            s.name = symbol
            return s

    numeric_cols = downloaded_df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        s = downloaded_df[numeric_cols[0]].copy()
        s.name = symbol
        return s

    return pd.Series(dtype=float, name=symbol)

@st.cache_data(ttl=60 * 30)
def get_yf_history(symbols, period="2y"):
    if not symbols:
        return pd.DataFrame()

    try:
        df = yf.download(
            tickers=symbols,
            period=period,
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=True,
            group_by="column"
        )
    except Exception:
        return pd.DataFrame()

    if df is None or df.empty:
        return pd.DataFrame()

    closes = {}
    for sym in symbols:
        try:
            s = extract_close_series(df, sym).dropna()
            if not s.empty:
                closes[sym] = s
        except Exception:
            continue

    if not closes:
        return pd.DataFrame()

    result = pd.DataFrame(closes)
    result.index = pd.to_datetime(result.index)
    return result.sort_index()

# =========================================================
# FRED helpers
# =========================================================
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

    return pd.Series(
        data=[v for _, v in rows],
        index=[d for d, _ in rows],
        name=series_id
    ).sort_index()

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
# Grouped chart data builder
# =========================================================
def build_grouped_chart_data(yf_hist, fred_data, selected_groups, lookback_days):
    min_date = pd.Timestamp.today().normalize() - pd.Timedelta(days=lookback_days)
    grouped_data = {}

    # Yahoo assets
    for item in MARKET_ASSETS:
        grp = item["group"]
        sym = item["symbol"]

        if selected_groups and grp not in selected_groups:
            continue
        if yf_hist.empty or sym not in yf_hist.columns:
            continue

        s = yf_hist[sym].dropna()
        s = s[s.index >= min_date]
        if s.empty:
            continue

        base = s.iloc[0]
        if pd.isna(base) or base == 0:
            continue

        plot_s = (s / base) * 100.0

        temp = pd.DataFrame({
            "Date": plot_s.index,
            "Value": plot_s.values,
            "Series": sym,
            "Asset": item["name"],
            "Group": grp
        })

        grouped_data.setdefault(grp, []).append(temp)

    # FRED assets
    for item in FRED_SERIES:
        grp = item["group"]
        sym = item["symbol"]

        if selected_groups and grp not in selected_groups:
            continue
        if sym not in fred_data:
            continue

        s = fred_data[sym].dropna()
        if s.empty:
            continue

        s = s.resample("D").ffill()
        s = s[s.index >= min_date]
        if s.empty:
            continue

        if grp in ["Liquidity", "Consumption", "Labor"]:
            base = s.iloc[0]
            if pd.isna(base) or base == 0:
                continue
            plot_s = (s / base) * 100.0
        else:
            plot_s = s.copy()

        temp = pd.DataFrame({
            "Date": plot_s.index,
            "Value": plot_s.values,
            "Series": sym,
            "Asset": item["name"],
            "Group": grp
        })

        grouped_data.setdefault(grp, []).append(temp)

    return grouped_data

# =========================================================
# Sidebar
# =========================================================
st.title("📊 Grouped Signal Charts")
st.caption("All group charts shown directly on one page without tabs.")

with st.sidebar:
    st.header("Settings")

    if FRED_API_KEY:
        st.success("FRED API key loaded")
    else:
        st.error("FRED_API_KEY is not set")

    selected_groups = st.multiselect(
        "Select groups",
        options=GROUP_ORDER,
        default=[
            "Equity",
            "Rates",
            "Inflation / Commodity",
            "Liquidity",
            "FX / Dollar",
            "Risk Asset"
        ]
    )

    chart_lookback = st.selectbox(
        "Chart lookback",
        options=["6M", "1Y", "2Y"],
        index=1
    )

    show_group_table = st.checkbox("Show component table", value=True)

# =========================================================
# Load data
# =========================================================
yf_symbols = [x["symbol"] for x in MARKET_ASSETS]

try:
    yf_hist = get_yf_history(yf_symbols, period="2y")
except Exception as e:
    st.error(f"Failed to load Yahoo Finance data: {e}")
    yf_hist = pd.DataFrame()

try:
    fred_data = load_all_fred(FRED_SERIES)
except Exception as e:
    st.error(f"Failed to load FRED data: {e}")
    fred_data = {}

# =========================================================
# Main page: Grouped Signal Charts only
# =========================================================
st.subheader("Grouped Signal Charts")

if not selected_groups:
    st.info("Please select at least one group in the sidebar.")
else:
    lookback_days = {"6M": 180, "1Y": 365, "2Y": 730}[chart_lookback]

    grouped_data = build_grouped_chart_data(
        yf_hist=yf_hist,
        fred_data=fred_data,
        selected_groups=selected_groups,
        lookback_days=lookback_days
    )

    plotted_any = False

    for grp in GROUP_ORDER:
        if grp not in grouped_data:
            continue

        frames = grouped_data[grp]
        if not frames:
            continue

        group_df = pd.concat(frames, ignore_index=True)
        if group_df.empty:
            continue

        plotted_any = True
        st.markdown(f"### {grp}")

        if grp in ["Equity", "Rates Proxy", "Inflation / Commodity", "FX / Dollar", "Risk Asset"]:
            y_title = "Normalized Index (Start = 100)"
        elif grp in ["Liquidity", "Consumption", "Labor"]:
            y_title = "Normalized Macro Index (Start = 100)"
        else:
            y_title = "Level"

        fig = px.line(
            group_df,
            x="Date",
            y="Value",
            color="Series",
            hover_name="Asset",
            title=f"{grp} — All Items"
        )

        fig.update_layout(
            height=480,
            legend_title_text="Ticker",
            xaxis_title="Date",
            yaxis_title=y_title
        )

        st.plotly_chart(fig, use_container_width=True)

        if show_group_table:
            latest_rows = []
            for series_name in group_df["Series"].unique():
                sub = group_df[group_df["Series"] == series_name].sort_values("Date")
                if sub.empty:
                    continue

                latest_rows.append({
                    "Ticker": series_name,
                    "Asset": sub["Asset"].iloc[-1],
                    "Latest Chart Value": round(float(sub["Value"].iloc[-1]), 2)
                })

            if latest_rows:
                latest_df = pd.DataFrame(latest_rows)
                st.dataframe(latest_df, use_container_width=True, hide_index=True)

        st.markdown("---")

    if not plotted_any:
        st.warning("No chartable data available for the selected groups.")
