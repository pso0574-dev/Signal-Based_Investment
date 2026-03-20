# streamlit_app.py
# ============================================================
# Global Macro Monitoring Dashboard (stabilized version)
# - More robust Yahoo loader
# - FRED fallback for 10Y / 2Y / 3M
# - Better empty-data handling
# - Safer summary / chart rendering
#
# Run:
#   streamlit run streamlit_app.py
#
# Install:
#   pip install streamlit pandas numpy plotly yfinance requests python-dateutil
#
# Optional:
#   Set FRED_API_KEY in environment variables
#   or .streamlit/secrets.toml:
#   FRED_API_KEY="YOUR_FRED_API_KEY"
# ============================================================

from __future__ import annotations

import os
from typing import Dict, Optional, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf
from dateutil.relativedelta import relativedelta

# ============================================================
# Page setup
# ============================================================
st.set_page_config(
    page_title="Global Macro Monitoring Dashboard",
    page_icon="🌍",
    layout="wide",
)

st.title("🌍 Global Macro Monitoring Dashboard")
st.caption(
    "Oil / Rates / Gold / BTC / Equity / FX • Current level + 3M / 6M / 1Y / 5Y / 10Y changes"
)

# ============================================================
# Constants
# ============================================================
TODAY = pd.Timestamp.today().normalize()
START_DATE = TODAY - relativedelta(years=11)

DISPLAY_PERIODS = {
    "3M": relativedelta(months=3),
    "6M": relativedelta(months=6),
    "1Y": relativedelta(years=1),
    "5Y": relativedelta(years=5),
    "10Y": relativedelta(years=10),
}

GRAPH_PERIODS = {
    "3M": relativedelta(months=3),
    "6M": relativedelta(months=6),
    "1Y": relativedelta(years=1),
    "5Y": relativedelta(years=5),
    "10Y": relativedelta(years=10),
    "ALL": relativedelta(years=11),
}

ASSET_CONFIG = [
    {"category": "Oil", "asset": "WTI Crude Oil", "symbol": "CL=F", "source": "Yahoo", "unit": "USD/bbl", "type": "price"},
    {"category": "Oil", "asset": "Brent Crude Oil", "symbol": "BZ=F", "source": "Yahoo", "unit": "USD/bbl", "type": "price"},

    {"category": "Rate", "asset": "US 10Y Yield", "symbol": "DGS10", "source": "FRED", "unit": "%", "type": "rate"},
    {"category": "Rate", "asset": "US 2Y Yield", "symbol": "DGS2", "source": "FRED", "unit": "%", "type": "rate"},
    {"category": "Rate", "asset": "US 3M Yield", "symbol": "DGS3MO", "source": "FRED", "unit": "%", "type": "rate"},

    {"category": "Metal", "asset": "Gold", "symbol": "GC=F", "source": "Yahoo", "unit": "USD/oz", "type": "price"},
    {"category": "Commodity", "asset": "Broad Commodities ETF", "symbol": "DBC", "source": "Yahoo", "unit": "ETF", "type": "price"},

    {"category": "Crypto", "asset": "Bitcoin / EUR", "symbol": "BTC-EUR", "source": "Yahoo", "unit": "EUR", "type": "price"},

    {"category": "Equity", "asset": "S&P 500", "symbol": "^GSPC", "source": "Yahoo", "unit": "Index", "type": "price"},
    {"category": "Equity", "asset": "Dow Jones", "symbol": "^DJI", "source": "Yahoo", "unit": "Index", "type": "price"},
    {"category": "Equity", "asset": "Nasdaq 100", "symbol": "^NDX", "source": "Yahoo", "unit": "Index", "type": "price"},

    {"category": "FX", "asset": "USD/KRW", "symbol": "KRW=X", "source": "Yahoo", "unit": "KRW per USD", "type": "price"},
    {"category": "FX", "asset": "EUR/KRW", "symbol": "EURKRW=X", "source": "Yahoo", "unit": "KRW per EUR", "type": "price"},
]

# Yahoo fallbacks for rate data
# ^TNX = 10Y * 10, ^IRX = 13-week bill * 100? historically often quoted in percent*100
# ^FVX = 5Y, not 2Y. For 2Y exact Yahoo fallback is not always reliable, so use ^IRX only for 3M.
# For 2Y we keep a synthetic fallback via Treasury index if available; otherwise leave empty.
FRED_FALLBACK = {
    "DGS10": "^TNX",   # divide by 10
    "DGS3MO": "^IRX",  # divide by 100
    # DGS2 exact Yahoo fallback is unreliable; handled separately below
}

RATE_ASSETS = {
    "US 10Y Yield",
    "US 2Y Yield",
    "US 3M Yield",
    "US 10Y-2Y Spread",
    "US 10Y-3M Spread",
}

# ============================================================
# Helpers
# ============================================================
def get_fred_api_key() -> Optional[str]:
    key = None
    try:
        key = st.secrets.get("FRED_API_KEY", None)
    except Exception:
        key = None
    if not key:
        key = os.getenv("FRED_API_KEY")
    return key


def safe_float(x) -> Optional[float]:
    try:
        if pd.isna(x):
            return None
        return float(x)
    except Exception:
        return None


def empty_series() -> pd.Series:
    return pd.Series(dtype=float)


def format_current(value: Optional[float], unit: str, asset_type: str) -> str:
    if value is None:
        return "N/A"
    if asset_type == "rate":
        return f"{value:.2f}%"
    if "KRW" in unit:
        return f"{value:,.1f}"
    return f"{value:,.2f}"


def format_change(value: Optional[float], asset_type: str, as_bp: bool = False) -> str:
    if value is None:
        return "N/A"
    if asset_type == "rate":
        if as_bp:
            return f"{value * 100:.0f}bp"
        return f"{value:+.2f}pp"
    return f"{value:+.2f}%"


def latest_value(series: pd.Series) -> Optional[float]:
    if series is None or series.empty:
        return None
    s = series.dropna()
    if s.empty:
        return None
    return safe_float(s.iloc[-1])


def nearest_value(series: pd.Series, target_date: pd.Timestamp, tolerance_days: int = 10) -> Optional[float]:
    if series is None or series.empty:
        return None

    s = series.dropna().sort_index()
    if s.empty:
        return None

    idx = s.index.searchsorted(target_date, side="right") - 1
    candidate = None

    if idx >= 0:
        candidate_date = s.index[idx]
        if abs((target_date - candidate_date).days) <= tolerance_days:
            candidate = safe_float(s.iloc[idx])

    if candidate is None:
        diffs = np.abs((s.index - target_date).days)
        if len(diffs) > 0:
            min_i = int(np.argmin(diffs))
            if diffs[min_i] <= tolerance_days:
                candidate = safe_float(s.iloc[min_i])

    return candidate


def compute_pct_change(current: Optional[float], past: Optional[float]) -> Optional[float]:
    if current is None or past is None or past == 0:
        return None
    return (current / past - 1.0) * 100.0


def compute_pp_change(current: Optional[float], past: Optional[float]) -> Optional[float]:
    if current is None or past is None:
        return None
    return current - past


def normalize_to_100(series: pd.Series) -> pd.Series:
    s = series.dropna().copy()
    if s.empty:
        return s
    base = s.iloc[0]
    if pd.isna(base) or base == 0:
        return empty_series()
    return s / base * 100.0


def compute_return(series: pd.Series) -> pd.Series:
    s = series.dropna()
    if s.empty:
        return empty_series()
    return s.pct_change()


def compute_diff(series: pd.Series) -> pd.Series:
    s = series.dropna()
    if s.empty:
        return empty_series()
    return s.diff()


def rolling_corr(series_a: pd.Series, series_b: pd.Series, window: int = 30) -> pd.Series:
    df = pd.concat([series_a.rename("a"), series_b.rename("b")], axis=1).dropna()
    if df.empty:
        return empty_series()
    return df["a"].rolling(window).corr(df["b"])


def filter_series_by_period(series: pd.Series, period_label: str) -> pd.Series:
    if series is None or series.empty:
        return empty_series()
    start = TODAY - GRAPH_PERIODS[period_label]
    return series.loc[series.index >= start].dropna()


def get_row_value(df: pd.DataFrame, asset_name: str, col: str = "Current_raw") -> Optional[float]:
    try:
        return safe_float(df.loc[df["Asset"] == asset_name, col].iloc[0])
    except Exception:
        return None


def resample_for_scatter(rate_series: pd.Series, equity_series: pd.Series, freq_label: str) -> pd.DataFrame:
    rate_series = rate_series.dropna().sort_index()
    equity_series = equity_series.dropna().sort_index()

    if rate_series.empty or equity_series.empty:
        return pd.DataFrame(columns=["rate_change", "equity_return"])

    if freq_label == "Daily":
        rate_change = rate_series.diff()
        equity_return = equity_series.pct_change()

    elif freq_label == "Weekly":
        rate_rs = rate_series.resample("W-FRI").last()
        eq_rs = equity_series.resample("W-FRI").last()
        rate_change = rate_rs.diff()
        equity_return = eq_rs.pct_change()

    elif freq_label == "Monthly":
        rate_rs = rate_series.resample("ME").last()
        eq_rs = equity_series.resample("ME").last()
        rate_change = rate_rs.diff()
        equity_return = eq_rs.pct_change()

    else:
        rate_change = rate_series.diff()
        equity_return = equity_series.pct_change()

    df = pd.concat(
        [rate_change.rename("rate_change"), equity_return.rename("equity_return")],
        axis=1
    ).dropna()

    return df


def transform_series_for_corr(series: pd.Series, asset_type: str, freq_label: str) -> pd.Series:
    s = series.dropna().sort_index()
    if s.empty:
        return empty_series()

    if freq_label == "Daily":
        base = s
    elif freq_label == "Weekly":
        base = s.resample("W-FRI").last()
    elif freq_label == "Monthly":
        base = s.resample("ME").last()
    else:
        base = s

    if asset_type == "rate":
        return base.diff()
    return base.pct_change()


def build_corr_input_map(
    series_map: Dict[str, pd.Series],
    freq_label: str,
    selected_graph_period: str,
) -> Dict[str, pd.Series]:
    selected_assets: List[Tuple[str, str, str]] = [
        ("WTI Crude Oil", "CL=F", "price"),
        ("Gold", "GC=F", "price"),
        ("Bitcoin / EUR", "BTC-EUR", "price"),
        ("S&P 500", "^GSPC", "price"),
        ("Nasdaq 100", "^NDX", "price"),
        ("USD/KRW", "KRW=X", "price"),
        ("EUR/KRW", "EURKRW=X", "price"),
        ("US 10Y Yield", "DGS10", "rate"),
        ("US 2Y Yield", "DGS2", "rate"),
        ("US 3M Yield", "DGS3MO", "rate"),
        ("US 10Y-2Y Spread", "SPREAD_10Y_2Y", "rate"),
        ("US 10Y-3M Spread", "SPREAD_10Y_3M", "rate"),
    ]

    out: Dict[str, pd.Series] = {}
    for name, symbol, asset_type in selected_assets:
        raw = filter_series_by_period(series_map.get(symbol, empty_series()), selected_graph_period)
        tr = transform_series_for_corr(raw, asset_type, freq_label)
        if not tr.empty:
            out[name] = tr

    return out


# ============================================================
# Signal logic
# ============================================================
def build_signal(row: pd.Series, spread_10y_2y: Optional[float], spread_10y_3m: Optional[float]) -> str:
    asset = row["Asset"]
    category = row["Category"]
    ch_3m = row.get("3M_raw")
    ch_1y = row.get("1Y_raw")
    curr = row.get("Current_raw")

    if category == "Rate":
        if curr is None:
            return "N/A"
        if asset == "US 10Y Yield":
            if curr >= 4.5:
                return "Tight"
            if curr <= 3.0:
                return "Loose"
            return "Neutral"
        if asset == "US 2Y Yield":
            return "Policy Watch"
        if asset == "US 3M Yield":
            return "Liquidity Watch"

    if asset == "WTI Crude Oil":
        if ch_3m is not None and ch_3m >= 10:
            return "Inflation Watch"
        if ch_3m is not None and ch_3m <= -10:
            return "Demand Weakness"
        return "Neutral"

    if asset == "Gold":
        if ch_1y is not None and ch_1y > 10:
            return "Risk-Off / Hedge"
        return "Neutral"

    if asset == "Bitcoin / EUR":
        if ch_3m is not None and ch_3m > 15:
            return "Risk-On"
        if ch_3m is not None and ch_3m < -15:
            return "Risk-Off"
        return "Volatile"

    if asset == "S&P 500":
        if ch_1y is not None and ch_1y > 10:
            return "Growth"
        if ch_1y is not None and ch_1y < -10:
            return "Weakness"
        return "Neutral"

    if asset == "Nasdaq 100":
        if ch_1y is not None and ch_1y > 15:
            return "Growth / Tech"
        return "Neutral"

    if asset == "USD/KRW":
        if ch_3m is not None and ch_3m > 3:
            return "KRW Weakness"
        if ch_3m is not None and ch_3m < -3:
            return "KRW Strength"
        return "Neutral"

    if asset == "EUR/KRW":
        if ch_3m is not None and ch_3m > 3:
            return "EUR Strong"
        if ch_3m is not None and ch_3m < -3:
            return "EUR Weak"
        return "Neutral"

    if asset == "US 10Y-2Y Spread" and spread_10y_2y is not None:
        if spread_10y_2y < 0:
            return "Inverted"
        if spread_10y_2y < 0.5:
            return "Flat"
        return "Steep"

    if asset == "US 10Y-3M Spread" and spread_10y_3m is not None:
        if spread_10y_3m < 0:
            return "Inverted"
        if spread_10y_3m < 0.5:
            return "Flat"
        return "Steep"

    return "Neutral"


# ============================================================
# Data loaders
# ============================================================
def _extract_single_price_series(df: pd.DataFrame) -> pd.Series:
    if df is None or df.empty:
        return empty_series()

    # Flatten MultiIndex if present
    if isinstance(df.columns, pd.MultiIndex):
        flat_cols = []
        for col in df.columns:
            if isinstance(col, tuple):
                flat_cols.append(" | ".join([str(x) for x in col if x is not None]))
            else:
                flat_cols.append(str(col))
        df = df.copy()
        df.columns = flat_cols

    preferred = [
        "Adj Close",
        "Close",
    ]

    for pref in preferred:
        exact = [c for c in df.columns if c == pref or c.endswith(f"| {pref}") or c.startswith(f"{pref} |")]
        if exact:
            s = pd.to_numeric(df[exact[0]], errors="coerce")
            return s.dropna()

    # partial fallback
    for key in ["Adj Close", "Close", "Open", "High", "Low"]:
        cand = [c for c in df.columns if key in str(c)]
        if cand:
            s = pd.to_numeric(df[cand[0]], errors="coerce")
            return s.dropna()

    # first numeric column
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if numeric_cols:
        s = pd.to_numeric(df[numeric_cols[0]], errors="coerce")
        return s.dropna()

    return empty_series()


@st.cache_data(ttl=600, show_spinner=False)
def load_yahoo_history(symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    try:
        df = yf.download(
            tickers=symbol,
            start=start.date(),
            end=(end + pd.Timedelta(days=1)).date(),
            auto_adjust=False,
            progress=False,
            interval="1d",
            group_by="column",
            threads=False,
        )

        if df is None or df.empty:
            return empty_series()

        s = _extract_single_price_series(df)
        if s.empty:
            return empty_series()

        s.index = pd.to_datetime(s.index).tz_localize(None)
        s = s[~s.index.duplicated(keep="last")].sort_index()
        return s.dropna()
    except Exception:
        return empty_series()


@st.cache_data(ttl=600, show_spinner=False)
def load_fred_series(series_id: str, start: pd.Timestamp, end: pd.Timestamp, api_key: Optional[str]) -> pd.Series:
    if not api_key:
        return empty_series()

    try:
        url = "https://api.stlouisfed.org/fred/series/observations"
        params = {
            "series_id": series_id,
            "api_key": api_key,
            "file_type": "json",
            "observation_start": start.strftime("%Y-%m-%d"),
            "observation_end": end.strftime("%Y-%m-%d"),
        }
        r = requests.get(url, params=params, timeout=20)
        r.raise_for_status()
        data = r.json()

        obs = data.get("observations", [])
        if not obs:
            return empty_series()

        df = pd.DataFrame(obs)
        if "date" not in df.columns or "value" not in df.columns:
            return empty_series()

        df["date"] = pd.to_datetime(df["date"])
        df["value"] = pd.to_numeric(df["value"].replace(".", np.nan), errors="coerce")
        s = df.set_index("date")["value"].sort_index().dropna()
        s.index = s.index.tz_localize(None)
        s = s[~s.index.duplicated(keep="last")]
        return s
    except Exception:
        return empty_series()


def convert_yahoo_rate_fallback(series_id: str, s_fb: pd.Series) -> pd.Series:
    if s_fb.empty:
        return empty_series()

    if series_id == "DGS10":
        return s_fb / 10.0  # ^TNX is typically quoted as yield*10
    if series_id == "DGS3MO":
        return s_fb / 100.0  # ^IRX often comes as yield*100
    return empty_series()


@st.cache_data(ttl=600, show_spinner=False)
def load_all_series() -> Dict[str, pd.Series]:
    api_key = get_fred_api_key()
    out: Dict[str, pd.Series] = {}

    for item in ASSET_CONFIG:
        sym = item["symbol"]
        src = item["source"]

        if src == "Yahoo":
            s = load_yahoo_history(sym, START_DATE, TODAY)

        elif src == "FRED":
            s = load_fred_series(sym, START_DATE, TODAY, api_key)

            if s.empty and sym in FRED_FALLBACK:
                fb = FRED_FALLBACK[sym]
                s_fb = load_yahoo_history(fb, START_DATE, TODAY)
                s = convert_yahoo_rate_fallback(sym, s_fb)

            # Optional very rough 2Y proxy if FRED missing:
            # leave empty rather than use poor-quality proxy
            if s.empty and sym == "DGS2":
                s = empty_series()

        else:
            s = empty_series()

        out[sym] = s

    d10 = out.get("DGS10", empty_series())
    d2 = out.get("DGS2", empty_series())
    d3m = out.get("DGS3MO", empty_series())

    if not d10.empty and not d2.empty:
        aligned_10_2 = pd.concat([d10.rename("10Y"), d2.rename("2Y")], axis=1).dropna()
        out["SPREAD_10Y_2Y"] = aligned_10_2["10Y"] - aligned_10_2["2Y"]
    else:
        out["SPREAD_10Y_2Y"] = empty_series()

    if not d10.empty and not d3m.empty:
        aligned_10_3m = pd.concat([d10.rename("10Y"), d3m.rename("3M")], axis=1).dropna()
        out["SPREAD_10Y_3M"] = aligned_10_3m["10Y"] - aligned_10_3m["3M"]
    else:
        out["SPREAD_10Y_3M"] = empty_series()

    return out


# ============================================================
# Summary table
# ============================================================
def build_summary_table(series_map: Dict[str, pd.Series]) -> pd.DataFrame:
    rows = []

    for item in ASSET_CONFIG:
        category = item["category"]
        asset = item["asset"]
        symbol = item["symbol"]
        source = item["source"]
        unit = item["unit"]
        asset_type = item["type"]

        s = series_map.get(symbol, empty_series())
        current = latest_value(s)

        row = {
            "Category": category,
            "Asset": asset,
            "Symbol": symbol,
            "Current_raw": current,
            "Unit": unit,
            "Source": source,
            "Type": asset_type,
        }

        for label, delta in DISPLAY_PERIODS.items():
            past_date = TODAY - delta
            past = nearest_value(s, past_date)

            if asset_type == "rate":
                raw_change = compute_pp_change(current, past)
                row[f"{label}_raw"] = raw_change
                row[label] = format_change(raw_change, asset_type)
            else:
                raw_change = compute_pct_change(current, past)
                row[f"{label}_raw"] = raw_change
                row[label] = format_change(raw_change, asset_type)

        row["Current"] = format_current(current, unit, asset_type)
        rows.append(row)

    for spread_symbol, spread_name in [
        ("SPREAD_10Y_2Y", "US 10Y-2Y Spread"),
        ("SPREAD_10Y_3M", "US 10Y-3M Spread"),
    ]:
        s = series_map.get(spread_symbol, empty_series())
        current = latest_value(s)

        row = {
            "Category": "Rate",
            "Asset": spread_name,
            "Symbol": spread_symbol,
            "Current_raw": current,
            "Unit": "pp",
            "Source": "Derived",
            "Type": "rate",
        }

        for label, delta in DISPLAY_PERIODS.items():
            past_date = TODAY - delta
            past = nearest_value(s, past_date)
            raw_change = compute_pp_change(current, past)
            row[f"{label}_raw"] = raw_change
            row[label] = format_change(raw_change, "rate")

        row["Current"] = "N/A" if current is None else f"{current:+.2f}pp"
        rows.append(row)

    df = pd.DataFrame(rows)

    spread_10y_2y = get_row_value(df, "US 10Y-2Y Spread", "Current_raw")
    spread_10y_3m = get_row_value(df, "US 10Y-3M Spread", "Current_raw")

    df["Signal"] = df.apply(
        lambda r: build_signal(r, spread_10y_2y=spread_10y_2y, spread_10y_3m=spread_10y_3m),
        axis=1,
    )

    return df


# ============================================================
# Styling
# ============================================================
def color_change_cell(val: str) -> str:
    try:
        if val == "N/A":
            return "color: #999999;"
        s = str(val).replace("%", "").replace("pp", "").replace("bp", "")
        num = float(s)
        if num > 0:
            return "color: #0a7f2e; font-weight: 600;"
        if num < 0:
            return "color: #b00020; font-weight: 600;"
        return "color: #666666;"
    except Exception:
        return ""


def color_signal_cell(val: str) -> str:
    mapping = {
        "Risk-On": "#0a7f2e",
        "Growth": "#0a7f2e",
        "Growth / Tech": "#0a7f2e",
        "Risk-Off / Hedge": "#b00020",
        "Inflation Watch": "#b36b00",
        "Tight": "#b36b00",
        "Inverted": "#b00020",
        "KRW Weakness": "#b00020",
        "EUR Strong": "#0a4ea1",
        "Steep": "#0a7f2e",
        "Flat": "#b36b00",
    }
    color = mapping.get(val, "#444444")
    return f"color: {color}; font-weight: 600;"


# ============================================================
# Charts
# ============================================================
def make_line_chart(
    series_dict: Dict[str, pd.Series],
    title: str,
    normalize: bool = False,
    yaxis_title: str = "",
) -> go.Figure:
    fig = go.Figure()

    for name, series in series_dict.items():
        s = series.dropna()
        if s.empty:
            continue
        if normalize:
            s = normalize_to_100(s)
        fig.add_trace(go.Scatter(x=s.index, y=s.values, mode="lines", name=name))

    fig.update_layout(
        title=title,
        height=520,
        hovermode="x unified",
        margin=dict(l=40, r=20, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        xaxis_title="Date",
        yaxis_title=yaxis_title,
    )
    return fig


def make_dual_axis_chart(
    left_series: Dict[str, pd.Series],
    right_series: Dict[str, pd.Series],
    title: str,
    left_title: str,
    right_title: str,
) -> go.Figure:
    fig = go.Figure()

    for name, s in left_series.items():
        s = s.dropna()
        if s.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=s.index,
                y=s.values,
                mode="lines",
                name=name,
                yaxis="y",
            )
        )

    for name, s in right_series.items():
        s = s.dropna()
        if s.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=s.index,
                y=s.values,
                mode="lines",
                name=name,
                yaxis="y2",
            )
        )

    fig.update_layout(
        title=title,
        height=520,
        hovermode="x unified",
        margin=dict(l=40, r=40, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        xaxis=dict(title="Date"),
        yaxis=dict(title=left_title),
        yaxis2=dict(title=right_title, overlaying="y", side="right"),
    )
    return fig


def make_bar_chart(summary_df: pd.DataFrame, period_col: str, asset_filter: str) -> go.Figure:
    df = summary_df.copy()
    if asset_filter != "All":
        df = df[df["Category"] == asset_filter]

    chart_df = pd.DataFrame({
        "Asset": df["Asset"],
        "Value": df[f"{period_col}_raw"],
    }).dropna()

    fig = go.Figure()
    if not chart_df.empty:
        fig.add_trace(
            go.Bar(
                x=chart_df["Asset"],
                y=chart_df["Value"],
                text=[f"{v:+.2f}" for v in chart_df["Value"]],
                textposition="outside",
                name=period_col,
            )
        )

    fig.update_layout(
        title=f"{period_col} Change by Asset",
        height=500,
        margin=dict(l=40, r=20, t=50, b=80),
        xaxis_title="Asset",
        yaxis_title="Change (% for price assets, pp for rate assets)",
    )
    return fig


def make_scatter_with_regression(df: pd.DataFrame, title: str, x_title: str, y_title: str) -> go.Figure:
    fig = go.Figure()

    if not df.empty:
        fig.add_trace(
            go.Scatter(
                x=df["rate_change"],
                y=df["equity_return"],
                mode="markers",
                name="Observations",
                text=[d.strftime("%Y-%m-%d") for d in df.index],
                hovertemplate="Date=%{text}<br>Rate change=%{x:.4f}<br>Equity return=%{y:.4f}<extra></extra>",
            )
        )

        if len(df) >= 2:
            try:
                slope, intercept = np.polyfit(df["rate_change"], df["equity_return"], 1)
                x_line = np.linspace(df["rate_change"].min(), df["rate_change"].max(), 100)
                y_line = slope * x_line + intercept
                fig.add_trace(
                    go.Scatter(
                        x=x_line,
                        y=y_line,
                        mode="lines",
                        name=f"Fit (slope={slope:.3f})",
                    )
                )
            except Exception:
                pass

    fig.update_layout(
        title=title,
        height=520,
        margin=dict(l=40, r=20, t=50, b=40),
        xaxis_title=x_title,
        yaxis_title=y_title,
    )
    return fig


def make_correlation_heatmap(corr_df: pd.DataFrame, title: str) -> go.Figure:
    if corr_df.empty:
        return go.Figure()

    z = corr_df.values
    x = list(corr_df.columns)
    y = list(corr_df.index)

    text = [[f"{v:.2f}" if pd.notna(v) else "" for v in row] for row in z]

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=x,
            y=y,
            zmin=-1,
            zmax=1,
            text=text,
            texttemplate="%{text}",
            hovertemplate="X=%{x}<br>Y=%{y}<br>Corr=%{z:.2f}<extra></extra>",
            colorbar=dict(title="Corr"),
        )
    )

    fig.update_layout(
        title=title,
        height=720,
        margin=dict(l=40, r=20, t=50, b=120),
        xaxis=dict(tickangle=-45),
    )
    return fig


# ============================================================
# Sidebar
# ============================================================
with st.sidebar:
    st.header("Controls")
    refresh = st.button("🔄 Refresh Data", use_container_width=True)

    selected_graph_period = st.selectbox(
        "Graph Period",
        options=list(GRAPH_PERIODS.keys()),
        index=2,
    )

    category_options = ["All", "Oil", "Rate", "Metal", "Commodity", "Crypto", "Equity", "FX"]
    selected_category = st.selectbox("Table Category Filter", category_options, index=0)

    sort_options = [
        "Category",
        "Asset",
        "Current_raw",
        "3M_raw",
        "6M_raw",
        "1Y_raw",
        "5Y_raw",
        "10Y_raw",
    ]
    selected_sort = st.selectbox("Sort By", sort_options, index=3)
    sort_desc = st.checkbox("Descending", value=True)

    show_all_columns = st.checkbox("Show raw helper columns", value=False)
    normalize_chart = st.checkbox("Normalize Multi-Asset chart to 100", value=True)

    st.markdown("---")
    st.caption("Rates use FRED when available. Fallbacks are partial if FRED key is missing.")

if refresh:
    st.cache_data.clear()

# ============================================================
# Load data
# ============================================================
with st.spinner("Loading market data..."):
    series_map = load_all_series()
    summary_df = build_summary_table(series_map)

d10_ok = not series_map.get("DGS10", empty_series()).empty
d2_ok = not series_map.get("DGS2", empty_series()).empty
d3m_ok = not series_map.get("DGS3MO", empty_series()).empty

if not (d10_ok and d2_ok and d3m_ok):
    st.warning(
        "FRED rate data is incomplete. "
        "10Y may still work via Yahoo fallback, 3M may partially work, "
        "but 2Y/spread support is best with FRED_API_KEY."
    )

# ============================================================
# Summary table view
# ============================================================
view_df = summary_df.copy()

if selected_category != "All":
    view_df = view_df[view_df["Category"] == selected_category].copy()

if selected_sort in view_df.columns:
    view_df = view_df.sort_values(selected_sort, ascending=not sort_desc, na_position="last")

display_cols = [
    "Category",
    "Asset",
    "Symbol",
    "Current",
    "3M",
    "6M",
    "1Y",
    "5Y",
    "10Y",
    "Unit",
    "Source",
    "Signal",
]

if show_all_columns:
    raw_cols = [c for c in view_df.columns if c.endswith("_raw") or c == "Current_raw"]
    display_cols += raw_cols

st.subheader("Summary Table")

styled = (
    view_df[display_cols]
    .style
    .map(color_change_cell, subset=["3M", "6M", "1Y", "5Y", "10Y"])
    .map(color_signal_cell, subset=["Signal"])
)

st.dataframe(styled, use_container_width=True, hide_index=True)

# ============================================================
# Quick snapshot
# ============================================================
st.subheader("Quick Risk Snapshot")

oil_3m = get_row_value(summary_df, "WTI Crude Oil", "3M_raw")
gold_1y = get_row_value(summary_df, "Gold", "1Y_raw")
btc_3m = get_row_value(summary_df, "Bitcoin / EUR", "3M_raw")
spx_1y = get_row_value(summary_df, "S&P 500", "1Y_raw")
spread_10y_2y = get_row_value(summary_df, "US 10Y-2Y Spread", "Current_raw")

inflation_score = "High" if oil_3m is not None and oil_3m > 10 else "Moderate" if oil_3m is not None and oil_3m > 0 else "Low"
risk_sentiment = "Risk-On" if (btc_3m is not None and btc_3m > 10 and spx_1y is not None and spx_1y > 10) else "Mixed"
if gold_1y is not None and gold_1y > 10 and (spx_1y is not None and spx_1y < 5):
    risk_sentiment = "Risk-Off"
yield_curve = "Inverted" if spread_10y_2y is not None and spread_10y_2y < 0 else "Normal"
growth = "Strong" if spx_1y is not None and spx_1y > 10 else "Weak" if spx_1y is not None and spx_1y < 0 else "Moderate"

c1, c2, c3, c4 = st.columns(4)
c1.metric("Inflation Pressure", inflation_score)
c2.metric("Risk Sentiment", risk_sentiment)
c3.metric("Yield Curve", yield_curve)
c4.metric("Growth Momentum", growth)

# ============================================================
# Tabs
# ============================================================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    ["Multi-Asset Trend", "Rates", "Equity", "FX", "Change Bar", "Rates vs Equity"]
)

with tab1:
    st.markdown("### Multi-Asset Trend")

    asset_name_to_symbol = {
        "WTI Crude Oil": "CL=F",
        "Brent Crude Oil": "BZ=F",
        "US 10Y Yield": "DGS10",
        "US 2Y Yield": "DGS2",
        "US 3M Yield": "DGS3MO",
        "US 10Y-2Y Spread": "SPREAD_10Y_2Y",
        "US 10Y-3M Spread": "SPREAD_10Y_3M",
        "Gold": "GC=F",
        "Broad Commodities ETF": "DBC",
        "Bitcoin / EUR": "BTC-EUR",
        "S&P 500": "^GSPC",
        "Dow Jones": "^DJI",
        "Nasdaq 100": "^NDX",
        "USD/KRW": "KRW=X",
        "EUR/KRW": "EURKRW=X",
    }

    default_assets = [
        "WTI Crude Oil",
        "US 10Y Yield",
        "Gold",
        "Bitcoin / EUR",
        "S&P 500",
        "USD/KRW",
    ]

    selected_assets = st.multiselect(
        "Select assets",
        options=list(asset_name_to_symbol.keys()),
        default=default_assets,
    )

    has_rate_asset = any(asset in RATE_ASSETS for asset in selected_assets)

    if has_rate_asset:
        st.info("Rate/spread series detected. Dual-axis view is recommended for readability.")

    multi_chart_mode = st.radio(
        "Chart Mode",
        ["Single Axis", "Dual Axis (Rates left / Others right)"],
        index=1 if has_rate_asset else 0,
        horizontal=True,
    )

    chart_map = {}
    for name in selected_assets:
        sym = asset_name_to_symbol[name]
        s = filter_series_by_period(series_map.get(sym, empty_series()), selected_graph_period)
        if not s.empty:
            chart_map[name] = s

    if not chart_map:
        st.info("No data available for selected assets.")
    elif multi_chart_mode == "Single Axis":
        fig = make_line_chart(
            chart_map,
            title=f"Multi-Asset Trend ({selected_graph_period})",
            normalize=normalize_chart,
            yaxis_title="Indexed to 100" if normalize_chart else "Price / Level",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        left_map = {}
        right_map = {}

        for name, s in chart_map.items():
            if name in RATE_ASSETS:
                left_map[name] = normalize_to_100(s) if normalize_chart else s
            else:
                right_map[name] = normalize_to_100(s) if normalize_chart else s

        fig = make_dual_axis_chart(
            left_series=left_map,
            right_series=right_map,
            title=f"Multi-Asset Trend ({selected_graph_period})",
            left_title="Rates / Spreads" if not normalize_chart else "Indexed to 100",
            right_title="Assets" if not normalize_chart else "Indexed to 100",
        )
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.markdown("### Rates & Yield Curve")

    rate_map = {
        "US 10Y Yield": filter_series_by_period(series_map.get("DGS10", empty_series()), selected_graph_period),
        "US 2Y Yield": filter_series_by_period(series_map.get("DGS2", empty_series()), selected_graph_period),
        "US 3M Yield": filter_series_by_period(series_map.get("DGS3MO", empty_series()), selected_graph_period),
    }
    rate_map = {k: v for k, v in rate_map.items() if not v.empty}

    if rate_map:
        fig_rates = make_line_chart(
            rate_map,
            title=f"US Rates ({selected_graph_period})",
            normalize=False,
            yaxis_title="%",
        )
        st.plotly_chart(fig_rates, use_container_width=True)
    else:
        st.info("No rate data available.")

    spread_map = {
        "10Y-2Y Spread": filter_series_by_period(series_map.get("SPREAD_10Y_2Y", empty_series()), selected_graph_period),
        "10Y-3M Spread": filter_series_by_period(series_map.get("SPREAD_10Y_3M", empty_series()), selected_graph_period),
    }
    spread_map = {k: v for k, v in spread_map.items() if not v.empty}

    if spread_map:
        fig_spreads = make_line_chart(
            spread_map,
            title=f"Yield Curve Spreads ({selected_graph_period})",
            normalize=False,
            yaxis_title="pp",
        )
        st.plotly_chart(fig_spreads, use_container_width=True)
    else:
        st.info("No spread data available.")

with tab3:
    st.markdown("### Equity Indices")

    equity_map = {
        "S&P 500": filter_series_by_period(series_map.get("^GSPC", empty_series()), selected_graph_period),
        "Dow Jones": filter_series_by_period(series_map.get("^DJI", empty_series()), selected_graph_period),
        "Nasdaq 100": filter_series_by_period(series_map.get("^NDX", empty_series()), selected_graph_period),
    }
    equity_map = {k: v for k, v in equity_map.items() if not v.empty}

    if equity_map:
        fig_eq = make_line_chart(
            equity_map,
            title=f"US Equity Indices ({selected_graph_period})",
            normalize=True,
            yaxis_title="Indexed to 100",
        )
        st.plotly_chart(fig_eq, use_container_width=True)
    else:
        st.info("No equity data available.")

with tab4:
    st.markdown("### FX")

    fx_map = {
        "USD/KRW": filter_series_by_period(series_map.get("KRW=X", empty_series()), selected_graph_period),
        "EUR/KRW": filter_series_by_period(series_map.get("EURKRW=X", empty_series()), selected_graph_period),
    }
    fx_map = {k: v for k, v in fx_map.items() if not v.empty}

    if fx_map:
        fig_fx = make_line_chart(
            fx_map,
            title=f"KRW FX Pairs ({selected_graph_period})",
            normalize=False,
            yaxis_title="KRW",
        )
        st.plotly_chart(fig_fx, use_container_width=True)
    else:
        st.info("No FX data available.")

with tab5:
    st.markdown("### Period Change Comparison")
    selected_bar_period = st.selectbox("Change Period", list(DISPLAY_PERIODS.keys()), index=2, key="bar_period")
    fig_bar = make_bar_chart(summary_df, selected_bar_period, selected_category)
    st.plotly_chart(fig_bar, use_container_width=True)

with tab6:
    st.markdown("### Rates vs Equity")

    d10 = filter_series_by_period(series_map.get("DGS10", empty_series()), selected_graph_period)
    d2 = filter_series_by_period(series_map.get("DGS2", empty_series()), selected_graph_period)
    spx = filter_series_by_period(series_map.get("^GSPC", empty_series()), selected_graph_period)
    ndx = filter_series_by_period(series_map.get("^NDX", empty_series()), selected_graph_period)

    if d10.empty or spx.empty:
        st.warning("Not enough data to compare rates and equity.")
    else:
        st.markdown("#### 1) Level comparison")
        left_series = {"US 10Y Yield": d10}
        if not d2.empty:
            left_series["US 2Y Yield"] = d2

        right_series = {"S&P 500": spx}
        if not ndx.empty:
            right_series["Nasdaq 100"] = ndx

        fig_lvl = make_dual_axis_chart(
            left_series=left_series,
            right_series=right_series,
            title=f"Rates vs Equity Levels ({selected_graph_period})",
            left_title="Yield (%)",
            right_title="Equity Index",
        )
        st.plotly_chart(fig_lvl, use_container_width=True)

        st.markdown("#### 2) Normalized comparison")
        norm_map = {"US 10Y Yield": normalize_to_100(d10)}
        if not d2.empty:
            norm_map["US 2Y Yield"] = normalize_to_100(d2)
        norm_map["S&P 500"] = normalize_to_100(spx)
        if not ndx.empty:
            norm_map["Nasdaq 100"] = normalize_to_100(ndx)

        fig_norm = make_line_chart(
            norm_map,
            title=f"Rates vs Equity Normalized ({selected_graph_period})",
            normalize=False,
            yaxis_title="Indexed to 100",
        )
        st.plotly_chart(fig_norm, use_container_width=True)

        st.markdown("#### 3) Rolling correlation")
        corr_window = st.selectbox(
            "Rolling correlation window",
            options=[20, 30, 60, 90],
            index=1,
            key="corr_window",
        )

        d10_chg = compute_diff(d10)
        d2_chg = compute_diff(d2) if not d2.empty else empty_series()
        spx_ret = compute_return(spx)
        ndx_ret = compute_return(ndx) if not ndx.empty else empty_series()

        corr_map = {
            f"10Y Δ vs S&P500 return ({corr_window}d)": rolling_corr(d10_chg, spx_ret, corr_window),
        }

        if not ndx.empty:
            corr_map[f"10Y Δ vs Nasdaq100 return ({corr_window}d)"] = rolling_corr(d10_chg, ndx_ret, corr_window)

        if not d2.empty:
            corr_map[f"2Y Δ vs S&P500 return ({corr_window}d)"] = rolling_corr(d2_chg, spx_ret, corr_window)
            if not ndx.empty:
                corr_map[f"2Y Δ vs Nasdaq100 return ({corr_window}d)"] = rolling_corr(d2_chg, ndx_ret, corr_window)

        corr_map = {k: v for k, v in corr_map.items() if not v.empty}

        if corr_map:
            fig_corr = make_line_chart(
                corr_map,
                title=f"Rolling Correlation: Rate Changes vs Equity Returns ({selected_graph_period})",
                normalize=False,
                yaxis_title="Correlation",
            )
            st.plotly_chart(fig_corr, use_container_width=True)
        else:
            st.info("Not enough data for rolling correlation.")

        st.markdown("#### 4) Scatter relation")
        scatter_freq = st.selectbox(
            "Scatter aggregation",
            options=["Daily", "Weekly", "Monthly"],
            index=1,
            key="scatter_freq",
        )

        scatter_df_spx = resample_for_scatter(d10, spx, scatter_freq)
        fig_scatter_spx = make_scatter_with_regression(
            scatter_df_spx,
            title=f"10Y Yield Change vs S&P500 Return ({scatter_freq}, {selected_graph_period})",
            x_title=f"10Y yield change ({scatter_freq.lower()}) in pp",
            y_title=f"S&P500 return ({scatter_freq.lower()})",
        )
        st.plotly_chart(fig_scatter_spx, use_container_width=True)

        if not ndx.empty:
            scatter_df_ndx = resample_for_scatter(d10, ndx, scatter_freq)
            fig_scatter_ndx = make_scatter_with_regression(
                scatter_df_ndx,
                title=f"10Y Yield Change vs Nasdaq100 Return ({scatter_freq}, {selected_graph_period})",
                x_title=f"10Y yield change ({scatter_freq.lower()}) in pp",
                y_title=f"Nasdaq100 return ({scatter_freq.lower()})",
            )
            st.plotly_chart(fig_scatter_ndx, use_container_width=True)

        st.markdown("#### 5) Correlation heatmap")
        heatmap_freq = st.selectbox(
            "Heatmap aggregation",
            options=["Daily", "Weekly", "Monthly"],
            index=1,
            key="heatmap_freq",
        )

        corr_input_map = build_corr_input_map(series_map, heatmap_freq, selected_graph_period)
        if corr_input_map:
            corr_input_df = pd.concat(corr_input_map, axis=1).dropna(how="all")
            corr_matrix = corr_input_df.corr()
            fig_heatmap = make_correlation_heatmap(
                corr_matrix,
                title=f"Cross-Asset Correlation Heatmap ({heatmap_freq}, {selected_graph_period})",
            )
            st.plotly_chart(fig_heatmap, use_container_width=True)

            with st.expander("Show correlation matrix table"):
                st.dataframe(corr_matrix.round(3), use_container_width=True)
        else:
            st.info("Not enough data to build the heatmap.")

        st.markdown("#### 6) Latest correlation snapshot")
        latest_corr_rows = []
        for name, s in corr_map.items():
            val = latest_value(s)
            latest_corr_rows.append({
                "Pair": name,
                "Latest Corr": None if val is None else round(val, 3)
            })
        if latest_corr_rows:
            st.dataframe(pd.DataFrame(latest_corr_rows), use_container_width=True, hide_index=True)

# ============================================================
# Raw preview
# ============================================================
with st.expander("See raw latest series data"):
    preview_defaults = [sym for sym in ["CL=F", "DGS10", "GC=F", "BTC-EUR", "^GSPC", "KRW=X"] if sym in series_map]
    preview_options = st.multiselect(
        "Choose symbols to preview",
        options=list(series_map.keys()),
        default=preview_defaults,
    )
    for sym in preview_options:
        st.markdown(f"**{sym}**")
        s = series_map.get(sym, empty_series())
        if s.empty:
            st.write("No data")
        else:
            st.dataframe(s.tail(10).rename(sym).to_frame(), use_container_width=True)

st.markdown("---")
st.caption("Tip: set FRED_API_KEY for best rate/spread coverage.")
