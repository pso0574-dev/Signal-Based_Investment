from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd
import requests
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from dateutil.relativedelta import relativedelta

# =========================================================
# Streamlit page config
# =========================================================
st.set_page_config(
    page_title="FRED Macro Dashboard",
    layout="wide",
)

# =========================================================
# Constants
# =========================================================
FRED_BASE = "https://api.stlouisfed.org/fred"

LOOKBACK_OPTIONS = {
    "1M": relativedelta(months=1),
    "3M": relativedelta(months=3),
    "6M": relativedelta(months=6),
    "1Y": relativedelta(years=1),
    "3Y": relativedelta(years=3),
    "5Y": relativedelta(years=5),
    "10Y": relativedelta(years=10),
}

DEFAULT_SERIES = {
    "Rates": [
        "FEDFUNDS",   # Effective Federal Funds Rate
        "DGS2",       # 2-Year Treasury
        "DGS10",      # 10-Year Treasury
        "T10Y2Y",     # 10Y-2Y Spread
        "MORTGAGE30US",  # 30-Year Mortgage Rate
    ],
    "Liquidity": [
        "M2SL",       # M2 Money Supply
        "WALCL",      # Fed Balance Sheet
    ],
    "Assets": [
        "SP500",      # S&P 500
        "NASDAQCOM",  # Nasdaq Composite
        "CSUSHPINSA", # Case-Shiller US Home Price Index
        "GOLDAMGBD228NLBM",  # Gold
        "DEXUSEU",    # USD per EUR
    ],
    "Credit / Risk": [
        "BAMLH0A0HYM2",   # High Yield OAS
        "VIXCLS",         # VIX
        "UNRATE",         # Unemployment Rate
    ],
}

SERIES_NOTES = {
    "FEDFUNDS": "Fed policy rate",
    "DGS2": "US 2Y Treasury yield",
    "DGS10": "US 10Y Treasury yield",
    "T10Y2Y": "10Y minus 2Y Treasury spread",
    "MORTGAGE30US": "30Y US mortgage rate",
    "M2SL": "M2 money stock",
    "WALCL": "Federal Reserve total assets",
    "SP500": "S&P 500 index",
    "NASDAQCOM": "Nasdaq Composite index",
    "CSUSHPINSA": "US national home price index",
    "GOLDAMGBD228NLBM": "Gold price",
    "DEXUSEU": "USD per EUR",
    "BAMLH0A0HYM2": "US high yield spread",
    "VIXCLS": "VIX",
    "UNRATE": "US unemployment rate",
}

# =========================================================
# Data classes
# =========================================================
@dataclass
class FredSeriesMeta:
    series_id: str
    title: str
    units: str
    frequency: str


# =========================================================
# Helpers
# =========================================================
def safe_float(x: str) -> Optional[float]:
    try:
        if x in (".", "", None):
            return None
        return float(x)
    except Exception:
        return None


@st.cache_data(show_spinner=False, ttl=60 * 60)
def fred_get_series_meta(api_key: str, series_id: str) -> FredSeriesMeta:
    url = f"{FRED_BASE}/series"
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    payload = r.json()

    if "seriess" not in payload or not payload["seriess"]:
        raise ValueError(f"Metadata not found for {series_id}")

    s = payload["seriess"][0]
    return FredSeriesMeta(
        series_id=series_id,
        title=s.get("title", series_id),
        units=s.get("units", ""),
        frequency=s.get("frequency_short", s.get("frequency", "")),
    )


@st.cache_data(show_spinner=False, ttl=60 * 60)
def fred_get_observations(
    api_key: str,
    series_id: str,
    start_date: str,
    end_date: str,
) -> pd.Series:
    url = f"{FRED_BASE}/series/observations"
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "observation_start": start_date,
        "observation_end": end_date,
        "sort_order": "asc",
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    payload = r.json()

    obs = payload.get("observations", [])
    if not obs:
        return pd.Series(dtype="float64", name=series_id)

    df = pd.DataFrame(obs)
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = df["value"].apply(safe_float)
    df = df.dropna(subset=["value"]).copy()
    df = df.set_index("date").sort_index()

    s = df["value"].astype(float)
    s.name = series_id
    return s


def build_master_dataframe(
    api_key: str,
    series_ids: List[str],
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    data = []
    for sid in series_ids:
        try:
            s = fred_get_observations(api_key, sid, start_date, end_date)
            data.append(s)
        except Exception as e:
            st.warning(f"{sid} 로드 실패: {e}")

    if not data:
        return pd.DataFrame()

    df = pd.concat(data, axis=1).sort_index()
    df = df[~df.index.duplicated(keep="last")]
    df = df.ffill()
    return df


def normalize_to_100(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        first_valid = out[col].dropna()
        if first_valid.empty:
            out[col] = pd.NA
        else:
            base = first_valid.iloc[0]
            out[col] = out[col] / base * 100.0
    return out


def compute_change(series: pd.Series, delta: relativedelta) -> Optional[float]:
    s = series.dropna()
    if len(s) < 2:
        return None

    latest_date = s.index.max()
    target_date = latest_date - delta

    hist = s.loc[:target_date]
    if hist.empty:
        return None

    latest_val = s.iloc[-1]
    past_val = hist.iloc[-1]

    if pd.isna(latest_val) or pd.isna(past_val) or past_val == 0:
        return None

    return (latest_val / past_val - 1.0) * 100.0


def latest_snapshot_table(df: pd.DataFrame, meta_map: Dict[str, FredSeriesMeta]) -> pd.DataFrame:
    rows = []
    for col in df.columns:
        s = df[col].dropna()
        if s.empty:
            continue

        latest = s.iloc[-1]
        latest_date = s.index[-1].date()

        row = {
            "Series": col,
            "Title": meta_map.get(col, FredSeriesMeta(col, col, "", "")).title,
            "Latest": latest,
            "Latest Date": str(latest_date),
            "Units": meta_map.get(col, FredSeriesMeta(col, col, "", "")).units,
            "1M %": compute_change(s, LOOKBACK_OPTIONS["1M"]),
            "3M %": compute_change(s, LOOKBACK_OPTIONS["3M"]),
            "6M %": compute_change(s, LOOKBACK_OPTIONS["6M"]),
            "1Y %": compute_change(s, LOOKBACK_OPTIONS["1Y"]),
            "5Y %": compute_change(s, LOOKBACK_OPTIONS["5Y"]),
        }
        rows.append(row)

    snap = pd.DataFrame(rows)
    if snap.empty:
        return snap

    for c in ["Latest", "1M %", "3M %", "6M %", "1Y %", "5Y %"]:
        if c in snap.columns:
            snap[c] = pd.to_numeric(snap[c], errors="coerce")

    return snap.sort_values("Series").reset_index(drop=True)


def format_snapshot(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if out.empty:
        return out

    out["Latest"] = out["Latest"].map(lambda x: f"{x:,.2f}" if pd.notna(x) else "")
    for c in ["1M %", "3M %", "6M %", "1Y %", "5Y %"]:
        out[c] = out[c].map(lambda x: f"{x:+.2f}%" if pd.notna(x) else "")
    return out


def color_signal(x):
    if isinstance(x, str) and x.endswith("%"):
        try:
            val = float(x.replace("%", ""))
            if val > 0:
                return "background-color: rgba(0, 150, 0, 0.15)"
            if val < 0:
                return "background-color: rgba(200, 0, 0, 0.15)"
        except Exception:
            pass
    return ""


def make_dual_axis_plot(df: pd.DataFrame, left_series: str, right_series: str, title: str) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df[left_series],
            mode="lines",
            name=left_series,
            yaxis="y1",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df[right_series],
            mode="lines",
            name=right_series,
            yaxis="y2",
        )
    )

    fig.update_layout(
        title=title,
        xaxis=dict(title="Date"),
        yaxis=dict(title=left_series),
        yaxis2=dict(title=right_series, overlaying="y", side="right"),
        hovermode="x unified",
        height=500,
        legend=dict(orientation="h", y=1.08),
    )
    return fig


# =========================================================
# Sidebar
# =========================================================
st.title("FRED Macro Dashboard")
st.caption("금리 + 자산 + 유동성 자동 대시보드")

with st.sidebar:
    st.header("Settings")

    api_key_default = os.getenv("FRED_API_KEY", "")
    api_key = st.text_input(
        "FRED API Key",
        value=api_key_default,
        type="password",
        help="환경변수 FRED_API_KEY 또는 직접 입력",
    )

    years_back = st.selectbox("기간", [1, 3, 5, 10, 20], index=2)
    end_date = pd.Timestamp.today().normalize()
    start_date = end_date - relativedelta(years=years_back)

    st.markdown("### 기본 시리즈")
    selected_groups = st.multiselect(
        "그룹 선택",
        options=list(DEFAULT_SERIES.keys()),
        default=["Rates", "Liquidity", "Assets"],
    )

    selected_series: List[str] = []
    for grp in selected_groups:
        st.markdown(f"**{grp}**")
        picks = st.multiselect(
            f"{grp} series",
            options=DEFAULT_SERIES[grp],
            default=DEFAULT_SERIES[grp],
            key=f"group_{grp}",
            help=", ".join([f"{s}: {SERIES_NOTES.get(s, '')}" for s in DEFAULT_SERIES[grp]]),
        )
        selected_series.extend(picks)

    selected_series = list(dict.fromkeys(selected_series))

    st.markdown("### 비교 차트 옵션")
    normalize_chart = st.checkbox("Normalize to 100", value=True)
    rolling_corr_window = st.slider("상관관계 창(일)", 30, 252, 90, 10)

# =========================================================
# Validation
# =========================================================
if not api_key:
    st.info("왼쪽 사이드바에 FRED API Key를 입력하세요.")
    st.stop()

if not selected_series:
    st.warning("최소 1개 이상의 시리즈를 선택하세요.")
    st.stop()

# =========================================================
# Load metadata
# =========================================================
meta_map: Dict[str, FredSeriesMeta] = {}
with st.spinner("시리즈 메타데이터 로드 중..."):
    for sid in selected_series:
        try:
            meta_map[sid] = fred_get_series_meta(api_key, sid)
        except Exception as e:
            st.warning(f"{sid} 메타데이터 로드 실패: {e}")

# =========================================================
# Load observations
# =========================================================
with st.spinner("FRED 데이터 로드 중..."):
    df = build_master_dataframe(
        api_key=api_key,
        series_ids=selected_series,
        start_date=start_date.strftime("%Y-%m-%d"),
        end_date=end_date.strftime("%Y-%m-%d"),
    )

if df.empty:
    st.error("데이터를 불러오지 못했습니다. API key 또는 시리즈를 확인하세요.")
    st.stop()

# =========================================================
# Top metrics
# =========================================================
st.subheader("Market Snapshot")

snap_raw = latest_snapshot_table(df, meta_map)
snap_fmt = format_snapshot(snap_raw)

if snap_fmt.empty:
    st.warning("스냅샷을 생성할 수 없습니다.")
else:
    styled = snap_fmt.style.map(color_signal, subset=["1M %", "3M %", "6M %", "1Y %", "5Y %"])
    st.dataframe(styled, use_container_width=True, height=420)

# =========================================================
# Main time-series chart
# =========================================================
st.subheader("Multi-Asset Trend")

chart_df = normalize_to_100(df) if normalize_chart else df.copy()
chart_title = "Normalized to 100" if normalize_chart else "Raw Levels"

fig = go.Figure()
for col in chart_df.columns:
    fig.add_trace(
        go.Scatter(
            x=chart_df.index,
            y=chart_df[col],
            mode="lines",
            name=col,
        )
    )

fig.update_layout(
    title=chart_title,
    xaxis_title="Date",
    yaxis_title="Index=100" if normalize_chart else "Level",
    hovermode="x unified",
    height=600,
    legend=dict(orientation="h", y=1.08),
)
st.plotly_chart(fig, use_container_width=True)

# =========================================================
# Focus panels
# =========================================================
col1, col2 = st.columns(2)

with col1:
    st.subheader("Rates vs Assets")
    possible_left = [c for c in df.columns if c in ["FEDFUNDS", "DGS2", "DGS10", "T10Y2Y", "MORTGAGE30US"]]
    possible_right = [c for c in df.columns if c in ["SP500", "NASDAQCOM", "CSUSHPINSA", "GOLDAMGBD228NLBM"]]

    if possible_left and possible_right:
        left_series = st.selectbox("Left axis", possible_left, index=0)
        right_series = st.selectbox("Right axis", possible_right, index=0)
        dual_fig = make_dual_axis_plot(df[[left_series, right_series]].dropna(), left_series, right_series, "Rates vs Assets")
        st.plotly_chart(dual_fig, use_container_width=True)
    else:
        st.info("금리 1개 + 자산 1개를 선택하면 비교 가능합니다.")

with col2:
    st.subheader("Liquidity vs Assets")
    liq_candidates = [c for c in df.columns if c in ["M2SL", "WALCL"]]
    asset_candidates = [c for c in df.columns if c in ["SP500", "NASDAQCOM", "CSUSHPINSA", "GOLDAMGBD228NLBM"]]

    if liq_candidates and asset_candidates:
        left_series2 = st.selectbox("Liquidity axis", liq_candidates, index=0)
        right_series2 = st.selectbox("Asset axis", asset_candidates, index=0)
        dual_fig2 = make_dual_axis_plot(df[[left_series2, right_series2]].dropna(), left_series2, right_series2, "Liquidity vs Assets")
        st.plotly_chart(dual_fig2, use_container_width=True)
    else:
        st.info("유동성 1개 + 자산 1개를 선택하면 비교 가능합니다.")

# =========================================================
# Correlation
# =========================================================
st.subheader("Rolling Correlation")

corr_candidates = [c for c in df.columns if df[c].dropna().shape[0] > rolling_corr_window + 5]
if len(corr_candidates) >= 2:
    corr_a = st.selectbox("Series A", corr_candidates, index=0, key="corr_a")
    corr_b = st.selectbox("Series B", corr_candidates, index=min(1, len(corr_candidates)-1), key="corr_b")

    tmp = df[[corr_a, corr_b]].dropna().copy()
    tmp["ret_a"] = tmp[corr_a].pct_change()
    tmp["ret_b"] = tmp[corr_b].pct_change()
    tmp["rolling_corr"] = tmp["ret_a"].rolling(rolling_corr_window).corr(tmp["ret_b"])
    tmp = tmp.dropna(subset=["rolling_corr"])

    if not tmp.empty:
        corr_fig = px.line(
            tmp,
            x=tmp.index,
            y="rolling_corr",
            title=f"{corr_a} vs {corr_b} rolling correlation ({rolling_corr_window}d)",
        )
        corr_fig.update_layout(height=420)
        st.plotly_chart(corr_fig, use_container_width=True)
    else:
        st.info("선택한 시리즈의 공통 구간이 충분하지 않습니다.")
else:
    st.info("상관관계를 보려면 최소 2개 이상의 시리즈가 필요합니다.")

# =========================================================
# Scatter view
# =========================================================
st.subheader("Scatter: Rate / Liquidity / Asset Regime")

scatter_x_candidates = [c for c in df.columns if c in selected_series]
scatter_y_candidates = [c for c in df.columns if c in selected_series]
scatter_color_candidates = [c for c in df.columns if c in selected_series]

if len(scatter_x_candidates) >= 2:
    x_col = st.selectbox("X axis", scatter_x_candidates, index=0, key="scatter_x")
    y_col = st.selectbox("Y axis", scatter_y_candidates, index=min(1, len(scatter_y_candidates)-1), key="scatter_y")
    color_col = st.selectbox("Color", scatter_color_candidates, index=min(2, len(scatter_color_candidates)-1), key="scatter_c")

    scatter_df = df[[x_col, y_col, color_col]].dropna().copy()
    scatter_df["date"] = scatter_df.index.strftime("%Y-%m-%d")

    if not scatter_df.empty:
        sc_fig = px.scatter(
            scatter_df,
            x=x_col,
            y=y_col,
            color=color_col,
            hover_data=["date"],
            title=f"{x_col} vs {y_col} (color={color_col})",
        )
        sc_fig.update_layout(height=500)
        st.plotly_chart(sc_fig, use_container_width=True)
    else:
        st.info("선택한 시리즈의 공통 데이터가 부족합니다.")

# =========================================================
# Raw data expander
# =========================================================
with st.expander("Raw Data"):
    st.dataframe(df.tail(300), use_container_width=True)

# =========================================================
# Interpretation helper
# =========================================================
st.subheader("Quick Interpretation")

interp_lines = []

if "DGS10" in df.columns and "SP500" in df.columns:
    dgs10_3m = compute_change(df["DGS10"].dropna(), LOOKBACK_OPTIONS["3M"])
    sp500_3m = compute_change(df["SP500"].dropna(), LOOKBACK_OPTIONS["3M"])
    if dgs10_3m is not None and sp500_3m is not None:
        if dgs10_3m > 0 and sp500_3m > 0:
            interp_lines.append("- 금리 상승과 주가 상승이 동시에 진행 중: 할인율 부담 대비 자산 강세 구간")
        elif dgs10_3m > 0 and sp500_3m < 0:
            interp_lines.append("- 금리 상승과 주가 하락 동행: 전형적인 긴축 압박 구간")
        elif dgs10_3m < 0 and sp500_3m > 0:
            interp_lines.append("- 금리 하락과 주가 상승 동행: 완화 기대 또는 유동성 회복 구간")

if "M2SL" in df.columns and "SP500" in df.columns:
    m2_1y = compute_change(df["M2SL"].dropna(), LOOKBACK_OPTIONS["1Y"])
    sp_1y = compute_change(df["SP500"].dropna(), LOOKBACK_OPTIONS["1Y"])
    if m2_1y is not None and sp_1y is not None:
        if m2_1y > 0 and sp_1y > 0:
            interp_lines.append("- 유동성과 주가가 함께 증가: 자산시장에 우호적")
        elif m2_1y < 0 and sp_1y > 0:
            interp_lines.append("- 유동성 둔화에도 주가 강세: 밸류에이션 확장 가능성 점검 필요")

if "CSUSHPINSA" in df.columns and "MORTGAGE30US" in df.columns:
    mort_1y = compute_change(df["MORTGAGE30US"].dropna(), LOOKBACK_OPTIONS["1Y"])
    home_1y = compute_change(df["CSUSHPINSA"].dropna(), LOOKBACK_OPTIONS["1Y"])
    if mort_1y is not None and home_1y is not None:
        if mort_1y > 0 and home_1y > 0:
            interp_lines.append("- 모기지 금리 상승에도 주택가격 상승 지속: 공급 부족 또는 후행 반응 가능성")
        elif mort_1y > 0 and home_1y < 0:
            interp_lines.append("- 모기지 금리 상승과 주택가격 하락 동행: 전형적인 부동산 압박 구간")

if interp_lines:
    for line in interp_lines:
        st.write(line)
else:
    st.write("- 선택한 시리즈 조합으로 자동 해석 문장을 만들기에는 데이터가 부족합니다.")
