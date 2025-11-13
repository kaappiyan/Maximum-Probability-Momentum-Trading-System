# mpts_appt.py
"""
MPTS — Streamlit App 
- Single ticker search (yfinance)
- Upload constituents CSV (Symbol/Ticker)
- Sample constituents fallback
- Indicators, composite scoring
"""

import os
import time
from io import StringIO
from typing import List, Dict, Tuple

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objs as go
from concurrent.futures import ThreadPoolExecutor, as_completed

# Optional TA library
try:
    import ta
except Exception:
    ta = None


# ================================
# APP CONFIG
# ================================
st.set_page_config(page_title="MPTS — Scanner", layout="wide")
st.title("MPTS — Maximum Probability Trading System")

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
SNAPSHOT_DIR = os.path.join(PROJECT_DIR, "snapshots")
os.makedirs(SNAPSHOT_DIR, exist_ok=True)

DEFAULT_PERIOD = "1y"
DEFAULT_INTERVAL = "1d"
MAX_WORKERS = 8
TOP_N_DEFAULT = 20

WEIGHTS = {
    "breakout": 0.30,
    "volume": 0.20,
    "rs": 0.20,
    "trend": 0.15,
    "fa": 0.10,
    "atr": 0.05
}


# ================================
# SAMPLE LISTS
# ================================
SAMPLE_NIFTY_1_100 = ["RELIANCE","TCS","INFY","HDFCBANK","ICICIBANK",
                      "HINDUNILVR","KOTAKBANK","LT","HDFC","AXISBANK"]

SAMPLE_NIFTY_101_250 = ["AMARAJABAT","APOLLOHOSP","PIDILITIND",
                        "TITAN","BEL","IDFCFIRSTB"]

SAMPLE_NIFTY_251_500 = ["INDIABULLS","PNBHOUSING","MANAPPURAM","ELECTROSTEEL"]


# ================================
# FETCH OHLCV (YFINANCE)
# ================================
@st.cache_data(show_spinner=False)
def fetch_ohlcv_bulk(
    tickers: List[str], 
    period=DEFAULT_PERIOD, 
    interval=DEFAULT_INTERVAL,
    max_workers=MAX_WORKERS
) -> Dict[str, pd.DataFrame]:

    def fetch_one(tk):
        symbol = tk.strip().upper()
        yf_sym = symbol + ".NS" if "." not in symbol else symbol
        try:
            df = yf.Ticker(yf_sym).history(period=period, interval=interval)
            if df is None or df.empty:
                return symbol, None
            df = df.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
            df.index = pd.to_datetime(df.index).tz_localize(None)
            return symbol, df
        except:
            return symbol, None

    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(fetch_one, t): t for t in tickers}
        for fut in as_completed(futures):
            tk = futures[fut]
            results[tk] = fut.result()[1]
    return results


# ================================
# INDICATORS
# ================================
def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    if d is None or d.empty: return d

    d["sma20"] = d["close"].rolling(20).mean()
    d["sma50"] = d["close"].rolling(50).mean()
    d["sma200"] = d["close"].rolling(200).mean()

    # RSI + ATR (fallback if TA not installed)
    if ta:
        d["rsi14"] = ta.momentum.RSIIndicator(d["close"]).rsi()
        d["atr14"] = ta.volatility.AverageTrueRange(
            d["high"], d["low"], d["close"]
        ).average_true_range()
    else:
        delta = d["close"].diff()
        up = delta.clip(lower=0)
        down = -1 * delta.clip(upper=0)
        roll_up = up.ewm(alpha=1/14, adjust=False).mean()
        roll_down = down.ewm(alpha=1/14, adjust=False).mean()
        rs = roll_up / roll_down
        d["rsi14"] = 100 - (100/(1+rs))

        tr = pd.concat([
            d["high"] - d["low"],
            (d["high"] - d["close"].shift()).abs(),
            (d["low"] - d["close"].shift()).abs()
        ], axis=1).max(axis=1)
        d["atr14"] = tr.rolling(14).mean()

    d["vol20"] = d["volume"].rolling(20).mean()
    d["ret_3m"] = d["close"].pct_change(63)
    d["52wk_high"] = d["close"].rolling(252, min_periods=1).max()
    return d


# ================================
# SCORING SUB-MODULES
# ================================
def score_breakout(df):
    if len(df) < 60: return 0, "Insufficient"
    last = df.iloc[-1]
    prev_ath = df["52wk_high"].shift(1).iloc[-1]
    vol_ratio = last["volume"] / (last["vol20"] or 1)

    if last["close"] >= prev_ath and vol_ratio >= 1.2:
        return min(100, 60 + (vol_ratio-1.2)*200), f"ATH + volume {vol_ratio:.2f}"
    if last["close"] >= prev_ath:
        return 60, "ATH breakout"
    return 0, "No breakout"


def score_volume(df):
    last = df.iloc[-1]
    vol_ratio = last["volume"] / (last["vol20"] or 1)
    return min(100, vol_ratio / 1.5 * 100), f"vol_ratio {vol_ratio:.2f}"


def score_rs(df, seg_ret_series):
    if len(df) < 63: return 0, "Insufficient"
    ret = (df["close"].iloc[-1] / df["close"].iloc[-63]) - 1
    return np.clip(50 + ret * 100, 0, 100), f"3m_ret {ret:.2%}"


def score_trend(df):
    last = df.iloc[-1]
    score = 0
    notes = []

    if last["close"] > last["sma50"] > last["sma200"]:
        score += 70; notes.append("50>200 strong trend")
    elif last["close"] > last["sma200"]:
        score += 50; notes.append("close>200")

    sma200 = df["sma200"].dropna()
    if len(sma200) > 5 and sma200.iloc[-1] > sma200.iloc[-5]:
        score += 15; notes.append("200 rising")

    return min(100, score), "; ".join(notes)


def score_fundamental_sanity(tk):
    try:
        info = yf.Ticker(tk + ".NS").info
        roe = info.get("returnOnEquity") or 0
        de = info.get("debtToEquity") or 0

        s = 50
        n = []
        if roe > 0.12:
            s += 30; n.append("ROE>12%")
        if de < 1.5:
            s += 20; n.append("D/E<1.5")
        return min(100, s), "; ".join(n)

    except:
        return 50, "Info unavailable"


def score_atr(df):
    last = df.iloc[-1]
    atr_pct = (last["atr14"] / last["close"]) if last["atr14"] else 0
    score = max(0, min(100, 100 * (1 - atr_pct/0.08)))
    return score, f"atr_pct {atr_pct:.3f}"


def composite_momentum_score(df, seg_series, tk: str):
    b_s, b_n = score_breakout(df)
    v_s, v_n = score_volume(df)
    r_s, r_n = score_rs(df, seg_series)
    t_s, t_n = score_trend(df)
    fa_s, fa_n = score_fundamental_sanity(tk)
    a_s, a_n = score_atr(df)

    score = (
        WEIGHTS["breakout"] * b_s +
        WEIGHTS["volume"] * v_s +
        WEIGHTS["rs"] * r_s +
        WEIGHTS["trend"] * t_s +
        WEIGHTS["fa"] * fa_s +
        WEIGHTS["atr"] * a_s
    )

    notes = {
        "breakout": b_n, "volume": v_n, "rs": r_n,
        "trend": t_n, "fundamentals": fa_n, "atr": a_n
    }

    return round(score, 2), notes


# ================================
# UI — SIDEBAR
# ================================
st.sidebar.header("Scanner Controls")

segment = st.sidebar.selectbox("Segment", [
    "Nifty 1-100", "Nifty 101-250", "Nifty 251-500"
])

const_mode = st.sidebar.radio("Constituents Source", ["upload", "sample"], index=1)

uploaded = None
if const_mode == "upload":
    uploaded = st.sidebar.file_uploader("Upload CSV with Symbol/Ticker column", type=["csv"])

# Single ticker
st.sidebar.subheader("Single Ticker Search")
single_ticker = st.sidebar.text_input("Ticker (e.g., RELIANCE, INFY)")
search_btn = st.sidebar.button("Search")

add_to_const = st.sidebar.checkbox("Add searched ticker to constituents", value=False)

# Scanner options
st.sidebar.subheader("Scanning")
top_n = st.sidebar.number_input("Top N", 5, 500, TOP_N_DEFAULT)
show_all = st.sidebar.checkbox("Show all results", value=False)
min_avg_vol = st.sidebar.number_input("Min Avg Volume", value=0, step=1000)
scan_btn = st.sidebar.button("Scan")


# ================================
# LOAD CONSTITUENTS
# ================================
constituents = []
debug_msg = ""

if const_mode == "upload" and uploaded is not None:
    try:
        df_up = pd.read_csv(uploaded)
        col = None
        for c in df_up.columns:
            if c.lower() in ("symbol", "ticker"):
                col = c
                break
        if col is None:
            st.sidebar.error("CSV must contain 'Symbol' or 'Ticker' column.")
        else:
            constituents = (
                df_up[col]
                .astype(str)
                .str.upper()
                .str.replace(".NS", "", regex=False)
                .str.strip()
                .tolist()
            )
            debug_msg = "Loaded from upload"
    except Exception as e:
        st.sidebar.error(f"Error reading CSV: {e}")
else:
    # sample fallback
    if segment == "Nifty 1-100": constituents = SAMPLE_NIFTY_1_100
    elif segment == "Nifty 101-250": constituents = SAMPLE_NIFTY_101_250
    else: constituents = SAMPLE_NIFTY_251_500
    debug_msg = "Using sample list"

st.sidebar.markdown(f"Loaded **{len(constituents)} constituents**")
if len(constituents) > 0:
    st.sidebar.write(", ".join(constituents[:20]))

with st.expander("Constituent Debug Info"):
    st.write(debug_msg)


# ================================
# MAIN LAYOUT
# ================================
left, right = st.columns([2, 1])

# ---------------------------------------------------
# SINGLE TICKER ANALYSIS
# ---------------------------------------------------
with left:
    if search_btn and single_ticker.strip():
        tkt = single_ticker.strip().upper()
        st.header(f"Ticker Analysis — {tkt}")

        with st.spinner("Fetching OHLCV..."):
            data = fetch_ohlcv_bulk([tkt])
            df = data.get(tkt)

        if df is None or df.empty:
            st.error("No data found.")
        else:
            df_ind = compute_indicators(df)
            score, notes = composite_momentum_score(df_ind, pd.Series(dtype=float), tkt)

            st.metric("Composite Score", score)
            st.write("Score Breakdown:")
            st.json(notes)

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_ind.index, y=df_ind["close"], name="Close"))
            fig.add_trace(go.Scatter(x=df_ind.index, y=df_ind["sma50"], name="SMA50"))
            fig.add_trace(go.Scatter(x=df_ind.index, y=df_ind["sma200"], name="SMA200"))
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

            if add_to_const:
                if tkt not in constituents:
                    constituents.append(tkt)
                    st.success(f"Added {tkt} to constituents list.")
                else:
                    st.info("Ticker already added.")


# ---------------------------------------------------
# SEGMENT SCANNER
# ---------------------------------------------------
with left:
    st.subheader("Segment Scanner")

    if scan_btn:
        if not constituents:
            st.error("No constituents loaded.")
        else:
            st.info("Fetching data...")

            data = fetch_ohlcv_bulk(constituents)
            processed = {tk: compute_indicators(df) for tk, df in data.items() if df is not None}

            # build segment series
            frames = []
            for tk, df in processed.items():
                frames.append(df["close"].pct_change().rename(tk))
            seg_series = (pd.concat(frames, axis=1).mean(axis=1).cumsum()) if frames else pd.Series(dtype=float)

            rows = []
            for tk, df in processed.items():

                # volume filter
                if min_avg_vol > 0:
                    if df["volume"].rolling(20).mean().iloc[-1] < min_avg_vol:
                        continue

                score, notes = composite_momentum_score(df, seg_series, tk)
                last_close = df["close"].iloc[-1]
                vol_ratio = df["volume"].iloc[-1] / (df["vol20"].iloc[-1] or 1)

                rows.append({
                    "symbol": tk,
                    "score": score,
                    "last_price": last_close,
                    "vol_ratio": round(vol_ratio, 2),
                    "ret_3m": df["ret_3m"].iloc[-1],
                    "sma50": df["sma50"].iloc[-1],
                    "sma200": df["sma200"].iloc[-1],
                    "notes": notes
                })

            if not rows:
                st.warning("No results.")
            else:
                df_res = pd.DataFrame(rows).sort_values("score", ascending=False)
                df_res["rank"] = range(1, len(df_res)+1)

                display_df = df_res if show_all else df_res.head(top_n)
                st.dataframe(display_df[["rank","symbol","score","last_price","vol_ratio","ret_3m","sma50","sma200"]])

                st.download_button(
                    "Download Results",
                    df_res.to_csv(index=False).encode(),
                    file_name="mpts_scan_results.csv",
                    mime="text/csv"
                )

                # Detail expanders
                for _, r in display_df.iterrows():
                    with st.expander(f"{r['rank']}. {r['symbol']} — Score {r['score']}"):
                        st.json(r["notes"])
                        tk = r["symbol"]
                        dfp = processed[tk]

                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=dfp.index, y=dfp["close"], name="Close"))
                        fig.add_trace(go.Scatter(x=dfp.index, y=dfp["sma50"], name="SMA50"))
                        fig.add_trace(go.Scatter(x=dfp.index, y=dfp["sma200"], name="SMA200"))
                        fig.update_layout(height=300)
                        st.plotly_chart(fig, use_container_width=True)


with right:
    st.markdown("### Notes")
    st.write("- Use CSV upload or sample lists for scanning.")
    st.write("- Single ticker tool supports scoring + indicator plotting.")
    st.write("- For production: replace yfinance & sample lists with licensed data.")
