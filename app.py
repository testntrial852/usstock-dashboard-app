import math
import sqlite3
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="US Stock WebApp", page_icon="📈", layout="wide")

DB_FILE = "stocks.db"
ET = ZoneInfo("America/New_York")
RAW_SCORE_MAX = 20.0
TOP5_BUCKETS = ["04:00", "08:00", "10:30"]
DEFAULT_STOCKS = [
    {"ticker": "RKLB", "stock_type": "Watch", "buy_price": None, "shares": None},
    {"ticker": "TSLA", "stock_type": "Watch", "buy_price": None, "shares": None},
    {"ticker": "GOOG", "stock_type": "Holding", "buy_price": 165.0, "shares": 10},
    {"ticker": "NVDA", "stock_type": "Watch", "buy_price": None, "shares": None},
    {"ticker": "AMD", "stock_type": "Watch", "buy_price": None, "shares": None},
    {"ticker": "MU", "stock_type": "Watch", "buy_price": None, "shares": None},
    {"ticker": "VRT", "stock_type": "Watch", "buy_price": None, "shares": None},
]
TOP5_UNIVERSE = [
    "NVDA", "AMD", "MU", "VRT", "TSLA", "GOOG", "META", "AMZN", "AVGO", "ANET",
    "CRWD", "PANW", "SNOW", "PLTR", "RKLB", "SMCI", "ARM", "TSM", "MRVL", "DELL"
]


def now_et():
    return datetime.now(ET)


def get_et_date_str(dt=None):
    dt = dt or now_et()
    return dt.strftime("%Y-%m-%d")


def format_et_dt(dt=None):
    dt = dt or now_et()
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def format_num(val, decimals=2):
    try:
        if val is None or pd.isna(val):
            return "-"
        return f"{float(val):,.{decimals}f}"
    except Exception:
        return "-"


def safe_round(value, digits=2):
    try:
        if value is None or pd.isna(value):
            return None
        return round(float(value), digits)
    except Exception:
        return None


def normalize_score_100(raw_score, max_score=RAW_SCORE_MAX):
    if raw_score is None:
        return None
    try:
        return round((float(raw_score) / float(max_score)) * 100, 1)
    except Exception:
        return None


def get_score_band(score_100):
    if score_100 is None:
        return "N/A"
    if score_100 >= 80:
        return "Elite"
    if score_100 >= 65:
        return "High"
    if score_100 >= 50:
        return "Medium"
    if score_100 >= 35:
        return "Watch"
    return "Low"


def short_text(text, max_len=90):
    if not text:
        return "-"
    text = str(text).strip()
    return text if len(text) <= max_len else text[:max_len - 3] + "..."


def get_conn():
    return sqlite3.connect(DB_FILE, check_same_thread=False)


def ensure_column(conn, table_name, column_name, column_type):
    c = conn.cursor()
    c.execute(f"PRAGMA table_info({table_name})")
    columns = [row[1] for row in c.fetchall()]
    if column_name not in columns:
        c.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type}")


def init_db():
    conn = get_conn()
    c = conn.cursor()

    c.execute("""
    CREATE TABLE IF NOT EXISTS watchlist (
        ticker TEXT PRIMARY KEY,
        stock_type TEXT,
        buy_price REAL,
        shares REAL,
        note TEXT,
        added_at TEXT
    )
    """)

    c.execute("""
    CREATE TABLE IF NOT EXISTS daily_picks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        pick_date TEXT,
        bucket TEXT,
        ticker TEXT,
        price REAL,
        action TEXT,
        confidence TEXT,
        score_raw REAL,
        score_max REAL,
        score_100 REAL,
        score_band TEXT,
        suggested_entry REAL,
        entry_type TEXT,
        entry_zone TEXT,
        fill_probability_today TEXT,
        execution_note TEXT,
        pt REAL,
        sl REAL,
        short_reason TEXT,
        full_reason TEXT,
        change_vs_prev_bucket TEXT,
        created_at TEXT
    )
    """)

    ensure_column(conn, "watchlist", "stock_type", "TEXT")
    ensure_column(conn, "watchlist", "buy_price", "REAL")
    ensure_column(conn, "watchlist", "shares", "REAL")
    ensure_column(conn, "daily_picks", "bucket", "TEXT")
    ensure_column(conn, "daily_picks", "score_max", "REAL")
    ensure_column(conn, "daily_picks", "score_100", "REAL")
    ensure_column(conn, "daily_picks", "score_band", "TEXT")
    ensure_column(conn, "daily_picks", "entry_type", "TEXT")
    ensure_column(conn, "daily_picks", "entry_zone", "TEXT")
    ensure_column(conn, "daily_picks", "fill_probability_today", "TEXT")
    ensure_column(conn, "daily_picks", "execution_note", "TEXT")
    ensure_column(conn, "daily_picks", "short_reason", "TEXT")
    ensure_column(conn, "daily_picks", "full_reason", "TEXT")
    ensure_column(conn, "daily_picks", "change_vs_prev_bucket", "TEXT")

    conn.commit()
    conn.close()


def seed_default_stocks():
    conn = get_conn()
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM watchlist")
    count = c.fetchone()[0]
    if count == 0:
        now = format_et_dt()
        for row in DEFAULT_STOCKS:
            c.execute(
                "INSERT OR IGNORE INTO watchlist (ticker, stock_type, buy_price, shares, note, added_at) VALUES (?, ?, ?, ?, ?, ?)",
                (row["ticker"], row["stock_type"], row["buy_price"], row["shares"], "", now),
            )
    conn.commit()
    conn.close()


def get_watchlist():
    conn = get_conn()
    df = pd.read_sql_query("SELECT * FROM watchlist ORDER BY ticker", conn)
    conn.close()
    return df


def add_stock(ticker, stock_type="Watch", buy_price=None, shares=None, note=""):
    conn = get_conn()
    c = conn.cursor()
    c.execute(
        "INSERT OR REPLACE INTO watchlist (ticker, stock_type, buy_price, shares, note, added_at) VALUES (?, ?, ?, ?, ?, ?)",
        (ticker.upper().strip(), stock_type, buy_price, shares, note, format_et_dt()),
    )
    conn.commit()
    conn.close()


def delete_stock(ticker):
    conn = get_conn()
    c = conn.cursor()
    c.execute("DELETE FROM watchlist WHERE ticker = ?", (ticker,))
    conn.commit()
    conn.close()


@st.cache_data(ttl=900)
def load_price_data(ticker, period="6mo", interval="1d"):
    df = yf.download(ticker, period=period, interval=interval, auto_adjust=True, progress=False, prepost=True)
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    df = df.reset_index()
    return df


def add_indicators(df):
    if df.empty:
        return df
    df = df.copy()
    df["MA20"] = df["Close"].rolling(20).mean()
    df["MA50"] = df["Close"].rolling(50).mean()
    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["RSI14"] = 100 - (100 / (1 + rs))
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["ATR14"] = tr.rolling(14).mean()
    return df


def optimize_entry_execution(action, price, suggested_entry, breakout_level, ma20, atr14):
    if price is None or pd.isna(price):
        return {"entry_type": "Wait", "entry_zone": None, "fill_probability_today": "Low", "execution_note": "Price unavailable.", "pt": None, "sl": None}
    if atr14 is None or pd.isna(atr14) or atr14 <= 0:
        atr14 = max(price * 0.02, 1.0)
    pt = round(price + 2.2 * atr14, 2)
    sl = round(price - 1.4 * atr14, 2)
    if action == "Breakout Confirmed":
        trigger = breakout_level if breakout_level is not None and not pd.isna(breakout_level) else price
        return {"entry_type": "Buy Stop / Stop-Limit", "entry_zone": f"{round(trigger * 1.001, 2)} to {round(trigger * 1.008, 2)}", "fill_probability_today": "Medium", "execution_note": "Breakout confirmed; chase only in a tight breakout zone.", "pt": pt, "sl": sl}
    if action in ["Near Entry", "Buy Setup"]:
        base = suggested_entry if suggested_entry is not None and not pd.isna(suggested_entry) else (ma20 if ma20 is not None and not pd.isna(ma20) else price)
        return {"entry_type": "Limit", "entry_zone": f"{round(base - 0.25 * atr14, 2)} to {round(base + 0.25 * atr14, 2)}", "fill_probability_today": "High" if abs(price - base) <= 0.3 * atr14 else "Medium", "execution_note": "Prefer entering on controlled pullback / support retest.", "pt": pt, "sl": sl}
    if action == "Breakout Watch":
        trigger = breakout_level if breakout_level is not None and not pd.isna(breakout_level) else price
        return {"entry_type": "Watch Trigger", "entry_zone": f"{round(trigger * 0.995, 2)} to {round(trigger * 1.003, 2)}", "fill_probability_today": "Low", "execution_note": "Watch for decisive break with volume before entry.", "pt": pt, "sl": sl}
    if action == "Hold":
        return {"entry_type": "Manage Only", "entry_zone": None, "fill_probability_today": "Low", "execution_note": "Existing position only; manage risk, not a fresh entry.", "pt": pt, "sl": sl}
    return {"entry_type": "Wait", "entry_zone": None, "fill_probability_today": "Low", "execution_note": "No efficient entry now.", "pt": pt, "sl": sl}


def analyze_stock(ticker, stock_type="Watch", buy_price=None, shares=None):
    df = load_price_data(ticker, period="6mo", interval="1d")
    if df is None or df.empty:
        return {
            "ticker": ticker, "stock_type": stock_type, "buy_price": buy_price, "shares": shares, "df": pd.DataFrame(),
            "price": None, "action": "Watch", "confidence": "Low", "score_raw": 0, "score_max": RAW_SCORE_MAX,
            "score_100": 0, "score_band": "Low", "suggested_entry": None, "entry_type": "Wait", "entry_zone": None,
            "fill_probability_today": "Low", "execution_note": "No data available.", "pt": None, "sl": None,
            "short_reason": "No price data available.", "full_reason": "Yahoo Finance returned no data for this ticker.",
            "ma20": None, "ma50": None, "rsi14": None, "atr14": None, "support": None, "resistance": None, "vol_ratio": None,
        }

    df = add_indicators(df)
    last = df.iloc[-1]
    price = safe_round(last.get("Close"))
    ma20 = safe_round(last.get("MA20"))
    ma50 = safe_round(last.get("MA50"))
    rsi14 = safe_round(last.get("RSI14"))
    atr14 = safe_round(last.get("ATR14"))
    recent_20 = df.tail(20)
    resistance = safe_round(recent_20["High"].max()) if "High" in recent_20.columns else None
    support = safe_round(recent_20["Low"].min()) if "Low" in recent_20.columns else None
    avg20_volume = df["Volume"].tail(20).mean() if "Volume" in df.columns else None
    vol_ratio = round(float(last["Volume"]) / float(avg20_volume), 2) if avg20_volume is not None and avg20_volume > 0 else None

    trend_ok = price is not None and ma20 is not None and ma50 is not None and price >= ma20 and ma20 >= ma50
    breakout_confirmed = resistance is not None and price is not None and vol_ratio is not None and price > resistance * 1.002 and vol_ratio >= 1.2
    breakout_watch = resistance is not None and price is not None and vol_ratio is not None and price >= resistance * 0.99 and price <= resistance * 1.002 and vol_ratio >= 0.9
    near_entry_condition = ma20 is not None and atr14 is not None and atr14 > 0 and price is not None and abs(price - ma20) <= 0.75 * atr14 and rsi14 is not None and rsi14 < 72 and ma50 is not None and price >= ma20 and ma20 >= ma50
    overheated = rsi14 is not None and rsi14 >= 75

    raw_score = 0.0
    if trend_ok:
        raw_score += 5.0
    if rsi14 is not None and 50 <= rsi14 <= 68:
        raw_score += 3.0
    elif rsi14 is not None and rsi14 < 75:
        raw_score += 1.5
    if vol_ratio is not None and vol_ratio >= 1.2:
        raw_score += 3.0
    elif vol_ratio is not None and vol_ratio >= 1.0:
        raw_score += 1.5
    if breakout_confirmed:
        raw_score += 5.0
    elif breakout_watch:
        raw_score += 3.0
    elif near_entry_condition:
        raw_score += 3.0
    if overheated:
        raw_score -= 2.0

    raw_score = max(0.0, min(raw_score, RAW_SCORE_MAX))
    score_100 = normalize_score_100(raw_score, RAW_SCORE_MAX)
    score_band = get_score_band(score_100)

    if breakout_confirmed:
        action = "Breakout Confirmed"
        confidence = "High"
        suggested_entry = price
        short_reason = "Price cleared resistance with confirming volume."
        full_reason = f"{ticker} broke above recent resistance near {resistance} with volume ratio {vol_ratio}. Trend is constructive above MA20/MA50, so this is treated as a breakout continuation setup."
    elif near_entry_condition:
        action = "Near Entry"
        confidence = "High" if score_100 is not None and score_100 >= 65 else "Medium"
        suggested_entry = ma20 if ma20 is not None else price
        short_reason = "Trend intact and price is near pullback entry zone."
        full_reason = f"{ticker} remains above MA20 ({ma20}) and MA50 ({ma50}), while price is close to MA20 within ATR tolerance. RSI at {rsi14} suggests the stock is not yet overheated, so this looks like a controlled pullback entry."
    elif trend_ok and not overheated:
        action = "Buy Setup"
        confidence = "Medium"
        suggested_entry = ma20 if ma20 is not None else price
        short_reason = "Trend is healthy, but ideal trigger is not fully formed yet."
        full_reason = f"{ticker} still has a healthy uptrend structure with price {price}, MA20 {ma20}, and MA50 {ma50}. It is not yet a confirmed breakout, but the setup is constructive enough to prepare a limit-style entry plan."
    elif breakout_watch:
        action = "Breakout Watch"
        confidence = "Medium"
        suggested_entry = resistance
        short_reason = "Price is testing resistance; wait for breakout confirmation."
        full_reason = f"{ticker} is sitting close to resistance around {resistance}. Volume ratio at {vol_ratio} is not weak, but a cleaner breakout confirmation is still preferred."
    elif stock_type == "Holding" and trend_ok:
        action = "Hold"
        confidence = "Medium"
        suggested_entry = None
        short_reason = "Trend still intact; manage position rather than add now."
        full_reason = f"{ticker} is already classified as a holding and trend structure remains constructive. This is better treated as position management instead of a fresh entry."
    elif overheated:
        action = "Avoid"
        confidence = "High"
        suggested_entry = None
        short_reason = "Breakout already too extended or overheated."
        full_reason = f"{ticker} looks extended relative to its trend, with RSI around {rsi14}. Risk/reward for a fresh entry is poor unless price resets or consolidates."
    else:
        action = "Watch"
        confidence = "Medium" if score_100 is not None and score_100 >= 35 else "Low"
        suggested_entry = None
        short_reason = "No strong executable setup today."
        full_reason = f"{ticker} does not currently meet breakout or pullback entry conditions. It stays on watch until trend, support, or resistance interaction improves."

    execution = optimize_entry_execution(action, price, suggested_entry, resistance, ma20, atr14)

    return {
        "ticker": ticker, "stock_type": stock_type, "buy_price": buy_price, "shares": shares, "df": df,
        "price": price, "action": action, "confidence": confidence, "score_raw": round(raw_score, 1),
        "score_max": RAW_SCORE_MAX, "score_100": score_100, "score_band": score_band,
        "suggested_entry": suggested_entry, "entry_type": execution["entry_type"], "entry_zone": execution["entry_zone"],
        "fill_probability_today": execution["fill_probability_today"], "execution_note": execution["execution_note"],
        "pt": execution["pt"], "sl": execution["sl"], "short_reason": short_reason, "full_reason": full_reason,
        "ma20": ma20, "ma50": ma50, "rsi14": rsi14, "atr14": atr14, "support": support, "resistance": resistance, "vol_ratio": vol_ratio,
    }


def build_analysis_results():
    watchlist_df = get_watchlist()
    results = []
    for _, row in watchlist_df.iterrows():
        results.append(analyze_stock(row["ticker"], row.get("stock_type", "Watch"), row.get("buy_price"), row.get("shares")))
    return results


def get_current_bucket_label(now_et_dt=None):
    now_et_dt = now_et_dt or now_et()
    current_minutes = now_et_dt.hour * 60 + now_et_dt.minute
    if 4 * 60 <= current_minutes < 8 * 60:
        return "04:00"
    if 8 * 60 <= current_minutes < 10 * 60 + 30:
        return "08:00"
    if current_minutes >= 10 * 60 + 30:
        return "10:30"
    return None


def get_daily_picks_for_bucket(pick_date, bucket_label):
    conn = get_conn()
    df = pd.read_sql_query(
        "SELECT * FROM daily_picks WHERE pick_date = ? AND bucket = ? ORDER BY score_100 DESC, ticker ASC",
        conn,
        params=(pick_date, bucket_label),
    )
    conn.close()
    return df


def replace_daily_picks_for_bucket(bucket_label, rows):
    conn = get_conn()
    c = conn.cursor()

    c.execute("DELETE FROM daily_picks WHERE pick_date = ? AND bucket = ?", (get_et_date_str(), bucket_label))

    for row in rows:
        values = (
            get_et_date_str(),
            bucket_label,
            row.get("ticker"),
            row.get("price"),
            row.get("action"),
            row.get("confidence"),
            row.get("score_raw"),
            row.get("score_max"),
            row.get("score_100"),
            row.get("score_band"),
            row.get("suggested_entry"),
            row.get("entry_type"),
            row.get("entry_zone"),
            row.get("fill_probability_today"),
            row.get("execution_note"),
            row.get("pt"),
            row.get("sl"),
            row.get("short_reason"),
            row.get("full_reason"),
            row.get("change_vs_prev_bucket", "Refreshed"),
            format_et_dt(),
        )

        c.execute("""
            INSERT INTO daily_picks (
                pick_date, bucket, ticker, price, action, confidence,
                score_raw, score_max, score_100, score_band,
                suggested_entry, entry_type, entry_zone, fill_probability_today, execution_note,
                pt, sl, short_reason, full_reason, change_vs_prev_bucket, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, values)

    conn.commit()
    conn.close()

def refresh_active_bucket(force=False):
    bucket_label = get_current_bucket_label()
    if bucket_label is None:
        return None

    universe_rows = [{"ticker": t, "stock_type": "Watch", "buy_price": None, "shares": None} for t in TOP5_UNIVERSE]
    results = []
    for row in universe_rows:
        try:
            results.append(analyze_stock(row["ticker"], row["stock_type"], row["buy_price"], row["shares"]))
        except Exception:
            continue

    results = [r for r in results if r.get("score_100") is not None]
    results.sort(key=lambda x: (x.get("score_100", -1), x.get("score_raw", -1)), reverse=True)
    top5 = results[:5]
    replace_daily_picks_for_bucket(bucket_label, top5)
    return bucket_label


def render_stock_chart(result):
    df = result["df"].copy()
    if df.empty:
        st.warning("No chart data.")
        return
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.72, 0.28])
    fig.add_trace(go.Candlestick(x=df["Date"], open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="Price"), row=1, col=1)
    if "MA20" in df.columns:
        fig.add_trace(go.Scatter(x=df["Date"], y=df["MA20"], mode="lines", name="MA20"), row=1, col=1)
    if "MA50" in df.columns:
        fig.add_trace(go.Scatter(x=df["Date"], y=df["MA50"], mode="lines", name="MA50"), row=1, col=1)
    fig.add_trace(go.Bar(x=df["Date"], y=df["Volume"], name="Volume"), row=2, col=1)
    fig.update_layout(height=560, xaxis_rangeslider_visible=False, margin=dict(l=20, r=20, t=30, b=20))
    st.plotly_chart(fig, use_container_width=True)


def render_dashboard_table(result_subset):
    rows = []
    for r in result_subset:
        rows.append({
            "Ticker": r["ticker"],
            "Type": r["stock_type"],
            "Price": format_num(r["price"], 2),
            "Action": r["action"],
            "Recommendation Strength": r["confidence"],
            "Entry": format_num(r.get("suggested_entry"), 2) if r.get("entry_type") not in ["Wait", "N/A", "Manage Only"] else "-",
            "Entry Type": r.get("entry_type", "-"),
            "Entry Zone": r.get("entry_zone") if r.get("entry_zone") is not None else "-",
            "Fill Prob": r.get("fill_probability_today", "-"),
            "PT": format_num(r.get("pt"), 2),
            "SL": format_num(r.get("sl"), 2),
            "Setup Score /100": format_num(r.get("score_100"), 1),
            "Band": r.get("score_band", "N/A"),
            "Reason": short_text(r.get("short_reason", "-"), 95),
        })
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)


def render_dashboard(results):
    st.subheader("Main Dashboard")
    action_needed = [r for r in results if r.get("action") not in ["Watch", "Avoid", None]]
    holdings = [r for r in results if r.get("stock_type") == "Holding"]

    tab1, tab2, tab3 = st.tabs(["Action Needed", "All Stocks", "Holdings"])

    with tab1:
        if action_needed:
            render_dashboard_table(action_needed)
        else:
            st.info("No action-needed stocks right now.")

    with tab2:
        render_dashboard_table(results)

    with tab3:
        if holdings:
            render_dashboard_table(holdings)
        else:
            st.info("No holdings in watchlist yet.")


def render_top5_section():
    st.subheader("Daily Top 5 High Potential Picks")
    active_bucket = refresh_active_bucket(force=True)
    st.caption(f"Active ET window bucket: {active_bucket or 'Before 04:00 ET'}")
    with st.expander("Top 5 debug"):
        st.caption(f"ET now: {format_et_dt()} | Active bucket: {active_bucket or 'Before 04:00 ET'} | Date: {get_et_date_str()}")
        

    tabs = st.tabs([f"{b} ET" for b in TOP5_BUCKETS])
    today = get_et_date_str()

    for tab, label in zip(tabs, TOP5_BUCKETS):
        with tab:
            df = get_daily_picks_for_bucket(today, label)
            if df.empty:
                st.info("No Top 5 picks stored for this bucket yet.")
                continue
            latest_ts = df["created_at"].iloc[0] if "created_at" in df.columns and not df.empty else format_et_dt()
            st.caption(f"Updated as of {latest_ts} ET")
            show = df[[
                "ticker", "price", "action", "confidence", "score_raw", "score_max", "score_100", "score_band",
                "suggested_entry", "entry_type", "entry_zone", "fill_probability_today", "pt", "sl", "short_reason"
            ]].copy()
            show.columns = [
                "Ticker", "Price", "Action", "Recommendation Strength", "Raw Score", "Max", "Score /100", "Band",
                "Entry", "Entry Type", "Entry Zone", "Fill Prob", "PT", "SL", "Reason"
            ]
            st.dataframe(show, use_container_width=True, hide_index=True)


def render_detail_section(results):
    st.subheader("Stock Detail View")
    tickers = [r["ticker"] for r in results]
    selected_ticker = st.selectbox("Select ticker", tickers)
    selected = next(r for r in results if r["ticker"] == selected_ticker)
    tab1, tab2, tab3, tab4 = st.tabs(["Chart", "Execution Plan", "Reasoning", "Technical"])
    with tab1:
        render_stock_chart(selected)
    with tab2:
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Action", selected.get("action", "-"))
            st.metric("Entry Type", selected.get("entry_type", "-"))
            st.metric("Suggested Entry", format_num(selected.get("suggested_entry"), 2))
            st.metric("Entry Zone", selected.get("entry_zone") or "-")
        with c2:
            st.metric("PT", format_num(selected.get("pt"), 2))
            st.metric("SL", format_num(selected.get("sl"), 2))
            st.metric("Fill Prob", selected.get("fill_probability_today", "-"))
            st.write(selected.get("execution_note", "-"))
    with tab3:
        st.write(selected.get("short_reason", "-"))
        st.write(selected.get("full_reason", "-"))
    with tab4:
        tech = pd.DataFrame([
            {"Metric": "MA20", "Value": format_num(selected.get("ma20"), 2)},
            {"Metric": "MA50", "Value": format_num(selected.get("ma50"), 2)},
            {"Metric": "RSI14", "Value": format_num(selected.get("rsi14"), 2)},
            {"Metric": "ATR14", "Value": format_num(selected.get("atr14"), 2)},
            {"Metric": "Support", "Value": format_num(selected.get("support"), 2)},
            {"Metric": "Resistance", "Value": format_num(selected.get("resistance"), 2)},
            {"Metric": "Volume Ratio", "Value": format_num(selected.get("vol_ratio"), 2)},
        ])
        st.dataframe(tech, use_container_width=True, hide_index=True)


def render_manage_section():
    st.subheader("Manage Stocks")
    with st.form("add_stock_form"):
        c1, c2, c3, c4 = st.columns(4)
        ticker = c1.text_input("Ticker").upper().strip()
        stock_type = c2.selectbox("Type", ["Watch", "Holding"])
        buy_price = c3.number_input("Buy Price", min_value=0.0, value=0.0)
        shares = c4.number_input("Shares", min_value=0.0, value=0.0)
        submitted = st.form_submit_button("Add / Update")
        if submitted and ticker:
            add_stock(ticker=ticker, stock_type=stock_type, buy_price=buy_price if buy_price > 0 else None, shares=shares if shares > 0 else None)
            st.success(f"Saved {ticker}")

    watchlist_df = get_watchlist()
    if not watchlist_df.empty:
        st.dataframe(watchlist_df, use_container_width=True, hide_index=True)
        to_delete = st.selectbox("Delete ticker", [""] + watchlist_df["ticker"].tolist())
        if st.button("Delete Selected") and to_delete:
            delete_stock(to_delete)
            st.success(f"Deleted {to_delete}")


def render_legends():
    st.caption("Top 5 buckets now work as ET time windows: 04:00–07:59, 08:00–10:29, and 10:30 onward. Refreshing the page updates the active bucket with the latest available data in that window.")


def main():
    init_db()
    seed_default_stocks()
    results = build_analysis_results()
    st.title("US Stock WebApp")
    st.caption(f"ET now: {format_et_dt()}")
    render_dashboard(results)
    render_legends()
    render_top5_section()
    render_detail_section(results)
    render_manage_section()


if __name__ == "__main__":
    main()
