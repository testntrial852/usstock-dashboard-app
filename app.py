import sqlite3
from datetime import datetime, time
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
RAW_SCORE_MAX = 22.0
TOP5_BUCKETS = ["04:00", "08:00", "10:30"]
PRIMARY_REGIME_SNAPSHOT_TIME = "09:50 ET"
DEFAULT_STOCKS = [
    {"ticker": "RKLB", "stock_type": "Watch", "buy_price": None, "shares": None},
    {"ticker": "TSLA", "stock_type": "Watch", "buy_price": None, "shares": None},
    {"ticker": "GOOG", "stock_type": "Holding", "buy_price": 165.0, "shares": 10},
    {"ticker": "NVDA", "stock_type": "Watch", "buy_price": None, "shares": None},
    {"ticker": "AMD", "stock_type": "Watch", "buy_price": None, "shares": None},
    {"ticker": "MU", "stock_type": "Watch", "buy_price": None, "shares": None},
    {"ticker": "VRT", "stock_type": "Watch", "buy_price": None, "shares": None},
]

M7_UNIVERSE = ["AAPL", "MSFT", "GOOG", "AMZN", "META", "NVDA", "TSLA"]
AI_INFRA_SEMIS_UNIVERSE = ["AMD", "MU", "AVGO", "ANET", "VRT", "SMCI", "ARM", "TSM", "MRVL", "DELL"]
CLOUD_UNIVERSE = ["SNOW", "CRWD", "PANW", "PLTR", "NET", "DDOG"]
DEFENSE_UNIVERSE = ["LMT", "RTX", "NOC", "GD"]
ENERGY_UNIVERSE = ["XOM", "CVX", "SLB", "FSLR"]
METALS_UNIVERSE = ["FCX", "NEM", "AA", "CLF"]
TOP5_UNIVERSE = list(dict.fromkeys(
    M7_UNIVERSE + AI_INFRA_SEMIS_UNIVERSE + CLOUD_UNIVERSE + DEFENSE_UNIVERSE + ENERGY_UNIVERSE + METALS_UNIVERSE
))


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


def short_text(text, max_len=90):
    if not text:
        return "-"
    text = str(text).strip()
    return text if len(text) <= max_len else text[: max_len - 3] + "..."


def normalize_to_100(raw_score, max_score):
    if raw_score is None or max_score in [None, 0]:
        return None
    try:
        return round((float(raw_score) / float(max_score)) * 100, 1)
    except Exception:
        return None


def score_band(score_100):
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
        readiness_score REAL,
        setup_state TEXT,
        setup_type TEXT,
        trigger_needed TEXT,
        chase_risk TEXT,
        market_alignment TEXT,
        final_score REAL,
        premarket_price REAL,
        premarket_gap_pct REAL,
        premarket_volume REAL,
        premarket_volume_vs_avg REAL,
        created_at TEXT
    )
    """)

    c.execute("""
    CREATE TABLE IF NOT EXISTS market_regime_snapshots (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        regime_date TEXT,
        snapshot_slot TEXT,
        trend_score REAL,
        breadth_score REAL,
        volatility_score REAL,
        total_score REAL,
        regime_label TEXT,
        confidence TEXT,
        live_note TEXT,
        universe_total INTEGER,
        universe_ready INTEGER,
        universe_actionable INTEGER,
        created_at TEXT
    )
    """)

    for col, typ in [("stock_type", "TEXT"), ("buy_price", "REAL"), ("shares", "REAL")]:
        ensure_column(conn, "watchlist", col, typ)

    for col, typ in [
        ("bucket", "TEXT"), ("score_max", "REAL"), ("score_100", "REAL"), ("score_band", "TEXT"),
        ("entry_type", "TEXT"), ("entry_zone", "TEXT"), ("fill_probability_today", "TEXT"),
        ("execution_note", "TEXT"), ("short_reason", "TEXT"), ("full_reason", "TEXT"),
        ("change_vs_prev_bucket", "TEXT"), ("readiness_score", "REAL"), ("setup_state", "TEXT"),
        ("setup_type", "TEXT"), ("trigger_needed", "TEXT"), ("chase_risk", "TEXT"),
        ("market_alignment", "TEXT"), ("final_score", "REAL"), ("premarket_price", "REAL"),
        ("premarket_gap_pct", "REAL"), ("premarket_volume", "REAL"), ("premarket_volume_vs_avg", "REAL"),
    ]:
        ensure_column(conn, "daily_picks", col, typ)

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
    return df.reset_index()


@st.cache_data(ttl=300)
def load_intraday_data(ticker, period="5d", interval="30m"):
    df = yf.download(ticker, period=period, interval=interval, auto_adjust=True, progress=False, prepost=True)
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    return df.reset_index()


def add_indicators(df):
    if df.empty:
        return df
    out = df.copy()
    out["MA20"] = out["Close"].rolling(20).mean()
    out["MA50"] = out["Close"].rolling(50).mean()
    out["MA200"] = out["Close"].rolling(200).mean()
    delta = out["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    out["RSI14"] = 100 - (100 / (1 + rs))
    high_low = out["High"] - out["Low"]
    high_close = (out["High"] - out["Close"].shift()).abs()
    low_close = (out["Low"] - out["Close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    out["ATR14"] = tr.rolling(14).mean()
    return out


def parse_entry_zone(zone):
    if not zone or not isinstance(zone, str) or "to" not in zone:
        return None, None
    try:
        left, right = [x.strip() for x in zone.split("to")]
        return float(left), float(right)
    except Exception:
        return None, None


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


def get_snapshot_slot(dt=None):
    dt = dt or now_et()
    t = dt.time()
    if t < time(8, 0):
        return "Pre-market"
    if t < time(10, 30):
        return "Primary"
    return "Confirmation"


def classify_setup_type(action):
    return {
        "Breakout Confirmed": "Breakout",
        "Breakout Watch": "Breakout",
        "Near Entry": "Pullback",
        "Buy Setup": "Pullback",
        "Hold": "Manage",
        "Avoid": "Avoid",
        "Watch": "Watch",
    }.get(action, "Watch")


def get_trigger_needed(action, resistance=None):
    if action == "Breakout Confirmed":
        return "Hold above breakout zone with volume support"
    if action == "Breakout Watch":
        return f"Break above {format_num(resistance, 2)} with stronger volume"
    if action in ["Near Entry", "Buy Setup"]:
        return "Hold zone and reclaim short-term momentum"
    if action == "Hold":
        return "Manage existing position"
    return "No trigger yet"


def get_chase_risk(price, zone_high, atr14):
    if price is None or zone_high is None:
        return "High"
    stretch = price - zone_high
    if stretch <= 0:
        return "Near"
    if atr14 is None or atr14 <= 0:
        return "Medium"
    if stretch <= 0.35 * atr14:
        return "Medium"
    return "High"


def get_setup_state(action, fill_prob, chase_risk):
    if action == "Avoid":
        return "Broken Setup"
    if action == "Hold":
        return "Manage Only"
    if action == "Breakout Confirmed" and chase_risk != "High":
        return "Ready Now"
    if action == "Near Entry" and fill_prob == "High":
        return "Ready Now"
    if action in ["Near Entry", "Buy Setup"] and fill_prob in ["Medium", "High"]:
        return "Near Zone"
    if action == "Breakout Watch":
        return "Wait Trigger"
    if chase_risk == "High":
        return "Extended"
    if action == "Watch":
        return "Low Priority"
    return "Wait Trigger"


def calc_readiness_score(action, fill_prob, chase_risk, price, base, atr14, vol_ratio):
    score = 0
    score += {"Breakout Confirmed": 35, "Near Entry": 34, "Buy Setup": 24, "Breakout Watch": 20, "Hold": 8}.get(action, 0)
    score += {"High": 26, "Medium": 16, "Low": 6}.get(fill_prob, 0)
    score += {"Near": 18, "Medium": 10, "High": 2}.get(chase_risk, 0)
    if vol_ratio is not None:
        if vol_ratio >= 1.3:
            score += 12
        elif vol_ratio >= 1.0:
            score += 7
        else:
            score += 3
    if price is not None and base is not None and atr14 is not None and atr14 > 0:
        distance_atr = abs(price - base) / atr14
        if distance_atr <= 0.35:
            score += 9
        elif distance_atr <= 0.75:
            score += 6
        else:
            score += 2
    return round(max(0, min(score, 100)), 1)


def optimize_entry_execution(action, price, suggested_entry, breakout_level, ma20, atr14):
    if price is None or pd.isna(price):
        return {
            "entry_type": "Wait",
            "entry_zone": None,
            "fill_probability_today": "Low",
            "execution_note": "Price unavailable.",
            "pt": None,
            "sl": None,
        }
    if atr14 is None or pd.isna(atr14) or atr14 <= 0:
        atr14 = max(price * 0.02, 1.0)

    pt = round(price + 2.2 * atr14, 2)
    sl = round(price - 1.4 * atr14, 2)

    if action == "Breakout Confirmed":
        trigger = breakout_level if breakout_level is not None and not pd.isna(breakout_level) else price
        zone_low = round(trigger * 1.001, 2)
        zone_high = round(trigger * 1.008, 2)
        return {
            "entry_type": "Buy Stop / Stop-Limit",
            "entry_zone": f"{zone_low} to {zone_high}",
            "fill_probability_today": "Medium",
            "execution_note": "Breakout confirmed; chase only in a tight breakout zone.",
            "pt": pt,
            "sl": sl,
        }

    if action in ["Near Entry", "Buy Setup"]:
        base = suggested_entry if suggested_entry is not None and not pd.isna(suggested_entry) else (ma20 if ma20 is not None and not pd.isna(ma20) else price)
        zone_low = round(base - 0.25 * atr14, 2)
        zone_high = round(base + 0.25 * atr14, 2)
        distance_pct = abs(price - base) / base * 100 if base > 0 else 999
        fill_prob = "High" if distance_pct <= 2.5 else ("Medium" if distance_pct <= 5 else "Low")
        return {
            "entry_type": "Limit",
            "entry_zone": f"{zone_low} to {zone_high}",
            "fill_probability_today": fill_prob,
            "execution_note": "Prefer entering on controlled pullback or support retest.",
            "pt": pt,
            "sl": sl,
        }

    if action == "Breakout Watch":
        trigger = breakout_level if breakout_level is not None and not pd.isna(breakout_level) else price
        zone_low = round(trigger * 0.995, 2)
        zone_high = round(trigger * 1.003, 2)
        return {
            "entry_type": "Watch Trigger",
            "entry_zone": f"{zone_low} to {zone_high}",
            "fill_probability_today": "Low",
            "execution_note": "Watch for decisive break with volume before entry.",
            "pt": pt,
            "sl": sl,
        }

    if action == "Hold":
        return {
            "entry_type": "Manage Only",
            "entry_zone": None,
            "fill_probability_today": "Low",
            "execution_note": "Existing position only; manage risk, not a fresh entry.",
            "pt": pt,
            "sl": sl,
        }

    return {
        "entry_type": "Wait",
        "entry_zone": None,
        "fill_probability_today": "Low",
        "execution_note": "No efficient entry now.",
        "pt": pt,
        "sl": sl,
    }


def get_premarket_metrics(ticker, avg20_volume=None):
    pm_price = None
    pm_gap_pct = None
    pm_volume = None
    pm_vol_vs_avg = None

    try:
        ticker_obj = yf.Ticker(ticker)
        fast_info = ticker_obj.fast_info or {}
        pm_price = fast_info.get("preMarketPrice") or fast_info.get("lastPrice")
        prev_close = fast_info.get("previousClose")

        if pm_price is not None and prev_close not in [None, 0]:
            pm_gap_pct = round(((float(pm_price) / float(prev_close)) - 1) * 100, 2)

        intraday = load_intraday_data(ticker, period="5d", interval="30m")
        if not intraday.empty and "Datetime" in intraday.columns and "Volume" in intraday.columns:
            intraday["Datetime"] = pd.to_datetime(intraday["Datetime"], utc=True, errors="coerce")
            intraday["ET"] = intraday["Datetime"].dt.tz_convert(ET)
            today_str = get_et_date_str()
            pm_rows = intraday[
                (intraday["ET"].dt.strftime("%Y-%m-%d") == today_str)
                & (intraday["ET"].dt.time < time(9, 30))
            ]
            if not pm_rows.empty:
                pm_volume = float(pm_rows["Volume"].sum())

        if pm_volume is not None and avg20_volume not in [None, 0]:
            pm_vol_vs_avg = round((pm_volume / float(avg20_volume)) * 100, 2)
    except Exception:
        pass

    return {
        "premarket_price": safe_round(pm_price, 2),
        "premarket_gap_pct": safe_round(pm_gap_pct, 2),
        "premarket_volume": safe_round(pm_volume, 0),
        "premarket_volume_vs_avg": safe_round(pm_vol_vs_avg, 2),
    }


def analyze_stock(ticker, stock_type="Watch", buy_price=None, shares=None, use_premarket=False):
    price_df = load_price_data(ticker, period="6mo", interval="1d")
    if price_df is None or price_df.empty:
        return {
            "ticker": ticker,
            "stock_type": stock_type,
            "buy_price": buy_price,
            "shares": shares,
            "df": pd.DataFrame(),
            "price": None,
            "action": "Watch",
            "confidence": "Low",
            "score_raw": 0.0,
            "score_max": RAW_SCORE_MAX,
            "score_100": 0.0,
            "score_band": "Low",
            "suggested_entry": None,
            "entry_type": "Wait",
            "entry_zone": None,
            "fill_probability_today": "Low",
            "execution_note": "No data available.",
            "pt": None,
            "sl": None,
            "short_reason": "No price data available.",
            "full_reason": "Yahoo Finance returned no data for this ticker.",
            "ma20": None,
            "ma50": None,
            "ma200": None,
            "rsi14": None,
            "atr14": None,
            "support": None,
            "resistance": None,
            "vol_ratio": None,
            "readiness_score": 0.0,
            "setup_state": "Low Priority",
            "setup_type": "Watch",
            "trigger_needed": "No trigger yet",
            "chase_risk": "High",
            "market_alignment": "Neutral",
            "final_score": 0.0,
            "premarket_price": None,
            "premarket_gap_pct": None,
            "premarket_volume": None,
            "premarket_volume_vs_avg": None,
        }

    price_df = add_indicators(price_df)
    last_row = price_df.iloc[-1]

    daily_close = safe_round(last_row.get("Close"))
    ma20 = safe_round(last_row.get("MA20"))
    ma50 = safe_round(last_row.get("MA50"))
    ma200 = safe_round(last_row.get("MA200"))
    rsi14 = safe_round(last_row.get("RSI14"))
    atr14 = safe_round(last_row.get("ATR14"))
    recent_window = price_df.tail(20)
    resistance = safe_round(recent_window["High"].max()) if "High" in recent_window.columns else None
    support = safe_round(recent_window["Low"].min()) if "Low" in recent_window.columns else None
    avg20_volume = price_df["Volume"].tail(20).mean() if "Volume" in price_df.columns else None
    vol_ratio = round(float(last_row["Volume"]) / float(avg20_volume), 2) if avg20_volume is not None and avg20_volume > 0 else None

    pm = get_premarket_metrics(ticker, avg20_volume=avg20_volume) if use_premarket else {
        "premarket_price": None,
        "premarket_gap_pct": None,
        "premarket_volume": None,
        "premarket_volume_vs_avg": None,
    }

    price = pm["premarket_price"] if use_premarket and pm.get("premarket_price") is not None else daily_close

    trend_ok = price is not None and ma20 is not None and ma50 is not None and price >= ma20 and ma20 >= ma50
    strong_trend = trend_ok and ma200 is not None and price >= ma200
    breakout_confirmed = resistance is not None and price is not None and vol_ratio is not None and price > resistance * 1.002 and vol_ratio >= 1.2
    breakout_watch = resistance is not None and price is not None and vol_ratio is not None and price >= resistance * 0.99 and price <= resistance * 1.002 and vol_ratio >= 0.9
    near_entry_condition = (
        ma20 is not None
        and atr14 is not None
        and atr14 > 0
        and price is not None
        and abs(price - ma20) <= 0.75 * atr14
        and rsi14 is not None
        and rsi14 < 72
        and ma50 is not None
        and price >= ma20
        and ma20 >= ma50
    )
    overheated = rsi14 is not None and rsi14 >= 75

    raw_score = 0.0
    if trend_ok:
        raw_score += 5.0
    if strong_trend:
        raw_score += 2.0
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
    if pm.get("premarket_gap_pct") is not None and abs(pm["premarket_gap_pct"]) >= 2:
        raw_score += 1.5
    if pm.get("premarket_volume_vs_avg") is not None and pm["premarket_volume_vs_avg"] >= 3:
        raw_score += 1.0
    if overheated:
        raw_score -= 2.0

    raw_score = max(0.0, min(raw_score, RAW_SCORE_MAX))
    score_100 = normalize_to_100(raw_score, RAW_SCORE_MAX)
    band = score_band(score_100)

    if breakout_confirmed:
        action = "Breakout Confirmed"
        confidence = "High"
        suggested_entry = price
        short_reason = "Price cleared resistance with confirming volume."
        full_reason = f"{ticker} broke above recent resistance near {resistance} with volume ratio {vol_ratio}. Trend remains constructive above MA20 and MA50."
    elif near_entry_condition:
        action = "Near Entry"
        confidence = "High" if score_100 is not None and score_100 >= 65 else "Medium"
        suggested_entry = ma20 if ma20 is not None else price
        short_reason = "Trend intact and price is near pullback entry zone."
        full_reason = f"{ticker} remains above MA20 ({ma20}) and MA50 ({ma50}), while price is close to MA20 within ATR tolerance."
    elif trend_ok and not overheated:
        action = "Buy Setup"
        confidence = "Medium"
        suggested_entry = ma20 if ma20 is not None else price
        short_reason = "Trend is healthy, but ideal trigger is not fully formed yet."
        full_reason = f"{ticker} still has a healthy uptrend structure with price {price}, MA20 {ma20}, and MA50 {ma50}."
    elif breakout_watch:
        action = "Breakout Watch"
        confidence = "Medium"
        suggested_entry = resistance
        short_reason = "Price is testing resistance; wait for breakout confirmation."
        full_reason = f"{ticker} is sitting close to resistance around {resistance}. Volume ratio at {vol_ratio} is not weak, but a cleaner breakout confirmation is preferred."
    elif stock_type == "Holding" and trend_ok:
        action = "Hold"
        confidence = "Medium"
        suggested_entry = None
        short_reason = "Trend still intact; manage position rather than add now."
        full_reason = f"{ticker} is already classified as a holding and trend structure remains constructive."
    elif overheated:
        action = "Avoid"
        confidence = "High"
        suggested_entry = None
        short_reason = "Breakout already too extended or overheated."
        full_reason = f"{ticker} looks extended relative to trend, with RSI around {rsi14}."
    else:
        action = "Watch"
        confidence = "Medium" if score_100 is not None and score_100 >= 35 else "Low"
        suggested_entry = None
        short_reason = "No strong executable setup today."
        full_reason = f"{ticker} does not currently meet breakout or pullback entry conditions."

    execution = optimize_entry_execution(action, price, suggested_entry, resistance, ma20, atr14)
    _, zone_high = parse_entry_zone(execution.get("entry_zone"))
    base_price = suggested_entry if suggested_entry is not None else ma20
    chase_risk = get_chase_risk(price, zone_high, atr14)
    readiness_score = calc_readiness_score(action, execution.get("fill_probability_today"), chase_risk, price, base_price, atr14, vol_ratio)
    setup_state = get_setup_state(action, execution.get("fill_probability_today"), chase_risk)
    setup_type = classify_setup_type(action)
    trigger_needed = get_trigger_needed(action, resistance)

    return {
        "ticker": ticker,
        "stock_type": stock_type,
        "buy_price": buy_price,
        "shares": shares,
        "df": price_df,
        "price": price,
        "daily_close": daily_close,
        "action": action,
        "confidence": confidence,
        "score_raw": round(raw_score, 1),
        "score_max": RAW_SCORE_MAX,
        "score_100": score_100,
        "score_band": band,
        "suggested_entry": suggested_entry,
        "entry_type": execution["entry_type"],
        "entry_zone": execution["entry_zone"],
        "fill_probability_today": execution["fill_probability_today"],
        "execution_note": execution["execution_note"],
        "pt": execution["pt"],
        "sl": execution["sl"],
        "short_reason": short_reason,
        "full_reason": full_reason,
        "ma20": ma20,
        "ma50": ma50,
        "ma200": ma200,
        "rsi14": rsi14,
        "atr14": atr14,
        "support": support,
        "resistance": resistance,
        "vol_ratio": vol_ratio,
        "readiness_score": readiness_score,
        "setup_state": setup_state,
        "setup_type": setup_type,
        "trigger_needed": trigger_needed,
        "chase_risk": chase_risk,
        "market_alignment": "Neutral",
        "final_score": 0.0,
        **pm,
    }


def compute_market_regime(results_for_universe):
    spy_result = analyze_stock("SPY")
    qqq_result = analyze_stock("QQQ")
    vix_data = load_price_data("^VIX", period="3mo", interval="1d")
    vix_level = safe_round(vix_data.iloc[-1].get("Close"), 2) if vix_data is not None and not vix_data.empty else None

    trend_score = 0.0
    for idx_result in [spy_result, qqq_result]:
        if idx_result.get("price") is not None and idx_result.get("ma20") is not None and idx_result.get("ma50") is not None:
            if idx_result["price"] >= idx_result["ma20"]:
                trend_score += 8
            if idx_result["ma20"] >= idx_result["ma50"]:
                trend_score += 8
            if idx_result.get("rsi14") is not None and 45 <= idx_result["rsi14"] <= 70:
                trend_score += 1.5
    trend_score = min(35.0, round(trend_score, 1))

    actionable = [r for r in results_for_universe if r.get("action") not in ["Watch", "Avoid", None]]
    ready = [r for r in results_for_universe if r.get("setup_state") == "Ready Now"]
    above_ma50 = [r for r in results_for_universe if r.get("price") is not None and r.get("ma50") is not None and r["price"] >= r["ma50"]]

    breadth_ratio = len(above_ma50) / len(results_for_universe) if results_for_universe else 0
    actionable_ratio = len(actionable) / len(results_for_universe) if results_for_universe else 0
    ready_ratio = len(ready) / len(results_for_universe) if results_for_universe else 0
    breadth_score = round(min(30.0, breadth_ratio * 16 + actionable_ratio * 8 + ready_ratio * 6), 1)

    if vix_level is None:
        volatility_score = 12.0
    elif vix_level >= 30:
        volatility_score = 4.0
    elif vix_level >= 24:
        volatility_score = 8.0
    elif vix_level >= 20:
        volatility_score = 12.0
    elif vix_level >= 16:
        volatility_score = 16.0
    else:
        volatility_score = 20.0

    total_score = round(trend_score + breadth_score + volatility_score, 1)
    if total_score >= 70:
        regime_label, confidence = "Buy Bias", "High"
    elif total_score >= 45:
        regime_label, confidence = "Selective Buy", "Medium"
    else:
        regime_label, confidence = "Defensive", "High"

    return {
        "trend_score": trend_score,
        "breadth_score": breadth_score,
        "volatility_score": volatility_score,
        "total_score": total_score,
        "regime_label": regime_label,
        "confidence": confidence,
        "live_note": f"SPY {spy_result.get('action','-')} | QQQ {qqq_result.get('action','-')} | VIX {format_num(vix_level,2)} | Ready {len(ready)}/{len(results_for_universe)}",
        "universe_total": len(results_for_universe),
        "universe_ready": len(ready),
        "universe_actionable": len(actionable),
        "created_at": format_et_dt(),
    }


def save_market_regime_snapshot(regime):
    conn = get_conn()
    c = conn.cursor()
    regime_date = get_et_date_str()
    snapshot_slot = get_snapshot_slot()
    c.execute("DELETE FROM market_regime_snapshots WHERE regime_date = ? AND snapshot_slot = ?", (regime_date, snapshot_slot))
    c.execute(
        "INSERT INTO market_regime_snapshots (regime_date, snapshot_slot, trend_score, breadth_score, volatility_score, total_score, regime_label, confidence, live_note, universe_total, universe_ready, universe_actionable, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            regime_date,
            snapshot_slot,
            regime["trend_score"],
            regime["breadth_score"],
            regime["volatility_score"],
            regime["total_score"],
            regime["regime_label"],
            regime["confidence"],
            regime["live_note"],
            regime["universe_total"],
            regime["universe_ready"],
            regime["universe_actionable"],
            regime["created_at"],
        ),
    )
    conn.commit()
    conn.close()


def apply_market_alignment(results, regime_label):
    for result in results:
        bonus = 0
        alignment = "Neutral"

        if regime_label == "Buy Bias":
            if result.get("setup_state") == "Ready Now":
                alignment, bonus = "Strong", 10
            elif result.get("setup_state") in ["Near Zone", "Wait Trigger"]:
                alignment, bonus = "Supportive", 5
        elif regime_label == "Selective Buy":
            if result.get("setup_state") == "Ready Now":
                alignment, bonus = "Supportive", 5
            elif result.get("setup_state") in ["Near Zone", "Wait Trigger"]:
                alignment, bonus = "Neutral", 2
            else:
                alignment, bonus = "Cautious", -4
        else:
            if result.get("setup_state") == "Ready Now":
                alignment, bonus = "Cautious", -8
            else:
                alignment, bonus = "Risk-Off", -12

        result["market_alignment"] = alignment
        result["final_score"] = round(
            0.55 * result.get("score_100", 0)
            + 0.30 * result.get("readiness_score", 0)
            + 0.15 * (50 + bonus),
            1,
        )
    return results


def get_daily_picks_for_bucket(pick_date, bucket_label):
    conn = get_conn()
    stored_df = pd.read_sql_query(
        "SELECT * FROM daily_picks WHERE pick_date = ? AND bucket = ? ORDER BY final_score DESC, score_100 DESC, ticker ASC",
        conn,
        params=(pick_date, bucket_label),
    )
    conn.close()
    return stored_df


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
            row.get("readiness_score"),
            row.get("setup_state"),
            row.get("setup_type"),
            row.get("trigger_needed"),
            row.get("chase_risk"),
            row.get("market_alignment"),
            row.get("final_score"),
            row.get("premarket_price"),
            row.get("premarket_gap_pct"),
            row.get("premarket_volume"),
            row.get("premarket_volume_vs_avg"),
            format_et_dt(),
        )
        c.execute(
            "INSERT INTO daily_picks (pick_date, bucket, ticker, price, action, confidence, score_raw, score_max, score_100, score_band, suggested_entry, entry_type, entry_zone, fill_probability_today, execution_note, pt, sl, short_reason, full_reason, change_vs_prev_bucket, readiness_score, setup_state, setup_type, trigger_needed, chase_risk, market_alignment, final_score, premarket_price, premarket_gap_pct, premarket_volume, premarket_volume_vs_avg, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            values,
        )

    conn.commit()
    conn.close()


def build_watchlist_results(regime_label):
    watchlist_df = get_watchlist()
    watchlist_results = []

    for _, row in watchlist_df.iterrows():
        analyzed = analyze_stock(
            row["ticker"],
            row.get("stock_type", "Watch"),
            row.get("buy_price"),
            row.get("shares"),
            use_premarket=False,
        )
        watchlist_results.append(analyzed)

    return apply_market_alignment(watchlist_results, regime_label)


def refresh_active_bucket():
    bucket_label = get_current_bucket_label()
    if bucket_label is None:
        return None, None

    use_premarket = bucket_label == "08:00"
    universe_results = [analyze_stock(ticker, use_premarket=use_premarket) for ticker in TOP5_UNIVERSE]
    regime = compute_market_regime([analyze_stock(ticker) for ticker in TOP5_UNIVERSE])
    save_market_regime_snapshot(regime)
    universe_results = apply_market_alignment(universe_results, regime["regime_label"])

    if bucket_label == "08:00":
        for result in universe_results:
            pm_bonus = 0
            if result.get("premarket_gap_pct") is not None and abs(result["premarket_gap_pct"]) >= 2:
                pm_bonus += 4
            if result.get("premarket_volume_vs_avg") is not None and result["premarket_volume_vs_avg"] >= 3:
                pm_bonus += 3
            if result.get("chase_risk") == "High":
                pm_bonus -= 3
            result["final_score"] = round(result.get("final_score", 0) + pm_bonus, 1)

    actionable = [r for r in universe_results if r.get("action") not in ["Avoid", "Watch", None]]
    actionable.sort(key=lambda x: x.get("final_score", 0), reverse=True)
    top5 = actionable[:5]
    replace_daily_picks_for_bucket(bucket_label, top5)
    return bucket_label, regime


def render_regime_bar(regime):
    st.subheader("Market Regime + Snapshot")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Today Regime", regime.get("regime_label", "-"))
    c1.caption(f"Confidence: {regime.get('confidence', '-')}")
    c2.metric("Snapshot Time", regime.get("created_at", "-"))
    c2.caption(f"Primary schedule: {PRIMARY_REGIME_SNAPSHOT_TIME}")
    c3.metric("Universe Ready", f"{regime.get('universe_ready', 0)}/{regime.get('universe_total', 0)}")
    c3.caption(f"Actionable: {regime.get('universe_actionable', 0)}")
    c4.metric("Regime Score", format_num(regime.get("total_score"), 1))
    c4.caption(
        f"Trend {format_num(regime.get('trend_score'),1)} | Breadth {format_num(regime.get('breadth_score'),1)} | Vol {format_num(regime.get('volatility_score'),1)}"
    )
    st.caption(f"Live monitor: {regime.get('live_note', '-')}")


def render_dashboard_table(result_subset):
    rows = []
    for result in result_subset:
        rows.append({
            "Ticker": result["ticker"],
            "Type": result["stock_type"],
            "Price": format_num(result["price"], 2),
            "Action": result["action"],
            "Setup State": result.get("setup_state", "-"),
            "Setup Type": result.get("setup_type", "-"),
            "Readiness /100": format_num(result.get("readiness_score"), 1),
            "Setup Score /100": format_num(result.get("score_100"), 1),
            "Final Rank": format_num(result.get("final_score"), 1),
            "Entry Zone": result.get("entry_zone") or "-",
            "Trigger Needed": short_text(result.get("trigger_needed", "-"), 55),
            "Chase Risk": result.get("chase_risk", "-"),
            "Alignment": result.get("market_alignment", "-"),
            "PT": format_num(result.get("pt"), 2),
            "SL": format_num(result.get("sl"), 2),
            "Reason": short_text(result.get("short_reason", "-"), 95),
        })

    table_df = pd.DataFrame(rows)
    st.dataframe(table_df, width="stretch", hide_index=True)


def render_dashboard(results):
    st.subheader("Main Dashboard")
    ready = [r for r in results if r.get("setup_state") == "Ready Now"]
    near_zone = [r for r in results if r.get("setup_state") == "Near Zone"]
    holdings = [r for r in results if r.get("stock_type") == "Holding"]
    all_sorted = sorted(results, key=lambda x: (x.get("final_score", 0), x.get("readiness_score", 0)), reverse=True)

    tab1, tab2, tab3, tab4 = st.tabs(["Ready / Action", "Near Zone", "All Stocks", "Holdings"])
    with tab1:
        render_dashboard_table(ready if ready else all_sorted)
    with tab2:
        if near_zone:
            render_dashboard_table(sorted(near_zone, key=lambda x: x.get("readiness_score", 0), reverse=True))
        else:
            st.info("No near-zone setups right now.")
    with tab3:
        render_dashboard_table(all_sorted)
    with tab4:
        if holdings:
            render_dashboard_table(holdings)
        else:
            st.info("No holdings in watchlist yet.")


def render_top5_section():
    st.subheader("Daily Top 5 High Potential Picks")
    active_bucket, _ = refresh_active_bucket()
    st.caption(f"Active ET window bucket: {active_bucket or 'Before 04:00 ET'}")
    st.caption(f"Universe: {len(TOP5_UNIVERSE)} stocks | 08:00 bucket uses pre-market overlay for ranking.")

    tabs = st.tabs([f"{b} ET" for b in TOP5_BUCKETS])
    today = get_et_date_str()

    for tab, label in zip(tabs, TOP5_BUCKETS):
        with tab:
            stored_df = get_daily_picks_for_bucket(today, label)
            if stored_df.empty:
                st.info("No Top 5 picks stored for this bucket yet.")
                continue

            st.caption(f"Updated as of {stored_df['created_at'].iloc[0]} ET")
            cols = [
                "ticker", "price", "action", "setup_state", "setup_type", "score_100", "readiness_score", "final_score",
                "entry_zone", "trigger_needed", "chase_risk", "market_alignment", "pt", "sl", "short_reason"
            ]
            if label == "08:00":
                cols = [
                    "ticker", "price", "premarket_price", "premarket_gap_pct", "premarket_volume_vs_avg", "action",
                    "setup_state", "score_100", "readiness_score", "final_score", "entry_zone", "trigger_needed",
                    "chase_risk", "short_reason"
                ]
            show_df = stored_df[cols].copy()
            st.dataframe(show_df, width="stretch", hide_index=True)


def render_stock_chart(result):
    chart_df = result["df"].copy()
    if chart_df.empty:
        st.warning("No chart data.")
        return

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.72, 0.28])
    fig.add_trace(
        go.Candlestick(
            x=chart_df["Date"],
            open=chart_df["Open"],
            high=chart_df["High"],
            low=chart_df["Low"],
            close=chart_df["Close"],
            name="Price",
        ),
        row=1,
        col=1,
    )
    for ma in ["MA20", "MA50", "MA200"]:
        if ma in chart_df.columns:
            fig.add_trace(go.Scatter(x=chart_df["Date"], y=chart_df[ma], mode="lines", name=ma), row=1, col=1)

    fig.add_trace(go.Bar(x=chart_df["Date"], y=chart_df["Volume"], name="Volume"), row=2, col=1)
    fig.update_layout(height=560, xaxis_rangeslider_visible=False, margin=dict(l=20, r=20, t=30, b=20))
    st.plotly_chart(fig, width="stretch")


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
        c1.metric("Action", selected.get("action", "-"))
        c1.metric("Setup State", selected.get("setup_state", "-"))
        c1.metric("Setup Type", selected.get("setup_type", "-"))
        c1.metric("Entry Zone", selected.get("entry_zone") or "-")
        c1.metric("Trigger Needed", selected.get("trigger_needed", "-"))
        c2.metric("Readiness /100", format_num(selected.get("readiness_score"), 1))
        c2.metric("Chase Risk", selected.get("chase_risk", "-"))
        c2.metric("PT", format_num(selected.get("pt"), 2))
        c2.metric("SL", format_num(selected.get("sl"), 2))
        st.write(selected.get("execution_note", "-"))
    with tab3:
        st.write(selected.get("short_reason", "-"))
        st.write(selected.get("full_reason", "-"))
    with tab4:
        tech_df = pd.DataFrame([
            {"Metric": "MA20", "Value": format_num(selected.get("ma20"), 2)},
            {"Metric": "MA50", "Value": format_num(selected.get("ma50"), 2)},
            {"Metric": "MA200", "Value": format_num(selected.get("ma200"), 2)},
            {"Metric": "RSI14", "Value": format_num(selected.get("rsi14"), 2)},
            {"Metric": "ATR14", "Value": format_num(selected.get("atr14"), 2)},
            {"Metric": "Support", "Value": format_num(selected.get("support"), 2)},
            {"Metric": "Resistance", "Value": format_num(selected.get("resistance"), 2)},
            {"Metric": "Volume Ratio", "Value": format_num(selected.get("vol_ratio"), 2)},
            {"Metric": "Pre-market Price", "Value": format_num(selected.get("premarket_price"), 2)},
            {"Metric": "Pre-market Gap %", "Value": format_num(selected.get("premarket_gap_pct"), 2)},
        ])
        st.dataframe(tech_df, width="stretch", hide_index=True)


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
            add_stock(
                ticker=ticker,
                stock_type=stock_type,
                buy_price=buy_price if buy_price > 0 else None,
                shares=shares if shares > 0 else None,
            )
            st.success(f"Saved {ticker}")

    watchlist_df = get_watchlist()
    if not watchlist_df.empty:
        st.dataframe(watchlist_df, width="stretch", hide_index=True)
        to_delete = st.selectbox("Delete ticker", [""] + watchlist_df["ticker"].tolist())
        if st.button("Delete Selected") and to_delete:
            delete_stock(to_delete)
            st.success(f"Deleted {to_delete}")


def render_legends():
    st.caption("Market Regime uses snapshot logic rather than pure real-time flipping. Primary decision snapshot is intended for 09:50 ET.")
    st.caption("Top 5 ranking combines Setup Score, Readiness Score, and Market Alignment. The 08:00 ET bucket adds a pre-market overlay instead of fully replacing daily setup logic.")
    st.caption("Suggested local smoke test tickers: NVDA, AMD, TSLA, RKLB.")


def main():
    init_db()
    seed_default_stocks()

    base_universe = [analyze_stock(ticker) for ticker in TOP5_UNIVERSE]
    regime = compute_market_regime(base_universe)
    save_market_regime_snapshot(regime)
    watchlist_results = build_watchlist_results(regime["regime_label"])

    st.title("US Stock WebApp")
    st.caption(f"ET now: {format_et_dt()}")
    render_regime_bar(regime)
    render_dashboard(watchlist_results)
    render_legends()
    render_top5_section()
    render_detail_section(watchlist_results)
    render_manage_section()


if __name__ == "__main__":
    main()
