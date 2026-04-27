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
DEFAULT_STOCKS = [
    ("AMD", "Watch", None, None, ""),
    ("NVDA", "Watch", None, None, ""),
    ("MU", "Watch", None, None, ""),
    ("RKLB", "Watch", None, None, ""),
    ("TSLA", "Watch", None, None, ""),
]
BUCKETS = [("04:00 ET", 4, 0), ("08:00 ET", 8, 0), ("10:30 ET", 10, 30)]


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
    try:
        if raw_score is None or pd.isna(raw_score):
            return None
        return round((float(raw_score) / float(max_score)) * 100, 1)
    except Exception:
        return None


def get_score_band(score_100):
    if score_100 is None or pd.isna(score_100):
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
    return text if len(text) <= max_len else text[: max_len - 3] + "..."


def color_action_badge(action):
    mapping = {
        "Ready": "#1f9d55",
        "Near Entry": "#d97706",
        "Breakout Watch": "#2563eb",
        "Hold": "#6b7280",
        "Watch": "#7c3aed",
        "Avoid": "#b91c1c",
    }
    return mapping.get(action, "#475569")


def color_confidence_badge(conf):
    mapping = {"High": "#1d4ed8", "Medium": "#b45309", "Low": "#6b7280"}
    return mapping.get(conf, "#64748b")


def score_band_color(band):
    mapping = {
        "Elite": "#166534",
        "High": "#1d4ed8",
        "Medium": "#b45309",
        "Watch": "#7c3aed",
        "Low": "#6b7280",
        "N/A": "#6b7280",
    }
    return mapping.get(band, "#6b7280")


def render_badge(text, bg):
    return f'''<span style="display:inline-block;padding:4px 10px;border-radius:999px;font-size:12px;font-weight:700;color:white;background:{bg};white-space:nowrap;">{text}</span>'''


def get_bucket_datetime_et(hour, minute, dt=None):
    dt = dt or now_et()
    return dt.replace(hour=hour, minute=minute, second=0, microsecond=0)


def get_due_buckets(dt=None):
    dt = dt or now_et()
    due = []
    for label, hour, minute in BUCKETS:
        if dt >= get_bucket_datetime_et(hour, minute, dt):
            due.append((label, hour, minute))
    return due


def get_latest_due_bucket(dt=None):
    due = get_due_buckets(dt)
    return due[-1] if due else None


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

    c.execute(
        """
    CREATE TABLE IF NOT EXISTS watchlist (
        ticker TEXT PRIMARY KEY,
        stock_type TEXT DEFAULT 'Watch',
        buy_price REAL,
        shares REAL,
        note TEXT,
        added_at TEXT
    )
    """
    )

    c.execute(
        """
    CREATE TABLE IF NOT EXISTS daily_picks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        pick_date TEXT,
        bucket_label TEXT,
        bucket_time_et TEXT,
        updated_at_et TEXT,
        ticker TEXT,
        stock_type TEXT,
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
        created_at TEXT
    )
    """
    )
    conn.commit()

    for col, typ in [
        ("stock_type", "TEXT DEFAULT 'Watch'"),
        ("buy_price", "REAL"),
        ("shares", "REAL"),
        ("note", "TEXT"),
    ]:
        ensure_column(conn, "watchlist", col, typ)

    for col, typ in [
        ("pick_date", "TEXT"),
        ("bucket_label", "TEXT"),
        ("bucket_time_et", "TEXT"),
        ("updated_at_et", "TEXT"),
        ("stock_type", "TEXT"),
        ("score_max", "REAL"),
        ("score_100", "REAL"),
        ("score_band", "TEXT"),
        ("entry_type", "TEXT"),
        ("entry_zone", "TEXT"),
        ("fill_probability_today", "TEXT"),
        ("execution_note", "TEXT"),
        ("pt", "REAL"),
        ("sl", "REAL"),
        ("short_reason", "TEXT"),
        ("full_reason", "TEXT"),
        ("created_at", "TEXT"),
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
        now_str = format_et_dt()
        for ticker, stock_type, buy_price, shares, note in DEFAULT_STOCKS:
            c.execute(
                """
                INSERT OR IGNORE INTO watchlist (ticker, stock_type, buy_price, shares, note, added_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (ticker, stock_type, buy_price, shares, note, now_str),
            )
    conn.commit()
    conn.close()


def get_watchlist():
    conn = get_conn()
    df = pd.read_sql_query("SELECT * FROM watchlist ORDER BY ticker ASC", conn)
    conn.close()
    return df


def add_stock(ticker, stock_type="Watch", buy_price=None, shares=None, note=""):
    conn = get_conn()
    c = conn.cursor()
    c.execute(
        """
        INSERT OR REPLACE INTO watchlist (ticker, stock_type, buy_price, shares, note, added_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
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


def clear_bucket_daily_picks(bucket_label):
    conn = get_conn()
    c = conn.cursor()
    c.execute(
        "DELETE FROM daily_picks WHERE pick_date = ? AND bucket_label = ?",
        (get_et_date_str(), bucket_label),
    )
    conn.commit()
    conn.close()


def save_daily_pick(row, bucket_label, bucket_time_et):
    conn = get_conn()
    c = conn.cursor()
    c.execute(
        """
        INSERT INTO daily_picks (
            pick_date, bucket_label, bucket_time_et, updated_at_et,
            ticker, stock_type, price, action, confidence,
            score_raw, score_max, score_100, score_band,
            suggested_entry, entry_type, entry_zone, fill_probability_today, execution_note,
            pt, sl, short_reason, full_reason, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            get_et_date_str(),
            bucket_label,
            bucket_time_et,
            format_et_dt(),
            row.get("ticker"),
            row.get("stock_type"),
            row.get("price"),
            row.get("action"),
            row.get("Recommendation Strength"),
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
            format_et_dt(),
        ),
    )
    conn.commit()
    conn.close()


def get_daily_picks_by_bucket(bucket_label):
    conn = get_conn()
    df = pd.read_sql_query(
        """
        SELECT * FROM daily_picks
        WHERE pick_date = ? AND bucket_label = ?
        ORDER BY score_100 DESC, ticker ASC
        """,
        conn,
        params=(get_et_date_str(), bucket_label),
    )
    conn.close()
    return df

def get_live_top5_rows(bucket_label):
    snap_df = get_daily_picks_by_bucket(bucket_label)
    if snap_df.empty:
        return pd.DataFrame()

    live_rows = []

    for _, snap in snap_df.iterrows():
        ticker = snap["ticker"]
        stock_type = snap.get("stock_type", "Watch")

        watchlist_df = get_watchlist()
        matched = watchlist_df[watchlist_df["ticker"] == ticker]

        buy_price = None
        shares = None
        if not matched.empty:
            buy_price = matched.iloc[0].get("buy_price")
            shares = matched.iloc[0].get("shares")

        live = analyze_stock(
            ticker=ticker,
            stock_type=stock_type,
            buy_price=buy_price,
            shares=shares
        )

        if "error" in live:
            live_rows.append({
                "Ticker": ticker,
                "Type": stock_type,
                "Price": format_num(snap.get("price"), 2),
                "Action": render_badge(str(snap.get("action", "-")), color_action_badge(str(snap.get("action", "-")))),
                "Recommendation Strength": render_badge(str(snap.get("confidence", "-")), color_confidence_badge(str(snap.get("confidence", "-")))),
                "Setup Score /100": format_num(snap.get("score_100"), 1),
                "Band": render_badge(str(snap.get("score_band", "N/A")), score_band_color(str(snap.get("score_band", "N/A")))),
                "Entry Type": snap.get("entry_type", "-"),
                "Entry Zone": snap.get("entry_zone", "-"),
                "Fill Prob": snap.get("fill_probability_today", "-"),
                "PT": format_num(snap.get("pt"), 2),
                "SL": format_num(snap.get("sl"), 2),
                "Reason": short_text(snap.get("short_reason", "-"), 95),
            })
            continue

        live_rows.append({
            "Ticker": ticker,
            "Type": live["stock_type"],
            "Price": format_num(live["price"], 2),
            "Action": render_badge(live["action"], color_action_badge(live["action"])),
            "Recommendation Strength": render_badge(live["confidence"], color_confidence_badge(live["Confidence"])),
            "Setup Score/100": format_num(live["Setup Score/100"], 1),
            "Band": render_badge(live["score_band"], score_band_color(live["score_band"])),
            "Entry Type": live["entry_type"],
            "Entry Zone": live["entry_zone"],
            "Fill Prob": live["fill_probability_today"],
            "PT": format_num(live["pt"], 2),
            "SL": format_num(live["sl"], 2),
            "Reason": short_text(live["short_reason"], 95),
        })

    return pd.DataFrame(live_rows)

@st.cache_data(ttl=300)
def load_price_data(ticker, period="6mo", interval="1d"):
    df = yf.download(ticker, period=period, interval=interval, auto_adjust=True, progress=False)
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.reset_index()
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    return df


def add_indicators(df):
    if df.empty:
        return df.copy()
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

    df["VolAvg20"] = df["Volume"].rolling(20).mean()
    df["VolumeRatio"] = df["Volume"] / df["VolAvg20"]
    df["Support20"] = df["Low"].rolling(20).min()
    df["Resistance20"] = df["High"].rolling(20).max()
    return df


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
        entry_low = round(trigger * 1.001, 2)
        entry_high = round(trigger * 1.008, 2)
        return {
            "entry_type": "Buy Stop / Stop-Limit",
            "entry_zone": f"{entry_low} to {entry_high}",
            "fill_probability_today": "Medium",
            "execution_note": "Breakout confirmed; chase only in a tight breakout zone.",
            "pt": pt,
            "sl": sl,
        }

    if action in ["Near Entry", "Buy Setup"]:
        base = suggested_entry
        if base is None or pd.isna(base):
            if ma20 is not None and not pd.isna(ma20):
                base = ma20
            else:
                base = price

        entry_low = round(base - 0.25 * atr14, 2)
        entry_high = round(base + 0.25 * atr14, 2)

        fill_prob = "High" if abs(price - base) <= 0.3 * atr14 else "Medium"

        return {
            "entry_type": "Limit",
            "entry_zone": f"{entry_low} to {entry_high}",
            "fill_probability_today": fill_prob,
            "execution_note": "Prefer entering on controlled pullback / support retest.",
            "pt": pt,
            "sl": sl,
        }

    if action == "Breakout Watch":
        trigger = breakout_level if breakout_level is not None and not pd.isna(breakout_level) else price
        watch_low = round(trigger * 0.995, 2)
        watch_high = round(trigger * 1.003, 2)
        return {
            "entry_type": "Watch Trigger",
            "entry_zone": f"{watch_low} to {watch_high}",
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

def analyze_stock(ticker, stock_type="Watch", buy_price=None, shares=None):
    df = load_price_data(ticker, period="6mo", interval="1d")
    if df is None or df.empty:
        return {
            "ticker": ticker,
            "stock_type": stock_type,
            "buy_price": buy_price,
            "shares": shares,
            "df": pd.DataFrame(),
            "price": None,
            "action": "Watch",
            "confidence": "Low",
            "score_raw": 0,
            "score_max": RAW_SCORE_MAX,
            "score_100": 0,
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
    vol_ratio = None
    if avg20_volume and avg20_volume > 0:
        vol_ratio = round(float(last["Volume"]) / float(avg20_volume), 2)

    trend_ok = (
        price is not None and ma20 is not None and ma50 is not None
        and price >= ma20 and ma20 >= ma50
    )

    breakout_confirmed = (
        resistance is not None and price is not None
        and vol_ratio is not None
        and price > resistance * 1.002
        and vol_ratio >= 1.2
    )

    breakout_watch = (
        resistance is not None and price is not None
        and vol_ratio is not None
        and price >= resistance * 0.99
        and price <= resistance * 1.002
        and vol_ratio >= 0.9
    )

    near_entry_condition = (
        ma20 is not None and atr14 is not None and atr14 > 0
        and price is not None
        and abs(price - ma20) <= 0.75 * atr14
        and rsi14 is not None and rsi14 < 72
        and ma50 is not None
        and price >= ma20 and ma20 >= ma50
    )

    overheated = (rsi14 is not None and rsi14 >= 75)

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
        full_reason = (
            f"{ticker} broke above recent resistance near {resistance} with volume ratio {vol_ratio}. "
            f"Trend is constructive above MA20/MA50, so this is treated as a breakout continuation setup."
        )
    elif near_entry_condition:
        action = "Near Entry"
        confidence = "High" if score_100 is not None and score_100 >= 65 else "Medium"
        suggested_entry = ma20 if ma20 is not None else price
        short_reason = "Trend intact and price is near pullback entry zone."
        full_reason = (
            f"{ticker} remains above MA20 ({ma20}) and MA50 ({ma50}), while price is close to MA20 within ATR tolerance. "
            f"RSI at {rsi14} suggests the stock is not yet overheated, so this looks like a controlled pullback entry."
        )
    elif trend_ok and not overheated:
        action = "Buy Setup"
        confidence = "Medium"
        suggested_entry = ma20 if ma20 is not None else price
        short_reason = "Trend is healthy, but ideal trigger is not fully formed yet."
        full_reason = (
            f"{ticker} still has a healthy uptrend structure with price {price}, MA20 {ma20}, and MA50 {ma50}. "
            f"It is not yet a confirmed breakout, but the setup is constructive enough to prepare a limit-style entry plan."
        )
    elif breakout_watch:
        action = "Breakout Watch"
        confidence = "Medium"
        suggested_entry = resistance
        short_reason = "Price is testing resistance; wait for breakout confirmation."
        full_reason = (
            f"{ticker} is sitting close to resistance around {resistance}. "
            f"Volume ratio at {vol_ratio} is not weak, but a cleaner breakout confirmation is still preferred."
        )
    elif stock_type == "Holding" and trend_ok:
        action = "Hold"
        confidence = "Medium"
        suggested_entry = None
        short_reason = "Trend still intact; manage position rather than add now."
        full_reason = (
            f"{ticker} is already classified as a holding and trend structure remains constructive. "
            f"This is better treated as position management instead of a fresh entry."
        )
    elif overheated:
        action = "Avoid"
        confidence = "High"
        suggested_entry = None
        short_reason = "Breakout already too extended or overheated."
        full_reason = (
            f"{ticker} looks extended relative to its trend, with RSI around {rsi14}. "
            f"Risk/reward for a fresh entry is poor unless price resets or consolidates."
        )
    else:
        action = "Watch"
        confidence = "Medium" if score_100 is not None and score_100 >= 35 else "Low"
        suggested_entry = None
        short_reason = "No strong executable setup today."
        full_reason = (
            f"{ticker} does not currently meet breakout or pullback entry conditions. "
            f"It stays on watch until trend, support, or resistance interaction improves."
        )

    execution = optimize_entry_execution(
        action=action,
        price=price,
        suggested_entry=suggested_entry,
        breakout_level=resistance,
        ma20=ma20,
        atr14=atr14
    )

    return {
        "ticker": ticker,
        "stock_type": stock_type,
        "buy_price": buy_price,
        "shares": shares,
        "df": df,
        "price": price,
        "action": action,
        "confidence": confidence,
        "score_raw": round(raw_score, 1),
        "score_max": RAW_SCORE_MAX,
        "score_100": score_100,
        "score_band": score_band,
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
        "rsi14": rsi14,
        "atr14": atr14,
        "support": support,
        "resistance": resistance,
        "vol_ratio": vol_ratio,
    }

def render_html_table(df, column_types=None, title=None):
    column_types = column_types or {}
    if df is None or df.empty:
        return {
        "ticker": ticker,
        "stock_type": stock_type,
        "buy_price": buy_price,
        "shares": shares,
        "df": df,
        "price": price,
        "action": action,
        "confidence": confidence,
        "score_raw": round(raw_score, 1),
        "score_max": RAW_SCORE_MAX,
        "score_100": score_100,
        "score_band": score_band,
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
        "rsi14": rsi14,
        "atr14": atr14,
        "support": support,
        "resistance": resistance,
        "vol_ratio": vol_ratio,
    }


    st.html(
        """
    <style>
    .custom-table-wrap{margin:8px 0 24px 0;border:1px solid #e5e7eb;border-radius:16px;overflow:auto;background:white;box-shadow:0 4px 18px rgba(15,23,42,.05)}
    table.custom-table{width:100%;border-collapse:separate;border-spacing:0;min-width:980px;font-size:14px}
    .custom-table thead th{position:sticky;top:0;background:#f8fafc;color:#334155;text-align:left;padding:14px;border-bottom:1px solid #e5e7eb;font-weight:700;white-space:nowrap}
    .custom-table tbody td{padding:14px;border-bottom:1px solid #eef2f7;color:#0f172a;vertical-align:top;line-height:1.45}
    .custom-table tbody tr:nth-child(even){background:#fcfcfd}
    .custom-table tbody tr:hover{background:#f8fbff}
    .num-col{text-align:right;white-space:nowrap;font-variant-numeric:tabular-nums}
    .reason-col{min-width:260px;color:#334155}
    </style>
    """.strip()
    )

    if title:
        st.markdown(f"### {title}")

    headers = "".join([f"<th>{col}</th>" for col in df.columns])
    rows_html = ""
    for _, row in df.iterrows():
        row_html = "<tr>"
        for col in df.columns:
            value = row[col]
            class_name = column_types.get(col, "")
            row_html += f'<td class="{class_name}">{value}</td>'
        row_html += "</tr>"
        rows_html += row_html

    table_html = f"""
    <div class="custom-table-wrap">
        <table class="custom-table">
            <thead><tr>{headers}</tr></thead>
            <tbody>{rows_html}</tbody>
        </table>
    </div>
    """.strip()
    st.html(table_html)


def render_stock_chart(result):
    df = result["df"].copy()
    if df.empty:
        st.warning("No chart data.")
        return

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06, row_heights=[0.72, 0.28])
    fig.add_trace(
        go.Candlestick(x=df["Date"], open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="Price"),
        row=1,
        col=1,
    )
    fig.add_trace(go.Scatter(x=df["Date"], y=df["MA20"], mode="lines", name="MA20", line=dict(width=1.8)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["Date"], y=df["MA50"], mode="lines", name="MA50", line=dict(width=1.8)), row=1, col=1)
    if result["pt"] is not None:
        fig.add_hline(y=result["pt"], line_dash="dot", line_color="green", row=1, col=1)
    if result["sl"] is not None:
        fig.add_hline(y=result["sl"], line_dash="dot", line_color="red", row=1, col=1)
    bar_colors = ["#16a34a" if c >= o else "#dc2626" for c, o in zip(df["Close"], df["Open"])]
    fig.add_trace(go.Bar(x=df["Date"], y=df["Volume"], marker_color=bar_colors, name="Volume", opacity=0.8), row=2, col=1)
    fig.add_trace(go.Scatter(x=df["Date"], y=df["VolAvg20"], mode="lines", name="Vol Avg20", line=dict(width=1.6, dash="dash")), row=2, col=1)
    fig.update_layout(height=620, margin=dict(l=20, r=20, t=20, b=20), xaxis_rangeslider_visible=False, legend=dict(orientation="h"))
    st.plotly_chart(fig, use_container_width=True)


def build_analysis_results():
    watchlist_df = get_watchlist()
    results = []
    for _, row in watchlist_df.iterrows():
        ticker = row["ticker"]
        stock_type = row.get("stock_type", "Watch")
        buy_price = row.get("buy_price")
        shares = row.get("shares")
        res = analyze_stock(ticker, stock_type=stock_type, buy_price=buy_price, shares=shares)
        results.append(res)
    return [r for r in results if "error" not in r]


def results_to_dataframe(results):
    rows = []
    for r in results:
        rows.append(
            {
                "Ticker": r["ticker"],
                "Type": r["stock_type"],
                "Price": format_num(r["price"], 2),
                "Action": render_badge(
                    r["action"],
                    color_action_badge(r["action"])
                ),
                "Recommendation Strength": render_badge(
                    r["confidence"],
                    color_confidence_badge(r["confidence"])
                ),
                "Entry": (
                    format_num(r.get("suggested_entry"), 2)
                    if r.get("entry_type") not in ["Wait", "N/A", "Manage Only"]
                    else "-"
                ),
                "Entry Type": r.get("entry_type", "-"),
                "Entry Zone": r.get("entry_zone") if r.get("entry_zone") is not None else "-",
                "Fill Prob": r.get("fill_probability_today", "-"),
                "PT": format_num(r.get("pt"), 2),
                "SL": format_num(r.get("sl"), 2),
                "Setup Score /100": format_num(r.get("score_100"), 1),
                "Band": render_badge(
                    r.get("score_band", "N/A"),
                    score_band_color(r.get("score_band", "N/A"))
                ),
                "Reason": short_text(r.get("short_reason", "-"), 95),
                "_sort_score": r.get("score_100") if r.get("score_100") is not None else -1,
            }
        )

    df = pd.DataFrame(rows)
    if not df.empty and "_sort_score" in df.columns:
        df = df.sort_values(by="_sort_score", ascending=False).drop(columns=["_sort_score"])
    return df
    
def is_positive_top_pick(r):
    positive_actions = ["Near Entry", "Breakout Watch", "Breakout Confirmed", "Ready"]

    if r.get("action") not in positive_actions:
        return False

    if r.get("action") == "Ready" and r.get("Recommendation Strength") != "High":
        return False

    if r.get("Recommendation Strength") not in ["High", "Medium"]:
        return False

    if r.get("fill_probability_today") == "Low":
        return False

    score_100 = r.get("score_100")
    if score_100 is None or pd.isna(score_100) or score_100 < 60:
        return False

    return True

def save_top5_to_db(results):
    latest_bucket = get_latest_due_bucket()
    if not latest_bucket:
        return

    bucket_label, hour, minute = latest_bucket
    bucket_time_et = f"{hour:02d}:{minute:02d}"
    clear_bucket_daily_picks(bucket_label)

    filtered = [r for r in results if is_positive_top_pick(r)]

    ranked = sorted(
        filtered,
        key=lambda x: (
            x["score_100"] if x["score_100"] is not None else 0,
            1 if x["Recommendation Strength"] == "High" else 0
        ),
        reverse=True
    )

    for r in ranked[:5]:
        save_daily_pick(r, bucket_label=bucket_label, bucket_time_et=bucket_time_et)


def render_header():
    st.title("📈 US Stock Monitoring WebApp")
    st.caption(f"Updated as of {format_et_dt()} ET")
    st.caption("Market data is fetched from yfinance; indicators are calculated from that data inside the app.")


def render_dashboard(results):
    st.subheader("Main Dashboard")

    tab1, tab2 = st.tabs(["Action Needed", "All Stocks"])

    all_df = results_to_dataframe(results)

    actionable_results = [
        r for r in results
        if r["action"] in ["Near Entry", "Ready", "Breakout Watch", "Breakout Confirmed", "Hold", "Watch"]
    ]
    actionable_df = results_to_dataframe(actionable_results)

    with tab1:
        st.caption("Quick decision view for stocks that currently need attention.")
        if actionable_df.empty:
            st.info("No actionable stocks today.")
        else:
            render_html_table(
                actionable_df,
                column_types={
                    "Price": "num-col",
                    "Entry": "num-col",
                    "PT": "num-col",
                    "SL": "num-col",
                    "Setup Score /100": "num-col",
                    "Reason": "reason-col",
                },
            )

    with tab2:
        st.caption("Complete list of all stocks in your current watchlist / holdings.")
        if all_df.empty:
            st.info("No stocks available.")
        else:
            render_html_table(
                all_df,
                column_types={
                    "Price": "num-col",
                    "Entry": "num-col",
                    "PT": "num-col",
                    "SL": "num-col",
                    "Setup Score /100": "num-col",
                    "Reason": "reason-col",
                },
            )

def render_top5_section():
    st.subheader("Daily Top 5 High Potential Picks")

    st.caption("List membership is fixed by each ET bucket snapshot. Price and setup fields below refresh with latest available data.")

    tabs = st.tabs([b[0] for b in BUCKETS])

    for i, (bucket_label, hour, minute) in enumerate(BUCKETS):
        with tabs[i]:
            snap_df = get_daily_picks_by_bucket(bucket_label)

            st.caption(f"Bucket time: {bucket_label}")

            if snap_df.empty:
                st.info(f"No picks saved yet for {bucket_label}. Refresh after this ET bucket is due.")
                continue

            updated_at = snap_df["updated_at_et"].iloc[0] if "updated_at_et" in snap_df.columns else "-"
            st.caption(f"List fixed as of {updated_at} ET")

            live_df = get_live_top5_rows(bucket_label)

            if live_df.empty:
                st.warning("Unable to refresh live price data for this bucket.")
                continue

            render_html_table(
                live_df,
                column_types={
                    "Price": "num-col",
                    "Setup Score /100": "num-col",
                    "PT": "num-col",
                    "SL": "num-col",
                    "Reason": "reason-col",
                },
            )

def render_detail_section(results):
    st.subheader("Stock Detail View")
    ticker_list = [r["ticker"] for r in results]
    selected_ticker = st.selectbox("Select stock", ticker_list, index=0)
    selected = next((r for r in results if r["ticker"] == selected_ticker), None)
    if not selected:
        st.warning("No detail available.")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Price", format_num(selected["price"], 2))
    c2.metric("Action", selected["action"])
    c3.metric("Recommendation Strength", selected["confidence"])
    c4.metric("Setup Score /100", format_num(selected["score_100"], 1))

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("MA20", format_num(selected["ma20"], 2))
    c6.metric("MA50", format_num(selected["ma50"], 2))
    c7.metric("RSI14", format_num(selected["rsi14"], 2))
    c8.metric("ATR14", format_num(selected["atr14"], 2))

    if selected["stock_type"] == "Holding":
        h1, h2, h3 = st.columns(3)
        h1.metric("Buy Price", format_num(selected.get("buy_price"), 2))
        h2.metric("Shares", format_num(selected.get("shares"), 0))
        h3.metric("Unrealized %", f"{format_num(selected.get('unrealized_pct'), 2)}%" if selected.get("unrealized_pct") is not None else "-")

    tab1, tab2, tab3, tab4 = st.tabs(["Chart", "Execution Plan", "Reasoning", "Score Detail"])

    with tab1:
        render_stock_chart(selected)

    with tab2:
        a1, a2 = st.columns(2)
        a1.markdown(f"**Suggested Entry:** {format_num(selected['suggested_entry'], 2)}")
        a1.markdown(f"**Entry Type:** {selected['entry_type']}")
        a1.markdown(f"**Entry Zone:** {selected['entry_zone']}")
        a2.markdown(f"**Fill Probability Today:** {selected['fill_probability_today']}")
        a2.markdown(f"**PT:** {format_num(selected['pt'], 2)}")
        a2.markdown(f"**SL:** {format_num(selected['sl'], 2)}")
        st.info(selected["execution_note"])

    with tab3:
        st.markdown(f"**Short Reason:** {selected['short_reason']}")
        st.write(selected["full_reason"])

    with tab4:
        st.markdown(f"**Raw Setup Score:** {format_num(selected['score_raw'], 1)}")
        st.markdown(f"**Max Setup Score:** {format_num(selected['score_max'], 1)}")
        st.markdown(f"**Setup Score /100:** {format_num(selected['score_100'], 1)}")
        st.markdown(f"**Band:** {selected['score_band']}")
        st.caption("Setup Score measures overall setup quality. Recommendation Strength reflects how strong the current analyzed recommendation is.")


def render_legends():
    with st.expander("Action Badge Legend"):
        st.markdown(
            """
- **Ready**: Setup may improve soon, but not confirmed yet.
- **Near Entry**: Price is near a workable pullback zone.
- **Breakout Watch**: Close to breakout trigger; wait for confirmation.
- **Hold**: Existing holding can continue to be managed.
- **Watch**: No strong trigger yet; keep monitoring.
- **Avoid**: Too extended or poor setup today.
        """
        )

    with st.expander("Signal Confidence Legend"):
        st.markdown(
            """
- **High**: Multiple conditions align well.
- **Medium**: Setup is reasonable but not ideal.
- **Low**: Weak conviction or incomplete setup.
        """
        )

    with st.expander("Score Band Legend"):
        st.markdown(
            """
- **Elite**: Very strong overall setup.
- **High**: Strong setup with good alignment.
- **Medium**: Decent setup but not top-tier.
- **Watch**: Early-stage or weaker setup; monitor only.
- **Low**: Weak setup; low priority.
        """
        )
    with st.expander("How to Read Recommendation Strength and Setup Score"):
        st.markdown(
            """
- **Recommendation Strength**: How strong the system believes the current recommendation is.
- **Setup Score /100**: Overall quality of the current stock setup, used for comparison and ranking.
- A stock can have a strong setup but still be **Avoid** if it is too extended to enter safely now.
            """
        )

def render_manage_section():
    st.subheader("Manage Stocks")
    with st.form("add_stock_form"):
        col1, col2, col3, col4, col5 = st.columns([1.4, 1.1, 1.1, 1.0, 1.8])
        ticker = col1.text_input("Ticker *").upper().strip()
        stock_type = col2.selectbox("Type *", ["Watch", "Holding"])
        buy_price = col3.number_input("Buy Price", min_value=0.0, step=0.01, value=0.0)
        shares = col4.number_input("Shares", min_value=0.0, step=1.0, value=0.0)
        note = col5.text_input("Note")
        submitted = st.form_submit_button("Add / Update Stock")
        if submitted:
            if not ticker:
                st.error("Ticker is required.")
            elif stock_type == "Holding" and buy_price <= 0:
                st.error("Buy Price is required for Holding.")
            elif stock_type == "Holding" and shares <= 0:
                st.error("Shares is required for Holding.")
            else:
                if stock_type == "Holding":
                    add_stock(ticker, stock_type, buy_price, shares, note)
                else:
                    add_stock(ticker, stock_type, None, None, note)
                st.success(f"Saved {ticker}.")
                st.rerun()

    current = get_watchlist()
    if not current.empty:
        st.dataframe(current, use_container_width=True)
        remove_ticker = st.selectbox("Remove stock", current["ticker"].tolist())
        if st.button("Delete Selected Stock"):
            delete_stock(remove_ticker)
            st.success(f"Deleted {remove_ticker}.")
            st.rerun()


def main():
    init_db()
    seed_default_stocks()
    render_header()

    if st.button("Refresh Analysis"):
        st.cache_data.clear()
        st.rerun()

    results = build_analysis_results()
    if not results:
        st.error("No valid stock data loaded.")
        return

    save_top5_to_db(results)
    render_dashboard(results)
    render_legends()
    render_top5_section()
    render_detail_section(results)
    render_manage_section()


if __name__ == "__main__":
    main()
