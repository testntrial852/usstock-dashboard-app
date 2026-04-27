import streamlit as st
import pandas as pd
import yfinance as yf
import sqlite3
import math
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, time
from zoneinfo import ZoneInfo

st.set_page_config(page_title="US Stock Decision Dashboard 6B.7", layout="wide")

DB_PATH = "stocks.db"
ET_TZ = ZoneInfo("America/New_York")
TOP_PICKS_BUCKETS = [
    ("0400", time(4, 0)),
    ("0800", time(8, 0)),
    ("1030", time(10, 30)),
]
MAX_DAILY_PICKS = 5
RAW_SCORE_MAX = 15.0


# ----------------------------
# DB
# ----------------------------
def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


conn = get_conn()


def table_columns(table_name):
    rows = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
    return [row["name"] for row in rows]


def ensure_column(table_name, column_name, column_type):
    cols = table_columns(table_name)
    if column_name not in cols:
        conn.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type}")
        conn.commit()


def init_db():
    conn.execute("""
    CREATE TABLE IF NOT EXISTS stocks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        sort_order INTEGER,
        ticker TEXT UNIQUE,
        stock_type TEXT,
        buy_price REAL,
        shares REAL,
        price REAL,
        suggested_entry REAL,
        pt REAL,
        sl REAL,
        weekly_est_high REAL,
        weekly_est_low REAL,
        breakout_status TEXT,
        breakout_level REAL,
        support_level REAL,
        resistance_level REAL,
        volume_vs_avg20 REAL,
        rsi REAL,
        ma20 REAL,
        ma50 REAL,
        atr14 REAL,
        position_value REAL,
        unrealized_pl REAL,
        action TEXT,
        signal_confidence TEXT,
        color TEXT,
        reason TEXT
    )
    """)

    conn.execute("""
    CREATE TABLE IF NOT EXISTS daily_picks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        pick_date_et TEXT,
        bucket_code TEXT,
        bucket_label TEXT,
        ticker TEXT,
        universe_source TEXT,
        rank_num INTEGER,
        score REAL,
        action TEXT,
        signal_confidence TEXT,
        risk_level TEXT,
        current_price REAL,
        suggested_entry REAL,
        pt REAL,
        sl REAL,
        short_reason TEXT,
        full_reason TEXT,
        change_vs_prev_bucket TEXT,
        updated_at_et TEXT
    )
    """)

    conn.execute("""
    CREATE TABLE IF NOT EXISTS system_meta (
        meta_key TEXT PRIMARY KEY,
        meta_value TEXT
    )
    """)
    conn.commit()

    ensure_column("daily_picks", "score_max", "REAL")
    ensure_column("daily_picks", "score_100", "REAL")
    ensure_column("daily_picks", "score_band", "TEXT")


init_db()


# ----------------------------
# Helpers
# ----------------------------
def now_et():
    return datetime.now(ET_TZ)


def get_et_date_str(dt=None):
    dt = dt or now_et()
    return dt.strftime("%Y-%m-%d")


def format_et_dt(dt=None):
    dt = dt or now_et()
    return dt.strftime("%Y-%m-%d %I:%M %p ET")


def set_meta(key, value):
    conn.execute("""
    INSERT INTO system_meta (meta_key, meta_value)
    VALUES (?, ?)
    ON CONFLICT(meta_key) DO UPDATE SET meta_value=excluded.meta_value
    """, (key, value))
    conn.commit()


def get_meta(key, default=None):
    row = conn.execute("SELECT meta_value FROM system_meta WHERE meta_key=?", (key,)).fetchone()
    return row["meta_value"] if row else default


def format_num(val, decimals=2):
    if val is None or pd.isna(val):
        return "-"
    return f"{float(val):,.{decimals}f}"


def safe_round(value):
    if value is None or pd.isna(value):
        return None
    return round(float(value), 2)


def normalize_score_100(raw_score, max_score=RAW_SCORE_MAX):
    if raw_score is None:
        return None
    normalized = (raw_score / max_score) * 100
    normalized = max(0, min(normalized, 100))
    return round(normalized, 1)


def get_score_band(raw_score):
    if raw_score is None:
        return "N/A"
    if raw_score >= 12:
        return "Very High"
    elif raw_score >= 9:
        return "High"
    elif raw_score >= 6:
        return "Medium"
    return "Low"


def render_badge(text, color):
    return f'<span class="status-badge {color}">{text}</span>'


def render_score_band_badge(score_band):
    mapping = {
        "Very High": "green",
        "High": "blue",
        "Medium": "yellow",
        "Low": "gray",
        "N/A": "gray",
    }
    color = mapping.get(score_band, "gray")
    return f'<span class="status-badge {color}">{score_band}</span>'


def render_html_table(df, column_types=None, title=None):
    column_types = column_types or {}

    if df is None or df.empty:
        st.info("No data to display.")
        return

    headers = list(df.columns)
    header_html = "".join([f"<th>{h}</th>" for h in headers])

    rows_html = []
    for _, row in df.iterrows():
        cells = []
        for col in headers:
            value = row[col]
            cls = column_types.get(col, "")
            display = "-" if pd.isna(value) else value
            cells.append(f'<td class="{cls}">{display}</td>')
        rows_html.append(f"<tr>{''.join(cells)}</tr>")

    title_html = f'<div class="table-title">{title}</div>' if title else ""

    table_html = f"""
<div class="custom-table-wrap">
  {title_html}
  <div class="custom-table-scroll">
    <table class="custom-table">
      <thead>
        <tr>{header_html}</tr>
      </thead>
      <tbody>
        {''.join(rows_html)}
      </tbody>
    </table>
  </div>
</div>
""".strip()

    try:
        st.html(table_html)
    except Exception:
        st.markdown(table_html, unsafe_allow_html=True)


# ----------------------------
# CSS
# ----------------------------
CSS_BLOCK = """
<style>
.block-container { padding-top: 1.2rem; padding-bottom: 2rem; }

.status-badge {
    display: inline-block;
    padding: 6px 12px;
    border-radius: 999px;
    font-size: 12.5px;
    font-weight: 700;
    letter-spacing: 0.01em;
    text-align: center;
    line-height: 1.25;
    white-space: nowrap;
}
.green { background-color: #DCFCE7; color: #166534; }
.yellow { background-color: #FEF3C7; color: #92400E; }
.orange { background-color: #FFEDD5; color: #9A3412; }
.red { background-color: #FEE2E2; color: #991B1B; }
.blue { background-color: #DBEAFE; color: #1D4ED8; }
.gray { background-color: #E5E7EB; color: #374151; }

.custom-table-wrap {
    border: 1px solid #e5e7eb;
    border-radius: 16px;
    overflow: hidden;
    background: #ffffff;
    box-shadow: 0 6px 18px rgba(15, 23, 42, 0.04);
    margin-top: 0.35rem;
    margin-bottom: 1.1rem;
}
.custom-table-scroll { overflow-x: auto; }
.custom-table {
    width: 100%;
    border-collapse: separate;
    border-spacing: 0;
    min-width: 980px;
    font-size: 14.5px;
    line-height: 1.48;
}
.custom-table thead th {
    background: #f8fafc;
    color: #0f172a;
    font-weight: 700;
    text-align: left;
    padding: 14px 14px;
    border-bottom: 1px solid #e5e7eb;
    white-space: nowrap;
}
.custom-table tbody td {
    padding: 14px 14px;
    border-bottom: 1px solid #f1f5f9;
    vertical-align: top;
    color: #1e293b;
}
.custom-table tbody tr:nth-child(even) { background: #fcfcfd; }
.custom-table tbody tr:hover { background: #f8fbff; }
.custom-table tbody tr:last-child td { border-bottom: none; }
.num-col {
    text-align: right;
    white-space: nowrap;
    font-variant-numeric: tabular-nums;
}
.reason-col {
    min-width: 320px;
    color: #334155;
    line-height: 1.55;
}
.table-title {
    padding: 14px 16px 6px 16px;
    font-size: 15px;
    font-weight: 700;
    color: #0f172a;
}
.small-note {
    color: #64748b;
    font-size: 13px;
}
</style>
""".strip()

try:
    st.html(CSS_BLOCK)
except Exception:
    st.markdown(CSS_BLOCK, unsafe_allow_html=True)


# ----------------------------
# Indicators
# ----------------------------
def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def calculate_atr(df, period=14):
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return true_range.rolling(period).mean()


def get_hist_with_indicators(ticker, period="6mo"):
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)
        if hist.empty or len(hist) < 30:
            return None

        hist = hist.copy()
        hist["MA20"] = hist["Close"].rolling(20).mean()
        hist["MA50"] = hist["Close"].rolling(50).mean()
        hist["MA150"] = hist["Close"].rolling(150).mean()
        hist["MA200"] = hist["Close"].rolling(200).mean()
        hist["RSI"] = calculate_rsi(hist["Close"])
        hist["ATR14"] = calculate_atr(hist, 14)
        hist["AvgVol20"] = hist["Volume"].rolling(20).mean()
        hist["DollarVol20"] = hist["Close"] * hist["AvgVol20"]
        hist["52WHigh"] = hist["High"].rolling(252, min_periods=60).max()
        return hist
    except Exception:
        return None


# ----------------------------
# Stocks logic
# ----------------------------
def build_short_reason(row):
    if row["action"] == "Breakout Confirmed":
        return f"Breakout above {row['breakout_level']} with volume {row['volume_vs_avg20']}x."
    elif row["action"] == "Breakout Watch":
        return f"Near resistance {row['breakout_level']}; waiting for volume confirmation."
    elif row["action"] == "Near Entry":
        return f"Price near MA20 {row['ma20']} with manageable RSI {row['rsi']}."
    elif row["action"] == "Buy Setup":
        return f"Pullback setup: below MA20 {row['ma20']} and RSI {row['rsi']}."
    elif row["action"] == "Wait Pullback":
        return f"Momentum too hot: RSI {row['rsi']}. Wait for retracement."
    elif row["action"] == "Avoid":
        return f"RSI {row['rsi']} is elevated without strong confirmation."
    elif row["action"] == "Trim":
        return f"Profitable position with overheated momentum (RSI {row['rsi']})."
    elif row["action"] == "Hold":
        return f"Trend still acceptable; MA20 reference = {row['ma20']}."
    elif row["action"] == "Stop Loss":
        return "Price weakened materially versus cost basis."
    return "No clear setup yet."


def get_action_priority(action):
    priority_map = {
        "Stop Loss": 1,
        "Trim": 2,
        "Breakout Confirmed": 3,
        "Near Entry": 4,
        "Buy Setup": 5,
        "Breakout Watch": 6,
        "Wait Pullback": 7,
        "Avoid": 8,
        "Hold": 9,
        "Watch": 10,
        "No Data": 11,
        "Error": 12
    }
    return priority_map.get(action, 99)


def analyze_stock(ticker, stock_type, buy_price=None, shares=None):
    try:
        hist = get_hist_with_indicators(ticker, "6mo")
        if hist is None or hist.empty or len(hist) < 60:
            return {
                "price": None, "suggested_entry": None, "pt": None, "sl": None,
                "weekly_est_high": None, "weekly_est_low": None,
                "breakout_status": "No Data", "breakout_level": None,
                "support_level": None, "resistance_level": None,
                "volume_vs_avg20": None, "rsi": None, "ma20": None, "ma50": None, "atr14": None,
                "position_value": None, "unrealized_pl": None,
                "action": "No Data", "signal_confidence": "Low", "color": "gray",
                "reason": "Yahoo Finance 歷史數據不足，暫未能完成完整分析。"
            }

        latest = hist.iloc[-1]
        prev_20 = hist.iloc[-21:-1]

        price = safe_round(latest["Close"])
        ma20 = safe_round(latest["MA20"])
        ma50 = safe_round(latest["MA50"])
        rsi = safe_round(latest["RSI"])
        atr14 = safe_round(latest["ATR14"])
        avg_vol20 = latest["AvgVol20"]
        latest_vol = latest["Volume"]

        recent_high = safe_round(prev_20["High"].max()) if not prev_20.empty else None
        recent_low = safe_round(prev_20["Low"].min()) if not prev_20.empty else None

        volume_ratio = None
        if avg_vol20 and not pd.isna(avg_vol20) and avg_vol20 != 0:
            volume_ratio = round(float(latest_vol / avg_vol20), 2)

        weekly_est_high = None
        weekly_est_low = None
        if price is not None and atr14 is not None:
            weekly_move = atr14 * math.sqrt(5)
            weekly_est_high = round(price + weekly_move, 2)
            weekly_est_low = round(price - weekly_move, 2)

        suggested_entry = None
        pt = None
        sl = None
        position_value = None
        unrealized_pl = None
        breakout_status = "Neutral"
        action = "Watch"
        signal_confidence = "Low"
        color = "blue"
        reason_parts = []

        breakout_confirmed = (
            recent_high is not None and
            price is not None and
            price > recent_high and
            volume_ratio is not None and
            volume_ratio >= 1.2
        )

        breakout_watch = (
            recent_high is not None and
            price is not None and
            price >= recent_high * 0.99 and
            not breakout_confirmed
        )

        if stock_type == "Watchlist":
            if breakout_confirmed and atr14 is not None:
                breakout_status = "Confirmed"
                suggested_entry = round(recent_high * 1.005, 2)
                range_height = (recent_high - recent_low) if recent_low is not None else atr14 * 2
                pt = round(recent_high + range_height, 2)
                sl = round(suggested_entry - 1.5 * atr14, 2)
                action = "Breakout Confirmed"
                signal_confidence = "High"
                color = "green"
                reason_parts.append(f"現價 {price} 已突破近 20 日阻力 {recent_high}，而成交量為 20 日平均的 {volume_ratio} 倍，屬放量突破。")
                reason_parts.append(f"建議買入價可參考突破位上方約 0.5% 的 {suggested_entry}。")
                reason_parts.append(f"PT 約 {pt}；SL 約 {sl}。")
            elif breakout_watch and atr14 is not None:
                breakout_status = "Watch"
                suggested_entry = round(recent_high * 1.005, 2)
                pt = round(suggested_entry + 2 * atr14, 2)
                sl = round(suggested_entry - 1.5 * atr14, 2)
                action = "Breakout Watch"
                signal_confidence = "Medium"
                color = "yellow"
                reason_parts.append(f"現價 {price} 已接近近 20 日阻力 {recent_high}，但成交量確認未足夠。")
                reason_parts.append(f"如後續放量突破，可留意 {suggested_entry}；PT 約 {pt}；SL 約 {sl}。")
            elif ma20 is not None and price is not None and atr14 is not None and rsi is not None and price <= ma20 * 1.03 and rsi < 60:
                breakout_status = "Pullback Setup"
                suggested_entry = round(ma20, 2)
                pt = round(price + 2 * atr14, 2)
                sl = round(suggested_entry - 1.5 * atr14, 2)
                action = "Near Entry"
                signal_confidence = "High"
                color = "yellow"
                reason_parts.append(f"現價 {price} 貼近 20 日線 {ma20}。")
                reason_parts.append(f"RSI {rsi} 未見過熱，風險回報較追高合理。")
                reason_parts.append(f"建議留意 {suggested_entry}；PT 約 {pt}；SL 約 {sl}。")
            elif ma20 is not None and price is not None and atr14 is not None and rsi is not None and price < ma20 and rsi < 45:
                breakout_status = "Mean Reversion Setup"
                suggested_entry = round(price, 2)
                pt = round(ma20, 2)
                sl = round(price - 1.2 * atr14, 2)
                action = "Buy Setup"
                signal_confidence = "High"
                color = "green"
                reason_parts.append(f"現價 {price} 低於 20 日線 {ma20}，RSI 只有 {rsi}。")
                reason_parts.append(f"偏向回調買入 setup；PT 可先看 {pt}；SL 約 {sl}。")
            elif rsi is not None and rsi >= 75:
                breakout_status = "Extended"
                action = "Wait Pullback"
                signal_confidence = "High"
                color = "red"
                reason_parts.append(f"RSI 高達 {rsi}，短線過熱，不宜即追。")
                if ma20 is not None:
                    suggested_entry = round(ma20, 2)
                    reason_parts.append(f"可等回調接近 20 日線 {ma20} 再評估。")
            elif rsi is not None and rsi >= 65:
                breakout_status = "Extended"
                action = "Avoid"
                signal_confidence = "Medium"
                color = "red"
                reason_parts.append(f"RSI {rsi} 偏熱，但未見有量突破支持。")
                if ma20 is not None:
                    suggested_entry = round(ma20, 2)
                    reason_parts.append(f"如要等更佳入場位，可先觀察會否回到 {ma20}。")
            else:
                breakout_status = "Neutral"
                action = "Watch"
                signal_confidence = "Low"
                color = "blue"
                reason_parts.append(f"現價 {price} 暫未形成清晰 breakout 或 pullback setup。")
                if recent_high is not None and recent_low is not None:
                    reason_parts.append(f"短線留意阻力 {recent_high} / 支撐 {recent_low}。")

        else:
            breakout_status = "Holding Review"
            if buy_price is not None and buy_price > 0 and shares is not None and shares > 0 and price is not None:
                position_value = round(price * shares, 2)
                unrealized_pl = round((price - buy_price) * shares, 2)

                if atr14 is not None:
                    sl = round(buy_price - 1.5 * atr14, 2)
                    pt = round(buy_price + 2.5 * atr14, 2)

                suggested_entry = round(buy_price, 2)

                if price < buy_price * 0.93:
                    action = "Stop Loss"
                    signal_confidence = "High"
                    color = "red"
                    reason_parts.append(f"現價 {price} 較買入價 {buy_price} 回落超過約 7%。")
                    if sl is not None:
                        reason_parts.append(f"防守位可參考 {sl}。")
                elif rsi is not None and rsi >= 75 and unrealized_pl > 0:
                    action = "Trim"
                    signal_confidence = "High"
                    color = "orange"
                    reason_parts.append(f"持倉未實現盈利 {unrealized_pl}，而 RSI 高達 {rsi}。")
                    if pt is not None:
                        reason_parts.append(f"可考慮在 {pt} 附近分段止賺。")
                elif ma20 is not None and price >= ma20:
                    action = "Hold"
                    signal_confidence = "High"
                    color = "yellow"
                    reason_parts.append(f"現價 {price} 仍高於 20 日線 {ma20}，趨勢未見破壞。")
                    reason_parts.append(f"持倉市值約 {position_value}，未實現盈虧為 {unrealized_pl}。")
                else:
                    action = "Hold"
                    signal_confidence = "Medium"
                    color = "yellow"
                    reason_parts.append("持倉仍可觀察，但短線優勢減弱。")
            else:
                action = "Hold"
                signal_confidence = "Low"
                color = "gray"
                reason_parts.append("Holding 資料未齊，至少需要買入價及股數。")

        reason = " ".join(reason_parts) if reason_parts else "暫未形成明確分析結論。"

        return {
            "price": price, "suggested_entry": suggested_entry, "pt": pt, "sl": sl,
            "weekly_est_high": weekly_est_high, "weekly_est_low": weekly_est_low,
            "breakout_status": breakout_status, "breakout_level": recent_high,
            "support_level": recent_low, "resistance_level": recent_high,
            "volume_vs_avg20": volume_ratio, "rsi": rsi, "ma20": ma20, "ma50": ma50, "atr14": atr14,
            "position_value": position_value, "unrealized_pl": unrealized_pl,
            "action": action, "signal_confidence": signal_confidence, "color": color, "reason": reason
        }

    except Exception as e:
        return {
            "price": None, "suggested_entry": None, "pt": None, "sl": None,
            "weekly_est_high": None, "weekly_est_low": None,
            "breakout_status": "Error", "breakout_level": None, "support_level": None, "resistance_level": None,
            "volume_vs_avg20": None, "rsi": None, "ma20": None, "ma50": None, "atr14": None,
            "position_value": None, "unrealized_pl": None,
            "action": "Error", "signal_confidence": "Low", "color": "gray",
            "reason": f"抓取數據失敗：{e}"
        }


# ----------------------------
# Stocks CRUD
# ----------------------------
def load_stocks_from_db():
    rows = conn.execute("SELECT * FROM stocks ORDER BY sort_order ASC, id ASC").fetchall()
    return [dict(row) for row in rows]


def save_stock_to_db(row):
    conn.execute("""
    INSERT INTO stocks (
        sort_order, ticker, stock_type, buy_price, shares, price,
        suggested_entry, pt, sl, weekly_est_high, weekly_est_low,
        breakout_status, breakout_level, support_level, resistance_level,
        volume_vs_avg20, rsi, ma20, ma50, atr14,
        position_value, unrealized_pl, action, signal_confidence, color, reason
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        row["sort_order"], row["ticker"], row["stock_type"], row["buy_price"], row["shares"], row["price"],
        row["suggested_entry"], row["pt"], row["sl"], row["weekly_est_high"], row["weekly_est_low"],
        row["breakout_status"], row["breakout_level"], row["support_level"], row["resistance_level"],
        row["volume_vs_avg20"], row["rsi"], row["ma20"], row["ma50"], row["atr14"],
        row["position_value"], row["unrealized_pl"], row["action"], row["signal_confidence"], row["color"], row["reason"]
    ))
    conn.commit()


def update_stock_in_db(stock_id, row):
    conn.execute("""
    UPDATE stocks
    SET sort_order=?, ticker=?, stock_type=?, buy_price=?, shares=?, price=?,
        suggested_entry=?, pt=?, sl=?, weekly_est_high=?, weekly_est_low=?,
        breakout_status=?, breakout_level=?, support_level=?, resistance_level=?,
        volume_vs_avg20=?, rsi=?, ma20=?, ma50=?, atr14=?,
        position_value=?, unrealized_pl=?, action=?, signal_confidence=?, color=?, reason=?
    WHERE id=?
    """, (
        row["sort_order"], row["ticker"], row["stock_type"], row["buy_price"], row["shares"], row["price"],
        row["suggested_entry"], row["pt"], row["sl"], row["weekly_est_high"], row["weekly_est_low"],
        row["breakout_status"], row["breakout_level"], row["support_level"], row["resistance_level"],
        row["volume_vs_avg20"], row["rsi"], row["ma20"], row["ma50"], row["atr14"],
        row["position_value"], row["unrealized_pl"], row["action"], row["signal_confidence"], row["color"], row["reason"],
        stock_id
    ))
    conn.commit()


def delete_stock_from_db(stock_id):
    conn.execute("DELETE FROM stocks WHERE id=?", (stock_id,))
    conn.commit()


def update_sort_orders():
    for idx, row in enumerate(st.session_state.stock_data):
        row["sort_order"] = idx
        conn.execute("UPDATE stocks SET sort_order=? WHERE id=?", (idx, row["id"]))
    conn.commit()


def refresh_all_analysis():
    refreshed = []
    for row in st.session_state.stock_data:
        analysis = analyze_stock(row["ticker"], row["stock_type"], row.get("buy_price"), row.get("shares"))
        updated_row = row.copy()
        updated_row.update(analysis)
        refreshed.append(updated_row)
        update_stock_in_db(updated_row["id"], updated_row)
    st.session_state.stock_data = refreshed


def recalc_and_update_stock(row):
    analysis = analyze_stock(row["ticker"], row["stock_type"], row.get("buy_price"), row.get("shares"))
    row.update(analysis)
    update_stock_in_db(row["id"], row)


# ----------------------------
# Chart
# ----------------------------
def build_price_chart(ticker, selected_row, period="6mo"):
    hist = get_hist_with_indicators(ticker, period)
    if hist is None or hist.empty:
        return None

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.75, 0.25], vertical_spacing=0.04
    )

    fig.add_trace(go.Candlestick(
        x=hist.index, open=hist["Open"], high=hist["High"], low=hist["Low"], close=hist["Close"], name="Price"
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=hist.index, y=hist["MA20"], mode="lines", name="MA20", line=dict(width=1.8, color="#f59e0b")
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=hist.index, y=hist["MA50"], mode="lines", name="MA50", line=dict(width=1.8, color="#3b82f6")
    ), row=1, col=1)

    fig.add_trace(go.Bar(
        x=hist.index, y=hist["Volume"], name="Volume", marker_color="#94a3b8"
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=hist.index, y=hist["AvgVol20"], mode="lines", name="AvgVol20", line=dict(width=1.5, color="#ef4444")
    ), row=2, col=1)

    line_items = [
        ("Support", selected_row.get("support_level"), "#10b981"),
        ("Resistance", selected_row.get("resistance_level"), "#ef4444"),
        ("Breakout", selected_row.get("breakout_level"), "#8b5cf6"),
        ("Entry", selected_row.get("suggested_entry"), "#f59e0b"),
        ("PT", selected_row.get("pt"), "#14b8a6"),
        ("SL", selected_row.get("sl"), "#dc2626"),
    ]

    for label, value, color in line_items:
        if value is not None and not pd.isna(value):
            fig.add_hline(
                y=float(value),
                line_dash="dot",
                line_color=color,
                annotation_text=f"{label}: {value}",
                annotation_position="right",
                row=1, col=1
            )

    fig.update_layout(
        height=620,
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis_rangeslider_visible=False,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        title=f"{ticker} Price & Volume Chart"
    )
    return fig


# ----------------------------
# Top 5 scan
# ----------------------------
NASDAQ_SAMPLE = [
    "AAPL", "MSFT", "NVDA", "AMD", "AVGO", "META", "AMZN", "NFLX", "TSLA", "PLTR",
    "CRWD", "PANW", "SNOW", "DDOG", "MDB", "SHOP", "ZS", "TEAM", "INTC", "MU"
]

RUSSELL_SAMPLE = [
    "APP", "IOT", "AAOI", "ALGM", "DOCN", "RKLB", "RXRX", "RUN", "ONON", "BILL",
    "EXAS", "FUBO", "IONQ", "UPWK", "SOUN", "CELH", "TTD", "FN", "CFLT", "SMCI"
]


def get_screening_universe():
    universe = []
    for t in NASDAQ_SAMPLE:
        universe.append((t, "NASDAQ"))
    for t in RUSSELL_SAMPLE:
        universe.append((t, "Russell 2000"))
    return universe


def score_candidate(ticker, universe_source):
    hist = get_hist_with_indicators(ticker, "1y")
    if hist is None or hist.empty:
        return None

    latest = hist.iloc[-1]
    prev_20 = hist.iloc[-21:-1]

    price = float(latest["Close"])
    ma20 = latest["MA20"]
    ma50 = latest["MA50"]
    ma150 = latest["MA150"]
    ma200 = latest["MA200"]
    rsi = latest["RSI"]
    atr14 = latest["ATR14"]
    avg_vol20 = latest["AvgVol20"]
    dollar_vol20 = latest["DollarVol20"]
    high_52w = latest["52WHigh"]

    if pd.isna(price) or pd.isna(avg_vol20) or pd.isna(dollar_vol20):
        return None
    if any(pd.isna(x) for x in [ma20, ma50, ma150, ma200, rsi, atr14]):
        return None
    if price < 10 or avg_vol20 < 1_000_000 or dollar_vol20 < 20_000_000:
        return None

    recent_high = float(prev_20["High"].max()) if not prev_20.empty else price
    recent_low = float(prev_20["Low"].min()) if not prev_20.empty else price
    volume_ratio = float(latest["Volume"] / avg_vol20) if avg_vol20 else 0.0

    quality = 0
    setup = 0
    penalty = 0

    if price > ma50:
        quality += 1
    if price > ma150:
        quality += 1
    if price > ma200:
        quality += 1
    if ma50 > ma150 > ma200:
        quality += 2
    if high_52w and price >= high_52w * 0.85:
        quality += 2

    if price >= recent_high * 0.99:
        setup += 2
    if price > recent_high and volume_ratio >= 1.2:
        setup += 3
    if volume_ratio >= 1.2:
        setup += 2
    if 55 <= rsi <= 70:
        setup += 1
    elif 70 < rsi <= 75:
        setup += 0.5

    if rsi > 75:
        penalty += 1.5
    if price > ma20 * 1.10:
        penalty += 1
    if universe_source == "Russell 2000":
        penalty += 0.5

    score = round(quality + setup - penalty, 2)

    breakout_confirmed = price > recent_high and volume_ratio >= 1.2
    breakout_watch = price >= recent_high * 0.99 and not breakout_confirmed

    if breakout_confirmed:
        action = "Breakout Confirmed"
        confidence = "High"
        suggested_entry = round(recent_high * 1.005, 2)
        pt = round(recent_high + (recent_high - recent_low), 2)
        sl = round(suggested_entry - 1.5 * atr14, 2)
    elif breakout_watch:
        action = "Breakout Watch"
        confidence = "Medium"
        suggested_entry = round(recent_high * 1.005, 2)
        pt = round(suggested_entry + 2 * atr14, 2)
        sl = round(suggested_entry - 1.5 * atr14, 2)
    elif price <= ma20 * 1.03 and rsi < 60:
        action = "Near Entry"
        confidence = "High"
        suggested_entry = round(ma20, 2)
        pt = round(price + 2 * atr14, 2)
        sl = round(suggested_entry - 1.5 * atr14, 2)
    else:
        action = "Watch"
        confidence = "Low"
        suggested_entry = round(ma20, 2)
        pt = round(price + 1.5 * atr14, 2)
        sl = round(price - 1.2 * atr14, 2)

    risk_level = "High" if universe_source == "Russell 2000" else "Medium"

    short_reason = (
        f"{universe_source}; price={round(price,2)}, RVOL={round(volume_ratio,2)}x, "
        f"RSI={round(rsi,2)}, breakout ref={round(recent_high,2)}."
    )

    full_reason = (
        f"{ticker} 來自 {universe_source} universe。現價 {round(price,2)}，相對20日均量為 "
        f"{round(volume_ratio,2)} 倍，RSI 為 {round(rsi,2)}。價格相對 MA20/50/150/200 "
        f"結構良好，近期20日阻力位約為 {round(recent_high,2)}。建議入場價 "
        f"{suggested_entry}，PT {pt}，SL {sl}。"
    )

    return {
        "ticker": ticker,
        "universe_source": universe_source,
        "score": score,
        "score_max": RAW_SCORE_MAX,
        "score_100": normalize_score_100(score),
        "score_band": get_score_band(score),
        "action": action,
        "signal_confidence": confidence,
        "risk_level": risk_level,
        "current_price": round(price, 2),
        "suggested_entry": suggested_entry,
        "pt": pt,
        "sl": sl,
        "short_reason": short_reason,
        "full_reason": full_reason,
    }


def get_bucket_label(bucket_code):
    return {
        "0400": "04:00 ET",
        "0800": "08:00 ET",
        "1030": "10:30 ET",
    }.get(bucket_code, bucket_code)


def get_due_buckets(now_dt=None):
    now_dt = now_dt or now_et()
    due = []
    for bucket_code, bucket_time in TOP_PICKS_BUCKETS:
        bucket_dt = datetime.combine(now_dt.date(), bucket_time, tzinfo=ET_TZ)
        if now_dt >= bucket_dt:
            due.append((bucket_code, bucket_time))
    return due


def bucket_meta_key(bucket_code, date_str):
    return f"daily_pick_last_refresh::{date_str}::{bucket_code}"


def bucket_already_refreshed(bucket_code, date_str):
    return get_meta(bucket_meta_key(bucket_code, date_str)) is not None


def get_previous_bucket(bucket_code):
    order = ["0400", "0800", "1030"]
    idx = order.index(bucket_code)
    return None if idx == 0 else order[idx - 1]


def load_daily_picks(date_str, bucket_code):
    rows = conn.execute("""
        SELECT *
        FROM daily_picks
        WHERE pick_date_et=? AND bucket_code=?
        ORDER BY rank_num ASC, score DESC
    """, (date_str, bucket_code)).fetchall()
    return [dict(r) for r in rows]


def compare_vs_previous_bucket(date_str, bucket_code, picks):
    prev_bucket = get_previous_bucket(bucket_code)
    if not prev_bucket:
        for p in picks:
            p["change_vs_prev_bucket"] = "First bucket of day"
        return picks

    prev_rows = load_daily_picks(date_str, prev_bucket)
    prev_map = {row["ticker"]: row for row in prev_rows}

    for p in picks:
        if p["ticker"] not in prev_map:
            p["change_vs_prev_bucket"] = "New"
        else:
            prev_score = prev_map[p["ticker"]].get("score", 0)
            curr_score = p.get("score", 0)
            if curr_score > prev_score:
                p["change_vs_prev_bucket"] = "Strengthened"
            elif curr_score < prev_score:
                p["change_vs_prev_bucket"] = "Weakened"
            else:
                p["change_vs_prev_bucket"] = "Stayed"
    return picks


def save_daily_picks(date_str, bucket_code, picks):
    conn.execute("DELETE FROM daily_picks WHERE pick_date_et=? AND bucket_code=?", (date_str, bucket_code))
    updated_at = format_et_dt()

    for idx, p in enumerate(picks, start=1):
        conn.execute("""
            INSERT INTO daily_picks (
                pick_date_et, bucket_code, bucket_label, ticker, universe_source, rank_num,
                score, score_max, score_100, score_band,
                action, signal_confidence, risk_level, current_price, suggested_entry,
                pt, sl, short_reason, full_reason, change_vs_prev_bucket, updated_at_et
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            date_str, bucket_code, get_bucket_label(bucket_code), p["ticker"], p["universe_source"], idx,
            p.get("score"), p.get("score_max", RAW_SCORE_MAX), p.get("score_100"), p.get("score_band", "N/A"),
            p.get("action"), p.get("signal_confidence"), p.get("risk_level"), p.get("current_price"),
            p.get("suggested_entry"), p.get("pt"), p.get("sl"), p.get("short_reason"),
            p.get("full_reason"), p.get("change_vs_prev_bucket"), updated_at
        ))

    conn.commit()
    set_meta(bucket_meta_key(bucket_code, date_str), updated_at)


def run_bucket_scan(bucket_code):
    universe = get_screening_universe()
    candidates = []

    for ticker, source in universe:
        scored = score_candidate(ticker, source)
        if scored:
            candidates.append(scored)

    candidates = sorted(
        candidates,
        key=lambda x: (
            x.get("score", 0),
            1 if x.get("signal_confidence") == "High" else 0,
            1 if x.get("action") == "Breakout Confirmed" else 0
        ),
        reverse=True
    )[:MAX_DAILY_PICKS]

    date_str = get_et_date_str()
    candidates = compare_vs_previous_bucket(date_str, bucket_code, candidates)
    save_daily_picks(date_str, bucket_code, candidates)


def refresh_due_scan_buckets():
    today_et = get_et_date_str()
    due = get_due_buckets()
    refreshed = []
    for bucket_code, _ in due:
        if not bucket_already_refreshed(bucket_code, today_et):
            run_bucket_scan(bucket_code)
            refreshed.append(bucket_code)
    return refreshed


# ----------------------------
# Session
# ----------------------------
if "stock_data" not in st.session_state:
    st.session_state.stock_data = load_stocks_from_db()
else:
    if len(st.session_state.stock_data) == 0:
        st.session_state.stock_data = load_stocks_from_db()


# ----------------------------
# Header
# ----------------------------
refreshed_buckets = refresh_due_scan_buckets()

st.title("US Stock Decision Dashboard")
st.caption("Version 6B.7 — Main Dashboard → Top 5 → Stock Detail")

if refreshed_buckets:
    st.success(f"Daily picks refreshed for bucket(s): {', '.join(refreshed_buckets)}")

st.markdown(
    '<div class="small-note">Market data fields are data-driven from Yahoo Finance history. '
    'Score, recommendation, PT/SL, risk level, and reasons are rule-based outputs built on top of those data.</div>',
    unsafe_allow_html=True
)


# ----------------------------
# Sidebar
# ----------------------------
with st.sidebar:
    st.header("Add Stock")

    with st.form("add_stock_form", clear_on_submit=False):
        ticker = st.text_input("Ticker *").upper().strip()
        stock_type = st.selectbox("Type *", ["Holding", "Watchlist"])

        buy_price = None
        shares = None

        if stock_type == "Holding":
            buy_price = st.number_input("Buy Price *", min_value=0.0, value=0.0, step=0.1)
            shares = st.number_input("Shares *", min_value=1.0, value=1.0, step=1.0)

        submitted = st.form_submit_button("Add Stock")

        if submitted:
            errors = []
            if not ticker:
                errors.append("Ticker is required.")
            existing_tickers = [row["ticker"] for row in st.session_state.stock_data]
            if ticker in existing_tickers:
                errors.append(f"{ticker} already exists.")
            if stock_type == "Holding" and (buy_price is None or buy_price <= 0):
                errors.append("Buy Price is required for Holding.")
            if stock_type == "Holding" and (shares is None or shares <= 0):
                errors.append("Shares is required for Holding.")

            if errors:
                for err in errors:
                    st.error(err)
            else:
                analysis = analyze_stock(ticker, stock_type, buy_price, shares)
                new_row = {
                    "sort_order": len(st.session_state.stock_data),
                    "ticker": ticker,
                    "stock_type": stock_type,
                    "buy_price": buy_price if stock_type == "Holding" else None,
                    "shares": shares if stock_type == "Holding" else None,
                    **analysis
                }
                save_stock_to_db(new_row)
                st.session_state.stock_data = load_stocks_from_db()
                st.success(f"{ticker} added successfully!")
                st.rerun()

    if st.button("Refresh All Analysis", use_container_width=True):
        refresh_all_analysis()
        st.success("All stocks refreshed.")
        st.rerun()

df = pd.DataFrame(st.session_state.stock_data)

with st.sidebar:
    st.markdown("---")
    st.header("Filters")

    if not df.empty:
        all_types = list(df["stock_type"].dropna().unique())
        all_actions = list(df["action"].dropna().unique())
        all_confidence = list(df["signal_confidence"].dropna().unique())

        type_filter = st.multiselect("Type", options=all_types, default=all_types)
        action_filter = st.multiselect("Action", options=all_actions, default=all_actions)
        confidence_filter = st.multiselect("Signal Confidence", options=all_confidence, default=all_confidence)
    else:
        type_filter = []
        action_filter = []
        confidence_filter = []

    search = st.text_input("Search ticker")

    st.markdown("---")
    st.header("Dashboard View")
    view_mode = st.radio("View Mode", ["Compact", "Standard", "Detailed"], index=0)


# ----------------------------
# Filter data
# ----------------------------
if not df.empty:
    filtered_df = df.copy()
    if type_filter:
        filtered_df = filtered_df[filtered_df["stock_type"].isin(type_filter)]
    if action_filter:
        filtered_df = filtered_df[filtered_df["action"].isin(action_filter)]
    if confidence_filter:
        filtered_df = filtered_df[filtered_df["signal_confidence"].isin(confidence_filter)]
    if search:
        filtered_df = filtered_df[filtered_df["ticker"].str.contains(search.upper(), na=False)]

    filtered_df["action_priority"] = filtered_df["action"].apply(get_action_priority)
    filtered_df = filtered_df.sort_values(by=["action_priority", "sort_order", "ticker"], ascending=[True, True, True])
else:
    filtered_df = pd.DataFrame()


# ----------------------------
# 1) MAIN DASHBOARD
# ----------------------------
st.markdown("---")
st.subheader("Main Dashboard")

if not df.empty:
    holdings_df = df[df["stock_type"] == "Holding"].copy()
    total_holdings = len(holdings_df)
    total_watchlist = len(df[df["stock_type"] == "Watchlist"])
    total_market_value = round(holdings_df["position_value"].fillna(0).sum(), 2) if not holdings_df.empty else 0
    total_unrealized_pl = round(holdings_df["unrealized_pl"].fillna(0).sum(), 2) if not holdings_df.empty else 0
    high_conf_count = len(df[df["signal_confidence"] == "High"])
    action_needed_count = len(df[df["action"].isin([
        "Breakout Confirmed", "Breakout Watch", "Near Entry", "Buy Setup", "Trim", "Stop Loss", "Wait Pullback"
    ])])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Holdings", total_holdings)
    c2.metric("Watchlist", total_watchlist)
    c3.metric("Market Value", total_market_value)
    c4.metric("Unrealized P/L", total_unrealized_pl)

    c5, c6 = st.columns(2)
    c5.metric("High Confidence", high_conf_count)
    c6.metric("Action Needed", action_needed_count)
else:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Holdings", 0)
    c2.metric("Watchlist", 0)
    c3.metric("Market Value", 0)
    c4.metric("Unrealized P/L", 0)

if filtered_df.empty:
    st.info("No stocks to display.")
else:
    display_df = filtered_df.copy()
    display_df["Action"] = [render_badge(a, c) for a, c in zip(display_df["action"], display_df["color"])]
    display_df["Type"] = display_df["stock_type"].replace({"Watchlist": "Watch"})
    display_df["Ticker"] = display_df["ticker"]
    display_df["Price"] = display_df["price"].apply(format_num)
    display_df["Entry"] = display_df["suggested_entry"].apply(format_num)
    display_df["PT"] = display_df["pt"].apply(format_num)
    display_df["SL"] = display_df["sl"].apply(format_num)
    display_df["Confidence"] = display_df["signal_confidence"]
    display_df["Short Reason"] = display_df.apply(build_short_reason, axis=1)

    if view_mode == "Compact":
        final_df = display_df[["Ticker", "Type", "Price", "Action", "Confidence", "Entry", "PT", "SL", "Short Reason"]].copy()
    elif view_mode == "Standard":
        display_df["Breakout"] = display_df["breakout_status"]
        display_df["RSI"] = display_df["rsi"].apply(lambda x: format_num(x, 1))
        display_df["MA20"] = display_df["ma20"].apply(format_num)
        display_df["RVOL"] = display_df["volume_vs_avg20"].apply(lambda x: format_num(x, 2))
        final_df = display_df[["Ticker", "Type", "Price", "Action", "Confidence", "Entry", "PT", "SL", "Breakout", "RSI", "MA20", "RVOL", "Short Reason"]].copy()
    else:
        display_df["Breakout"] = display_df["breakout_status"]
        display_df["RSI"] = display_df["rsi"].apply(lambda x: format_num(x, 1))
        display_df["MA20"] = display_df["ma20"].apply(format_num)
        display_df["MA50"] = display_df["ma50"].apply(format_num)
        display_df["RVOL"] = display_df["volume_vs_avg20"].apply(lambda x: format_num(x, 2))
        display_df["Buy Price"] = display_df["buy_price"].apply(format_num)
        display_df["Shares"] = display_df["shares"].apply(lambda x: format_num(x, 0))
        display_df["Position Value"] = display_df["position_value"].apply(format_num)
        display_df["Unrealized P/L"] = display_df["unrealized_pl"].apply(format_num)
        final_df = display_df[[
            "Ticker", "Type", "Price", "Action", "Confidence", "Entry", "PT", "SL",
            "Breakout", "RSI", "MA20", "MA50", "RVOL",
            "Buy Price", "Shares", "Position Value", "Unrealized P/L", "Short Reason"
        ]].copy()

    render_html_table(
        final_df,
        column_types={
            "Price": "num-col",
            "Entry": "num-col",
            "PT": "num-col",
            "SL": "num-col",
            "RSI": "num-col",
            "MA20": "num-col",
            "MA50": "num-col",
            "RVOL": "num-col",
            "Buy Price": "num-col",
            "Shares": "num-col",
            "Position Value": "num-col",
            "Unrealized P/L": "num-col",
            "Short Reason": "reason-col"
        },
        title="Portfolio Dashboard"
    )


# ----------------------------
# 2) TOP 5
# ----------------------------
st.markdown("---")
st.subheader("Daily Top 5 High Potential Picks")
st.caption("Universe: NASDAQ + Russell 2000 | Buckets: 04:00 / 08:00 / 10:30 ET")

with st.expander("How to read Score", expanded=False):
    st.markdown("""
- **Raw Score**：內部 ranking 分數，按 trend quality + setup quality - penalty 計。
- **Max**：目前理論上限 15。
- **Score / 100**：將 raw score 正規化到 100。
- **Score Band**：Very High / High / Medium / Low。
""")

if st.button("Refresh Scan Buckets", key="refresh_scan_buckets_btn"):
    refreshed = refresh_due_scan_buckets()
    if refreshed:
        st.success(f"Refreshed buckets: {', '.join(refreshed)}")
    else:
        st.info("No new bucket due yet. Existing data remains current.")
    st.rerun()

today_et = get_et_date_str()
tab_0400, tab_0800, tab_1030 = st.tabs(["04:00 ET", "08:00 ET", "10:30 ET"])

for tab, bucket_code in zip([tab_0400, tab_0800, tab_1030], ["0400", "0800", "1030"]):
    with tab:
        rows = load_daily_picks(today_et, bucket_code)

        if not rows:
            st.warning(f"No data yet for {get_bucket_label(bucket_code)} on {today_et}.")
            continue

        updated_at = rows[0].get("updated_at_et", "Not available")
        st.markdown(f"**Updated as of:** {updated_at}")

        safe_rows = []
        for r in rows:
            raw_score = r.get("score")
            score_max = r.get("score_max", RAW_SCORE_MAX)
            score_100 = r.get("score_100")
            if score_100 is None and raw_score is not None:
                score_100 = normalize_score_100(raw_score, score_max)
            score_band = r.get("score_band")
            if not score_band:
                score_band = get_score_band(raw_score)

            action_text = r.get("action", "N/A")
            action_color = "green" if action_text == "Breakout Confirmed" else "yellow" if action_text in ["Breakout Watch", "Near Entry"] else "blue"

            safe_rows.append({
                "Rank": r.get("rank_num", "-"),
                "Ticker": r.get("ticker", "-"),
                "Universe": r.get("universe_source", "-"),
                "Action": render_badge(action_text, action_color),
                "Confidence": r.get("signal_confidence", "-"),
                "Risk": r.get("risk_level", "-"),
                "Price": format_num(r.get("current_price")),
                "Entry": format_num(r.get("suggested_entry")),
                "PT": format_num(r.get("pt")),
                "SL": format_num(r.get("sl")),
                "Raw Score": format_num(raw_score, 1),
                "Max": format_num(score_max, 0),
                "Score / 100": format_num(score_100, 1),
                "Score Band": render_score_band_badge(score_band),
                "Change": r.get("change_vs_prev_bucket", "-"),
                "Reason": r.get("short_reason", "-"),
            })

        display_df = pd.DataFrame(safe_rows)

        render_html_table(
            display_df,
            column_types={
                "Rank": "num-col",
                "Price": "num-col",
                "Entry": "num-col",
                "PT": "num-col",
                "SL": "num-col",
                "Raw Score": "num-col",
                "Max": "num-col",
                "Score / 100": "num-col",
                "Reason": "reason-col",
            },
            title=f"{get_bucket_label(bucket_code)} Picks"
        )


# ----------------------------
# 3) STOCK DETAIL
# ----------------------------
st.markdown("---")
st.subheader("Stock Detail View")

if filtered_df.empty:
    st.info("Add your first stock from the sidebar.")
else:
    selected_ticker = st.selectbox(
        "Choose a stock to view details",
        filtered_df["ticker"].tolist(),
        key="detail_select"
    )
    selected_row = filtered_df[filtered_df["ticker"] == selected_ticker].iloc[0]

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Overview",
        "Trade Plan",
        "Technical",
        "Reason",
        "Position"
    ])

    with tab1:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Current Price", selected_row["price"])
        c2.metric("Action", selected_row["action"])
        c3.metric("Signal Confidence", selected_row["signal_confidence"])
        c4.metric("Breakout Status", selected_row["breakout_status"])
        st.markdown("**Short Reason**")
        st.write(build_short_reason(selected_row))

    with tab2:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Suggested Entry", selected_row["suggested_entry"])
        c2.metric("PT", selected_row["pt"])
        c3.metric("SL", selected_row["sl"])
        c4.metric("Volume vs Avg20", selected_row["volume_vs_avg20"])

        c5, c6 = st.columns(2)
        c5.metric("Weekly Est High", selected_row["weekly_est_high"])
        c6.metric("Weekly Est Low", selected_row["weekly_est_low"])

    with tab3:
        chart_period = st.radio(
            "Chart Range",
            ["3mo", "6mo"],
            index=1,
            horizontal=True,
            key="chart_period"
        )

        fig = build_price_chart(selected_ticker, selected_row, chart_period)
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Not enough historical data to render chart.")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("RSI", selected_row["rsi"])
        c2.metric("MA20", selected_row["ma20"])
        c3.metric("MA50", selected_row["ma50"])
        c4.metric("ATR14", selected_row["atr14"])

        c5, c6, c7, c8 = st.columns(4)
        c5.metric("Support", selected_row["support_level"])
        c6.metric("Resistance", selected_row["resistance_level"])
        c7.metric("Breakout Level", selected_row["breakout_level"])
        c8.metric("Price", selected_row["price"])

    with tab4:
        st.markdown("**Full Reason**")
        st.write(selected_row["reason"])

    with tab5:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Type", selected_row["stock_type"])
        c2.metric("Buy Price", selected_row["buy_price"])
        c3.metric("Shares", selected_row["shares"])
        c4.metric("Position Value", selected_row["position_value"])

        c5, c6 = st.columns(2)
        c5.metric("Unrealized P/L", selected_row["unrealized_pl"])
        c6.metric("Sort Order", selected_row["sort_order"])


# ----------------------------
# Edit / manage
# ----------------------------
st.markdown("---")
st.subheader("Edit Stock")

if not df.empty:
    selected_edit_ticker = st.selectbox(
        "Choose stock to edit",
        df["ticker"].tolist(),
        key="edit_stock_select"
    )
    selected_edit_row = next(
        (row for row in st.session_state.stock_data if row["ticker"] == selected_edit_ticker),
        None
    )

    if selected_edit_row:
        with st.form("edit_stock_form"):
            new_ticker = st.text_input("Ticker *", value=selected_edit_row["ticker"]).upper().strip()
            new_type = st.selectbox(
                "Type *",
                ["Holding", "Watchlist"],
                index=0 if selected_edit_row["stock_type"] == "Holding" else 1
            )

            default_buy_price = float(selected_edit_row["buy_price"]) if selected_edit_row["buy_price"] is not None else 0.0
            default_shares = float(selected_edit_row["shares"]) if selected_edit_row["shares"] is not None else 1.0

            if new_type == "Holding":
                new_buy_price = st.number_input("Buy Price *", min_value=0.0, value=default_buy_price, step=0.1)
                new_shares = st.number_input("Shares *", min_value=1.0, value=default_shares, step=1.0)
            else:
                new_buy_price = None
                new_shares = None
                st.info("Watchlist stock does not require buy price or shares.")

            save_edit = st.form_submit_button("Save Changes")

            if save_edit:
                errors = []
                if not new_ticker:
                    errors.append("Ticker is required.")

                duplicate_tickers = [
                    row["ticker"] for row in st.session_state.stock_data
                    if row["id"] != selected_edit_row["id"]
                ]
                if new_ticker in duplicate_tickers:
                    errors.append(f"{new_ticker} already exists.")

                if new_type == "Holding" and (new_buy_price is None or new_buy_price <= 0):
                    errors.append("Buy Price is required for Holding.")

                if new_type == "Holding" and (new_shares is None or new_shares <= 0):
                    errors.append("Shares is required for Holding.")

                if errors:
                    for err in errors:
                        st.error(err)
                else:
                    selected_edit_row["ticker"] = new_ticker
                    selected_edit_row["stock_type"] = new_type
                    selected_edit_row["buy_price"] = new_buy_price
                    selected_edit_row["shares"] = new_shares

                    recalc_and_update_stock(selected_edit_row)
                    st.session_state.stock_data = load_stocks_from_db()
                    st.success(f"{new_ticker} updated successfully!")
                    st.rerun()
else:
    st.info("No stock available for editing.")

st.markdown("---")
st.subheader("Manage Stocks")

if not filtered_df.empty:
    filtered_ids = filtered_df["id"].tolist()

    for row in st.session_state.stock_data:
        if row["id"] not in filtered_ids:
            continue

        c1, c2, c3, c4, c5, c6 = st.columns([2, 2, 2, 1, 1, 1])
        c1.write(f"**{row['ticker']}**")
        c2.write(row["stock_type"])
        c3.write(row["action"])

        if c4.button("↑", key=f"up_{row['id']}"):
            pos = st.session_state.stock_data.index(row)
            if pos > 0:
                st.session_state.stock_data[pos - 1], st.session_state.stock_data[pos] = (
                    st.session_state.stock_data[pos],
                    st.session_state.stock_data[pos - 1],
                )
                update_sort_orders()
            st.rerun()

        if c5.button("↓", key=f"down_{row['id']}"):
            pos = st.session_state.stock_data.index(row)
            if pos < len(st.session_state.stock_data) - 1:
                st.session_state.stock_data[pos + 1], st.session_state.stock_data[pos] = (
                    st.session_state.stock_data[pos],
                    st.session_state.stock_data[pos + 1],
                )
                update_sort_orders()
            st.rerun()

        if c6.button("Remove", key=f"remove_{row['id']}"):
            delete_stock_from_db(row["id"])
            st.session_state.stock_data = load_stocks_from_db()
            st.rerun()
else:
    st.info("No stocks added yet.")
