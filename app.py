import streamlit as st
import pandas as pd
import yfinance as yf
import sqlite3
import math
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="US Stock Decision Dashboard 6B.3", layout="wide")

DB_PATH = "stocks.db"


def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


conn = get_conn()


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
    conn.commit()


init_db()


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
    SET sort_order=?,
        ticker=?,
        stock_type=?,
        buy_price=?,
        shares=?,
        price=?,
        suggested_entry=?,
        pt=?,
        sl=?,
        weekly_est_high=?,
        weekly_est_low=?,
        breakout_status=?,
        breakout_level=?,
        support_level=?,
        resistance_level=?,
        volume_vs_avg20=?,
        rsi=?,
        ma20=?,
        ma50=?,
        atr14=?,
        position_value=?,
        unrealized_pl=?,
        action=?,
        signal_confidence=?,
        color=?,
        reason=?
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


def safe_round(value):
    if value is None or pd.isna(value):
        return None
    return round(float(value), 2)


def render_badge(text, color):
    return f'<span class="status-badge {color}">{text}</span>'


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


def get_action_group(action):
    if action in ["Stop Loss", "Trim", "Breakout Confirmed", "Near Entry", "Buy Setup", "Breakout Watch", "Wait Pullback"]:
        return "Action Needed"
    elif action in ["Avoid", "Hold", "Watch"]:
        return "Monitor"
    return "Low Priority"


def get_hist_with_indicators(ticker, period="6mo"):
    stock = yf.Ticker(ticker)
    hist = stock.history(period=period)

    if hist.empty or len(hist) < 30:
        return None

    hist = hist.copy()
    hist["MA20"] = hist["Close"].rolling(20).mean()
    hist["MA50"] = hist["Close"].rolling(50).mean()
    hist["RSI"] = calculate_rsi(hist["Close"])
    hist["ATR14"] = calculate_atr(hist, 14)
    hist["AvgVol20"] = hist["Volume"].rolling(20).mean()
    return hist


def analyze_stock(ticker, stock_type, buy_price=None, shares=None):
    try:
        hist = get_hist_with_indicators(ticker, "6mo")

        if hist is None or hist.empty or len(hist) < 60:
            return {
                "price": None,
                "suggested_entry": None,
                "pt": None,
                "sl": None,
                "weekly_est_high": None,
                "weekly_est_low": None,
                "breakout_status": "No Data",
                "breakout_level": None,
                "support_level": None,
                "resistance_level": None,
                "volume_vs_avg20": None,
                "rsi": None,
                "ma20": None,
                "ma50": None,
                "atr14": None,
                "position_value": None,
                "unrealized_pl": None,
                "action": "No Data",
                "signal_confidence": "Low",
                "color": "gray",
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
                reason_parts.append(f"建議買入價可參考突破位上方約 0.5% 的 {suggested_entry}，避免太早追入假突破。")
                reason_parts.append(f"以近期區間高度推算，初步 PT 約 {pt}；SL 設於 {sl}，約為 1.5 倍 ATR 防守。")

            elif breakout_watch and atr14 is not None:
                breakout_status = "Watch"
                suggested_entry = round(recent_high * 1.005, 2)
                pt = round(suggested_entry + 2 * atr14, 2)
                sl = round(suggested_entry - 1.5 * atr14, 2)
                action = "Breakout Watch"
                signal_confidence = "Medium"
                color = "yellow"
                reason_parts.append(f"現價 {price} 已接近近 20 日阻力 {recent_high}，但成交量確認仍未足夠，現階段屬 breakout watch。")
                reason_parts.append(f"如後續放量突破，可考慮以 {suggested_entry} 作追入參考；PT 約 {pt}；SL 約 {sl}。")

            elif ma20 is not None and price is not None and atr14 is not None and rsi is not None and price <= ma20 * 1.03 and rsi < 60:
                breakout_status = "Pullback Setup"
                suggested_entry = round(ma20, 2)
                pt = round(price + 2 * atr14, 2)
                sl = round(suggested_entry - 1.5 * atr14, 2)
                action = "Near Entry"
                signal_confidence = "High"
                color = "yellow"
                reason_parts.append(f"現價 {price} 貼近 20 日線 {ma20}，屬趨勢中的回踩位置。")
                reason_parts.append(f"RSI 為 {rsi}，未見明顯過熱，風險回報較追高更合理。")
                reason_parts.append(f"建議買入價可先看 {suggested_entry}；PT 約 {pt}；SL 約 {sl}。")

            elif ma20 is not None and price is not None and atr14 is not None and rsi is not None and price < ma20 and rsi < 45:
                breakout_status = "Mean Reversion Setup"
                suggested_entry = round(price, 2)
                pt = round(ma20, 2)
                sl = round(price - 1.2 * atr14, 2)
                action = "Buy Setup"
                signal_confidence = "High"
                color = "green"
                reason_parts.append(f"現價 {price} 低於 20 日線 {ma20}，而 RSI 只有 {rsi}，顯示短線已回調到較弱區域。")
                reason_parts.append(f"呢種 setup 偏向回調買入，第一目標可先看回 MA20 附近，即 {pt}。")
                reason_parts.append(f"建議買入價可參考現價附近 {suggested_entry}；SL 約 {sl}。")

            elif rsi is not None and rsi >= 75:
                breakout_status = "Extended"
                action = "Wait Pullback"
                signal_confidence = "High"
                color = "red"
                reason_parts.append(f"RSI 高達 {rsi}，屬非常過熱區，即使趨勢強，現階段追入風險高。")
                if ma20 is not None:
                    suggested_entry = round(ma20, 2)
                    reason_parts.append(f"較合理做法係等股價回調接近 20 日線 {ma20} 附近再重新評估。")

            elif rsi is not None and rsi >= 65:
                breakout_status = "Extended"
                action = "Avoid"
                signal_confidence = "Medium"
                color = "red"
                reason_parts.append(f"RSI 為 {rsi}，動能偏熱，但未見有量突破支持，風險回報未算吸引。")
                if ma20 is not None:
                    suggested_entry = round(ma20, 2)
                    reason_parts.append(f"如要等更佳入場位，可先觀察股價會否回到 {ma20} 附近。")

            else:
                breakout_status = "Neutral"
                action = "Watch"
                signal_confidence = "Low"
                color = "blue"
                reason_parts.append(f"現價 {price} 暫未形成清晰 breakout 或 pullback setup，結構偏中性。")
                if recent_high is not None and recent_low is not None:
                    reason_parts.append(f"短線可先觀察上方阻力 {recent_high} 及下方支撐 {recent_low}。")

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
                    reason_parts.append(f"現價 {price} 已較買入價 {buy_price} 回落超過約 7%，走勢明顯轉弱。")
                    if sl is not None:
                        reason_parts.append(f"按 ATR 波幅估算，防守位可參考 {sl}。")

                elif rsi is not None and rsi >= 75 and unrealized_pl > 0:
                    action = "Trim"
                    signal_confidence = "High"
                    color = "orange"
                    reason_parts.append(f"持倉已有未實現盈利 {unrealized_pl}，而 RSI 高達 {rsi}，短線動能過熱。")
                    if pt is not None:
                        reason_parts.append(f"可考慮在 {pt} 附近分段止賺，或先鎖定部分盈利。")

                elif ma20 is not None and price >= ma20:
                    action = "Hold"
                    signal_confidence = "High"
                    color = "yellow"
                    reason_parts.append(f"現價 {price} 仍高於 20 日線 {ma20}，趨勢結構未見破壞。")
                    reason_parts.append(f"按目前股數 {shares} 計，持倉市值約 {position_value}，未實現盈虧為 {unrealized_pl}。")

                else:
                    action = "Hold"
                    signal_confidence = "Medium"
                    color = "yellow"
                    reason_parts.append("持倉仍可觀察，但現價對比短期均線優勢減弱，需留意是否跌穿關鍵支撐。")

            else:
                action = "Hold"
                signal_confidence = "Low"
                color = "gray"
                reason_parts.append("Holding 資料未齊，至少需要買入價及股數先可以完整分析持倉風險與盈虧。")

        reason = " ".join(reason_parts) if reason_parts else "暫未形成明確分析結論。"

        return {
            "price": price,
            "suggested_entry": suggested_entry,
            "pt": pt,
            "sl": sl,
            "weekly_est_high": weekly_est_high,
            "weekly_est_low": weekly_est_low,
            "breakout_status": breakout_status,
            "breakout_level": recent_high,
            "support_level": recent_low,
            "resistance_level": recent_high,
            "volume_vs_avg20": volume_ratio,
            "rsi": rsi,
            "ma20": ma20,
            "ma50": ma50,
            "atr14": atr14,
            "position_value": position_value,
            "unrealized_pl": unrealized_pl,
            "action": action,
            "signal_confidence": signal_confidence,
            "color": color,
            "reason": reason
        }

    except Exception as e:
        return {
            "price": None,
            "suggested_entry": None,
            "pt": None,
            "sl": None,
            "weekly_est_high": None,
            "weekly_est_low": None,
            "breakout_status": "Error",
            "breakout_level": None,
            "support_level": None,
            "resistance_level": None,
            "volume_vs_avg20": None,
            "rsi": None,
            "ma20": None,
            "ma50": None,
            "atr14": None,
            "position_value": None,
            "unrealized_pl": None,
            "action": "Error",
            "signal_confidence": "Low",
            "color": "gray",
            "reason": f"抓取數據失敗：{e}"
        }


def build_price_chart_with_volume(ticker, selected_row, period="6mo"):
    hist = get_hist_with_indicators(ticker, period)
    if hist is None or hist.empty:
        return None

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.72, 0.28]
    )

    fig.add_trace(
        go.Candlestick(
            x=hist.index,
            open=hist["Open"],
            high=hist["High"],
            low=hist["Low"],
            close=hist["Close"],
            name="Price",
            increasing_line_color="#16a34a",
            decreasing_line_color="#dc2626"
        ),
        row=1,
        col=1
    )

    fig.add_trace(
        go.Scatter(
            x=hist.index,
            y=hist["MA20"],
            mode="lines",
            name="MA20",
            line=dict(width=1.8, color="#f59e0b")
        ),
        row=1,
        col=1
    )

    fig.add_trace(
        go.Scatter(
            x=hist.index,
            y=hist["MA50"],
            mode="lines",
            name="MA50",
            line=dict(width=1.8, color="#2563eb")
        ),
        row=1,
        col=1
    )

    vol_colors = ["#16a34a" if c >= o else "#dc2626" for c, o in zip(hist["Close"], hist["Open"])]
    fig.add_trace(
        go.Bar(
            x=hist.index,
            y=hist["Volume"],
            name="Volume",
            marker_color=vol_colors,
            opacity=0.75
        ),
        row=2,
        col=1
    )

    fig.add_trace(
        go.Scatter(
            x=hist.index,
            y=hist["AvgVol20"],
            mode="lines",
            name="AvgVol20",
            line=dict(width=2, color="#7c3aed")
        ),
        row=2,
        col=1
    )

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
                row=1,
                col=1
            )

    fig.update_layout(
        height=680,
        margin=dict(l=20, r=20, t=45, b=20),
        xaxis_rangeslider_visible=False,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        title=f"{ticker} Price + Volume"
    )

    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)

    return fig


if "stock_data" not in st.session_state:
    st.session_state.stock_data = load_stocks_from_db()

if "selected_ticker" not in st.session_state:
    if len(st.session_state.stock_data) > 0:
        st.session_state.selected_ticker = st.session_state.stock_data[0]["ticker"]
    else:
        st.session_state.selected_ticker = None


def refresh_all_analysis():
    refreshed = []
    for row in st.session_state.stock_data:
        analysis = analyze_stock(
            row["ticker"],
            row["stock_type"],
            row.get("buy_price"),
            row.get("shares")
        )
        updated_row = row.copy()
        updated_row.update(analysis)
        refreshed.append(updated_row)
        update_stock_in_db(updated_row["id"], updated_row)
    st.session_state.stock_data = refreshed


def recalc_and_update_stock(row):
    analysis = analyze_stock(
        row["ticker"],
        row["stock_type"],
        row.get("buy_price"),
        row.get("shares")
    )
    row.update(analysis)
    update_stock_in_db(row["id"], row)


st.markdown(
    """
    <style>
    .status-badge {
        display: inline-block;
        padding: 6px 12px;
        border-radius: 999px;
        font-size: 13px;
        font-weight: 700;
        text-align: center;
        min-width: 120px;
    }
    .green { background-color: #DCFCE7; color: #166534; }
    .yellow { background-color: #FEF3C7; color: #92400E; }
    .orange { background-color: #FFEDD5; color: #9A3412; }
    .red { background-color: #FEE2E2; color: #991B1B; }
    .blue { background-color: #DBEAFE; color: #1D4ED8; }
    .gray { background-color: #E5E7EB; color: #374151; }

    .stock-card {
        border: 1px solid #e5e7eb;
        border-radius: 16px;
        padding: 14px 16px;
        margin-bottom: 12px;
        background: #ffffff;
        box-shadow: 0 1px 2px rgba(0,0,0,0.04);
    }

    .detail-header {
        border: 1px solid #e5e7eb;
        border-radius: 16px;
        padding: 16px 18px;
        margin-bottom: 12px;
        background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
    }

    .muted {
        color: #64748b;
        font-size: 13px;
    }

    .section-pill {
        display: inline-block;
        padding: 6px 12px;
        border-radius: 999px;
        font-size: 13px;
        font-weight: 700;
        margin-bottom: 12px;
        background: #eef2ff;
        color: #3730a3;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("US Stock Decision Dashboard")
st.caption("Version 6B.3 - Grouped dashboard + polished detail + volume average line")

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
                st.session_state.selected_ticker = ticker
                st.success(f"{ticker} added and saved successfully!")
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
    filtered_df["action_group"] = filtered_df["action"].apply(get_action_group)
    filtered_df = filtered_df.sort_values(by=["action_priority", "sort_order", "ticker"], ascending=[True, True, True])

    available_tickers = filtered_df["ticker"].tolist()
    if st.session_state.selected_ticker not in available_tickers and len(available_tickers) > 0:
        st.session_state.selected_ticker = available_tickers[0]
else:
    filtered_df = pd.DataFrame()

st.subheader("Portfolio Summary")

if not df.empty:
    holdings_df = df[df["stock_type"] == "Holding"].copy()

    total_holdings = len(holdings_df)
    total_watchlist = len(df[df["stock_type"] == "Watchlist"])
    total_market_value = round(holdings_df["position_value"].fillna(0).sum(), 2) if not holdings_df.empty else 0
    total_unrealized_pl = round(holdings_df["unrealized_pl"].fillna(0).sum(), 2) if not holdings_df.empty else 0
    high_conf_count = len(df[df["signal_confidence"] == "High"])
    action_needed_count = len(df[df["action"].isin([
        "Breakout Confirmed",
        "Breakout Watch",
        "Near Entry",
        "Buy Setup",
        "Trim",
        "Stop Loss",
        "Wait Pullback"
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

st.markdown("---")
st.subheader("Dashboard")

if filtered_df.empty:
    st.info("No stocks to display.")
else:
    for group_name in ["Action Needed", "Monitor", "Low Priority"]:
        group_df = filtered_df[filtered_df["action_group"] == group_name]
        if group_df.empty:
            continue

        st.markdown(f"<div class='section-pill'>{group_name}</div>", unsafe_allow_html=True)

        for _, row in group_df.iterrows():
            short_reason = build_short_reason(row)
            stock_type_label = "Watch" if row["stock_type"] == "Watchlist" else row["stock_type"]

            st.markdown('<div class="stock-card">', unsafe_allow_html=True)
            col1, col2, col3, col4, col5, col6 = st.columns([1.2, 1.2, 1.8, 1.0, 1.0, 1.0])

            with col1:
                st.markdown(f"### {row['ticker']}")
                st.caption(stock_type_label)

            with col2:
                st.metric("Price", row["price"])
                st.markdown(render_badge(row["action"], row["color"]), unsafe_allow_html=True)

            with col3:
                st.markdown("**Short Reason**")
                st.write(short_reason)
                st.markdown(f"<div class='muted'>Confidence: {row['signal_confidence']} | Breakout: {row['breakout_status']}</div>", unsafe_allow_html=True)

            with col4:
                st.metric("Entry", row["suggested_entry"])
                st.metric("PT", row["pt"])

            with col5:
                st.metric("SL", row["sl"])
                st.metric("RSI", row["rsi"])

            with col6:
                if st.button("Open Detail", key=f"open_{row['id']}"):
                    st.session_state.selected_ticker = row["ticker"]
                    st.rerun()

            st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")
st.subheader("Stock Detail View")

if not filtered_df.empty and st.session_state.selected_ticker:
    available_tickers = filtered_df["ticker"].tolist()

    selected_ticker = st.selectbox(
        "Choose a stock to view details",
        available_tickers,
        index=available_tickers.index(st.session_state.selected_ticker) if st.session_state.selected_ticker in available_tickers else 0,
        key="detail_selectbox"
    )
    st.session_state.selected_ticker = selected_ticker
    selected_row = filtered_df[filtered_df["ticker"] == selected_ticker].iloc[0]

    st.markdown('<div class="detail-header">', unsafe_allow_html=True)
    h1, h2, h3, h4, h5 = st.columns([1.4, 1.4, 1.0, 1.0, 1.2])
    h1.markdown(f"## {selected_ticker}")
    h2.markdown(render_badge(selected_row["action"], selected_row["color"]), unsafe_allow_html=True)
    h3.metric("Price", selected_row["price"])
    h4.metric("Confidence", selected_row["signal_confidence"])
    h5.metric("Type", selected_row["stock_type"])
    st.markdown(
        f"<div class='muted'>Breakout Status: {selected_row['breakout_status']} | Support: {selected_row['support_level']} | Resistance: {selected_row['resistance_level']}</div>",
        unsafe_allow_html=True
    )
    st.markdown('</div>', unsafe_allow_html=True)

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

        fig = build_price_chart_with_volume(selected_ticker, selected_row, chart_period)
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
        c8.metric("Volume vs Avg20", selected_row["volume_vs_avg20"])

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
else:
    st.info("Add your first stock from the sidebar.")

st.markdown("---")
st.subheader("Legend")

with st.expander("Action Badge Meaning", expanded=False):
    st.markdown("""
- **Breakout Confirmed**：已突破近期阻力，並有成交量確認，可開始留意追入策略。
- **Breakout Watch**：接近或測試阻力位，但未有足夠量能確認，先觀察。
- **Near Entry**：股價接近理想入場區，例如 MA20 / pullback 位，風險回報開始合理。
- **Buy Setup**：出現較清晰買入 setup，例如回調後 RSI 回落、接近支撐。
- **Wait Pullback**：走勢太熱，唔建議即追，等回調更合理。
- **Avoid**：現階段風險回報唔吸引，或者技術條件未支持入場。
- **Hold**：如果你已持有，暫時可繼續持有觀察。
- **Trim**：持倉有盈利而且動能過熱，可考慮減持部分倉位。
- **Stop Loss**：技術走勢轉差，或者已跌穿風險控制區，應優先處理風險。
""")

with st.expander("Signal Confidence Meaning", expanded=False):
    st.markdown("""
- **High**：多個條件同時支持呢個判斷，例如價位、RSI、breakout、volume 或趨勢結構一致。
- **Medium**：有部分條件支持，但未算非常完整，建議再觀察或等確認。
- **Low**：訊號未夠清晰，暫時只作參考，唔應單靠呢個訊號做決定。
""")

with st.expander("Type Meaning", expanded=False):
    st.markdown("""
- **Holding**：你已經持有嘅股票，重點係睇 Hold / Trim / Stop Loss / PT / SL。
- **Watch**：你未持有、但想觀察入場機會嘅股票，重點係睇 Buy Setup / Near Entry / Breakout Watch。
""")

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
                    old_ticker = selected_edit_row["ticker"]
                    selected_edit_row["ticker"] = new_ticker
                    selected_edit_row["stock_type"] = new_type
                    selected_edit_row["buy_price"] = new_buy_price
                    selected_edit_row["shares"] = new_shares

                    recalc_and_update_stock(selected_edit_row)
                    st.session_state.stock_data = load_stocks_from_db()

                    if st.session_state.selected_ticker == old_ticker:
                        st.session_state.selected_ticker = new_ticker

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
            if st.session_state.stock_data:
                st.session_state.selected_ticker = st.session_state.stock_data[0]["ticker"]
            else:
                st.session_state.selected_ticker = None
            st.rerun()
else:
    st.info("No stocks added yet.")