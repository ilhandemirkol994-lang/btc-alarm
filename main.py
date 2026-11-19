
import os
import json
import math
import requests
import pandas as pd
import numpy as np

BASE_URL = "https://fapi.binance.com"

SYMBOL = os.getenv("SYMBOL", "BTCUSDT")
TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]

MIN_SCORE = int(os.getenv("MIN_SCORE", "78"))
RISK_USDT = float(os.getenv("RISK_USDT", "5"))
CAPITAL = float(os.getenv("CAPITAL_USDT", "250"))
LEVERAGE = float(os.getenv("LEVERAGE", "9"))

STATE_FILE = ".state/last_signal.json"


def get_data(interval, limit=300):
    response = requests.get(
        f"{BASE_URL}/fapi/v1/klines",
        params={
            "symbol": SYMBOL,
            "interval": interval,
            "limit": limit
        },
        timeout=20
    )
    response.raise_for_status()

    columns = [
        "time", "open", "high", "low", "close",
        "volume", "close_time", "qvol", "trades",
        "taker_base", "taker_quote", "ignore"
    ]

    df = pd.DataFrame(response.json(), columns=columns)

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)

    df["close_time"] = pd.to_datetime(
        df["close_time"], unit="ms", utc=True
    )

    # Sadece kapanmış mumları kullan.
    now = pd.Timestamp.now(tz="UTC")
    df = df[df["close_time"] < now]

    return df.reset_index(drop=True)


def ema(series, period):
    return series.ewm(
        span=period, adjust=False
    ).mean()


def rsi(series, period=14):
    delta = series.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1/period, adjust=False
    ).mean()

    avg_loss = loss.ewm(
        alpha=1/period, adjust=False
    ).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    return 100 - (100 / (1 + rs))


def atr(df, period=14):
    previous_close = df["close"].shift(1)

    tr = pd.concat([
        df["high"] - df["low"],
        abs(df["high"] - previous_close),
        abs(df["low"] - previous_close)
    ], axis=1).max(axis=1)

    return tr.ewm(
        alpha=1/period, adjust=False
    ).mean()


def adx(df, period=14):
    up = df["high"].diff()
    down = -df["low"].diff()

    plus_dm = pd.Series(
        np.where((up > down) & (up > 0), up, 0),
        index=df.index
    )

    minus_dm = pd.Series(
        np.where((down > up) & (down > 0), down, 0),
        index=df.index
    )

    average_atr = atr(df, period).replace(0, np.nan)

    plus_di = 100 * (
        plus_dm.ewm(alpha=1/period, adjust=False).mean()
        / average_atr
    )

    minus_di = 100 * (
        minus_dm.ewm(alpha=1/period, adjust=False).mean()
        / average_atr
    )

    dx = 100 * abs(plus_di - minus_di) / (
        plus_di + minus_di
    ).replace(0, np.nan)

    return (
        dx.ewm(alpha=1/period, adjust=False).mean(),
        plus_di,
        minus_di
    )


def macd(series):
    line = ema(series, 12) - ema(series, 26)
    signal = ema(line, 9)

    return line, signal, line - signal


def supertrend(df, period=10, multiplier=3):
    average_atr = atr(df, period)

    middle = (df["high"] + df["low"]) / 2

    upper = middle + multiplier * average_atr
    lower = middle - multiplier * average_atr

    final_upper = upper.copy()
    final_lower = lower.copy()

    direction = pd.Series(
        1.0, index=df.index
    )

    for i in range(1, len(df)):

        if (
            upper.iloc[i] < final_upper.iloc[i-1]
            or df["close"].iloc[i-1] > final_upper.iloc[i-1]
        ):
            final_upper.iloc[i] = upper.iloc[i]
        else:
            final_upper.iloc[i] = final_upper.iloc[i-1]

        if (
            lower.iloc[i] > final_lower.iloc[i-1]
            or df["close"].iloc[i-1] < final_lower.iloc[i-1]
        ):
            final_lower.iloc[i] = lower.iloc[i]
        else:
            final_lower.iloc[i] = final_lower.iloc[i-1]

        if direction.iloc[i-1] == -1:
            direction.iloc[i] = (
                1 if df["close"].iloc[i] > final_upper.iloc[i]
                else -1
            )
        else:
            direction.iloc[i] = (
                -1 if df["close"].iloc[i] < final_lower.iloc[i]
                else 1
            )

    return direction


def ichimoku(df):
    tenkan = (
        df["high"].rolling(9).max()
        + df["low"].rolling(9).min()
    ) / 2

    kijun = (
        df["high"].rolling(26).max()
        + df["low"].rolling(26).min()
    ) / 2

    span_a = (tenkan + kijun) / 2

    span_b = (
        df["high"].rolling(52).max()
        + df["low"].rolling(52).min()
    ) / 2

    return tenkan, kijun, span_a, span_b


def analyze():

    h1 = get_data("1h")
    m15 = get_data("15m")

    if min(len(h1), len(m15)) < 220:
        raise RuntimeError("Yeterli mum verisi yok.")

    # 1 saatlik ana trend
    h1["ema50"] = ema(h1["close"], 50)
    h1["ema200"] = ema(h1["close"], 200)
    h1["adx"], h1["pdi"], h1["mdi"] = adx(h1)

    ht = h1.iloc[-1]

    trend_up = (
        ht["close"] > ht["ema200"]
        and ht["ema50"] > ht["ema200"]
    )

    trend_down = (
        ht["close"] < ht["ema200"]
        and ht["ema50"] < ht["ema200"]
    )

    # 15 dakikalık analiz
    df = m15.copy()

    df["ema20"] = ema(df["close"], 20)
    df["ema50"] = ema(df["close"], 50)

    df["rsi"] = rsi(df["close"])
    df["atr"] = atr(df)

    df["adx"], df["pdi"], df["mdi"] = adx(df)

    df["supertrend"] = supertrend(df)

    df["macd"], df["macd_signal"], df["macd_hist"] = (
        macd(df["close"])
    )

    tenkan, kijun, span_a, span_b = ichimoku(df)

    df["tenkan"] = tenkan
    df["kijun"] = kijun

    df["cloud_top"] = pd.concat(
        [span_a, span_b], axis=1
    ).max(axis=1)

    df["cloud_bottom"] = pd.concat(
        [span_a, span_b], axis=1
    ).min(axis=1)

    # Gaussian kanal yaklaşımı
    df["gaussian"] = ema(
        ema(df["close"], 20), 20
    )

    df["volume_avg"] = df["volume"].rolling(20).mean()

    x = df.iloc[-1]

    price = float(x["close"])

    long_score = 0
    short_score = 0

    long_reasons = []
    short_reasons = []

    def add(condition, side, points, reason):

        nonlocal long_score, short_score

        if condition:

            if side == "LONG":
                long_score += points
                long_reasons.append(reason)

            else:
                short_score += points
                short_reasons.append(reason)

    # Trend ve ADX
    add(
        trend_up and ht["adx"] >= 18,
        "LONG", 20, "1 saatlik trend yukarı"
    )

    add(
        trend_down and ht["adx"] >= 18,
        "SHORT", 20, "1 saatlik trend aşağı"
    )

    # Supertrend
    add(
        x["supertrend"] == 1,
        "LONG", 18, "Supertrend pozitif"
    )

    add(
        x["supertrend"] == -1,
        "SHORT", 18, "Supertrend negatif"
    )

    # MACD ve RSI
    add(
        x["macd"] > x["macd_signal"]
        and x["macd_hist"] > 0
        and 52 <= x["rsi"] <= 75,
        "LONG", 18, "MACD ve RSI pozitif"
    )

    add(
        x["macd"] < x["macd_signal"]
        and x["macd_hist"] < 0
        and 25 <= x["rsi"] <= 48,
        "SHORT", 18, "MACD ve RSI negatif"
    )

    # Gaussian ve EMA
    add(
        price > x["gaussian"]
        and price > x["ema20"]
        and x["ema20"] > x["ema50"],
        "LONG", 14, "Gaussian ve EMA uyumu"
    )

    add(
        price < x["gaussian"]
        and price < x["ema20"]
        and x["ema20"] < x["ema50"],
        "SHORT", 14, "Gaussian ve EMA uyumu"
    )

    # Ichimoku
    add(
        price > x["cloud_top"]
        and x["tenkan"] > x["kijun"],
        "LONG", 14, "Ichimoku pozitif"
    )

    add(
        price < x["cloud_bottom"]
        and x["tenkan"] < x["kijun"],
        "SHORT", 14, "Ichimoku negatif"
    )

    # Hacim ve DI
    volume_ok = x["volume"] > x["volume_avg"]

    add(
        volume_ok and x["pdi"] > x["mdi"],
        "LONG", 16, "Hacim ve pozitif DI onayı"
    )

    add(
        volume_ok and x["mdi"] > x["pdi"],
        "SHORT", 16, "Hacim ve negatif DI onayı"
    )

    side = "WAIT"
    score = max(long_score, short_score)
    reasons = []

    if long_score >= MIN_SCORE and long_score > short_score:
        side = "LONG"
        score = long_score
        reasons = long_reasons

    elif short_score >= MIN_SCORE and short_score > long_score:
        side = "SHORT"
        score = short_score
        reasons = short_reasons

    # ATR stop ve hedefler
    current_atr = float(x["atr"])

    if side == "LONG":
        stop = price - 1.5 * current_atr
        target1 = price + 3 * current_atr
        target2 = price + 4.5 * current_atr

    elif side == "SHORT":
        stop = price + 1.5 * current_atr
        target1 = price - 3 * current_atr
        target2 = price - 4.5 * current_atr

    else:
        stop = target1 = target2 = None

    if stop is not None:
        stop_pct = abs(price - stop) / price

        notional = min(
            RISK_USDT / stop_pct,
            CAPITAL * LEVERAGE
        )

        margin = notional / LEVERAGE

    else:
        notional = margin = None

    return {
        "side": side,
        "score": int(score),
        "long_score": int(long_score),
        "short_score": int(short_score),
        "price": price,
        "stop": stop,
        "target1": target1,
        "target2": target2,
        "notional": notional,
        "margin": margin,
        "candle": x["close_time"].isoformat(),
        "reasons": reasons,
        "rsi": float(x["rsi"]),
        "adx": float(x["adx"]),
        "trend": (
            "YUKARI" if trend_up
            else "AŞAĞI" if trend_down
            else "BELİRSİZ"
        )
    }


def send_telegram(message):

    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"

    response = requests.post(
        url,
        json={
            "chat_id": CHAT_ID,
            "text": message
        },
        timeout=20
    )

    response.raise_for_status()


def load_state():
    try:
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_state(state):
    os.makedirs(".state", exist_ok=True)

    with open(STATE_FILE, "w") as f:
        json.dump(state, f)


def fmt(value):
    return f"{value:,.2f}" if value is not None else "-"


def main():

    signal = analyze()

    print(json.dumps(signal, ensure_ascii=False, indent=2))

    if signal["side"] == "WAIT":
        print("Sinyal yok.")
        return

    state = load_state()

    key = f'{signal["candle"]}:{signal["side"]}'

    if state.get("last_key") == key:
        print("Bu sinyal daha önce gönderildi.")
        return

    message = (
        f"🚨 BTCUSDT FUTURES {signal['side']}\n\n"
        f"Skor: {signal['score']}/100\n"
        f"Fiyat: {fmt(signal['price'])}\n"
        f"Stop: {fmt(signal['stop'])}\n"
        f"Hedef 1: {fmt(signal['target1'])}\n"
        f"Hedef 2: {fmt(signal['target2'])}\n\n"
        f"Pozisyon büyüklüğü: {fmt(signal['notional'])} USDT\n"
        f"Tahmini teminat: {fmt(signal['margin'])} USDT\n"
        f"Risk bütçesi: {RISK_USDT} USDT\n\n"
        f"1 saatlik trend: {signal['trend']}\n"
        f"RSI: {signal['rsi']:.1f}\n"
        f"ADX: {signal['adx']:.1f}\n\n"
        "Teyitler:\n- "
        + "\n- ".join(signal["reasons"])
        + "\n\n⚠️ Bu bildirim otomatik emir değildir. "
        "İşlem öncesi piyasa ve risk koşullarını kontrol edin."
    )

    send_telegram(message)

    state["last_key"] = key
    save_state(state)

    print("Telegram bildirimi gönderildi.")


if __name__ == "__main__":
    main()
