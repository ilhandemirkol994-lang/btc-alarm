import ccxt
import pandas as pd
import numpy as np
import requests
import os

# ============ 配置（从 GitHub Secrets 读取）============
TG_TOKEN = os.environ.get("TG_TOKEN")
TG_CHAT_ID = os.environ.get("TG_CHAT_ID")

SYMBOLS = ["BTC/USDT:USDT", "ETH/USDT:USDT"]
TF_ENTRY = "15m"
TF_TREND = "4h"
LIMIT = 500  # GitHub Actions 里跑，不用太多

STOP_PCT = 0.01
TP1_PCT = 0.004
TP1_RATIO = 0.60
TP2_PCT = 0.01

exchange = ccxt.binanceusdm({"enableRateLimit": True})

def send_telegram(msg):
    if not TG_TOKEN or not TG_CHAT_ID:
        print("Telegram 配置缺失，跳过通知")
        return
    url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
    try:
        requests.post(url, json={"chat_id": TG_CHAT_ID, "text": msg, "parse_mode": "HTML"}, timeout=10)
    except Exception as e:
        print(f"Telegram 发送失败: {e}")

def fetch_df(symbol, tf, limit=300):
    ohlcv = exchange.fetch_ohlcv(symbol, tf, limit=limit)
    df = pd.DataFrame(ohlcv, columns=["ts","open","high","low","close","vol"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms")
    return df

def ema(s, n): return s.ewm(span=n, adjust=False).mean()

def rsi(s, n=14):
    d = s.diff()
    up = d.clip(lower=0).rolling(n).mean()
    dn = -d.clip(upper=0).rolling(n).mean()
    return 100 - 100/(1 + up/dn)

def check_signal(symbol):
    try:
        df4 = fetch_df(symbol, TF_TREND)
        df15 = fetch_df(symbol, TF_ENTRY)
    except Exception as e:
        print(f"{symbol} 数据获取失败: {e}")
        return None, None

    df4["ema50"] = ema(df4["close"], 50)
    df4["ema200"] = ema(df4["close"], 200)
    trend_up = df4["ema50"].iloc[-1] > df4["ema200"].iloc[-1]
    trend_dn = df4["ema50"].iloc[-1] < df4["ema200"].iloc[-1]

    df15["ema50"] = ema(df15["close"], 50)
    df15["rsi"] = rsi(df15["close"])
    df15["vol_sma"] = df15["vol"].rolling(20).mean()

    last = df15.iloc[-1]; prev = df15.iloc[-2]
    near_ema = abs(last["low"] - last["ema50"])/last["ema50"] < 0.003 or \
               abs(last["close"] - last["ema50"])/last["ema50"] < 0.003
    vol_ok = last["vol"] > last["vol_sma"]

    if trend_up and near_ema and prev["rsi"] < 40 and last["rsi"] > 40 and vol_ok:
        return "LONG", last["close"]
    if trend_dn and near_ema and prev["rsi"] > 60 and last["rsi"] < 60 and vol_ok:
        return "SHORT", last["close"]
    return None, None

def main():
    signals = []
    for sym in SYMBOLS:
        side, price = check_signal(sym)
        if side:
            if side == "LONG":
                stop = price*(1-STOP_PCT); tp1 = price*(1+TP1_PCT); tp2 = price*(1+TP2_PCT)
            else:
                stop = price*(1+STOP_PCT); tp1 = price*(1-TP1_PCT); tp2 = price*(1-TP2_PCT)
            
            msg = (f"<b>{side} {sym}</b>\n"
                   f"Giriş: {price:.2f}\n"
                   f"Stop: {stop:.2f} (~-40$)\n"
                   f"TP1 (%60): {tp1:.2f} (~+20$)\n"
                   f"TP1 sonrası stop → giriş\n"
                   f"TP2 (%40): {tp2:.2f}")
            signals.append(msg)

    if signals:
        full_msg = "📊 <b>Vadeli Sinyal</b>\n\n" + "\n\n".join(signals)
        send_telegram(full_msg)
        print("Sinyal gönderildi")
    else:
        print("Sinyal yok")

if __name__ == "__main__":
    main()
