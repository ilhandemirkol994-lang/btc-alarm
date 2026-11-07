import os, time, requests
from collections import deque

TOKEN = os.environ["TG_TOKEN"]
CHAT = os.environ["TG_CHAT_ID"]
SYMBOL = "BTCUSDT"
DROP = 0.30
WINDOW = 15 * 60
RUN_FOR = 5 * 3600 + 50 * 60
COOLDOWN = 300

URLS = [
    "https://fapi.binance.com/fapi/v1/ticker/price",
    "https://data-api.binance.vision/api/v3/ticker/price",
]

def price():
    for u in URLS:
        try:
            r = requests.get(u, params={"symbol": SYMBOL}, timeout=5)
            if r.status_code == 200:
                return float(r.json()["price"])
        except Exception:
            pass
    return None

def send(text):
    requests.post(f"https://api.telegram.org/bot{TOKEN}/sendMessage",
                  data={"chat_id": CHAT, "text": text}, timeout=10)

start = time.time()
hist = deque()
last_alert = 0
send("BTC alarm botu başladı ✅")

while time.time() - start < RUN_FOR:
    p = price()
    now = time.time()
    if p:
        hist.append((now, p))
        while hist and now - hist[0][0] > WINDOW:
            hist.popleft()
        high = max(x[1] for x in hist)
        change = (p - high) / high * 100
        if change <= -DROP and now - last_alert > COOLDOWN:
            send(f"🔻 {SYMBOL} son 15 dk'da %{abs(change):.2f} düştü\n"
                 f"Zirve: {high:.1f} → Şimdi: {p:.1f}")
            last_alert = now
            hist.clear()
    time.sleep(2)
