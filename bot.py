import os, time, requests
from collections import deque

TOKEN = os.environ["TG_TOKEN"]
CHAT = os.environ["TG_CHAT_ID"]
SYMBOL = "BTCUSDT"

# pencere (dakika): eşik (yüzde)
WINDOWS = {15: 0.38, 30: 0.52}

RUN_FOR = 5 * 3600 + 50 * 60
COOLDOWN = 300
MAX_WINDOW = max(WINDOWS) * 60

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
    try:
        requests.post(f"https://api.telegram.org/bot{TOKEN}/sendMessage",
                      data={"chat_id": CHAT, "text": text}, timeout=10)
    except Exception:
        pass

start = time.time()
hist = deque()
last_alert = {w: 0 for w in WINDOWS}
send("BTC alarm botu başladı ✅ (15 dk + 30 dk)")

while time.time() - start < RUN_FOR:
    p = price()
    now = time.time()
    if p:
        hist.append((now, p))
        while hist and now - hist[0][0] > MAX_WINDOW:
            hist.popleft()

        hits = []
        for w, drop in WINDOWS.items():
            since = max(now - w * 60, last_alert[w])
            high = max(x[1] for x in hist if x[0] >= since)
            change = (p - high) / high * 100
            if change <= -drop and now - last_alert[w] > COOLDOWN:
                hits.append((w, abs(change), high))

        if hits:
            lines = [f"🔻 {SYMBOL} düşüş alarmı"]
            for w, ch, high in hits:
                lines.append(f"Son {w} dk: %{ch:.2f} düştü (zirve {high:.1f})")
                last_alert[w] = now
            lines.append(f"Şimdi: {p:.1f}")
            send("\n".join(lines))
    time.sleep(2)
