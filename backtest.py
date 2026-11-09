import io, os, zipfile, requests
import numpy as np, pandas as pd

TOKEN = os.environ["TG_TOKEN"]
CHAT = os.environ["TG_CHAT_ID"]
SYMBOL = "BTCUSDT"
MONTHS = ["2025-11", "2025-12"] + [f"2026-{m:02d}" for m in range(1, 11)]
RULES = [(15, 0.25), (30, 0.27)]
TGT = 0.31
HORIZON = 7 * 24 * 60
LEVELS = [0.5, 1, 2, 3, 5, 10]

def load(m):
    url = (f"https://data.binance.vision/data/futures/um/monthly/klines/"
           f"{SYMBOL}/1m/{SYMBOL}-1m-{m}.zip")
    r = requests.get(url, timeout=180)
    if r.status_code != 200:
        print("yok:", m, r.status_code)
        return None
    z = zipfile.ZipFile(io.BytesIO(r.content))
    df = pd.read_csv(z.open(z.namelist()[0]), header=None)
    df = df[pd.to_numeric(df[0], errors="coerce").notna()]
    df = df.iloc[:, :6].astype(float)
    df.columns = ["t", "o", "h", "l", "c", "v"]
    return df

def find_events(h, c, W, thr, cool=5):
    ev, last = [], -10**9
    for i in range(W, len(c) - 1):
        start = max(i - W, last)
        hi = h[start:i + 1].max()
        if (c[i] - hi) / hi * 100 <= -thr and i - last > cool:
            ev.append(i)
            last = i
    return ev

def measure(ev, h, l, c):
    rows = []
    for i in ev:
        end = min(i + 1 + HORIZON, len(c))
        if end - i - 1 < 60:
            continue
        entry = c[i]
        hh, ll = h[i + 1:end], l[i + 1:end]
        hit = np.nonzero(hh >= entry * (1 + TGT / 100))[0]
        kh = int(hit[0]) if len(hit) else None
        seg = ll[:kh + 1] if kh is not None else ll
        mae = (seg.min() / entry - 1) * 100
        rows.append((kh, mae))
    return rows

def report(name, rows):
    n = len(rows)
    if n == 0:
        return [f"■ {name}: olay yok"]
    hit = np.array([r[0] + 1 for r in rows if r[0] is not None])
    mae = np.array([r[1] for r in rows])
    within = lambda m: (hit <= m).sum() / n * 100
    L = [f"■ {name}: {n} olay",
         f"Hedefe ulaşma: 30dk %{within(30):.0f} | 1sa %{within(60):.0f} | "
         f"4sa %{within(240):.0f} | 24sa %{within(1440):.0f} | 7g %{len(hit)/n*100:.0f}"]
    if len(hit):
        L.append(f"Bekleme (ulaşanlarda): ortanca {np.median(hit):.0f} dk, "
                 f"%90'ı {np.percentile(hit, 90):.0f} dk içinde")
    L.append(f"Hedefe kadar en derin düşüş: ortanca %{np.median(mae):.2f}, "
             f"p99 %{np.percentile(mae, 1):.2f}, en kötü %{mae.min():.2f}")
    L.append("Hedefe gelmeden şu kadar düşen olay: " +
             " | ".join(f"%{x}: %{(mae <= -x).mean() * 100:.1f}" for x in LEVELS))
    return L

def send(text):
    for k in range(0, len(text), 3500):
        requests.post(f"https://api.telegram.org/bot{TOKEN}/sendMessage",
                      data={"chat_id": CHAT, "text": text[k:k + 3500]}, timeout=20)

parts = [d for d in (load(m) for m in MONTHS) if d is not None]
if not parts:
    send("Backtest: veri indirilemedi")
    raise SystemExit
df = pd.concat(parts).sort_values("t").reset_index(drop=True)
h, l, c = df["h"].values, df["l"].values, df["c"].values

out = [f"Veri: {len(df)} dakika ({len(parts)} ay), hedef %{TGT}"]
for W, thr in RULES:
    rows = measure(find_events(h, c, W, thr), h, l, c)
    half = len(rows) // 2
    out += report(f"{W}dk %{thr} TÜMÜ", rows)
    out += report(f"{W}dk %{thr} ilk yarı", rows[:half])
    out += report(f"{W}dk %{thr} son yarı", rows[half:])
text = "\n".join(out)
print(text)
send(text)
