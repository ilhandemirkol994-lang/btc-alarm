import io, os, zipfile, requests
import numpy as np, pandas as pd
from collections import defaultdict

TOKEN = os.environ["TG_TOKEN"]
CHAT = os.environ["TG_CHAT_ID"]
SYMBOL = "BTCUSDT"
MONTHS = ["2025-11", "2025-12"] + [f"2026-{m:02d}" for m in range(1, 11)]
W, THR = 15, 0.25
TGT = 0.31
STOP = 3          # hesap için varsayılan stop (%)
NOTIONAL = 1600
FEE = NOTIONAL * 0.001
HORIZON = 7 * 24 * 60
COOL = 60

def load(m):
    url = (f"https://data.binance.vision/data/futures/um/monthly/klines/"
           f"{SYMBOL}/1m/{SYMBOL}-1m-{m}.zip")
    r = requests.get(url, timeout=180)
    if r.status_code != 200:
        return None
    z = zipfile.ZipFile(io.BytesIO(r.content))
    df = pd.read_csv(z.open(z.namelist()[0]), header=None)
    df = df[pd.to_numeric(df[0], errors="coerce").notna()]
    df = df.iloc[:, :6].astype(float)
    df.columns = ["t", "o", "h", "l", "c", "v"]
    return df

def send(text):
    for k in range(0, len(text), 3500):
        requests.post(f"https://api.telegram.org/bot{TOKEN}/sendMessage",
                      data={"chat_id": CHAT, "text": text[k:k + 3500]}, timeout=20)

parts = [d for d in (load(m) for m in MONTHS) if d is not None]
if not parts:
    send("Filtre testi: veri indirilemedi")
    raise SystemExit
df = pd.concat(parts).sort_values("t").reset_index(drop=True)
t = df["t"].values
if t.max() > 1e14:
    t = t // 1000
h, l, c, v = df["h"].values, df["l"].values, df["c"].values, df["v"].values

sc, sv = pd.Series(c), pd.Series(v)
sma240 = sc.rolling(240).mean().values
sma1440 = sc.rolling(1440).mean().values
v5 = sv.rolling(5).mean().values
v60 = sv.rolling(60).mean().values
rg = (pd.Series(h) - pd.Series(l)) / sc * 100
rg60 = rg.rolling(60).mean().values
rg1440 = rg.rolling(1440).mean().values

def find_events():
    ev, last = [], -10**9
    for i in range(1440, len(c) - 1):
        start = max(i - W, last)
        seg = h[start:i + 1]
        hi = seg.max()
        if (c[i] - hi) / hi * 100 <= -THR and i - last > COOL:
            ev.append((i, start + int(seg.argmax())))
            last = i
    return ev

def pnl_for(i):
    end = min(i + 1 + HORIZON, len(c))
    if end - i - 1 < 60:
        return None
    entry = c[i]
    hh, ll = h[i + 1:end], l[i + 1:end]
    hit = np.nonzero(hh >= entry * (1 + TGT / 100))[0]
    st = np.nonzero(ll <= entry * (1 - STOP / 100))[0]
    kh = hit[0] if len(hit) else None
    ks = st[0] if len(st) else None
    if kh is not None and (ks is None or kh < ks):
        return NOTIONAL * TGT / 100 - FEE
    if ks is not None:
        return -NOTIONAL * STOP / 100 - FEE
    return NOTIONAL * (c[end - 1] / entry - 1) - FEE

def feats(i, hi_idx):
    f = {}
    f["Trend (24sa ort.)"] = "üstünde" if c[i] > sma1440[i] else "altında"
    f["Trend (4sa ort.)"] = "üstünde" if c[i] > sma240[i] else "altında"
    sp = i - hi_idx
    f["Düşüş hızı"] = "hızlı ≤3dk" if sp <= 3 else ("orta 4-10dk" if sp <= 10 else "yavaş >10dk")
    r = v5[i] / v60[i] if v60[i] > 0 else 1
    f["Hacim"] = "yüksek >2x" if r > 2 else ("normal 1-2x" if r >= 1 else "düşük <1x")
    q = rg60[i] / rg1440[i] if rg1440[i] > 0 else 1
    f["Oynaklık"] = "yüksek" if q > 1.3 else "normal/düşük"
    ch4 = (c[i] / c[i - 240] - 1) * 100
    f["4sa değişim"] = "sert düşüş <-1%" if ch4 < -1 else "diğer"
    hr = int((t[i] // 3600000) % 24)
    f["Saat (UTC)"] = "00-08" if hr < 8 else ("08-16" if hr < 16 else "16-24")
    return f

rows = []
for i, hi_idx in find_events():
    p = pnl_for(i)
    if p is None:
        continue
    rows.append((len(rows), feats(i, hi_idx), p))

half = len(rows) // 2

def line(name, items):
    p = np.array([x[1] for x in items])
    n = len(p)
    if n < 30:
        return None
    h1 = np.array([x[1] for x in items if x[0] < half])
    h2 = np.array([x[1] for x in items if x[0] >= half])
    ev, se = p.mean(), p.std() / np.sqrt(n)
    fm = lambda a: f"{a.mean():+.1f}" if len(a) else "-"
    star = ""
    if ev - 2 * se > 0 and len(h1) and len(h2) and h1.mean() > 0 and h2.mean() > 0:
        star = " ⭐"
    return (f"{name}: n={n} | kazanç %{(p > 0).mean() * 100:.0f} | "
            f"EV {ev:+.2f}±{se:.2f}$ (ilk {fm(h1)} / son {fm(h2)}){star}")

out = [f"Filtre testi: {len(rows)} olay, 15dk %{THR}, hedef %{TGT}, stop %{STOP} (varsayım)",
       f"Rastgele beklenen kazanç oranı: %{STOP / (STOP + TGT) * 100:.1f}"]
out.append(line("TÜMÜ", [(r[0], r[2]) for r in rows]))
groups = defaultdict(list)
for k, f, p in rows:
    for name, b in f.items():
        groups[(name, b)].append((k, p))
last_name = None
for (name, b) in sorted(groups):
    if name != last_name:
        out.append("")
        last_name = name
    ln = line(f"{name} {b}", groups[(name, b)])
    if ln:
        out.append(ln)

text = "\n".join(x for x in out if x is not None)
print(text)
send(text)
