import io, os, zipfile, requests
import numpy as np, pandas as pd

TOKEN = os.environ["TG_TOKEN"]
CHAT = os.environ["TG_CHAT_ID"]
SYMBOL = "BTCUSDT"
FEE = 0.0012
NOTIONAL = 1600
H = 96
KS = [1.0, 1.5, 2.0]
KT = [1.0, 1.5, 2.0, 3.0, "7$"]

def months():
    out, y, m = [], 2023, 11
    while (y, m) <= (2026, 10):
        out.append(f"{y}-{m:02d}")
        m += 1
        if m > 12:
            y, m = y + 1, 1
    return out

def load15(m):
    url = (f"https://data.binance.vision/data/futures/um/monthly/klines/"
           f"{SYMBOL}/1m/{SYMBOL}-1m-{m}.zip")
    try:
        r = requests.get(url, timeout=180)
    except Exception:
        return None
    if r.status_code != 200:
        return None
    z = zipfile.ZipFile(io.BytesIO(r.content))
    df = pd.read_csv(z.open(z.namelist()[0]), header=None)
    df = df[pd.to_numeric(df[0], errors="coerce").notna()]
    df = df.iloc[:, :6].astype(float)
    df.columns = ["t", "o", "h", "l", "c", "v"]
    t = df["t"].values.astype("int64")
    if t.max() > 10**14:
        t = t // 1000
    df["g"] = t // 900000
    g = df.groupby("g").agg(o=("o", "first"), h=("h", "max"),
                            l=("l", "min"), c=("c", "last"), v=("v", "sum"))
    g["t"] = g.index * 900000
    return g.reset_index(drop=True)

def send(text):
    for k in range(0, len(text), 3500):
        requests.post(f"https://api.telegram.org/bot{TOKEN}/sendMessage",
                      data={"chat_id": CHAT, "text": text[k:k + 3500]}, timeout=20)

parts = [d for d in (load15(m) for m in months()) if d is not None]
if not parts:
    send("Araştırma: veri indirilemedi")
    raise SystemExit
bars = pd.concat(parts).sort_values("t").drop_duplicates("t").reset_index(drop=True)
o, h, l, c = (bars[k].values for k in ("o", "h", "l", "c"))
n = len(bars)
split = int(n * 2 / 3)

close, high, low = pd.Series(c), pd.Series(h), pd.Series(l)
tr = pd.concat([high - low, (high - close.shift()).abs(),
                (low - close.shift()).abs()], axis=1).max(axis=1)
atr = tr.rolling(14).mean().values
ema = close.ewm(span=384, adjust=False).mean()
up = (close > ema).values
delta = close.diff()
gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
rsi = (100 - 100 / (1 + gain / loss)).values

S = {}
for x in (25, 30, 35, 40):
    S[f"Pullback RSI<{x}"] = up & (rsi < x)
for N in (32, 64, 128):
    hh = high.shift(1).rolling(N).max().values
    S[f"Kırılım {N} bar"] = up & (c > hh)
for N in (16, 48):
    for k in (3, 5):
        S[f"Momentum N{N} k{k}"] = up & ((c - close.shift(N).values) / atr > k)
mid = close.rolling(20).mean()
sd = close.rolling(20).std()
bw = (4 * sd) / mid
upper = (mid + 2 * sd).values
for q in (0.1, 0.2):
    sq = (bw < bw.rolling(480).quantile(q)).astype(float).rolling(12).max().values > 0
    S[f"Sıkışma kırılımı q{q}"] = up & sq & (c > upper)
for k in (1.5, 2.5):
    dip = (high.rolling(4).max().values - c) / atr
    S[f"Trendde dip k{k}"] = up & (dip > k)

def sim(sig, ks, kt):
    res, busy = [], -1
    for i in sig:
        if i <= busy or i + 1 + H >= n:
            continue
        a = atr[i]
        if not a > 0:
            continue
        e = o[i + 1]
        stop = e - ks * a
        tgt = e * 1.004375 if kt == "7$" else e + kt * a
        lo, hi = l[i + 1:i + 1 + H], h[i + 1:i + 1 + H]
        s = np.flatnonzero(lo <= stop)
        t_ = np.flatnonzero(hi >= tgt)
        js = s[0] if len(s) else H
        jt = t_[0] if len(t_) else H
        if js <= jt and js < H:
            ex, k = stop, js
        elif jt < js:
            ex, k = tgt, jt
        else:
            ex, k = c[i + H], H - 1
        res.append((i, ex / e - 1 - FEE))
        busy = i + 1 + k
    return res

def stats(r):
    r = np.array(r)
    m = len(r)
    if m < 2:
        return None
    mean, sdv = r.mean(), r.std(ddof=1)
    t = mean / (sdv / np.sqrt(m)) if sdv > 0 else 0
    return m, (r > 0).mean() * 100, mean * NOTIONAL, t

results = []
for name, s in S.items():
    onset = s & ~np.r_[False, s[:-1]]
    sig = np.flatnonzero(onset)
    for ks in KS:
        for kt in KT:
            tr_ = sim(sig, ks, kt)
            a = stats([x[1] for x in tr_ if x[0] < split])
            b = stats([x[1] for x in tr_ if x[0] >= split])
            if a and b:
                results.append((name, ks, kt, a, b))

cands = [r for r in results if r[3][0] >= 150 and r[3][3] >= 2 and r[3][2] > 0]
cands.sort(key=lambda r: -r[3][3])
ok = [r for r in cands if r[4][0] >= 50 and r[4][2] > 0 and r[4][3] >= 1]

out = [f"Veri: {n} adet 15dk mum, ~36 ay. Eğitim ilk 24 ay, test son 12 ay",
       f"Ortanca 15dk ATR: %{np.nanmedian(atr / c) * 100:.2f}",
       f"Denenen kombinasyon: {len(results)} | eğitimde anlamlı: {len(cands)} | "
       f"testte de geçen: {len(ok)}",
       "(Şans eseri beklenen yanlış pozitif: birkaç tane. Test sonucu belirleyici.)", ""]
for name, ks, kt, a, b in cands[:8]:
    good = b[0] >= 50 and b[2] > 0 and b[3] >= 1
    hedef = kt if kt == "7$" else f"{kt}×ATR"
    out.append(f"{name} | stop {ks}×ATR hedef {hedef} {'✅' if good else '❌'}")
    out.append(f"  EĞİTİM n={a[0]} kazanç %{a[1]:.0f} EV {a[2]:+.1f}$ t={a[3]:.1f}")
    out.append(f"  TEST   n={b[0]} kazanç %{b[1]:.0f} EV {b[2]:+.1f}$ t={b[3]:.1f}")
if not cands:
    out.append("Eğitimde anlamlı çıkan strateji yok.")
text = "\n".join(out)
print(text)
send(text)
