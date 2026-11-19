import io, os, zipfile, requests
import numpy as np, pandas as pd

TOKEN = os.environ["TG_TOKEN"]
CHAT = os.environ["TG_CHAT_ID"]
SYMBOLS = ["BTCUSDT", "ETHUSDT"]
CAPITAL, LEV = 250, 8
NOTIONAL = CAPITAL * LEV
TGT = 20 / NOTIONAL * 100
STOP = 40 / NOTIONAL * 100
FEE_PCT = 0.12
HOLD = 72

def months():
    out, y, m = [], 2022, 1
    while (y, m) <= (2026, 10):
        out.append(f"{y}-{m:02d}")
        m += 1
        if m > 12:
            y, m = y + 1, 1
    return out

def load1h(sym, m):
    url = (f"https://data.binance.vision/data/futures/um/monthly/klines/"
           f"{sym}/1m/{sym}-1m-{m}.zip")
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
    df["g"] = t // 3600000
    g = df.groupby("g").agg(o=("o", "first"), h=("h", "max"),
                            l=("l", "min"), c=("c", "last"))
    g["t"] = g.index * 3600000
    return g.reset_index(drop=True)

def send(text):
    for k in range(0, len(text), 3500):
        requests.post(f"https://api.telegram.org/bot{TOKEN}/sendMessage",
                      data={"chat_id": CHAT, "text": text[k:k + 3500]}, timeout=20)

def fmt(r):
    m = len(r)
    if m < 2:
        return f"n={m}"
    p = np.array([x[1] for x in r])
    sd = p.std(ddof=1)
    t = p.mean() / (sd / np.sqrt(m)) if sd > 0 else 0
    return (f"n={m} kazanç %{(p > 0).mean() * 100:.0f} "
            f"EV{p.mean() / 100 * NOTIONAL:+.1f}$ t={t:.1f}")

def streak(r):
    best = cur = 0
    for x in r:
        cur = cur + 1 if x[1] < 0 else 0
        best = max(best, cur)
    return best

def mean_of(r):
    return np.mean([x[1] for x in r]) if r else -9

def onset(s):
    return s & ~np.r_[False, s[:-1]]

rows, ctrl = [], []
for sym in SYMBOLS:
    parts = [d for d in (load1h(sym, m) for m in months()) if d is not None]
    if not parts:
        continue
    b = pd.concat(parts).sort_values("t").drop_duplicates("t").reset_index(drop=True)
    o, h, l, c = (b[k].values for k in ("o", "h", "l", "c"))
    n = len(b)
    split = int(n * 2 / 3)
    close, high, low = pd.Series(c), pd.Series(h), pd.Series(l)
    ema50 = close.ewm(span=50, adjust=False).mean()
    ema200 = close.ewm(span=200, adjust=False).mean()
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    rsi = (100 - 100 / (1 + gain / loss.replace(0, np.nan))).fillna(50).values
    up, dn = (ema50 > ema200).values, (ema50 < ema200).values
    cl = (close > high.shift(1)).values
    cs = (close < low.shift(1)).values
    ones = np.ones(n, bool)

    setups = []
    for r in (35, 40, 45):
        for conf in (0, 1):
            tag = "+onay" if conf else ""
            setups.append((f"Trend geri çekilme RSI{r}{tag}",
                           up & (rsi <= r) & (cl if conf else ones),
                           dn & (rsi >= 100 - r) & (cs if conf else ones)))
    for r in (25, 30):
        for conf in (0, 1):
            tag = "+onay" if conf else ""
            setups.append((f"Aşırı-dönüş RSI{r}{tag}",
                           (rsi <= r) & (cl if conf else ones),
                           (rsi >= 100 - r) & (cs if conf else ones)))
    for N in (24, 48):
        hh = high.shift(1).rolling(N).max().values
        ll = low.shift(1).rolling(N).min().values
        setups.append((f"Kırılım {N}s", up & (c > hh), dn & (c < ll)))

    def sim(sig, sg):
        res, busy = [], -1
        for i in sig:
            if i <= busy or i + 1 + HOLD >= n:
                continue
            e = o[i + 1]
            lo, hi = l[i + 1:i + 1 + HOLD], h[i + 1:i + 1 + HOLD]
            if sg == 1:
                stop, tgt = e * (1 - STOP / 100), e * (1 + TGT / 100)
                s, t_ = np.flatnonzero(lo <= stop), np.flatnonzero(hi >= tgt)
            else:
                stop, tgt = e * (1 + STOP / 100), e * (1 - TGT / 100)
                s, t_ = np.flatnonzero(hi >= stop), np.flatnonzero(lo <= tgt)
            js = s[0] if len(s) else HOLD
            jt = t_[0] if len(t_) else HOLD
            if js <= jt and js < HOLD:
                pct, k = -STOP, js
            elif jt < js:
                pct, k = TGT, jt
            else:
                pct, k = sg * (c[i + HOLD] / e - 1) * 100, HOLD - 1
            res.append((i, pct - FEE_PCT))
            busy = i + 1 + k
        return res

    def add(name, sig, sg, bucket):
        res = sim(sig, sg)
        a = [x for x in res if x[0] < split]
        t_ = [x for x in res if x[0] >= split]
        ok = len(a) >= 30 and len(t_) >= 30 and mean_of(a) > 0 and mean_of(t_) > 0
        bucket.append((f"{sym[:3]} {'LONG' if sg == 1 else 'SHORT'} | {name}", a, t_, ok))

    for name, L, S in setups:
        add(name, np.flatnonzero(onset(L)), 1, rows)
        add(name, np.flatnonzero(onset(S)), -1, rows)
    add("KONTROL (rastgele giriş)", np.arange(0, n, 24), 1, ctrl)
    add("KONTROL (rastgele giriş)", np.arange(0, n, 24), -1, ctrl)

def block(x):
    name, a, b, ok = x
    return (f"{'✅ ' if ok else ''}{name}\n  EĞİTİM {fmt(a)}\n"
            f"  TEST   {fmt(b)} | en uzun zarar serisi {streak(b)}")

good = [x for x in rows if x[3]]
rest = sorted([x for x in rows if not x[3]], key=lambda x: -mean_of(x[1]))[:10]
out = [f"Hedef {TGT:.2f}% (20$), stop {STOP:.2f}% (40$), {LEV}x, 1 saatlik mum, 2022-2026",
       f"Rastgele girişte hedefe ulaşma ≈ %{STOP / (STOP + TGT) * 100:.0f}, "
       f"başa baş ≈ %{(STOP + FEE_PCT) / (TGT + STOP) * 100:.0f}",
       f"Denenen: {len(rows)} | eğitim ve testte pozitif: {len(good)}", ""]
out += ["--- GEÇENLER ---"] + ([block(x) for x in good] or ["Geçen yok"])
out += ["", "--- KONTROL ---"] + [block(x) for x in ctrl]
out += ["", "--- EĞİTİMDE EN İYİ GEÇMEYENLER ---"] + [block(x) for x in rest]
text = "\n".join(out)
print(text)
send(text)
