import io, os, zipfile, requests
import numpy as np, pandas as pd

TOKEN = os.environ["TG_TOKEN"]
CHAT = os.environ["TG_CHAT_ID"]
SYMBOL = "BTCUSDT"
CAPITAL, LEV = 250, 8
NOTIONAL = CAPITAL * LEV
FEE_PCT = 0.12
LIQ_PCT = 12.0

def months():
    out, y, m = [], 2020, 1
    while (y, m) <= (2026, 10):
        out.append(f"{y}-{m:02d}")
        m += 1
        if m > 12:
            y, m = y + 1, 1
    return out

def load1h(m):
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
    df["g"] = t // 3600000
    g = df.groupby("g").agg(o=("o", "first"), h=("h", "max"),
                            l=("l", "min"), c=("c", "last"))
    g["t"] = g.index * 3600000
    return g.reset_index(drop=True)

def agg(b, ms):
    g = b.groupby(b["t"] // ms).agg(o=("o", "first"), h=("h", "max"),
                                    l=("l", "min"), c=("c", "last"))
    g["t"] = g.index * ms
    return g.reset_index(drop=True)

def send(text):
    buf = ""
    for line in text.split("\n"):
        if len(buf) + len(line) + 1 > 3500:
            requests.post(f"https://api.telegram.org/bot{TOKEN}/sendMessage",
                          data={"chat_id": CHAT, "text": buf}, timeout=20)
            buf = ""
        buf += line + "\n"
    if buf:
        requests.post(f"https://api.telegram.org/bot{TOKEN}/sendMessage",
                      data={"chat_id": CHAT, "text": buf}, timeout=20)

def rma(x, n):
    return pd.Series(x).ewm(alpha=1 / n, adjust=False).mean().values

def true_range(h, l, c):
    pc = np.r_[c[0], c[:-1]]
    return np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))

def supertrend(h, l, c, n=10, m=3.0):
    atr = rma(true_range(h, l, c), n)
    hl2 = (h + l) / 2
    up, dn = hl2 - m * atr, hl2 + m * atr
    trend = np.ones(len(c))
    for i in range(1, len(c)):
        if c[i - 1] > up[i - 1]:
            up[i] = max(up[i], up[i - 1])
        if c[i - 1] < dn[i - 1]:
            dn[i] = min(dn[i], dn[i - 1])
        t = trend[i - 1]
        if t == -1 and c[i] > dn[i - 1]:
            t = 1
        elif t == 1 and c[i] < up[i - 1]:
            t = -1
        trend[i] = t
    trend[:n + 5] = 0
    return trend

def utbot(h, l, c, a=1.0, n=10):
    nl = a * rma(true_range(h, l, c), n)
    stop, pos = np.zeros(len(c)), np.zeros(len(c))
    stop[0] = c[0]
    for i in range(1, len(c)):
        s = stop[i - 1]
        if c[i] > s and c[i - 1] > s:
            stop[i] = max(s, c[i] - nl[i])
        elif c[i] < s and c[i - 1] < s:
            stop[i] = min(s, c[i] + nl[i])
        elif c[i] > s:
            stop[i] = c[i] - nl[i]
        else:
            stop[i] = c[i] + nl[i]
        if c[i - 1] < s and c[i] > s:
            pos[i] = 1
        elif c[i - 1] > s and c[i] < s:
            pos[i] = -1
        else:
            pos[i] = pos[i - 1]
    pos[:n + 5] = 0
    return pos

def linreg_end(y, n):
    out = np.full(len(y), np.nan)
    if len(y) < n:
        return out
    w = np.lib.stride_tricks.sliding_window_view(y, n)
    x = np.arange(n) - (n - 1) / 2
    b = (w * x).sum(axis=1) / (x * x).sum()
    out[n - 1:] = w.mean(axis=1) + b * (n - 1) / 2
    return out

def squeeze(h, l, c, n=20):
    s, hs, ls = pd.Series(c), pd.Series(h), pd.Series(l)
    basis = s.rolling(n).mean()
    dev = 2.0 * s.rolling(n).std(ddof=0)
    ub, lb = basis + dev, basis - dev
    rm = pd.Series(true_range(h, l, c)).rolling(n).mean()
    uk, lk = basis + 1.5 * rm, basis - 1.5 * rm
    on = ((lb > lk) & (ub < uk)).values
    mid = ((hs.rolling(n).max() + ls.rolling(n).min()) / 2 + basis) / 2
    val = linreg_end((s - mid).values, n)
    state, pos = np.zeros(len(c)), 0
    for i in range(1, len(c)):
        if np.isnan(val[i]):
            continue
        if pos == 0:
            if on[i - 1] and not on[i]:
                pos = 1 if val[i] > 0 else -1
        elif (pos == 1 and val[i] < 0) or (pos == -1 and val[i] > 0):
            pos = 0
        state[i] = pos
    return state

def macd_state(h, l, c):
    s = pd.Series(c)
    m = s.ewm(span=12, adjust=False).mean() - s.ewm(span=26, adjust=False).mean()
    sg = m.ewm(span=9, adjust=False).mean()
    st = np.where(m > sg, 1.0, -1.0)
    st[:40] = 0
    return st

def ichimoku(h, l, c):
    H, L = pd.Series(h), pd.Series(l)
    ten = (H.rolling(9).max() + L.rolling(9).min()) / 2
    kij = (H.rolling(26).max() + L.rolling(26).min()) / 2
    A = ((ten + kij) / 2).shift(26).values
    B = ((H.rolling(52).max() + L.rolling(52).min()) / 2).shift(26).values
    top, bot = np.maximum(A, B), np.minimum(A, B)
    tv, kv = ten.values, kij.values
    longs = (c > top) & (tv > kv)
    shorts = (c < bot) & (tv < kv)
    return np.where(longs, 1.0, np.where(shorts, -1.0, 0.0))

STRATS = [("SuperTrend", supertrend), ("UT Bot", utbot), ("Squeeze Momentum", squeeze),
          ("MACD", macd_state), ("Ichimoku", ichimoku)]

def make_trades(state, o, h, l):
    tr, cur, ent = [], 0, 0

    def close(d, a, b):
        e, x = o[a], o[b]
        mae = (e - l[a:b].min()) / e * 100 if d == 1 else (h[a:b].max() - e) / e * 100
        if mae >= LIQ_PCT:
            return (d, -LIQ_PCT - 0.5, a, 1)
        return (d, d * (x / e - 1) * 100 - FEE_PCT, a, 0)

    for i in range(len(state) - 1):
        s = state[i]
        if s != cur:
            if cur != 0:
                tr.append(close(cur, ent, i + 1))
            if s != 0:
                ent = i + 1
            cur = s
    return tr

def stats(tr, split):
    if len(tr) < 5:
        return None
    p = np.array([x[1] for x in tr])
    d = np.array([x[0] for x in tr])
    ix = np.array([x[2] for x in tr])
    liq = sum(x[3] for x in tr)
    pnl = p / 100 * NOTIONAL
    cum = np.cumsum(pnl)
    dd = (np.maximum.accumulate(np.r_[0, cum])[1:] - cum).max()
    gl = -pnl[pnl < 0].sum()
    pf = pnl[pnl > 0].sum() / gl if gl > 0 else 9.99
    sd = p.std(ddof=1)
    t = p.mean() / (sd / np.sqrt(len(p))) if sd > 0 else 0
    av = lambda m: p[m].mean() if m.sum() > 0 else float("nan")
    return dict(n=len(p), w=(p > 0).mean() * 100, avg=p.mean(), pf=pf, tot=pnl.sum(),
                dd=dd, liq=liq, la=av(d == 1), sa=av(d == -1),
                h1=av(ix < split), h2=av(ix >= split), t=t)

parts = [d for d in (load1h(m) for m in months()) if d is not None]
if not parts:
    send("TV testi: veri indirilemedi")
    raise SystemExit
b1 = pd.concat(parts).sort_values("t").drop_duplicates("t").reset_index(drop=True)
drift = (b1["c"].iloc[-1] / b1["o"].iloc[0] - 1) * 100
out = [f"BTC vadeli, {NOTIONAL}$ pozisyon ({CAPITAL}$ x {LEV}), komisyon+kayma %{FEE_PCT}, "
       f"tasfiye ~%{LIQ_PCT:.0f}",
       f"Dönem BTC fiyat değişimi: %{drift:+.0f} (long tarafı bu yüzden şişebilir)",
       "✅ = ilk yarı>0, son yarı>0, t>=2, long>0 ve short>0", ""]
frames = [("1sa", b1), ("4sa", agg(b1, 14400000)), ("1g", agg(b1, 86400000))]
for tf, fr in frames:
    o, h, l, c = (fr[k].values for k in ("o", "h", "l", "c"))
    split = len(fr) // 2
    for name, fn in STRATS:
        s = stats(make_trades(fn(h, l, c), o, h, l), split)
        if s is None:
            out.append(f"{name} {tf}: yetersiz işlem")
            continue
        ok = s["h1"] > 0 and s["h2"] > 0 and s["t"] >= 2 and s["la"] > 0 and s["sa"] > 0
        out.append(f"{'✅ ' if ok else ''}{name} {tf}: n={s['n']} kazanç %{s['w']:.0f} "
                   f"ort {s['avg']:+.2f}% PF {s['pf']:.2f} | toplam {s['tot']:+.0f}$ "
                   f"DD -{s['dd']:.0f}$ tasfiye {s['liq']}")
        out.append(f"   LONG {s['la']:+.2f}% SHORT {s['sa']:+.2f}% | "
                   f"ilk yarı {s['h1']:+.2f}% son yarı {s['h2']:+.2f}% t={s['t']:.1f}")
text = "\n".join(out)
print(text)
send(text)
