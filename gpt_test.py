import io, os, zipfile, requests
import numpy as np, pandas as pd

TOKEN = os.environ["TG_TOKEN"]
CHAT = os.environ["TG_CHAT_ID"]
SYMBOLS = ["ETHUSDT", "BTCUSDT"]
FEE = 0.0012
HOLD = 288
MIN_SCORE = 78

def months():
    out, y, m = [], 2023, 11
    while (y, m) <= (2026, 10):
        out.append(f"{y}-{m:02d}")
        m += 1
        if m > 12:
            y, m = y + 1, 1
    return out

def load5(sym, m):
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
    df["g"] = t // 300000
    g = df.groupby("g").agg(o=("o", "first"), h=("h", "max"),
                            l=("l", "min"), c=("c", "last"), v=("v", "sum"))
    g["t"] = g.index * 300000
    return g.reset_index(drop=True)

def agg(b, ms):
    g = b.groupby(b["t"] // ms).agg(o=("o", "first"), h=("h", "max"),
                                    l=("l", "min"), c=("c", "last"), v=("v", "sum"))
    g["t"] = g.index * ms
    return g.reset_index(drop=True)

def enrich(d):
    c = d["c"]
    d["ema20"] = c.ewm(span=20, adjust=False).mean()
    d["ema50"] = c.ewm(span=50, adjust=False).mean()
    d["ema200"] = c.ewm(span=200, adjust=False).mean()
    delta = c.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    d["rsi"] = (100 - 100 / (1 + gain / loss.replace(0, np.nan))).fillna(50)
    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    d["hist"] = macd - macd.ewm(span=9, adjust=False).mean()
    d["hist_prev"] = d["hist"].shift(1)
    pc = c.shift(1)
    tr = pd.concat([d["h"] - d["l"], (d["h"] - pc).abs(),
                    (d["l"] - pc).abs()], axis=1).max(axis=1)
    d["atr"] = tr.ewm(alpha=1 / 14, adjust=False).mean()
    d["vsma"] = d["v"].rolling(20).mean()
    tp = (d["h"] + d["l"] + c) / 3
    d["vwap"] = (tp * d["v"]).rolling(20).sum() / d["v"].rolling(20).sum().replace(0, np.nan)
    return d

def prep(sym):
    parts = [d for d in (load5(sym, m) for m in months()) if d is not None]
    if not parts:
        return None
    b = pd.concat(parts).sort_values("t").drop_duplicates("t").reset_index(drop=True)
    d15, d1 = enrich(agg(b, 900000)), enrich(agg(b, 3600000))
    b["ct"] = b["t"] + 300000
    b["prior_h"] = b["h"].shift(1).rolling(12).max()
    b["prior_l"] = b["l"].shift(1).rolling(12).min()
    mc = ["t", "c", "ema20", "ema50", "rsi", "hist", "hist_prev", "v", "vsma", "vwap", "atr"]
    m = d15[mc].copy()
    m["ct"] = m["t"] + 900000
    m = m.drop(columns="t").add_prefix("m_").rename(columns={"m_ct": "ct"})
    hh = d1[["t", "c", "ema20", "ema50", "ema200"]].copy()
    hh["ct"] = hh["t"] + 3600000
    hh = hh.drop(columns="t").add_prefix("h_").rename(columns={"h_ct": "ct"})
    out = pd.merge_asof(b.sort_values("ct"), m.sort_values("ct"), on="ct")
    out = pd.merge_asof(out.sort_values("ct"), hh.sort_values("ct"), on="ct")
    return out.reset_index(drop=True)

def score(d):
    mc, ma = d["m_c"], d["m_atr"]
    vol_ok = d["m_v"] > 1.15 * d["m_vsma"]
    rng = (0.0015 <= 1.4 * ma / mc) & (1.4 * ma / mc <= 0.02)
    common = 15 * vol_ok + 5 * rng
    tu = (d["h_ema20"] > d["h_ema50"]) & (d["h_ema50"] > d["h_ema200"]) & (d["h_c"] > d["h_ema20"])
    td = (d["h_ema20"] < d["h_ema50"]) & (d["h_ema50"] < d["h_ema200"]) & (d["h_c"] < d["h_ema20"])
    ls = (common + 20 * tu + 20 * ((d["m_ema20"] > d["m_ema50"]) & (mc > d["m_ema20"]))
          + 10 * ((d["m_rsi"] >= 52) & (d["m_rsi"] <= 72))
          + 10 * ((d["m_hist"] > 0) & (d["m_hist"] >= d["m_hist_prev"]))
          + 10 * (mc > d["m_vwap"]) + 10 * (d["c"] > d["prior_h"]))
    ss = (common + 20 * td + 20 * ((d["m_ema20"] < d["m_ema50"]) & (mc < d["m_ema20"]))
          + 10 * ((d["m_rsi"] >= 28) & (d["m_rsi"] <= 48))
          + 10 * ((d["m_hist"] < 0) & (d["m_hist"] <= d["m_hist_prev"]))
          + 10 * (mc < d["m_vwap"]) + 10 * (d["c"] < d["prior_l"]))
    return ls.values.astype(float), ss.values.astype(float)

def simulate(d):
    o, h, l, c = (d[k].values for k in ("o", "h", "l", "c"))
    atr = d["m_atr"].values
    n = len(d)
    ls, ss = score(d)
    sc = np.maximum(ls, ss)
    side = np.where(ls > ss, 1, -1)
    cand = np.flatnonzero((sc >= MIN_SCORE) & (ls != ss))
    res, busy = [], -1
    for i in cand:
        if i <= busy or i + 1 + HOLD >= n:
            continue
        a = atr[i]
        if not a > 0:
            continue
        e, sd, sg = o[i + 1], 1.4 * a, side[i]
        stop, t1 = e - sg * sd, e + sg * 2 * sd
        lo, hi = l[i + 1:i + 1 + HOLD], h[i + 1:i + 1 + HOLD]
        if sg == 1:
            s, t_ = np.flatnonzero(lo <= stop), np.flatnonzero(hi >= t1)
        else:
            s, t_ = np.flatnonzero(hi >= stop), np.flatnonzero(lo <= t1)
        js = s[0] if len(s) else HOLD
        jt = t_[0] if len(t_) else HOLD
        if js <= jt and js < HOLD:
            r, k, win = -1.0, js, 0
        elif jt < js:
            r, k, win = 2.0, jt, 1
        else:
            r, k, win = sg * (c[i + HOLD] - e) / sd, HOLD - 1, 0
        r -= FEE * e / sd
        res.append((i, int(sg), int(sc[i]), r, win, FEE * e / sd))
        busy = i + 1 + k
    return res, n

def st(rows):
    m = len(rows)
    if m < 2:
        return f"n={m}"
    r = np.array([x[3] for x in rows])
    w = np.mean([x[4] for x in rows]) * 100
    sdv = r.std(ddof=1)
    t = r.mean() / (sdv / np.sqrt(m)) if sdv > 0 else 0
    return f"n={m} kazanç %{w:.0f} R={r.mean():+.2f} t={t:.1f}"

def send(text):
    for k in range(0, len(text), 3500):
        requests.post(f"https://api.telegram.org/bot{TOKEN}/sendMessage",
                      data={"chat_id": CHAT, "text": text[k:k + 3500]}, timeout=20)

out = ["ChatGPT skor stratejisi testi (mum kapanışlı, 36 ay, eğitim 24 / test 12 ay)",
       "Hedef 2R, stop 1.4×ATR(15dk). Başa baş kazanç oranı yaklaşık %35-37.", ""]
for sym in SYMBOLS:
    d = prep(sym)
    if d is None:
        out.append(f"{sym}: veri yok")
        continue
    res, n = simulate(d)
    split = int(n * 2 / 3)
    for sg, name in ((1, "LONG"), (-1, "SHORT")):
        for lab, lo, hi in (("hepsi", 0, 101), ("78-84", 78, 85),
                            ("85-91", 85, 92), ("92+", 92, 101)):
            rows = [x for x in res if x[1] == sg and lo <= x[2] < hi]
            a = [x for x in rows if x[0] < split]
            b = [x for x in rows if x[0] >= split]
            out.append(f"{sym} {name} skor {lab}")
            out.append(f"  EĞİTİM {st(a)}")
            out.append(f"  TEST   {st(b)}")
text = "\n".join(out)
print(text)
send(text)
