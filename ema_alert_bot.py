"""
Bot de alertas swing (Telegram). Solo avisa, no opera.
  Diario : precio entrando en zona de Order Block grande
  Horario: divergencia grande del RSI
  5 min  : cruce de medias (EMA rápida/lenta)

Modos:
  python ema_alert_bot.py          -> loop continuo (PC/VPS/Raspberry), las 3 alertas
  python ema_alert_bot.py --once   -> una pasada, solo diario + horario (GitHub Actions)
"""
import os, sys, json, time
import numpy as np
import pandas as pd
import requests

# ---------------- ACTIVOS ----------------
WATCHLIST = [
    # ("crypto", símbolo en Kraken)  /  ("yf", ticker de Yahoo Finance)
    ("crypto", "BTC/USD"),
    ("crypto", "ETH/USD"),
    ("yf", "TSLA"),
    ("yf", "NVDA"),
    ("yf", "COIN"),
    ("yf", "MSTR"),
    ("yf", "EURUSD=X"),
    ("yf", "USDJPY=X"),
]

# ---------------- PARÁMETROS ----------------
# Order block diario
OB_LOOKBACK = 120        # velas diarias hacia atrás
OB_IMPULSE_ATR = 3.0     # "grande": el impulso posterior mueve >= 3 ATR
OB_IMPULSE_CANDLES = 5   # ...dentro de 5 velas
# Divergencia RSI horaria
RSI_LEN = 14
PIVOT_N = 5              # un pivote se confirma con 5 velas a cada lado (5 h de demora)
DIV_LOOKBACK = 120
DIV_MIN_GAP = 12         # velas mínimas entre los dos pivotes
DIV_MIN_RSI_DIFF = 5     # diferencia mínima de RSI entre pivotes
DIV_MIN_PRICE_ATR = 0.5  # el precio debe superar el pivote anterior por >= 0.5 ATR
# Cruce 5 min
FAST, SLOW = 5, 10
GATE_5M = True           # True: solo avisa el cruce si coincide con OB diario o divergencia 1H
GATE_DIV_HOURS = 24      # cuánto tiempo vale una divergencia como contexto

STATE_FILE = "alert_state.json"
TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
# --------------------------------------------

NAME = {"bull": "ALCISTA", "bear": "BAJISTA"}
DELTA = {"5m": pd.Timedelta(minutes=5), "1h": pd.Timedelta(hours=1), "1d": pd.Timedelta(days=1)}
YF_PERIOD = {"5m": "5d", "1h": "60d", "1d": "1y"}
OB_CACHE = {}
_kraken = None


def candles(kind, sym, tf):
    global _kraken
    if kind == "crypto":
        if _kraken is None:
            import ccxt
            _kraken = ccxt.kraken()
        raw = _kraken.fetch_ohlcv(sym, tf, limit=720)
        df = pd.DataFrame(raw, columns=["t", "open", "high", "low", "close", "vol"])
        df.index = pd.to_datetime(df["t"], unit="ms", utc=True)
    else:
        import yfinance as yf
        df = yf.download(sym, period=YF_PERIOD[tf], interval=tf, progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [c.lower() for c in df.columns]
        df = df.dropna()
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
    now = pd.Timestamp.now(tz="UTC")
    return df[df.index + DELTA[tf] <= now]  # solo velas cerradas


def atr(df, n=14):
    c = df["close"]
    tr = pd.concat([df.high - df.low, (df.high - c.shift()).abs(),
                    (df.low - c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()


def rsi(c, n=RSI_LEN):
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn)


def pivots(x, n, kind):
    out = []
    for i in range(n, len(x) - n):
        w = x[i - n:i + n + 1]
        if (kind == "low" and x[i] == w.min()) or (kind == "high" and x[i] == w.max()):
            if not out or i - out[-1] > n:  # evita pivotes duplicados por empates
                out.append(i)
    return out


# ---------- DIARIO: order blocks ----------
def find_obs(df):
    """OB alcista: última vela bajista antes de un impulso alcista grande (espejo para bajista).
    Vigente mientras ningún cierre posterior la atraviese."""
    d = df.iloc[-OB_LOOKBACK:]
    a = atr(d).values
    o, h, l, c = (d[k].values for k in ("open", "high", "low", "close"))
    n = len(d)
    obs = []
    for i in range(14, n - 2):
        if np.isnan(a[i]):
            continue
        end = min(i + 1 + OB_IMPULSE_CANDLES, n)
        if c[i] < o[i] and c[i + 1] > o[i + 1]:
            if h[i + 1:end].max() - c[i] >= OB_IMPULSE_ATR * a[i] and not (c[i + 1:] < l[i]).any():
                obs.append(("bull", l[i], h[i], d.index[i]))
        elif c[i] > o[i] and c[i + 1] < o[i + 1]:
            if c[i] - l[i + 1:end].min() >= OB_IMPULSE_ATR * a[i] and not (c[i + 1:] > h[i]).any():
                obs.append(("bear", l[i], h[i], d.index[i]))
    return obs


def check_ob(sym, px, obs, st, ctx):
    prev = st["ob"].get(sym, {})
    new = {}
    ctx["ob"] = None
    for dirn, lo, hi, ts in obs:
        oid = f"{dirn}|{ts:%Y-%m-%d}"
        inside = bool(lo <= px <= hi)
        new[oid] = inside
        if inside:
            ctx["ob"] = dirn
            if not prev.get(oid, False):  # solo al ENTRAR a la zona
                send(f"{sym} · DIARIO\nPrecio dentro de OB {NAME[dirn]} grande\n"
                     f"Zona: {lo:.5g} – {hi:.5g}\nFormado: {ts:%Y-%m-%d}\nPrecio: {px:.5g}")
    st["ob"][sym] = new


# ---------- HORARIO: divergencia RSI ----------
def find_divergence(df):
    r = rsi(df.close).iloc[-DIV_LOOKBACK:].values
    d = df.iloc[-DIV_LOOKBACK:]
    a = atr(df).iloc[-1]
    n = len(d)
    if n < 40 or np.isnan(a):
        return None
    recent = lambda i: i >= n - 1 - PIVOT_N - 2  # pivote recién confirmado
    lo, hi = d.low.values, d.high.values
    pl = pivots(lo, PIVOT_N, "low")
    if len(pl) >= 2:
        i1, i2 = pl[-2], pl[-1]
        if (recent(i2) and i2 - i1 >= DIV_MIN_GAP and lo[i1] - lo[i2] >= DIV_MIN_PRICE_ATR * a
                and r[i2] - r[i1] >= DIV_MIN_RSI_DIFF and r[i1] < 40):
            return "bull", d.index[i2], r[i1], r[i2], lo[i1], lo[i2]
    ph = pivots(hi, PIVOT_N, "high")
    if len(ph) >= 2:
        i1, i2 = ph[-2], ph[-1]
        if (recent(i2) and i2 - i1 >= DIV_MIN_GAP and hi[i2] - hi[i1] >= DIV_MIN_PRICE_ATR * a
                and r[i1] - r[i2] >= DIV_MIN_RSI_DIFF and r[i1] > 60):
            return "bear", d.index[i2], r[i1], r[i2], hi[i1], hi[i2]
    return None


def check_div(sym, h1, st, ctx):
    res = find_divergence(h1)
    if not res:
        return
    dirn, ts, r1, r2, p1, p2 = res
    ctx["div"] = {"dir": dirn, "seen": pd.Timestamp.now(tz="UTC").isoformat()}
    key = f"{dirn}|{ts}"
    if st["last"].get(f"{sym}|1h") == key:
        return
    st["last"][f"{sym}|1h"] = key
    send(f"{sym} · HORARIO\nDivergencia {NAME[dirn]} grande en RSI\n"
         f"Precio: {p1:.5g} → {p2:.5g}\nRSI: {r1:.1f} → {r2:.1f}\nPivote: {ts:%Y-%m-%d %H:%M} UTC")


# ---------- 5 MIN: cruce de medias ----------
def check_cross(sym, d5, st, ctx):
    c = d5.close
    gap = c.ewm(span=FAST, adjust=False).mean() - c.ewm(span=SLOW, adjust=False).mean()
    g0, g1 = gap.iloc[-2], gap.iloc[-1]
    if g0 <= 0 < g1:
        dirn = "bull"
    elif g0 >= 0 > g1:
        dirn = "bear"
    else:
        return
    ts = d5.index[-1]
    key = f"{dirn}|{ts}"
    if st["last"].get(f"{sym}|5m") == key:
        return
    st["last"][f"{sym}|5m"] = key
    why = []
    if ctx.get("ob") == dirn:
        why.append("precio en OB diario " + NAME[dirn].lower())
    div = ctx.get("div")
    if div and div["dir"] == dirn and \
            pd.Timestamp.now(tz="UTC") - pd.Timestamp(div["seen"]) <= pd.Timedelta(hours=GATE_DIV_HOURS):
        why.append("divergencia RSI 1H " + NAME[dirn].lower())
    if GATE_5M and not why:
        return
    send(f"{sym} · 5 MIN\nCruce {NAME[dirn]} EMA{FAST}/EMA{SLOW}\nPrecio: {c.iloc[-1]:.5g}\n"
         f"Contexto: {', '.join(why) if why else 'sin confluencia'}\nVela: {ts:%Y-%m-%d %H:%M} UTC")


def send(msg):
    if not TOKEN or not CHAT_ID:
        print("[sin Telegram configurado]\n" + msg + "\n")
        return
    requests.post(f"https://api.telegram.org/bot{TOKEN}/sendMessage",
                  json={"chat_id": CHAT_ID, "text": msg}, timeout=15)


def run_symbol(kind, sym, st, do_5m, do_hourly):
    ctx = st["ctx"].setdefault(sym, {})
    d5 = h1 = None
    px = None
    if do_5m:
        d5 = candles(kind, sym, "5m")
        px = float(d5.close.iloc[-1])
    if do_hourly:
        h1 = candles(kind, sym, "1h")
        if px is None:
            px = float(h1.close.iloc[-1])
        check_div(sym, h1, st, ctx)
    if do_hourly or sym not in OB_CACHE:
        OB_CACHE[sym] = find_obs(candles(kind, sym, "1d"))
    check_ob(sym, px, OB_CACHE[sym], st, ctx)
    if do_5m:
        check_cross(sym, d5, st, ctx)


def load_state():
    try:
        st = json.load(open(STATE_FILE))
    except Exception:
        st = {}
    for k in ("last", "ob", "ctx"):
        st.setdefault(k, {})
    return st


def main():
    once = "--once" in sys.argv
    st = load_state()
    if once:
        for kind, sym in WATCHLIST:
            try:
                run_symbol(kind, sym, st, do_5m=False, do_hourly=True)
            except Exception as e:
                print(f"Error con {sym}: {e}")
        json.dump(st, open(STATE_FILE, "w"))
        return
    first = True
    while True:
        now = pd.Timestamp.now(tz="UTC")
        hourly = first or now.minute < 5
        first = False
        for kind, sym in WATCHLIST:
            try:
                run_symbol(kind, sym, st, do_5m=True, do_hourly=hourly)
            except Exception as e:
                print(f"Error con {sym}: {e}")
        json.dump(st, open(STATE_FILE, "w"))
        nxt = now.floor("5min") + pd.Timedelta(minutes=5, seconds=20)
        time.sleep(max(5, (nxt - pd.Timestamp.now(tz="UTC")).total_seconds()))


if __name__ == "__main__":
    main()
