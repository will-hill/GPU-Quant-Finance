"""Step 1 probe: Theta Data history endpoints for the SPY GEX study.

    uv run python gex_history/probe.py

Prints the account's subscription fields, schemas, row counts, the OI date semantics test,
history depth by year, per-call wall times and an 8-worker concurrency sample. Read-only:
nothing is cached here (download.py does that).
"""
import datetime as dt
import os
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import polars as pl
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=True)  # the file wins over a stale shell variable
from thetadata import ThetaClient  # noqa: E402

pl.Config.set_tbl_cols(40)
pl.Config.set_tbl_width_chars(240)
pl.Config.set_tbl_rows(12)

client = ThetaClient(api_key=os.environ["THETADATA_API_KEY"])
print("subscription fields reported by the client (0 = FREE tier):",
      {a: getattr(client, a, None) for a in ("stock_subscription", "options_subscription", "index_subscription")})
TIMES = {}


def err_class(e):
    code = det = ""
    if hasattr(e, "code"):
        try:
            code = f" code={e.code()}"
        except Exception:
            pass
    if hasattr(e, "details"):
        try:
            det = f" details={str(e.details())[:200]}"
        except Exception:
            pass
    return f"{type(e).__name__}{code}{det} :: {str(e)[:200]}"


def call(label, fn, **kw):
    t0 = time.perf_counter()
    try:
        df = fn(**kw)
        TIMES[label] = t = time.perf_counter() - t0
        print(f"[{label}] OK  {df.height:,} rows x {df.width} cols  {t:.2f}s")
        return df
    except Exception as e:
        TIMES[label] = t = time.perf_counter() - t0
        print(f"[{label}] FAIL after {t:.2f}s: {err_class(e)}")
        return None


def datecols(df):
    return [c for c, d in df.schema.items() if isinstance(d, (pl.Date, pl.Datetime)) or d in (pl.Date,)]


def as_date_expr(df):
    """Expression giving a pl.Date for the row's report date, whatever the column type."""
    for c in datecols(df):
        return pl.col(c).dt.date().alias("d")
    for c in df.columns:
        if any(k in c.lower() for k in ("date", "created", "timestamp", "time")):
            return pl.col(c).cast(pl.Utf8).str.slice(0, 10).str.strptime(pl.Date, "%Y-%m-%d", strict=False).alias("d")
    raise KeyError(f"no date-like column in {df.columns}")


def describe(label, df, n=3):
    if df is None:
        return
    print(f"--- {label}: schema ---")
    for c, d in df.schema.items():
        print(f"    {c:24s} {d}")
    print(f"--- {label}: head({n}) ---")
    print(df.head(n))
    for c in datecols(df):
        u = df[c].unique().sort()
        print(f"--- {label}: {c} unique ({u.len()}): {u.head(3).to_list()} ... {u.tail(2).to_list()}")
    for c in ("expiration", "strike", "right", "date"):
        if c in df.columns:
            u = df[c].unique().sort()
            print(f"--- {label}: {c} n_unique={u.len()} min={u.min()} max={u.max()}")


D = dt.date(2026, 9, 10)
print(f"\n=== probe date D = {D} ({D.strftime('%A')}) ===")

# ---- 1. EOD quotes, all expirations, one day
eod = call("1.eod", client.option_history_eod, start_date=D, end_date=D, symbol="SPY", expiration="*", max_dte=120)
if eod is None:
    print("expiration='*' rejected on option_history_eod; falling back to option_list_expirations")
    print(call("1.list_exp", client.option_list_expirations, symbol="SPY"))
describe("1.eod", eod)

# ---- 2. open interest, all expirations, one day
oi = call("2.oi", client.option_history_open_interest, symbol="SPY", expiration="*", start_date=D, end_date=D, max_dte=120)
if oi is None:
    oi = call("2.oi(date=)", client.option_history_open_interest, symbol="SPY", expiration="*", date=D, max_dte=120)
describe("2.oi", oi)
if oi is not None:
    for c in oi.columns:
        if any(k in c.lower() for k in ("date", "created", "timestamp", "time")):
            u = oi[c].unique().sort()
            print(f"--- 2.oi: {c} distinct values ({u.len()}): {u.head(5).to_list()}")
    oicol = [c for c in oi.columns if "interest" in c.lower() or c.lower() == "oi"]
    print(f"--- 2.oi: OI column candidates {oicol}; total OI on D = {oi[oicol[0]].sum():,}" if oicol else "--- 2.oi: no OI column found")

# ---- 3. vendor greeks, all expirations, one day
gk = call("3.greeks", client.option_history_greeks_eod, symbol="SPY", expiration="*", start_date=D, end_date=D, max_dte=120)
describe("3.greeks", gk)
if gk is not None:
    cand = [c for c in gk.columns if any(k in c.lower() for k in ("gamma", "iv", "impl", "vol", "under", "spot", "delta", "vega", "theta", "rho"))]
    print("3.greeks candidate greek / iv / underlying columns:", cand)
    print(gk.select(cand).describe())

# ---- 4. stock EOD
st = call("4.stock_eod", client.stock_history_eod, symbol="SPY", start_date=dt.date(2026, 9, 1), end_date=D)
describe("4.stock_eod", st, n=10)

# ---- 5. SOFR
for sym in ("SOFR", "sofr"):
    ir = call(f"5.rate[{sym}]", client.interest_rate_history_eod, symbol=sym, start_date=dt.date(2026, 9, 1), end_date=D)
    if ir is not None:
        describe(f"5.rate[{sym}]", ir, n=10)
        break

# ---- OI date semantics: one expiration over its life, quotes vs OI
print("\n=== OI date semantics test: expiration 2026-09-11 (Friday), quotes vs OI, 2026-08-24 .. 2026-09-14 ===")
EXP = dt.date(2026, 9, 11)
q1 = call("sem.eod", client.option_history_eod, start_date=dt.date(2026, 8, 24), end_date=dt.date(2026, 9, 14), symbol="SPY", expiration=EXP)
o1 = call("sem.oi", client.option_history_open_interest, symbol="SPY", expiration=EXP, start_date=dt.date(2026, 8, 24), end_date=dt.date(2026, 9, 14))
try:
    if q1 is not None and o1 is not None:
        q1 = q1.with_columns(as_date_expr(q1))
        o1 = o1.with_columns(as_date_expr(o1))
        print("quote dates :", q1["d"].unique().sort().to_list())
        print("OI dates    :", o1["d"].unique().sort().to_list())
        oicol = [c for c in o1.columns if "interest" in c.lower() or c.lower() == "oi"][0]
        volcol = [c for c in q1.columns if "volume" in c.lower()][0]
        print(f"OI column = {oicol!r}, volume column = {volcol!r}")
        print("OI rows/date, sum OI:", o1.group_by("d").agg(pl.len(), pl.col(oicol).sum()).sort("d"))
        print("quote rows/date, sum vol:", q1.group_by("d").agg(pl.len(), pl.col(volcol).sum()).sort("d"))
        key = ["strike", "right"]
        fq = q1.filter(pl.col(volcol) > 0).group_by(key).agg(pl.col("d").min().alias("first_trade_d"))
        fo = o1.filter(pl.col(oicol) > 0).group_by(key).agg(pl.col("d").min().alias("first_oi_d"))
        j = fq.join(fo, on=key, how="inner").with_columns((pl.col("first_oi_d") - pl.col("first_trade_d")).dt.total_days().alias("lag_days"))
        print("first OI>0 date minus first traded date, per contract (days):", j["lag_days"].value_counts().sort("lag_days"))
        oo = o1.sort(key + ["d"]).with_columns(pl.col(oicol).diff().over(key).alias("dOI"), pl.col("d").shift(1).over(key).alias("d_prev"))
        vv = q1.select(key + ["d", volcol])
        m0 = oo.join(vv, on=key + ["d"], how="inner").select("dOI", pl.col(volcol).alias("vol_same_day")).drop_nulls()
        m1 = oo.join(vv.rename({"d": "d_prev"}), on=key + ["d_prev"], how="inner").select("dOI", pl.col(volcol).alias("vol_prev_day")).drop_nulls()
        c0 = np.corrcoef(np.abs(m0["dOI"].to_numpy()), m0["vol_same_day"].to_numpy())[0, 1]
        c1 = np.corrcoef(np.abs(m1["dOI"].to_numpy()), m1["vol_prev_day"].to_numpy())[0, 1]
        print(f"corr(|dOI on D|, volume on D)   = {c0:.3f}  (n={m0.height})")
        print(f"corr(|dOI on D|, volume on D-1) = {c1:.3f}  (n={m1.height})")
        # |dOI| cannot exceed the volume that produced it: count violations under each hypothesis
        v0 = (np.abs(m0["dOI"].to_numpy()) > m0["vol_same_day"].to_numpy()).mean()
        v1 = (np.abs(m1["dOI"].to_numpy()) > m1["vol_prev_day"].to_numpy()).mean()
        print(f"share of contract-days with |dOI| > volume: same-day {v0:.3f}, prev-day {v1:.3f}  (the true pairing must be near 0)")
        print("If the D-1 pairing wins, an OI row dated D is the position as of the close of D-1 (OCC morning publication).")
except Exception:
    traceback.print_exc()

# ---- 6. history depth
print("\n=== history depth ===")
for Dk in (dt.date(2025, 9, 10), dt.date(2024, 9, 11), dt.date(2023, 9, 13)):
    e = call(f"6.eod[{Dk}]", client.option_history_eod, start_date=Dk, end_date=Dk, symbol="SPY", expiration="*", max_dte=120)
    o = call(f"6.oi[{Dk}]", client.option_history_open_interest, symbol="SPY", expiration="*", start_date=Dk, end_date=Dk, max_dte=120)
    g = call(f"6.greeks[{Dk}]", client.option_history_greeks_eod, symbol="SPY", expiration="*", start_date=Dk, end_date=Dk, max_dte=120)
    for lab, f in (("eod", e), ("oi", o), ("greeks", g)):
        if f is not None:
            print(f"    {Dk} {lab}: {f.height:,} rows; exp n={f['expiration'].n_unique() if 'expiration' in f.columns else '?'}; "
                  f"date col values={[f[c].unique().sort().to_list()[:2] for c in f.columns if any(k in c.lower() for k in ('date', 'created'))]}")

# ---- 7. timing summary + 8-worker concurrency sample
print("\n=== per-call wall times (s) ===")
for k, v in TIMES.items():
    print(f"    {k:26s} {v:7.2f}")
one_day = sum(TIMES.get(k, 0.0) for k in ("1.eod", "2.oi", "3.greeks"))
print(f"one full day (eod+oi+greeks), serial = {one_day:.1f}s; x 756 = {one_day*756/60:.1f} min serial")

days8 = [dt.date(2026, 8, 3) + dt.timedelta(days=i) for i in (0, 1, 2, 3, 4, 7, 8, 9)]


def one_day_pull(Dk):
    t0 = time.perf_counter()
    n = 0
    for fn, kw in ((client.option_history_eod, dict(start_date=Dk, end_date=Dk, symbol="SPY", expiration="*", max_dte=120)),
                   (client.option_history_open_interest, dict(symbol="SPY", expiration="*", start_date=Dk, end_date=Dk, max_dte=120)),
                   (client.option_history_greeks_eod, dict(symbol="SPY", expiration="*", start_date=Dk, end_date=Dk, max_dte=120))):
        try:
            n += fn(**kw).height
        except Exception as e:
            return Dk, time.perf_counter() - t0, f"FAIL {err_class(e)[:80]}"
    return Dk, time.perf_counter() - t0, n


t0 = time.perf_counter()
with ThreadPoolExecutor(max_workers=8) as ex:
    res = list(ex.map(one_day_pull, days8))
wall = time.perf_counter() - t0
for Dk, t, n in res:
    print(f"    {Dk}: {t:6.1f}s  rows={n}")
print(f"8 days x (eod+oi+greeks) with 8 workers: wall {wall:.1f}s = {wall/8:.2f}s per day effective; x 756 days = {wall/8*756/60:.1f} min")
