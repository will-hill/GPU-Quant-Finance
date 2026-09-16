"""Pull and cache option EOD quotes + open interest (+ optional 1-minute bars) for many symbols.

Same layout as download.py, one parquet per symbol per endpoint per trading day under gex_history/cache/:
    {sym}_eod_YYYY-MM-DD.parquet, {sym}_oi_YYYY-MM-DD.parquet, {sym}_stock_eod.parquet (or {sym}_index_eod.parquet
    for index roots), {sym}_1m_YYYY-MM.parquet.  Symbols are lower-cased in file names (SPY files stay spy_*).

Resumable, atomic writes, one Theta session (re-authenticates on the one-session UNAUTHENTICATED error).
Do not run another ThetaClient while this runs.

    uv run python gex_history/download_universe.py --start 2025-09-15 --end 2026-09-14 --symbols NVDA TSLA ... --intraday
"""
import argparse
import datetime as dt
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import grpc
import polars as pl
from dotenv import load_dotenv

HERE = Path(__file__).resolve().parent
load_dotenv(HERE.parent / ".env", override=True)
from thetadata import ThetaClient  # noqa: E402
from thetadata.errors import NoDataFoundError  # noqa: E402

CACHE = HERE / "cache"
RESULTS = HERE / "results"
MAX_DTE = 120
INDEX_ROOTS = {"SPX": "SPX", "SPXW": "SPX", "NDX": "NDX", "NDXP": "NDX", "RUT": "RUT", "RUTW": "RUT", "VIX": "VIX"}
RETRY_CODES = {grpc.StatusCode.UNAVAILABLE, grpc.StatusCode.DEADLINE_EXCEEDED, grpc.StatusCode.INTERNAL,
               grpc.StatusCode.UNKNOWN, grpc.StatusCode.RESOURCE_EXHAUSTED}

client: ThetaClient = None
_auth_lock = threading.Lock()
_auth_gen = 0


def reauth(gen_seen):
    global client, _auth_gen
    with _auth_lock:
        if _auth_gen == gen_seen:
            client = ThetaClient(api_key=os.environ["THETADATA_API_KEY"])
            _auth_gen += 1
            print("re-authenticated: new Theta session", flush=True)


def weekdays(start, end):
    d = start
    while d <= end:
        if d.weekday() < 5:
            yield d
        d += dt.timedelta(days=1)


def month_chunks(start, end):
    a = dt.date(start.year, start.month, 1)
    while a <= end:
        nxt = dt.date(a.year + (a.month == 12), a.month % 12 + 1, 1)
        yield max(a, start), min(nxt - dt.timedelta(days=1), end)
        a = nxt


def write_atomic(df, path):
    tmp = path.with_suffix(".parquet.tmp")
    df.write_parquet(tmp)
    os.replace(tmp, path)


def with_retries(label, fn, retries=4):
    """Run fn() with the download.py retry policy. Returns (status, df_or_None, error)."""
    for attempt in range(retries):
        gen = _auth_gen
        try:
            df = fn()
            return ("empty" if df.height == 0 else "ok"), df, None
        except NoDataFoundError:
            return "no_data", None, None
        except grpc.RpcError as e:
            code = e.code()
            if code == grpc.StatusCode.UNAUTHENTICATED and attempt < retries - 1:
                reauth(gen); time.sleep(1.0); continue
            if code in RETRY_CODES and attempt < retries - 1:
                time.sleep(2.0 * (attempt + 1)); continue
            return "error", None, f"{type(e).__name__} {code} {str(e.details())[:120]}"
        except Exception as e:  # noqa: BLE001
            return "error", None, f"{type(e).__name__}: {str(e)[:120]}"
    return "error", None, "retries exhausted"


def pull_one(task):
    sym, day, kind = task
    path = CACHE / f"{sym.lower()}_{kind}_{day.isoformat()}.parquet"
    rec = {"sym": sym, "day": day.isoformat(), "kind": kind, "status": None, "rows": None, "secs": 0.0, "error": None}
    if path.exists():
        rec["status"] = "cached"
        return rec
    t0 = time.perf_counter()
    if kind == "eod":
        fn = lambda: client.option_history_eod(start_date=day, end_date=day, symbol=sym, expiration="*", max_dte=MAX_DTE)  # noqa: E731
    elif kind == "oi":
        fn = lambda: client.option_history_open_interest(symbol=sym, expiration="*", start_date=day, end_date=day, max_dte=MAX_DTE)  # noqa: E731
    else:
        fn = lambda: client.option_history_greeks_eod(symbol=sym, expiration="*", start_date=day, end_date=day, max_dte=MAX_DTE)  # noqa: E731
    status, df, err = with_retries(f"{sym} {day} {kind}", fn)
    rec.update(status=status, secs=round(time.perf_counter() - t0, 2), error=err)
    if status == "ok":
        write_atomic(df, path)
        rec["rows"] = df.height
    return rec


def pull_spot(sym, start, end):
    is_index = sym.upper() in INDEX_ROOTS
    root = INDEX_ROOTS.get(sym.upper(), sym)
    path = CACHE / f"{sym.lower()}_{'index' if is_index else 'stock'}_eod.parquet"
    if path.exists():
        return {"status": "cached", "rows": pl.read_parquet(path).height}
    lead = start - dt.timedelta(days=120)
    frames = []
    a = lead
    while a <= end:
        b = min(dt.date(a.year, 12, 31), end)
        fn = (lambda a=a, b=b: client.index_history_eod(symbol=root, start_date=a, end_date=b)) if is_index else \
             (lambda a=a, b=b: client.stock_history_eod(symbol=sym, start_date=a, end_date=b))
        status, df, err = with_retries(f"{sym} spot {a}..{b}", fn)
        if status == "ok":
            frames.append(df)
        elif status == "error":
            print(f"{sym} spot {a}..{b}: {err}", flush=True)
        a = b + dt.timedelta(days=1)
    if not frames:
        return {"status": "no_data", "rows": 0}
    df = pl.concat(frames, how="vertical_relaxed")
    tcol = [c for c in ("created", "timestamp", "date") if c in df.columns][0]
    df = df.unique(subset=[tcol]).sort(tcol)
    write_atomic(df, path)
    return {"status": "ok", "rows": df.height, "cols": df.columns}


def pull_month(task):
    sym, a, b = task
    path = CACHE / f"{sym.lower()}_1m_{a:%Y-%m}.parquet"
    rec = {"sym": sym, "month": f"{a:%Y-%m}", "status": None, "rows": None, "secs": 0.0, "error": None}
    if path.exists():
        rec["status"] = "cached"
        return rec
    if sym.upper() in INDEX_ROOTS:
        root = INDEX_ROOTS[sym.upper()]
        fn = lambda: client.index_history_price(symbol=root, interval="1m", start_date=a, end_date=b)  # noqa: E731
    else:
        fn = lambda: client.stock_history_ohlc(symbol=sym, interval="1m", start_date=a, end_date=b)  # noqa: E731
    t0 = time.perf_counter()
    status, df, err = with_retries(f"{sym} 1m {a:%Y-%m}", fn)
    rec.update(status=status, secs=round(time.perf_counter() - t0, 1), error=err)
    if status == "ok":
        write_atomic(df, path)
        rec["rows"] = df.height
    return rec


def main():
    global client
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=dt.date.fromisoformat, required=True)
    ap.add_argument("--end", type=dt.date.fromisoformat, required=True)
    ap.add_argument("--symbols", nargs="+", required=True)
    ap.add_argument("--kinds", nargs="+", default=["eod", "oi"])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--intraday", action="store_true")
    ap.add_argument("--log", default="download_universe_log.json")
    args = ap.parse_args()
    CACHE.mkdir(exist_ok=True); RESULTS.mkdir(exist_ok=True)
    client = ThetaClient(api_key=os.environ["THETADATA_API_KEY"])
    subs = {a: getattr(client, a, None) for a in ("stock_subscription", "options_subscription", "index_subscription")}
    syms = [s.upper() for s in args.symbols]
    days = list(weekdays(args.start, args.end))
    print(f"{len(syms)} symbols x {len(days)} weekdays x {args.kinds}, {args.workers} workers, subscription {subs}", flush=True)
    t_all = time.perf_counter()
    # index roots: the spot for SPXW is SPX; both option roots are pulled as separate symbols
    tasks = [(s, d, k) for s in syms for d in days for k in args.kinds]
    recs = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for i, rec in enumerate(ex.map(pull_one, tasks), 1):
            recs.append(rec)
            if rec["status"] not in ("cached", "ok", "no_data"):
                print(f"{rec['sym']} {rec['day']} {rec['kind']} {rec['status']} {rec['error']}", flush=True)
            if i % 1000 == 0:
                ok = sum(r["status"] == "ok" for r in recs)
                print(f"... {i}/{len(tasks)} tasks, {ok} new files, {time.perf_counter() - t_all:.0f}s", flush=True)
    t_opt = time.perf_counter() - t_all
    spot = {s: pull_spot(s, args.start, args.end) for s in syms}
    for s, v in spot.items():
        print(f"{s} spot: {v}", flush=True)
    intraday = None
    if args.intraday:
        t0 = time.perf_counter()
        mtasks = [(s, a, b) for s in syms for a, b in month_chunks(args.start, args.end)]
        with ThreadPoolExecutor(max_workers=4) as ex:
            mrecs = list(ex.map(pull_month, mtasks))
        bad = [r for r in mrecs if r["status"] not in ("ok", "cached")]
        intraday = {"tasks": len(mtasks), "ok": len(mtasks) - len(bad), "problems": [(r["sym"], r["month"], r["error"] or r["status"]) for r in bad],
                    "wall_seconds": round(time.perf_counter() - t0, 1)}
        print(f"intraday: {intraday['ok']}/{len(mtasks)} months ok, {intraday['wall_seconds']}s", flush=True)
    # summary per symbol
    per_sym = {}
    for s in syms:
        rs = [r for r in recs if r["sym"] == s]
        by_day = {}
        for r in rs:
            by_day.setdefault(r["day"], {})[r["kind"]] = r["status"]
        complete = [d for d, ks in by_day.items() if all(ks.get(k) in ("ok", "cached") for k in args.kinds)]
        nodata = [d for d, ks in by_day.items() if all(ks.get(k) == "no_data" for k in args.kinds)]
        partial = {d: ks for d, ks in by_day.items() if d not in complete and d not in nodata}
        per_sym[s] = {"complete_days": len(complete), "first": min(complete) if complete else None, "last": max(complete) if complete else None,
                      "no_data_days": len(nodata), "partial_days": partial, "spot": spot[s]}
    summary = {"generated_at": dt.datetime.now().isoformat(timespec="seconds"), "window": [args.start.isoformat(), args.end.isoformat()],
               "symbols": syms, "kinds": args.kinds, "weekdays": len(days), "option_wall_seconds": round(t_opt, 1),
               "new_files": sum(r["status"] == "ok" for r in recs), "errors": [r for r in recs if r["status"] == "error"][:50],
               "per_symbol": per_sym, "intraday": intraday,
               "cache_mb": round(sum(p.stat().st_size for p in CACHE.glob("*.parquet")) / 1e6, 1)}
    (RESULTS / args.log).write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps({s: (v["complete_days"], v["no_data_days"], len(v["partial_days"])) for s, v in per_sym.items()}, indent=0))
    print(f"total wall {time.perf_counter() - t_all:.0f}s; wrote {RESULTS / args.log}", flush=True)


if __name__ == "__main__":
    main()
