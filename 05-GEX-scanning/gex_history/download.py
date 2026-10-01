"""Pull and cache SPY option EOD quotes, open interest and vendor Greeks from Theta Data.

One parquet per endpoint per trading day under gex_history/cache/:
    spy_eod_YYYY-MM-DD.parquet     option_history_eod          quotes at the 17:15 ET EOD report of that day
    spy_oi_YYYY-MM-DD.parquet      option_history_open_interest published about 06:30 ET that day = OI as of the prior close
    spy_greeks_YYYY-MM-DD.parquet  option_history_greeks_eod   vendor Greeks at the EOD report
plus spy_stock_eod.parquet (SPY daily bars, with a lead-in before the window) and sofr.parquet
(SOFR in percent, one row per calendar day) over the whole window.

Every option pull uses expiration="*" and max_dte=120, one day per request (the server requires
day-by-day requests for expiration="*"). Files that exist are skipped, writes are atomic, so the
script is safe to rerun and resumes where it stopped. Holidays raise NoDataFoundError and are
logged, not retried. Transient gRPC failures are retried three times.

With --intraday, SPY 1-minute bars are pulled per calendar month into spy_1m_YYYY-MM.parquet (step 7).

Theta Data allows one session per account: a second ThetaClient anywhere invalidates this one
(UNAUTHENTICATED "Invalid session ID"). The script re-authenticates and retries when that happens,
but do not run probe.py or a notebook cell that builds a ThetaClient while this is running.

Download time is not part of any number in the notebook.

    uv run python gex_history/download.py [--start 2023-09-15] [--end 2026-09-14] [--workers 8] [--intraday]
"""
import argparse
import datetime as dt
import json
import os
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import grpc
import polars as pl
from dotenv import load_dotenv

HERE = Path(__file__).resolve().parent
load_dotenv(HERE.parent / ".env", override=True)  # the file wins over a stale shell variable
from thetadata import ThetaClient  # noqa: E402
from thetadata.errors import NoDataFoundError  # noqa: E402

CACHE = HERE / "cache"
RESULTS = HERE / "results"
SYMBOL = "SPY"
MAX_DTE = 120
STOCK_LEAD_IN = dt.date(2023, 6, 1)   # SPY bars start here so prev_close and pre-episode windows exist
RETRY_CODES = {grpc.StatusCode.UNAVAILABLE, grpc.StatusCode.DEADLINE_EXCEEDED, grpc.StatusCode.INTERNAL,
               grpc.StatusCode.UNKNOWN, grpc.StatusCode.RESOURCE_EXHAUSTED}

client: ThetaClient = None  # set in main()
_auth_lock = threading.Lock()
_auth_gen = 0


def reauth(gen_seen: int):
    """Open a new Theta session once per invalidation (many threads may see the same failure)."""
    global client, _auth_gen
    with _auth_lock:
        if _auth_gen == gen_seen:
            client = ThetaClient(api_key=os.environ["THETADATA_API_KEY"])
            _auth_gen += 1
            print("re-authenticated: new Theta session", flush=True)


def month_chunks(start: dt.date, end: dt.date):
    a = dt.date(start.year, start.month, 1)
    while a <= end:
        nxt = dt.date(a.year + (a.month == 12), a.month % 12 + 1, 1)
        yield max(a, start), min(nxt - dt.timedelta(days=1), end)
        a = nxt


def weekdays(start: dt.date, end: dt.date):
    d = start
    while d <= end:
        if d.weekday() < 5:
            yield d
        d += dt.timedelta(days=1)


def year_chunks(start: dt.date, end: dt.date):
    a = start
    while a <= end:
        b = min(dt.date(a.year, 12, 31), end)
        yield a, b
        a = b + dt.timedelta(days=1)


def fetch(kind: str, day: dt.date) -> pl.DataFrame:
    if kind == "eod":
        return client.option_history_eod(start_date=day, end_date=day, symbol=SYMBOL, expiration="*", max_dte=MAX_DTE)
    if kind == "oi":
        return client.option_history_open_interest(symbol=SYMBOL, expiration="*", start_date=day, end_date=day, max_dte=MAX_DTE)
    if kind == "greeks":
        return client.option_history_greeks_eod(symbol=SYMBOL, expiration="*", start_date=day, end_date=day, max_dte=MAX_DTE)
    raise ValueError(kind)


def write_atomic(df: pl.DataFrame, path: Path):
    tmp = path.with_suffix(".parquet.tmp")
    df.write_parquet(tmp)
    os.replace(tmp, path)


def pull_one(task, retries=4):
    day, kind = task
    path = CACHE / f"spy_{kind}_{day.isoformat()}.parquet"
    rec = {"day": day.isoformat(), "kind": kind, "status": None, "rows": None, "secs": 0.0, "error": None}
    if path.exists():
        rec["status"] = "cached"
        return rec
    t0 = time.perf_counter()
    for attempt in range(retries):
        gen = _auth_gen
        try:
            df = fetch(kind, day)
            rec["secs"] = round(time.perf_counter() - t0, 2)
            if df.height == 0:
                rec["status"] = "empty"
                return rec
            write_atomic(df, path)
            rec.update(status="ok", rows=df.height)
            return rec
        except NoDataFoundError:
            rec.update(status="no_data", secs=round(time.perf_counter() - t0, 2))
            return rec
        except grpc.RpcError as e:
            code = e.code()
            if code == grpc.StatusCode.UNAUTHENTICATED and attempt < retries - 1:
                reauth(gen)
                time.sleep(1.0)
                continue
            if code in RETRY_CODES and attempt < retries - 1:
                time.sleep(2.0 * (attempt + 1))
                continue
            rec.update(status="error", secs=round(time.perf_counter() - t0, 2), error=f"{type(e).__name__} {code} {str(e.details())[:120]}")
            return rec
        except Exception as e:  # noqa: BLE001
            rec.update(status="error", secs=round(time.perf_counter() - t0, 2), error=f"{type(e).__name__}: {str(e)[:120]}")
            return rec
    return rec


def pull_whole_window(start: dt.date, end: dt.date):
    out = {}
    p = CACHE / "spy_stock_eod.parquet"
    if not p.exists():
        frames = [client.stock_history_eod(symbol=SYMBOL, start_date=a, end_date=b) for a, b in year_chunks(STOCK_LEAD_IN, end)]
        df = pl.concat(frames, how="vertical_relaxed").unique(subset=["created"]).sort("created")
        write_atomic(df, p)
        out["spy_stock_eod_rows"] = df.height
        print(f"stock EOD: {df.height} rows {df['created'].min()} .. {df['created'].max()} -> {p.name}", flush=True)
    else:
        out["spy_stock_eod_rows"] = pl.read_parquet(p).height
        print("stock EOD: cached", flush=True)
    p = CACHE / "sofr.parquet"
    if not p.exists():
        frames, problems = [], {}
        for a, b in year_chunks(start, end):
            try:
                frames.append(client.interest_rate_history_eod(symbol="SOFR", start_date=a, end_date=b))
            except Exception as e:  # noqa: BLE001  (this account's rate history starts 2024-01-01)
                problems[f"{a}..{b}"] = f"{type(e).__name__} {getattr(e, 'code', lambda: '')()} {str(getattr(e, 'details', lambda: '')())[:140]}"
                print(f"SOFR {a}..{b}: SKIP {problems[f'{a}..{b}']}", flush=True)
        out["sofr_problems"] = problems
        if frames:
            df = pl.concat(frames, how="vertical_relaxed").unique(subset=["created"]).sort("created")
            write_atomic(df, p)
            out["sofr_rows"] = df.height
            out["sofr_first_date"] = str(df["created"].min())
            print(f"SOFR: {df.height} rows {df['created'].min()} .. {df['created'].max()} -> {p.name}", flush=True)
    else:
        df = pl.read_parquet(p)
        out["sofr_rows"] = df.height
        out["sofr_first_date"] = str(df["created"].min())
        print("SOFR: cached", flush=True)
    return out


def pull_month(span, retries=4):
    a, b = span
    path = CACHE / f"spy_1m_{a:%Y-%m}.parquet"
    rec = {"month": f"{a:%Y-%m}", "status": None, "rows": None, "secs": 0.0, "error": None}
    if path.exists():
        rec["status"] = "cached"
        return rec
    t0 = time.perf_counter()
    for attempt in range(retries):
        gen = _auth_gen
        try:
            df = client.stock_history_ohlc(symbol=SYMBOL, interval="1m", start_date=a, end_date=b)
            rec["secs"] = round(time.perf_counter() - t0, 1)
            if df.height == 0:
                rec["status"] = "empty"
                return rec
            write_atomic(df, path)
            rec.update(status="ok", rows=df.height)
            return rec
        except NoDataFoundError:
            rec.update(status="no_data", secs=round(time.perf_counter() - t0, 1))
            return rec
        except grpc.RpcError as e:
            code = e.code()
            if code == grpc.StatusCode.UNAUTHENTICATED and attempt < retries - 1:
                reauth(gen)
                time.sleep(1.0)
                continue
            if code in RETRY_CODES and attempt < retries - 1:
                time.sleep(2.0 * (attempt + 1))
                continue
            rec.update(status="error", secs=round(time.perf_counter() - t0, 1), error=f"{type(e).__name__} {code} {str(e.details())[:120]}")
            return rec
        except Exception as e:  # noqa: BLE001
            rec.update(status="error", secs=round(time.perf_counter() - t0, 1), error=f"{type(e).__name__}: {str(e)[:120]}")
            return rec
    return rec


def pull_intraday(start: dt.date, end: dt.date, workers: int = 4):
    t0 = time.perf_counter()
    spans = list(month_chunks(start, end))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        recs = list(ex.map(pull_month, spans))
    for r in recs:
        if r["status"] != "cached":
            print(f"1m {r['month']} {r['status']:7s} rows={r['rows']} {r['secs']}s {r['error'] or ''}", flush=True)
    rows = sum(pl.scan_parquet(CACHE / f"spy_1m_{r['month']}.parquet").select(pl.len()).collect().item()
               for r in recs if r["status"] in ("ok", "cached"))
    out = {"intraday_months": len(spans), "intraday_months_ok": sum(r["status"] in ("ok", "cached") for r in recs),
           "intraday_rows": rows, "intraday_wall_seconds_this_run": round(time.perf_counter() - t0, 1),
           "intraday_problems": {r["month"]: (r["error"] or r["status"]) for r in recs if r["status"] not in ("ok", "cached")}}
    print(f"intraday: {out['intraday_months_ok']}/{len(spans)} months, {rows:,} 1-minute rows, {out['intraday_wall_seconds_this_run']}s", flush=True)
    return out


def main():
    global client
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=dt.date.fromisoformat, default=dt.date(2023, 9, 15))
    ap.add_argument("--end", type=dt.date.fromisoformat, default=dt.date(2026, 9, 14))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--intraday", action="store_true", help="also pull SPY 1-minute bars per month (step 7)")
    args = ap.parse_args()
    CACHE.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    client = ThetaClient(api_key=os.environ["THETADATA_API_KEY"])
    subs = {a: getattr(client, a, None) for a in ("stock_subscription", "options_subscription", "index_subscription")}
    print(f"window {args.start} .. {args.end}, {args.workers} workers, subscription fields {subs}", flush=True)

    days = list(weekdays(args.start, args.end))
    tasks = [(d, k) for d in days for k in ("eod", "oi", "greeks")]
    t_all = time.perf_counter()
    recs = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for i, rec in enumerate(ex.map(pull_one, tasks), 1):
            recs.append(rec)
            if rec["status"] != "cached":
                print(f"{rec['day']} {rec['kind']:6s} {rec['status']:7s} rows={rec['rows']} {rec['secs']}s {rec['error'] or ''}", flush=True)
            if i % 300 == 0:
                print(f"... {i}/{len(tasks)} tasks, {time.perf_counter() - t_all:.0f}s elapsed", flush=True)
    extra = pull_whole_window(args.start, args.end)
    wall = time.perf_counter() - t_all
    if args.intraday:
        extra.update(pull_intraday(args.start, args.end))

    # ---- summary
    by_day = {}
    for r in recs:
        by_day.setdefault(r["day"], {})[r["kind"]] = r
    complete, holidays, partial = [], [], {}
    for d, kinds in by_day.items():
        st = {k: kinds[k]["status"] for k in ("eod", "oi", "greeks")}
        if all(s in ("ok", "cached") for s in st.values()):
            complete.append(d)
        elif all(s == "no_data" for s in st.values()):
            holidays.append(d)
        else:
            partial[d] = {k: (st[k] if st[k] != "error" else kinds[k]["error"]) for k in st}
    rows_by_kind = {}
    for k in ("eod", "oi", "greeks"):
        rows = [pl.scan_parquet(CACHE / f"spy_{k}_{d}.parquet").select(pl.len()).collect().item() for d in complete]
        rows_by_kind[k] = {"min": min(rows), "median": int(statistics.median(rows)), "max": max(rows), "total": sum(rows)} if rows else None
    cache_bytes = sum(p.stat().st_size for p in CACHE.glob("*.parquet"))
    newly = sum(1 for r in recs if r["status"] == "ok")
    summary = {
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "window": {"start": args.start.isoformat(), "end": args.end.isoformat()},
        "symbol": SYMBOL, "max_dte": MAX_DTE, "workers": args.workers,
        "weekdays_in_window": len(days),
        "trading_days_complete": len(complete),
        "first_day": complete[0] if complete else None, "last_day": complete[-1] if complete else None,
        "holidays_no_data": holidays,
        "partial_days": partial,
        "files_downloaded_this_run": newly,
        "wall_seconds_this_run": round(wall, 1),
        "rows_per_day": rows_by_kind,
        **extra,
        "cache_bytes": cache_bytes,
        "cache_mb": round(cache_bytes / 1e6, 1),
    }
    (RESULTS / "download_log.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k not in ("holidays_no_data", "partial_days")}, indent=2))
    print(f"holidays (no data on all three endpoints): {len(holidays)} -> {holidays}")
    print(f"partial days: {len(partial)} -> {json.dumps(partial, indent=1) if partial else '{}'}")
    print(f"wrote {RESULTS / 'download_log.json'}", flush=True)


if __name__ == "__main__":
    main()
