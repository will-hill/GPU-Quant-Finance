"""Helpers for gex_history_spy.ipynb.

Chain prep (step 3), engine call (step 4), GEX aggregation, flip level and walls (step 4),
regimes and episodes (step 5), statistics (step 6), figures (step 8), vendor cross-check (step 9).
Everything reads the parquet cache written by download.py; nothing here touches the network.
"""
from __future__ import annotations

import datetime as dt
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

HERE = Path(__file__).resolve().parent
CACHE = HERE / "cache"
RESULTS = HERE / "results"
FIGURES = HERE / "figures"
ENGINE_DIR = HERE.parent / "engine_alo"
if str(ENGINE_DIR) not in sys.path:
    sys.path.insert(0, str(ENGINE_DIR))

SYMBOL = "SPY"
Q_FLAT = 0.012            # SPY trailing dividend yield, held flat (assumption stated in the notebook)
R_FALLBACK = 0.043        # only if SOFR is missing for a day
STRIKE_BAND = (0.80, 1.20)
DTE_MIN, DTE_MAX = 1, 120
KEY = ["expiration", "strike", "right"]
DROP_ORDER = ["dte_lt_1", "dte_gt_120", "outside_band", "oi_zero", "no_quote", "bid_le_0", "ask_lt_bid", "mid_le_intrinsic", "kept"]
DROP_LABEL = {
    "dte_lt_1": "expires on D (gone at the close)",
    "dte_gt_120": "more than 120 days to expiry",
    "outside_band": "strike outside 0.80 S to 1.20 S",
    "oi_zero": "open interest 0 (row count only, no GEX)",
    "no_quote": "open interest but no EOD quote row",
    "bid_le_0": "bid = 0",
    "ask_lt_bid": "ask below bid",
    "mid_le_intrinsic": "mid at or below intrinsic (exercise region, gamma 0)",
    "kept": "kept, sent to the engine",
}


# ----------------------------------------------------------------------------- cache access
INDEX_ROOTS = {"SPX", "SPXW", "XSP", "NDX", "NDXP", "RUT", "RUTW", "VIX", "VIXW", "DJX", "OEX", "XEO"}
# trailing dividend yields used as flat q per symbol (assumptions); anything not listed uses 0
Q_BY_SYMBOL = {"SPY": 0.012, "QQQ": 0.006, "IWM": 0.011, "DIA": 0.016, "SPX": 0.012, "SPXW": 0.012, "AAPL": 0.004, "MSFT": 0.007,
               "META": 0.004, "GOOGL": 0.004, "GOOG": 0.004, "NVDA": 0.0003, "MU": 0.004, "ORCL": 0.010, "TSM": 0.014, "IBM": 0.026,
               "GS": 0.020, "DELL": 0.012, "AVGO": 0.008, "ASML": 0.009, "WDC": 0.001, "STX": 0.020, "EWY": 0.015, "SMH": 0.004}


def q_for(symbol: str) -> float:
    return Q_BY_SYMBOL.get(symbol.upper(), 0.0)


def cached_days(kind: str = "eod", symbol: str = "spy") -> list[dt.date]:
    return sorted(dt.date.fromisoformat(p.stem.split("_")[-1]) for p in CACHE.glob(f"{symbol.lower()}_{kind}_????-??-??.parquet"))


def study_days(symbol: str = "spy") -> list[dt.date]:
    """Trading days with both a quote file and an OI file."""
    return sorted(set(cached_days("eod", symbol)) & set(cached_days("oi", symbol)))


def session_range_1m(symbol: str = "spy") -> pd.DataFrame | None:
    """Per-day high and low of the traded 1-minute bars in the regular session, if the intraday cache exists."""
    files = sorted(CACHE.glob(f"{symbol.lower()}_1m_????-??.parquet"))
    if not files:
        return None
    b = pl.concat([pl.read_parquet(f) for f in files], how="vertical_relaxed")
    if "close" not in b.columns and "price" in b.columns:          # index roots: one price per minute, no volume
        b = b.with_columns(pl.col("price").alias("high"), pl.col("price").alias("low"), pl.lit(1).alias("volume"))
    b = (b.with_columns(pl.col("timestamp").dt.date().alias("date"),
                         (pl.col("timestamp").dt.hour().cast(pl.Int32) * 60 + pl.col("timestamp").dt.minute().cast(pl.Int32)).alias("mod"))
           .filter((pl.col("volume") > 0) & (pl.col("mod") >= 570) & (pl.col("mod") <= 960))
           .group_by("date").agg(pl.col("high").max().alias("high_1m"), pl.col("low").min().alias("low_1m"))
           .sort("date").to_pandas())
    b["date"] = pd.to_datetime(b["date"])
    return b.set_index("date")


def load_stock(symbol: str = "spy", verbose: bool = False) -> pd.DataFrame:
    """SPY daily bars indexed by date, with prev_close, ret_cc, range_pct, gap and open-to-close moves.

    High and low are the narrower of the EOD report and the 1-minute session range: a single bad print in
    either source widens the range (2026-02-02 EOD low 69.005), the intersection removes it. The two
    sources agree exactly on all but a handful of days; the count is printed with verbose=True."""
    sym = symbol.lower()
    p_stock, p_index = CACHE / f"{sym}_stock_eod.parquet", CACHE / f"{sym}_index_eod.parquet"
    df = pl.read_parquet(p_stock if p_stock.exists() else p_index)
    tcol = "created" if "created" in df.columns else [c for c in df.columns if "time" in c.lower() or "date" in c.lower()][0]
    if "volume" not in df.columns:
        df = df.with_columns(pl.lit(0).alias("volume"))
    out = (df.with_columns(pl.col(tcol).dt.date().alias("date"))
             .select("date", "open", "high", "low", "close", "volume").sort("date").to_pandas())
    out["date"] = pd.to_datetime(out["date"])
    out = out.set_index("date")
    out["high_eod"], out["low_eod"] = out["high"], out["low"]
    sr = session_range_1m(symbol)
    if sr is not None:
        j = out.join(sr, how="left")
        hi = np.fmax(np.fmin(j["high_eod"], j["high_1m"]), np.fmax(j["open"], j["close"]))   # the 16:00 auction print is not in the 1m bars
        lo = np.fmin(np.fmax(j["low_eod"], j["low_1m"]), np.fmin(j["open"], j["close"]))
        d_range = ((j["high_eod"] - j["low_eod"]) - (hi - lo)) / j["close"]
        out["high"], out["low"] = hi, lo
        out["range_change"] = d_range
        if verbose:
            big = d_range > 0.001
            print(f"daily high/low: narrower of the EOD report and the 1-minute session range, open and close included, on {int(j['high_1m'].notna().sum())} days; "
                  f"range narrowed by more than 0.1% of the close on {int(big.sum())} days: "
                  f"{[(d.strftime('%Y-%m-%d'), round(float(v) * 100, 2)) for d, v in d_range[big].items()]}")
    out = out.reset_index()
    out["prev_close"] = out["close"].shift(1)
    out["ret_cc"] = np.log(out["close"] / out["prev_close"])
    out["range_pct"] = (out["high"] - out["low"]) / out["prev_close"]
    out["gap_pct"] = out["open"] / out["prev_close"] - 1.0     # close of D-1 to open of D
    out["oc_pct"] = out["close"] / out["open"] - 1.0           # open of D to close of D
    out["date"] = pd.to_datetime(out["date"])
    return out.set_index("date")


def load_sofr() -> pd.Series:
    """SOFR as a decimal, indexed by calendar date (Theta reports percent)."""
    df = pl.read_parquet(CACHE / "sofr.parquet")
    s = (df.with_columns(pl.col("created").cast(pl.Utf8).str.slice(0, 10).str.strptime(pl.Date, "%Y-%m-%d").alias("date"))
           .select("date", "rate").sort("date").to_pandas())
    s["date"] = pd.to_datetime(s["date"])
    return s.set_index("date")["rate"] / 100.0


def rate_for(sofr: pd.Series, day) -> float:
    day = pd.Timestamp(day)
    if day in sofr.index and np.isfinite(sofr.loc[day]):
        return float(sofr.loc[day])
    prior = sofr.loc[:day].dropna()
    if len(prior):
        return float(prior.iloc[-1])
    later = sofr.dropna()          # days before the first SOFR record take the first record (2023 on this account)
    return float(later.iloc[0]) if len(later) else R_FALLBACK


# ----------------------------------------------------------------------------- step 3: chain prep
def prep_day(day: dt.date, S: float, r: float, symbol: str = "spy", q: float | None = None) -> tuple[pl.DataFrame, dict]:
    """One day's chain in the run_chain.py schema plus open_interest, r, q, date, dte.

    Universe = the OI record dated `day` (published ~06:30 ET that morning = positions as of the
    prior close, the last OI available during the session). Quotes of `day` are left-joined on
    (expiration, strike, right). Returns the kept rows and the share of the day's OI in each
    drop bucket (DROP_ORDER).
    """
    sym = symbol.lower()
    q = Q_FLAT if (q is None and sym == "spy") else (q_for(sym) if q is None else q)
    eod = (pl.read_parquet(CACHE / f"{sym}_eod_{day}.parquet")
             .select(KEY + ["bid", "ask"])
             .with_columns(pl.col("right").str.slice(0, 1))
             .unique(subset=KEY, keep="first"))
    oi = (pl.read_parquet(CACHE / f"{sym}_oi_{day}.parquet")
            .with_columns(pl.col("right").str.slice(0, 1))
            .sort("timestamp")
            .unique(subset=KEY, keep="last")            # latest OI message of the morning wins
            .select(KEY + ["open_interest"]))
    ch = (oi.join(eod, on=KEY, how="left")
            .with_columns(
                (pl.col("expiration").str.strptime(pl.Date, "%Y-%m-%d") - pl.lit(day)).dt.total_days().cast(pl.Int32).alias("dte"),
                ((pl.col("bid") + pl.col("ask")) / 2).alias("mid"),
                (pl.col("right") == "C").alias("is_call"))
            .with_columns(
                (pl.col("dte") / 365.25).alias("T_years"),
                pl.when(pl.col("is_call")).then(pl.lit(S) - pl.col("strike")).otherwise(pl.col("strike") - pl.lit(S))
                  .clip(lower_bound=0.0).alias("intrinsic")))
    reason = (pl.when(pl.col("dte") < DTE_MIN).then(pl.lit("dte_lt_1"))
                .when(pl.col("dte") > DTE_MAX).then(pl.lit("dte_gt_120"))
                .when(~pl.col("strike").is_between(STRIKE_BAND[0] * S, STRIKE_BAND[1] * S)).then(pl.lit("outside_band"))
                .when(pl.col("open_interest") <= 0).then(pl.lit("oi_zero"))
                .when(pl.col("bid").is_null()).then(pl.lit("no_quote"))
                .when(pl.col("bid") <= 0).then(pl.lit("bid_le_0"))
                .when(pl.col("ask") < pl.col("bid")).then(pl.lit("ask_lt_bid"))
                .when(pl.col("mid") <= pl.col("intrinsic") + 1e-6).then(pl.lit("mid_le_intrinsic"))
                .otherwise(pl.lit("kept")))
    ch = ch.with_columns(reason.alias("drop_reason"))
    tot = ch["open_interest"].sum()
    sh = dict.fromkeys(DROP_ORDER, 0.0)
    for k, v in ch.group_by("drop_reason").agg(pl.col("open_interest").sum()).iter_rows():
        sh[k] = v / tot if tot else np.nan
    sh["oi_total"] = int(tot)
    sh["n_oi_rows"] = ch.height
    for k, v in ch.group_by("drop_reason").agg(pl.len()).iter_rows():
        sh[f"rows_{k}"] = v
    kept = (ch.filter(pl.col("drop_reason") == "kept")
              .with_columns(pl.lit(sym.upper()).alias("symbol"), pl.lit(float(S)).alias("S"), pl.lit(float(r)).alias("r"),
                            pl.lit(float(q)).alias("q"), pl.lit(day).alias("date"))
              .select("symbol", "expiration", "strike", "right", "S", "T_years", "mid", "is_call", "bid", "ask",
                      "open_interest", "r", "q", "date", "dte"))
    return kept, sh


def prep_all(days, stock: pd.DataFrame, sofr: pd.Series, verbose: bool = True, symbol: str = "spy", q: float | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Concatenate prep_day over `days`. Returns (chain as pandas, per-day OI shares as pandas)."""
    chains, shares, skipped = [], [], []
    t0 = time.perf_counter()
    for d in days:
        ts = pd.Timestamp(d)
        if ts not in stock.index:
            skipped.append(d)
            continue
        kept, sh = prep_day(d, float(stock.loc[ts, "close"]), rate_for(sofr, d), symbol=symbol, q=q)
        chains.append(kept)
        shares.append({"date": ts, **sh})
    chain = pl.concat(chains).to_pandas()
    chain["date"] = pd.to_datetime(chain["date"])
    sh = pd.DataFrame(shares).set_index("date")
    if verbose:
        rows = chain.groupby("date").size()
        print(f"chain prep: {len(shares)} days, {len(chain):,} rows; rows/day min {rows.min()} median {int(rows.median())} max {rows.max()}; "
              f"{time.perf_counter() - t0:.1f}s; skipped (no SPY bar): {skipped}")
    return chain, sh


def filter_table(shares: pd.DataFrame) -> pd.DataFrame:
    """Median (and max) share of the day's OI in each drop bucket, for the notebook."""
    rows = []
    for k in DROP_ORDER:
        rows.append({"bucket": k, "what": DROP_LABEL[k], "median OI share": shares[k].median(), "max OI share": shares[k].max()})
    return pd.DataFrame(rows).set_index("bucket")


# ----------------------------------------------------------------------------- step 4: engine
def solve_chain(df: pd.DataFrame, target: str = "cpu", dtype: str = "fp64", evals: int = 12, sig0: float = 0.3):
    """IV, model value, delta, gamma for every row of `df` (run_chain.py sequence). Returns (df, timings).

    iv_status: 0 ok, 1 no root (mid below the sigma->0 model value), 2 vol ceiling, 3 not converged.
    Adds gex = sign * gamma * open_interest * 100 * S * S * 0.01 (calls +, puts -), in dollars per 1% move.
    """
    import numba
    from alo_numba import Engine, Tables, build_iv_cpu, build_iv_cuda, make_core, style_codes

    f = np.float32 if dtype == "fp32" else np.float64
    n = len(df)
    S = df["S"].to_numpy(np.float64); K = df["strike"].to_numpy(np.float64); T = df["T_years"].to_numpy(np.float64)
    mid = df["mid"].to_numpy(np.float64); r = df["r"].to_numpy(np.float64); q = df["q"].to_numpy(np.float64)
    is_euro = df["symbol"].astype(str).str.upper().isin(INDEX_ROOTS).to_numpy()   # index roots are European
    style = style_codes(df["is_call"].to_numpy(), is_euro)
    tab = Tables(f, 7, 7, 27)
    core = make_core(f, target, 7, 7, 27, m_iter=4)
    eng = Engine(f, target)
    arrs = [np.ascontiguousarray(a.astype(f)) for a in (mid, S, K, r, q, T)]
    sig0f = np.full(n, sig0, dtype=f)
    iv = np.zeros(n, dtype=f)
    m = min(n, 256)
    t0 = time.perf_counter()
    if target == "cuda":
        from numba import cuda
        kern = build_iv_cuda(core, f, evals=evals)
        d = [cuda.to_device(a) for a in arrs] + [cuda.to_device(style), cuda.to_device(sig0f)]
        dt_ = [cuda.to_device(a) for a in tab.args()]
        d_sc = cuda.to_device(tab.scratch(n)); d_iv = cuda.to_device(iv)
        blocks = (n + 255) // 256
        kern[blocks, 256](*d, *dt_, d_sc, d_iv); cuda.synchronize()          # includes compile
        t_compile = time.perf_counter() - t0
        t0 = time.perf_counter()
        kern[blocks, 256](*d, *dt_, d_sc, d_iv); cuda.synchronize()
        t_iv = time.perf_counter() - t0
        iv = d_iv.copy_to_host()
        device = str(cuda.get_current_device().name, "utf-8") if isinstance(cuda.get_current_device().name, bytes) else str(cuda.get_current_device().name)
    else:
        run_iv = build_iv_cpu(core, f, evals=evals)
        run_iv(*(a[:m] for a in arrs), style[:m], sig0f[:m], *tab.args(), tab.scratch(m), iv[:m])   # compile on a slice
        t_compile = time.perf_counter() - t0
        t0 = time.perf_counter()
        run_iv(*arrs, style, sig0f, *tab.args(), tab.scratch(n), iv)
        t_iv = time.perf_counter() - t0
        device = f"CPU, {numba.get_num_threads()} numba threads"
    iv = iv.astype(np.float64)
    solved = np.isfinite(iv)
    sig_use = np.where(solved, iv, 1e-3)
    eng.price(S[:m], K[:m], r[:m], q[:m], sig_use[:m], T[:m], style[:m])   # compile
    t0 = time.perf_counter()
    out = eng.price(S, K, r, q, sig_use, T, style).astype(np.float64)
    t_greeks = time.perf_counter() - t0
    resid = np.abs(out[:, 0] - mid)
    status = np.zeros(n, dtype=np.int8)
    status[solved & (iv <= 1.001e-3)] = 1
    status[solved & (iv >= 5.99)] = 2
    status[solved & (status == 0) & (resid > 1e-3 * np.maximum(1.0, mid))] = 3
    status[~solved] = 1
    df = df.copy()
    df["iv"] = iv; df["value"] = out[:, 0]; df["delta"] = out[:, 1]; df["gamma"] = out[:, 2]
    df["resid"] = resid; df["iv_status"] = status
    sign = np.where(df["is_call"].to_numpy(), 1.0, -1.0)
    df["gex"] = sign * df["gamma"].to_numpy() * df["open_interest"].to_numpy(np.float64) * 100.0 * S * S * 0.01
    timings = {"target": target, "dtype": dtype, "device": device, "n_contracts": int(n), "evals": evals,
               "t_compile_s": round(t_compile, 2), "t_iv_s": round(t_iv, 3), "t_greeks_s": round(t_greeks, 3)}
    return df, timings


def solve_report(sol: pd.DataFrame) -> dict:
    good = sol["iv_status"].to_numpy() == 0
    if not good.any():
        raise ValueError("no contract with iv_status 0")
    resid = sol["resid"].to_numpy()[good]
    oi = sol["open_interest"].to_numpy(np.float64)
    return {
        "rows": int(len(sol)),
        "iv_status_counts": {int(k): int(v) for k, v in sol["iv_status"].value_counts().sort_index().items()},
        "oi_share_status_ne_0": float(oi[~good].sum() / oi.sum()),
        "resid_median": float(np.median(resid)), "resid_p99": float(np.percentile(resid, 99)), "resid_max": float(resid.max()),
        "gamma_zero_rows": int((sol["gamma"].to_numpy() == 0).sum()),
    }


# ----------------------------------------------------------------------------- step 4: daily series
def flip_levels(sol: pd.DataFrame, lo: float = 0.93, hi: float = 1.07, step: float = 0.0025, chunk_days: int = 25,
                target: str = "cpu", verbose: bool = True) -> pd.DataFrame:
    """Re-evaluate every solved contract on a spot grid lo..hi x S (IV fixed), sum signed GEX per grid
    point and day, return per day: flip_level (zero crossing nearest S, NaN if none), n_crossings,
    gex_at_lo, gex_at_hi. Same construction as the deck's gex_by_strike(spot)."""
    from alo_numba import Engine, style_codes
    eng = Engine(np.float64, target)
    rel = np.round(np.arange(lo, hi + step / 2, step), 6)
    m = len(rel)
    ok = sol[sol["iv_status"] == 0]
    days = sorted(ok["date"].unique())
    rows = []
    t0 = time.perf_counter()
    for i in range(0, len(days), chunk_days):
        sub = ok[ok["date"].isin(days[i:i + chunk_days])]
        n = len(sub)
        S0 = sub["S"].to_numpy(np.float64)
        Sg = (S0[:, None] * rel[None, :]).ravel()
        rep = lambda col: np.repeat(sub[col].to_numpy(np.float64), m)  # noqa: E731
        style = np.repeat(style_codes(sub["is_call"].to_numpy(), False), m)
        res = eng.price(Sg, rep("strike"), rep("r"), rep("q"), rep("iv"), rep("T_years"), style)
        gamma = res[:, 2].reshape(n, m)
        sign = np.where(sub["is_call"].to_numpy(), 1.0, -1.0)[:, None]
        gex = sign * gamma * sub["open_interest"].to_numpy(np.float64)[:, None] * 100.0 * Sg.reshape(n, m) ** 2 * 0.01
        codes, uniq = pd.factorize(sub["date"], sort=True)
        net = np.zeros((len(uniq), m))
        np.add.at(net, codes, gex)
        for j, d in enumerate(uniq):
            prof = net[j]
            sgn = np.sign(prof)
            cross = np.where(sgn[:-1] * sgn[1:] < 0)[0]
            level = np.nan
            if len(cross):
                xs = rel[cross] + step * prof[cross] / (prof[cross] - prof[cross + 1])
                level = float(S0[codes == j][0] * xs[np.argmin(np.abs(xs - 1.0))])
            rows.append({"date": d, "flip_level": level, "n_crossings": int(len(cross)),
                         "gex_at_lo": prof[0], "gex_at_hi": prof[-1]})
        if verbose:
            print(f"flip grid: days {i + 1}-{min(i + chunk_days, len(days))} of {len(days)}, {n * m:,} evaluations, {time.perf_counter() - t0:.0f}s", flush=True)
    return pd.DataFrame(rows).set_index("date")


def daily_gex(sol: pd.DataFrame, stock: pd.DataFrame, flips: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per trading day: the spy_gex_daily.csv columns."""
    ok = sol[sol["iv_status"] == 0]
    g = ok.groupby("date")
    a = ok["gex"].abs()
    daily = pd.DataFrame({
        "gex_net_usd": g["gex"].sum(),
        "gex_call_usd": ok[ok["is_call"]].groupby("date")["gex"].sum(),
        "gex_put_usd": ok[~ok["is_call"]].groupby("date")["gex"].sum(),
        "gex_le30d_share": a.where(ok["dte"] <= 30, 0.0).groupby(ok["date"]).sum() / a.groupby(ok["date"]).sum(),
        "n_contracts": g.size(),
        "oi_total": sol.groupby("date")["open_interest"].sum(),
        "oi_solved_share": g["open_interest"].sum() / sol.groupby("date")["open_interest"].sum(),
    })
    by_strike = ok.groupby(["date", "strike"])["gex"].sum()
    daily["wall_pos_strike"] = by_strike.groupby(level=0).idxmax().map(lambda t: t[1])
    daily["wall_neg_strike"] = by_strike.groupby(level=0).idxmin().map(lambda t: t[1])
    daily["flip_level"] = flips["flip_level"] if flips is not None else np.nan
    st = stock.loc[daily.index, ["close", "open", "high", "low", "ret_cc", "range_pct"]].rename(columns={"close": "S"})
    st["close"] = st["S"]
    daily = st.join(daily)
    daily["regime"] = np.sign(daily["gex_net_usd"]).astype(int)
    cols = ["S", "open", "high", "low", "close", "ret_cc", "range_pct", "gex_net_usd", "gex_call_usd", "gex_put_usd",
            "gex_le30d_share", "n_contracts", "oi_total", "oi_solved_share", "flip_level", "wall_pos_strike", "wall_neg_strike", "regime"]
    daily.index.name = "date"
    return daily[cols]


# ----------------------------------------------------------------------------- step 3 extra: strike band check
def band_share_day(day, S: float, r: float) -> pd.DataFrame:
    """Solve one day with no strike band and report the share of total |GEX| by K/S band."""
    global STRIKE_BAND
    saved = STRIKE_BAND
    STRIKE_BAND = (0.0, 1e9)
    try:
        kept, _ = prep_day(day, S, r)
    finally:
        STRIKE_BAND = saved
    sol, _ = solve_chain(kept.to_pandas())
    ok = sol[sol["iv_status"] == 0].copy()
    m = ok["strike"] / ok["S"]
    ok["band"] = pd.cut(m, [0, 0.8, 0.9, 1.1, 1.2, np.inf], labels=["< 0.80 S", "0.80 to 0.90 S", "0.90 to 1.10 S", "1.10 to 1.20 S", "> 1.20 S"])
    a = ok["gex"].abs()
    out = pd.DataFrame({"abs GEX share": a.groupby(ok["band"], observed=False).sum() / a.sum(),
                        "OI share": ok["open_interest"].groupby(ok["band"], observed=False).sum() / ok["open_interest"].sum(),
                        "contracts": ok.groupby("band", observed=False).size()})
    return out


# ----------------------------------------------------------------------------- step 5: regimes and episodes
def sign_runs(daily: pd.DataFrame, col: str = "regime") -> pd.DataFrame:
    """Consecutive runs of equal `col`: sign, start, end, length, median GEX."""
    reg = daily[col].to_numpy()
    idx = daily.index
    rows, start = [], 0
    for i in range(1, len(reg) + 1):
        if i == len(reg) or reg[i] != reg[start]:
            seg = daily.iloc[start:i]
            rows.append({"sign": int(reg[start]), "start": idx[start], "end": idx[i - 1], "length": i - start,
                         "median_gex": float(seg["gex_net_usd"].median())})
            start = i
    return pd.DataFrame(rows)


def episode_stats(stock: pd.DataFrame, start, end, pre: int = 20) -> dict:
    """Outcomes while the regime is in force (start+1 .. end+1) against the `pre` days ending at start."""
    i0, i1 = stock.index.get_indexer([start, end])
    inside = stock.iloc[i0 + 1: i1 + 2]
    before = stock.iloc[max(0, i0 - pre + 1): i0 + 1]
    last = min(i1 + 1, len(stock) - 1)
    ann = np.sqrt(252.0)
    return {"spy_return_in_force": float(stock["close"].iloc[last] / stock["close"].iloc[i0] - 1.0),
            "range_inside": float(inside["range_pct"].mean()), "range_before": float(before["range_pct"].mean()),
            "rvol_inside": float(inside["ret_cc"].std(ddof=1) * ann), "rvol_before": float(before["ret_cc"].std(ddof=1) * ann),
            "n_inside": int(len(inside)), "n_before": int(len(before))}


def select_episodes(daily: pd.DataFrame, stock: pd.DataFrame, n_each: int = 4, neg_min: int = 3, pos_min: int = 10):
    """Rule: negative = 4 most recent runs of >= neg_min days; positive = 4 most recent runs of >= pos_min days
    whose median GEX is in the top tercile of the whole history. Returns (episodes, info)."""
    runs = sign_runs(daily)
    top_tercile = float(daily["gex_net_usd"].quantile(2.0 / 3.0))
    neg = runs[(runs["sign"] == -1) & (runs["length"] >= neg_min)].sort_values("start").tail(n_each)
    pos = runs[(runs["sign"] == 1) & (runs["length"] >= pos_min) & (runs["median_gex"] >= top_tercile)].sort_values("start").tail(n_each)
    rows = []
    for kind, sel in (("neg", neg), ("pos", pos)):
        for _, r in sel.iterrows():
            rows.append({"kind": kind, **r.to_dict(), **episode_stats(stock, r["start"], r["end"])})
    eps = pd.DataFrame(rows)
    info = {"n_runs": int(len(runs)), "top_tercile_gex_usd": top_tercile,
            "n_neg_runs_ge_min": int(((runs["sign"] == -1) & (runs["length"] >= neg_min)).sum()),
            "n_pos_runs_ge_min": int(((runs["sign"] == 1) & (runs["length"] >= pos_min)).sum()),
            "n_pos_runs_ge_min_top_tercile": int(len(pos)),
            "longest_pos_run": int(runs.loc[runs["sign"] == 1, "length"].max()) if (runs["sign"] == 1).any() else 0,
            "longest_neg_run": int(runs.loc[runs["sign"] == -1, "length"].max()) if (runs["sign"] == -1).any() else 0}
    return eps, info


# ----------------------------------------------------------------------------- step 6: leading-indicator tests
TESTS = [
    ("range", "range_next", "range_pct on D+1", "lower in +GEX"),
    ("move", "absret_next", "abs(ret_cc) on D+1", "lower in +GEX"),
    ("forward vol", "fvol_5d", "annualized std of ret_cc over D+1..D+5", "lower in +GEX"),
    ("continuation", "continuation", "sign(ret D+1) == sign(ret D)", "lower in +GEX"),
    ("gap fade", "gap_fade", "sign(open->close D+1) != sign(close D -> open D+1)", "higher in +GEX"),
    ("autocorrelation", "ac_product", "lag-1 autocorrelation of ret_cc within regime", "negative in +GEX"),
]


def next_day_outcomes(daily: pd.DataFrame, stock: pd.DataFrame) -> pd.DataFrame:
    """Per regime day D (index): the D+1 / D+1..D+5 outcomes, the regime and GEX known at the close of D."""
    pos = stock.index.get_indexer(daily.index)
    n = len(stock)

    def take(col, off):
        p = pos + off
        ok = (p >= 0) & (p < n)
        v = np.full(len(pos), np.nan)
        v[ok] = stock[col].to_numpy()[p[ok]]
        return v

    out = pd.DataFrame(index=daily.index)
    out["regime"] = daily["regime"].to_numpy()
    out["gex"] = daily["gex_net_usd"].to_numpy()
    out["ret_D"] = take("ret_cc", 0)
    out["ret_next"] = take("ret_cc", 1)
    out["range_next"] = take("range_pct", 1)
    out["absret_next"] = np.abs(out["ret_next"])
    fwd = np.column_stack([take("ret_cc", k) for k in range(1, 6)])
    fv = np.nanstd(fwd, axis=1, ddof=1) * np.sqrt(252.0)
    fv[np.isnan(fwd).any(axis=1)] = np.nan
    out["fvol_5d"] = fv
    cont = (np.sign(out["ret_next"]) == np.sign(out["ret_D"])).astype(float)
    cont[out["ret_next"].isna() | out["ret_D"].isna()] = np.nan
    out["continuation"] = cont
    oc, gap = take("oc_pct", 1), take("gap_pct", 1)
    fade = (np.sign(oc) != np.sign(gap)).astype(float)
    fade[np.isnan(oc) | np.isnan(gap)] = np.nan
    out["gap_fade"] = fade
    out["ac_product"] = out["ret_D"] * out["ret_next"]
    return out


def tercile_cut(out: pd.DataFrame) -> pd.Series:
    """+1 for the top third of GEX, -1 for the bottom third, 0 for the middle (a second regime definition)."""
    lo, hi = out["gex"].quantile([1.0 / 3.0, 2.0 / 3.0])
    return pd.Series(np.where(out["gex"] >= hi, 1, np.where(out["gex"] <= lo, -1, 0)), index=out.index)


def regime_tests(out: pd.DataFrame, group: pd.Series, n_boot: int = 10000, seed: int = 0) -> pd.DataFrame:
    """For each TESTS row: n, mean, median per regime, the difference (+ minus -), bootstrap 95% CI of the
    difference (n_boot resamples, seed), Mann-Whitney U p-value. The autocorrelation row reports the
    lag-1 correlation per regime, a bootstrap CI of the difference in correlations, and the Mann-Whitney
    p-value of the per-day products ret_D * ret_next."""
    from scipy.stats import mannwhitneyu
    rng = np.random.default_rng(seed)
    rows = []
    for name, col, desc, expect in TESTS:
        A = out[(group == 1).to_numpy()]
        B = out[(group == -1).to_numpy()]
        if name == "autocorrelation":
            pa = A[["ret_D", "ret_next"]].dropna().to_numpy()
            pb = B[["ret_D", "ret_next"]].dropna().to_numpy()
            corr = lambda m: float(np.corrcoef(m[:, 0], m[:, 1])[0, 1])  # noqa: E731
            sa, sb = corr(pa), corr(pb)
            boots = np.array([corr(pa[rng.integers(0, len(pa), len(pa))]) - corr(pb[rng.integers(0, len(pb), len(pb))]) for _ in range(n_boot)])
            a, b = A[col].dropna().to_numpy(), B[col].dropna().to_numpy()
            p = float(mannwhitneyu(a, b, alternative="two-sided").pvalue)
            rows.append({"test": name, "outcome": desc, "n_pos": len(pa), "n_neg": len(pb), "mean_pos": sa, "mean_neg": sb,
                         "median_pos": np.nan, "median_neg": np.nan, "diff": sa - sb,
                         "ci_lo": float(np.percentile(boots, 2.5)), "ci_hi": float(np.percentile(boots, 97.5)), "p_mwu": p, "expectation": expect})
            continue
        a, b = A[col].dropna().to_numpy(), B[col].dropna().to_numpy()
        ba = rng.choice(a, (n_boot, len(a))).mean(axis=1)
        bb = rng.choice(b, (n_boot, len(b))).mean(axis=1)
        boots = ba - bb
        p = float(mannwhitneyu(a, b, alternative="two-sided").pvalue)
        rows.append({"test": name, "outcome": desc, "n_pos": len(a), "n_neg": len(b), "mean_pos": float(a.mean()), "mean_neg": float(b.mean()),
                     "median_pos": float(np.median(a)), "median_neg": float(np.median(b)), "diff": float(a.mean() - b.mean()),
                     "ci_lo": float(np.percentile(boots, 2.5)), "ci_hi": float(np.percentile(boots, 97.5)), "p_mwu": p, "expectation": expect})
    return pd.DataFrame(rows).set_index("test")


def verdicts(tests: pd.DataFrame) -> pd.Series:
    """Does the measured difference have the expected sign, with a bootstrap CI excluding 0?"""
    v = {}
    for name, r in tests.iterrows():
        want_lower = r["expectation"].startswith("lower") or r["expectation"].startswith("negative")
        sign_ok = (r["diff"] < 0) if want_lower else (r["diff"] > 0)
        ci_excl = (r["ci_hi"] < 0) or (r["ci_lo"] > 0)
        v[name] = "supported" if (sign_ok and ci_excl) else ("direction only, CI includes 0" if sign_ok else "not supported")
    return pd.Series(v, name="verdict")


def cross_correlation(daily: pd.DataFrame, stock: pd.DataFrame, ks=range(-5, 6), col: str = "range_pct") -> pd.DataFrame:
    """corr(gex_net_usd on D, `col` on D+k). k > 0 is the leading side, k < 0 the reacting side."""
    from scipy.stats import spearmanr
    g = daily["gex_net_usd"].to_numpy()
    pos = stock.index.get_indexer(daily.index)
    rows = []
    for k in ks:
        p = pos + k
        ok = (p >= 0) & (p < len(stock))
        x, y = g[ok], stock[col].to_numpy()[p[ok]]
        m = np.isfinite(x) & np.isfinite(y)
        rows.append({"k": k, "pearson": float(np.corrcoef(x[m], y[m])[0, 1]), "spearman": float(spearmanr(x[m], y[m]).correlation), "n": int(m.sum())})
    return pd.DataFrame(rows).set_index("k")


def pinning_test(daily: pd.DataFrame) -> pd.DataFrame:
    """Distance from the close of D to wall_pos_strike known at the close of D-1, in % of S, by the regime in
    force on D. Baseline: distance to the nearest 5-dollar strike. SPY lists expirations every weekday in
    this window, so every day is an expiration day; the monthly (third Friday) subgroup is shown too."""
    from scipy.stats import mannwhitneyu
    prev = daily[["wall_pos_strike", "regime", "close"]].shift(1)
    close = daily["close"]
    d_wall = (close - prev["wall_pos_strike"]).abs() / close * 100.0
    d_prev = (prev["close"] - prev["wall_pos_strike"]).abs() / close * 100.0
    d_5 = (close - 5.0 * np.round(close / 5.0)).abs() / close * 100.0
    approach = (d_wall < d_prev).astype(float)
    idx = daily.index
    third_fri = pd.Series([(t.weekday() == 4) and (15 <= t.day <= 21) for t in idx], index=idx)
    rows = []
    for label, mask in (("all days", pd.Series(True, index=idx)), ("third Fridays", third_fri)):
        for reg in (1, -1):
            m = mask & (prev["regime"] == reg) & d_wall.notna()
            rows.append({"days": label, "regime in force": reg, "n": int(m.sum()), "mean dist to wall %": float(d_wall[m].mean()),
                         "median dist to wall %": float(d_wall[m].median()), "mean dist to nearest $5 %": float(d_5[m].mean()),
                         "share of days closer to wall than prev close": float(approach[m].mean())})
        a = d_wall[mask & (prev["regime"] == 1) & d_wall.notna()]
        b = d_wall[mask & (prev["regime"] == -1) & d_wall.notna()]
        p = float(mannwhitneyu(a, b, alternative="two-sided").pvalue) if len(a) > 2 and len(b) > 2 else np.nan
        rows.append({"days": label, "regime in force": "p (MWU, + vs -)", "n": int(len(a) + len(b)), "mean dist to wall %": p,
                     "median dist to wall %": np.nan, "mean dist to nearest $5 %": np.nan, "share of days closer to wall than prev close": np.nan})
    return pd.DataFrame(rows).set_index(["days", "regime in force"])


# ----------------------------------------------------------------------------- step 9: vendor cross-check
def vendor_frame(days) -> pd.DataFrame:
    """Vendor gamma and IV for every cached day, keyed like the chain."""
    frames = []
    for d in days:
        day = pd.Timestamp(d).date()
        p = CACHE / f"spy_greeks_{day}.parquet"
        if not p.exists():
            continue
        frames.append(pl.read_parquet(p).select(KEY + ["gamma", "implied_vol", "underlying_price"])
                        .with_columns(pl.col("right").str.slice(0, 1), pl.lit(day).alias("date"))
                        .unique(subset=KEY, keep="first"))
    v = pl.concat(frames).rename({"gamma": "gamma_vendor", "implied_vol": "iv_vendor"}).to_pandas()
    v["date"] = pd.to_datetime(v["date"])
    return v


def vendor_compare(sol: pd.DataFrame) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """Vendor GEX with the same rows, OI and formula. Returns (per-day frame, stats dict, per-bucket gamma table)."""
    ok = sol[sol["iv_status"] == 0].copy()
    v = vendor_frame(sorted(ok["date"].unique()))
    m = ok.merge(v, on=["date"] + KEY, how="left")
    has = m["gamma_vendor"].notna() & np.isfinite(m["gamma_vendor"])
    sign = np.where(m["is_call"], 1.0, -1.0)
    m["gex_vendor"] = sign * m["gamma_vendor"] * m["open_interest"] * 100.0 * m["S"] ** 2 * 0.01
    both = m[has]

    def per_day(frame):
        d = pd.DataFrame({"engine": frame.groupby("date")["gex"].sum(), "vendor": frame.groupby("date")["gex_vendor"].sum()})
        d["rel_diff"] = (d["vendor"] - d["engine"]).abs() / d["engine"].abs()
        return d

    def stats(d):
        ad = (d["vendor"] - d["engine"]).abs()
        return {"n_days": int(len(d)), "corr": float(np.corrcoef(d["engine"], d["vendor"])[0, 1]),
                "sign_agreement": float((np.sign(d["engine"]) == np.sign(d["vendor"])).mean()),
                "rel_diff_median": float(d["rel_diff"].median()), "rel_diff_p99": float(d["rel_diff"].quantile(0.99)),
                "abs_diff_usd_median": float(ad.median()), "abs_diff_usd_p99": float(ad.quantile(0.99)),
                "vendor_minus_engine_usd_median": float((d["vendor"] - d["engine"]).median())}

    all_days = per_day(both)
    near = per_day(both[both["dte"] <= 7])
    ratio = {}
    for lab, sub in (("calls", both[both["is_call"]]), ("puts", both[~both["is_call"]])):
        r = sub.groupby("date")["gex_vendor"].sum() / sub.groupby("date")["gex"].sum()
        ratio[lab] = {"median": float(r.median()), "p10": float(r.quantile(0.1)), "p90": float(r.quantile(0.9))}
    out = {"vendor_gamma_missing_share_rows": float(1.0 - has.mean()),
           "vendor_gamma_missing_share_oi": float(m.loc[~has, "open_interest"].sum() / m["open_interest"].sum()),
           "vendor_gamma_zero_share_rows": float((both["gamma_vendor"] == 0).mean()),
           "daily_vendor_over_engine_gex_by_right": ratio,
           "all": stats(all_days), "dte_le_7": stats(near)}
    both = both.copy()
    both["rel_gamma_diff"] = (both["gamma_vendor"] - both["gamma"]) / both["gamma"].where(both["gamma"] > 0)
    both["moneyness"] = pd.cut(both["strike"] / both["S"], [0, 0.95, 0.99, 1.01, 1.05, 9], labels=["K < 0.95 S", "0.95 to 0.99", "0.99 to 1.01", "1.01 to 1.05", "K > 1.05 S"])
    both["dte_bucket"] = pd.cut(both["dte"], [0, 7, 30, 120], labels=["1 to 7 d", "8 to 30 d", "31 to 120 d"])
    tab = both.groupby(["dte_bucket", "moneyness", "right"], observed=True)["rel_gamma_diff"].median().unstack("right")
    return all_days.join(near, rsuffix="_le7d"), out, tab


# ----------------------------------------------------------------------------- step 8: figure style
BG, FG, GRID, DIM, DIM2 = "#000000", "#e6edf3", "#21262d", "#8b949e", "#484f58"
CYAN, ORANGE, PURPLE = "#58a6ff", "#f0883e", "#bc8cff"
FIG_W, FIG_H, DPI = 9.6, 5.4, 200


def style():
    import matplotlib as mpl
    from matplotlib import font_manager
    names = {f.name for f in font_manager.fontManager.ttflist}
    fam = [n for n in ("Inter", "JetBrains Mono") if n in names] + ["DejaVu Sans"]
    mpl.rcParams.update({
        "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
        "text.color": FG, "axes.labelcolor": FG, "axes.edgecolor": DIM2,
        "xtick.color": DIM, "ytick.color": DIM, "xtick.major.size": 0, "ytick.major.size": 0,
        "xtick.minor.size": 0, "ytick.minor.size": 0,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "font.family": fam, "font.size": 12, "axes.titlesize": 17, "axes.titleweight": "bold",
        "axes.labelsize": 12, "legend.frameon": False, "legend.fontsize": 11,
        "figure.dpi": 100, "savefig.dpi": DPI,
    })
    FIGURES.mkdir(exist_ok=True)
    return fam[0]


def regime_color(sign):
    return CYAN if sign > 0 else ORANGE


def shade_regimes(ax, dates, in_force, alpha=0.18):
    """Background tint per day by the regime in force that day (cyan +, orange -)."""
    import matplotlib.dates as mdates
    d = mdates.date2num(dates)
    for x, s in zip(d, in_force):
        if s == 0 or np.isnan(s):
            continue
        ax.axvspan(x - 0.5, x + 0.5, color=regime_color(s), alpha=alpha, lw=0)


# ----------------------------------------------------------------------------- step 8: figures
def _bn(x):
    return np.asarray(x, dtype=float) / 1e9


def in_force(daily: pd.DataFrame) -> pd.Series:
    """Regime in force on each day = sign of GEX at the previous close (NaN on the first day)."""
    return daily["regime"].shift(1)


def fig_overview(daily: pd.DataFrame, path=None):
    import matplotlib.pyplot as plt
    style()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(FIG_W, FIG_H), sharex=True, gridspec_kw={"height_ratios": [2.0, 1.0], "hspace": 0.08})
    shade_regimes(ax1, daily.index, in_force(daily).to_numpy(), alpha=0.22)
    ax1.plot(daily.index, daily["close"], color=FG, lw=1.3)
    ax1.set_ylabel("SPY close, $")
    neg_share = (daily["regime"] < 0).mean()
    ax1.set_title("SPY close shaded by the GEX regime in force", loc="left")
    ax1.text(0.99, 0.04, f"cyan: positive GEX in force   orange: negative GEX in force   ({neg_share:.0%} of days negative)",
             transform=ax1.transAxes, ha="right", va="bottom", color=DIM, fontsize=10)
    g = _bn(daily["gex_net_usd"])
    ax2.bar(daily.index, g, width=1.0, color=[regime_color(s) for s in np.sign(g)], lw=0)
    ax2.axhline(0, color=DIM2, lw=0.8)
    ax2.set_ylabel("net GEX, $bn per 1% move")
    ax2.set_xlabel("")
    fig.align_ylabels()
    if path:
        fig.savefig(path, bbox_inches="tight", pad_inches=0.25)
    return fig


def fig_episode(daily: pd.DataFrame, stock: pd.DataFrame, start, end, kind: str, path=None, pad: int = 10):
    """Slide chart: candles shaded by the regime in force, flip level dotted, walls at the right edge,
    daily range bars below, `pad` trading days on each side of the episode."""
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    style()
    i0, i1 = stock.index.get_indexer([start, end])
    win = stock.iloc[max(0, i0 - pad): i1 + pad + 1]
    d = daily.reindex(win.index)
    force = in_force(daily).reindex(win.index)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(FIG_W, FIG_H), sharex=True, gridspec_kw={"height_ratios": [2.6, 1.0], "hspace": 0.08})
    shade_regimes(ax1, win.index, force.to_numpy(), alpha=0.25)
    shade_regimes(ax2, win.index, force.to_numpy(), alpha=0.25)
    x = mdates.date2num(win.index)
    up = win["close"] >= win["open"]
    ax1.vlines(x, win["low"], win["high"], color=FG, lw=1.0)
    ax1.vlines(x, win["open"], win["close"], color=np.where(up, FG, DIM), lw=5.5)
    fl = d["flip_level"]
    if fl.notna().any():
        ax1.plot(win.index, fl, ls=":", color=PURPLE, lw=2.0, label="flip level")
    ax1.axvspan(mdates.date2num(start) - 0.5, mdates.date2num(end) + 0.5, fill=False, edgecolor=FG, lw=1.0, ls="--")
    last = d.loc[end]
    xr = x[-1] + 0.9
    for col, c, lab in (("wall_pos_strike", CYAN, "+ wall"), ("wall_neg_strike", ORANGE, "- wall")):
        y = last[col]
        if np.isfinite(y):
            ax1.plot([xr - 0.4, xr + 0.6], [y, y], color=c, lw=3.5, solid_capstyle="butt")
            ax1.text(xr + 0.8, y, f"{lab} {y:g}", color=c, va="center", fontsize=11, fontweight="bold")
    ax1.set_xlim(x[0] - 1.0, x[-1] + 6.0)
    ax1.set_ylabel("SPY, $")
    label = "Negative" if kind == "neg" else "Positive"
    ax1.set_title(f"{label} GEX episode, {pd.Timestamp(start):%d %b %Y} to {pd.Timestamp(end):%d %b %Y}", loc="left", pad=22)
    med = d.loc[start:end, "gex_net_usd"].median() / 1e9
    ax1.text(0.0, 1.015, f"median net GEX in the episode {med:+.1f} $bn per 1% move.  Dashed box: days the sign was observed.  Shading: regime in force (cyan +, orange -).",
             transform=ax1.transAxes, ha="left", va="bottom", color=DIM, fontsize=10)
    if fl.notna().any():
        ax1.legend(loc="upper left", fontsize=10)
    ax2.bar(x, win["range_pct"] * 100.0, width=0.8, color=[regime_color(s) if np.isfinite(s) and s != 0 else DIM2 for s in force.to_numpy()], lw=0)
    ax2.set_ylabel("daily range, %")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax2.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
    fig.align_ylabels()
    if path:
        fig.savefig(path, bbox_inches="tight", pad_inches=0.25)
    return fig


def fig_lead_test(out: pd.DataFrame, xc: pd.DataFrame, group: pd.Series, cut_label: str, path=None, seed: int = 0):
    """Left: next-day range distribution by regime (strip, medians labeled, n in legend).
    Right: corr(GEX on D, range on D+k) for k = -5..5 with k = 0 marked."""
    import matplotlib.pyplot as plt
    style()
    rng = np.random.default_rng(seed)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 4.6), gridspec_kw={"wspace": 0.34})
    for pos_, reg, c, lab in ((0, 1, CYAN, "positive GEX"), (1, -1, ORANGE, "negative GEX")):
        v = out.loc[(group == reg).to_numpy(), "range_next"].dropna().to_numpy() * 100.0
        ax1.scatter(pos_ + rng.uniform(-0.28, 0.28, len(v)), v, s=9, color=c, alpha=0.45, lw=0, label=f"{lab}, n = {len(v)}")
        m = np.median(v)
        ax1.plot([pos_ - 0.36, pos_ + 0.36], [m, m], color=FG, lw=2.2)
        ax1.text(pos_, m, f"median {m:.2f}%", color=FG, ha="center", va="bottom", fontsize=10.5,
                 bbox={"facecolor": BG, "alpha": 0.75, "lw": 0, "pad": 1.5})
    ax1.set_xticks([0, 1]); ax1.set_xticklabels(["positive GEX\nat close of D", "negative GEX\nat close of D"])
    ax1.set_ylabel("next-day range (high - low) / prev close, %")
    ax1.set_title(f"Next-day range by regime ({cut_label})", loc="left", fontsize=14)
    ax1.legend(loc="upper right", fontsize=9.5)
    ax1.set_xlim(-0.6, 1.6)
    ax1.set_ylim(0, min(np.nanpercentile(out["range_next"].dropna().to_numpy() * 100.0, 99.5) * 1.35, 12.0))
    ks = xc.index.to_numpy()
    cols = [PURPLE if k == 0 else (CYAN if k > 0 else DIM) for k in ks]
    ax2.bar(ks, xc["pearson"], color=cols, width=0.7, lw=0)
    ax2.axhline(0, color=DIM2, lw=0.8)
    ax2.axvline(0, color=PURPLE, lw=1.0, ls=":")
    ax2.set_xticks(ks)
    ax2.set_xlabel("k, trading days (k > 0: GEX leads range)")
    ax2.set_ylabel("corr(GEX on D, range on D+k)")
    ax2.set_title("Lead (k > 0) versus reaction (k < 0)", loc="left", fontsize=14, pad=20)
    ax2.text(0.0, 1.01, "gray: range before the GEX print.  cyan: range after it.  purple: same day.", transform=ax2.transAxes, color=DIM, fontsize=9.5, va="bottom")
    if path:
        fig.savefig(path, bbox_inches="tight", pad_inches=0.25)
    return fig


def fig_vendor(vd: pd.DataFrame, stats: dict, path=None):
    import matplotlib.pyplot as plt
    style()
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    e, v = _bn(vd["engine"]), _bn(vd["vendor"])
    lim = [min(e.min(), v.min()) * 1.05, max(e.max(), v.max()) * 1.05]
    ax.plot(lim, lim, color=DIM2, lw=1.0, ls="--")
    ax.scatter(e, v, s=14, color=PURPLE, alpha=0.7, lw=0)
    ax.set_xlabel("engine GEX (ALO American gamma), $bn per 1% move")
    ax.set_ylabel("vendor GEX (Theta Data gamma), $bn per 1% move")
    ax.set_title("Vendor gamma and engine gamma agree on daily GEX", loc="left")
    s = stats["all"]
    ax.text(0.02, 0.96, f"{s['n_days']} days   corr {s['corr']:.3f}   sign agreement {s['sign_agreement']:.1%}\n"
                        f"median relative difference {s['rel_diff_median']:.1%}   p99 {s['rel_diff_p99']:.1%}",
            transform=ax.transAxes, va="top", color=FG, fontsize=11)
    ax.set_xlim(lim); ax.set_ylim(lim)
    if path:
        fig.savefig(path, bbox_inches="tight", pad_inches=0.25)
    return fig


def fig_intraday(tab: pd.DataFrame, path=None):
    """tab: index regime (+1/-1), columns mean, ci_lo, ci_hi, n of the per-day lag-1 autocorrelation of 30-minute returns."""
    import matplotlib.pyplot as plt
    style()
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    xs = np.arange(len(tab))
    cols = [regime_color(r) for r in tab.index]
    ax.bar(xs, tab["mean"], color=cols, width=0.55, lw=0)
    ax.errorbar(xs, tab["mean"], yerr=[tab["mean"] - tab["ci_lo"], tab["ci_hi"] - tab["mean"]], fmt="none", ecolor=FG, elinewidth=1.6, capsize=6)
    ax.axhline(0, color=DIM2, lw=0.8)
    ax.set_xticks(xs); ax.set_xticklabels([f"{'positive' if r > 0 else 'negative'} GEX in force\nn = {int(n)} days" for r, n in zip(tab.index, tab["n"])])
    ax.set_ylabel("mean lag-1 autocorrelation of 30-minute returns")
    ax.set_title("Intraday reversal by GEX regime", loc="left", pad=22)
    ax.text(0.0, 1.015, "bars: mean of the per-day lag-1 autocorrelation of 30-minute returns.  whiskers: bootstrap 95% CI.", transform=ax.transAxes, color=DIM, fontsize=10, va="bottom")
    if path:
        fig.savefig(path, bbox_inches="tight", pad_inches=0.25)
    return fig


# ----------------------------------------------------------------------------- step 7: intraday reversal
def load_intraday() -> pl.DataFrame | None:
    files = sorted(CACHE.glob("spy_1m_????-??.parquet"))
    if not files:
        return None
    return pl.concat([pl.read_parquet(f) for f in files], how="vertical_relaxed").sort("timestamp")


def intraday_autocorr(daily: pd.DataFrame, n_boot: int = 10000, seed: int = 0):
    """Per day: lag-1 autocorrelation of 30-minute log returns (12 returns per full session, from the
    last 1-minute close in each 30-minute bucket 09:30 to 16:00). Averaged by the regime in force
    (sign of GEX at the previous close). Returns (per-day frame, summary table by regime, p-value)."""
    from scipy.stats import mannwhitneyu
    bars = load_intraday()
    if bars is None:
        return None, None, None
    # Bars with no trade carry a NaN close and zero volume; market holidays arrive as 391 zero-volume
    # bars. Keep traded bars only, take the last traded close per 30-minute bucket, drop days with
    # fewer than 8 returns.
    b = (bars.with_columns(pl.col("timestamp").dt.date().alias("date"),
                           (pl.col("timestamp").dt.hour().cast(pl.Int32) * 60 + pl.col("timestamp").dt.minute().cast(pl.Int32)).alias("mod"))
             .filter((pl.col("mod") >= 570) & (pl.col("mod") <= 960) & (pl.col("volume") > 0) & pl.col("close").is_not_nan() & pl.col("close").is_not_null())
             .with_columns(((pl.col("mod") - 570).clip(0, 389) // 30).alias("bucket"))
             .sort("timestamp")
             .group_by(["date", "bucket"]).agg(pl.col("close").last().alias("close"))
             .sort(["date", "bucket"])
             .with_columns((pl.col("close") / pl.col("close").shift(1).over("date")).log().alias("r"))
             .drop_nulls("r"))
    rows = []
    for (d,), g in b.group_by(["date"], maintain_order=True):
        r = g["r"].to_numpy()
        r = r[np.isfinite(r)]
        if len(r) >= 8 and np.std(r[1:]) > 0 and np.std(r[:-1]) > 0:
            rows.append({"date": pd.Timestamp(d), "ac1": float(np.corrcoef(r[:-1], r[1:])[0, 1]), "n_ret": int(len(r))})
    per_day = pd.DataFrame(rows).set_index("date")
    per_day["in_force"] = in_force(daily).reindex(per_day.index)
    per_day = per_day.dropna(subset=["in_force"])
    rng = np.random.default_rng(seed)
    out = []
    for reg in (1, -1):
        v = per_day.loc[per_day["in_force"] == reg, "ac1"].to_numpy()
        if len(v) == 0:
            out.append({"regime": reg, "mean": np.nan, "median": np.nan, "ci_lo": np.nan, "ci_hi": np.nan, "n": 0})
            continue
        boots = rng.choice(v, (n_boot, len(v))).mean(axis=1)
        out.append({"regime": reg, "mean": float(v.mean()), "median": float(np.median(v)),
                    "ci_lo": float(np.percentile(boots, 2.5)), "ci_hi": float(np.percentile(boots, 97.5)), "n": int(len(v))})
    tab = pd.DataFrame(out).set_index("regime")
    a = per_day.loc[per_day["in_force"] == 1, "ac1"]; c = per_day.loc[per_day["in_force"] == -1, "ac1"]
    p = float(mannwhitneyu(a, c, alternative="two-sided").pvalue) if len(a) > 2 and len(c) > 2 else np.nan
    return per_day, tab, p


# ----------------------------------------------------------------------------- misc
def json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (pd.Timestamp, dt.date, dt.datetime)):
        return pd.Timestamp(o).strftime("%Y-%m-%d")
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    if isinstance(o, float) and not np.isfinite(o):
        return None
    return str(o)


def save_json(obj, path):
    Path(path).write_text(json.dumps(obj, indent=2, default=json_default))
