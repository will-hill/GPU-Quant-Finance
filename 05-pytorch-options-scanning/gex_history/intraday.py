"""Intraday value-of-real-time tests for the GEX series. Reads the cache only; no network.

The real-time quantity a scanner produces is the previous close's book re-marked at the current
spot (open interest arrives once a day). These helpers build that re-mark from the daily profile
on a spot grid and test it against the 1-minute bars.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import polars as pl

import gexlib as G

REL = np.round(np.arange(0.90, 1.10 + 1e-9, 0.0025), 6)      # 81 spot points, -10% .. +10%


# ----------------------------------------------------------------------------- daily profiles
def daily_profiles(sol: pd.DataFrame, rel: np.ndarray = REL, chunk_days: int = 25, target: str = "cpu", dtype=np.float64, verbose: bool = True) -> pd.DataFrame:
    """Net signed GEX (USD per 1% move) of each day's solved chain re-evaluated on spot = rel x S,
    IV fixed. Index: date; columns: rel. Same construction as gexlib.flip_levels. target="cuda" with
    dtype=np.float32 is the GPU build; the engine's Engine.price copies host arrays per call."""
    from alo_numba import Engine, style_codes
    eng = Engine(dtype, target)
    ok = sol[sol["iv_status"] == 0]
    days = sorted(ok["date"].unique())
    m = len(rel)
    rows, idx = [], []
    t0 = time.perf_counter()
    for i in range(0, len(days), chunk_days):
        sub = ok[ok["date"].isin(days[i:i + chunk_days])]
        n = len(sub)
        S0 = sub["S"].to_numpy(np.float64)
        Sg = (S0[:, None] * rel[None, :]).ravel()
        rep = lambda col: np.repeat(sub[col].to_numpy(np.float64), m)  # noqa: E731
        style = np.repeat(style_codes(sub["is_call"].to_numpy(), False), m)
        gamma = eng.price(Sg, rep("strike"), rep("r"), rep("q"), rep("iv"), rep("T_years"), style)[:, 2].reshape(n, m)
        sign = np.where(sub["is_call"].to_numpy(), 1.0, -1.0)[:, None]
        gex = sign * gamma * sub["open_interest"].to_numpy(np.float64)[:, None] * 100.0 * Sg.reshape(n, m) ** 2 * 0.01
        codes, uniq = pd.factorize(sub["date"], sort=True)
        net = np.zeros((len(uniq), m))
        np.add.at(net, codes, gex)
        rows.append(net); idx.extend(uniq)
        if verbose:
            print(f"profiles: days {i + 1}-{min(i + chunk_days, len(days))} of {len(days)}, {time.perf_counter() - t0:.0f}s", flush=True)
    return pd.DataFrame(np.vstack(rows), index=pd.DatetimeIndex(idx, name="date"), columns=rel)


def remark(profiles: pd.DataFrame, S0: float, day_prev, spot) -> np.ndarray:
    """Re-marked net GEX at spot(s), from the profile of day_prev whose close was S0. Linear on the grid, clipped."""
    prof = profiles.loc[day_prev].to_numpy()
    rel = np.clip(np.asarray(spot, dtype=float) / S0, profiles.columns[0], profiles.columns[-1])
    return np.interp(rel, profiles.columns.to_numpy(dtype=float), prof)


# ----------------------------------------------------------------------------- intraday bars
def load_bars(symbol: str = "spy") -> pl.DataFrame:
    """Traded regular-session 1-minute bars: date, mod (minutes since midnight), open, high, low, close, volume."""
    files = sorted(G.CACHE.glob(f"{symbol.lower()}_1m_????-??.parquet"))
    if not files:
        raise FileNotFoundError(f"no {symbol} 1-minute cache")
    b = pl.concat([pl.read_parquet(f) for f in files], how="vertical_relaxed")
    if "close" not in b.columns and "price" in b.columns:          # index roots: one price per minute, no volume
        b = b.with_columns(pl.col("price").alias("close"), pl.col("price").alias("open"), pl.col("price").alias("high"),
                           pl.col("price").alias("low"), pl.lit(1).alias("volume"))
    return (b.with_columns(pl.col("timestamp").dt.date().alias("date"),
                           (pl.col("timestamp").dt.hour().cast(pl.Int32) * 60 + pl.col("timestamp").dt.minute().cast(pl.Int32)).alias("mod"))
             .filter((pl.col("mod") >= 570) & (pl.col("mod") <= 960) & (pl.col("volume") > 0) & pl.col("close").is_not_nan() & pl.col("close").is_not_null())
             .sort("timestamp").select("date", "mod", "open", "high", "low", "close", "volume"))


def bucketize(bars: pl.DataFrame, minutes: int) -> pd.DataFrame:
    """Per (date, bucket): first open, last close, high, low. Bucket 0 starts 09:30; the 16:00 print joins the last bucket."""
    nb = 390 // minutes
    b = (bars.with_columns(((pl.col("mod") - 570).clip(0, 389) // minutes).clip(0, nb - 1).alias("bucket"))
             .group_by(["date", "bucket"]).agg(pl.col("open").first(), pl.col("close").last(), pl.col("high").max(), pl.col("low").min(), pl.col("volume").sum())
             .sort(["date", "bucket"]).to_pandas())
    b["date"] = pd.to_datetime(b["date"])
    full = b.groupby("date")["bucket"].transform("size") == nb
    b = b[full].reset_index(drop=True)
    b["ret"] = np.log(b["close"] / b.groupby("date")["close"].shift(1))
    b.loc[b["bucket"] == 0, "ret"] = np.log(b.loc[b["bucket"] == 0, "close"] / b.loc[b["bucket"] == 0, "open"])
    return b


def attach_regime(b: pd.DataFrame, daily: pd.DataFrame, profiles: pd.DataFrame | None) -> pd.DataFrame:
    """Add the previous close's regime, GEX, walls, flip, S0, and the re-marked GEX at each bucket close."""
    prev = daily[["gex_net_usd", "regime", "close", "flip_level", "wall_pos_strike", "wall_neg_strike"]].shift(1)
    prev.columns = ["gex_prev", "regime_prev", "S0", "flip_prev", "wall_pos_prev", "wall_neg_prev"]
    prev["date_prev"] = pd.Series(daily.index, index=daily.index).shift(1)
    out = b.merge(prev, left_on="date", right_index=True, how="inner").dropna(subset=["regime_prev", "S0"])
    if profiles is not None:
        vals = np.full(len(out), np.nan)
        for dprev, g in out.groupby("date_prev"):
            if dprev in profiles.index:
                vals[g.index.to_numpy() if False else np.where(out["date_prev"].to_numpy() == dprev)[0]] = remark(profiles, float(g["S0"].iloc[0]), dprev, g["close"].to_numpy())
        out["gex_rt"] = vals
        out["regime_rt"] = np.sign(out["gex_rt"])
    out["range_b"] = (out["high"] - out["low"]) / out["S0"]
    out["absret_b"] = out["ret"].abs()
    med = out.groupby("bucket")["range_b"].transform("median")
    out["range_adj"] = out["range_b"] / med                       # removes the intraday U-shape
    out["next_range_adj"] = out.groupby("date")["range_adj"].shift(-1)
    out["next_absret"] = out.groupby("date")["absret_b"].shift(-1)
    out["next_ret"] = out.groupby("date")["ret"].shift(-1)
    return out.reset_index(drop=True)


# ----------------------------------------------------------------------------- tests
def boot_diff(a, b, n_boot=10000, seed=0):
    rng = np.random.default_rng(seed)
    d = rng.choice(a, (n_boot, len(a))).mean(1) - rng.choice(b, (n_boot, len(b))).mean(1)
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def t1_intraday_vol(bars: pl.DataFrame, daily: pd.DataFrame, horizons=(5, 15, 30, 60)) -> pd.DataFrame:
    """Per-day realized vol from intraday returns at each horizon, by the regime in force (previous close)."""
    from scipy.stats import mannwhitneyu
    rows = []
    inf = G.in_force(daily)
    for h in horizons:
        b = bucketize(bars, h)
        rv = b.groupby("date")["ret"].apply(lambda r: np.sqrt(np.nansum(r.to_numpy() ** 2)) * np.sqrt(252.0))
        rv = rv.to_frame("rv").join(inf.rename("in_force"), how="inner").dropna()
        a, c = rv.loc[rv["in_force"] == 1, "rv"].to_numpy(), rv.loc[rv["in_force"] == -1, "rv"].to_numpy()
        lo, hi = boot_diff(a, c)
        rows.append({"horizon_min": h, "n_pos": len(a), "n_neg": len(c), "rv_pos": a.mean(), "rv_neg": c.mean(), "median_pos": np.median(a), "median_neg": np.median(c),
                     "ratio_neg_over_pos": c.mean() / a.mean(), "diff": a.mean() - c.mean(), "ci_lo": lo, "ci_hi": hi, "p_mwu": float(mannwhitneyu(a, c).pvalue)})
    return pd.DataFrame(rows).set_index("horizon_min")


def t6_remark_tracks_next_print(daily: pd.DataFrame, profiles: pd.DataFrame) -> dict:
    """At the close of D: does the D-1 book re-marked at S_D predict the official GEX of D (new OI, new IV)
    better than the stale D-1 value?"""
    d = daily.copy()
    d["S0"] = d["close"].shift(1); d["gex_prev"] = d["gex_net_usd"].shift(1)
    d["date_prev"] = pd.Series(d.index, index=d.index).shift(1)
    rt = np.full(len(d), np.nan)
    for i, (dt_, row) in enumerate(d.iterrows()):
        if pd.notna(row["date_prev"]) and row["date_prev"] in profiles.index:
            rt[i] = remark(profiles, row["S0"], row["date_prev"], row["close"])[()]
    d["gex_rt_close"] = rt
    d = d.dropna(subset=["gex_rt_close", "gex_prev"])
    truth = np.sign(d["gex_net_usd"])
    out = {"n_days": int(len(d)),
           "sign_agreement_stale": float((np.sign(d["gex_prev"]) == truth).mean()),
           "sign_agreement_remark": float((np.sign(d["gex_rt_close"]) == truth).mean()),
           "corr_level_stale": float(np.corrcoef(d["gex_prev"], d["gex_net_usd"])[0, 1]),
           "corr_level_remark": float(np.corrcoef(d["gex_rt_close"], d["gex_net_usd"])[0, 1]),
           "mad_stale_bn": float((d["gex_prev"] - d["gex_net_usd"]).abs().median() / 1e9),
           "mad_remark_bn": float((d["gex_rt_close"] - d["gex_net_usd"]).abs().median() / 1e9),
           "days_sign_changed": int((np.sign(d["gex_prev"]) != truth).sum()),
           "remark_caught_change": float((np.sign(d.loc[np.sign(d["gex_prev"]) != truth, "gex_rt_close"]) == truth[np.sign(d["gex_prev"]) != truth]).mean())}
    return out


def t2_remark_vs_stale(ob: pd.DataFrame) -> dict:
    """Bucket level: does the re-marked GEX at the end of bucket b predict the next bucket's (seasonally
    adjusted) range better than the stale previous-close GEX? ob = attach_regime(bucketize(bars, 30), ...)."""
    from scipy.stats import mannwhitneyu, spearmanr
    d = ob.dropna(subset=["next_range_adj", "gex_rt", "gex_prev"]).copy()
    out = {"n_buckets": int(len(d)), "n_days": int(d["date"].nunique())}
    out["spearman_stale"] = float(spearmanr(d["gex_prev"], d["next_range_adj"]).correlation)
    out["spearman_remark"] = float(spearmanr(d["gex_rt"], d["next_range_adj"]).correlation)
    # within-day increment: the part of the re-mark that the stale value does not have
    d["d_gex"] = d["gex_rt"] - d["gex_prev"]
    d["resid_next"] = d["next_range_adj"] - d.groupby("date")["next_range_adj"].transform("mean")
    out["spearman_increment_within_day"] = float(spearmanr(d["d_gex"], d["resid_next"]).correlation)
    # sign flips inside the day
    flipped = d["regime_rt"] != d["regime_prev"]
    out["share_buckets_flipped"] = float(flipped.mean())
    out["share_days_with_any_flip"] = float(d.groupby("date").apply(lambda g: (g["regime_rt"] != g["regime_prev"]).any()).mean())
    res = {}
    for reg, lab in ((-1, "prev_negative"), (1, "prev_positive")):
        m = d["regime_prev"] == reg
        a = d.loc[m & flipped, "next_range_adj"].to_numpy(); b = d.loc[m & ~flipped, "next_range_adj"].to_numpy()
        if len(a) > 5 and len(b) > 5:
            lo, hi = boot_diff(a, b)
            res[lab] = {"n_flipped": int(len(a)), "n_same": int(len(b)), "next_range_adj_flipped": float(a.mean()), "next_range_adj_same": float(b.mean()),
                        "diff": float(a.mean() - b.mean()), "ci_lo": lo, "ci_hi": hi, "p_mwu": float(mannwhitneyu(a, b).pvalue)}
    out["flip_vs_same_by_prev_regime"] = res
    return out


def t3_intraday_flips(ob: pd.DataFrame) -> dict:
    """Day level: first bucket where the re-marked sign differs from the previous close's sign. Post-flip mean
    adjusted range on flip days against no-flip days with the same prior regime over the same buckets,
    unmatched and matched on the size of the move to the flip bucket (tercile within regime and bucket)."""
    d = ob.dropna(subset=["gex_rt"]).copy()
    d["move"] = (d["close"] / d["S0"] - 1.0).abs()
    days = []
    for dt_, g in d.groupby("date"):
        g = g.sort_values("bucket")
        fl = g.index[g["regime_rt"] != g["regime_prev"]]
        k = int(g.loc[fl[0], "bucket"]) if len(fl) else None
        days.append({"date": dt_, "regime_prev": int(g["regime_prev"].iloc[0]), "flip_bucket": k,
                     "move_at_flip": float(g.loc[fl[0], "move"]) if len(fl) else np.nan})
    days = pd.DataFrame(days).set_index("date")
    nb = int(d["bucket"].max()) + 1
    out = {"n_days": int(len(days)), "share_days_flipped": float(days["flip_bucket"].notna().mean()),
           "flip_bucket_distribution": days["flip_bucket"].value_counts().sort_index().to_dict()}
    res = {}
    for reg, lab in ((-1, "neg_to_pos"), (1, "pos_to_neg")):
        fd = days[(days["regime_prev"] == reg) & days["flip_bucket"].notna() & (days["flip_bucket"] <= nb - 3)]
        nf = days[(days["regime_prev"] == reg) & days["flip_bucket"].isna()]
        diffs_u, diffs_m = [], []
        for dt_, r in fd.iterrows():
            k = int(r["flip_bucket"])
            post = d[(d["date"] == dt_) & (d["bucket"] > k)]["range_adj"].mean()
            ref_days = nf.index
            ref = d[d["date"].isin(ref_days) & (d["bucket"] > k)]
            diffs_u.append(post - ref["range_adj"].mean())
            # matched: no-flip days whose move by bucket k is in the same tercile as this day's move
            mv = d[d["date"].isin(ref_days) & (d["bucket"] == k)].set_index("date")["move"]
            if len(mv) >= 9:
                q = pd.qcut(mv, 3, labels=False, duplicates="drop")
                tgt = pd.cut([r["move_at_flip"]], bins=np.quantile(mv, [0, 1/3, 2/3, 1.0]), labels=False, include_lowest=True)[0]
                sel = mv.index[q == tgt] if not np.isnan(tgt) else mv.index
                refm = d[d["date"].isin(sel) & (d["bucket"] > k)]
                if len(refm):
                    diffs_m.append(post - refm["range_adj"].mean())
        du, dm = np.array(diffs_u), np.array(diffs_m)
        rng = np.random.default_rng(0)
        ci = lambda x: (float(np.percentile(rng.choice(x, (10000, len(x))).mean(1), 2.5)), float(np.percentile(rng.choice(x, (10000, len(x))).mean(1), 97.5))) if len(x) > 5 else (np.nan, np.nan)  # noqa: E731
        res[lab] = {"n_flip_days": int(len(fd)), "n_noflip_days": int(len(nf)),
                    "post_flip_minus_reference_unmatched": float(du.mean()) if len(du) else np.nan, "ci_unmatched": ci(du),
                    "post_flip_minus_reference_matched": float(dm.mean()) if len(dm) else np.nan, "ci_matched": ci(dm), "n_matched": int(len(dm))}
    out["by_direction"] = res
    return out, days


def t4_wall_touches(bars: pl.DataFrame, daily: pd.DataFrame, tol: float = 0.0005, horizons=(30, 60), placebo_offsets=(5.0, -5.0)) -> dict:
    """First intraday touch of the previous close's call wall (from below) and put wall (from above): signed
    return over the next `horizons` minutes, against touches of placebo levels wall +/- 5 dollars."""
    b = bars.to_pandas(); b["date"] = pd.to_datetime(b["date"])
    prev = daily[["regime", "close", "wall_pos_strike", "wall_neg_strike"]].shift(1)
    prev.columns = ["regime_prev", "S0", "wall_pos", "wall_neg"]
    b = b.merge(prev, left_on="date", right_index=True, how="inner").dropna(subset=["S0", "wall_pos", "wall_neg"])
    by_day = {d: g.sort_values("mod").reset_index(drop=True) for d, g in b.groupby("date")}

    def touch_stats(level_col_fn, side):
        rows = []
        for d, g in by_day.items():
            lvl = level_col_fn(g.iloc[0])
            S0 = g["S0"].iloc[0]
            if side == "call" and not (lvl > S0 * (1 + tol)):
                continue
            if side == "put" and not (lvl < S0 * (1 - tol)):
                continue
            hit = (g["high"] >= lvl * (1 - tol)) if side == "call" else (g["low"] <= lvl * (1 + tol))
            if not hit.any():
                rows.append({"date": d, "touched": False, "regime_prev": g["regime_prev"].iloc[0], "dist": abs(lvl / S0 - 1)}); continue
            i = int(np.argmax(hit.to_numpy()))
            c0 = g.loc[i, "close"]; t0 = g.loc[i, "mod"]
            rec = {"date": d, "touched": True, "regime_prev": g["regime_prev"].iloc[0], "dist": abs(lvl / S0 - 1), "mod": t0}
            for h in horizons:
                later = g[g["mod"] >= t0 + h]
                rec[f"ret_{h}"] = np.log(later["close"].iloc[0] / c0) if len(later) else np.nan
            rows.append(rec)
        return pd.DataFrame(rows)

    out = {}
    for side, col, sgn in (("call", "wall_pos", -1.0), ("put", "wall_neg", 1.0)):
        real = touch_stats(lambda r, col=col: r[col], side)
        out[side] = {"n_days_wall_beyond_spot": int(len(real)), "touch_rate": float(real["touched"].mean()),
                     "touch_rate_by_prev_regime": real.groupby("regime_prev")["touched"].mean().round(3).to_dict()}
        for h in horizons:
            r = real.loc[real["touched"], f"ret_{h}"].dropna() * 1e4
            out[side][f"after_touch_{h}m_bp_mean"] = float(r.mean()); out[side][f"after_touch_{h}m_share_reversal"] = float((np.sign(r) == sgn).mean()); out[side][f"n_touch_{h}m"] = int(len(r))
            rng = np.random.default_rng(0); bs = rng.choice(r.to_numpy(), (10000, len(r))).mean(1)
            out[side][f"after_touch_{h}m_ci"] = (float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)))
        plc = {}
        for off in placebo_offsets:
            p = touch_stats(lambda r, col=col, off=off: r[col] + off, side)
            plc[f"wall{off:+.0f}"] = {"touch_rate": float(p["touched"].mean()),
                                     **{f"after_touch_{h}m_bp_mean": float((p.loc[p["touched"], f"ret_{h}"].dropna() * 1e4).mean()) for h in horizons},
                                     **{f"after_touch_{h}m_share_reversal": float((np.sign(p.loc[p["touched"], f"ret_{h}"].dropna()) == sgn).mean()) for h in horizons}}
        out[side]["placebo"] = plc
    # range bounded by both walls
    day = b.groupby("date").agg(high=("high", "max"), low=("low", "min"), wall_pos=("wall_pos", "first"), wall_neg=("wall_neg", "first"), regime_prev=("regime_prev", "first"))
    inside = (day["high"] < day["wall_pos"]) & (day["low"] > day["wall_neg"])
    out["day_inside_both_walls_share"] = float(inside.mean())
    out["day_inside_both_walls_by_prev_regime"] = inside.groupby(day["regime_prev"]).mean().round(3).to_dict()
    return out


def t5_time_of_day(ob: pd.DataFrame) -> pd.DataFrame:
    """Mean raw range (% of S0) per 30-minute bucket by the regime in force, and the ratio."""
    g = ob.groupby(["bucket", "regime_prev"])["range_b"].mean().unstack("regime_prev") * 100.0
    g.columns = [("pos" if c > 0 else "neg") for c in g.columns]
    g["ratio_neg_over_pos"] = g["neg"] / g["pos"]
    return g


def t3b_near_flip(ob: pd.DataFrame, days: pd.DataFrame, near_pct: float = 0.5) -> dict:
    """Control for the starting distance: among days whose previous close was within `near_pct` % of the
    flip level (the stale print already says 'near the boundary'), compare the rest-of-day adjusted range
    on days that crossed intraday against days that did not, over the same buckets after the median
    flip bucket of the crossers. Also a joint rank regression of the next bucket's range on the stale
    and re-marked GEX."""
    import statsmodels.api as sm
    d = ob.dropna(subset=["gex_rt"]).copy()
    dist0 = ((d.groupby("date")["S0"].first() / d.groupby("date")["flip_prev"].first()) - 1.0).abs() * 100.0
    near = dist0[dist0 <= near_pct].index
    out = {"near_pct": near_pct, "n_near_days": int(len(near)), "n_days": int(dist0.notna().sum())}
    res = {}
    for reg, lab in ((-1, "neg_to_pos"), (1, "pos_to_neg")):
        dd = days.loc[days.index.isin(near) & (days["regime_prev"] == reg)]
        cross = dd[dd["flip_bucket"].notna() & (dd["flip_bucket"] <= 9)]
        stay = dd[dd["flip_bucket"].isna()]
        k = int(cross["flip_bucket"].median()) if len(cross) else 0
        post = lambda idx: d[d["date"].isin(idx) & (d["bucket"] > k)].groupby("date")["range_adj"].mean()  # noqa: E731
        a, b = post(cross.index).to_numpy(), post(stay.index).to_numpy()
        if len(a) > 5 and len(b) > 5:
            from scipy.stats import mannwhitneyu
            lo, hi = boot_diff(a, b)
            res[lab] = {"n_cross": int(len(a)), "n_stay": int(len(b)), "median_flip_bucket": k, "post_range_adj_cross": float(a.mean()), "post_range_adj_stay": float(b.mean()),
                        "diff": float(a.mean() - b.mean()), "ci_lo": lo, "ci_hi": hi, "p_mwu": float(mannwhitneyu(a, b).pvalue)}
        else:
            res[lab] = {"n_cross": int(len(a)), "n_stay": int(len(b))}
    out["near_flip_cross_vs_stay"] = res
    # joint regression on ranks (pooled buckets): next_range_adj ~ rank(gex_prev) + rank(gex_rt)
    dd = d.dropna(subset=["next_range_adj", "gex_prev", "gex_rt"])
    X = pd.DataFrame({"stale_rank": dd["gex_prev"].rank(pct=True), "remark_rank": dd["gex_rt"].rank(pct=True)})
    X = sm.add_constant(X)
    fit = sm.OLS(dd["next_range_adj"].to_numpy(), X).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(dd["date"])[0]})
    out["joint_regression"] = {"coef_stale": float(fit.params["stale_rank"]), "t_stale": float(fit.tvalues["stale_rank"]),
                               "coef_remark": float(fit.params["remark_rank"]), "t_remark": float(fit.tvalues["remark_rank"]),
                               "r2": float(fit.rsquared), "n": int(fit.nobs), "se_clustered_by_day": True}
    fit1 = sm.OLS(dd["next_range_adj"].to_numpy(), sm.add_constant(X[["stale_rank"]])).fit()
    fit2 = sm.OLS(dd["next_range_adj"].to_numpy(), sm.add_constant(X[["remark_rank"]])).fit()
    out["r2_stale_only"] = float(fit1.rsquared); out["r2_remark_only"] = float(fit2.rsquared)
    return out


def t7_model_choice(sol: pd.DataFrame, daily: pd.DataFrame) -> dict:
    """Accuracy: re-solve the same chain with European exercise (Black-Scholes-Merton) and compare the daily net
    GEX sign and flip-level location against the American reference."""
    from alo_numba import Engine, Tables, build_iv_cpu, make_core, style_codes
    ok = sol[sol["iv_status"] == 0].reset_index(drop=True)
    f = np.float64
    n = len(ok)
    S = ok["S"].to_numpy(f); K = ok["strike"].to_numpy(f); T = ok["T_years"].to_numpy(f); mid = ok["mid"].to_numpy(f)
    r = ok["r"].to_numpy(f); q = ok["q"].to_numpy(f)
    style_eu = style_codes(ok["is_call"].to_numpy(), True)
    tab = Tables(f, 7, 7, 27); core = make_core(f, "cpu", 7, 7, 27, m_iter=4); eng = Engine(f, "cpu")
    run_iv = build_iv_cpu(core, f, evals=12)
    iv = np.zeros(n, dtype=f); sig0 = np.full(n, 0.3, dtype=f)
    t0 = time.perf_counter()
    run_iv(mid, S, K, r, q, T, style_eu, sig0, *tab.args(), tab.scratch(n), iv)
    t_iv = time.perf_counter() - t0
    solved = np.isfinite(iv) & (iv > 1.001e-3) & (iv < 5.99)
    out_eu = eng.price(S, K, r, q, np.where(solved, iv, 1e-3), T, style_eu)
    gamma_eu = out_eu[:, 2]
    sign = np.where(ok["is_call"], 1.0, -1.0)
    gex_eu = np.where(solved, sign * gamma_eu * ok["open_interest"].to_numpy(f) * 100.0 * S * S * 0.01, 0.0)
    day_eu = pd.Series(gex_eu).groupby(ok["date"].to_numpy()).sum()
    day_am = daily["gex_net_usd"].reindex(day_eu.index)
    rel = (ok["gamma"].to_numpy() / np.where(gamma_eu > 0, gamma_eu, np.nan))
    return {"n_contracts": int(n), "european_iv_unsolved_share": float(1.0 - solved.mean()), "t_iv_european_s": round(t_iv, 2),
            "sign_agreement_daily": float((np.sign(day_eu) == np.sign(day_am)).mean()),
            "days_sign_differs": int((np.sign(day_eu) != np.sign(day_am)).sum()),
            "median_abs_diff_bn": float((day_eu - day_am).abs().median() / 1e9),
            "median_rel_gamma_american_over_european": float(np.nanmedian(rel)),
            "p10_p90_rel_gamma": (float(np.nanpercentile(rel, 10)), float(np.nanpercentile(rel, 90))),
            "put_itm_median_ratio": float(np.nanmedian(rel[(~ok["is_call"].to_numpy()) & (K > S)])),
            "call_itm_median_ratio": float(np.nanmedian(rel[(ok["is_call"].to_numpy()) & (K < S)])),
            "otm_median_ratio": float(np.nanmedian(rel[((~ok["is_call"].to_numpy()) & (K < S)) | ((ok["is_call"].to_numpy()) & (K > S))]))}


# ----------------------------------------------------------------------------- figures
def fig_remark_vs_stale(daily: pd.DataFrame, profiles: pd.DataFrame, t6: dict, path=None):
    """Two scatters against the official next print: stale previous-close GEX and the re-marked book."""
    import matplotlib.pyplot as plt
    G.style()
    d = daily.copy()
    d["S0"] = d["close"].shift(1); d["gex_prev"] = d["gex_net_usd"].shift(1)
    d["date_prev"] = pd.Series(d.index, index=d.index).shift(1)
    rt = np.full(len(d), np.nan)
    for i, (_, row) in enumerate(d.iterrows()):
        if pd.notna(row["date_prev"]) and row["date_prev"] in profiles.index:
            rt[i] = remark(profiles, row["S0"], row["date_prev"], row["close"])[()]
    d["gex_rt"] = rt
    d = d.dropna(subset=["gex_rt", "gex_prev"])
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.6), gridspec_kw={"wspace": 0.3})
    lim = [d["gex_net_usd"].min() / 1e9 * 1.05, d["gex_net_usd"].max() / 1e9 * 1.05]
    for ax, col, c, title, agree, corr in ((axes[0], "gex_prev", G.DIM, "Stale: yesterday's print", t6["sign_agreement_stale"], t6["corr_level_stale"]),
                                            (axes[1], "gex_rt", G.CYAN, "Re-marked at today's close", t6["sign_agreement_remark"], t6["corr_level_remark"])):
        ax.plot(lim, lim, color=G.DIM2, lw=1.0, ls="--")
        ax.axhline(0, color=G.DIM2, lw=0.6); ax.axvline(0, color=G.DIM2, lw=0.6)
        ax.scatter(d[col] / 1e9, d["gex_net_usd"] / 1e9, s=10, color=c, alpha=0.65, lw=0)
        ax.set_xlabel(f"{'stale' if col == 'gex_prev' else 're-marked'} GEX, $bn per 1%")
        ax.set_ylabel("official GEX at today's close, $bn per 1%")
        ax.set_title(title, loc="left", fontsize=14, pad=20)
        ax.text(0.0, 1.01, f"sign agreement {agree:.1%}   level correlation {corr:.2f}", transform=ax.transAxes, color=G.FG, fontsize=10.5, va="bottom")
        ax.set_xlim(lim); ax.set_ylim(lim)
    if path:
        fig.savefig(path, bbox_inches="tight", pad_inches=0.25)
    return fig


def fig_intraday_flip(ob: pd.DataFrame, days: pd.DataFrame, t3b: dict, path=None):
    """Left: mean adjusted range in the buckets after an intraday flip against no-flip days, by direction.
    Right: the near-flip control (days that started within 0.5% of the flip) with bootstrap CIs."""
    import matplotlib.pyplot as plt
    G.style()
    d = ob.dropna(subset=["gex_rt"])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 4.6), gridspec_kw={"wspace": 0.36, "width_ratios": [1.3, 1.0]})
    for reg, lab, c in ((-1, "negative at prior close, flipped positive", G.CYAN), (1, "positive at prior close, flipped negative", G.ORANGE)):
        fd = days[(days["regime_prev"] == reg) & days["flip_bucket"].notna()]
        nf = days[(days["regime_prev"] == reg) & days["flip_bucket"].isna()]
        rel = []
        for dt_, r in fd.iterrows():
            k = int(r["flip_bucket"])
            g = d[d["date"] == dt_].set_index("bucket")["range_adj"]
            rel.append(pd.Series(g.values, index=g.index - k))
        ev = pd.concat(rel, axis=1).mean(axis=1)
        ev = ev[(ev.index >= -3) & (ev.index <= 6)]
        ref = d[d["date"].isin(nf.index)]["range_adj"].mean()
        ax1.plot(ev.index, ev.values, marker="o", color=c, lw=2.0, label=f"{lab} (n = {len(fd)})")
        ax1.axhline(ref, color=c, lw=1.0, ls=":", alpha=0.9)
    ax1.axvline(0, color=G.PURPLE, lw=1.2, ls="--")
    ax1.text(0.0, 1.01, "dashed: crossing bucket\ndotted: mean of same-regime days that never crossed", transform=ax1.transAxes, va="bottom", color=G.DIM, fontsize=8.5)
    ax1.set_xlabel("30-minute buckets relative to the crossing")
    ax1.set_ylabel("range, multiple of the time-of-day median")
    ax1.set_title("Range before and after an intraday flip", loc="left", fontsize=14, pad=30)
    ax1.legend(loc="upper right", fontsize=8.5)
    r = t3b["near_flip_cross_vs_stay"]
    labels, vals, los, his, cols = [], [], [], [], []
    for key, lab, c in (("neg_to_pos", "neg start", G.CYAN), ("pos_to_neg", "pos start", G.ORANGE)):
        x = r[key]
        labels += [f"{lab}\ncrossed\nn = {x['n_cross']}", f"{lab}\nstayed\nn = {x['n_stay']}"]
        vals += [x["post_range_adj_cross"], x["post_range_adj_stay"]]
        cols += [c, G.DIM2]
    xs = np.arange(len(vals))
    ax2.bar(xs, vals, color=cols, width=0.7, lw=0)
    ax2.set_xticks(xs); ax2.set_xticklabels(labels, fontsize=9)
    ax2.set_ylabel("rest-of-day range, multiple of median")
    ax2.set_title(f"Started within {t3b['near_pct']}% of the flip", loc="left", fontsize=14, pad=30)
    ax2.text(0.0, 1.01, "crossed: re-marked sign flipped intraday\nstayed: it did not", transform=ax2.transAxes, va="bottom", color=G.DIM, fontsize=8.5)
    for key, i in (("neg_to_pos", 0.5), ("pos_to_neg", 2.5)):
        x = r[key]
        ax2.text(i, max(vals) * 1.04, f"diff {x['diff']:+.2f}\nCI [{x['ci_lo']:+.2f}, {x['ci_hi']:+.2f}]", ha="center", va="bottom", color=G.FG, fontsize=9)
    ax2.set_ylim(0, max(vals) * 1.3)
    if path:
        fig.savefig(path, bbox_inches="tight", pad_inches=0.25)
    return fig


def fig_intraday_vol(t1: pd.DataFrame, path=None):
    import matplotlib.pyplot as plt
    G.style()
    fig, ax = plt.subplots(figsize=(G.FIG_W, G.FIG_H))
    xs = np.arange(len(t1)); w = 0.36
    ax.bar(xs - w / 2, t1["rv_pos"] * 100, width=w, color=G.CYAN, lw=0, label=f"positive GEX in force, n = {int(t1['n_pos'].iloc[0])} days")
    ax.bar(xs + w / 2, t1["rv_neg"] * 100, width=w, color=G.ORANGE, lw=0, label=f"negative GEX in force, n = {int(t1['n_neg'].iloc[0])} days")
    for i, (h, r) in enumerate(zip(t1.index, t1["ratio_neg_over_pos"])):
        ax.text(i, max(t1["rv_pos"].iloc[i], t1["rv_neg"].iloc[i]) * 100 + 0.3, f"x{r:.2f}", ha="center", color=G.FG, fontsize=11)
    ax.set_xticks(xs); ax.set_xticklabels([f"{h}-minute returns" for h in t1.index])
    ax.set_ylabel("mean realized vol from intraday returns, annualized %")
    ax.set_title("The regime is visible at every intraday horizon", loc="left", pad=22)
    ax.text(0.0, 1.015, "per-day realized vol from intraday log returns, averaged by the regime at the previous close; label: negative over positive", transform=ax.transAxes, color=G.DIM, fontsize=10, va="bottom")
    ax.set_ylim(0, max(t1["rv_neg"].max(), t1["rv_pos"].max()) * 100 * 1.22)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2, fontsize=10)
    if path:
        fig.savefig(path, bbox_inches="tight", pad_inches=0.25)
    return fig


def fig_time_of_day(t5: pd.DataFrame, path=None):
    import matplotlib.pyplot as plt
    G.style()
    fig, ax = plt.subplots(figsize=(G.FIG_W, G.FIG_H))
    xs = np.arange(len(t5)); w = 0.4
    ax.bar(xs - w / 2, t5["pos"], width=w, color=G.CYAN, lw=0, label="positive GEX in force")
    ax.bar(xs + w / 2, t5["neg"], width=w, color=G.ORANGE, lw=0, label="negative GEX in force")
    labels = [f"{(570 + 30 * int(b)) // 60:02d}:{(570 + 30 * int(b)) % 60:02d}" for b in t5.index]
    ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=9)
    ax.set_xlabel("30-minute bucket start, ET")
    ax.set_ylabel("mean range in the bucket, % of prior close")
    ax.set_title("Regime effect by time of day", loc="left", pad=22)
    ax.text(0.0, 1.015, f"negative over positive ratio: min {t5['ratio_neg_over_pos'].min():.2f}, max {t5['ratio_neg_over_pos'].max():.2f}; widest bucket is the open in both regimes", transform=ax.transAxes, color=G.DIM, fontsize=10, va="bottom")
    ax.legend(loc="upper right", fontsize=10)
    if path:
        fig.savefig(path, bbox_inches="tight", pad_inches=0.25)
    return fig


def fig_wall_touches(t4: dict, path=None):
    import matplotlib.pyplot as plt
    G.style()
    fig, ax = plt.subplots(figsize=(G.FIG_W, G.FIG_H))
    items = []
    for side, lab, c in (("call", "call wall touched from below", G.CYAN), ("put", "put wall touched from above", G.ORANGE)):
        x = t4[side]
        items.append((f"{side} wall\nn = {x['n_touch_30m']}", x["after_touch_30m_bp_mean"], x["after_touch_30m_ci"], c))
        for k, p in x["placebo"].items():
            items.append((f"{side} placebo\n{k.replace('wall', 'wall ')}", p["after_touch_30m_bp_mean"], None, G.DIM2))
    xs = np.arange(len(items))
    ax.bar(xs, [v for _, v, _, _ in items], color=[c for *_, c in items], width=0.65, lw=0)
    for i, (_, v, ci, _) in enumerate(items):
        if ci:
            ax.errorbar(i, v, yerr=[[v - ci[0]], [ci[1] - v]], fmt="none", ecolor=G.FG, elinewidth=1.6, capsize=6)
    ax.axhline(0, color=G.DIM2, lw=0.8)
    ax.set_xticks(xs); ax.set_xticklabels([t for t, *_ in items], fontsize=9.5)
    ax.set_ylabel("mean return in the 30 minutes after the first touch, bp")
    ax.set_title("Walls as intraday support and resistance: no effect", loc="left", pad=22)
    ax.text(0.0, 1.015, "whiskers: bootstrap 95% CI; placebo levels are the wall shifted by 5 dollars; a resistance effect would be negative for calls, positive for puts", transform=ax.transAxes, color=G.DIM, fontsize=9.5, va="bottom")
    if path:
        fig.savefig(path, bbox_inches="tight", pad_inches=0.25)
    return fig


# ----------------------------------------------------------------------------- the 0DTE layer
def zero_dte_layer(sol: pd.DataFrame, daily: pd.DataFrame, ob: pd.DataFrame, symbol: str = "spy", band: float = 0.03, minutes: int = 30) -> pd.DataFrame:
    """Same-day-expiry contracts re-marked intraday with the true remaining time.

    For day D: open interest from the OI file dated D (positions as of the D-1 close, the freshest
    available during D), IV from the D-1 solve of the same contracts (1 day to expiry then); strikes
    without a solved IV take the nearest solved strike of the same right within `band` of the prior
    close. Gamma at each bucket close with T = time to 16:00. Returns per (date, bucket): signed
    net 0DTE GEX, unsigned 0DTE gamma exposure, the standing-book re-mark, and the max-exposure strike.
    Positions opened during D are invisible (OI is once a day), so this is a lower bound on 0DTE gamma."""
    from alo_numba import Engine, style_codes
    sym = symbol.lower()
    days = list(daily.index)
    prev = {days[i]: days[i - 1] for i in range(1, len(days))}
    rows = []
    for d in days[1:]:
        dp = prev[d]
        oi = pl.read_parquet(G.CACHE / f"{sym}_oi_{d.date()}.parquet").filter(pl.col("expiration") == str(d.date())).with_columns(pl.col("right").str.slice(0, 1))
        oi = oi.sort("timestamp").unique(subset=["strike", "right"], keep="last").select("strike", "right", "open_interest").to_pandas()
        oi = oi[oi["open_interest"] > 0]
        if oi.empty:
            continue
        iv = sol[(sol["date"] == dp) & (sol["expiration"] == str(d.date())) & (sol["iv_status"] == 0)][["strike", "right", "iv", "r", "q"]]
        if iv.empty:
            continue
        S0 = float(daily.loc[dp, "close"])
        oi = oi[(oi["strike"] / S0 - 1).abs() <= band]
        m = oi.merge(iv, on=["strike", "right"], how="left")
        for right in ("C", "P"):
            src = iv[iv["right"] == right].sort_values("strike")
            need = m["iv"].isna() & (m["right"] == right)
            if need.any() and len(src):
                idx = np.abs(src["strike"].to_numpy()[None, :] - m.loc[need, "strike"].to_numpy()[:, None]).argmin(axis=1)
                m.loc[need, "iv"] = src["iv"].to_numpy()[idx]; m.loc[need, "r"] = src["r"].iloc[0]; m.loc[need, "q"] = src["q"].iloc[0]
        m = m.dropna(subset=["iv"])
        m["date"] = d
        rows.append(m)
    con = pd.concat(rows, ignore_index=True)
    # cross with the day's bucket closes
    b = ob[ob["date"].isin(con["date"].unique())][["date", "bucket", "close", "gex_rt", "gex_prev", "regime_prev", "range_adj", "next_range_adj"]]
    x = con.merge(b, on="date", how="inner")
    mod_close = 570 + minutes * (x["bucket"].to_numpy() + 1)
    T = np.maximum((960 - mod_close) / (60.0 * 24.0 * 365.25), 1e-9)      # 0 at the 16:00 print: expired
    eng = Engine(np.float64, "cpu")
    is_call = (x["right"] == "C").to_numpy()
    out = eng.price(x["close"].to_numpy(np.float64), x["strike"].to_numpy(np.float64), x["r"].to_numpy(np.float64), x["q"].to_numpy(np.float64),
                    x["iv"].to_numpy(np.float64), T, style_codes(is_call, False))
    x["gamma"] = out[:, 2]
    x["gex"] = np.where(is_call, 1.0, -1.0) * x["gamma"] * x["open_interest"] * x["close"] ** 2
    x["ugex"] = x["gamma"] * x["open_interest"] * x["close"] ** 2
    g = x.groupby(["date", "bucket"])
    layer = g.agg(gex_0dte=("gex", "sum"), ugex_0dte=("ugex", "sum"), n_0dte=("gex", "size"), gex_rt=("gex_rt", "first"), gex_prev=("gex_prev", "first"),
                  regime_prev=("regime_prev", "first"), close=("close", "first"), range_adj=("range_adj", "first"), next_range_adj=("next_range_adj", "first")).reset_index()
    kstar = x.loc[x.groupby(["date", "bucket"])["ugex"].idxmax(), ["date", "bucket", "strike"]].rename(columns={"strike": "k_star_0dte"})
    layer = layer.merge(kstar, on=["date", "bucket"], how="left")
    layer["share_0dte_of_book"] = layer["ugex_0dte"] / (layer["ugex_0dte"] + layer["gex_rt"].abs())
    return layer


def t8_zero_dte(layer: pd.DataFrame) -> dict:
    """(a) how big the 0DTE layer is against the standing book by time of day; (b) does adding it improve the
    next-bucket range prediction; (c) the last half hour: range of the 15:30 to 16:00 bucket by the 0DTE
    exposure at 15:30 (terciles) within each prior regime."""
    import statsmodels.api as sm
    from scipy.stats import mannwhitneyu, spearmanr
    d = layer.dropna(subset=["gex_rt"]).copy()
    out = {"n_day_buckets": int(len(d)), "n_days": int(d["date"].nunique()), "median_contracts_per_day": float(d.groupby("date")["n_0dte"].first().median())}
    by_b = d.groupby("bucket").agg(ugex_0dte_bn=("ugex_0dte", lambda v: float(np.median(v) / 1e9)), book_abs_bn=("gex_rt", lambda v: float(np.median(np.abs(v)) / 1e9)),
                                   share_0dte=("share_0dte_of_book", "median"), net_0dte_positive_share=("gex_0dte", lambda v: float((v > 0).mean())))
    out["by_bucket"] = by_b.round(4).reset_index().to_dict("records")
    dd = d.dropna(subset=["next_range_adj"])
    dd = dd[dd["bucket"] < 12]
    out["spearman_next_range"] = {"stale": float(spearmanr(dd["gex_prev"], dd["next_range_adj"]).correlation),
                                  "remark": float(spearmanr(dd["gex_rt"], dd["next_range_adj"]).correlation),
                                  "remark_plus_0dte_signed": float(spearmanr(dd["gex_rt"] + dd["gex_0dte"], dd["next_range_adj"]).correlation),
                                  "unsigned_0dte_alone": float(spearmanr(dd["ugex_0dte"], dd["next_range_adj"]).correlation),
                                  "signed_0dte_alone": float(spearmanr(dd["gex_0dte"], dd["next_range_adj"]).correlation)}
    X = sm.add_constant(pd.DataFrame({"remark_rank": dd["gex_rt"].rank(pct=True), "u0dte_rank": dd["ugex_0dte"].rank(pct=True), "s0dte_rank": dd["gex_0dte"].rank(pct=True)}))
    fit = sm.OLS(dd["next_range_adj"].to_numpy(), X).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(dd["date"])[0]})
    out["joint_regression_t"] = {k: float(v) for k, v in fit.tvalues.items() if k != "const"}
    out["joint_regression_coef"] = {k: float(v) for k, v in fit.params.items() if k != "const"}
    last = d[d["bucket"] == 11].dropna(subset=["next_range_adj"])          # 15:00-15:30 close -> outcome is the 15:30-16:00 bucket
    res = {}
    for reg, lab in ((1, "prev_positive"), (-1, "prev_negative"), (0, "all")):
        m = last if reg == 0 else last[last["regime_prev"] == reg]
        if len(m) < 30:
            continue
        q = pd.qcut(m["ugex_0dte"].rank(method="first"), 3, labels=False)
        lo_, hi_ = m.loc[q == 0, "next_range_adj"].to_numpy(), m.loc[q == 2, "next_range_adj"].to_numpy()
        ci = boot_diff(hi_, lo_)
        res[lab] = {"n": int(len(m)), "last_half_hour_range_adj_low_0dte": float(lo_.mean()), "high_0dte": float(hi_.mean()), "diff_high_minus_low": float(hi_.mean() - lo_.mean()),
                    "ci_lo": ci[0], "ci_hi": ci[1], "p_mwu": float(mannwhitneyu(hi_, lo_).pvalue),
                    "spearman_ugex_vs_last_range": float(spearmanr(m["ugex_0dte"], m["next_range_adj"]).correlation),
                    "spearman_signed_vs_last_range": float(spearmanr(m["gex_0dte"], m["next_range_adj"]).correlation)}
    out["last_half_hour_by_0dte_exposure_at_1530"] = res
    return out


def t8_controls(layer: pd.DataFrame, sol: pd.DataFrame, daily: pd.DataFrame, symbol: str = "spy") -> dict:
    """Controls for the last-half-hour result: (a) the day's own realized range so far (mean adjusted range of
    buckets 0..11) and the prior regime in a joint rank regression; (b) how much same-day OI was added on the
    day before expiry (OI dated D over OI dated D-1 for the same expiry), the part of 0DTE positioning the
    once-a-day OI does see."""
    import statsmodels.api as sm
    d = layer.dropna(subset=["gex_rt"]).copy()
    day_sofar = d[d["bucket"] <= 11].groupby("date")["range_adj"].mean().rename("range_sofar")
    last = d[d["bucket"] == 11].dropna(subset=["next_range_adj"]).merge(day_sofar, left_on="date", right_index=True)
    X = pd.DataFrame({"u0dte_rank": last["ugex_0dte"].rank(pct=True), "range_sofar_rank": last["range_sofar"].rank(pct=True),
                      "book_rank": last["gex_rt"].rank(pct=True), "prev_negative": (last["regime_prev"] < 0).astype(float)})
    fit = sm.OLS(last["next_range_adj"].to_numpy(), sm.add_constant(X)).fit(cov_type="HC1")
    out = {"n_days": int(len(last)), "last_half_hour_controlled_t": {k: float(v) for k, v in fit.tvalues.items() if k != "const"},
           "last_half_hour_controlled_coef": {k: float(v) for k, v in fit.params.items() if k != "const"}, "r2": float(fit.rsquared)}
    # within-day-vol strata: terciles of range so far, then high vs low 0DTE exposure inside each stratum
    strata = {}
    q_sofar = pd.qcut(last["range_sofar"].rank(method="first"), 3, labels=["calm so far", "middle", "wide so far"])
    for lab in ("calm so far", "middle", "wide so far"):
        m = last[q_sofar == lab]
        q = pd.qcut(m["ugex_0dte"].rank(method="first"), 3, labels=False)
        lo_, hi_ = m.loc[q == 0, "next_range_adj"].to_numpy(), m.loc[q == 2, "next_range_adj"].to_numpy()
        ci = boot_diff(hi_, lo_)
        strata[lab] = {"n": int(len(m)), "low_0dte": float(lo_.mean()), "high_0dte": float(hi_.mean()), "diff": float(hi_.mean() - lo_.mean()), "ci_lo": ci[0], "ci_hi": ci[1]}
    out["last_half_hour_within_day_vol_strata"] = strata
    # OI added on the last day before expiry
    sym = symbol.lower()
    days = list(daily.index)
    ratios = []
    for i in range(1, len(days)):
        d0, dp = days[i], days[i - 1]
        try:
            a = pl.read_parquet(G.CACHE / f"{sym}_oi_{d0.date()}.parquet").filter(pl.col("expiration") == str(d0.date()))["open_interest"].sum()
            b = pl.read_parquet(G.CACHE / f"{sym}_oi_{dp.date()}.parquet").filter(pl.col("expiration") == str(d0.date()))["open_interest"].sum()
        except Exception:  # noqa: BLE001
            continue
        if b and a:
            ratios.append(a / b)
    r = np.array(ratios)
    out["same_day_expiry_oi_growth_on_last_day"] = {"n_days": int(len(r)), "median_ratio_D_over_Dminus1": float(np.median(r)), "p25": float(np.percentile(r, 25)), "p75": float(np.percentile(r, 75))}
    return out


def fig_zero_dte(t8: dict, path=None):
    import matplotlib.pyplot as plt
    G.style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 4.6), gridspec_kw={"wspace": 0.34})
    bb = pd.DataFrame(t8["by_bucket"]).set_index("bucket")
    bb = bb[bb.index < 12]
    labels = [f"{(570 + 30 * int(b) + 30) // 60:02d}:{(570 + 30 * int(b) + 30) % 60:02d}" for b in bb.index]
    ax1.plot(bb.index, bb["ugex_0dte_bn"], marker="o", color=G.PURPLE, lw=2.0, label="same-day expiry, unsigned gamma exposure")
    ax1.plot(bb.index, bb["book_abs_bn"], marker="o", color=G.DIM, lw=2.0, label="standing book, abs net GEX (re-marked)")
    ax1.set_xticks(bb.index[::2]); ax1.set_xticklabels(labels[::2], fontsize=9)
    ax1.set_ylabel("median across days, $bn per 1% move")
    ax1.set_xlabel("bucket close, ET")
    ax1.set_title("0DTE layer against the standing book", loc="left", fontsize=14, pad=20)
    ax1.text(0.0, 1.01, f"median share of same-day expiry in total unsigned exposure {bb['share_0dte'].median():.0%}, positions as of the prior close only", transform=ax1.transAxes, va="bottom", color=G.DIM, fontsize=8.5)
    ax1.set_ylim(0, max(bb["ugex_0dte_bn"].max(), bb["book_abs_bn"].max()) * 1.35)
    ax1.legend(loc="upper right", fontsize=8.5)
    r = t8["last_half_hour_by_0dte_exposure_at_1530"]
    keys = [("all", "all days"), ("prev_positive", "prior regime positive"), ("prev_negative", "prior regime negative")]
    xs = np.arange(len(keys)); w = 0.36
    ax2.bar(xs - w / 2, [r[k]["last_half_hour_range_adj_low_0dte"] for k, _ in keys], width=w, color=G.DIM2, lw=0, label="bottom third of 0DTE exposure at 15:30")
    ax2.bar(xs + w / 2, [r[k]["high_0dte"] for k, _ in keys], width=w, color=G.PURPLE, lw=0, label="top third")
    for i, (k, _) in enumerate(keys):
        ax2.text(i, max(r[k]["last_half_hour_range_adj_low_0dte"], r[k]["high_0dte"]) + 0.05, f"diff {r[k]['diff_high_minus_low']:+.2f}\nCI [{r[k]['ci_lo']:+.2f}, {r[k]['ci_hi']:+.2f}]", ha="center", va="bottom", color=G.FG, fontsize=8.5)
    ax2.set_xticks(xs); ax2.set_xticklabels([f"{lab}\nn = {r[k]['n']}" for k, lab in (("all", "all days"), ("prev_positive", "prior +"), ("prev_negative", "prior -"))], fontsize=9)
    ax2.set_ylabel("15:30 to 16:00 range, multiple of median")
    ax2.set_title("0DTE gamma and the last half hour", loc="left", fontsize=14, pad=20)
    ax2.set_ylim(0, max(max(r[k]["last_half_hour_range_adj_low_0dte"], r[k]["high_0dte"]) for k, _ in keys) * 1.6)
    ax2.legend(loc="upper right", fontsize=8.5)
    if path:
        fig.savefig(path, bbox_inches="tight", pad_inches=0.25)
    return fig
