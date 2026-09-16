"""Cross-section: daily GEX and the intraday re-mark for many symbols, with pooled tests.

Per symbol: chain prep (gexlib), engine solve, daily table (gexlib.daily_gex), spot-grid profiles
(intraday.daily_profiles), saved to cache/{sym}_profiles.parquet and results/cross_section/{sym}_daily.csv.
Pooled: next-day range by regime within each name, a cross-sectional sort on GEX per dollar traded,
and the intraday re-mark tests (T6, T2, T3b) pooled across names. Cache only; no network.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import gexlib as G
import intraday as I

XS = G.RESULTS / "cross_section"


def flip_from_profiles(prof: pd.DataFrame, S: pd.Series) -> pd.DataFrame:
    """Zero crossing of the spot-grid profile nearest to spot (rel = 1), per day; NaN when none."""
    rel = prof.columns.to_numpy(dtype=float)
    rows = []
    for d, row in prof.iterrows():
        v = row.to_numpy(); sg = np.sign(v)
        cross = np.where(sg[:-1] * sg[1:] < 0)[0]
        level = np.nan
        if len(cross):
            xs = rel[cross] + (rel[cross + 1] - rel[cross]) * v[cross] / (v[cross] - v[cross + 1])
            level = float(S.loc[d] * xs[np.argmin(np.abs(xs - 1.0))])
        rows.append({"date": d, "flip_level": level, "n_crossings": int(len(cross))})
    return pd.DataFrame(rows).set_index("date")


def run_symbol(symbol: str, roots=None, target: str = "cpu", dtype: str = "fp64", verbose: bool = True) -> dict:
    """Daily GEX table and spot-grid profiles for one spot symbol. `roots` lists the option roots that share
    the spot (default: the symbol itself; SPX uses ("spx", "spxw")). Writes results/cross_section/{sym}_daily.csv
    and cache/{sym}_profiles.parquet."""
    sym = symbol.lower()
    roots = [r.lower() for r in (roots or [sym])]
    XS.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    stock = G.load_stock(sym)
    sofr = G.load_sofr()
    chains = []
    for root in roots:
        days = G.study_days(root)
        if days:
            ch, _ = G.prep_all(days, stock, sofr, verbose=False, symbol=root, q=G.q_for(sym))
            chains.append(ch)
    if not chains:
        return {"symbol": symbol.upper(), "status": "no option cache"}
    chain = pd.concat(chains, ignore_index=True)
    if len(chain) == 0:
        return {"symbol": symbol.upper(), "status": "empty chain"}
    sol, tm = G.solve_chain(chain, target=target, dtype=dtype)
    rep = G.solve_report(sol)
    prof = I.daily_profiles(sol, verbose=False)
    prof.to_parquet(G.CACHE / f"{sym}_profiles.parquet")
    flips = flip_from_profiles(prof, stock["close"])
    daily = G.daily_gex(sol, stock, flips)
    daily.to_csv(XS / f"{sym}_daily.csv", float_format="%.8g")
    out = {"symbol": symbol.upper(), "roots": [r.upper() for r in roots], "status": "ok", "days": int(len(daily)), "first": str(daily.index.min().date()), "last": str(daily.index.max().date()),
           "contracts": int(tm["n_contracts"]), "t_iv_s": tm["t_iv_s"], "t_greeks_s": tm["t_greeks_s"],
           "iv_status_ne_0_oi_share": rep["oi_share_status_ne_0"], "resid_p99": rep["resid_p99"],
           "share_negative_days": float((daily["regime"] < 0).mean()), "median_gex_bn": float(daily["gex_net_usd"].median() / 1e9),
           "median_abs_gex_bn": float(daily["gex_net_usd"].abs().median() / 1e9), "flip_found_share": float(daily["flip_level"].notna().mean()),
           "median_oi_total": float(daily["oi_total"].median()), "wall_s": round(time.perf_counter() - t0, 1)}
    if verbose:
        print(f"{symbol.upper():6s} {out['days']} days, {out['contracts']:,} contracts, IV {tm['t_iv_s']:.1f}s, "
              f"negative {out['share_negative_days']:.0%}, median |GEX| {out['median_abs_gex_bn']:.2f} $bn, {out['wall_s']}s", flush=True)
    return out


def load_daily(symbol: str) -> pd.DataFrame:
    return pd.read_csv(XS / f"{symbol.lower()}_daily.csv", parse_dates=["date"]).set_index("date")


# ----------------------------------------------------------------------------- pooled daily tests
def next_day_by_regime(symbols) -> pd.DataFrame:
    """Within each name: mean next-day range and abs return after positive against negative GEX, and the
    same on the name's own scale (range divided by its trailing 20-day mean range)."""
    from scipy.stats import mannwhitneyu
    rows = []
    for s in symbols:
        d = load_daily(s)
        d["range_next"] = d["range_pct"].shift(-1); d["absret_next"] = d["ret_cc"].abs().shift(-1)
        d["range_rel_next"] = d["range_next"] / d["range_pct"].rolling(20).mean()
        a = d[d["regime"] > 0]; b = d[d["regime"] < 0]
        if len(a) < 10 or len(b) < 10:
            rows.append({"symbol": s.upper(), "n_pos": len(a), "n_neg": len(b)}); continue
        p = float(mannwhitneyu(a["range_next"].dropna(), b["range_next"].dropna()).pvalue)
        rows.append({"symbol": s.upper(), "n_pos": int(len(a)), "n_neg": int(len(b)), "share_negative": float((d["regime"] < 0).mean()),
                     "range_next_pos": float(a["range_next"].mean()), "range_next_neg": float(b["range_next"].mean()),
                     "ratio_neg_over_pos": float(b["range_next"].mean() / a["range_next"].mean()),
                     "range_rel_pos": float(a["range_rel_next"].mean()), "range_rel_neg": float(b["range_rel_next"].mean()),
                     "absret_next_pos": float(a["absret_next"].mean()), "absret_next_neg": float(b["absret_next"].mean()), "p_mwu_range": p})
    return pd.DataFrame(rows).set_index("symbol")


def pooled_panel(symbols) -> pd.DataFrame:
    """Stack the daily tables with per-name normalizations: GEX per dollar of 20-day average traded value,
    range relative to the name's trailing 20-day mean, and the name's own GEX z-score."""
    frames = []
    for s in symbols:
        d = load_daily(s).copy()
        d["symbol"] = s.upper()
        d["dollar_vol20"] = (d["close"] * G.load_stock(s)["volume"].reindex(d.index)).rolling(20).mean()
        d["gex_per_dv"] = d["gex_net_usd"] / d["dollar_vol20"]
        d["abs_gex_per_dv"] = d["gex_net_usd"].abs() / d["dollar_vol20"]
        d["gex_z"] = (d["gex_net_usd"] - d["gex_net_usd"].rolling(60).mean()) / d["gex_net_usd"].rolling(60).std()
        d["range_next"] = d["range_pct"].shift(-1)
        d["range_rel_next"] = d["range_next"] / d["range_pct"].rolling(20).mean()
        d["absret_next"] = d["ret_cc"].abs().shift(-1)
        d["ret_next"] = d["ret_cc"].shift(-1)
        frames.append(d)
    return pd.concat(frames)


def cross_sectional_sort(panel: pd.DataFrame, key: str = "gex_per_dv", n_bins: int = 5) -> pd.DataFrame:
    """Each day, rank names by `key` into n_bins; average the next-day relative range and abs return per bin."""
    p = panel.dropna(subset=[key, "range_rel_next"]).copy()
    counts = p.groupby(level=0)["symbol"].transform("size")
    p = p[counts >= n_bins * 2]
    p["bin"] = p.groupby(level=0)[key].transform(lambda x: pd.qcut(x.rank(method="first"), n_bins, labels=False))
    g = p.groupby("bin").agg(n=("range_rel_next", "size"), range_rel_next=("range_rel_next", "mean"), absret_next=("absret_next", "mean"),
                             ret_next=("ret_next", "mean"), key_median=(key, "median"), share_negative=("regime", lambda r: float((r < 0).mean())))
    return g


def pooled_regime_test(panel: pd.DataFrame) -> dict:
    """All names pooled: next-day relative range and abs return after positive against negative GEX, with a
    bootstrap CI over name-days and a Mann-Whitney p."""
    from scipy.stats import mannwhitneyu
    p = panel.dropna(subset=["range_rel_next"])
    a = p.loc[p["regime"] > 0, "range_rel_next"].to_numpy(); b = p.loc[p["regime"] < 0, "range_rel_next"].to_numpy()
    lo, hi = I.boot_diff(a, b)
    return {"n_pos": int(len(a)), "n_neg": int(len(b)), "n_names": int(p["symbol"].nunique()), "range_rel_pos": float(a.mean()), "range_rel_neg": float(b.mean()),
            "diff": float(a.mean() - b.mean()), "ci_lo": lo, "ci_hi": hi, "p_mwu": float(mannwhitneyu(a, b).pvalue),
            "absret_pos": float(p.loc[p["regime"] > 0, "absret_next"].mean()), "absret_neg": float(p.loc[p["regime"] < 0, "absret_next"].mean())}


# ----------------------------------------------------------------------------- pooled intraday tests
def intraday_for_symbol(symbol: str) -> dict | None:
    sym = symbol.lower()
    pp = G.CACHE / f"{sym}_profiles.parquet"
    if not pp.exists() or not list(G.CACHE.glob(f"{sym}_1m_????-??.parquet")):
        return None
    daily = load_daily(sym)
    prof = pd.read_parquet(pp); prof.columns = prof.columns.astype(float)
    bars = I.load_bars(sym)
    ob = I.attach_regime(I.bucketize(bars, 30), daily, prof)
    if len(ob) < 200:
        return None
    t6 = I.t6_remark_tracks_next_print(daily, prof)
    t2 = I.t2_remark_vs_stale(ob)
    t3, days = I.t3_intraday_flips(ob)
    t3b = I.t3b_near_flip(ob, days, near_pct=1.0)
    ob["symbol"] = symbol.upper()
    return {"symbol": symbol.upper(), "t6": t6, "t2": t2, "t3": t3, "t3b": t3b, "ob": ob}


def pooled_intraday(results: list[dict]) -> dict:
    """Pool the bucket panels of all names: re-mark against stale (Spearman on within-name ranks), flip-versus-
    same next-bucket range, and the near-flip crossing test with the whole cross-section as the sample."""
    from scipy.stats import mannwhitneyu, spearmanr
    ob = pd.concat([r["ob"] for r in results], ignore_index=True)
    ob["gex_prev_rank"] = ob.groupby("symbol")["gex_prev"].rank(pct=True)
    ob["gex_rt_rank"] = ob.groupby("symbol")["gex_rt"].rank(pct=True)
    d = ob.dropna(subset=["next_range_adj", "gex_rt", "gex_prev"])
    out = {"n_names": int(d["symbol"].nunique()), "n_buckets": int(len(d)), "n_days": int(d.groupby(["symbol", "date"]).ngroups),
           "spearman_stale": float(spearmanr(d["gex_prev_rank"], d["next_range_adj"]).correlation),
           "spearman_remark": float(spearmanr(d["gex_rt_rank"], d["next_range_adj"]).correlation)}
    flipped = d["regime_rt"] != d["regime_prev"]
    out["share_days_with_flip"] = float(d.groupby(["symbol", "date"]).apply(lambda g: (g["regime_rt"] != g["regime_prev"]).any()).mean())
    res = {}
    for reg, lab in ((-1, "prev_negative_flipped_positive"), (1, "prev_positive_flipped_negative")):
        m = d["regime_prev"] == reg
        a = d.loc[m & flipped, "next_range_adj"].to_numpy(); b = d.loc[m & ~flipped, "next_range_adj"].to_numpy()
        lo, hi = I.boot_diff(a, b)
        res[lab] = {"n_flipped": int(len(a)), "n_same": int(len(b)), "next_range_adj_flipped": float(a.mean()), "next_range_adj_same": float(b.mean()),
                    "diff": float(a.mean() - b.mean()), "ci_lo": lo, "ci_hi": hi, "p_mwu": float(mannwhitneyu(a, b).pvalue)}
    out["flip_vs_same"] = res
    t6 = pd.DataFrame([r["t6"] for r in results], index=[r["symbol"] for r in results])
    out["t6_by_name"] = t6[["sign_agreement_stale", "sign_agreement_remark", "corr_level_stale", "corr_level_remark", "remark_caught_change"]].round(3).to_dict("index")
    out["t6_mean"] = t6[["sign_agreement_stale", "sign_agreement_remark", "corr_level_stale", "corr_level_remark"]].mean().round(3).to_dict()
    return out


# ----------------------------------------------------------------------------- figures
def fig_ratio_by_name(tbl: pd.DataFrame, path=None):
    """Next-day range after negative GEX over after positive GEX, one bar per name."""
    import matplotlib.pyplot as plt
    G.style()
    t = tbl.dropna(subset=["ratio_neg_over_pos"]).sort_values("ratio_neg_over_pos")
    fig, ax = plt.subplots(figsize=(G.FIG_W, max(G.FIG_H, 0.22 * len(t) + 1.5)))
    ys = np.arange(len(t))
    cols = [G.ORANGE if r > 1 else G.CYAN for r in t["ratio_neg_over_pos"]]
    ax.barh(ys, t["ratio_neg_over_pos"], color=cols, height=0.7, lw=0)
    ax.axvline(1.0, color=G.FG, lw=1.0, ls="--")
    ax.set_yticks(ys); ax.set_yticklabels([f"{s}  (n {int(r.n_pos)}/{int(r.n_neg)}, p {r.p_mwu_range:.2f})" for s, r in t.iterrows()], fontsize=9)
    ax.set_xlabel("mean next-day range after negative GEX / after positive GEX")
    ax.set_title("Does the SPY result hold name by name?", loc="left", pad=22)
    ax.text(0.0, 1.01, f"{int((t['ratio_neg_over_pos'] > 1).sum())} of {len(t)} names above 1; median ratio {t['ratio_neg_over_pos'].median():.2f}; n = positive/negative days, p from Mann-Whitney", transform=ax.transAxes, color=G.DIM, fontsize=10, va="bottom")
    if path:
        fig.savefig(path, bbox_inches="tight", pad_inches=0.25)
    return fig


def fig_xs_sort(sort_tbl: pd.DataFrame, key_label: str, path=None):
    import matplotlib.pyplot as plt
    G.style()
    fig, ax = plt.subplots(figsize=(G.FIG_W, G.FIG_H))
    xs = np.arange(len(sort_tbl))
    cols = [G.ORANGE if s > 0.5 else G.CYAN for s in sort_tbl["share_negative"]]
    ax.bar(xs, sort_tbl["range_rel_next"], color=cols, width=0.65, lw=0)
    ax.axhline(1.0, color=G.FG, lw=1.0, ls="--")
    ax.set_xticks(xs); ax.set_xticklabels([f"bin {int(b) + 1}\n{'most negative' if i == 0 else ('most positive' if i == len(sort_tbl) - 1 else '')}" for i, b in enumerate(sort_tbl.index)], fontsize=10)
    ax.set_xlabel(f"names ranked each day by {key_label}")
    ax.set_ylabel("next-day range / own trailing 20-day mean")
    ax.set_title("Cross-sectional sort: quiet and wide names tomorrow", loc="left", pad=22)
    ax.text(0.0, 1.01, "bar colour: orange where most names in the bin have negative GEX, cyan otherwise; dashed line = the name's usual range", transform=ax.transAxes, color=G.DIM, fontsize=10, va="bottom")
    for i, v in enumerate(sort_tbl["range_rel_next"]):
        ax.text(i, v + 0.01, f"{v:.3f}", ha="center", va="bottom", color=G.FG, fontsize=10)
    if path:
        fig.savefig(path, bbox_inches="tight", pad_inches=0.25)
    return fig


def fig_remark_by_name(t6_by_name: dict, path=None):
    import matplotlib.pyplot as plt
    G.style()
    t = pd.DataFrame(t6_by_name).T.sort_values("sign_agreement_remark")
    fig, ax = plt.subplots(figsize=(G.FIG_W, max(G.FIG_H, 0.22 * len(t) + 1.5)))
    ys = np.arange(len(t))
    ax.hlines(ys, t["sign_agreement_stale"] * 100, t["sign_agreement_remark"] * 100, color=G.DIM2, lw=2)
    ax.scatter(t["sign_agreement_stale"] * 100, ys, color=G.DIM, s=40, zorder=3, label="stale: yesterday's print")
    ax.scatter(t["sign_agreement_remark"] * 100, ys, color=G.CYAN, s=40, zorder=3, label="re-marked at today's close")
    ax.set_yticks(ys); ax.set_yticklabels(t.index, fontsize=9)
    ax.set_xlabel("agreement with the sign of the next official GEX print, %")
    ax.set_title("Re-marking the book tracks the next print in every name", loc="left", pad=22)
    ax.text(0.0, 1.01, f"mean across names: stale {t['sign_agreement_stale'].mean():.1%}, re-marked {t['sign_agreement_remark'].mean():.1%}", transform=ax.transAxes, color=G.DIM, fontsize=10, va="bottom")
    ax.legend(loc="lower right", fontsize=10)
    if path:
        fig.savefig(path, bbox_inches="tight", pad_inches=0.25)
    return fig


# ----------------------------------------------------------------------------- terciles (single names rarely change sign)
def next_day_by_tercile(symbols) -> pd.DataFrame:
    """Within each name: next-day range relative to the name's trailing 20-day mean, in the bottom third of the
    name's own net GEX (most negative or least positive) against the top third. Ratio bottom over top > 1 means
    low GEX goes with a wider next day, the SPY direction."""
    from scipy.stats import mannwhitneyu
    rows = []
    for s in symbols:
        d = load_daily(s).copy()
        d["range_rel_next"] = d["range_pct"].shift(-1) / d["range_pct"].rolling(20).mean()
        d["absret_next"] = d["ret_cc"].abs().shift(-1)
        d = d.dropna(subset=["range_rel_next"])
        if len(d) < 60:
            rows.append({"symbol": s.upper(), "n": int(len(d))}); continue
        lo_q, hi_q = d["gex_net_usd"].quantile([1 / 3, 2 / 3])
        lo = d[d["gex_net_usd"] <= lo_q]; hi = d[d["gex_net_usd"] >= hi_q]
        rows.append({"symbol": s.upper(), "n": int(len(d)), "share_negative": float((d["regime"] < 0).mean()),
                     "gex_bottom_third_max_bn": float(lo_q / 1e9), "gex_top_third_min_bn": float(hi_q / 1e9),
                     "range_rel_bottom": float(lo["range_rel_next"].mean()), "range_rel_top": float(hi["range_rel_next"].mean()),
                     "ratio_bottom_over_top": float(lo["range_rel_next"].mean() / hi["range_rel_next"].mean()),
                     "absret_bottom": float(lo["absret_next"].mean()), "absret_top": float(hi["absret_next"].mean()),
                     "p_mwu": float(mannwhitneyu(lo["range_rel_next"], hi["range_rel_next"]).pvalue)})
    return pd.DataFrame(rows).set_index("symbol")


def pooled_tercile_test(panel: pd.DataFrame) -> dict:
    """Pooled over names: bottom third against top third of each name's own GEX (computed within the name)."""
    from scipy.stats import mannwhitneyu
    p = panel.dropna(subset=["range_rel_next"]).copy()
    q = p.groupby("symbol")["gex_net_usd"].transform(lambda x: pd.qcut(x.rank(method="first"), 3, labels=False))
    a = p.loc[q == 0, "range_rel_next"].to_numpy(); b = p.loc[q == 2, "range_rel_next"].to_numpy()
    lo, hi = I.boot_diff(a, b)
    return {"n_bottom": int(len(a)), "n_top": int(len(b)), "n_names": int(p["symbol"].nunique()), "range_rel_bottom": float(a.mean()), "range_rel_top": float(b.mean()),
            "ratio_bottom_over_top": float(a.mean() / b.mean()), "diff": float(a.mean() - b.mean()), "ci_lo": lo, "ci_hi": hi, "p_mwu": float(mannwhitneyu(a, b).pvalue),
            "absret_bottom": float(p.loc[q == 0, "absret_next"].mean()), "absret_top": float(p.loc[q == 2, "absret_next"].mean())}


def fig_tercile_by_name(tbl: pd.DataFrame, path=None):
    """Next-day relative range in the name's bottom GEX third over its top third, one bar per name."""
    import matplotlib.pyplot as plt
    G.style()
    t = tbl.dropna(subset=["ratio_bottom_over_top"]).sort_values("ratio_bottom_over_top")
    fig, ax = plt.subplots(figsize=(G.FIG_W, max(G.FIG_H, 0.22 * len(t) + 1.5)))
    ys = np.arange(len(t))
    cols = [G.ORANGE if r > 1 else G.CYAN for r in t["ratio_bottom_over_top"]]
    ax.barh(ys, t["ratio_bottom_over_top"], color=cols, height=0.7, lw=0)
    ax.axvline(1.0, color=G.FG, lw=1.0, ls="--")
    ax.set_yticks(ys); ax.set_yticklabels([f"{s}  (negative {r.share_negative:.0%} of days, p {r.p_mwu:.2f})" for s, r in t.iterrows()], fontsize=9)
    ax.set_xlabel("next-day range relative to own mean: bottom GEX third / top GEX third")
    ax.set_title("Low GEX, wider next day: name by name", loc="left", pad=22)
    ax.text(0.0, 1.01, f"{int((t['ratio_bottom_over_top'] > 1).sum())} of {len(t)} names above 1; median ratio {t['ratio_bottom_over_top'].median():.2f}; terciles of each name's own net GEX", transform=ax.transAxes, color=G.DIM, fontsize=10, va="bottom")
    if path:
        fig.savefig(path, bbox_inches="tight", pad_inches=0.25)
    return fig


# ----------------------------------------------------------------------------- same-day discrimination across names
def _within_group_diff(frame: pd.DataFrame, group_cols, lo_mask: pd.Series, hi_mask: pd.Series, y: str = "y", seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    diffs = []
    for _, g in frame.groupby(group_cols):
        a, b = g.loc[lo_mask.loc[g.index], y], g.loc[hi_mask.loc[g.index], y]
        if len(a) >= 2 and len(b) >= 2:
            diffs.append(a.mean() - b.mean())
    d = np.array(diffs)
    boots = rng.choice(d, (10000, len(d))).mean(1)
    return {"n_groups": int(len(d)), "mean_diff": float(d.mean()), "ci_lo": float(np.percentile(boots, 2.5)), "ci_hi": float(np.percentile(boots, 97.5)), "share_positive": float((d > 0).mean())}


def same_day_tests(panel: pd.DataFrame, ob_all: pd.DataFrame) -> dict:
    """Does GEX separate names from each other at the same time? Daily: within each date, names in their own
    bottom GEX tercile against names in their own top tercile (next-day relative range). Intraday: within each
    (date, bucket), names whose re-marked sign flipped against names whose sign held, by prior regime. Also the
    time effect: the share of names in their bottom tercile on D against the market-average relative range on D+1."""
    from scipy.stats import spearmanr
    p = panel.dropna(subset=["range_rel_next"]).copy()
    p["own_tercile"] = p.groupby("symbol")["gex_net_usd"].transform(lambda x: pd.qcut(x.rank(method="first"), 3, labels=False))
    p["y"] = p["range_rel_next"]; p = p.reset_index()
    out = {"daily_same_day_bottom_minus_top": _within_group_diff(p, ["date"], p["own_tercile"] == 0, p["own_tercile"] == 2)}
    bydate = p.groupby("date").agg(rr=("range_rel_next", "mean"), share_low=("own_tercile", lambda t: float((t == 0).mean())))
    out["time_effect_spearman_share_low_vs_market_range_next"] = float(spearmanr(bydate["share_low"], bydate["rr"]).correlation)
    ob = ob_all.dropna(subset=["next_range_adj", "gex_rt"]).copy()
    ob["flipped"] = ob["regime_rt"] != ob["regime_prev"]; ob["y"] = ob["next_range_adj"]
    res = {}
    for reg, lab in ((-1, "prev_negative_flipped_positive"), (1, "prev_positive_flipped_negative")):
        q = ob[ob["regime_prev"] == reg].reset_index(drop=True)
        res[lab] = _within_group_diff(q, ["date", "bucket"], q["flipped"], ~q["flipped"])
    out["intraday_same_bucket_flipped_minus_same"] = res
    return out
