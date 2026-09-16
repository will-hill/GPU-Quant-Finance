"""One function per demonstrated GEX indication (the rows of the findings table), for the gex_rows notebooks.

load(sym) gathers everything a row needs from the cache; rowN(ctx) returns the row's statistics; figN(ctx, res)
draws its figure; per_name(cols) reads the per-symbol table written by per_symbol_tests.py.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, pearsonr, spearmanr

import cross_section as X
import gexlib as G
import intraday as I

NAMES = ["SPY", "QQQ", "IWM", "NVDA", "TSLA", "AAPL", "AMZN", "META", "MSFT", "AMD"]


def load(sym: str = "spy", with_chain: bool = False) -> dict:
    sym = sym.lower()
    p = X.XS / f"{sym}_daily.csv"
    daily = X.load_daily(sym) if p.exists() else pd.read_csv(G.RESULTS / "spy_gex_daily.csv", parse_dates=["date"]).set_index("date")
    stock = G.load_stock(sym)
    prof = pd.read_parquet(G.CACHE / f"{sym}_profiles.parquet"); prof.columns = prof.columns.astype(float)
    bars = I.load_bars(sym)
    ob = I.attach_regime(I.bucketize(bars, 30), daily, prof)
    ctx = {"sym": sym.upper(), "daily": daily, "stock": stock, "prof": prof, "bars": bars, "ob": ob}
    if with_chain:
        sofr = G.load_sofr(); roots = ["spx", "spxw"] if sym == "spx" else [sym]
        chain = pd.concat([G.prep_all(G.study_days(r), stock, sofr, verbose=False, symbol=r, q=G.q_for(sym))[0] for r in roots], ignore_index=True)
        ctx["sol"], ctx["timings"] = G.solve_chain(chain)
    return ctx


def per_name(cols: dict, names=NAMES) -> pd.DataFrame:
    """Per-symbol statistics from results/cross_section/per_symbol_tests.json; cols maps json key -> column label."""
    rows = json.loads((X.XS / "per_symbol_tests.json").read_text())
    t = pd.DataFrame(rows).set_index("symbol").reindex(names)
    return t[[k for k in cols if k in t.columns]].rename(columns=cols)


# ----------------------------------------------------------------------------- rows
def row1(ctx):
    return I.t6_live_tracks_next_print(ctx["daily"], ctx["prof"])


def fig1(ctx, res, path=None):
    return I.fig_live_versus_stale(ctx["daily"], ctx["prof"], res, path)


def row2(ctx):
    ob = ctx["ob"]; W = ob.pivot(index="date", columns="bucket", values="range_adj"); Gx = ob.pivot(index="date", columns="bucket", values="gex_live")
    stale = ob.groupby("date")["gex_prev"].first()
    def fwd(k, h):
        cols = [c for c in W.columns if c > k and (h is None or c <= k + h)]
        return W[cols].mean(axis=1)
    horizons = {}
    for k, lab in ((0, "10:00"), (1, "10:30"), (3, "11:30"), (5, "12:30"), (8, "14:00")):
        horizons[lab] = {}
        for h, hl in ((1, "next 30 min"), (2, "next 1 h"), (4, "next 2 h"), (None, "rest of day")):
            y = fwd(k, h); x = Gx[k]; m = x.notna() & y.notna()
            horizons[lab][hl] = (float(spearmanr(x[m], y[m]).correlation), float(spearmanr(stale.reindex(x.index)[m], y[m]).correlation))
    x = Gx[0]; y = fwd(0, None); m = x.notna() & y.notna(); x, y = x[m], y[m]
    q = pd.qcut(x.rank(method="first"), 3, labels=["low third", "middle", "high third"])
    terc = y.groupby(q, observed=True).mean()
    size = pd.qcut(x.abs().rank(method="first"), 3, labels=["small", "medium", "large"])
    by_size_sign = y.groupby([np.sign(x).map({-1.0: "negative", 1.0: "positive"}), size], observed=True).mean()
    return {"spearman_by_time_and_horizon (live, stale)": horizons, "rest_of_day_by_1000_tercile": terc.round(3).to_dict(),
            "rest_of_day_by_sign_and_size": {f"{a} {b}": round(v, 3) for (a, b), v in by_size_sign.items()},
            "p_low_vs_high": float(mannwhitneyu(y[q == "low third"], y[q == "high third"]).pvalue), "n_days": int(len(x))}


def fig2(ctx, res, path=None):
    import matplotlib.pyplot as plt
    G.style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 4.6), gridspec_kw={"wspace": 0.3})
    t = res["rest_of_day_by_1000_tercile"]; keys = ["low third", "middle", "high third"]
    ax1.bar(range(3), [t[k] for k in keys], color=[G.ORANGE, G.DIM2, G.CYAN], width=0.6, lw=0)
    for i, k in enumerate(keys): ax1.text(i, t[k] + 0.03, f"{t[k]:.2f} times", ha="center", color=G.FG)
    ax1.set_xticks(range(3)); ax1.set_xticklabels(keys, fontsize=10); ax1.set_xlabel("live GEX at 10:00, terciles")
    ax1.set_ylabel("rest-of-day range, multiple of time-of-day median"); ax1.set_title(f"{ctx['sym']}: level at 10:00 sets the day", loc="left", fontsize=14)
    s = res["rest_of_day_by_sign_and_size"]; order = ["negative large", "negative medium", "negative small", "positive small", "positive medium", "positive large"]
    vals = [s.get(k, np.nan) for k in order]
    ax2.bar(range(6), vals, color=[G.ORANGE] * 3 + [G.CYAN] * 3, width=0.6, lw=0)
    for i, v in enumerate(vals): ax2.text(i, v + 0.03, f"{v:.2f} times", ha="center", color=G.FG, fontsize=9)
    from matplotlib.patches import Patch
    ax2.set_xticks(range(6)); ax2.set_xticklabels([k.split()[1] for k in order], fontsize=10); ax2.set_xlabel("size of |GEX| at 10:00, terciles")
    ax2.legend(handles=[Patch(color=G.ORANGE, label="negative at 10:00"), Patch(color=G.CYAN, label="positive at 10:00")], loc="upper right", fontsize=9)
    ax2.set_ylim(0, max(vals) * 1.25)
    ax2.set_title("by sign and size of |GEX| at 10:00", loc="left", fontsize=14)
    if path: fig.savefig(path, bbox_inches="tight", pad_inches=0.25)
    return fig


def row3(ctx):
    import statsmodels.api as sm
    ob = ctx["ob"]; t2 = I.t2_live_versus_stale(ob)
    d = ob.dropna(subset=["next_range_adj", "gex_live", "gex_prev"])
    X_ = sm.add_constant(pd.DataFrame({"stale_rank": d["gex_prev"].rank(pct=True), "live_rank": d["gex_live"].rank(pct=True)}))
    fit = sm.OLS(d["next_range_adj"].to_numpy(), X_).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d["date"])[0]})
    by_bucket = {int(b): (float(spearmanr(g["gex_live"], g["next_range_adj"]).correlation), float(spearmanr(g["gex_prev"], g["next_range_adj"]).correlation)) for b, g in d.groupby("bucket") if len(g) > 50}
    return {"spearman_live": t2["spearman_live"], "spearman_stale": t2["spearman_stale"], "within_day_increment": t2["spearman_increment_within_day"],
            "joint_regression_t": {"live": float(fit.tvalues["live_rank"]), "stale": float(fit.tvalues["stale_rank"])}, "n_buckets": t2["n_buckets"],
            "share_days_with_flip": t2["share_days_with_any_flip"], "spearman_by_bucket (live, stale)": by_bucket}


def fig3(ctx, res, path=None):
    import matplotlib.pyplot as plt
    G.style()
    bb = res["spearman_by_bucket (live, stale)"]; ks = sorted(bb)
    fig, ax = plt.subplots(figsize=(G.FIG_W, G.FIG_H))
    ax.plot(ks, [bb[k][0] for k in ks], marker="o", color=G.CYAN, lw=2, label="live at the bucket close")
    ax.plot(ks, [bb[k][1] for k in ks], marker="o", color=G.DIM, lw=2, label="stale previous-close print")
    ax.set_xticks(ks); ax.set_xticklabels([f"{(570 + 30 * (k + 1)) // 60:02d}:{(570 + 30 * (k + 1)) % 60:02d}" for k in ks], fontsize=9)
    ax.set_xlabel("bucket close, ET"); ax.set_ylabel("Spearman with the next bucket's adjusted range")
    ax.set_title(f"{ctx['sym']}: the live value's edge grows through the day", loc="left"); ax.legend(loc="lower right")
    if path: fig.savefig(path, bbox_inches="tight", pad_inches=0.25)
    return fig


def row4(ctx):
    ob = ctx["ob"]; t3, days = I.t3_intraday_flips(ob)
    return {"share_days_flipped": t3["share_days_flipped"], "first_flip_bucket_counts": t3["flip_bucket_distribution"], "matched_on_move_size": t3["by_direction"],
            "near_flip_0.5pct": I.t3b_near_flip(ob, days, 0.5)["near_flip_cross_vs_stay"], "near_flip_1pct": I.t3b_near_flip(ob, days, 1.0)["near_flip_cross_vs_stay"], "_days": days}


def fig4(ctx, res, path=None):
    return I.fig_intraday_flip(ctx["ob"], res["_days"], {"near_pct": 0.5, "near_flip_cross_vs_stay": res["near_flip_0.5pct"]}, path)


def row5(ctx):
    layer = I.zero_dte_layer(ctx["sol"], ctx["daily"], ctx["ob"], symbol=ctx["sym"].lower() if ctx["sym"] != "SPX" else "spx")
    return {"t8": I.t8_zero_dte(layer), "controls": I.t8_controls(layer, ctx["sol"], ctx["daily"], symbol=ctx["sym"].lower()), "_layer": layer}


def fig5(ctx, res, path=None):
    return I.fig_zero_dte(res["t8"], path)


def row6(ctx):
    s = ctx["daily"]["regime"].to_numpy(); runs = G.sign_runs(ctx["daily"])
    return {"p_same_sign_next_day": float((s[:-1] == s[1:]).mean()), "phi_sign_D_vs_D1": float(pearsonr(s[:-1], s[1:])[0]),
            "n_runs": int(len(runs)), "longest_positive": int(runs.loc[runs["sign"] == 1, "length"].max()), "longest_negative": int(runs.loc[runs["sign"] == -1, "length"].max()),
            "median_run_positive": float(runs.loc[runs["sign"] == 1, "length"].median()), "median_run_negative": float(runs.loc[runs["sign"] == -1, "length"].median()), "_runs": runs}


def fig6(ctx, res, path=None):
    import matplotlib.pyplot as plt
    G.style()
    runs = res["_runs"]; fig, ax = plt.subplots(figsize=(G.FIG_W, G.FIG_H))
    bins = np.arange(0.5, max(runs["length"].max(), 10) + 1.5, 1)
    ax.hist(runs.loc[runs["sign"] == 1, "length"], bins=bins, color=G.CYAN, alpha=0.8, label=f"positive runs (n {int((runs['sign'] == 1).sum())})")
    ax.hist(runs.loc[runs["sign"] == -1, "length"], bins=bins, color=G.ORANGE, alpha=0.6, label=f"negative runs (n {int((runs['sign'] == -1).sum())})")
    ax.set_xlabel("run length, trading days"); ax.set_ylabel("count of runs"); ax.set_title(f"{ctx['sym']}: the sign holds {res['p_same_sign_next_day']:.0%} of days", loc="left"); ax.legend()
    if path: fig.savefig(path, bbox_inches="tight", pad_inches=0.25)
    return fig


def row7(ctx):
    return I.t1_intraday_vol(ctx["bars"], ctx["daily"])


def fig7(ctx, res, path=None):
    return I.fig_intraday_vol(res, path)


def row8(ctx):
    out = G.next_day_outcomes(ctx["daily"], ctx["stock"]); tests = G.regime_tests(out, out["regime"]); tests["verdict"] = G.verdicts(tests)
    lo_q, hi_q = out["gex"].quantile([1 / 3, 2 / 3]); lo, hi = out[out["gex"] <= lo_q], out[out["gex"] >= hi_q]
    xc = G.cross_correlation(ctx["daily"], ctx["stock"])
    return {"sign_cut": tests.loc[["range", "move"]], "tercile_range_next": {"bottom third": float(lo["range_next"].mean()), "top third": float(hi["range_next"].mean()),
            "p": float(mannwhitneyu(lo["range_next"].dropna(), hi["range_next"].dropna()).pvalue)}, "spearman_gex_next_range": float(spearmanr(out["gex"], out["range_next"], nan_policy="omit").correlation),
            "cross_correlation": xc, "_out": out}


def fig8(ctx, res, path=None):
    return G.fig_lead_test(res["_out"], res["cross_correlation"], res["_out"]["regime"], "sign cut", path)


def row9(ctx):
    out = G.next_day_outcomes(ctx["daily"], ctx["stock"]); tests = G.regime_tests(out, out["regime"])
    lo_q, hi_q = out["gex"].quantile([1 / 3, 2 / 3]); lo, hi = out[out["gex"] <= lo_q], out[out["gex"] >= hi_q]
    return {"sign_cut": tests.loc[["forward vol"]], "tercile_fvol": {"bottom third": float(lo["fvol_5d"].mean()), "top third": float(hi["fvol_5d"].mean()),
            "p": float(mannwhitneyu(lo["fvol_5d"].dropna(), hi["fvol_5d"].dropna()).pvalue)}, "spearman_gex_fvol": float(spearmanr(out["gex"], out["fvol_5d"], nan_policy="omit").correlation), "_out": out}


def fig9(ctx, res, path=None):
    import matplotlib.pyplot as plt
    G.style()
    out = res["_out"]; fig, ax = plt.subplots(figsize=(G.FIG_W, G.FIG_H))
    q = pd.qcut(out["gex"].rank(method="first"), 5, labels=False); v = out.groupby(q)["fvol_5d"].mean() * 100
    ax.bar(range(5), v.values, color=[G.ORANGE, G.ORANGE, G.DIM2, G.CYAN, G.CYAN], width=0.6, lw=0)
    for i, x in enumerate(v.values): ax.text(i, x + 0.2, f"{x:.1f}%", ha="center", color=G.FG)
    ax.set_xticks(range(5)); ax.set_xticklabels(["most negative", "2", "3", "4", "most positive"]); ax.set_xlabel("quintile of net GEX at the close of D")
    ax.set_ylabel("annualized vol of daily returns, D+1 to D+5, %"); ax.set_title(f"{ctx['sym']}: next-week vol by GEX quintile", loc="left")
    if path: fig.savefig(path, bbox_inches="tight", pad_inches=0.25)
    return fig


def row10(ctx):
    out = G.next_day_outcomes(ctx["daily"], ctx["stock"]); lo_q = out["gex"].quantile(1 / 3)
    res = {"base_share_negative": float((out["regime"] < 0).mean()), "base_share_bottom_third": 1 / 3}
    for n in (10, 20, 50):
        big = out["absret_next"].nlargest(n).index; wide = out["range_next"].nlargest(n).index
        res[f"top{n}_moves_after_negative"] = float((out.loc[big, "regime"] < 0).mean()); res[f"top{n}_moves_in_bottom_third"] = float((out.loc[big, "gex"] <= lo_q).mean())
        res[f"top{n}_ranges_after_negative"] = float((out.loc[wide, "regime"] < 0).mean())
    res["spearman_gex_absret_next"] = float(spearmanr(out["gex"], out["absret_next"], nan_policy="omit").correlation)
    res["_out"] = out
    return res


def fig10(ctx, res, path=None):
    import matplotlib.pyplot as plt
    G.style()
    fig, ax = plt.subplots(figsize=(G.FIG_W, G.FIG_H))
    ns = (10, 20, 50); xs = np.arange(3); w = 0.36
    ax.bar(xs - w / 2, [res[f"top{n}_moves_after_negative"] * 100 for n in ns], width=w, color=G.ORANGE, lw=0, label="largest next-day moves after a negative close")
    ax.bar(xs + w / 2, [res[f"top{n}_ranges_after_negative"] * 100 for n in ns], width=w, color=G.PURPLE, lw=0, label="widest next-day ranges after a negative close")
    ax.axhline(res["base_share_negative"] * 100, color=G.FG, ls="--", lw=1, label=f"base rate of negative closes {res['base_share_negative']:.0%}")
    ax.set_xticks(xs); ax.set_xticklabels([f"top {n}" for n in ns]); ax.set_ylabel("share, %"); ax.set_ylim(0, 105)
    ax.set_title(f"{ctx['sym']}: the tails sit in the negative regime", loc="left"); ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3, fontsize=9)
    if path: fig.savefig(path, bbox_inches="tight", pad_inches=0.25)
    return fig


def show(d, skip_private=True):
    """Print a result dict without the underscore-prefixed frames."""
    if isinstance(d, pd.DataFrame):
        return d
    return json.loads(json.dumps({k: v for k, v in d.items() if not str(k).startswith("_")}, default=lambda o: o.to_dict() if hasattr(o, "to_dict") else (float(o) if isinstance(o, (np.floating, np.integer)) else str(o))))


# ----------------------------------------------------------------------------- opener figures: plain language, dollars, percent, dates
def _day_path(ctx, day):
    b = ctx["bars"].filter(pl.col("date") == pd.Timestamp(day).date()).sort("mod").to_pandas() if hasattr(ctx["bars"], "filter") else None
    return b


def _pct_from_open(b):
    return (b["close"] / b["open"].iloc[0] - 1.0) * 100.0


import polars as pl  # noqa: E402


def fig_open1(ctx, res, path=None):
    """One day where yesterday's report said calm, the live reading turned during the day, and the official
    next report confirmed it."""
    import matplotlib.pyplot as plt
    G.style()
    d = ctx["daily"].copy(); ob = ctx["ob"]
    d["gex_prev"] = d["gex_net_usd"].shift(1)
    cand = d[(np.sign(d["gex_prev"]) != np.sign(d["gex_net_usd"]))].copy(); cand["move"] = d["ret_cc"].abs()
    day = cand.sort_values("move").index[-3] if len(cand) > 3 else cand.index[-1]
    b = _day_path(ctx, day); o = ob[ob["date"] == day].sort_values("bucket")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(G.FIG_W, G.FIG_H), sharex=False, gridspec_kw={"height_ratios": [1.2, 1.0], "hspace": 0.35})
    x = b["mod"] / 60.0
    ax1.plot(x, b["close"], color=G.FG, lw=1.6)
    ax1.set_ylabel(f"{ctx['sym']} price, $"); ax1.set_title(f"{pd.Timestamp(day):%d %B %Y}: the report said calm, the live reading said otherwise", loc="left", fontsize=14)
    ax1.set_xticks(range(10, 17)); ax1.set_xticklabels([f"{h:02d}:00" for h in range(10, 17)]); ax1.set_xlim(9.4, 16.1)
    ax1.text(0.99, 0.95, f"{ctx['sym']} {d.loc[day, 'ret_cc'] * 100:+.1f}% on the day", transform=ax1.transAxes, ha="right", va="top", color=G.FG, fontsize=11)
    xb = (570 + 30 * (o["bucket"] + 1)) / 60.0
    ax2.plot(xb, o["gex_live"] / 1e9, marker="o", color=G.CYAN, lw=2, label="live reading during the day")
    ax2.axhline(o["gex_prev"].iloc[0] / 1e9, color=G.DIM, ls="--", lw=1.5, label=f"yesterday's report: {o['gex_prev'].iloc[0] / 1e9:+.1f} billion")
    ax2.scatter([16.0], [d.loc[day, "gex_net_usd"] / 1e9], color=G.ORANGE, s=90, zorder=5, label=f"tonight's official report: {d.loc[day, 'gex_net_usd'] / 1e9:+.1f} billion")
    ax2.axhline(0, color=G.FG, lw=0.8)
    ax2.set_ylabel("gamma exposure\nbillion $ per 1% move"); ax2.set_xticks(range(10, 17)); ax2.set_xticklabels([f"{h:02d}:00" for h in range(10, 17)]); ax2.set_xlim(9.4, 16.1)
    ax2.text(0.01, 0.06, "above zero: dealers calm the market.  below zero: dealers amplify it.", transform=ax2.transAxes, color=G.DIM, fontsize=9.5)
    ax2.legend(loc="upper right", fontsize=9)
    if path: fig.savefig(path, bbox_inches="tight", pad_inches=0.25)
    return fig


def fig_open2(ctx, res, path=None):
    """Read the gauge at 10:00: the rest of the day's range in percent, and one example day of each kind."""
    import matplotlib.pyplot as plt
    G.style()
    ob = ctx["ob"]
    rest = ob[ob["bucket"] >= 1].groupby("date").agg(hi=("high", "max"), lo=("low", "min"), S0=("S0", "first"))
    rest["range_pct"] = (rest["hi"] - rest["lo"]) / rest["S0"] * 100.0
    g10 = ob[ob["bucket"] == 0].set_index("date")["gex_live"].reindex(rest.index)
    m = g10.notna(); rest, g10 = rest[m], g10[m]
    q = pd.qcut(g10.rank(method="first"), 3, labels=["low", "middle", "high"])
    means = rest["range_pct"].groupby(q, observed=True).mean()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 4.6), gridspec_kw={"wspace": 0.3, "width_ratios": [1, 1.4]})
    ax1.bar(range(3), means.values, color=[G.ORANGE, G.DIM2, G.CYAN], width=0.6, lw=0)
    for i, v in enumerate(means.values): ax1.text(i, v + 0.03, f"{v:.2f}%", ha="center", color=G.FG, fontsize=12)
    ax1.set_xticks(range(3)); ax1.set_xticklabels(["low gamma\nat 10:00", "middle", "high gamma\nat 10:00"], fontsize=10)
    ax1.set_ylabel(f"{ctx['sym']} high-to-low range, 10:00 to close, %"); ax1.set_title("Read the gauge at 10:00", loc="left", fontsize=14)
    ax1.set_ylim(0, means.max() * 1.25)
    # example days: median-range day within the high third and within the low third
    for lab, c in (("high", G.CYAN), ("low", G.ORANGE)):
        sub = rest[q == lab]["range_pct"]; day = (sub - sub.median()).abs().idxmin()
        b = _day_path(ctx, day); ax2.plot(b["mod"] / 60.0, _pct_from_open(b), color=c, lw=1.8, label=f"{lab} gamma at 10:00, {pd.Timestamp(day):%d %b %Y}: range {rest.loc[day, 'range_pct']:.2f}%")
    ax2.axvline(10.0, color=G.PURPLE, ls=":", lw=1.5); ax2.text(10.05, ax2.get_ylim()[1] * 0.9 if ax2.get_ylim()[1] > 0 else 0.5, "10:00 reading", color=G.PURPLE, fontsize=9)
    ax2.set_xticks(range(10, 17)); ax2.set_xticklabels([f"{h:02d}:00" for h in range(10, 17)]); ax2.set_ylabel("move from the open, %")
    ax2.set_title("What the rest of the day looked like", loc="left", fontsize=14); ax2.legend(loc="upper left", fontsize=8.5)
    if path: fig.savefig(path, bbox_inches="tight", pad_inches=0.25)
    return fig


def fig_open3(ctx, res, path=None):
    """One day: each half hour's range as bars, the live reading as a line, yesterday's report as a flat dashed line."""
    import matplotlib.pyplot as plt
    G.style()
    ob = ctx["ob"].dropna(subset=["gex_live"])
    spread = ob.groupby("date")["gex_live"].agg(lambda v: v.max() - v.min())
    full = ob.groupby("date").size() == 13
    day = spread[full].sort_values().index[int(0.9 * full.sum())]           # a day where the live reading moved a lot
    o = ob[ob["date"] == day].sort_values("bucket")
    xb = (570 + 30 * (o["bucket"] + 1)) / 60.0
    fig, ax = plt.subplots(figsize=(G.FIG_W, G.FIG_H)); ax2 = ax.twinx()
    ax.bar(xb, o["range_b"] * 100.0, width=0.4, color=G.DIM2, lw=0, label="range of each half hour, %")
    ax.set_ylabel("half-hour range, % of price"); ax.set_xticks(range(10, 17)); ax.set_xticklabels([f"{h:02d}:00" for h in range(10, 17)])
    ax2.plot(xb, o["gex_live"] / 1e9, marker="o", color=G.CYAN, lw=2, label="live gamma reading")
    ax2.axhline(o["gex_prev"].iloc[0] / 1e9, color=G.DIM, ls="--", lw=1.5, label="yesterday's report")
    ax2.axhline(0, color=G.FG, lw=0.6); ax2.set_ylabel("gamma exposure, billion $ per 1% move"); ax2.grid(False)
    ax.set_title(f"{pd.Timestamp(day):%d %B %Y}: the forecast that updates every half hour", loc="left", fontsize=14, pad=22)
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels(); ax.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=9)
    ax.text(0.0, 1.01, "lower reading, bigger bars: the live line moves with the day, the dashed line cannot", transform=ax.transAxes, color=G.DIM, fontsize=9.5, va="bottom")
    if path: fig.savefig(path, bbox_inches="tight", pad_inches=0.25)
    return fig


def fig_open4(ctx, res, path=None):
    """Two typical crossing days: price against the flip level, the rest-of-day range against the average of days that never crossed."""
    import matplotlib.pyplot as plt
    G.style()
    ob = ctx["ob"]; days = res["_days"]
    rest = ob[ob["bucket"] >= 1].groupby("date").agg(hi=("high", "max"), lo=("low", "min"), S0=("S0", "first"))
    rest["range_pct"] = (rest["hi"] - rest["lo"]) / rest["S0"] * 100.0
    picks = []
    for reg, lab, c in ((1, "crossed down", G.ORANGE), (-1, "crossed up", G.CYAN)):
        cross = days[(days["regime_prev"] == reg) & days["flip_bucket"].between(1, 6)]
        stay = days[(days["regime_prev"] == reg) & days["flip_bucket"].isna()]
        if len(cross) < 3:
            continue
        r_cross = rest["range_pct"].reindex(cross.index).dropna(); day = (r_cross - r_cross.median()).abs().idxmin()
        picks.append((day, lab, c, float(rest["range_pct"].reindex(stay.index).mean())))
    fig, axes = plt.subplots(1, len(picks), figsize=(9.6, 4.6), gridspec_kw={"wspace": 0.3}); axes = np.atleast_1d(axes)
    for ax, (day, lab, c, ref) in zip(axes, picks):
        b = _day_path(ctx, day); o = ob[ob["date"] == day].sort_values("bucket"); S0 = o["S0"].iloc[0]
        k = int(days.loc[day, "flip_bucket"]); t_cross = (570 + 30 * (k + 1)) / 60.0
        ax.plot(b["mod"] / 60.0, (b["close"] / S0 - 1.0) * 100.0, color=G.FG, lw=1.6)
        ax.axhline((o["flip_prev"].iloc[0] / S0 - 1.0) * 100.0, color=G.PURPLE, ls=":", lw=2, label="flip level from last night's book")
        ax.axvline(t_cross, color=c, ls="--", lw=1.5, label=f"{lab} at {int(t_cross):02d}:{int(round((t_cross % 1) * 60)):02d}")
        ax.set_title(f"{pd.Timestamp(day):%d %b %Y}: {lab}", loc="left", fontsize=13)
        ax.text(0.02, 0.97, f"a typical crossing day\nrange from 10:00 to the close: {rest.loc[day, 'range_pct']:.2f}%\ndays that never crossed averaged {ref:.2f}%", transform=ax.transAxes, color=G.FG, fontsize=9.5, va="top")
        ax.set_xticks(range(10, 17)); ax.set_xticklabels([f"{h:02d}" for h in range(10, 17)], fontsize=9); ax.set_ylabel("move from last night's close, %")
        ax.legend(loc="lower right", fontsize=8.5)
    if path: fig.savefig(path, bbox_inches="tight", pad_inches=0.25)
    return fig


def fig_open5(ctx, res, path=None):
    """The iceberg: what the daily report shows against what is alive at 10:00; and the last half hour in percent."""
    import matplotlib.pyplot as plt
    G.style()
    t8 = res["t8"]; bb = {int(x["bucket"]): x for x in t8["by_bucket"]}
    layer = res["_layer"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 4.6), gridspec_kw={"wspace": 0.35})
    book, same = bb[0]["book_abs_bn"], bb[0]["ugex_0dte_bn"]
    ax1.bar([0], [book], color=G.DIM, width=0.55, lw=0, label="options expiring later (in the daily report)")
    ax1.bar([1], [book], color=G.DIM, width=0.55, lw=0); ax1.bar([1], [same], bottom=[book], color=G.PURPLE, width=0.55, lw=0, label="options expiring today (not in the report)")
    ax1.set_xticks([0, 1]); ax1.set_xticklabels(["what the daily\nreport shows", "what is alive\nat 10:00"], fontsize=10)
    ax1.set_ylabel("gamma exposure, billion $ per 1% move (median day)"); ax1.set_title("What the daily report misses", loc="left", fontsize=14)
    ax1.text(1, book + same + 0.2, f"{same:.1f} bn", ha="center", color=G.PURPLE, fontsize=11); ax1.text(0, book + 0.2, f"{book:.1f} bn", ha="center", color=G.FG, fontsize=11)
    ax1.set_ylim(0, (book + same) * 1.3); ax1.legend(loc="upper left", fontsize=8.5)
    last = layer[layer["bucket"] == 11].dropna(subset=["ugex_0dte"]).set_index("date")
    ob = ctx["ob"]; r12 = ob[ob["bucket"] == 12].set_index("date")["range_b"] * 100.0
    last["last_range_pct"] = r12.reindex(last.index); last = last.dropna(subset=["last_range_pct"])
    q = pd.qcut(last["ugex_0dte"].rank(method="first"), 3, labels=["little same-day\ngamma at 15:30", "middle", "a lot of same-day\ngamma at 15:30"])
    v = last["last_range_pct"].groupby(q, observed=True).mean()
    ax2.bar(range(3), v.values, color=[G.ORANGE, G.DIM2, G.PURPLE], width=0.6, lw=0)
    for i, x in enumerate(v.values): ax2.text(i, x + 0.005, f"{x:.2f}%", ha="center", color=G.FG, fontsize=12)
    ax2.set_xticks(range(3)); ax2.set_xticklabels(v.index, fontsize=9.5); ax2.set_ylabel(f"{ctx['sym']} range 15:30 to 16:00, %"); ax2.set_ylim(0, v.max() * 1.3)
    ax2.set_title("The last half hour", loc="left", fontsize=14)
    if path: fig.savefig(path, bbox_inches="tight", pad_inches=0.25)
    return fig


def fig_open6(ctx, res, path=None):
    """A streak timeline: the regime as a coloured strip, longest streaks labelled."""
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    G.style()
    d = ctx["daily"]; runs = res["_runs"]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(G.FIG_W, G.FIG_H), sharex=True, gridspec_kw={"height_ratios": [2.2, 0.6], "hspace": 0.05})
    ax1.plot(d.index, d["close"], color=G.FG, lw=1.2); ax1.set_ylabel(f"{ctx['sym']} price, $")
    for _, r in runs.iterrows():
        ax2.axvspan(mdates.date2num(r["start"]) - 0.5, mdates.date2num(r["end"]) + 0.5, color=G.CYAN if r["sign"] > 0 else G.ORANGE, lw=0)
    ax2.set_yticks([]); ax2.set_ylabel("regime", rotation=0, labelpad=30, va="center")
    for sign, c in ((1, G.CYAN), (-1, G.ORANGE)):
        r = runs[runs["sign"] == sign].sort_values("length").iloc[-1]
        mid = r["start"] + (r["end"] - r["start"]) / 2
        ax1.annotate(f"{int(r['length'])} straight {'calm' if sign > 0 else 'wild'} days\n{r['start']:%d %b %Y} to {r['end']:%d %b %Y}", xy=(mid, d.loc[r["start"]:r["end"], "close"].min()),
                     xytext=(0, -55), textcoords="offset points", ha="center", color=c, fontsize=9.5, arrowprops={"arrowstyle": "-", "color": c, "lw": 1})
    ax1.set_title(f"{ctx['sym']}: the market stays in one mode for weeks (cyan calm, orange wild)", loc="left", fontsize=14)
    ax1.text(0.01, 0.95, f"tomorrow has the same colour as today {res['p_same_sign_next_day']:.0%} of the time", transform=ax1.transAxes, va="top", color=G.DIM, fontsize=10)
    if path: fig.savefig(path, bbox_inches="tight", pad_inches=0.25)
    return fig


def fig_open7(ctx, res, path=None):
    """Typical move size at four zoom levels, in percent, by yesterday's regime."""
    import matplotlib.pyplot as plt
    G.style()
    inf = G.in_force(ctx["daily"])
    rows = []
    for h in (5, 15, 30, 60):
        b = I.bucketize(ctx["bars"], h); b["in_force"] = inf.reindex(b["date"]).to_numpy()
        for reg, lab in ((1, "calm"), (-1, "wild")):
            rows.append({"h": h, "regime": lab, "typical_move_pct": float(b.loc[b["in_force"] == reg, "ret"].abs().median() * 100.0)})
    t = pd.DataFrame(rows).pivot(index="h", columns="regime", values="typical_move_pct")
    fig, ax = plt.subplots(figsize=(G.FIG_W, G.FIG_H)); xs = np.arange(len(t)); w = 0.36
    ax.bar(xs - w / 2, t["calm"], width=w, color=G.CYAN, lw=0, label="after a calm (positive gamma) close")
    ax.bar(xs + w / 2, t["wild"], width=w, color=G.ORANGE, lw=0, label="after a wild (negative gamma) close")
    for i, h in enumerate(t.index):
        ax.text(i - w / 2, t.loc[h, "calm"] + 0.003, f"{t.loc[h, 'calm']:.3f}%", ha="center", color=G.FG, fontsize=9); ax.text(i + w / 2, t.loc[h, "wild"] + 0.003, f"{t.loc[h, 'wild']:.3f}%", ha="center", color=G.FG, fontsize=9)
    ax.set_xticks(xs); ax.set_xticklabels([f"{h}-minute move" for h in t.index]); ax.set_ylabel(f"typical (median) {ctx['sym']} move, %")
    ax.set_title("The same day at four zoom levels", loc="left", fontsize=14); ax.legend(loc="upper left", fontsize=9.5); ax.set_ylim(0, t.max().max() * 1.3)
    if path: fig.savefig(path, bbox_inches="tight", pad_inches=0.25)
    return fig


def _candles(ax, st, days, color_up=G.FG, color_dn=G.DIM):
    import matplotlib.dates as mdates
    w = st.loc[days]; x = mdates.date2num(w.index)
    ax.vlines(x, w["low"], w["high"], color=G.FG, lw=1.2)
    ax.vlines(x, w["open"], w["close"], color=np.where(w["close"] >= w["open"], color_up, color_dn), lw=7)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))


def fig_open8(ctx, res, path=None):
    """The day after: candles for the day after the most positive and the most negative closes, and the averages."""
    import matplotlib.pyplot as plt
    G.style()
    d, st = ctx["daily"], ctx["stock"]
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 4.6), gridspec_kw={"wspace": 0.4, "width_ratios": [1, 1, 1.1]})
    for ax, pick, lab, c in ((axes[0], d["gex_net_usd"].idxmax(), "most positive close", G.CYAN), (axes[1], d["gex_net_usd"].idxmin(), "most negative close", G.ORANGE)):
        i = st.index.get_loc(pick); days = st.index[max(0, i - 2): i + 2]
        _candles(ax, st, days); nxt = st.index[i + 1]
        ax.axvspan(mdates_num(nxt) - 0.5, mdates_num(nxt) + 0.5, color=c, alpha=0.2, lw=0)
        ax.set_title(f"{lab}\n{pick:%d %b %Y}: {d.loc[pick, 'gex_net_usd'] / 1e9:+.1f} billion", loc="left", fontsize=11)
        ax.text(0.03, 0.04, f"next day range {st.loc[nxt, 'range_pct'] * 100:.2f}%", transform=ax.transAxes, color=c, fontsize=10)
        ax.set_ylabel(f"{ctx['sym']}, $"); ax.tick_params(axis="x", labelsize=8)
    out = res["_out"]; a = out.loc[out["regime"] > 0, "range_next"].mean() * 100; b = out.loc[out["regime"] < 0, "range_next"].mean() * 100
    axes[2].bar([0, 1], [a, b], color=[G.CYAN, G.ORANGE], width=0.6, lw=0)
    for i, v in enumerate((a, b)): axes[2].text(i, v + 0.02, f"{v:.2f}%", ha="center", color=G.FG, fontsize=12)
    axes[2].set_xticks([0, 1]); axes[2].set_xticklabels(["day after a\npositive close", "day after a\nnegative close"], fontsize=9.5)
    axes[2].set_ylabel("average next-day range, %"); axes[2].set_title("all days", loc="left", fontsize=11); axes[2].set_ylim(0, max(a, b) * 1.3)
    fig.suptitle("The day after", x=0.01, y=1.04, ha="left", fontsize=15, fontweight="bold", color=G.FG)
    if path: fig.savefig(path, bbox_inches="tight", pad_inches=0.25)
    return fig


def mdates_num(t):
    import matplotlib.dates as mdates
    return mdates.date2num(t)


def fig_open9(ctx, res, path=None):
    """The week after: typical daily move over the next five days, and the middle half of all five-day paths by regime."""
    import matplotlib.pyplot as plt
    G.style()
    out = res["_out"]; st = ctx["stock"]; close = st["close"]
    a = out.loc[out["regime"] > 0, "fvol_5d"].mean() / np.sqrt(252) * 100; b = out.loc[out["regime"] < 0, "fvol_5d"].mean() / np.sqrt(252) * 100
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 4.6), gridspec_kw={"wspace": 0.3, "width_ratios": [1, 1.5]})
    ax1.bar([0, 1], [a, b], color=[G.CYAN, G.ORANGE], width=0.6, lw=0)
    for i, v in enumerate((a, b)): ax1.text(i, v + 0.02, f"{v:.2f}%", ha="center", color=G.FG, fontsize=12)
    ax1.set_xticks([0, 1]); ax1.set_xticklabels(["week after a\npositive close", "week after a\nnegative close"], fontsize=9.5)
    ax1.set_ylabel("typical daily move over the next 5 days, %"); ax1.set_title("The week after", loc="left", fontsize=14); ax1.set_ylim(0, max(a, b) * 1.3)
    pos = st.index.get_indexer(out.index)
    for reg, c, lab in ((1, G.CYAN, "after a positive close"), (-1, G.ORANGE, "after a negative close")):
        rows_ = []
        for p0, r in zip(pos, out["regime"]):
            if r == reg and p0 + 5 < len(close):
                rows_.append([(close.iloc[p0 + k] / close.iloc[p0] - 1) * 100 for k in range(6)])
        m = np.array(rows_); lo, hi = np.percentile(m, 25, axis=0), np.percentile(m, 75, axis=0)
        ax2.fill_between(range(6), lo, hi, color=c, alpha=0.35, lw=0, label=f"{lab}: middle half of {len(m)} weeks, {hi[5] - lo[5]:.1f} points wide on day 5")
    ax2.axhline(0, color=G.DIM2, lw=0.8); ax2.set_xticks(range(6)); ax2.set_xticklabels(["close", "+1", "+2", "+3", "+4", "+5"]); ax2.set_xlabel("trading days after the close")
    ax2.set_ylabel("move from that close, %"); ax2.set_title("how wide the week can get", loc="left", fontsize=14); ax2.legend(loc="upper left", fontsize=8.5)
    if path: fig.savefig(path, bbox_inches="tight", pad_inches=0.25)
    return fig


def fig_open10(ctx, res, path=None):
    """Where the biggest moves sit: every daily return, the 20 largest marked by the colour of the prior close."""
    import matplotlib.pyplot as plt
    G.style()
    out = res["_out"]; st = ctx["stock"]
    nxt_idx = [st.index[st.index.get_loc(t) + 1] for t in out.index if st.index.get_loc(t) + 1 < len(st)]
    o = out.iloc[: len(nxt_idx)].copy(); o["next_day"] = nxt_idx
    fig, ax = plt.subplots(figsize=(G.FIG_W, G.FIG_H))
    ax.scatter(o["next_day"], o["ret_next"] * 100, s=6, color=G.DIM2, lw=0)
    big = o["absret_next"].nlargest(20).index
    for reg, c, lab in ((-1, G.ORANGE, "after a negative-gamma close"), (1, G.CYAN, "after a positive-gamma close")):
        sel = o.loc[big][o.loc[big, "regime"] == reg]
        ax.scatter(sel["next_day"], sel["ret_next"] * 100, s=70, color=c, lw=0, label=f"{len(sel)} of the 20 largest moves came {lab}", zorder=5)
    ax.axhline(0, color=G.FG, lw=0.6); ax.set_ylabel(f"{ctx['sym']} daily return, %")
    ax.set_title("Where the biggest moves live", loc="left", fontsize=14); ax.legend(loc="lower left", fontsize=9.5)
    ax.text(0.99, 0.96, f"negative-gamma closes are {res['base_share_negative']:.0%} of all days", transform=ax.transAxes, ha="right", va="top", color=G.DIM, fontsize=10)
    if path: fig.savefig(path, bbox_inches="tight", pad_inches=0.25)
    return fig
