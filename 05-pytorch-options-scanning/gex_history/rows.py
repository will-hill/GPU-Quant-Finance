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
    return I.t6_remark_tracks_next_print(ctx["daily"], ctx["prof"])


def fig1(ctx, res, path=None):
    return I.fig_remark_vs_stale(ctx["daily"], ctx["prof"], res, path)


def row2(ctx):
    ob = ctx["ob"]; W = ob.pivot(index="date", columns="bucket", values="range_adj"); Gx = ob.pivot(index="date", columns="bucket", values="gex_rt")
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
    return {"spearman_by_time_and_horizon (re-mark, stale)": horizons, "rest_of_day_by_1000_tercile": terc.round(3).to_dict(),
            "rest_of_day_by_sign_and_size": {f"{a} {b}": round(v, 3) for (a, b), v in by_size_sign.items()},
            "p_low_vs_high": float(mannwhitneyu(y[q == "low third"], y[q == "high third"]).pvalue), "n_days": int(len(x))}


def fig2(ctx, res, path=None):
    import matplotlib.pyplot as plt
    G.style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 4.6), gridspec_kw={"wspace": 0.3})
    t = res["rest_of_day_by_1000_tercile"]; keys = ["low third", "middle", "high third"]
    ax1.bar(range(3), [t[k] for k in keys], color=[G.ORANGE, G.DIM2, G.CYAN], width=0.6, lw=0)
    for i, k in enumerate(keys): ax1.text(i, t[k] + 0.03, f"{t[k]:.2f}x", ha="center", color=G.FG)
    ax1.set_xticks(range(3)); ax1.set_xticklabels(keys, fontsize=10); ax1.set_xlabel("re-marked GEX at 10:00, terciles")
    ax1.set_ylabel("rest-of-day range, multiple of time-of-day median"); ax1.set_title(f"{ctx['sym']}: level at 10:00 sets the day", loc="left", fontsize=14)
    s = res["rest_of_day_by_sign_and_size"]; order = ["negative large", "negative medium", "negative small", "positive small", "positive medium", "positive large"]
    vals = [s.get(k, np.nan) for k in order]
    ax2.bar(range(6), vals, color=[G.ORANGE] * 3 + [G.CYAN] * 3, width=0.6, lw=0)
    for i, v in enumerate(vals): ax2.text(i, v + 0.03, f"{v:.2f}x", ha="center", color=G.FG, fontsize=9)
    from matplotlib.patches import Patch
    ax2.set_xticks(range(6)); ax2.set_xticklabels([k.split()[1] for k in order], fontsize=10); ax2.set_xlabel("size of |GEX| at 10:00, terciles")
    ax2.legend(handles=[Patch(color=G.ORANGE, label="negative at 10:00"), Patch(color=G.CYAN, label="positive at 10:00")], loc="upper right", fontsize=9)
    ax2.set_ylim(0, max(vals) * 1.25)
    ax2.set_title("by sign and size of |GEX| at 10:00", loc="left", fontsize=14)
    if path: fig.savefig(path, bbox_inches="tight", pad_inches=0.25)
    return fig


def row3(ctx):
    import statsmodels.api as sm
    ob = ctx["ob"]; t2 = I.t2_remark_vs_stale(ob)
    d = ob.dropna(subset=["next_range_adj", "gex_rt", "gex_prev"])
    X_ = sm.add_constant(pd.DataFrame({"stale_rank": d["gex_prev"].rank(pct=True), "remark_rank": d["gex_rt"].rank(pct=True)}))
    fit = sm.OLS(d["next_range_adj"].to_numpy(), X_).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d["date"])[0]})
    by_bucket = {int(b): (float(spearmanr(g["gex_rt"], g["next_range_adj"]).correlation), float(spearmanr(g["gex_prev"], g["next_range_adj"]).correlation)) for b, g in d.groupby("bucket") if len(g) > 50}
    return {"spearman_remark": t2["spearman_remark"], "spearman_stale": t2["spearman_stale"], "within_day_increment": t2["spearman_increment_within_day"],
            "joint_regression_t": {"remark": float(fit.tvalues["remark_rank"]), "stale": float(fit.tvalues["stale_rank"])}, "n_buckets": t2["n_buckets"],
            "share_days_with_flip": t2["share_days_with_any_flip"], "spearman_by_bucket (re-mark, stale)": by_bucket}


def fig3(ctx, res, path=None):
    import matplotlib.pyplot as plt
    G.style()
    bb = res["spearman_by_bucket (re-mark, stale)"]; ks = sorted(bb)
    fig, ax = plt.subplots(figsize=(G.FIG_W, G.FIG_H))
    ax.plot(ks, [bb[k][0] for k in ks], marker="o", color=G.CYAN, lw=2, label="re-marked at the bucket close")
    ax.plot(ks, [bb[k][1] for k in ks], marker="o", color=G.DIM, lw=2, label="stale previous-close print")
    ax.set_xticks(ks); ax.set_xticklabels([f"{(570 + 30 * (k + 1)) // 60:02d}:{(570 + 30 * (k + 1)) % 60:02d}" for k in ks], fontsize=9)
    ax.set_xlabel("bucket close, ET"); ax.set_ylabel("Spearman with the next bucket's adjusted range")
    ax.set_title(f"{ctx['sym']}: the re-mark's edge grows through the day", loc="left"); ax.legend(loc="lower right")
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
    return {"t8": I.t8_zero_dte(layer), "controls": I.t8_controls(layer, ctx["sol"], ctx["daily"], symbol=ctx["sym"].lower())}


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
