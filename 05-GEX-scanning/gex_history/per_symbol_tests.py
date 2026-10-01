"""Run every demonstrated GEX indication for a list of symbols and tabulate one statistic per row.

    uv run python gex_history/per_symbol_tests.py SPY QQQ IWM NVDA TSLA AAPL AMZN META MSFT AMD
Writes results/cross_section/per_symbol_tests.json. Cache only; needs {sym}_daily.csv (cross_section.run_symbol),
{sym}_profiles.parquet and {sym}_1m_*.parquet. Single names list weekly expirations, so the 0DTE row has few days.
"""
import json
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr

import cross_section as X
import gexlib as G
import intraday as I

warnings.filterwarnings("ignore")


def run(sym: str) -> dict:
    sym = sym.lower()
    daily = X.load_daily(sym) if (X.XS / f"{sym}_daily.csv").exists() else pd.read_csv(G.RESULTS / "spy_gex_daily.csv", parse_dates=["date"]).set_index("date")
    stock = G.load_stock(sym)
    prof = pd.read_parquet(G.CACHE / f"{sym}_profiles.parquet"); prof.columns = prof.columns.astype(float)
    bars = I.load_bars(sym)
    ob = I.attach_regime(I.bucketize(bars, 30), daily, prof)
    r = {"symbol": sym.upper(), "days": int(len(daily)), "share_negative": float((daily["regime"] < 0).mean())}
    # row 1: live value tracks the next print
    t6 = I.t6_live_tracks_next_print(daily, prof)
    r["r1_sign_agree_live"], r["r1_sign_agree_stale"], r["r1_corr_live"], r["r1_corr_stale"] = t6["sign_agreement_live"], t6["sign_agreement_stale"], t6["corr_level_live"], t6["corr_level_stale"]
    # row 2: live level at 10:00 -> rest of day
    W = ob.pivot(index="date", columns="bucket", values="range_adj"); Gx = ob.pivot(index="date", columns="bucket", values="gex_live")
    rest = W[[c for c in W.columns if c > 0]].mean(axis=1); x = Gx[0]; m = x.notna() & rest.notna()
    q = pd.qcut(x[m].rank(method="first"), 3, labels=False)
    r["r2_spearman_1000_rest"] = float(spearmanr(x[m], rest[m]).correlation)
    r["r2_rest_low_third"], r["r2_rest_high_third"] = float(rest[m][q == 0].mean()), float(rest[m][q == 2].mean())
    # row 3: any bucket -> next 30 min
    t2 = I.t2_live_versus_stale(ob); r["r3_spearman_live"], r["r3_spearman_stale"], r["r3_share_days_flip"] = t2["spearman_live"], t2["spearman_stale"], t2["share_days_with_any_flip"]
    # row 4: flip crossing near the flip (1%)
    t3, days = I.t3_intraday_flips(ob); t3b = I.t3b_near_flip(ob, days, near_pct=1.0)
    for key, lab in (("neg_to_pos", "up"), ("pos_to_neg", "down")):
        v = t3b["near_flip_cross_vs_stay"].get(key, {})
        r[f"r4_{lab}_cross"], r[f"r4_{lab}_stay"], r[f"r4_{lab}_n"] = v.get("post_range_adj_cross", np.nan), v.get("post_range_adj_stay", np.nan), int(v.get("n_cross", 0))
        r[f"r4_{lab}_ci_excl0"] = bool(v.get("ci_hi", 1) < 0 or v.get("ci_lo", -1) > 0) if "ci_lo" in v else None
    # row 5: 0DTE layer (needs the solved chain)
    try:
        sofr = G.load_sofr(); roots = ["spx", "spxw"] if sym == "spx" else [sym]
        chain = pd.concat([G.prep_all(G.study_days(rt), stock, sofr, verbose=False, symbol=rt, q=G.q_for(sym))[0] for rt in roots], ignore_index=True)
        sol, _ = G.solve_chain(chain)
        layer = I.zero_dte_layer(sol, daily, ob, symbol=roots[0])
        t8 = I.t8_zero_dte(layer)
        last = layer[layer["bucket"] == 11].dropna(subset=["next_range_adj"])
        r["r5_days_with_0dte"] = int(layer["date"].nunique()); r["r5_share_0dte_of_book_1000"] = float(t8["by_bucket"][0]["share_0dte"])
        r["r5_spearman_1530_last"] = float(spearmanr(last["ugex_0dte"], last["next_range_adj"]).correlation) if len(last) > 20 else np.nan
        a = t8["last_half_hour_by_0dte_exposure_at_1530"].get("all", {}); r["r5_last_low"], r["r5_last_high"], r["r5_last_p"] = a.get("last_half_hour_range_adj_low_0dte", np.nan), a.get("high_0dte", np.nan), a.get("p_mwu", np.nan)
    except Exception as e:  # noqa: BLE001
        r["r5_error"] = f"{type(e).__name__}: {str(e)[:80]}"
    # row 6: persistence
    s = daily["regime"].to_numpy(); r["r6_persistence"] = float((s[:-1] == s[1:]).mean())
    # row 7: intraday realized vol by daily regime (30-min returns)
    t1 = I.t1_intraday_vol(bars, daily, horizons=(30,)); r["r7_rv_ratio_neg_over_pos"], r["r7_p"], r["r7_n_neg"] = float(t1.loc[30, "ratio_neg_over_pos"]), float(t1.loc[30, "p_mwu"]), int(t1.loc[30, "n_neg"])
    # rows 8, 9, 11: next-day outcomes by sign and by own tercile
    out = G.next_day_outcomes(daily, stock)
    lo_q, hi_q = out["gex"].quantile([1 / 3, 2 / 3]); lo, hi = out[out["gex"] <= lo_q], out[out["gex"] >= hi_q]
    pos, neg = out[out["regime"] > 0], out[out["regime"] < 0]
    r["r8_range_pos"], r["r8_range_neg"], r["r8_n_neg"] = float(pos["range_next"].mean()), float(neg["range_next"].mean()), int(len(neg))
    r["r8_range_top_third"], r["r8_range_bottom_third"] = float(hi["range_next"].mean()), float(lo["range_next"].mean())
    r["r8_p_tercile"] = float(mannwhitneyu(lo["range_next"].dropna(), hi["range_next"].dropna()).pvalue)
    r["r8_spearman_gex_next_range"] = float(spearmanr(out["gex"], out["range_next"], nan_policy="omit").correlation)
    r["r9_fvol_top_third"], r["r9_fvol_bottom_third"] = float(hi["fvol_5d"].mean()), float(lo["fvol_5d"].mean())
    r["r9_spearman_gex_fvol"] = float(spearmanr(out["gex"], out["fvol_5d"], nan_policy="omit").correlation)
    big = out["absret_next"].nlargest(20).index
    r["r11_share_top20_moves_bottom_third"] = float((out.loc[big, "gex"] <= lo_q).mean()); r["r11_share_top20_moves_negative"] = float((out.loc[big, "regime"] < 0).mean())
    return r


if __name__ == "__main__":
    syms = sys.argv[1:] or ["SPY", "QQQ", "IWM", "NVDA", "TSLA", "AAPL", "AMZN", "META", "MSFT", "AMD"]
    rows = []
    for s in syms:
        try:
            rows.append(run(s)); print(s, "ok", flush=True)
        except Exception as e:  # noqa: BLE001
            print(s, "FAILED", type(e).__name__, str(e)[:200], flush=True)
    X.XS.mkdir(parents=True, exist_ok=True)
    G.save_json(rows, X.XS / "per_symbol_tests.json")
    print("wrote", X.XS / "per_symbol_tests.json")
