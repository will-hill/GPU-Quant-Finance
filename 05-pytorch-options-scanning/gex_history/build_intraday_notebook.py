"""Build gex_intraday_spy.ipynb: what the live SPY book adds intraday over the daily print.

    uv run python gex_history/build_intraday_notebook.py
    uv run jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=3600 gex_intraday_spy.ipynb
The conclusions cell is filled from results/intraday_spy.json when it exists (run, execute, run, execute).
"""
import json
import os
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
OUT = Path(os.environ.get("GEX_INTRADAY_NB_OUT", HERE.parent / "gex_intraday_spy.ipynb"))
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text.strip("\n")))


def code(text):
    src = text.strip("\n")
    assert len(src.splitlines()) <= 25, f"code cell over 25 lines:\n{src}"
    cells.append(nbf.v4.new_code_cell(src))


def R():
    p = RESULTS / "intraday_spy.json"
    return json.loads(p.read_text()) if p.exists() else None


def zero_dte_line(r):
    t8, t8c = r.get("T8_zero_dte_layer"), r.get("T8b_zero_dte_controls")
    if not t8:
        return "7. 0DTE layer: (filled after execution)."
    bb = {int(x["bucket"]): x for x in t8["by_bucket"]}
    last = t8["last_half_hour_by_0dte_exposure_at_1530"]["all"]
    ct = t8c["last_half_hour_controlled_t"]; strata = t8c["last_half_hour_within_day_vol_strata"]; oi = t8c["same_day_expiry_oi_growth_on_last_day"]
    return (f"7. The 0DTE layer: same-day-expiry contracts carry a median {bb[0]['ugex_0dte_bn']:.1f} billion dollars of unsigned gamma exposure at 10:00 against "
            f"{bb[0]['book_abs_bn']:.1f} billion for the whole standing book, {bb[0]['share_0dte']:.0%} of the total, from positions as of the prior close alone, and that open interest roughly doubles on the last day before expiry (median ratio {oi['median_ratio_D_over_Dminus1']:.2f}). "
            f"Its magnitude damps the next half hour beyond the live book (joint regression t-statistic {t8['joint_regression_t']['u0dte_rank']:.1f} against {t8['joint_regression_t']['live_rank']:.1f} for the book; its sign adds nothing, t-statistic {t8['joint_regression_t']['s0dte_rank']:.1f}). "
            f"At 15:30 the top third of 0DTE exposure is followed by a 15:30 to 16:00 range of {last['high_0dte']:.2f} times the time-of-day median against {last['last_half_hour_range_adj_low_0dte']:.2f} for the bottom third (95% confidence interval of the difference [{last['ci_lo']:+.2f}, {last['ci_hi']:+.2f}]); "
            f"controlling for the day's range so far the effect shrinks to a t-statistic of {ct['u0dte_rank']:.1f}, holding in the narrow and wide thirds of days (confidence intervals [{strata['narrow so far']['ci_lo']:+.2f}, {strata['narrow so far']['ci_hi']:+.2f}] and [{strata['wide so far']['ci_lo']:+.2f}, {strata['wide so far']['ci_hi']:+.2f}]) and not in the middle third.")


def conclusions_md():
    r = R()
    if r is None:
        return "## Conclusions\n\n(filled after the first execution)\n\nEducational analysis, not a trading strategy."
    t6, t2, t3b, t4, t7 = r["T6_live_tracks_next_print"], r["T2_live_versus_stale"], r["T3b_near_flip_control_0.5pct"], r["T4_wall_touches"], r["T7_model_choice_american_vs_european"]
    t1 = {int(x["horizon_min"]): x for x in r["T1_intraday_vol_by_regime"]}
    nf = t3b["near_flip_cross_vs_stay"]; jr = t3b["joint_regression"]
    lines = ["## Conclusions", "",
             f"1. The live book tracks the next official print: sign agreement with the next day's GEX {t6['sign_agreement_live']:.1%} against {t6['sign_agreement_stale']:.1%} for the stale value, level correlation {t6['corr_level_live']:.2f} against {t6['corr_level_stale']:.2f}; of the {t6['days_sign_changed']} days on which the sign changed, the live value caught {t6['live_caught_change']:.0%} by the close.",
             f"2. Realized vol by regime at intraday horizons: under negative GEX it is {t1[5]['ratio_neg_over_pos']:.2f} times the positive-regime value from 5-minute returns and {t1[60]['ratio_neg_over_pos']:.2f} times from 60-minute returns, p-value below 0.001 at each horizon.",
             f"3. The live value predicts the next half hour better than the stale print: Spearman with the next bucket's adjusted range {t2['spearman_live']:+.2f} against {t2['spearman_stale']:+.2f}; in a joint rank regression the live value carries a t-statistic of {jr['t_live']:.1f} and the stale value {jr['t_stale']:.1f}. Within a day the live value's movement does not pick out individual buckets (within-day Spearman {t2['spearman_increment_within_day']:+.2f}); what it adds is an updated level for the day.",
             f"4. Intraday flips: the live sign differs from the prior close on {t2['share_days_with_any_flip']:.0%} of days. Among days that started within 0.5% of the flip, crossing up from a negative start cut the rest-of-day range to {nf['neg_to_pos']['post_range_adj_cross']:.2f} times the time-of-day median against {nf['neg_to_pos']['post_range_adj_stay']:.2f} times when it stayed (confidence interval [{nf['neg_to_pos']['ci_lo']:+.2f}, {nf['neg_to_pos']['ci_hi']:+.2f}]); crossing down from a positive start raised it to {nf['pos_to_neg']['post_range_adj_cross']:.2f} times against {nf['pos_to_neg']['post_range_adj_stay']:.2f} times (confidence interval [{nf['pos_to_neg']['ci_lo']:+.2f}, {nf['pos_to_neg']['ci_hi']:+.2f}]).",
             f"5. Walls as intraday levels: the mean 30-minute return after the first touch is {t4['call']['after_touch_30m_bp_mean']:+.1f} basis points for the call wall (confidence interval [{t4['call']['after_touch_30m_ci'][0]:+.1f}, {t4['call']['after_touch_30m_ci'][1]:+.1f}], n = {t4['call']['n_touch_30m']}) and {t4['put']['after_touch_30m_bp_mean']:+.1f} basis points for the put wall (confidence interval [{t4['put']['after_touch_30m_ci'][0]:+.1f}, {t4['put']['after_touch_30m_ci'][1]:+.1f}], n = {t4['put']['n_touch_30m']}), indistinguishable from the placebo levels.",
             f"6. Model choice: solving the same chain as European instead of American changes the daily sign on {t7['days_sign_differs']} of 751 days and moves net GEX by a median {t7['median_abs_diff_bn']:.2f} billion dollars; American gamma is {t7['put_itm_median_ratio']:.3f} times the European value for in-the-money puts and equal for calls.",
             zero_dte_line(r),
             "", "Educational analysis, not a trading strategy."]
    return "\n".join(lines)


md("""
# SPY real-time GEX: what the live book adds intraday

The daily notebook showed that the sign and level of net GEX at the close describe the next session's range. Open interest only arrives once a day, so a continuous scanner cannot see new positions; what it does is evaluate the previous close's book at the current spot, vol and time: the live book. This notebook measures what that live value is worth on SPY: whether it tracks the next official print, whether it predicts the next half hour better than the stale value, what an intraday crossing of the flip level does to the rest of the day, whether the walls act as intraday levels, and how much the exercise model matters.
""")

md("""
## Definitions

- Live GEX at time t: the previous close's solved chain (same open interest, each contract's IV fixed) re-evaluated at spot S_t. Computed from the daily profile on a spot grid from 0.90 S to 1.10 S in 0.25% steps and interpolated. Time decay inside the day is ignored.
- Regime at the prior close: sign of the official net GEX of D-1. Live regime: sign of the live value.
- Buckets: 30 minutes, 13 per session from the traded 1-minute bars. Adjusted range: the bucket's (high - low) / prior close divided by the median for that time of day, which removes the intraday U-shape.
- Flip level: zero crossing of the profile nearest to spot. Walls: strikes with the largest positive and most negative net GEX at the prior close.
""")

code("""
import json
import sys
import time
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore", category=RuntimeWarning)
sys.path.insert(0, "gex_history")
import gexlib as G
import intraday as I
pd.set_option("display.width", 220); pd.set_option("display.max_columns", 40)
stock, sofr, days = G.load_stock(), G.load_sofr(), G.study_days()
daily = pd.read_csv(G.RESULTS / "spy_gex_daily.csv", parse_dates=["date"]).set_index("date")
chain, shares = G.prep_all(days, stock, sofr, verbose=False)
sol, timings = G.solve_chain(chain, target="cpu", dtype="fp64")
print(f"{len(daily)} days; {timings['n_contracts']:,} contracts solved in {timings['t_iv_s'] + timings['t_greeks_s']:.1f}s on the CPU")
""")

md("""
## The live profile

One profile per day: net GEX of the solved chain at 81 spot points. The live value at any intraday spot is a lookup on it. On the RTX PRO 6000 Blackwell the same 81-point re-evaluation of the whole 2.4M-contract history is a fraction of a second in fp32; here the CPU fp64 build does it in about two minutes for the record.
""")

code("""
t0 = time.perf_counter()
prof = I.daily_profiles(sol, verbose=False)
prof.to_parquet(G.CACHE / "spy_profiles.parquet")
print(f"profiles: {prof.shape[0]} days x {prof.shape[1]} spot points in {time.perf_counter() - t0:.0f}s (CPU fp64)")
print(f"largest relative gap between the profile at spot and the daily series: {((prof[1.0].reindex(daily.index) - daily.gex_net_usd).abs() / daily.gex_net_usd.abs()).max():.1e}")
""")

md("""
## The live value against the next official print

At the close of D the scanner holds the D-1 book live at S_D. The next morning's official print uses the new open interest and the new closing quotes. If the live value tracks it, the spot move explains most of the day-to-day change in GEX and the live value carries information the stale print does not.
""")

code("""
t6 = I.t6_live_tracks_next_print(daily, prof)
print(json.dumps(t6, indent=1))
_ = I.fig_live_versus_stale(daily, prof, t6, G.FIGURES / "f6_live_versus_stale.png")
""")

md("""
## The regime inside the day

Per-day realized vol from intraday log returns at 5, 15, 30 and 60 minutes, averaged by the regime at the previous close, with a bootstrap confidence interval of the difference and a Mann-Whitney p-value.
""")

code("""
bars = I.load_bars("spy")
t1 = I.t1_intraday_vol(bars, daily)
display(t1.style.format({"rv_pos": "{:.3f}", "rv_neg": "{:.3f}", "median_pos": "{:.3f}", "median_neg": "{:.3f}", "ratio_neg_over_pos": "{:.2f}", "diff": "{:+.4f}", "ci_lo": "{:+.4f}", "ci_hi": "{:+.4f}", "p_mwu": "{:.2e}"}))
_ = I.fig_intraday_vol(t1, G.FIGURES / "f8_intraday_vol.png")
""")

md("""
## Live against stale, bucket by bucket

For every 30-minute bucket b: the stale value (prior close), the live value at the bucket close, and the next bucket's adjusted range. Pooled Spearman for each, then a joint rank regression with standard errors clustered by day. The within-day column asks whether the live value's movement inside a day picks out the wide buckets once the day's mean is removed.
""")

code("""
ob = I.attach_regime(I.bucketize(bars, 30), daily, prof)
t2 = I.t2_live_versus_stale(ob)
print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in t2.items() if k != "flip_vs_same_by_prev_regime"})
pd.DataFrame(t2["flip_vs_same_by_prev_regime"]).T.style.format("{:.3f}")
""")

md("""
## Intraday flips

29% of days see the live sign leave the prior close's sign at some point. The event study lines up buckets on the first crossing. The control restricts to days that started within 0.5% of the flip, where the stale print alone already places spot near the boundary, and compares the rest of the day on days that crossed against days that did not.
""")

code("""
t3, flipdays = I.t3_intraday_flips(ob)
t3b = I.t3b_near_flip(ob, flipdays, near_pct=0.5)
t3b_1 = I.t3b_near_flip(ob, flipdays, near_pct=1.0)
print("share of days with an intraday flip:", round(t3["share_days_flipped"], 3), "| first flip bucket counts:", t3["flip_bucket_distribution"])
print("matched on move size:", {k: (round(v["post_flip_minus_reference_matched"], 3), v["n_flip_days"]) for k, v in t3["by_direction"].items()})
print("joint regression:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in t3b["joint_regression"].items()})
display(pd.DataFrame(t3b["near_flip_cross_vs_stay"]).T.style.format("{:.3f}"))
_ = I.fig_intraday_flip(ob, flipdays, t3b, G.FIGURES / "f7_intraday_flip.png")
""")

md("""
## Walls as intraday levels

First intraday touch of the prior close's call wall from below and put wall from above, within 5 basis points. Signed return over the next 30 and 60 minutes, against the same measurement on placebo levels 5 dollars above and below the wall. The result is reported either way.
""")

code("""
t4 = I.t4_wall_touches(bars, daily)
print(json.dumps({s: {k: v for k, v in t4[s].items() if k != "placebo"} for s in ("call", "put")}, indent=1))
print("inside both walls all day:", round(t4["day_inside_both_walls_share"], 3), t4["day_inside_both_walls_by_prev_regime"])
_ = I.fig_wall_touches(t4, G.FIGURES / "f10_wall_touches.png")
""")

md("""
## Time of day
""")

code("""
t5 = I.t5_time_of_day(ob)
display(t5.style.format("{:.3f}"))
_ = I.fig_time_of_day(t5, G.FIGURES / "f9_time_of_day.png")
""")

md("""
## The 0DTE layer

Contracts expiring on day D are gone at the close and never enter the daily print, yet they are alive all session. Their open interest is in the open interest file dated D (positions as of the D-1 close, the freshest available during D), their IV comes from the D-1 solve of the same contracts, and gamma is evaluated at every bucket close with the true remaining time. Positions opened during D are invisible, so this layer is a lower bound. Tests: its size against the standing book by time of day, whether its magnitude adds to the next-bucket range prediction, and the 15:30 to 16:00 range against the 0DTE exposure at 15:30, with the day's own range so far as a control.
""")

code("""
t0 = time.perf_counter()
layer = I.zero_dte_layer(sol, daily, ob)
t8 = I.t8_zero_dte(layer); t8c = I.t8_controls(layer, sol, daily)
print(f"{t8['n_days']} days, median {t8['median_contracts_per_day']:.0f} same-day contracts within 3% of the prior close, {time.perf_counter() - t0:.0f}s")
print("Spearman with the next bucket's range:", {k: round(v, 3) for k, v in t8["spearman_next_range"].items()})
print("joint regression t-statistics (next bucket):", {k: round(v, 1) for k, v in t8["joint_regression_t"].items()})
print("last half hour, controlled for range so far and regime, t-statistics:", {k: round(v, 1) for k, v in t8c["last_half_hour_controlled_t"].items()})
print("same-day-expiry open interest on D over D-1:", {k: round(v, 2) for k, v in t8c["same_day_expiry_oi_growth_on_last_day"].items()})
display(pd.DataFrame(t8["by_bucket"]).set_index("bucket").style.format("{:.3f}"))
display(pd.DataFrame(t8["last_half_hour_by_0dte_exposure_at_1530"]).T.style.format("{:.3f}"))
display(pd.DataFrame(t8c["last_half_hour_within_day_vol_strata"]).T.style.format("{:.3f}"))
_ = I.fig_zero_dte(t8, G.FIGURES / "f14_zero_dte.png")
""")

md("""
## The exercise model's effect

The same chain solved with European exercise (Black-Scholes-Merton) instead of the American engine: daily sign agreement, median absolute difference in net GEX, and the gamma ratio by moneyness.
""")

code("""
t7 = I.t7_model_choice(sol, daily)
print(json.dumps(t7, indent=1))
res = {"symbol": "SPY", "buckets_minutes": 30, "T1_intraday_vol_by_regime": t1.reset_index().to_dict("records"), "T2_live_versus_stale": t2,
       "T3_intraday_flips": t3, "T3b_near_flip_control_0.5pct": t3b, "T3b_near_flip_control_1pct": t3b_1, "T4_wall_touches": t4,
       "T5_time_of_day_range_pct": t5.reset_index().to_dict("records"), "T6_live_tracks_next_print": t6, "T7_model_choice_american_vs_european": t7,
       "T8_zero_dte_layer": t8, "T8b_zero_dte_controls": t8c}
G.save_json(res, G.RESULTS / "intraday_spy.json")
""")

md(conclusions_md())

nb = nbf.v4.new_notebook(); nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}}
for c in cells:
    for bad in ("—", "–"):
        assert bad not in c["source"], f"dash in cell: {c['source'][:80]}"
nbf.write(nb, OUT)
print(f"wrote {OUT} with {len(cells)} cells ({sum(c['cell_type'] == 'code' for c in cells)} code)")
