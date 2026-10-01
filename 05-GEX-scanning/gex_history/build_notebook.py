"""Build gex_history_spy.ipynb (top level of 05-pytorch-options-scanning) from short cells.

Two passes: run once, execute the notebook (which writes gex_history/results/*), run again so the
conclusions cell carries the measured numbers, execute again. Every cell runs from the cache.

    uv run python gex_history/build_notebook.py
    uv run jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=3600 gex_history_spy.ipynb
"""
import json
import os
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
OUT = Path(os.environ.get("GEX_NB_OUT", HERE.parent / "gex_history_spy.ipynb"))

cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text.strip("\n")))


def code(text):
    src = text.strip("\n")
    assert len(src.splitlines()) <= 25, f"code cell over 25 lines:\n{src}"
    cells.append(nbf.v4.new_code_cell(src))


# ----------------------------------------------------------------------------- conclusions (pass 2 fills numbers)
def _stats():
    rs = RESULTS / "regime_stats.json"
    return json.loads(rs.read_text()) if rs.exists() else None


def lead_md():
    st = _stats()
    if st is None:
        return "(filled after the first execution)"
    xc = {int(r["k"]): r for r in st["cross_correlation_gex_vs_range"]}
    lead = sum(xc[k]["pearson"] for k in range(1, 6)) / 5
    react = sum(xc[k]["pearson"] for k in range(-5, 0)) / 5
    return (f"Result: the correlation of net GEX on D with the range on D+1 is {xc[1]['pearson']:+.2f} (Spearman {xc[1]['spearman']:+.2f}), "
            f"with the range on D-1 it is {xc[-1]['pearson']:+.2f} (Spearman {xc[-1]['spearman']:+.2f}), and at k = 0 it is {xc[0]['pearson']:+.2f}; "
            f"the k = 1..5 side averages {lead:+.2f} against {react:+.2f} for k = -5..-1. The leading side is the stronger one, but both sides are "
            f"large because GEX and the daily range are both persistent, so the sign describes the vol regime that is already in place and carries "
            f"into the next session rather than an independent trigger.")


def conclusions_md():
    st = _stats()
    if st is None:
        return "## Conclusions\n\n(filled after the first execution)\n\nEducational analysis, not a trading strategy."
    tests = {t["test"]: t for t in st["tests_sign_cut"]}
    pin = {(r["days"], str(r["regime in force"])): r for r in st["pinning"]}
    share_neg = st["share_negative_days"]

    def s(name, pct=True, sign="", digits=2):
        t = tests[name]
        f = (lambda v: f"{v:{sign}.{digits}%}") if pct else (lambda v: f"{v:{sign}.3f}")
        pv = "p-value below 0.001" if t["p_mwu"] < 0.0005 else f"p-value {t['p_mwu']:.3f}"
        return (f"{f(t['mean_pos'])} after a positive-GEX close against {f(t['mean_neg'])} after a negative one, difference {f(t['diff'])}, "
                f"95% confidence interval [{f(t['ci_lo'])}, {f(t['ci_hi'])}], {pv}")
    pa, pn = pin[("all days", "1")], pin[("all days", "-1")]
    lines = ["## Conclusions", "",
             f"1. Moves are damped after positive GEX: mean next-day range {s('range')} ({tests['range']['verdict']}); mean absolute next-day return {s('move')} ({tests['move']['verdict']}).",
             f"2. Realized vol is lower after positive GEX: mean annualized 5-day forward vol {s('forward vol', digits=1)} ({tests['forward vol']['verdict']}).",
             f"3. Trends do not extend more under negative GEX: the next-day continuation rate is {s('continuation', digits=1)}, the reverse of the claim ({tests['continuation']['verdict']}), and the lag-1 autocorrelation of daily returns is {tests['autocorrelation']['mean_pos']:+.3f} under positive GEX against {tests['autocorrelation']['mean_neg']:+.3f} under negative, again the reverse ({tests['autocorrelation']['verdict']}).",
             f"4. Gaps fade slightly more often after positive GEX: {s('gap fade', digits=1)} ({tests['gap fade']['verdict']}).",
             f"5. Pinning: the close sits {pa['mean dist to wall %']:.2f}% from the positive wall under positive GEX against {pn['mean dist to wall %']:.2f}% under negative, but the close moved toward the wall on {pa['share of days closer to the wall than the previous close']:.1%} of positive-GEX days against {pn['share of days closer to the wall than the previous close']:.1%} of negative-GEX days (not supported).",
             ]
    it = st.get("intraday_ac1_by_regime")
    if it:
        a = {int(r["regime"]): r for r in it}
        lines.append(f"6. Intraday reversal: mean lag-1 autocorrelation of 30-minute returns {a[1]['mean']:+.3f} under positive GEX (95% confidence interval [{a[1]['ci_lo']:+.3f}, {a[1]['ci_hi']:+.3f}], n = {a[1]['n']}) against {a[-1]['mean']:+.3f} under negative ([{a[-1]['ci_lo']:+.3f}, {a[-1]['ci_hi']:+.3f}], n = {a[-1]['n']}), Mann-Whitney p-value {st['intraday_mwu_p']:.3f} (direction only, the intervals overlap).")
    lines += ["", f"Net GEX was negative on {share_neg:.0%} of the {st['n_days']} days, so the positive-GEX sample is the smaller one. " + lead_md(),
              "", "Educational analysis, not a trading strategy."]
    return "\n".join(lines)



# ----------------------------------------------------------------------------- title, convention, provenance
md("""
# SPY net gamma exposure, 2023 to 2026: the sign of GEX and the next session

This notebook builds a daily history of SPY net gamma exposure (GEX) from Theta Data end-of-day option quotes and open interest, with gamma from the `engine_alo` American option engine. It then tests, by rule and without hand-picked dates, whether the sign of GEX at the close says anything about the next session: the size of the range, the size of the move, realized volatility over the following week, continuation against reversal, and whether gaps fade. Episodes of both regimes are selected by rule and charted. The result is reported either way.
""")

md("""
## Sign convention and caveats

- GEX per contract = gamma x open interest x 100 x S x S x 0.01, in dollars per 1% move of SPY. Calls count positive, puts negative. This assumes dealers are long the calls and short the puts that customers hold. Dealer positioning is inferred, not observed.
- Open interest is published once a day at about 06:30 ET and reflects positions at the previous close. The GEX for day D uses the record available during D. Positions opened during D, including all 0DTE flow, are not in it.
- SPY only. SPX and ES options carry most of the index gamma and are not included.
- Gamma comes from a Black-Scholes-Merton American model with constant rate and dividend yield, at the implied vol that reproduces the closing mid. r is SOFR on the day, q is a flat 1.2% (SPY trailing yield, an assumption).
- "Model value" is what the engine outputs. "Mid" is the input.
""")

md("""
## Data

Window: 2023-09-15 to 2026-09-14, the last completed session when the pull ran. Endpoints, one request per trading day with `expiration="*"` and `max_dte=120`: `option_history_eod` (quotes at the 17:15 ET report), `option_history_open_interest` (the 06:30 ET record), `option_history_greeks_eod` (vendor Greeks, used only in the cross-check). SPY daily bars from `stock_history_eod`, SOFR from `interest_rate_history_eod`. This account's rate history starts 2024-01-01, so the 2023 days use the first SOFR record (5.38%). The daily high and low are the narrower of the EOD report and the 1-minute session range, with the open and close folded in: the EOD bar for 2026-02-02 carries a low of 69.005, a decimal-shift print, and the intersection removes prints like it from either source. `gex_history/download.py` writes one parquet per endpoint per day under `gex_history/cache/`; this notebook never touches the network.
""")

code("""
import json
import sys
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore", category=RuntimeWarning)
sys.path.insert(0, "gex_history")
import gexlib as G
pd.set_option("display.width", 220); pd.set_option("display.max_columns", 40)
log = json.loads((G.RESULTS / "download_log.json").read_text())
print(f"window {log['window']['start']} to {log['window']['end']}: {log['trading_days_complete']} trading days with quotes, open interest and Greeks")
print(f"holidays with no data: {len(log['holidays_no_data'])}; partial days: {len(log['partial_days'])}; cache {log['cache_mb']} MB")
print("rows per day:", {k: (v['min'], v['median'], v['max']) for k, v in log['rows_per_day'].items()})
stock, sofr, days = G.load_stock(verbose=True), G.load_sofr(), G.study_days()
print(f"{len(days)} study days {days[0]} to {days[-1]}; SPY bars from {stock.index.min():%Y-%m-%d}; SOFR {sofr.index.min():%Y-%m-%d} to {sofr.index.max():%Y-%m-%d}, last {sofr.iloc[-1]:.4f}")
""")

# ----------------------------------------------------------------------------- step 3
md("""
## Chain prep

Per day D the universe is the open interest record dated D. Quotes of D are joined on (expiration, strike, right). S is the SPY close on D, T = (expiration - D) / 365.25 years. Rules, in order:

- 1 <= days to expiry <= 120. Contracts expiring on D are gone at the close and drop out.
- 0.80 S <= strike <= 1.20 S. Gamma outside that band is negligible for GEX; the table after the next one shows the |GEX| share by band on one day solved without the cut.
- open interest > 0 (zero open interest is zero GEX; those rows are not solved).
- bid > 0, ask >= bid, mid > intrinsic + 1e-6: the engine's own requirements. Contracts at or below intrinsic sit in the exercise region where the model gamma is 0, so they contribute 0 GEX by construction.

The scanner's `dte >= 7`, `mid >= 0.05` and `0.4 <= S/K <= 2.5` filters are not applied. They were for the IV benchmark, and near-dated, low-priced contracts are where SPY gamma lives. The table shows the share of the day's open interest in each bucket, median and max across days.
""")

code("""
chain, shares = G.prep_all(days, stock, sofr)
ft = G.filter_table(shares)
ft.style.format({"median open interest share": "{:.1%}", "largest open interest share": "{:.1%}"})
""")

code("""
day = [d for d in days if d.weekday() == 2][-10]        # a Wednesday near the end of the window
bs = G.band_share_day(day, float(stock.loc[pd.Timestamp(day), "close"]), G.rate_for(sofr, day))
print(f"{day}: share of total absolute GEX and of open interest by strike band, all strikes solved")
bs.style.format({"absolute GEX share": "{:.2%}", "open interest share": "{:.1%}"})
""")

# ----------------------------------------------------------------------------- step 4
md("""
## Engine run

`gexlib.solve_chain` mirrors `engine_alo/run_chain.py`: `Tables(7, 7, 27)`, `make_core(m_iter=4)`, a bracketed-secant IV solve with 12 model evaluations per contract, then one `Engine.price` call for model value, delta and gamma at the solved IV. Reference build: CPU, fp64. `iv_status` 0 means solved with the model value at the solved IV within 1e-3 of the mid; other codes are excluded from GEX and their open interest share is reported. The excluded contracts are deep in-the-money calls whose mid sits below the zero-vol model value under the flat r and q. They cluster in the two weeks before SPY's quarterly ex-dividend dates, which is dividend-capture positioning that a continuous-yield model cannot fit. The vendor's gamma for those contracts is also 0, so the choice does not affect the cross-check.
""")

code("""
sol, timings = G.solve_chain(chain, target="cpu", dtype="fp64")
rep = G.solve_report(sol)
print(timings)
print({k: v for k, v in rep.items() if k != "iv_status_counts"})
print("iv_status counts:", rep["iv_status_counts"])
""")

code("""
from numba import cuda
try:
    if cuda.is_available():
        if len(cuda.gpus) > 1:
            cuda.select_device(1)          # on this box index 1 is the NVIDIA RTX PRO 6000 Blackwell (96GB)
        sol32, t32 = G.solve_chain(chain, target="cuda", dtype="fp32")
        d_net = (sol32[sol32.iv_status == 0].groupby("date")["gex"].sum() - sol[sol.iv_status == 0].groupby("date")["gex"].sum()).abs()
        print(t32)
        print(f"GPU fp32 versus CPU fp64 daily net GEX: largest absolute difference {d_net.max():,.0f} dollars, "
              f"largest relative difference {(d_net / sol[sol.iv_status == 0].groupby('date')['gex'].sum().abs()).max():.2e}; timings for scale only")
    else:
        print("no CUDA device: GPU timing skipped")
except Exception as e:
    print(f"GPU pass skipped: {type(e).__name__}: {str(e)[:120]}")
print("all GEX numbers below come from the CPU fp64 run")
""")

md("""
### Daily series

`flip_level`: the chain is re-evaluated on a spot grid from 0.93 S to 1.07 S in 0.25% steps with each contract's IV held fixed, signed GEX is summed per grid point, and the zero crossing nearest to S is taken. NaN when the profile does not change sign on the grid. `wall_pos_strike` and `wall_neg_strike`: the strike with the largest positive and most negative net GEX summed over expirations. `regime` is the sign of `gex_net_usd`.
""")

code("""
import time; t0 = time.perf_counter()
flips = G.flip_levels(sol, verbose=False)
print(f"flip grid: {len(flips)} days x 57 spot points, {time.perf_counter() - t0:.0f}s on the CPU; crossings per day {flips.n_crossings.value_counts().to_dict()}")
daily = G.daily_gex(sol, stock, flips)
daily.to_csv(G.RESULTS / "spy_gex_daily.csv", float_format="%.8g")
print(f"{len(daily)} days; negative net GEX on {(daily.regime < 0).mean():.1%} of days; "
      f"flip level on the grid on {daily.flip_level.notna().mean():.1%} of days; "
      f"median net GEX {daily.gex_net_usd.median() / 1e9:+.2f} billion dollars per 1%; "
      f"median share of |GEX| within 30 days to expiry {daily.gex_le30d_share.median():.1%}")
daily.head(8)
""")

code("""
cols = ["S", "ret_cc", "range_pct", "gex_net_usd", "gex_call_usd", "gex_put_usd", "flip_level", "wall_pos_strike", "wall_neg_strike"]
fmt = {"gex_net_usd": "{:,.3e}", "gex_call_usd": "{:,.3e}", "gex_put_usd": "{:,.3e}", "ret_cc": "{:+.2%}", "range_pct": "{:.2%}", "flip_level": "{:.1f}"}
print("five most negative days"); display(daily.nsmallest(5, "gex_net_usd")[cols].style.format(fmt))
print("five most positive days"); display(daily.nlargest(5, "gex_net_usd")[cols].style.format(fmt))
""")

md("""
## F1: the whole window
""")

code("""
_ = G.fig_overview(daily, G.FIGURES / "f1_overview.png")
""")

# ----------------------------------------------------------------------------- step 5
md("""
## Episodes, selected by rule

The regime on day D is the sign of GEX at the close of D and is in force for D+1. An episode is a run of at least 3 consecutive same-sign days. Negative episodes: the 4 most recent runs. Positive episodes: the 4 most recent runs of at least 10 days whose median GEX is in the top tercile of the whole history. For each: the SPY move while the regime was in force (close of the first day to the close of the day after the last), mean daily range and annualized realized vol inside the episode against the 20 days before it.
""")

code("""
eps, info = G.select_episodes(daily, stock)
G.save_json({"rule": {"negative": "4 most recent runs of >= 3 days", "positive": "4 most recent runs of >= 10 days with median GEX in the top tercile"},
             "info": info, "episodes": eps.to_dict("records")}, G.RESULTS / "episodes.json")
print(info)
efmt = {"median_gex": "{:,.3e}", "spy_return_in_force": "{:+.2%}", "range_inside": "{:.2%}", "range_before": "{:.2%}", "rvol_inside": "{:.1%}", "rvol_before": "{:.1%}"}
eps.style.format(efmt)
""")

md("""
## F2: one chart per episode

Candles shaded by the regime in force, the flip level dotted, the walls on the last episode day as ticks at the right edge, the daily range below. The dashed box marks the days on which the sign was observed; the shading is shifted by one day because the regime acts on the next session.
""")

code("""
for _, r in eps.iterrows():
    _ = G.fig_episode(daily, stock, r["start"], r["end"], r["kind"], G.FIGURES / f"f2_ep_{r['kind']}_{pd.Timestamp(r['start']):%Y-%m-%d}.png")
""")

# ----------------------------------------------------------------------------- step 6
md("""
## Next-session tests conditioned on the sign

All outcomes are measured on D+1 (or D+1 to D+5) conditioned on the regime known at the close of D. For each test: n per regime, mean and median per regime, the difference (positive minus negative), a bootstrap 95% confidence interval of the difference (10,000 resamples, seed 0) and a two-sided Mann-Whitney U p-value. The autocorrelation row reports the lag-1 correlation of daily returns within each regime, a bootstrap confidence interval of the difference in correlations, and the Mann-Whitney p-value of the per-day products ret(D) x ret(D+1). "Supported" means the difference has the expected sign and the confidence interval excludes zero.
""")

code("""
out = G.next_day_outcomes(daily, stock)
tests = G.regime_tests(out, out["regime"])
tests["verdict"] = G.verdicts(tests)
share_min = min((out.regime < 0).mean(), (out.regime > 0).mean())
print(f"sign cut: {int((out.regime > 0).sum())} positive days, {int((out.regime < 0).sum())} negative days")
tfmt = {"mean_pos": "{:.4f}", "mean_neg": "{:.4f}", "median_pos": "{:.4f}", "median_neg": "{:.4f}", "diff": "{:+.4f}", "ci_lo": "{:+.4f}", "ci_hi": "{:+.4f}", "p_mwu": "{:.3f}"}
tests.drop(columns=["outcome"]).style.format(tfmt)
""")

code("""
tests_t = None
if share_min < 0.05:
    terc = G.tercile_cut(out)
    tests_t = G.regime_tests(out, terc)
    tests_t["verdict"] = G.verdicts(tests_t)
    print(f"one sign holds fewer than 5% of days, so a second cut on terciles of GEX is added: "
          f"top third ({int((terc == 1).sum())} days) against bottom third ({int((terc == -1).sum())} days)")
    display(tests_t.drop(columns=["outcome"]).style.format(tfmt))
else:
    print(f"both signs hold at least 5% of days (smaller share {share_min:.1%}); the tercile cut is not needed")
""")

md("""
### Cross-correlation of GEX with the range at leads and lags

Negative GEX is partly a consequence of a selloff: puts get bid and spot falls toward the put wall. The leading claim needs the k > 0 side of the cross-correlation between GEX on D and the daily range on D+k to hold up, and the k < 0 side is shown next to it.
""")

code("""
xc = G.cross_correlation(daily, stock)
lead = xc.loc[1:5, "pearson"].mean(); react = xc.loc[-5:-1, "pearson"].mean()
print(f"correlation of GEX on D with the range on D+k: k = 0 {xc.loc[0, 'pearson']:+.3f}; mean over k = 1 to 5 {lead:+.3f}; mean over k = -5 to -1 {react:+.3f}")
print("negative values mean lower GEX goes with a wider range")
_ = G.fig_lead_test(out, xc, out["regime"], "sign cut", G.FIGURES / "f3_lead_test.png")
if tests_t is not None:
    _ = G.fig_lead_test(out, xc, terc, "tercile cut", G.FIGURES / "f3b_lead_test_terciles.png")
xc.T.style.format("{:+.3f}", subset=pd.IndexSlice[["pearson", "spearman"], :])
""")

md(lead_md())

md("""
### Pinning

SPY lists expirations every weekday in this window, so every day is an expiration day. Distance from the close of D to the positive wall known at the close of D-1, in percent of S, by the regime in force on D. Baseline: distance to the nearest 5-dollar strike. Third Fridays, where the monthly open interest sits, are shown as a subgroup. The result is reported either way.
""")

code("""
pin = G.pinning_test(daily)
pin.style.format("{:.3f}")
""")

# ----------------------------------------------------------------------------- step 7
md("""
## Intraday reversal

30-minute log returns from the SPY 1-minute bars (last 1-minute close per 30-minute bucket, 12 returns per full session). Per day, the lag-1 autocorrelation of those returns; then the mean by the regime in force with a bootstrap 95% confidence interval. Expectation: negative under positive GEX, less negative or positive under negative GEX.
""")

code("""
per_day, itab, ip = G.intraday_autocorr(daily)
stats = {"cut": "sign of gex_net_usd at the close of D, in force for D+1", "n_days": int(len(out)),
         "share_negative_days": float((out.regime < 0).mean()), "tests_sign_cut": tests.reset_index().to_dict("records"),
         "tests_tercile_cut": None if tests_t is None else tests_t.reset_index().to_dict("records"),
         "cross_correlation_gex_vs_range": xc.reset_index().to_dict("records"),
         "pinning": pin.reset_index().to_dict("records")}
if itab is None:
    print("no 1-minute cache: step 7 skipped")
else:
    stats["intraday_ac1_by_regime"] = itab.reset_index().to_dict("records"); stats["intraday_mwu_p"] = ip
    print(f"{len(per_day)} days with 30-minute returns; Mann-Whitney p (+ against -): {ip:.3f}")
    display(itab.style.format({"mean": "{:+.3f}", "median": "{:+.3f}", "ci_lo": "{:+.3f}", "ci_hi": "{:+.3f}"}))
    _ = G.fig_intraday(itab, G.FIGURES / "f5_intraday_autocorr.png")
G.save_json(stats, G.RESULTS / "regime_stats.json")
""")

# ----------------------------------------------------------------------------- step 9
md("""
## Vendor gamma cross-check

Theta Data provides Greeks with the EOD report. The series recomputes them for a consistent American model across the whole chain and for refresh rate. Here the vendor gamma replaces the engine gamma with the same rows, open interest and formula. Table: daily correlation of vendor and engine net GEX, sign agreement, median and 99th percentile of the relative difference, the absolute difference in dollars, and the same restricted to contracts within 7 days of expiry. The relative difference of a net figure is large on days when the net is near zero, so the absolute difference is the number to read. The ratio of vendor to engine GEX is given for calls and puts separately, and the second table shows the median relative gamma difference by expiry and moneyness bucket.
""")

code("""
vd, vstats, vtab = G.vendor_compare(sol)
G.save_json(vstats, G.RESULTS / "vendor_vs_engine.json")
print(json.dumps(vstats, indent=1))
_ = G.fig_vendor(vd, vstats, G.FIGURES / "f4_vendor_vs_engine.png")
vtab.style.format("{:+.3f}")
""")


md(conclusions_md())

# ----------------------------------------------------------------------------- write
nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                  "language_info": {"name": "python"}}
for c in cells:
    for bad in ("—", "–"):
        assert bad not in c["source"], f"dash in cell: {c['source'][:80]}"
nbf.write(nb, OUT)
print(f"wrote {OUT} with {len(cells)} cells ({sum(c['cell_type'] == 'code' for c in cells)} code)")
