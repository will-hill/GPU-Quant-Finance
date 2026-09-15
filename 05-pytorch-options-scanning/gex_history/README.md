# gex_history: SPY net gamma exposure, 2023 to 2026

Daily history of SPY net gamma exposure (GEX) from Theta Data end-of-day option quotes and open
interest, gamma from the `engine_alo` American option engine, and rule-based tests of whether the
sign of GEX at the close leads the next session. The notebook is `../gex_history_spy.ipynb`
(executed, outputs committed). Educational analysis, not a trading strategy.

## Layout

```
../gex_history_spy.ipynb        the notebook (executed, outputs committed)
probe.py                        step 1 probe of the Theta endpoints: schemas, OI date rule, depth, timing
download.py                     pull + cache, resumable, standalone (the only network step)
gexlib.py                       chain prep, engine call, GEX aggregation, flip level, regimes, tests, figures
build_notebook.py               writes the notebook from short cells; run, execute, run again, execute again
cache/                          one parquet per endpoint per day (gitignored)
figures/                        PNG, 1920x1080 (gitignored; the notebook outputs carry them)
results/
  download_log.json             what was pulled: window, days, holidays, rows per day, cache size
  spy_gex_daily.csv             one row per trading day
  episodes.json                 the rule-selected episodes and their stats
  regime_stats.json             next-session tests, cross-correlation, pinning, intraday
  vendor_vs_engine.json         vendor gamma against engine gamma
```

## How to rerun

```
cd 05-pytorch-options-scanning
uv sync                                                    # numba, numba-cuda[cu13], quantlib, scipy, thetadata, polars, jupyter
uv run python gex_history/download.py --intraday           # ~15 min with 8 workers; resumable; needs THETADATA_API_KEY in .env
uv run python gex_history/build_notebook.py
uv run jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=3600 gex_history_spy.ipynb
uv run python gex_history/build_notebook.py                # second pass fills the conclusions cell from results/
uv run jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=3600 gex_history_spy.ipynb
```

Every notebook cell runs from `cache/`; copy the cache to another machine and skip the download.
Theta Data allows one live session per account: do not run `probe.py` or another `ThetaClient`
while `download.py` runs (the server answers UNAUTHENTICATED "Invalid session ID"; the script
re-authenticates and retries, but the other process loses its session).

## Data and rules

**Window.** 2023-09-15 to 2026-09-14, the last completed session when the pull ran (2026-09-15,
15:25 ET). 782 weekdays, 751 trading days with quotes, open interest and vendor Greeks, 31 market
holidays with no data on any endpoint (listed in `results/download_log.json`), 0 partial days.
History depth was probed at 1, 2 and 3 years back (`probe.py`); all three endpoints returned data at
2023-09-13, so the window is the 3 years the prompt allows.

**Endpoints.** One request per trading day with `expiration="*"` and `max_dte=120`:
`option_history_eod` (quotes at the 17:15 ET report), `option_history_open_interest`,
`option_history_greeks_eod`. `stock_history_eod` for SPY bars (from 2023-06-01, so the 20 days
before the first episode exist), `interest_rate_history_eod("SOFR")`, and with `--intraday`
`stock_history_ohlc(interval="1m")` per calendar month.

**Open interest rule.** An OI row dated D carries a 06:30 ET timestamp on D and is the open interest
as of the close of D-1. Established in `probe.py` on the 2026-09-11 expiration: the correlation of
|dOI on D| with volume on D-1 is 0.667 against 0.162 with volume on D, and |dOI| exceeds volume on
1.1% of contract-days under the D-1 pairing against 15.9% under the same-day pairing. The GEX at the
close of D uses the OI record dated D, the last one available during D. Positions opened during D,
including all 0DTE flow, are not in it. No record published after the close of D is used.

**Chain rules per day.** Universe = the OI record of D, left-joined to the quotes of D on
(expiration, strike, right). S = SPY close on D. T = (expiration - D) / 365.25.
Kept: 1 <= days to expiry <= 120, 0.80 S <= K <= 1.20 S, open interest > 0, bid > 0, ask >= bid,
mid > intrinsic + 1e-6. Contracts expiring on D are gone at the close. Strikes outside the band
hold 29.5% of OI but 0.84% of |GEX| on the day checked without the cut (2026-07-08). The scanner's
`dte >= 7`, `mid >= 0.05` and `0.4 <= S/K <= 2.5` filters are not applied.

**Rate and yield.** r = SOFR on D from Theta Data, in decimal, used as a continuously compounded
rate. This account's rate history starts 2024-01-01; the 2023 days use the first record, 5.38%.
q = 0.012 flat (SPY trailing yield, an assumption).

**Daily bars.** Open and close from the EOD report. High and low are the narrower of the EOD report
and the traded 1-minute session range, with the open and close folded in. The two sources agree
exactly on most days; the EOD bar of 2026-02-02 has a low of 69.005 (a decimal-shift print) and the
rule removes it. `load_stock(verbose=True)` lists the days where the range narrowed by more than
0.1% of the close.

**Engine.** `gexlib.solve_chain` mirrors `engine_alo/run_chain.py`: Tables(7, 7, 27), make_core with
4 fixed-point sweeps, bracketed secant IV with 12 model evaluations, one Engine.price call for value,
delta and gamma at the solved IV. Reference build CPU fp64. Contracts with iv_status != 0 (all deep
in-the-money calls quoted below the zero-vol model value, clustered before SPY's quarterly
ex-dividend dates) are excluded; the vendor's gamma for them is 0 as well.

**GEX.** Per contract gamma x open interest x 100 x S x S x 0.01, calls +, puts -, dollars per 1%
move of SPY. Dealer positioning is inferred, not observed. Flip level: signed GEX summed on a spot
grid 0.93 S to 1.07 S in 0.25% steps with each contract's IV fixed, zero crossing nearest to S.
Walls: strikes with the largest positive and most negative net GEX summed over expirations.

**Regimes and episodes.** Regime on D = sign of net GEX at the close of D, in force for D+1.
Negative episodes: the 4 most recent runs of at least 3 same-sign days. Positive episodes: the 4 most
recent runs of at least 10 days whose median GEX is in the top tercile of the whole history.

**Hardware.** Timings in the notebook: AMD Threadripper PRO 7965WX (48 numba threads) for the
reference run; NVIDIA RTX PRO 6000 Blackwell (96GB) for the fp32 comparison, selected with
`cuda.select_device(1)` because numba enumerates the RTX 6000 Ada in this box first. All GEX
numbers come from the CPU fp64 run. Download time is not part of any number.

## Provenance of the numbers in the conclusions

Every number in the notebook's conclusions cell, with the cell that produced it (cells are named by
their first line) and the results file that stores it. All numbers are from the committed execution
of `gex_history_spy.ipynb` on this machine; nothing is typed in by hand.

| number | value | cell | file |
|---|---|---|---|
| trading days, share with negative net GEX | 751, 57.9% (435 negative, 316 positive) | `flips = G.flip_levels(...)` / `daily = G.daily_gex(...)` | `results/spy_gex_daily.csv` |
| contracts solved, iv_status 1 (excluded), OI share excluded | 2,392,386; 39,881; 0.47% overall, 0.014% median day, 11.07% worst day (2024-06-18) | `sol, timings = G.solve_chain(chain, target="cpu", dtype="fp64")` | printed in the notebook |
| repricing residual, status 0 | median 6.1e-12, p99 3.7e-8, max 4.6e-4 | same cell | printed |
| engine wall time, CPU fp64, 48 threads | IV pass 10.5 s, Greeks pass 1.3 s | same cell | printed |
| engine wall time, RTX PRO 6000 Blackwell fp32 | IV pass 0.21 s, Greeks pass 0.05 s; daily net GEX differs from CPU fp64 by at most 2.8e-3 relative | `from numba import cuda` cell | printed |
| flip level found | 747 of 751 days; median 0.19% above spot | flip cell | `spy_gex_daily.csv` |
| next-day range, + against - | 0.75% against 1.22%, diff -0.47%, CI [-0.57%, -0.39%], p < 0.001 | `out = G.next_day_outcomes(daily, stock)` | `results/regime_stats.json` tests_sign_cut[range] |
| next-day absolute return | 0.46% against 0.80%, diff -0.34%, CI [-0.43%, -0.25%], p < 0.001 | same | tests_sign_cut[move] |
| 5-day forward vol, annualized | 9.8% against 14.7%, diff -4.9%, CI [-6.1%, -3.8%], p < 0.001 | same | tests_sign_cut[forward vol] |
| next-day continuation rate | 54.7% against 49.5%, diff +5.2 points, CI [-2.1, +12.4], p = 0.159 | same | tests_sign_cut[continuation] |
| lag-1 autocorrelation of daily returns | +0.064 against -0.088, CI of the difference [-0.10, +0.37], p = 0.051 | same | tests_sign_cut[autocorrelation] |
| gap-fade rate | 54.1% against 49.8%, diff +4.3 points, CI [-2.7, +11.8], p = 0.240 | same | tests_sign_cut[gap fade] |
| corr(GEX on D, range on D+k) | k = +1: -0.39 (Spearman -0.52); k = -1: -0.31 (Spearman -0.37); k = 0: -0.40; mean k = 1..5 -0.32, mean k = -5..-1 -0.24 | `xc = G.cross_correlation(daily, stock)` | cross_correlation_gex_vs_range |
| pinning: distance to the positive wall, share of days moving toward it | 0.79% against 2.89%; 44.6% against 54.6% of days | `pin = G.pinning_test(daily)` | pinning |
| intraday 30-minute lag-1 autocorrelation | -0.093 (CI [-0.125, -0.060], n = 310) against -0.066 (CI [-0.094, -0.037], n = 433), p = 0.275 | `per_day, itab, ip = G.intraday_autocorr(daily)` | intraday_ac1_by_regime |
| episodes | 4 negative runs (2026-07-16, 08-18, 08-28, 09-04), 4 positive runs (2024-06-05, 2024-11-19, 2025-11-25, 2026-08-03) | `eps, info = G.select_episodes(daily, stock)` | `results/episodes.json` |
| vendor against engine net GEX | corr 0.9992, sign agreement 94.5%, abs difference median $0.72bn, p99 $1.51bn; within 7 days to expiry corr 0.9998, sign agreement 98.3%, abs difference median $0.08bn; vendor put GEX is 0.957 of the engine's on the median day, calls 1.006 | `vd, vstats, vtab = G.vendor_compare(sol)` | `results/vendor_vs_engine.json` |
| strike band check | 94.97% of abs GEX inside 0.90 to 1.10 S, 0.84% outside 0.80 to 1.20 S, on 2026-07-08 | `day = [d for d in days if d.weekday() == 2][-10]` cell | printed |

Step 1 numbers (schemas, the OI rule test, history depth, pull timing) come from `uv run python
gex_history/probe.py`; the pull summary is `results/download_log.json`.
