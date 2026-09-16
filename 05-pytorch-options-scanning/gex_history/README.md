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
intraday.py                     the live book (the previous close's book evaluated at the current spot, from daily spot-grid profiles) and the intraday tests T1..T7
build_intraday_notebook.py      writes ../gex_intraday_spy.ipynb (same two-pass scheme)
live_gpu.py                   device-resident live value kernel around the engine's device function; timing table
download_universe.py            multi-symbol pull (quotes, OI, spot, 1-minute bars) for the cross-section
cross_section.py                per-name daily GEX + profiles, pooled tests, cross-sectional sort, pooled intraday tests
build_cross_section_notebook.py writes ../gex_cross_section.ipynb
per_symbol_tests.py            every indication as one statistic per symbol -> results/cross_section/per_symbol_tests.json
rows.py, build_row_notebooks.py one notebook per indication into ../gex_rows/
cache/                          one parquet per endpoint per day (gitignored)
figures/                        PNG, 1920x1080 (gitignored; the notebook outputs carry them)
results/
  download_log.json             what was pulled: window, days, holidays, rows per day, cache size
  spy_gex_daily.csv             one row per trading day
  episodes.json                 the rule-selected episodes and their stats
  regime_stats.json             next-session tests, cross-correlation, pinning, intraday
  vendor_vs_engine.json         vendor gamma against engine gamma
  intraday_spy.json             SPY intraday tests (live value against stale, flips, walls, time of day, model choice)
  live_timing.json            GPU kernel against CPU timing of the spot-grid live value
  download_universe_log.json    the multi-symbol pull
  cross_section/                per-name daily tables ({sym}_daily.csv), symbols.json, summary.json
```

Notebooks at the module top level: `gex_history_spy.ipynb` (daily regime tests, 3 years), `gex_intraday_spy.ipynb`
(what the live book adds intraday), `gex_cross_section.ipynb` (the most active names, 1 year).

`../gex_rows/row01..row10_*.ipynb`: one notebook per demonstrated indication (title line plus code), built by
`build_row_notebooks.py` on `rows.py`; each computes SPY live and shows the per-name table from
`results/cross_section/per_symbol_tests.json` (`per_symbol_tests.py`, SPY QQQ IWM NVDA TSLA AAPL AMZN META MSFT AMD).
Rows: 1 live value tracks the next print; 2 level at 10:00 sets the rest of day; 3 live value beats the stale print for the
next 30 minutes; 4 flip crossings; 5 the 0DTE layer; 6 persistence; 7 intraday realized vol by daily regime;
8 next-day range; 9 next-week vol; 10 tail days.

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

## Intraday: what the live book adds (`gex_intraday_spy.ipynb`)

Open interest arrives once a day, so the quantity a continuous scanner produces is the previous
close's book evaluated at the current spot, vol and time: the live book. `intraday.daily_profiles` evaluates each
day's solved chain on a spot grid from 0.90 S to 1.10 S in 0.25% steps (IV fixed); the live value at any
intraday spot is a lookup on that profile. Tests on the traded 1-minute SPY bars, 30-minute buckets,
range normalized by the time-of-day median. Numbers in `results/intraday_spy.json`.

| test | result |
|---|---|
| T6 live value against the next official print | sign agreement 92.9% for the live book at today's close against 79.5% for yesterday's print; level correlation 0.96 against 0.77; 77% of the 154 sign changes caught by the close |
| T1 realized vol by regime | negative over positive 1.58x (5-minute returns) to 1.64x (30-minute), p < 0.001 at every horizon |
| T2 live value against stale, next 30-minute range | Spearman -0.52 against -0.43; joint rank regression t = -10.6 for the live value, -1.5 for the stale value; within-day increment Spearman -0.03 (the live value updates the day's level, it does not time buckets) |
| T3 intraday flips | live sign leaves the prior close's sign on 29% of days; after a negative-to-positive crossing the rest of the day runs 0.60 to 0.64 multiples below no-crossing days of the same regime (matched on move size), after positive-to-negative 0.31 above |
| T3b near-flip control | days that started within 0.5% of the flip: crossed up 0.97 against stayed 1.24 (CI [-0.40, -0.13]); crossed down 1.12 against 0.81 (CI [+0.21, +0.40]) |
| T4 wall touches | 30 minutes after the first touch: call wall -1.2 bp (CI [-3.9, +1.5], n 122), put wall -2.7 bp (CI [-8.6, +3.2], n 86); placebo levels 5 dollars away look the same. Not support or resistance |
| T5 time of day | negative over positive range ratio between 1.38 and 1.74 in every half hour; widest bucket is the open |
| T7 model choice | European instead of American changes the daily sign on 15 of 751 days, median 0.26 $bn; American gamma is 1.067x European for in-the-money puts, equal for calls |

## GPU: the live value as one kernel (`live_gpu.py`, `results/live_timing.json`)

One thread per contract, 81 spot points, the engine's CUDA device function, per-day sums reduced on
the device in fp64. Full 3-year SPY history, 2,352,505 contracts x 81 points = 190.6M evaluations:
kernel 0.87 s, 0.95 s end to end on the NVIDIA RTX PRO 6000 Blackwell (96GB) in fp32, against 107 s for
the same evaluations through the engine's CPU batch driver in fp64 on the AMD Threadripper PRO 7965WX
(48 threads). One day (4,086 contracts): 36 ms kernel against 1.8 s CPU. At the measured rate a full
live evaluation of 200 names with 1,500 to 3,000 contracts each is 0.11 to 0.22 s on the GPU and 14 to 27 s on
the CPU path. The fp32 profile differs from the CPU fp64 profile by at most a few 1e-4 of its level.

## Cross-section: the most active names (`gex_cross_section.ipynb`)

`download_universe.py` pulled one year (2025-09-15 to 2026-09-14) of quotes and open interest for the
40 most active names in the scanner's universe plus the SPX book (SPX and SPXW roots on the SPX spot,
European exercise), their daily bars and 1-minute bars. `cross_section.run_symbol` runs the SPY pipeline
per name (flat dividend yield per name from `gexlib.Q_BY_SYMBOL`, 0 when not listed). Structure of the
books, GEX in dollars per 1% move, from `results/cross_section/{sym}_daily.csv`:

| symbol | days | negative-GEX days | median net GEX, $bn per 1% | median abs GEX | contracts per day |
|---|---|---|---|---|---|
| SPX | 251.0 | 38% | +14.55 | 32.18 | 10711.0 |
| SPY | 251.0 | 63% | -2.69 | 4.59 | 3595.0 |
| QQQ | 251.0 | 59% | -0.86 | 2.37 | 2988.0 |
| GLD | 251.0 | 20% | +1.38 | 1.38 | 1850.0 |
| IWM | 251.0 | 82% | -0.97 | 1.02 | 1487.0 |
| AAPL | 251.0 | 2% | +0.97 | 0.97 | 500.0 |
| NVDA | 251.0 | 5% | +0.80 | 0.80 | 490.0 |
| MSFT | 251.0 | 19% | +0.43 | 0.43 | 753.0 |
| AMZN | 251.0 | 4% | +0.41 | 0.41 | 461.0 |
| GOOGL | 251.0 | 11% | +0.30 | 0.30 | 602.0 |
| META | 251.0 | 29% | +0.22 | 0.28 | 1114.0 |
| TSLA | 251.0 | 20% | +0.25 | 0.27 | 838.0 |
| GOOG | 251.0 | 21% | +0.15 | 0.15 | 477.0 |
| AVGO | 251.0 | 30% | +0.08 | 0.11 | 642.0 |
| AMD | 251.0 | 23% | +0.09 | 0.11 | 384.0 |
| SMH | 251.0 | 66% | -0.07 | 0.10 | 664.0 |
| MU | 251.0 | 19% | +0.08 | 0.10 | 604.0 |
| SLV | 251.0 | 16% | +0.09 | 0.09 | 966.0 |
| PLTR | 251.0 | 24% | +0.08 | 0.09 | 374.0 |
| TSM | 251.0 | 33% | +0.05 | 0.08 | 454.0 |
| MSTR | 251.0 | 15% | +0.05 | 0.06 | 437.0 |
| NFLX | 251.0 | 51% | -0.00 | 0.05 | 495.0 |
| ORCL | 251.0 | 51% | -0.00 | 0.05 | 366.0 |
| GS | 251.0 | 32% | +0.03 | 0.05 | 952.0 |
| INTC | 251.0 | 11% | +0.04 | 0.05 | 363.0 |
| SPCX | 81.0 | 63% | -0.00 | 0.04 | 422.0 |
| ASML | 251.0 | 31% | +0.03 | 0.04 | 883.0 |
| IBM | 251.0 | 22% | +0.02 | 0.03 | 340.0 |
| EWY | 251.0 | 34% | +0.01 | 0.02 | 455.0 |
| SNDK | 251.0 | 33% | +0.01 | 0.02 | 791.0 |
| MRVL | 251.0 | 24% | +0.02 | 0.02 | 350.0 |
| DELL | 251.0 | 24% | +0.02 | 0.02 | 403.0 |
| CRWV | 251.0 | 41% | +0.01 | 0.02 | 427.0 |
| ARM | 251.0 | 45% | +0.00 | 0.01 | 376.0 |
| DRAM | 110.0 | 25% | +0.01 | 0.01 | 442.0 |
| WDC | 251.0 | 26% | +0.01 | 0.01 | 419.0 |
| BE | 251.0 | 22% | +0.01 | 0.01 | 400.0 |
| SKHY | 43.0 | 51% | -0.00 | 0.01 | 398.0 |
| NBIS | 251.0 | 38% | +0.00 | 0.01 | 391.0 |
| STX | 251.0 | 23% | +0.01 | 0.01 | 454.0 |
| SOXL | 251.0 | 29% | +0.01 | 0.01 | 416.0 |
| ASTS | 251.0 | 43% | +0.00 | 0.00 | 325.0 |

Index and ETF books are put-dominated; the mega-caps are call-dominated and almost never negative, so
the within-name tests use each name's own GEX terciles rather than the sign. Test results and the
pooled intraday live-book tests are in `results/cross_section/summary.json` and the notebook's
conclusions cell. Results. Within a name, low GEX goes with a wider next day in 27 of 39 names (median ratio 1.06; SPX 1.50, QQQ 1.39, DRAM 1.37); pooled bottom third 1.046 against top third 0.978, CI [+0.044, +0.090]. That is a market-wide time effect, not a way to pick names: on the same day, names in their own bottom tercile against names in their own top tercile differ by -0.002 (CI [-0.036, +0.032], 207 days), the share of names in their low state on D correlates +0.20 with the market's average relative range on D+1, and ranking names across the cross-section does not order tomorrow's relative range under any key (most negative to most positive bin: GEX per dollar traded 1.019, 0.995, 0.986, 1.019, 1.023; own z-score 0.992, 1.005, 1.010, 0.998, 1.012; raw GEX 1.025, 0.996, 1.008, 1.013, 1.000). Intraday, over 41 names and 115,644 half-hour buckets: mean sign agreement with the next official print 92.2% for the live book against 85.3% stale, better in every name; the live sign leaves the prior close's sign on 14% of name-days. Pooled flips move the next bucket in the predicted direction (negative to positive 1.11 against 1.26, positive to negative 1.23 against 1.10), but at the same date and bucket flipped names are not wider than names that held (-0.009, CI [-0.038, +0.018]; -0.039, CI [-0.056, -0.021]). A multi-name scanner is therefore shown to keep each name's regime current between prints; it is not shown to tell which name will be wider or quieter than its peers.

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

Intraday numbers come from the cells of `gex_intraday_spy.ipynb` named by their first line (`t6 = I.t6_live_tracks_next_print(...)`,
`t1 = I.t1_intraday_vol(...)`, `t2 = I.t2_live_versus_stale(ob)`, `t3, flipdays = I.t3_intraday_flips(ob)`, `t4 = I.t4_wall_touches(...)`,
`t5 = I.t5_time_of_day(ob)`, `t7 = I.t7_model_choice(...)`, `layer = I.zero_dte_layer(...)`) and are stored in `results/intraday_spy.json`.
GPU timings come from `uv run python gex_history/live_gpu.py` (`results/live_timing.json`). Cross-section numbers come from the
cells of `gex_cross_section.ipynb` (`runs = [X.run_symbol(...)]`, `bytercile = X.next_day_by_tercile(names)`, `pooled = ...`,
`xs = X.cross_sectional_sort(...)`, `pooled_intra = X.pooled_intraday(intra)`, `same = X.same_day_tests(panel, ob_all)`) and are stored in
`results/cross_section/summary.json`.

Step 1 numbers (schemas, the OI rule test, history depth, pull timing) come from `uv run python
gex_history/probe.py`; the pull summary is `results/download_log.json`.
