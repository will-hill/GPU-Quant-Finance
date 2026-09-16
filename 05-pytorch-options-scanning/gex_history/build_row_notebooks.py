"""Build one notebook per demonstrated GEX indication into ../gex_rows/. Markdown kept to a title line.

    uv run python gex_history/build_row_notebooks.py
    for nb in gex_rows/*.ipynb; do uv run jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=3600 "$nb"; done
"""
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "gex_rows"
OUT.mkdir(exist_ok=True)

SETUP = '''import sys
import json
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, "../gex_history")
import rows as R
import gexlib as G
pd.set_option("display.width", 220); pd.set_option("display.max_columns", 40)
ctx = R.load("spy"{chain})
print(ctx["sym"], len(ctx["daily"]), "days")'''

FIELDS = ["frequency (GEX period)", "strength", "tradability (1 to 10)", "holds for", "how to trade", "numbers (SPY)", "example", "video opener"]

OPENERS = {1: "The market makers who sell you options hedge them by buying and selling the stock, and the sum of that hedging, called gamma exposure or GEX, decides whether the market gets pushed around or held still. Most people look it up once a day from the morning open interest report. Take last night's positions and recompute the hedging at the price the stock trades right now and you already know tonight's number nine times out of ten, so you stop trading on yesterday's regime.", 2: 'By 10:00 you can measure how much option hedging is pinning SPY in place today. When that live gamma reading is in its top third, the rest of the day runs about half the range of a bottom-third day. Set your stops and decide whether you are a seller or a buyer of options before the day is a quarter over.', 3: 'Gamma exposure is not a once-a-day number; it changes every time SPY moves, because how much hedging dealers must do depends on where price sits against the strikes people own. Recomputing it every half hour tells you more about the next half hour than the morning print does, and the gap widens all afternoon. A live number is worth more the later in the day it gets.', 4: "There is a price level, the gamma flip, where dealer hedging switches from calming the market to chasing it. When SPY crosses that level during the day, the rest of the session gets quieter after an upward cross and wilder after a downward one. Watch the live flip level and adjust your risk when price goes through it, instead of waiting for tomorrow's report.", 5: 'More than half of the gamma alive in SPY on any day sits in options that expire that same afternoon, and the daily report never shows them because they are gone by the close. When that same-day gamma is large going into 15:30, the last half hour tends to be quiet. Plan how you trade the close around a number only a live calculation can give you.', 6: "Once the market is in a calm or a wild hedging regime it tends to stay there: about four days in five, tomorrow's reading has the same sign as today's. A regime is a condition you can plan a few days around, not a one-day signal, as long as you check each morning that it has not flipped.", 7: "Yesterday's closing gamma reading does not only predict tomorrow's daily range; it shows up at every time scale during the day. Five-minute, fifteen-minute and hourly moves are all about 60 percent larger when dealers are short gamma. A day trader can set the whole day's expectations from one number before the open.", 8: "When option dealers are long gamma they sell rallies and buy dips to stay hedged, and the market moves less; when they are short gamma they do the opposite and moves get amplified. You can read which state they are in from the day's option data, and it predicts tomorrow's trading range well enough to decide tonight whether to sell options or buy them.", 9: 'The same hedging reading works for the week ahead, not only tomorrow: after a positive gamma close, realized volatility over the next five days runs about a third lower than after a negative one. If you trade weekly options, that is the difference between selling premium into a calm week and buying protection before a rough one.', 10: 'The biggest one-day moves in SPY, the ones that blow up positions, almost all arrive after days when dealer gamma was negative: 85 percent of the twenty largest moves in three years. You cannot predict which day, but you can know when you are in the zone. That is when to cut size or own some tail protection.'}

ROWS = [
    (1, "live_tracking", "The live book tracks the next official GEX print",
     "res = R.row1(ctx)\nprint(json.dumps(R.show(res), indent=1))\n_ = R.fig1(ctx, res, G.FIGURES / 'row01_live_tracking.png')",
     {"r1_sign_agree_live": "sign agreement, live", "r1_sign_agree_stale": "sign agreement, stale print", "r1_corr_live": "level correlation, live", "r1_corr_stale": "level correlation, stale print"},
     ["every tick, compared with the next daily print", "0.96 correlation of levels (a tracking measure, not a market outcome)", "3",
      "all ten names. Sign agreement live versus stale print: SPY 93% versus 79%, QQQ 93% versus 82%, IWM 94% versus 87%, NVDA 99% versus 94%, TSLA 92% versus 81%, AAPL 99% versus 97%, AMZN 96% versus 94%, META 94% versus 86%, MSFT 97% versus 90%, AMD 93% versus 84%",
      "run the live book instead of waiting for the morning print; it feeds rows 2 to 5",
      "sign agreement 92.9% live versus 79.5% stale; level correlation 0.96 versus 0.77; across 41 names 92.2% versus 85.3%",
      "2026-06-05: stale value +2.3 billion dollars, live value at the close -12.5 billion, official next print -6.2 billion, SPY -2.61%"]),
    (2, "level_at_1000", "The live GEX level at 10:00 sets the rest-of-day range",
     "res = R.row2(ctx)\nprint(json.dumps({k: v for k, v in R.show(res).items() if 'horizon' not in k}, indent=1))\ndisplay(pd.DataFrame({t: {h: f'{a:+.2f} (stale {b:+.2f})' for h, (a, b) in d.items()} for t, d in res['spearman_by_time_and_horizon (live, stale)'].items()}).T)\n_ = R.fig2(ctx, res, G.FIGURES / 'row02_level_at_1000.png')",
     {"r2_spearman_1000_rest": "Spearman correlation, 10:00 level versus rest of day", "r2_rest_low_third": "rest-of-day range, low third (times the median)", "r2_rest_high_third": "rest-of-day range, high third"},
     ["live value at a 30-minute close", "0.58 Spearman correlation", "7",
      "yes SPY and QQQ (1.48 versus 0.89 times the median); weak IWM, NVDA, AAPL, AMZN, META, MSFT (correlations -0.15 to -0.25); no TSLA, AMD",
      "at 10:00: top third of live GEX, sell intraday premium and expect about 0.9 times the usual range; bottom third, buy gamma or widen stops and expect about 1.75 times",
      "terciles 1.75 versus 0.91 times the time-of-day median, p-value below 0.001; large positive 0.83 times, large negative 1.88 times",
      "bottom-third mornings run 1.96 times the median on prior-negative days versus 1.14 times for the top third"]),
    (3, "next_30_minutes", "At any half hour the live level predicts the next 30 minutes better than the stale print",
     "res = R.row3(ctx)\nprint(json.dumps({k: v for k, v in R.show(res).items() if 'bucket (' not in k}, indent=1))\n_ = R.fig3(ctx, res, G.FIGURES / 'row03_next_30_minutes.png')",
     {"r3_spearman_live": "Spearman correlation, live", "r3_spearman_stale": "Spearman correlation, stale print", "r3_share_days_flip": "share of days with an intraday sign flip"},
     ["live value at a 30-minute close", "0.52 Spearman correlation (stale print 0.43)", "6",
      "yes SPY, QQQ (-0.48 versus -0.34), IWM (-0.25 versus -0.18); weak NVDA; no edge in the other six stocks",
      "refresh the volatility expectation every half hour; scale same-day-expiry premium selling or gamma scalping to the current level",
      "joint regression t-statistics -10.6 for the live value versus -1.5 for the stale print; at 14:00 the rest-of-day correlation is -0.54 live versus -0.42 stale",
      "at 11:30 the live value's correlation with the next hour's range is -0.60, the stale print's -0.52"]),
    (4, "flip_crossing", "Spot crossing the flip level intraday changes the rest of the day",
     "res = R.row4(ctx)\nprint(json.dumps(R.show(res), indent=1))\n_ = R.fig4(ctx, res, G.FIGURES / 'row04_flip_crossing.png')",
     {"r4_up_cross": "crossed up, rest-of-day range", "r4_up_stay": "stayed, rest-of-day range", "r4_up_n": "count crossed up", "r4_up_ci_excl0": "confidence interval excludes zero (up)", "r4_down_cross": "crossed down, rest-of-day range", "r4_down_stay": "stayed", "r4_down_n": "count crossed down", "r4_down_ci_excl0": "confidence interval excludes zero (down)"},
     ["live value at a 30-minute close", "0.45 (crossing down) and 0.31 (crossing up), point-biserial correlation", "6",
      "yes SPY and QQQ (0.97 versus 1.21 crossing up, 1.13 versus 0.82 crossing down); direction only IWM; TSLA crossing down only (11 events); no META, MSFT; too few events in NVDA, AAPL, AMZN, AMD",
      "crossing up: fade volatility and sell premium; crossing down: buy protection and widen stops",
      "29% of days cross, 13% by 10:00; among days that started within 0.5% of the flip, crossing up 0.97 versus staying 1.24 times the median, crossing down 1.12 versus 0.81; both confidence intervals exclude zero",
      "2024-11-21: prior close +0.7 billion dollars, crossed down at 10:30, rest of day 1.39 times the median. 2026-07-09: prior close -3.7 billion, crossed up at 10:00, rest of day 0.74 times"]),
    (5, "zero_dte", "More same-day-expiry gamma at 15:30, quieter close; the same-day layer is larger than the standing book",
     "res = R.row5(ctx)\nprint(json.dumps({k: v for k, v in R.show(res['t8']).items() if k != 'by_bucket'}, indent=1))\nprint(json.dumps(R.show(res['controls']), indent=1))\ndisplay(pd.DataFrame(res['t8']['by_bucket']).set_index('bucket').round(3))\n_ = R.fig5(ctx, res, G.FIGURES / 'row05_zero_dte.png')",
     {"r5_days_with_0dte": "days with a same-day expiry", "r5_share_0dte_of_book_1000": "same-day-expiry share of gamma at 10:00", "r5_spearman_1530_last": "Spearman correlation, 15:30 exposure versus last half hour", "r5_last_low": "last half hour, low third (times the median)", "r5_last_high": "last half hour, high third", "r5_last_p": "p-value"},
     ["live value at a 30-minute close with the true remaining time to expiry", "0.31 Spearman correlation", "4",
      "yes SPY, QQQ (1.44 versus 1.09, p-value 0.006), IWM (1.27 versus 1.04, p-value 0.005), TSLA (p-value 0.02), MSFT (p-value 0.04); no NVDA, AAPL, AMZN, META, AMD. Stocks expire weekly, so about 112 days each",
      "on days in the top third of same-day gamma at 15:30, sell the last-half-hour straddle or expect a pinned close",
      "15:30 to 16:00 range 1.07 versus 1.59 times the median, t-statistic -2.4 after controlling for the day's range so far; the same-day layer holds 5.6 billion dollars of unsigned gamma exposure at 10:00 against 4.2 billion for the whole standing book, 57% of the total; that open interest doubles on the last day before expiry",
      "2025-07-02: 7.5 billion dollars of same-day exposure at 15:30, standing book +4.1 billion, last half hour 0.60 times the median"]),
    (6, "persistence", "The GEX regime is sticky day to day",
     "res = R.row6(ctx)\nprint(json.dumps(R.show(res), indent=1))\n_ = R.fig6(ctx, res, G.FIGURES / 'row06_persistence.png')",
     {"share_negative": "share of negative days", "r6_persistence": "probability the sign holds the next day"},
     ["daily print", "0.58 phi correlation between today's sign and tomorrow's", "3",
      "all names, 79% to 97%; the stock figures are high because their books are positive almost every day",
      "hold regime-based positions overnight; re-check with row 1", "the sign holds on 79% of days; longest runs 20 days positive, 49 days negative",
      "19 November to 17 December 2024: 20 straight positive days"]),
    (7, "intraday_realized_vol", "The daily print sets intraday realized volatility at every horizon",
     "res = R.row7(ctx)\ndisplay(res.round(4))\n_ = R.fig7(ctx, res, G.FIGURES / 'row07_intraday_realized_vol.png')",
     {"r7_rv_ratio_neg_over_pos": "realized volatility, negative over positive regime (30-minute returns)", "r7_n_neg": "negative days", "r7_p": "p-value"},
     ["daily print", "0.57 Spearman correlation with same-day realized volatility", "5",
      "yes SPY 1.64 times, QQQ 1.46 times; weak IWM 1.18, TSLA 1.17, NVDA 1.24 (13 negative days), MSFT 1.11; no AAPL, AMZN, META, AMD",
      "intraday volatility scalping and position sizing by the prior close's regime",
      "negative over positive 1.58 times at 5 minutes to 1.64 times at 30 minutes, p-value below 0.001", "5-minute realized volatility 11.8% versus 7.4% annualized"]),
    (8, "next_day_range", "A positive close means a quieter next session, a negative close a wider one",
     "res = R.row8(ctx)\ndisplay(res['sign_cut'].drop(columns=['outcome']).round(4))\nprint(json.dumps({k: v for k, v in R.show(res).items() if k not in ('sign_cut', 'cross_correlation')}, indent=1))\ndisplay(res['cross_correlation'].T.round(3))\n_ = R.fig8(ctx, res, G.FIGURES / 'row08_next_day_range.png')",
     {"r8_range_pos": "next-day range after a positive close", "r8_range_neg": "after a negative close", "r8_n_neg": "negative days", "r8_range_top_third": "own top third", "r8_range_bottom_third": "own bottom third", "r8_p_tercile": "p-value (terciles)", "r8_spearman_gex_next_range": "Spearman correlation"},
     ["daily print", "0.52 Spearman correlation", "6",
      "yes SPY, QQQ (1.13% versus 1.61%, correlation -0.39), IWM (correlation -0.20), AAPL (own tercile 1.78% versus 2.28%), MSFT (2.05% versus 2.40%); weak AMZN, META; no NVDA, TSLA, AMD. Stocks need the own-tercile cut because they are rarely negative",
      "next-day volatility filter: short gamma and tighter stops after a positive close, long gamma and wider stops after a negative one",
      "range 0.75% versus 1.22%, absolute move 0.46% versus 0.80%, p-value below 0.001",
      "2026-08-13 +19.5 billion dollars, next-day range 0.43% (median day 0.85%); 2026-03-19 -19.3 billion, next-day range 1.81%"]),
    (9, "next_week_vol", "A positive close means lower realized volatility over the next week",
     "res = R.row9(ctx)\ndisplay(res['sign_cut'].drop(columns=['outcome']).round(4))\nprint(json.dumps({k: v for k, v in R.show(res).items() if k != 'sign_cut'}, indent=1))\n_ = R.fig9(ctx, res, G.FIGURES / 'row09_next_week_vol.png')",
     {"r9_fvol_top_third": "5-day volatility, own top third", "r9_fvol_bottom_third": "own bottom third", "r9_spearman_gex_fvol": "Spearman correlation"},
     ["daily print", "0.42 Spearman correlation", "5", "yes SPY, QQQ (-0.32); weak IWM, AAPL, AMZN, MSFT (-0.15 to -0.27); no NVDA, TSLA, META, AMD",
      "the same filter at the weekly options horizon", "5-day forward volatility 9.8% versus 14.7%, p-value below 0.001",
      "25 November to 11 December 2025 positive episode: volatility 8.1% versus 15.1% in the 20 days before"]),
    (10, "tail_days", "The largest moves follow low GEX",
     "res = R.row10(ctx)\nprint(json.dumps(R.show(res), indent=1))\n_ = R.fig10(ctx, res, G.FIGURES / 'row10_tail_days.png')",
     {"r11_share_top20_moves_negative": "share of the 20 largest moves after a negative close", "share_negative": "base rate of negative closes", "r11_share_top20_moves_bottom_third": "share of the 20 largest moves in the own bottom third (base rate 33%)"},
     ["daily print", "0.31 Spearman correlation with the absolute next-day return", "5",
      "by own bottom third against a 33% base rate: SPY 55%, IWM 55%, QQQ 50%, AAPL 50%, AMZN, META, MSFT, AMD 45%; no NVDA 35%, TSLA 25%",
      "risk control: cut size or hold tail hedges when GEX is low",
      "85% of the 20 largest next-day moves and 94% of the 50 widest ranges followed a negative close, base rate 58%",
      "2025-04-08 -9.1 billion dollars, next day +10.0%, range 11.2%"]),
]

for n, slug, title, body, cols, fields in ROWS:
    table = "| field | content |\n|---|---|\n" + "\n".join(f"| {k} | {v} |" for k, v in zip(FIELDS, fields + [OPENERS[n]]))
    cells = [nbf.v4.new_markdown_cell(f"# Row {n}: {title}\n\n{table}"),
             nbf.v4.new_code_cell(SETUP.format(chain=", with_chain=True" if n == 5 else "")),
             nbf.v4.new_code_cell(body),
             nbf.v4.new_code_cell(f"R.per_name({cols!r}).round(3)")]
    for c in cells:
        if c["cell_type"] == "code":
            assert len(c["source"].splitlines()) <= 25
            assert not any(l.strip().startswith("import ") and "," in l for l in c["source"].splitlines()), "multiple imports on one line"
        for bad in ("\u2014", "\u2013"):
            assert bad not in c["source"]
        assert "remark" not in c["source"].lower() and "re-mark" not in c["source"].lower()
    nb = nbf.v4.new_notebook(); nb["cells"] = cells
    nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}}
    out = OUT / f"row{n:02d}_{slug}.ipynb"
    nbf.write(nb, out)
    print("wrote", out.name)
