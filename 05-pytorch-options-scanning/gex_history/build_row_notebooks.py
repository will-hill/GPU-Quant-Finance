"""Build one notebook per demonstrated GEX indication into ../gex_rows/. Markdown kept to a title line.

    uv run python gex_history/build_row_notebooks.py
    for nb in gex_rows/*.ipynb; do uv run jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=3600 "$nb"; done
"""
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "gex_rows"
OUT.mkdir(exist_ok=True)

SETUP = '''import sys, json, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, "../gex_history")
import rows as R, gexlib as G
pd.set_option("display.width", 220); pd.set_option("display.max_columns", 40)
ctx = R.load("spy"{chain})
print(ctx["sym"], len(ctx["daily"]), "days")'''

ROWS = [
    (1, "remark_tracking", "Re-marking the previous close's book at the current spot tracks the next official GEX print",
     "res = R.row1(ctx)\nprint(json.dumps(R.show(res), indent=1))\n_ = R.fig1(ctx, res, G.FIGURES / 'row01_remark_tracking.png')",
     {"r1_sign_agree_remark": "sign agreement, re-mark", "r1_sign_agree_stale": "sign agreement, stale", "r1_corr_remark": "level corr, re-mark", "r1_corr_stale": "level corr, stale"}),
    (2, "level_at_1000", "The re-marked GEX level at 10:00 sets the rest-of-day range",
     "res = R.row2(ctx)\nprint(json.dumps({k: v for k, v in R.show(res).items() if 'horizon' not in k}, indent=1))\ndisplay(pd.DataFrame({t: {h: f'{a:+.2f} ({b:+.2f})' for h, (a, b) in d.items()} for t, d in res['spearman_by_time_and_horizon (re-mark, stale)'].items()}).T)\n_ = R.fig2(ctx, res, G.FIGURES / 'row02_level_at_1000.png')",
     {"r2_spearman_1000_rest": "Spearman, 10:00 level vs rest of day", "r2_rest_low_third": "rest of day, low third", "r2_rest_high_third": "rest of day, high third"}),
    (3, "next_30_minutes", "At any half hour the re-marked level predicts the next 30 minutes better than the stale print",
     "res = R.row3(ctx)\nprint(json.dumps({k: v for k, v in R.show(res).items() if 'bucket (' not in k}, indent=1))\n_ = R.fig3(ctx, res, G.FIGURES / 'row03_next_30_minutes.png')",
     {"r3_spearman_remark": "Spearman, re-mark", "r3_spearman_stale": "Spearman, stale", "r3_share_days_flip": "share of days with an intraday flip"}),
    (4, "flip_crossing", "Spot crossing the flip level intraday changes the rest of the day",
     "res = R.row4(ctx)\nprint(json.dumps(R.show(res), indent=1))\n_ = R.fig4(ctx, res, G.FIGURES / 'row04_flip_crossing.png')",
     {"r4_up_cross": "crossed up, rest of day", "r4_up_stay": "stayed, rest of day", "r4_up_n": "n crossed up", "r4_up_ci_excl0": "CI excludes 0 (up)", "r4_down_cross": "crossed down", "r4_down_stay": "stayed", "r4_down_n": "n crossed down", "r4_down_ci_excl0": "CI excludes 0 (down)"}),
    (5, "zero_dte", "More same-day-expiry gamma at 15:30, quieter last half hour; the 0DTE layer is larger than the standing book",
     "res = R.row5(ctx)\nprint(json.dumps({k: v for k, v in R.show(res['t8']).items() if k != 'by_bucket'}, indent=1))\nprint(json.dumps(R.show(res['controls']), indent=1))\ndisplay(pd.DataFrame(res['t8']['by_bucket']).set_index('bucket').round(3))\n_ = R.fig5(ctx, res, G.FIGURES / 'row05_zero_dte.png')",
     {"r5_days_with_0dte": "days with same-day expiry", "r5_share_0dte_of_book_1000": "0DTE share of gamma at 10:00", "r5_spearman_1530_last": "Spearman, 15:30 exposure vs last half hour", "r5_last_low": "last half hour, low third", "r5_last_high": "high third", "r5_last_p": "p"}),
    (6, "persistence", "The GEX regime is sticky day to day",
     "res = R.row6(ctx)\nprint(json.dumps(R.show(res), indent=1))\n_ = R.fig6(ctx, res, G.FIGURES / 'row06_persistence.png')",
     {"share_negative": "share of negative days", "r6_persistence": "P(same sign next day)"}),
    (7, "intraday_realized_vol", "The daily print sets intraday realized vol at every horizon",
     "res = R.row7(ctx)\ndisplay(res.round(4))\n_ = R.fig7(ctx, res, G.FIGURES / 'row07_intraday_realized_vol.png')",
     {"r7_rv_ratio_neg_over_pos": "realized vol, negative over positive (30-min)", "r7_n_neg": "negative days", "r7_p": "p"}),
    (8, "next_day_range", "A positive close means a quieter next session, a negative close a wider one",
     "res = R.row8(ctx)\ndisplay(res['sign_cut'].drop(columns=['outcome']).round(4))\nprint(json.dumps({k: v for k, v in R.show(res).items() if k not in ('sign_cut', 'cross_correlation')}, indent=1))\ndisplay(res['cross_correlation'].T.round(3))\n_ = R.fig8(ctx, res, G.FIGURES / 'row08_next_day_range.png')",
     {"r8_range_pos": "next-day range after positive", "r8_range_neg": "after negative", "r8_n_neg": "negative days", "r8_range_top_third": "own top third", "r8_range_bottom_third": "own bottom third", "r8_p_tercile": "p (terciles)", "r8_spearman_gex_next_range": "Spearman"}),
    (9, "next_week_vol", "A positive close means lower realized vol over the next week",
     "res = R.row9(ctx)\ndisplay(res['sign_cut'].drop(columns=['outcome']).round(4))\nprint(json.dumps({k: v for k, v in R.show(res).items() if k != 'sign_cut'}, indent=1))\n_ = R.fig9(ctx, res, G.FIGURES / 'row09_next_week_vol.png')",
     {"r9_fvol_top_third": "5-day vol, own top third", "r9_fvol_bottom_third": "own bottom third", "r9_spearman_gex_fvol": "Spearman"}),
    (10, "tail_days", "The largest moves follow low GEX",
     "res = R.row10(ctx)\nprint(json.dumps(R.show(res), indent=1))\n_ = R.fig10(ctx, res, G.FIGURES / 'row10_tail_days.png')",
     {"r11_share_top20_moves_negative": "top-20 moves after a negative close", "share_negative": "base rate", "r11_share_top20_moves_bottom_third": "top-20 moves in own bottom third (base 33%)"}),
]

for n, slug, title, body, cols in ROWS:
    cells = [nbf.v4.new_markdown_cell(f"# Row {n}: {title}"),
             nbf.v4.new_code_cell(SETUP.format(chain=", with_chain=True" if n == 5 else "")),
             nbf.v4.new_code_cell(body),
             nbf.v4.new_code_cell(f"R.per_name({cols!r}).round(3)")]
    for c in cells:
        assert len(c["source"].splitlines()) <= 25
        for bad in ("—", "–"):
            assert bad not in c["source"]
    nb = nbf.v4.new_notebook(); nb["cells"] = cells
    nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}}
    out = OUT / f"row{n:02d}_{slug}.ipynb"
    nbf.write(nb, out)
    print("wrote", out.name)
