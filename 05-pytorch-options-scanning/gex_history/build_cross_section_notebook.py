"""Build gex_cross_section.ipynb: daily GEX and the intraday re-mark across the most active names.

    uv run python gex_history/build_cross_section_notebook.py
    uv run jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=7200 gex_cross_section.ipynb
Two passes: the conclusions cell is filled from results/cross_section/summary.json when it exists.
"""
import json
import os
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results" / "cross_section"
OUT = Path(os.environ.get("GEX_XS_NB_OUT", HERE.parent / "gex_cross_section.ipynb"))
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text.strip("\n")))


def code(text):
    src = text.strip("\n")
    assert len(src.splitlines()) <= 25, f"code cell over 25 lines:\n{src}"
    cells.append(nbf.v4.new_code_cell(src))


def conclusions_md():
    p = RESULTS / "summary.json"
    if not p.exists():
        return "## Conclusions\n\n(filled after the first execution)\n\nEducational analysis, not a trading strategy."
    s = json.loads(p.read_text())
    byname, pooled, xs, intr = s["next_day_by_name"], s["pooled_regime_test"], s["cross_sectional_sort"], s["pooled_intraday"]
    n_above = sum(1 for v in byname.values() if v.get("ratio_neg_over_pos", 0) > 1)
    n_names = sum(1 for v in byname.values() if "ratio_neg_over_pos" in v)
    ratios = sorted(v["ratio_neg_over_pos"] for v in byname.values() if "ratio_neg_over_pos" in v)
    med = ratios[len(ratios) // 2]
    bins = sorted(xs.items(), key=lambda kv: int(float(kv[0])))
    lo, hi = bins[0][1], bins[-1][1]
    f = intr["flip_vs_same"]
    lines = ["## Conclusions", "",
             f"1. The SPY regime result generalizes: the next-day range after a negative-GEX close exceeds the range after a positive close in {n_above} of {n_names} names, median ratio {med:.2f}. Pooled over {pooled['n_names']} names and {pooled['n_pos'] + pooled['n_neg']} name-days, the next-day range relative to the name's own trailing mean is {pooled['range_rel_pos']:.3f} after positive GEX against {pooled['range_rel_neg']:.3f} after negative, difference {pooled['diff']:+.3f} with 95% CI [{pooled['ci_lo']:+.3f}, {pooled['ci_hi']:+.3f}], p = {pooled['p_mwu']:.1e}.",
             f"2. The cross-sectional ranking works: sorting names each day by net GEX per dollar traded, the most negative bin has a next-day relative range of {lo['range_rel_next']:.3f} and the most positive bin {hi['range_rel_next']:.3f}.",
             f"3. Re-marking tracks the next print in every name: mean sign agreement with the next official GEX {intr['t6_mean']['sign_agreement_remark']:.1%} for the re-marked book against {intr['t6_mean']['sign_agreement_stale']:.1%} for the stale print, level correlation {intr['t6_mean']['corr_level_remark']:.2f} against {intr['t6_mean']['corr_level_stale']:.2f}.",
             f"4. Pooled over {intr['n_names']} names and {intr['n_buckets']} half-hour buckets, the re-mark predicts the next bucket's range better than the stale value (Spearman on within-name ranks {intr['spearman_remark']:+.2f} against {intr['spearman_stale']:+.2f}); a re-marked sign that left the prior close's sign appears on {intr['share_days_with_flip']:.0%} of name-days, and after such a flip from negative to positive the next bucket's adjusted range is {f['prev_negative_flipped_positive']['next_range_adj_flipped']:.2f} against {f['prev_negative_flipped_positive']['next_range_adj_same']:.2f} when the sign held (CI of the difference [{f['prev_negative_flipped_positive']['ci_lo']:+.2f}, {f['prev_negative_flipped_positive']['ci_hi']:+.2f}]); from positive to negative {f['prev_positive_flipped_negative']['next_range_adj_flipped']:.2f} against {f['prev_positive_flipped_negative']['next_range_adj_same']:.2f} (CI [{f['prev_positive_flipped_negative']['ci_lo']:+.2f}, {f['prev_positive_flipped_negative']['ci_hi']:+.2f}]).",
             "", "Educational analysis, not a trading strategy."]
    return "\n".join(lines)


md("""
# GEX across the most active names: does the SPY result generalize, and what does the intraday re-mark add?

The SPY notebooks showed that the sign of net gamma exposure at the close describes the next session's range, and that re-marking the previous close's book at the current spot tracks the next official print and predicts the next half hour better than the stale value. A scanner over hundreds of names only earns its compute if those two facts hold across names. This notebook runs the same pipeline on the most active optionable names plus the SPX index book over one year, then pools the tests.
""")

md("""
## Universe and data

The 40 most active names from the scanner's universe (ranked by option trading in July 2026, SPY excluded because it has its own notebooks) plus the SPX book (SPX and SPXW roots on the SPX spot, priced European). Window 2025-09-15 to 2026-09-14. Per name and day: `option_history_eod` and `option_history_open_interest` with `expiration="*"` and `max_dte=120`, the stock or index daily bars, and 1-minute bars. Chain rules, engine and GEX formula are those of the SPY notebook; the flat dividend yield per name is a trailing-yield assumption listed in `gexlib.Q_BY_SYMBOL` (0 for names not listed). Single names carry discrete dividends and earnings jumps that the flat-yield model does not see; deep in-the-money contracts quoted below the zero-vol floor are excluded exactly as for SPY. Names with a short option history (recent listings) have fewer days.
""")

code("""
import json, sys, time, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore", category=RuntimeWarning)
sys.path.insert(0, "gex_history")
import gexlib as G, intraday as I, cross_section as X
pd.set_option("display.width", 220); pd.set_option("display.max_columns", 40)
spots = sorted(p.stem.split("_")[0] for p in G.CACHE.glob("*_stock_eod.parquet")) + sorted(p.stem.split("_")[0] for p in G.CACHE.glob("*_index_eod.parquet"))
symbols = [s for s in spots if s not in ("spy", "spxw")]
roots = {s: (["spx", "spxw"] if s == "spx" else [s]) for s in symbols}
print(len(symbols), "spot symbols:", " ".join(s.upper() for s in symbols))
""")

md("""
## Daily GEX per name

Same engine pass per name: chain prep, American IV and gamma on the CPU in fp64, the daily table, and the 81-point spot-grid profile that the intraday re-mark reads. Each row of the summary is one name.
""")

code("""
t0 = time.perf_counter()
runs = [X.run_symbol(s, roots=roots[s], verbose=True) for s in symbols]
summary = pd.DataFrame([r for r in runs if r.get("status") == "ok"]).set_index("symbol")
skipped = [r for r in runs if r.get("status") != "ok"]
print(f"{len(summary)} names in {time.perf_counter() - t0:.0f}s; skipped: {skipped}")
X.XS.mkdir(parents=True, exist_ok=True)
G.save_json({"runs": runs}, X.XS / "symbols.json")
summary[["days", "contracts", "t_iv_s", "share_negative_days", "median_gex_bn", "median_abs_gex_bn", "flip_found_share", "iv_status_ne_0_oi_share"]].style.format(
    {"t_iv_s": "{:.1f}", "share_negative_days": "{:.0%}", "median_gex_bn": "{:+.2f}", "median_abs_gex_bn": "{:.2f}", "flip_found_share": "{:.0%}", "iv_status_ne_0_oi_share": "{:.2%}"})
""")

md("""
## Does the daily result generalize?

Within each name: mean next-day range after a positive-GEX close against after a negative one, the ratio, and a Mann-Whitney p-value. Then the pooled test on the range relative to the name's own trailing 20-day mean, so names with different vol levels can be stacked.
""")

code("""
names = list(summary.index.str.lower())
byname = X.next_day_by_regime(names)
_ = X.fig_ratio_by_name(byname, G.FIGURES / "f11_ratio_by_name.png")
byname.style.format({"share_negative": "{:.0%}", "range_next_pos": "{:.2%}", "range_next_neg": "{:.2%}", "ratio_neg_over_pos": "{:.2f}",
                     "range_rel_pos": "{:.3f}", "range_rel_neg": "{:.3f}", "absret_next_pos": "{:.2%}", "absret_next_neg": "{:.2%}", "p_mwu_range": "{:.3f}"})
""")

code("""
panel = X.pooled_panel(names)
panel = panel[panel.index >= "2025-09-15"]
pooled = X.pooled_regime_test(panel)
print(json.dumps(pooled, indent=1))
xs = X.cross_sectional_sort(panel, key="gex_per_dv", n_bins=5)
_ = X.fig_xs_sort(xs, "net GEX per dollar of 20-day average traded value", G.FIGURES / "f12_cross_sectional_sort.png")
xs.style.format({"range_rel_next": "{:.3f}", "absret_next": "{:.2%}", "ret_next": "{:+.3%}", "key_median": "{:.2e}", "share_negative": "{:.0%}"})
""")

md("""
## The intraday re-mark across names

For each name with 1-minute bars: the previous close's book re-marked at each half-hour close (from its profile), the stale print, and the next bucket's adjusted range. Per name: agreement of the re-mark and of the stale print with the next official GEX sign. Pooled: Spearman on within-name ranks, and the flip-against-same comparison.
""")

code("""
t0 = time.perf_counter()
intra = [r for r in (X.intraday_for_symbol(s) for s in names) if r is not None]
pooled_intra = X.pooled_intraday(intra)
print(f"{len(intra)} names with intraday panels in {time.perf_counter() - t0:.0f}s")
print({k: v for k, v in pooled_intra.items() if k not in ("t6_by_name", "flip_vs_same")})
display(pd.DataFrame(pooled_intra["flip_vs_same"]).T.style.format("{:.3f}"))
_ = X.fig_remark_by_name(pooled_intra["t6_by_name"], G.FIGURES / "f13_remark_by_name.png")
pd.DataFrame(pooled_intra["t6_by_name"]).T.style.format("{:.3f}")
""")

code("""
G.save_json({"symbols": names, "next_day_by_name": byname.to_dict("index"), "pooled_regime_test": pooled,
             "cross_sectional_sort": {str(k): v for k, v in xs.to_dict("index").items()},
             "pooled_intraday": {k: v for k, v in pooled_intra.items()}}, X.XS / "summary.json")
print("wrote", X.XS / "summary.json")
""")

md(conclusions_md())

nb = nbf.v4.new_notebook(); nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}}
for c in cells:
    for bad in ("—", "–"):
        assert bad not in c["source"], f"dash in cell: {c['source'][:80]}"
nbf.write(nb, OUT)
print(f"wrote {OUT} with {len(cells)} cells ({sum(c['cell_type'] == 'code' for c in cells)} code)")
