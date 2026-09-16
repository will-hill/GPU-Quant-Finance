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
    if not all(k in s for k in ("next_day_by_tercile", "pooled_tercile_test", "cross_sectional_sort_by_key", "pooled_intraday", "same_day_tests")):
        return "## Conclusions\n\n(filled after the first execution)\n\nEducational analysis, not a trading strategy."
    byname, pt, keys, intr, same = s["next_day_by_tercile"], s["pooled_tercile_test"], s["cross_sectional_sort_by_key"], s["pooled_intraday"], s["same_day_tests"]
    ratios = {k: v["ratio_bottom_over_top"] for k, v in byname.items() if isinstance(v.get("ratio_bottom_over_top"), float) and v["ratio_bottom_over_top"] == v["ratio_bottom_over_top"]}
    n_above = sum(r > 1 for r in ratios.values()); med = sorted(ratios.values())[len(ratios) // 2]
    top3 = sorted(ratios.items(), key=lambda kv: -kv[1])[:3]
    f = intr["flip_vs_same"]; sd = same["daily_same_day_bottom_minus_top"]; si = same["intraday_same_bucket_flipped_minus_same"]
    fmt = lambda xs: ", ".join(f"{v:.3f}" for v in xs)  # noqa: E731
    lines = ["## Conclusions", "",
             f"1. Within a name, low GEX goes with a wider next day in {n_above} of {len(ratios)} names (median ratio {med:.2f}), strongest in the index books: " + ", ".join(f"{k} {v:.2f}" for k, v in top3) + f". Pooled over {pt['n_names']} names, bottom third {pt['range_rel_bottom']:.3f} against top third {pt['range_rel_top']:.3f}, CI of the difference [{pt['ci_lo']:+.3f}, {pt['ci_hi']:+.3f}]. Single-name books are call-heavy and rarely negative, so the cut is the name's own tercile, not the sign.",
             f"2. That within-name effect is a market-wide time effect, not a way to pick names. On the same day, names in their own bottom GEX tercile are not wider than names in their own top tercile: difference {sd['mean_diff']:+.3f}, CI [{sd['ci_lo']:+.3f}, {sd['ci_hi']:+.3f}] over {sd['n_groups']} days. The share of names in their low state on D correlates {same['time_effect_spearman_share_low_vs_market_range_next']:+.2f} with the market's average relative range on D+1. Ranking names across the cross-section does not order tomorrow's relative range under any key (bins most negative to most positive: GEX per dollar traded {fmt(keys['gex_per_dv'])}; own z-score {fmt(keys['gex_z'])}; raw GEX {fmt(keys['gex_net_usd'])}).",
             f"3. Re-marking tracks the next official print in every name: mean sign agreement {intr['t6_mean']['sign_agreement_remark']:.1%} for the re-marked book against {intr['t6_mean']['sign_agreement_stale']:.1%} for the stale print, level correlation {intr['t6_mean']['corr_level_remark']:.2f} against {intr['t6_mean']['corr_level_stale']:.2f}; no name is better served by the stale value. The re-marked sign leaves the prior close's sign on {intr['share_days_with_flip']:.0%} of name-days.",
             f"4. Pooled over {intr['n_names']} names and {intr['n_buckets']:,} half-hour buckets, a flipped re-marked sign is followed by a next bucket in the predicted direction (negative to positive {f['prev_negative_flipped_positive']['next_range_adj_flipped']:.2f} against {f['prev_negative_flipped_positive']['next_range_adj_same']:.2f}; positive to negative {f['prev_positive_flipped_negative']['next_range_adj_flipped']:.2f} against {f['prev_positive_flipped_negative']['next_range_adj_same']:.2f}), but at the same date and bucket flipped names are not wider than names that held (differences {si['prev_negative_flipped_positive']['mean_diff']:+.3f}, CI [{si['prev_negative_flipped_positive']['ci_lo']:+.3f}, {si['prev_negative_flipped_positive']['ci_hi']:+.3f}], and {si['prev_positive_flipped_negative']['mean_diff']:+.3f}, CI [{si['prev_positive_flipped_negative']['ci_lo']:+.3f}, {si['prev_positive_flipped_negative']['ci_hi']:+.3f}]). Across names, intraday flips are the market's move showing through every book.",
             "5. What a multi-name scanner is therefore demonstrated to do: keep each name's gamma regime current between prints, at a cost the GPU makes negligible. What it is not demonstrated to do: tell which name will be wider or quieter than its peers tomorrow or in the next half hour.",
             "", "Educational analysis, not a trading strategy."]
    return "\n".join(lines)


md("""
# GEX across the most active names: whether the SPY result generalizes and what the intraday re-mark adds

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
## Whether the daily result generalizes

Within each name: mean next-day range after a positive-GEX close against after a negative one, the ratio, and a Mann-Whitney p-value. Then the pooled test on the range relative to the name's own trailing 20-day mean, so names with different vol levels can be stacked.
""")

code("""
names = list(summary.index.str.lower())
bytercile = X.next_day_by_tercile(names)
_ = X.fig_tercile_by_name(bytercile, G.FIGURES / "f11_tercile_by_name.png")
bytercile.style.format({"share_negative": "{:.0%}", "gex_bottom_third_max_bn": "{:+.2f}", "gex_top_third_min_bn": "{:+.2f}", "range_rel_bottom": "{:.3f}",
                        "range_rel_top": "{:.3f}", "ratio_bottom_over_top": "{:.2f}", "absret_bottom": "{:.2%}", "absret_top": "{:.2%}", "p_mwu": "{:.3f}"})
""")

md("""
Single names are call-heavy books that rarely turn negative, so the sign cut leaves few negative days; the table above uses each name's own GEX terciles. The sign cut is shown next for completeness, then the pooled tests.
""")

code("""
byname = X.next_day_by_regime(names)
byname.style.format({"share_negative": "{:.0%}", "range_next_pos": "{:.2%}", "range_next_neg": "{:.2%}", "ratio_neg_over_pos": "{:.2f}",
                     "range_rel_pos": "{:.3f}", "range_rel_neg": "{:.3f}", "absret_next_pos": "{:.2%}", "absret_next_neg": "{:.2%}", "p_mwu_range": "{:.3f}"})
""")

code("""
panel = X.pooled_panel(names)
panel = panel[panel.index >= "2025-09-15"]
pooled = X.pooled_regime_test(panel)
pooled_t = X.pooled_tercile_test(panel)
print("sign cut:", json.dumps(pooled, indent=1))
print("tercile cut:", json.dumps(pooled_t, indent=1))
xs = X.cross_sectional_sort(panel, key="gex_per_dv", n_bins=5)
xs_z = X.cross_sectional_sort(panel, key="gex_z", n_bins=5)
xs_raw = X.cross_sectional_sort(panel, key="gex_net_usd", n_bins=5)
_ = X.fig_xs_sort(xs, "net GEX per dollar of 20-day average traded value", G.FIGURES / "f12_cross_sectional_sort.png")
print("next-day relative range by bin, most negative to most positive:")
for lab, t in (("GEX per dollar traded", xs), ("GEX z-score against the name's own 60 days", xs_z), ("raw net GEX", xs_raw)):
    print(f"  {lab}: {[round(v, 3) for v in t['range_rel_next']]}")
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

md("""
## Whether GEX separates names from each other on the same day

The pooled tests above stack name-days, so a market-wide effect (every book turns negative on the same wide days) would pass them. The scanner question is different: whether, at the same time, a name's GEX state says anything about that name against the others. Daily: within each date, names in their own bottom GEX tercile against names in their own top tercile. Intraday: within each date and bucket, names whose re-marked sign flipped against names whose sign held. The time effect itself is the correlation between the share of names in their low state on D and the market's average relative range on D+1.
""")

code("""
ob_all = pd.concat([r["ob"] for r in intra], ignore_index=True)
same = X.same_day_tests(panel, ob_all)
print(json.dumps(same, indent=1))
""")

code("""
G.save_json({"symbols": names, "next_day_by_name": byname.to_dict("index"), "next_day_by_tercile": bytercile.to_dict("index"),
             "pooled_regime_test": pooled, "pooled_tercile_test": pooled_t,
             "cross_sectional_sort": {str(k): v for k, v in xs.to_dict("index").items()},
             "cross_sectional_sort_by_key": {"gex_per_dv": [float(v) for v in xs["range_rel_next"]], "gex_z": [float(v) for v in xs_z["range_rel_next"]], "gex_net_usd": [float(v) for v in xs_raw["range_rel_next"]]},
             "same_day_tests": same,
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
