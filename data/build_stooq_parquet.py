"""Convert Stooq's free bulk daily US dump into the parquet the notebook reads.

1. Download d_us_txt.zip from https://stooq.com/db/h/ and unzip it.
2. python build_stooq_parquet.py /path/to/d_us_txt
   -> writes stooq_daily_us.parquet and stooq_us_etf_tickers.txt into this data/ dir

Requires polars (`pip install polars`).
"""
import glob
import os
import sys

import polars as pl

root = sys.argv[1] if len(sys.argv) > 1 else "d_us_txt"
files = [f for f in glob.glob(f"{root}/daily/us/**/*.us.txt", recursive=True)
         if os.path.getsize(f) > 100]
print(f"files found (non-empty): {len(files)}")

df = (
    pl.scan_csv(files, has_header=True, schema_overrides={"<DATE>": pl.Utf8},
                include_file_paths="path")
    .select(
        pl.col("<TICKER>").str.strip_suffix(".US").alias("ticker"),
        pl.col("<DATE>").str.to_date("%Y%m%d").alias("date"),
        pl.col("<CLOSE>").alias("close"),
        pl.col("<VOL>").alias("vol"),
        pl.col("path").str.contains("etfs").alias("is_etf"),
    )
    .collect()
)
out_dir = os.path.dirname(os.path.abspath(__file__))  # this script lives in data/
os.makedirs(out_dir, exist_ok=True)
df.drop("is_etf").write_parquet(os.path.join(out_dir, "stooq_daily_us.parquet"))
etfs = df.filter(pl.col("is_etf"))["ticker"].unique().sort()
with open(os.path.join(out_dir, "stooq_us_etf_tickers.txt"), "w") as f:
    f.write("\n".join(etfs.to_list()) + "\n")
print(f"{len(df):,} rows, {df['ticker'].n_unique():,} tickers, "
      f"{len(etfs):,} ETFs -> {out_dir}")
