# GPU-Quant-Finance

Four self-contained **hero notebooks**, each applying a standard scikit-learn technique to
real financial data and benchmarking **CPU vs GPU** through a *one-line* change. The value is
always the same: **same scikit-learn API, no rewrite, NVIDIA GPU speed.** 

> Benchmarks measured on an **NVIDIA RTX PRO 6000 Blackwell (96 GB)** GPU vs an
> **AMD Threadripper PRO 7965WX** CPU with **256 GB RAM**. Numbers are measured will vary by
> hardware and library version. CPU and GPU results agree structurally, not bit-for-bit.
> All notebook are for **educational and benchmark purposes, not trading strategy, not financial advice"**.

## The four notebooks

| # | notebook | One-line GPU fix | Measured CPU → GPU | Short |
|---|---|---|---|---|
| **01** | [Spectral clustering of an equity-options vol surface](01-spec-clust/Spectral_Clustering_Options_Demo_with_Benchmarking.ipynb) | `%load_ext cuml.accel` | **254 s → 1.5 s (~171×)**; up to ~240× at scale | NEED LINK |
| **02** | [Ledoit-Wolf shrinkage cuts portfolio turnover](02-lw-turnover/LedoitWolf_Turnover_Demo.ipynb) | `%load_ext cuml.accel` | **2.5 h → 9 min (~17×)**; −57% turnover | NEED LINK |
| **03** | [KDE of return distributions — the fat tails the normal model misses](03-kde-stylized-facts/KDE_Stylized_Facts_Demo.ipynb) | `%load_ext cuml.accel` | **~6 min → seconds** for 100 bootstrap KDE refits (grows with sample size) | NEED LINK |
| **04** | [HDBSCAN finds "statistical sectors" in a detoned correlation matrix](04-hdbscan-corr-stats-sectors/HDBSCAN_Statistical_Sectors_Demo_with_Benchmarking.ipynb) | `%load_ext cudf.pandas`<br>`%load_ext cuml.accel` | **9 min → ~0.5 s** (HDBSCAN on 8,471 tickers / 72 M correlations) | NEED LINK |

The "one-line, zero-code change, GPU acceleration" is the main point:   
Load NVIDIA's [cuML](https://github.com/rapidsai/cuml)
zero-code-change accelerator (and [cuDF](https://github.com/rapidsai/cudf) for dataframes) before
your imports, and the *same* scikit-learn / pandas code runs on the GPU.

## Data

| Notebook | Dataset | In this repo? |
|---|---|---|
| 01 | `01-spec-clust/options_2026_04_16.parquet` (2.3 MB) | ✅ included — runs as-is |
| 03 | `data/intraday_returns.parquet` (5.5 MB) | ✅ included — runs as-is |
| 02 | Stooq 5-minute + daily OHLCV panels (hundreds of MB) | ➖ not included — **falls back to a synthetic factor-model panel** of the same shape, so the notebook runs end-to-end without them (committed outputs are from the real data) |
| 04 | `data/stooq_daily_us.parquet` (240 MB) | ❌ too large for GitHub — **rebuild from Stooq** (below). `data/stooq_us_etf_tickers.txt` is included. |

### Rebuilding the Stooq dataset (notebook 04)

The 240 MB daily-price parquet is derived from Stooq's free bulk dump, so it is not committed.
To rebuild it:

1. Download the free **bulk daily US** data — `d_us_txt.zip` — from **<https://stooq.com/db/h/>**
2. Unzip it, then run the included builder:
   ```bash
   python data/build_stooq_parquet.py /path/to/unzipped/d_us_txt
   ```
   This writes `data/stooq_daily_us.parquet` and `data/stooq_us_etf_tickers.txt`, which notebook 04 reads.

> Stooq's bulk dump covers *currently listed* tickers only — fine for structure discovery,
> but survivorship-biased for backtests (the notebook calls this out).

## Running

Each notebook runs top-to-bottom on a GPU machine, or on a free GPU in
[Google Colab](https://colab.research.google.com/) / Kaggle (where cuML and cuDF are
pre-installed). Locally, this repo uses a [uv](https://github.com/astral-sh/uv)-managed
Python 3.12 virtualenv with RAPIDS `cu13` (`cuml-cu13`, `cudf-cu13`).
