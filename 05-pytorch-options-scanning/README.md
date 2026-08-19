# 05 - PyTorch Options Scanning

Pull, solve IV/Greeks, rank, repeat.

**Goal:** scan every US equity option, continuously
**Result:** 12.4 scans/min (~4,827 full-market scans per trading day). fp32 GPU solves the whole market in ~25 ms.

## Measured Results (live market)

| engine | contracts | sec | contracts/sec |
|---|---|---|---|
| PyTorch GPU fp32 | 270,480 | 0.02 | 10,908,824 |
| PyTorch GPU | 270,480 | 1.29 | 209,432 |
| PyTorch CPU fp32 | 270,480 | 1.69 | 159,640 |
| PyTorch CPU | 270,480 | 2.02 | 133,626 |
| CuPy | 270,480 | 4.46 | 60,580 |
| CuPy fp32 | 270,480 | 4.61 | 58,662 |
| NumPy | 150,000 | 80.82 | 1,856 (extrapolated) |

- Warm timings (compile pre-market)
- Full IV solve (30 Newton iterations) + 6 Greeks, fused
- fp32 vs fp64: a millionth of a vol point -- does not matter at penny precision
- Also tested on DGX Spark (GB10, aarch64): whole market in 0.18s fp32 -- [as-run notebook](options_scanner_pytorch_DGX_Spark.ipynb)

## Quickstart

```
cd 05-pytorch-options-scanning
uv sync
cp .env.example .env   # add your THETADATA_API_KEY
uv run jupyter lab options_scanner_pytorch.ipynb
```

**Requirements:**
- NVIDIA GPU
- Running Theta Terminal
- Linux: C toolchain for torch.compile -- `sudo apt install build-essential python3-dev`

## Validation

Results are validated using QuantLib: [validate_engines_vs_quantlib.ipynb](validate_engines_vs_quantlib.ipynb)
Rendered record -- re-run needs a cached market day (not included).

## Hardware

RTX PRO 6000 Blackwell (96GB) | 24-core Threadripper | 256GB RAM
Data: [Theta Data](https://www.thetadata.net)
