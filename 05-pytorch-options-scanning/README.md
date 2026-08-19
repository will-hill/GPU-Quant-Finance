# 05 - PyTorch Options Scanning

Pull, solve IV/Greeks, rank, repeat.

**Goal:** scan every US equity option, continuously
**Result:** 12.3 scans/min (~4,800 full-market scans per trading day). fp32 GPU solves the whole market in ~25 ms.

## Measured Results (live market)

| engine | contracts | sec | contracts/sec |
|---|---|---|---|
| PyTorch GPU fp32 | 244,516 | 0.02 | 10,222,109 |
| PyTorch GPU | 244,516 | 1.27 | 192,250 |
| PyTorch CPU fp32 | 244,516 | 1.43 | 171,318 |
| PyTorch CPU | 244,516 | 1.77 | 137,868 |
| CuPy | 244,516 | 4.46 | 54,826 |
| CuPy fp32 | 244,516 | 4.60 | 53,138 |
| NumPy | 150,000 | 81.42 | 1,842 (extrapolated) |

- Warm timings (compile pre-market)
- Full IV solve (30 Newton iterations) + 6 Greeks, fused
- fp32 vs fp64: a millionth of a vol point -- does not matter at penny precision
- Also tested on DGX Spark (GB10, aarch64): whole market in 0.13s fp32

## Quickstart

```
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

MIT License
