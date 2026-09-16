"""Device-resident re-mark of a solved chain on a spot grid: the scanner's intraday workload as one kernel.

One thread per contract loops over the grid points, calls the engine's device function for gamma at
spot = S0 x rel, and writes signed GEX per (contract, grid point). The per-day sums are reduced on the
device with CuPy; only the (days x grid) profile comes back to the host. The engine's boundary solve
runs once per evaluation (for calls the put-call symmetry makes the boundary depend on spot), so this
is the full cost of a re-mark, not a shortcut.

    uv run python gex_history/remark_gpu.py            # times the SPY history and one day; writes results/remark_timing.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "engine_alo"))
import gexlib as G  # noqa: E402
import intraday as I  # noqa: E402


def build_remark_kernel(dtype=np.float32):
    from numba import cuda
    from alo_numba import make_core
    core = make_core(dtype, "cuda", 7, 7, 27, m_iter=4)

    @cuda.jit
    def kern(S0, K, r, q, sig, T, style, w, rel, glx_l, glw_l, glx_p, glw_p, chx, chw, scratch, out):
        i = cuda.grid(1)
        if i < S0.shape[0]:
            for j in range(rel.shape[0]):
                s = S0[i] * rel[j]
                v, d, g = core(s, K[i], r[i], q[i], sig[i], T[i], style[i], glx_l, glw_l, glx_p, glw_p, chx, chw, scratch[i])
                out[i, j] = w[i] * g * s * s          # w = sign x open interest; x100 x 0.01 = 1
    return kern


class RemarkGPU:
    """Holds the compiled kernel and tables; `profiles(sol_ok, rel)` returns (days x grid) net GEX and timings."""

    def __init__(self, dtype=np.float32, threads=256):
        from alo_numba import Tables
        self.f, self.threads = dtype, threads
        self.kern = build_remark_kernel(dtype)
        self.tab = Tables(dtype, 7, 7, 27)

    def profiles(self, sol_ok: pd.DataFrame, rel: np.ndarray = I.REL, return_timing=True):
        import cupy as cp
        from numba import cuda
        from alo_numba import style_codes
        f = self.f
        sol_ok = sol_ok.sort_values(["date"], kind="stable")
        n, m = len(sol_ok), len(rel)
        host = [np.ascontiguousarray(sol_ok[c].to_numpy(np.float64).astype(f)) for c in ("S", "strike", "r", "q", "iv", "T_years")]
        is_euro = sol_ok["symbol"].astype(str).str.upper().isin(G.INDEX_ROOTS).to_numpy()
        style = style_codes(sol_ok["is_call"].to_numpy(), is_euro)
        w = (np.where(sol_ok["is_call"].to_numpy(), 1.0, -1.0) * sol_ok["open_interest"].to_numpy(np.float64)).astype(f)
        codes, days = pd.factorize(sol_ok["date"], sort=True)
        starts = np.flatnonzero(np.r_[True, codes[1:] != codes[:-1]])
        t0 = time.perf_counter()
        d_in = [cuda.to_device(a) for a in host] + [cuda.to_device(style), cuda.to_device(w), cuda.to_device(rel.astype(f))]
        d_tab = [cuda.to_device(a) for a in self.tab.args()]
        d_sc = cuda.device_array((n, 2 * self.tab.n_nodes), dtype=f)
        d_out = cuda.device_array((n, m), dtype=f)
        cuda.synchronize(); t_h2d = time.perf_counter() - t0
        blocks = (n + self.threads - 1) // self.threads
        t0 = time.perf_counter()
        self.kern[blocks, self.threads](*d_in, *d_tab, d_sc, d_out); cuda.synchronize()
        t_kernel = time.perf_counter() - t0
        t0 = time.perf_counter()
        prof = cp.add.reduceat(cp.asarray(d_out).astype(cp.float64), cp.asarray(starts), axis=0).get()   # fp64 accumulation over contracts
        cuda.synchronize(); t_reduce = time.perf_counter() - t0
        out = pd.DataFrame(prof.astype(np.float64), index=pd.DatetimeIndex(days, name="date"), columns=rel)
        timing = {"n_contracts": int(n), "grid_points": int(m), "evaluations": int(n * m), "t_h2d_s": t_h2d, "t_kernel_s": t_kernel, "t_reduce_and_d2h_s": t_reduce,
                  "t_total_s": t_h2d + t_kernel + t_reduce}
        return (out, timing) if return_timing else out


def main():
    from numba import cuda
    if len(cuda.gpus) > 1:
        cuda.select_device(1)
    dev = cuda.get_current_device().name
    dev = dev.decode() if isinstance(dev, bytes) else str(dev)
    SCR = Path("/tmp/claude-1000/-home-will-git-GPU-Quant-Finance/256e06b7-f713-4c41-a576-406ce3558e40/scratchpad")
    sol = pd.read_parquet(SCR / "sol.parquet") if (SCR / "sol.parquet").exists() else None
    if sol is None:
        stock, sofr, days = G.load_stock(), G.load_sofr(), G.study_days()
        chain, _ = G.prep_all(days, stock, sofr, verbose=False)
        sol, _ = G.solve_chain(chain)
    ok = sol[sol["iv_status"] == 0]
    rm = RemarkGPU(np.float32)
    one = ok[ok["date"] == pd.Timestamp("2026-07-08")]
    rm.profiles(one)                                   # compile
    res = {"device": dev, "dtype": "fp32", "cpu": "AMD Threadripper PRO 7965WX, 48 numba threads, fp64"}
    t = [rm.profiles(one)[1] for _ in range(5)]
    res["one_day"] = {k: (float(np.median([x[k] for x in t])) if k.startswith("t_") else t[0][k]) for k in t[0]}
    prof_gpu, tt = rm.profiles(ok)
    res["full_history"] = tt
    prof_cpu = pd.read_parquet(G.CACHE / "spy_profiles.parquet"); prof_cpu.columns = prof_cpu.columns.astype(float)
    rel = (prof_gpu.to_numpy() - prof_cpu.to_numpy()) / np.maximum(np.abs(prof_cpu.to_numpy()), 1e8)
    res["max_rel_diff_vs_cpu_fp64_profile"] = float(np.abs(rel).max())
    # CPU reference on the same day and the whole history (Engine.price path, fp64) for the record
    t0 = time.perf_counter(); I.daily_profiles(one, target="cpu", dtype=np.float64, verbose=False); res["one_day"]["cpu_fp64_engine_price_s"] = time.perf_counter() - t0
    res["full_history"]["cpu_fp64_engine_price_s_measured_earlier"] = 107.0
    rate = res["full_history"]["evaluations"] / res["full_history"]["t_kernel_s"]
    res["kernel_rate_evaluations_per_s"] = rate
    # scanner-scale extrapolation: 200 names x contracts per name-day x 81 spot points
    for per_name in (1500, 3000):
        ev = 200 * per_name * len(I.REL)
        res[f"scanner_200_names_{per_name}_contracts_each"] = {"evaluations": ev, "gpu_kernel_s": ev / rate,
                                                                "cpu_fp64_s_at_measured_rate": ev / (res["full_history"]["evaluations"] / 107.0)}
    G.save_json(res, G.RESULTS / "remark_timing.json")
    print(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
