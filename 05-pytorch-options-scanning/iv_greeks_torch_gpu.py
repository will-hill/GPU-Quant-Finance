"""American-option IV + Greeks -- PyTorch engine, torch.compile optimized

One function. NumPy arrays in, NumPy arrays out:

    solve_iv_greeks(price, S, K, T, r, q, is_call)
        -> (iv, dict: price/delta/gamma/vega/rho/theta)

Same math as iv_greeks_numpy.py (Bjerksund-Stensland 2002, A&S normal CDF,
20-point Gauss-Legendre bivariate, fixed-30 Newton, bump-and-reprice Greeks).

Optimizations vs the original eager version:
- torch.compile fuses the pricer; autotuner picks CUDA tile sizes for this GPU
- _BUCKET padding gives jit a fixed shape so it compiles once per bucket boundary
- pinned memory + non_blocking H2D overlaps PCIe transfers with Python setup
- torch.stack batches iv + all 6 Greeks into ONE D2H transfer

Single-function fusion (vs the earlier split solve_iv/greeks API):
- inputs upload once instead of twice; sigma never round-trips through the host
- the Newton loop's final-iteration price & vega are reused as the Greek
  base/vega: 95 pricer evals instead of 98 (~3% faster at 1M contracts).
  base/vega sit at sigma_29 while the bump evals sit at sigma_30; at
  convergence those agree, so iv/delta/rho match the split engines
  bit-for-bit and price/gamma/vega/theta match to ~1e-8 at p99.9.
"""
import math

import numpy as np
import torch

_DEV = "cuda"
_BUCKET = 50_000

GOLDEN = 0.5 * (math.sqrt(5.0) - 1.0)
TAU = math.sqrt(GOLDEN)

_XG = [0.9931285991850949, 0.9639719272779138, 0.9122344282513259,
       0.8391169718222188, 0.7463319064601508, 0.6360536807265150,
       0.5108670019508271, 0.3737060887154196, 0.2277858511416451,
       0.07652652113349733]
_WG = [0.01761400713915212, 0.04060142980038694, 0.06267204833410906,
       0.08327674157670475, 0.1019301198172404, 0.1181945319615184,
       0.1316886384491766, 0.1420961093183821, 0.1491729864726037,
       0.1527533871307259]


def _in(a):
    arr = np.ascontiguousarray(a, dtype=np.float64)
    return torch.from_numpy(arr).pin_memory().to(_DEV, non_blocking=True)


def _padded(arrays, fills, n):
    n_pad = (-n) % _BUCKET
    if n_pad == 0:
        return arrays
    return [torch.cat([a, torch.full((n_pad,), f, dtype=a.dtype, device=_DEV)])
            for a, f in zip(arrays, fills)]


def _norm_cdf(x):
    z = torch.abs(x)
    t = 1.0 / (1.0 + 0.2316419 * z)
    poly = t * (0.319381530 + t * (-0.356563782 + t * (1.781477937
               + t * (-1.821255978 + t * 1.330274429))))
    tail = 0.3989422804014327 * torch.exp(-0.5 * z * z) * poly
    return torch.where(x >= 0.0, 1.0 - tail, tail)


def _cbnd(a, b, rho):
    """P(X<=a, Y<=b; rho) -- BS2002 only ever uses rho = +/-sqrt(GOLDEN)."""
    h, k = -a, -b
    hk, hs = h * k, 0.5 * (h * h + k * k)
    asr = math.asin(rho)
    bvn = torch.zeros_like(h)
    for xg, wg in zip(_XG, _WG):
        for sgn in (-1.0, 1.0):
            sn = math.sin(asr * (sgn * xg + 1.0) / 2.0)
            bvn = bvn + wg * torch.exp((sn * hk - hs) / (1.0 - sn * sn))
    return bvn * asr / (4.0 * math.pi) + _norm_cdf(-h) * _norm_cdf(-k)


def _phi(fs, t, gamma, h, i, r, b, v):
    v2 = v * v
    vsqt = v * torch.sqrt(t)
    d1 = -(torch.log(fs / h) + (b + (gamma - 0.5) * v2) * t) / vsqt
    d2 = d1 - 2.0 * torch.log(i / fs) / vsqt
    lam = -r + gamma * b + 0.5 * gamma * (gamma - 1.0) * v2
    kappa = 2.0 * b / v2 + (2.0 * gamma - 1.0)
    return (torch.exp(lam * t) * fs ** gamma
            * (_norm_cdf(d1) - (i / fs) ** kappa * _norm_cdf(d2)))


def _psi(fs, t2, gamma, h, i2, i1, t1, r, b, v):
    v2 = v * v
    vsqt1, vsqt2 = v * torch.sqrt(t1), v * torch.sqrt(t2)
    bg1 = (b + (gamma - 0.5) * v2) * t1
    bg2 = (b + (gamma - 0.5) * v2) * t2
    d1 = (torch.log(fs / i1) + bg1) / vsqt1
    d3 = (torch.log(fs / i1) - bg1) / vsqt1
    d2 = (torch.log(i2 * i2 / (fs * i1)) + bg1) / vsqt1
    d4 = (torch.log(i2 * i2 / (fs * i1)) - bg1) / vsqt1
    e1 = (torch.log(fs / h) + bg2) / vsqt2
    e2 = (torch.log(i2 * i2 / (fs * h)) + bg2) / vsqt2
    e3 = (torch.log(i1 * i1 / (fs * h)) + bg2) / vsqt2
    e4 = (torch.log(fs * i1 * i1 / (h * i2 * i2)) + bg2) / vsqt2
    lam = -r + gamma * b + 0.5 * gamma * (gamma - 1.0) * v2
    kappa = 2.0 * b / v2 + (2.0 * gamma - 1.0)
    return (torch.exp(lam * t2) * fs ** gamma
            * (_cbnd(-d1, -e1, TAU)
               - (i2 / fs) ** kappa * _cbnd(-d2, -e2, TAU)
               - (i1 / fs) ** kappa * _cbnd(-d3, -e3, -TAU)
               + (i1 / i2) ** kappa * _cbnd(-d4, -e4, -TAU)))


def _bs2002_call(fs, x, t, r, b, v):
    v2 = v * v
    d1 = (torch.log(fs / x) + (b + 0.5 * v2) * t) / (v * torch.sqrt(t))
    d2 = d1 - v * torch.sqrt(t)
    e_value = (fs * torch.exp((b - r) * t) * _norm_cdf(d1)
               - x * torch.exp(-r * t) * _norm_cdf(d2))

    t1 = GOLDEN * t
    beta = (0.5 - b / v2) + torch.sqrt(torch.abs((b / v2 - 0.5) ** 2 + 2.0 * r / v2))
    beta_m1 = torch.where(torch.abs(beta - 1.0) < 1e-12, 1e-12, beta - 1.0)
    b_inf = (beta / beta_m1) * x
    rmb = torch.where(torch.abs(r - b) < 1e-12, 1e-12, r - b)
    b_zero = torch.maximum(x, (r / rmb) * x)

    denom = (b_inf - b_zero) * b_zero
    i1 = b_zero + (b_inf - b_zero) * (1.0 - torch.exp(-(b * t1 + 2.0 * v * torch.sqrt(t1)) * (x * x / denom)))
    i2 = b_zero + (b_inf - b_zero) * (1.0 - torch.exp(-(b * t + 2.0 * v * torch.sqrt(t)) * (x * x / denom)))
    alpha1 = (i1 - x) * i1 ** (-beta)
    alpha2 = (i2 - x) * i2 ** (-beta)

    big = (alpha2 * fs ** beta
           - alpha2 * _phi(fs, t1, beta, i2, i2, r, b, v)
           + _phi(fs, t1, 1.0, i2, i2, r, b, v)
           - _phi(fs, t1, 1.0, i1, i2, r, b, v)
           - x * _phi(fs, t1, 0.0, i2, i2, r, b, v)
           + x * _phi(fs, t1, 0.0, i1, i2, r, b, v)
           + alpha1 * _phi(fs, t1, beta, i1, i2, r, b, v)
           - alpha1 * _psi(fs, t, beta, i1, i2, i1, t1, r, b, v)
           + _psi(fs, t, 1.0, i1, i2, i1, t1, r, b, v)
           - _psi(fs, t, 1.0, x, i2, i1, t1, r, b, v)
           - x * _psi(fs, t, 0.0, i1, i2, i1, t1, r, b, v)
           + x * _psi(fs, t, 0.0, x, i2, i1, t1, r, b, v))

    american = torch.where(fs >= i2, fs - x, big)
    value = torch.where(b >= r, e_value, american)         # b>=r: never exercise early
    return torch.maximum(value, e_value)


def _price(S, K, T, r, q, sigma, is_call):
    b = r - q
    call = _bs2002_call(S, K, T, r, b, sigma)
    put = _bs2002_call(K, S, T, r - b, -b, sigma)          # put via the BS transform
    return torch.where(is_call, call, put)


# torch.compile the PRICER ONLY, not the Newton loop. A plain Python
# `for _ in range(30)` gets UNROLLED by torch.compile into a ~90-pricer graph
# whose codegen is pathological. PyTorch has no clean fori_loop, so we fuse the
# bounded pricer graph and keep the loop eager: each iteration calls the
# compiled kernel. Same per-eval fusion, no compile explosion. The eager loop
# is also what makes the base/vega reuse below free: they're just Python
# references to the last iteration's tensors.
#
# mode="default" (Triton fusion), NOT reduce-overhead/max-autotune: those wrap
# the kernel in CUDA graphs, which capture a retained memory pool at EACH of the
# ~95 pricer call sites -> multi-GB blowup / OOM. Plain fusion gives the
# throughput win without the per-call-site graph retention.
_price = torch.compile(_price, fullgraph=True, mode="default")


def _solve_greeks(price, S, K, T, r, q, ic):
    """Newton IV solve, then Greeks -- all on device, one pass.

    The final loop iteration's model price and vega ARE the Greek base and
    vega (both at sigma_29); only the delta/gamma/rho/theta bumps re-price
    at the converged sigma_30. 95 pricer evals vs 98 for a split solve+greeks.
    """
    dsig = 1e-4
    ebrt, ert = torch.exp(-q * T), torch.exp(-r * T)
    payoff = torch.where(ic, S * ebrt - K * ert, K * ert - S * ebrt)
    aa = math.sqrt(2.0 * math.pi) / (S * ebrt + K * ert)
    bb = price - payoff / 2.0
    sigma = torch.clamp(aa * (bb + torch.sqrt(bb * bb + payoff * payoff / math.pi)) / torch.sqrt(T), 0.05, 2.0)
    base = vega = None
    for _ in range(30):
        sigma = torch.clamp(sigma, 1e-4, 5.0)
        base = _price(S, K, T, r, q, sigma, ic)
        vega = (_price(S, K, T, r, q, sigma + dsig, ic)
                - _price(S, K, T, r, q, sigma - dsig, ic)) / (2.0 * dsig)
        vega = torch.where(torch.abs(vega) < 1e-8, 1e-8, vega)
        sigma = sigma - (base - price) / vega
    sigma = torch.clamp(sigma, 1e-4, 5.0)

    dS_rel, dr, dT = 1e-3, 1e-4, 1e-4
    hS = dS_rel * S
    up = _price(S + hS, K, T, r, q, sigma, ic)
    dn = _price(S - hS, K, T, r, q, sigma, ic)
    T_dn = torch.clamp(T - dT, min=1e-6)
    return (
        sigma,
        base,
        (up - dn) / (2.0 * hS),
        (up - 2.0 * base + dn) / (hS * hS),
        vega,
        (_price(S, K, T, r + dr, q, sigma, ic)
         - _price(S, K, T, r - dr, q, sigma, ic)) / (2.0 * dr),
        (_price(S, K, T_dn, r, q, sigma, ic) - base) / (T - T_dn),
    )


def solve_iv_greeks(price, S, K, T, r, q, is_call):
    n = len(price)
    ic = torch.as_tensor(np.asarray(is_call)).to(_DEV, non_blocking=True)
    price_t, S_t, K_t, T_t = [_in(a) for a in (price, S, K, T)]
    r_t = torch.full((n,), float(r), dtype=torch.float64, device=_DEV)
    q_t = torch.full((n,), float(q), dtype=torch.float64, device=_DEV)
    price_t, S_t, K_t, T_t, r_t, q_t, ic = _padded(
        [price_t, S_t, K_t, T_t, r_t, q_t, ic],
        [1.0, 100.0, 100.0, 0.5, float(r), float(q), True], n)
    out = _solve_greeks(price_t, S_t, K_t, T_t, r_t, q_t, ic)
    block = torch.stack(out).cpu().numpy()                 # iv + 6 Greeks, one D2H
    keys = ("price", "delta", "gamma", "vega", "rho", "theta")
    return block[0, :n], {k: block[i + 1, :n] for i, k in enumerate(keys)}
