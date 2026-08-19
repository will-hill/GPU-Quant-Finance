"""American-option IV + Greeks

One function. NumPy arrays in, NumPy arrays out:

    solve_iv_greeks(price, S, K, T, r, q, is_call)
        -> (iv, dict: price/delta/gamma/vega/rho/theta)

Pricer: Bjerksund-Stensland 2002 American approximation (golden-ratio two-step).
Normal CDF: Abramowitz-Stegun 26.2.17. Bivariate CDF: 20-point Gauss-Legendre.
IV: fixed 30 Newton iterations, no early exit -- every contract does identical
work, so the whole batch is pure array math. Greeks: bump-and-reprice, reusing
the final Newton iteration's price & vega as the Greek base/vega (95 pricer
evals vs 98 for a split solve+greeks; base/vega sit at sigma_29, the bump
evals at sigma_30 -- identical at convergence).

fp32 variant of iv_greeks_cupy.py
"""

import math

import cupy as xp

# CPU only xp.seterr(all="ignore")        # BS2002 evaluates both call/put branches; unused one overflows

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
    return xp.asarray(a, dtype=xp.float32)


def _out(a):
    return xp.asnumpy(a)       # copy GPU result back to a NumPy array (synchronizes)


def _norm_cdf(x):
    z = xp.abs(x)
    t = 1.0 / (1.0 + 0.2316419 * z)
    poly = t * (0.319381530 + t * (-0.356563782 + t * (1.781477937
               + t * (-1.821255978 + t * 1.330274429))))
    tail = 0.3989422804014327 * xp.exp(-0.5 * z * z) * poly
    return xp.where(x >= 0.0, 1.0 - tail, tail)


def _cbnd(a, b, rho):
    """P(X<=a, Y<=b; rho) -- BS2002 only ever uses rho = +/-sqrt(GOLDEN)."""
    h, k = -a, -b
    hk, hs = h * k, 0.5 * (h * h + k * k)
    asr = math.asin(rho)
    bvn = xp.zeros_like(h)
    for xg, wg in zip(_XG, _WG):
        for sgn in (-1.0, 1.0):
            sn = math.sin(asr * (sgn * xg + 1.0) / 2.0)
            bvn = bvn + wg * xp.exp((sn * hk - hs) / (1.0 - sn * sn))
    return bvn * asr / (4.0 * math.pi) + _norm_cdf(-h) * _norm_cdf(-k)


def _phi(fs, t, gamma, h, i, r, b, v):
    v2 = v * v
    vsqt = v * xp.sqrt(t)
    d1 = -(xp.log(fs / h) + (b + (gamma - 0.5) * v2) * t) / vsqt
    d2 = d1 - 2.0 * xp.log(i / fs) / vsqt
    lam = -r + gamma * b + 0.5 * gamma * (gamma - 1.0) * v2
    kappa = 2.0 * b / v2 + (2.0 * gamma - 1.0)
    return (xp.exp(lam * t) * fs ** gamma
            * (_norm_cdf(d1) - (i / fs) ** kappa * _norm_cdf(d2)))


def _psi(fs, t2, gamma, h, i2, i1, t1, r, b, v):
    v2 = v * v
    vsqt1, vsqt2 = v * xp.sqrt(t1), v * xp.sqrt(t2)
    bg1 = (b + (gamma - 0.5) * v2) * t1
    bg2 = (b + (gamma - 0.5) * v2) * t2
    d1 = (xp.log(fs / i1) + bg1) / vsqt1
    d3 = (xp.log(fs / i1) - bg1) / vsqt1
    d2 = (xp.log(i2 * i2 / (fs * i1)) + bg1) / vsqt1
    d4 = (xp.log(i2 * i2 / (fs * i1)) - bg1) / vsqt1
    e1 = (xp.log(fs / h) + bg2) / vsqt2
    e2 = (xp.log(i2 * i2 / (fs * h)) + bg2) / vsqt2
    e3 = (xp.log(i1 * i1 / (fs * h)) + bg2) / vsqt2
    e4 = (xp.log(fs * i1 * i1 / (h * i2 * i2)) + bg2) / vsqt2
    lam = -r + gamma * b + 0.5 * gamma * (gamma - 1.0) * v2
    kappa = 2.0 * b / v2 + (2.0 * gamma - 1.0)
    return (xp.exp(lam * t2) * fs ** gamma
            * (_cbnd(-d1, -e1, TAU)
               - (i2 / fs) ** kappa * _cbnd(-d2, -e2, TAU)
               - (i1 / fs) ** kappa * _cbnd(-d3, -e3, -TAU)
               + (i1 / i2) ** kappa * _cbnd(-d4, -e4, -TAU)))


def _bs2002_call(fs, x, t, r, b, v):
    v2 = v * v
    d1 = (xp.log(fs / x) + (b + 0.5 * v2) * t) / (v * xp.sqrt(t))
    d2 = d1 - v * xp.sqrt(t)
    e_value = (fs * xp.exp((b - r) * t) * _norm_cdf(d1)
               - x * xp.exp(-r * t) * _norm_cdf(d2))

    t1 = GOLDEN * t
    beta = (0.5 - b / v2) + xp.sqrt(xp.abs((b / v2 - 0.5) ** 2 + 2.0 * r / v2))
    beta_m1 = xp.where(xp.abs(beta - 1.0) < 1e-6, 1e-6, beta - 1.0)
    b_inf = (beta / beta_m1) * x
    rmb = xp.where(xp.abs(r - b) < 1e-6, 1e-6, r - b)
    b_zero = xp.maximum(x, (r / rmb) * x)

    denom = (b_inf - b_zero) * b_zero
    i1 = b_zero + (b_inf - b_zero) * (1.0 - xp.exp(-(b * t1 + 2.0 * v * xp.sqrt(t1)) * (x * x / denom)))
    i2 = b_zero + (b_inf - b_zero) * (1.0 - xp.exp(-(b * t + 2.0 * v * xp.sqrt(t)) * (x * x / denom)))
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

    american = xp.where(fs >= i2, fs - x, big)
    value = xp.where(b >= r, e_value, american)          # b>=r: never exercise early
    return xp.maximum(value, e_value)


def _price(S, K, T, r, q, sigma, is_call):
    b = r - q
    call = _bs2002_call(S, K, T, r, b, sigma)
    put = _bs2002_call(K, S, T, r - b, -b, sigma)        # put via the BS transform
    return xp.where(is_call, call, put)


def solve_iv_greeks(price, S, K, T, r, q, is_call, n_iter=30,
                    dS_rel=1e-3, dsig=1e-4, dr=1e-4, dT=1e-4):
    price, S, K, T = _in(price), _in(S), _in(K), _in(T)
    ic = xp.asarray(is_call)
    r = xp.full(price.shape, float(r), dtype=xp.float32)
    q = xp.full(price.shape, float(q), dtype=xp.float32)

    # Brenner-Subrahmanyam seed, then fixed-count Newton with bumped vega
    ebrt, ert = xp.exp(-q * T), xp.exp(-r * T)
    payoff = xp.where(ic, S * ebrt - K * ert, K * ert - S * ebrt)
    aa = math.sqrt(2.0 * math.pi) / (S * ebrt + K * ert)
    bb = price - payoff / 2.0
    sigma = xp.clip(aa * (bb + xp.sqrt(bb * bb + payoff * payoff / math.pi)) / xp.sqrt(T), 0.05, 2.0)

    base = vega = None
    for _ in range(n_iter):
        sigma = xp.clip(sigma, 1e-4, 5.0)
        base = _price(S, K, T, r, q, sigma, ic)
        vega = (_price(S, K, T, r, q, sigma + dsig, ic)
                - _price(S, K, T, r, q, sigma - dsig, ic)) / (2.0 * dsig)
        vega = xp.where(xp.abs(vega) < 1e-8, 1e-8, vega)
        sigma = sigma - (base - price) / vega
    sigma = xp.clip(sigma, 1e-4, 5.0)

    # Greeks -- base/vega reused from the final Newton iteration above; only
    # the delta/gamma/rho/theta bumps re-price at the converged sigma
    hS = dS_rel * S
    up, dn = _price(S + hS, K, T, r, q, sigma, ic), _price(S - hS, K, T, r, q, sigma, ic)
    T_dn = xp.maximum(T - dT, 1e-6)
    return _out(sigma), {
        "price": _out(base),
        "delta": _out((up - dn) / (2.0 * hS)),
        "gamma": _out((up - 2.0 * base + dn) / (hS * hS)),
        "vega": _out(vega),
        "rho": _out((_price(S, K, T, r + dr, q, sigma, ic)
                     - _price(S, K, T, r - dr, q, sigma, ic)) / (2.0 * dr)),
        "theta": _out((_price(S, K, T_dn, r, q, sigma, ic) - base) / (T - T_dn)),
    }
