"""Black-Scholes prices and greeks. Zero rates, no dividends.

All functions are vectorised over S and tau. tau is time to expiry in years;
tau <= 0 is handled by returning the terminal values.
"""

import numpy as np
from scipy.stats import norm

_EPS = 1e-12


def _d1_d2(S, K, tau, sigma):
    S = np.asarray(S, dtype=float)
    tau = np.maximum(np.asarray(tau, dtype=float), _EPS)
    vol_t = sigma * np.sqrt(tau)
    d1 = (np.log(S / K) + 0.5 * sigma**2 * tau) / vol_t
    return d1, d1 - vol_t


def price(S, K, tau, sigma, kind="call"):
    d1, d2 = _d1_d2(S, K, tau, sigma)
    if kind == "call":
        out = S * norm.cdf(d1) - K * norm.cdf(d2)
        intrinsic = np.maximum(np.asarray(S, dtype=float) - K, 0.0)
    elif kind == "put":
        out = K * norm.cdf(-d2) - S * norm.cdf(-d1)
        intrinsic = np.maximum(K - np.asarray(S, dtype=float), 0.0)
    else:
        raise ValueError(kind)
    return np.where(np.asarray(tau) <= 0, intrinsic, out)


def delta(S, K, tau, sigma, kind="call"):
    d1, _ = _d1_d2(S, K, tau, sigma)
    if kind == "call":
        out = norm.cdf(d1)
        terminal = (np.asarray(S, dtype=float) > K).astype(float)
    elif kind == "put":
        out = norm.cdf(d1) - 1.0
        terminal = -(np.asarray(S, dtype=float) < K).astype(float)
    else:
        raise ValueError(kind)
    return np.where(np.asarray(tau) <= 0, terminal, out)


def gamma(S, K, tau, sigma):
    """Same for calls and puts (follows from put-call parity)."""
    d1, _ = _d1_d2(S, K, tau, sigma)
    tau_c = np.maximum(np.asarray(tau, dtype=float), _EPS)
    out = norm.pdf(d1) / (np.asarray(S, dtype=float) * sigma * np.sqrt(tau_c))
    return np.where(np.asarray(tau) <= 0, 0.0, out)


def vega(S, K, tau, sigma):
    """Per unit of sigma, i.e. multiply by 0.01 for a one-vol-point move."""
    d1, _ = _d1_d2(S, K, tau, sigma)
    tau_c = np.maximum(np.asarray(tau, dtype=float), _EPS)
    out = np.asarray(S, dtype=float) * norm.pdf(d1) * np.sqrt(tau_c)
    return np.where(np.asarray(tau) <= 0, 0.0, out)


def theta(S, K, tau, sigma):
    """Per year. Equals -0.5 * sigma^2 * S^2 * gamma when r = 0."""
    return -0.5 * sigma**2 * np.asarray(S, dtype=float) ** 2 * gamma(S, K, tau, sigma)
