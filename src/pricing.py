"""Black-Scholes option pricing and Greeks, and an Ornstein-Uhlenbeck mean-
reversion fit for the Brent-WTI spread.

This module only computes; it never touches the network or the LLM. Every
number produced here is a plain Python float derived from a closed-form
formula or a numpy least-squares fit, so it can be passed straight into a
grounded LLM payload without violating the "LLM never computes" rule.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

_SQRT_2 = math.sqrt(2.0)
_SQRT_2PI = math.sqrt(2.0 * math.pi)


def _norm_cdf(x: float) -> float:
    """Standard normal CDF via the error function (no scipy dependency)."""
    return 0.5 * (1.0 + math.erf(x / _SQRT_2))


def _norm_pdf(x: float) -> float:
    """Standard normal PDF."""
    return math.exp(-0.5 * x * x) / _SQRT_2PI


def _d1_d2(S: float, K: float, T: float, r: float, sigma: float) -> tuple[float, float]:
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return d1, d2


@dataclass
class Greeks:
    """Black-Scholes Greeks for a single option contract."""
    price: float
    delta: float
    gamma: float
    theta: float  # per calendar day
    vega: float   # per 1.0 (100 percentage points) change in sigma
    rho: float    # per 1.0 (100 percentage points) change in r


def bs_price(S: float, K: float, T: float, r: float, sigma: float, option_type: str = "call") -> float:
    """Black-Scholes price for a European call or put.

    S: spot price, K: strike, T: years to expiry, r: risk-free rate (annual,
    decimal), sigma: implied volatility (annual, decimal).
    """
    if T <= 0 or sigma <= 0:
        intrinsic = max(0.0, S - K) if option_type == "call" else max(0.0, K - S)
        return intrinsic
    d1, d2 = _d1_d2(S, K, T, r, sigma)
    if option_type == "call":
        return S * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)
    return K * math.exp(-r * T) * _norm_cdf(-d2) - S * _norm_cdf(-d1)


def bs_greeks(S: float, K: float, T: float, r: float, sigma: float, option_type: str = "call") -> Greeks:
    """Full Greeks set for a European call or put via closed-form Black-Scholes."""
    if T <= 0 or sigma <= 0:
        price = bs_price(S, K, T, r, sigma, option_type)
        delta = 1.0 if (option_type == "call" and S > K) else (-1.0 if S < K else 0.0)
        return Greeks(price=price, delta=delta, gamma=0.0, theta=0.0, vega=0.0, rho=0.0)

    d1, d2 = _d1_d2(S, K, T, r, sigma)
    pdf_d1 = _norm_pdf(d1)
    sqrt_T = math.sqrt(T)

    price = bs_price(S, K, T, r, sigma, option_type)
    gamma = pdf_d1 / (S * sigma * sqrt_T)
    vega = S * pdf_d1 * sqrt_T / 100.0  # per 1 vol point (0.01 sigma)

    if option_type == "call":
        delta = _norm_cdf(d1)
        theta_annual = (
            -(S * pdf_d1 * sigma) / (2 * sqrt_T) - r * K * math.exp(-r * T) * _norm_cdf(d2)
        )
        rho = K * T * math.exp(-r * T) * _norm_cdf(d2) / 100.0
    else:
        delta = _norm_cdf(d1) - 1.0
        theta_annual = (
            -(S * pdf_d1 * sigma) / (2 * sqrt_T) + r * K * math.exp(-r * T) * _norm_cdf(-d2)
        )
        rho = -K * T * math.exp(-r * T) * _norm_cdf(-d2) / 100.0

    return Greeks(
        price=price, delta=delta, gamma=gamma,
        theta=theta_annual / 365.0, vega=vega, rho=rho,
    )


def implied_vol(
    market_price: float, S: float, K: float, T: float, r: float,
    option_type: str = "call", low: float = 1e-4, high: float = 5.0, tol: float = 1e-6, max_iter: int = 100,
) -> float | None:
    """Solve for implied volatility by bisection on the Black-Scholes price.

    Returns None if the market price is outside the achievable range (e.g.
    below intrinsic value or the bracket doesn't converge).
    """
    if T <= 0 or market_price <= 0:
        return None
    f_low = bs_price(S, K, T, r, low, option_type) - market_price
    f_high = bs_price(S, K, T, r, high, option_type) - market_price
    if f_low * f_high > 0:
        return None
    for _ in range(max_iter):
        mid = (low + high) / 2.0
        f_mid = bs_price(S, K, T, r, mid, option_type) - market_price
        if abs(f_mid) < tol:
            return mid
        if f_low * f_mid < 0:
            high = mid
        else:
            low, f_low = mid, f_mid
    return (low + high) / 2.0


@dataclass
class OUFit:
    """Ornstein-Uhlenbeck mean-reversion fit for a spread series.

    Model: dX_t = kappa * (mu - X_t) dt + sigma dW_t, discretized as the
    AR(1) regression X_{t+1} = a + b * X_t + e_t, fit by ordinary least
    squares. kappa = -ln(b) / dt; the long-run mean mu = a / (1 - b); the
    half-life is the time for a deviation from mu to decay by half.
    """
    kappa: float          # mean-reversion speed, per period
    mu: float             # long-run mean
    half_life: float      # in periods (days, if fit on daily data)
    r_squared: float
    n_obs: int


def fit_ou(series: pd.Series, dt: float = 1.0) -> OUFit | None:
    """Fit an OU process to a spread series via AR(1) least squares.

    `series` should be a clean (no-NaN) time-ordered sequence of spread
    values. `dt` is the time step between observations in the same units the
    caller wants the half-life reported in (default 1.0 = one period).
    Returns None if there are fewer than 10 observations, the fit is
    degenerate (b outside (0, 1), i.e. not mean-reverting), or the sample
    doesn't span enough estimated half-lives to trust the estimate. That last
    guard matters: OLS on a true random walk still yields b slightly below 1
    in any finite sample (a well-known finite-sample bias), which would
    otherwise read as spurious mean reversion. Requiring at least 5 half-lives
    of history rejects those spurious fits while still accepting genuinely
    mean-reverting series, mirroring the same "not enough history to trust
    this number" caution CLAUDE.md applies to the z-score window.
    """
    x = series.dropna().to_numpy(dtype=float)
    if len(x) < 10:
        return None
    x_t, x_next = x[:-1], x[1:]
    A = np.column_stack([np.ones_like(x_t), x_t])
    coeffs, residuals, _, _ = np.linalg.lstsq(A, x_next, rcond=None)
    a, b = coeffs
    if not (0.0 < b < 1.0):
        return None  # not mean-reverting (b>=1 is a random walk or explosive)

    kappa = -math.log(b) / dt
    mu = a / (1.0 - b)
    half_life = math.log(2.0) / kappa
    if len(x) * dt < 5 * half_life:
        return None  # sample too short to distinguish from a random walk

    fitted = A @ coeffs
    ss_res = float(np.sum((x_next - fitted) ** 2))
    ss_tot = float(np.sum((x_next - x_next.mean()) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return OUFit(kappa=kappa, mu=mu, half_life=half_life, r_squared=r_squared, n_obs=len(x))
