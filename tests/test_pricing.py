import math

import numpy as np
import pandas as pd

from src import pricing


def test_bs_price_call_put_parity():
    S, K, T, r, sigma = 80.0, 80.0, 0.5, 0.05, 0.3
    call = pricing.bs_price(S, K, T, r, sigma, "call")
    put = pricing.bs_price(S, K, T, r, sigma, "put")
    # Put-call parity: C - P = S - K * exp(-rT)
    assert abs((call - put) - (S - K * np.exp(-r * T))) < 1e-8


def test_bs_price_known_reference_value():
    # Standard textbook example: S=100, K=100, T=1, r=0.05, sigma=0.2 -> call ~10.4506
    price = pricing.bs_price(100.0, 100.0, 1.0, 0.05, 0.2, "call")
    assert abs(price - 10.4506) < 1e-3


def test_bs_greeks_call_delta_in_range():
    g = pricing.bs_greeks(80.0, 80.0, 0.5, 0.05, 0.3, "call")
    assert 0.0 < g.delta < 1.0
    assert g.gamma > 0.0
    assert g.vega > 0.0


def test_bs_greeks_put_delta_in_range():
    g = pricing.bs_greeks(80.0, 80.0, 0.5, 0.05, 0.3, "put")
    assert -1.0 < g.delta < 0.0
    assert g.gamma > 0.0


def test_implied_vol_round_trip():
    S, K, T, r, sigma = 80.0, 78.0, 0.25, 0.05, 0.35
    price = pricing.bs_price(S, K, T, r, sigma, "call")
    iv = pricing.implied_vol(price, S, K, T, r, "call")
    assert iv is not None
    assert abs(iv - sigma) < 1e-4


def test_implied_vol_returns_none_for_price_below_intrinsic():
    iv = pricing.implied_vol(0.0001, 100.0, 80.0, 0.5, 0.05, "call")
    assert iv is None


def test_fit_ou_recovers_known_kappa():
    rng = np.random.default_rng(42)
    n, dt = 2000, 1.0
    kappa_true, mu_true, sigma_ou = 0.05, 5.0, 0.5
    x = np.zeros(n)
    x[0] = mu_true
    for i in range(1, n):
        x[i] = (
            x[i - 1] + kappa_true * (mu_true - x[i - 1]) * dt
            + sigma_ou * math.sqrt(dt) * rng.standard_normal()
        )
    fit = pricing.fit_ou(pd.Series(x), dt=dt)
    assert fit is not None
    assert abs(fit.kappa - kappa_true) < 0.02
    # mu = a / (1 - b) divides by a small number when b is close to 1 (slow
    # reversion), so its sampling error is much larger than kappa's even
    # though kappa itself is recovered tightly. A wide but bounded tolerance.
    assert abs(fit.mu - mu_true) < 2.0


def test_fit_ou_returns_none_on_random_walk():
    rng = np.random.default_rng(1)
    x = np.cumsum(rng.standard_normal(500))  # pure random walk, b ~ 1
    fit = pricing.fit_ou(pd.Series(x))
    assert fit is None


def test_fit_ou_returns_none_on_too_few_observations():
    assert pricing.fit_ou(pd.Series([1.0, 2.0, 3.0])) is None
