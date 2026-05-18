"""
iv_solver.py
------------
Black-Scholes formula and implied volatility solver.
Implemented from scratch using Brent's method with no external pricing libraries.
"""

from __future__ import annotations

from datetime import datetime, time

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm


SECONDS_PER_YEAR = 365.25 * 24 * 60 * 60


def black_scholes_call(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
) -> float:
    """
    Black-Scholes price for a European call option.
    We use the BSM formula as an approximation;
    the error introduced is small given SPY's low dividend yield
    Parameters
    ----------
    S     : spot price
    K     : strike price
    T     : time to expiry in years
    r     : risk-free rate (annualized)
    sigma : volatility (annualized)
    q     : continuous dividend yield (annualized)

    Returns
    -------
    call price : float
    """
    if T <= 0:
        return max(S - K, 0.0)
    if sigma <= 0:
        return max(S * np.exp(-q * T) - K * np.exp(-r * T), 0.0)

    d_plus = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d_minus = d_plus - sigma * np.sqrt(T)

    return S * np.exp(-q * T) * norm.cdf(d_plus) - K * np.exp(-r * T) * norm.cdf(d_minus)


def implied_volatility(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    q: float = 0.0,
    tol: float = 1e-6,
    max_iter: int = 200,
) -> float:
    """
    Compute implied volatility by numerically inverting Black-Scholes
    using Brent's method.

    Parameters
    ----------
    market_price : observed call option price (use mid price)
    S            : spot price
    K            : strike price
    T            : time to expiry in years
    r            : risk-free rate
    q            : continuous dividend yield
    tol          : convergence tolerance
    max_iter     : maximum iterations

    Returns
    -------
    implied volatility : float, or np.nan if no solution is found
    """
    if T <= 0:
        return np.nan

    intrinsic = max(S * np.exp(-q * T) - K * np.exp(-r * T), 0.0)
    if market_price <= intrinsic:
        return np.nan

    if market_price >= S * np.exp(-q * T):
        return np.nan

    objective = lambda sigma: black_scholes_call(S, K, T, r, sigma, q=q) - market_price

    try:
        #Brent's method is a hybrid root-finding algorithm used in numerical analysis to solve equations of the form $f(x) = 0$.
        #It combines the advantages of the bisection method, the secant method, and inverse quadratic interpolation.
        iv = brentq(objective, 1e-4, 10.0, xtol=tol, maxiter=max_iter)
        return iv
    except ValueError:
        return np.nan


def time_to_expiry_years(expiry: str, valuation_dt: datetime | None = None) -> float:
    """Convert an expiry date string into a positive year fraction."""
    expiry_dt = datetime.combine(datetime.strptime(expiry, "%Y-%m-%d").date(), time(hour=16))
    now = valuation_dt or datetime.now()
    return max((expiry_dt - now).total_seconds() / SECONDS_PER_YEAR, 1e-6)


def compute_iv_surface(
    calls: pd.DataFrame,
    S: float,
    r: float,
    expiry: str,
    q: float = 0.0,
) -> pd.DataFrame:
    """
    Compute implied volatility for each option in the calls DataFrame.

    Parameters
    ----------
    calls  : DataFrame from data.get_options_chain()
    S      : spot price
    r      : risk-free rate
    expiry : expiry date string "YYYY-MM-DD"
    q      : continuous dividend yield

    Returns
    -------
    DataFrame with additional columns: T, iv, moneyness, log_moneyness
    """
    required_columns = {"mid", "strike"}
    missing = required_columns.difference(calls.columns)
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise ValueError(f"Missing required option columns: {missing_list}")

    df = calls.copy()
    if df.empty:
        for column in ["T", "iv", "moneyness", "log_moneyness"]:
            df[column] = pd.Series(dtype=float)
        return df

    T = time_to_expiry_years(expiry)
    df["T"] = T

    df["iv"] = df.apply(
        lambda row: implied_volatility(row["mid"], S, row["strike"], T, r, q=q),
        axis=1,
    )

    df["moneyness"] = df["strike"] / S
    df["log_moneyness"] = np.log(df["strike"] / S)

    df = df.dropna(subset=["iv"])
    df = df[(df["iv"] > 0.01) & (df["iv"] < 5.0)]

    return df.reset_index(drop=True)
