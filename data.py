"""
data.py
-------
Parameters estimator choice
Fetch options chain data from Yahoo Finance.
"""

from __future__ import annotations

import yfinance as yf
import pandas as pd



def get_risk_free_rate() -> float:
    """
    Approximate risk-free rate using 3-month T-bill yield (^IRX).
    the U.S.3-month Treasury bill rate, this rate indicates the yield for a 3-month investment
    Falls back to 0.05 if unavailable.
    """
    try:
        tbill = yf.Ticker("^IRX")
        rate = tbill.history(period="1d")["Close"].iloc[-1] / 100
        return float(rate)
    except Exception:
        return 0.05


def get_dividend_yield(ticker: str) -> float:
    """
    Approximate the continuous dividend yield used in Black-Scholes.
    Falls back to 0.0 if unavailable.
    """
    try:
        dividend_yield = yf.Ticker(ticker).info.get("dividendYield")
        if dividend_yield is None:
            return 0.0

        q = float(dividend_yield)
        if q > 1.0:
            q /= 100.0
        return max(q, 0.0)
    except Exception:
        return 0.0


def get_options_chain(ticker: str, expiry: str | None = None) -> tuple[pd.DataFrame, float, float, str]:
    """
    Fetch call options chain for a given ticker and expiry date.

    Parameters
    ----------
    ticker  : str   e.g. "SPY", "AAPL"
    expiry  : str   e.g. "2025-06-20". If None, uses the nearest expiry.

    Returns
    -------
    calls   : pd.DataFrame  cleaned call options with columns:
                            strike, lastPrice, bid, ask, mid, volume, openInterest
    S0      : float         current spot price
    r       : float         risk-free rate
    expiry  : str           expiry date actually used
    """
    tk = yf.Ticker(ticker)

    # current spot price
    hist = tk.history(period="1d")
    if hist.empty:
        raise ValueError(f"Could not fetch spot price for {ticker}")
    S0 = float(hist["Close"].iloc[-1])

    # available expiry dates
    expiries = tk.options
    if not expiries:
        raise ValueError(f"No options data available for {ticker}")

    if expiry is None:
        expiry = expiries[0]
    elif expiry not in expiries:
        raise ValueError(f"Expiry {expiry} not available. Choose from: {list(expiries)}")

    chain = tk.option_chain(expiry)
    calls = chain.calls.copy()

    # compute mid price; prefer mid over lastPrice for accuracy
    # bid price: the highest price a buyer is willing to pay for the option
    # ask price: the lowest price a seller is willing to accept for the option
    calls["mid"] = (calls["bid"] + calls["ask"]) / 2

    # filter: remove zero-volume, zero-bid, obviously stale quotes
    calls = calls[
        (calls["bid"] > 0) &
        (calls["ask"] > 0) &
        (calls["mid"] > 0.01) &
        (calls["strike"] > 0)
    ].copy()

    # keep only relevant columns
    calls = calls[["strike", "lastPrice", "bid", "ask", "mid", "volume", "openInterest"]].reset_index(drop=True)

    r = get_risk_free_rate()

    return calls, S0, r, expiry


def list_expiries(ticker: str) -> list[str]:
    """Return all available option expiry dates for a ticker."""
    tk = yf.Ticker(ticker)
    return list(tk.options)
