# vol-smile-analyzer

Implied volatility smile analysis on real options data.
Built from scratch with no external pricing libraries.

## What it does

1. Fetches real options data via Yahoo Finance.
2. Computes implied volatility by numerically inverting the Black-Scholes formula with Brent's method.
3. Visualizes the volatility smile across strikes and maturities.
4. Runs statistical diagnostics on historical returns to explain why the smile exists.

## The core idea

Black-Scholes assumes stock returns are lognormal with constant volatility.
If that were true, all options on the same stock with the same expiry would share the same implied volatility.

In reality, implied volatility varies with strike. That curve is the volatility smile.
Its shape reflects what Black-Scholes gets wrong: real returns are skewed and fat-tailed.

## Project structure

```text
volatility smiles/
|-- data.py
|-- iv_solver.py
|-- visualizer.py
|-- demo.ipynb
|-- requirements.txt
`-- results/
    `-- figures/
```

## Quickstart

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook demo.ipynb
```

## Dependency note

The pinned `requirements.txt` keeps NumPy below 2.0 because the current SciPy,
Matplotlib, and Pandas stack used by this project is not compatible with NumPy 2.x.

## Key results

- Volatility smile is typically downward sloping for broad index ETFs such as SPY.
- The smile often flattens at longer maturities.
- Empirical returns usually show negative skewness and positive excess kurtosis.
- Jarque-Bera strongly rejects normality for equity index returns.

These statistical facts help explain the observed smile shape.

## Tech stack

`Python`, `yfinance`, `scipy`, `numpy`, `matplotlib`, `pandas`
