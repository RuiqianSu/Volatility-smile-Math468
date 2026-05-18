"""
visualizer.py
-------------
Plotting functions for implied volatility smiles and surfaces.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm


plt.rcParams.update({
    "figure.facecolor": "#0f0f0f",
    "axes.facecolor": "#0f0f0f",
    "axes.edgecolor": "#333333",
    "axes.labelcolor": "#cccccc",
    "xtick.color": "#888888",
    "ytick.color": "#888888",
    "text.color": "#cccccc",
    "grid.color": "#222222",
    "grid.linestyle": "--",
    "grid.alpha": 0.6,
    "font.family": "monospace",
})

ACCENT = "#00d4ff"
COLORS = ["#00d4ff", "#ff6b6b", "#a8ff78", "#f7971e", "#c471ed"]
X_AXIS_LABELS = {
    "moneyness": "Moneyness K/S",
    "log_moneyness": "Log-Moneyness ln(K/S)",
    "strike": "Strike Price",
}


def _prepare_save_path(save_path: str | None) -> str | None:
    if save_path is None:
        return None

    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return str(path)


def _validate_x_axis(x_axis: str) -> str:
    if x_axis not in X_AXIS_LABELS:
        valid = ", ".join(sorted(X_AXIS_LABELS))
        raise ValueError(f"Unsupported x_axis '{x_axis}'. Choose from: {valid}")
    return X_AXIS_LABELS[x_axis]


def _ensure_non_empty(df: pd.DataFrame, label: str) -> pd.DataFrame:
    if df.empty:
        raise ValueError(f"{label} is empty. No data available to plot.")
    return df


def _marker_sizes(open_interest: pd.Series) -> np.ndarray:
    if open_interest.empty:
        return np.array([], dtype=float)

    max_open_interest = float(open_interest.max())
    if not np.isfinite(max_open_interest) or max_open_interest <= 0:
        return np.full(len(open_interest), 30.0)

    sizes = open_interest.to_numpy(dtype=float) / max_open_interest * 80 + 10
    return np.clip(sizes, 10, 90)


def plot_smile(
    df: pd.DataFrame,
    ticker: str,
    expiry: str,
    S: float,
    x_axis: str = "strike",
    save_path: str | None = None,
):
    """
    Plot the implied volatility smile for a single expiry.

    Parameters
    ----------
    df       : output of iv_solver.compute_iv_surface()
    ticker   : e.g. "SPY"
    expiry   : expiry date string
    S        : spot price
    x_axis   : "strike", "moneyness", or "log_moneyness"
    save_path: if provided, saves the figure to this path
    """
    df = _ensure_non_empty(df, "IV surface")
    x_label = _validate_x_axis(x_axis)
    save_path = _prepare_save_path(save_path)

    fig, ax = plt.subplots(figsize=(10, 5))

    x = df[x_axis]
    y = df["iv"] * 100

    ax.scatter(
        x,
        y,
        s=_marker_sizes(df["openInterest"]),
        color=ACCENT,
        alpha=0.5,
        zorder=2,
        label="Options data",
    )

    idx = x.argsort()
    ax.plot(x.values[idx], y.values[idx], color=ACCENT, linewidth=1.5, alpha=0.9, zorder=3)

    atm_x = {"moneyness": 1.0, "log_moneyness": 0.0, "strike": S}[x_axis]
    ax.axvline(atm_x, color="#ffffff", linewidth=0.8, linestyle=":", alpha=0.5, label="ATM")

    ax.set_xlabel(x_label)
    ax.set_ylabel("Implied Volatility (%)")
    ax.set_title(f"{ticker} | Volatility Smile | Expiry: {expiry}\nSpot S0 = {S:.2f}", pad=14)
    ax.legend(framealpha=0.1)
    ax.grid(True)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_smile_comparison(
    dfs: list[pd.DataFrame],
    labels: list[str],
    ticker: str,
    S: float | None = None,
    x_axis: str = "moneyness",
    save_path: str | None = None,
    title: str | None = None,
):
    """
    Overlay volatility smiles for multiple expiries on the same plot.
    Use an explicit x_axis="strike" when you want the raw option-market strike axis.
    """
    x_label = _validate_x_axis(x_axis)
    save_path = _prepare_save_path(save_path)

    pairs = [(df, label) for df, label in zip(dfs, labels) if not df.empty]
    if not pairs:
        raise ValueError("No non-empty IV surfaces are available to compare.")

    if x_axis == "strike" and S is None:
        raise ValueError("S must be provided when x_axis='strike'.")

    fig, ax = plt.subplots(figsize=(11, 5))

    for i, (df, label) in enumerate(pairs):
        color = COLORS[i % len(COLORS)]
        x = df[x_axis]
        y = df["iv"] * 100
        idx = x.argsort()
        ax.plot(x.values[idx], y.values[idx], color=color, linewidth=2, label=label)
        ax.scatter(x, y, s=15, color=color, alpha=0.4, zorder=2)

    atm_x = {"moneyness": 1.0, "log_moneyness": 0.0, "strike": S}[x_axis]
    ax.axvline(atm_x, color="#ffffff", linewidth=0.8, linestyle=":", alpha=0.5, label="ATM")

    ax.set_xlabel(x_label)
    ax.set_ylabel("Implied Volatility (%)")
    if title is not None:
        plot_title = title
    elif S is not None:
        plot_title = f"{ticker} | Volatility Smile by Maturity | Spot S0 = {S:.2f}"
    else:
        plot_title = f"{ticker} | Volatility Smile by Maturity"
    ax.set_title(plot_title, pad=14)
    ax.legend(framealpha=0.1)
    ax.grid(True)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_return_diagnostics(returns: pd.Series, ticker: str, save_path: str | None = None):
    """
    Statistical diagnostics of log returns:
    - Histogram vs normal
    - QQ plot
    """
    returns = returns.dropna()
    if returns.empty:
        raise ValueError("Returns series is empty. No diagnostics can be plotted.")

    save_path = _prepare_save_path(save_path)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    mu, sigma = returns.mean(), returns.std()

    ax.hist(
        returns,
        bins=80,
        density=True,
        color=ACCENT,
        alpha=0.35,
        edgecolor="none",
        label="Empirical returns",
    )

    if sigma > 0:
        x = np.linspace(returns.min(), returns.max(), 300)
        ax.plot(
            x,
            norm.pdf(x, mu, sigma),
            color="#ff6b6b",
            linewidth=2,
            label=f"Normal  mu={mu:.4f}  sigma={sigma:.4f}",
        )

    skew = float(returns.skew())
    kurt = float(returns.kurt())
    ax.set_title(f"Return Distribution\nskewness={skew:.3f}   excess kurtosis={kurt:.3f}")
    ax.set_xlabel("Log Return")
    ax.set_ylabel("Density")
    ax.legend(framealpha=0.1)
    ax.grid(True)

    ax = axes[1]
    sorted_r = np.sort(returns)
    n = len(sorted_r)
    quantiles = norm.ppf(np.linspace(0.5 / n, 1 - 0.5 / n, n), loc=mu, scale=max(sigma, 1e-12))

    ax.scatter(quantiles, sorted_r, s=6, color=ACCENT, alpha=0.5)
    ref_line = np.linspace(quantiles[0], quantiles[-1], 100)
    ax.plot(ref_line, ref_line, color="#ff6b6b", linewidth=1.5, label="Normal reference")

    ax.set_title("QQ Plot vs Normal\n(deviations at tails -> fat tails)")
    ax.set_xlabel("Normal quantiles")
    ax.set_ylabel("Empirical quantiles")
    ax.legend(framealpha=0.1)
    ax.grid(True)

    fig.suptitle(f"{ticker} | Return Diagnostics", fontsize=13, y=1.02)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_vol_surface_3d(
    dfs: list[pd.DataFrame],
    maturities_years: list[float],
    ticker: str,
    save_path: str | None = None,
):
    """
    3D plot of the implied volatility surface: log-moneyness x maturity x IV.
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    if len(dfs) != len(maturities_years):
        raise ValueError("dfs and maturities_years must have the same length.")

    pairs = [(df, T) for df, T in zip(dfs, maturities_years) if not df.empty]
    if not pairs:
        raise ValueError("No non-empty IV surfaces are available for the 3D plot.")

    save_path = _prepare_save_path(save_path)
    maturity_scale = max(max(maturities_years), 1e-6)

    fig = plt.figure(figsize=(12, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("#0f0f0f")
    fig.patch.set_facecolor("#0f0f0f")

    cmap = cm.cool

    for df, T in pairs:
        x = df["log_moneyness"].values
        z = df["iv"].values * 100
        y = np.full_like(x, T)

        idx = np.argsort(x)
        color = cmap(T / maturity_scale)
        ax.plot(x[idx], y[idx], z[idx], color=color, linewidth=1.5)
        ax.scatter(x, y, z, s=8, color=color, alpha=0.4)

    ax.set_xlabel("Log-Moneyness", labelpad=8)
    ax.set_ylabel("Maturity (yrs)", labelpad=8)
    ax.set_zlabel("IV (%)", labelpad=8)
    ax.set_title(f"{ticker} | Implied Volatility Surface", pad=16)
    ax.tick_params(colors="#888888")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
