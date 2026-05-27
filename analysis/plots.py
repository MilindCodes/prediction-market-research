from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import config


FIGURE_DIR = config.DATA_DIR / "processed" / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

DPI = 300
sns.set_theme(style="whitegrid", font_scale=1.1)


def plot_implied_vol_series(ticker: str,
                            data_dir: Path | None = None) -> Path:
    """Plot price and sigma_implied over time with FOMC windows shaded.

    Parameters
    ----------
    ticker : str
        Contract identifier.
    data_dir : Path
        Root data directory (default from config).

    Returns
    -------
    Path
        Path to saved figure.
    """
    data_dir = data_dir or config.DATA_DIR
    iv_path = data_dir / "processed" / "implied_vol" / f"{ticker}.parquet"
    df = pd.read_parquet(iv_path)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    ax1.plot(df["timestamp"], df["price"], color="steelblue", linewidth=0.8)
    ax1.set_ylabel("Price (probability)")
    ax1.set_title(f"{ticker} — Price & Implied Volatility")

    ax2.plot(df["timestamp"], df["sigma_implied"], color="darkorange",
             linewidth=0.8)
    ax2.set_ylabel("σ_implied")
    ax2.set_xlabel("Time (UTC)")

    fomc_rows = df[df["is_jump_window"]]
    if not fomc_rows.empty:
        for _, row in fomc_rows.iterrows():
            ts = row["timestamp"]
            ax1.axvspan(ts - pd.Timedelta(minutes=30),
                        ts + pd.Timedelta(minutes=30),
                        alpha=0.15, color="red", label="_nolegend_")
            ax2.axvspan(ts - pd.Timedelta(minutes=30),
                        ts + pd.Timedelta(minutes=30),
                        alpha=0.15, color="red", label="_nolegend_")

    fig.tight_layout()
    out = FIGURE_DIR / f"iv_series_{ticker}.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")
    return out


def plot_heston_fit(ticker: str,
                    data_dir: Path | None = None) -> Path:
    """Plot observed vs Heston-fitted variance path.

    Parameters
    ----------
    ticker : str
        Contract identifier.
    data_dir : Path
        Root data directory.

    Returns
    -------
    Path
        Path to saved figure.
    """
    data_dir = data_dir or config.DATA_DIR

    iv_path = data_dir / "processed" / "implied_vol" / f"{ticker}.parquet"
    iv_df = pd.read_parquet(iv_path)
    sigma_obs = iv_df["sigma_implied"].dropna().values
    var_obs = sigma_obs ** 2
    timestamps = iv_df.loc[iv_df["sigma_implied"].notna(), "timestamp"].values

    param_path = data_dir / "processed" / "heston_params" / f"{ticker}.json"
    with open(param_path) as f:
        params = json.load(f)

    from src.models.heston import HestonCalibrator
    p = np.array([params["kappa"], params["theta"], params["xi"],
                  params["rho"], params["v0"]])
    dt = 1 / 6552
    var_model = HestonCalibrator._simulate_variance_path(p, len(var_obs), dt)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(timestamps, var_obs, color="steelblue", linewidth=0.8,
            alpha=0.7, label="Observed σ²")
    ax.plot(timestamps, var_model, color="darkorange", linewidth=1.2,
            label="Heston fit")
    ax.set_ylabel("Variance (σ²)")
    ax.set_xlabel("Time (UTC)")
    ax.set_title(f"{ticker} — Heston Model Fit")
    ax.legend()

    fig.tight_layout()
    out = FIGURE_DIR / f"heston_fit_{ticker}.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")
    return out


def plot_parameter_distribution(param_name: str,
                                data_dir: Path | None = None) -> Path:
    """Histogram of a Heston parameter across all calibrated contracts.

    Parameters
    ----------
    param_name : str
        One of: kappa, theta, xi, rho, v0.
    data_dir : Path
        Root data directory.

    Returns
    -------
    Path
        Path to saved figure.
    """
    data_dir = data_dir or config.DATA_DIR
    param_dir = data_dir / "processed" / "heston_params"

    values: list[float] = []
    for path in param_dir.glob("*.json"):
        with open(path) as f:
            result = json.load(f)
        val = result.get(param_name)
        if val is not None and np.isfinite(val):
            values.append(val)

    if not values:
        print(f"  No valid values for {param_name}")
        return FIGURE_DIR / f"param_dist_{param_name}.png"

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(values, bins=30, color="steelblue", edgecolor="white", alpha=0.85)
    ax.axvline(np.median(values), color="darkorange", linestyle="--",
               label=f"Median: {np.median(values):.4f}")
    ax.set_xlabel(param_name)
    ax.set_ylabel("Count")
    ax.set_title(f"Distribution of Heston {param_name} (n={len(values)})")
    ax.legend()

    fig.tight_layout()
    out = FIGURE_DIR / f"param_dist_{param_name}.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")
    return out


def plot_model_comparison_table(data_dir: Path | None = None) -> Path:
    """Heatmap of MSE by model (Heston/Bates) x market type (fed/political).

    Parameters
    ----------
    data_dir : Path
        Root data directory.

    Returns
    -------
    Path
        Path to saved figure.
    """
    data_dir = data_dir or config.DATA_DIR
    summary_path = data_dir / "processed" / "model_comparison_summary.parquet"
    df = pd.read_parquet(summary_path)

    kalshi_cat = data_dir / "raw" / "kalshi" / "catalog_filtered.parquet"
    poly_cat = data_dir / "raw" / "polymarket" / "catalog_filtered.parquet"

    type_map: dict[str, str] = {}
    for cat_path in [kalshi_cat, poly_cat]:
        if cat_path.exists():
            cat = pd.read_parquet(cat_path)
            id_col = "ticker" if "ticker" in cat.columns else "condition_id"
            for _, row in cat.iterrows():
                cid = row.get(id_col, "")
                if row.get("is_fed", False):
                    type_map[cid] = "Fed/FOMC"
                elif row.get("is_political", False):
                    type_map[cid] = "Political"

    df["market_type"] = df["ticker"].map(type_map).fillna("Other")

    pivot_data: dict[str, dict[str, float]] = {}
    for mt in df["market_type"].unique():
        subset = df[df["market_type"] == mt]
        pivot_data[mt] = {
            "Heston MSE": subset["mse_heston"].mean(),
            "Bates MSE": subset["mse_bates"].mean(),
        }

    heatmap_df = pd.DataFrame(pivot_data).T
    if heatmap_df.empty:
        print("  No data for heatmap")
        return FIGURE_DIR / "model_comparison_heatmap.png"

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(heatmap_df, annot=True, fmt=".6f", cmap="YlOrRd",
                linewidths=0.5, ax=ax)
    ax.set_title("Mean MSE: Model × Market Type")
    ax.set_ylabel("Market Type")

    fig.tight_layout()
    out = FIGURE_DIR / "model_comparison_heatmap.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")
    return out
