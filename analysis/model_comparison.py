from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import config


class ModelComparison:
    """Compare Heston and Bates calibration results across all contracts.

    Computes AIC, BIC, MSE, QLIKE loss, and likelihood ratio statistics,
    then splits by platform, market type, and near-resolution windows.

    Parameters
    ----------
    data_dir : Path
        Root data directory (default from config).
    """

    BS_K = 1        # Black-Scholes: one free parameter (constant sigma)
    HESTON_K = 5
    BATES_K = 8
    VAR_FLOOR = 1e-4  # σ²_floor = 0.01² — prevents QLIKE ratio blow-up near σ̂²→0

    def __init__(self, data_dir: Path | None = None):
        self.data_dir = data_dir or config.DATA_DIR
        self.bs_dir = self.data_dir / "processed" / "bs_params"
        self.heston_dir = self.data_dir / "processed" / "heston_params"
        self.bates_dir = self.data_dir / "processed" / "bates_params"
        self.iv_dir = self.data_dir / "processed" / "implied_vol"

    def run(self) -> pd.DataFrame:
        """Load all results, compute comparison metrics, and save summary.

        Returns
        -------
        pd.DataFrame
            One row per contract with all comparison metrics.
        """
        bs_results = self._load_results(self.bs_dir)
        heston_results = self._load_results(self.heston_dir)
        bates_results = self._load_results(self.bates_dir)

        tickers = set(bs_results.keys()) & set(heston_results.keys()) & set(bates_results.keys())
        if not tickers:
            print("No contracts with all three models calibrated.")
            print("Run backtest-bs, backtest-heston, and backtest-bates first.")
            return pd.DataFrame()

        records: list[dict] = []
        for ticker in sorted(tickers):
            bs = bs_results[ticker]
            h = heston_results[ticker]
            b = bates_results[ticker]
            row = self._compare_contract(ticker, bs, h, b)
            records.append(row)

        summary = pd.DataFrame(records)
        out_path = self.data_dir / "processed" / "model_comparison_summary.parquet"
        summary.to_parquet(out_path, index=False)
        print(f"Model comparison summary saved to {out_path}")
        print(f"  Contracts compared: {len(summary)}")

        self._print_split_summary(summary)
        return summary

    def _compare_contract(self, ticker: str, bs: dict, h: dict,
                          b: dict) -> dict:
        """Compute all comparison metrics for one contract.

        Parameters
        ----------
        ticker : str
            Contract identifier.
        h : dict
            Heston calibration result.
        b : dict
            Bates calibration result.

        Returns
        -------
        dict
            Row with all metrics for the summary DataFrame.
        """
        n = h.get("n_obs", 0)
        ll_h = h.get("log_likelihood", np.nan)
        ll_b = b.get("log_likelihood", np.nan)

        aic_h = self._aic(ll_h, self.HESTON_K)
        aic_b = self._aic(ll_b, self.BATES_K)
        bic_h = self._bic(ll_h, self.HESTON_K, n)
        bic_b = self._bic(ll_b, self.BATES_K, n)

        qlike_h = self._qlike(ticker, "heston")
        qlike_b = self._qlike(ticker, "bates")
        qlike_bs = self._qlike_bs(ticker, bs)

        lr_stat = b.get("lr_statistic", np.nan)
        p_value = b.get("p_value_jump_significance", np.nan)

        iv_path = self.iv_dir / f"{ticker}.parquet"
        near_resolution_frac = np.nan
        if iv_path.exists():
            iv_df = pd.read_parquet(iv_path)
            if "T" in iv_df.columns and len(iv_df) > 0:
                near_mask = iv_df["T"] <= (5 / 365.25)
                near_resolution_frac = float(near_mask.mean())

        return {
            "ticker": ticker,
            "platform": h.get("platform", "unknown"),
            "n_obs": n,
            # Black-Scholes Cash-or-Nothing (k=1 — constant volatility baseline)
            "sigma_bs": bs.get("sigma_bs", np.nan),
            "mse_bs": bs.get("mse", np.nan),
            "ll_bs": bs.get("log_likelihood", np.nan),
            "aic_bs": self._aic(bs.get("log_likelihood", np.nan), self.BS_K),
            "bic_bs": self._bic(bs.get("log_likelihood", np.nan), self.BS_K, n),
            "qlike_median_bs": qlike_bs["median"],
            "qlike_mean_bs": qlike_bs["mean"],
            # Heston (k=5)
            "mse_heston": h.get("mse", np.nan),
            "ll_heston": ll_h,
            "aic_heston": aic_h,
            "bic_heston": bic_h,
            "qlike_median_heston": qlike_h["median"],
            "qlike_mean_heston": qlike_h["mean"],
            # Bates (k=8)
            "mse_bates": b.get("mse", np.nan),
            "ll_bates": ll_b,
            "aic_bates": aic_b,
            "bic_bates": bic_b,
            "qlike_median_bates": qlike_b["median"],
            "qlike_mean_bates": qlike_b["mean"],
            # Likelihood ratio test: Bates vs Heston (chi-sq, 3 df)
            "lr_statistic": lr_stat,
            "p_value_jump": p_value,
            "feller_heston": h.get("feller_satisfied", False),
            "feller_bates": b.get("feller_satisfied", False),
            "converged_heston": h.get("converged", False),
            "converged_bates": b.get("converged", False),
            "near_resolution_frac": near_resolution_frac,
        }

    def _qlike_bs(self, ticker: str, bs: dict) -> dict:
        """QLIKE loss for the Black-Scholes constant-variance model.

        Returns dict with 'median' (primary) and 'mean' (secondary).
        """
        nan_result = {"median": np.nan, "mean": np.nan}
        iv_path = self.iv_dir / f"{ticker}.parquet"
        if not iv_path.exists():
            return nan_result
        sigma_bs = bs.get("sigma_bs", np.nan)
        if not np.isfinite(sigma_bs) or sigma_bs <= 0:
            return nan_result
        iv_df = pd.read_parquet(iv_path)
        if "near_boundary" in iv_df.columns:
            iv_df = iv_df[~iv_df["near_boundary"]]
        sigma_obs = iv_df["sigma_implied"].dropna().values
        if len(sigma_obs) == 0:
            return nan_result
        var_obs = np.maximum(sigma_obs ** 2, self.VAR_FLOOR)
        var_bs = max(sigma_bs ** 2, self.VAR_FLOOR)
        ratio = var_obs / var_bs
        pointwise = ratio - np.log(ratio) - 1.0
        return {"median": float(np.median(pointwise)), "mean": float(np.mean(pointwise))}

    def _qlike(self, ticker: str, model: str) -> dict:
        """Compute QLIKE loss for a fitted model.

        QLIKE = σ²/σ̂² - log(σ²/σ̂²) - 1, with a variance floor to prevent
        blow-up when σ̂²→0. Near-boundary rows (price<0.05 or >0.95) are
        excluded because Jacobian amplification makes σ_implied unreliable there.

        Returns dict with 'median' (primary) and 'mean' (secondary).
        """
        nan_result = {"median": np.nan, "mean": np.nan}
        iv_path = self.iv_dir / f"{ticker}.parquet"
        if not iv_path.exists():
            return nan_result

        param_dir = self.heston_dir if model == "heston" else self.bates_dir
        param_path = param_dir / f"{ticker}.json"
        if not param_path.exists():
            return nan_result

        iv_df = pd.read_parquet(iv_path)
        if "near_boundary" in iv_df.columns:
            iv_df = iv_df[~iv_df["near_boundary"]]
        sigma_obs = iv_df["sigma_implied"].dropna().values
        if len(sigma_obs) == 0:
            return nan_result

        with open(param_path) as f:
            params = json.load(f)

        dt = 1 / 6552
        if model == "heston":
            from src.models.heston import HestonCalibrator
            p = np.array([params["kappa"], params["theta"], params["xi"],
                          params["rho"], params["v0"]])
            var_model_full = HestonCalibrator._simulate_variance_path(p, len(iv_df), dt)
        else:
            from src.models.bates import BatesCalibrator
            p = np.array([params["kappa"], params["theta"], params["xi"],
                          params["rho"], params["v0"], params["lambda_j"],
                          params["mu_j"], params["sigma_j"]])
            var_model_full = BatesCalibrator._simulate_bates_variance_path(p, len(iv_df), dt)

        # Align model path with non-NaN, non-boundary observed rows.
        valid_idx = iv_df.index[iv_df["sigma_implied"].notna()].tolist()
        if len(valid_idx) == 0:
            return nan_result

        # var_model_full is indexed 0..len(iv_df)-1 over the filtered DataFrame.
        # Re-index to match filtered df positional indices.
        filtered_positions = np.arange(len(iv_df))[iv_df["sigma_implied"].notna().values]
        var_model = var_model_full[filtered_positions]

        var_obs = np.maximum(sigma_obs ** 2, self.VAR_FLOOR)
        var_model = np.maximum(var_model, self.VAR_FLOOR)

        ratio = var_obs / var_model
        pointwise = ratio - np.log(ratio) - 1.0
        return {"median": float(np.median(pointwise)), "mean": float(np.mean(pointwise))}

    @staticmethod
    def _aic(log_likelihood: float, k: int) -> float:
        """Akaike Information Criterion.

        Parameters
        ----------
        log_likelihood : float
            Model log-likelihood.
        k : int
            Number of estimated parameters.

        Returns
        -------
        float
            AIC value (lower is better).
        """
        if not np.isfinite(log_likelihood):
            return np.nan
        return 2 * k - 2 * log_likelihood

    @staticmethod
    def _bic(log_likelihood: float, k: int, n: int) -> float:
        """Bayesian Information Criterion.

        Parameters
        ----------
        log_likelihood : float
            Model log-likelihood.
        k : int
            Number of estimated parameters.
        n : int
            Number of observations.

        Returns
        -------
        float
            BIC value (lower is better).
        """
        if not np.isfinite(log_likelihood) or n <= 0:
            return np.nan
        return k * np.log(n) - 2 * log_likelihood

    @staticmethod
    def _print_split_summary(df: pd.DataFrame) -> None:
        """Print mean metrics split by platform and market type.

        Parameters
        ----------
        df : pd.DataFrame
            Full comparison summary.
        """
        metrics = ["mse_bs", "mse_heston", "mse_bates",
                   "aic_bs", "aic_heston", "aic_bates",
                   "qlike_median_bs", "qlike_median_heston", "qlike_median_bates"]

        print("\n--- By Platform ---")
        if "platform" in df.columns:
            for platform, group in df.groupby("platform"):
                print(f"  {platform} (n={len(group)}):")
                for m in metrics:
                    print(f"    {m}: {group[m].mean():.6f}")

        print("\n--- Near Resolution (last 5 days) ---")
        if "near_resolution_frac" in df.columns:
            near = df[df["near_resolution_frac"] > 0.5]
            far = df[df["near_resolution_frac"] <= 0.5]
            for label, subset in [("Near", near), ("Far", far)]:
                if len(subset) > 0:
                    print(f"  {label} (n={len(subset)}):")
                    for m in metrics:
                        print(f"    {m}: {subset[m].mean():.6f}")

    @staticmethod
    def _load_results(directory: Path) -> dict[str, dict]:
        """Load all JSON result files from a directory.

        Parameters
        ----------
        directory : Path
            Directory containing {ticker}.json files.

        Returns
        -------
        dict[str, dict]
            Mapping from ticker to parsed result dict.
        """
        results = {}
        if not directory.exists():
            return results
        for path in directory.glob("*.json"):
            with open(path) as f:
                data = json.load(f)
            ticker = data.get("ticker", path.stem)
            results[ticker] = data
        return results
