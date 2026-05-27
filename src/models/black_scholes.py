from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import norm

import config


@dataclass
class BSResult:
    """Calibration result for the Black-Scholes Cash-or-Nothing model.

    Fields
    ------
    ticker : str
        Contract identifier.
    platform : str
        Source platform.
    sigma_bs : float
        Calibrated constant implied volatility (single free parameter).
    mse : float
        In-sample MSE: mean((observed_variance - sigma_bs^2)^2).
    log_likelihood : float
        Gaussian log-likelihood of the variance residuals.
    n_obs : int
        Number of valid observations used in calibration.
    converged : bool
        Whether the optimizer found a valid minimum.
    """
    ticker: str
    platform: str
    sigma_bs: float
    mse: float
    log_likelihood: float
    n_obs: int
    converged: bool


class BSCalibrator:
    """Calibrate the Black-Scholes Cash-or-Nothing model.

    BS assumes constant volatility. We find the single sigma_BS that
    minimises MSE between the BS-implied variance (sigma_BS^2, constant)
    and the observed implied variance series (sigma_implied^2).

    This is the correct apples-to-apples baseline for comparing against
    Heston (k=5) and Bates (k=8): all three are evaluated on how well
    their model variance fits the observed implied variance time series.

    Parameters
    ----------
    data_dir : Path
        Root data directory (default from config).
    """

    def __init__(self, data_dir: Path | None = None):
        self.data_dir = data_dir or config.DATA_DIR
        self.out_dir = self.data_dir / "processed" / "bs_params"
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def calibrate(self, sigma_series: np.ndarray, ticker: str,
                  platform: str) -> BSResult:
        """Fit a constant sigma_BS to the implied volatility series.

        Parameters
        ----------
        sigma_series : np.ndarray
            1-D array of sigma_implied values. NaN values are dropped.
        ticker : str
            Contract identifier.
        platform : str
            Source platform name.

        Returns
        -------
        BSResult
            Fitted constant volatility and diagnostics.
        """
        sigma_clean = sigma_series[np.isfinite(sigma_series)]
        if len(sigma_clean) < 2:
            return self._empty_result(ticker, platform, len(sigma_clean))

        var_obs = sigma_clean ** 2

        # Optimal constant sigma minimises MSE vs observed variance.
        # The closed-form solution is sigma_BS = sqrt(mean(var_obs)).
        result = minimize_scalar(
            lambda s: float(np.mean((var_obs - s ** 2) ** 2)),
            bounds=(1e-6, 10.0),
            method="bounded",
        )

        sigma_bs = float(result.x)
        var_bs = sigma_bs ** 2
        mse = float(np.mean((var_obs - var_bs) ** 2))
        ll = self._log_likelihood(var_obs, var_bs)

        bs_result = BSResult(
            ticker=ticker,
            platform=platform,
            sigma_bs=sigma_bs,
            mse=mse,
            log_likelihood=ll,
            n_obs=len(var_obs),
            converged=bool(result.success),
        )
        self._save(bs_result)
        return bs_result

    @staticmethod
    def _log_likelihood(var_obs: np.ndarray, var_const: float) -> float:
        """Gaussian log-likelihood for a constant-variance model."""
        residuals = var_obs - var_const
        n = len(residuals)
        sigma2 = float(np.var(residuals))
        if sigma2 < 1e-30:
            sigma2 = 1e-30
        return float(-0.5 * n * np.log(2 * np.pi * sigma2)
                     - np.sum(residuals ** 2) / (2 * sigma2))

    def _empty_result(self, ticker: str, platform: str, n_obs: int) -> BSResult:
        return BSResult(
            ticker=ticker, platform=platform,
            sigma_bs=np.nan, mse=np.nan, log_likelihood=np.nan,
            n_obs=n_obs, converged=False,
        )

    def _save(self, result: BSResult) -> None:
        path = self.out_dir / f"{result.ticker}.json"
        with open(path, "w") as f:
            json.dump(asdict(result), f, indent=2, default=_json_default)


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
