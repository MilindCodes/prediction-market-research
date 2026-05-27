from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

import config


@dataclass
class HestonResult:
    """Calibration result for the Heston stochastic volatility model.

    Fields
    ------
    ticker : str
        Contract identifier.
    platform : str
        Source platform (kalshi or polymarket).
    kappa : float
        Mean reversion speed of variance.
    theta : float
        Long-run mean variance.
    xi : float
        Volatility of volatility.
    rho : float
        Correlation between price and variance shocks.
    v0 : float
        Initial variance.
    feller_satisfied : bool
        Whether 2*kappa*theta >= xi^2 holds.
    converged : bool
        Whether the optimizer reported convergence.
    mse : float
        In-sample mean squared error.
    log_likelihood : float
        Log-likelihood of the fitted model.
    n_obs : int
        Number of observations used in calibration.
    """
    ticker: str
    platform: str
    kappa: float
    theta: float
    xi: float
    rho: float
    v0: float
    feller_satisfied: bool
    converged: bool
    mse: float
    log_likelihood: float
    n_obs: int


class HestonCalibrator:
    """Calibrate the Heston stochastic volatility model to an implied
    volatility time series.

    The Heston model parameters:
        kappa  — mean reversion speed of variance
        theta  — long-run mean variance
        xi     — vol of vol
        rho    — correlation between price and variance shocks
        v0     — initial variance

    Fits by minimizing sum of squared differences between model-implied
    variance path and observed sigma_implied^2 series.

    Parameters
    ----------
    data_dir : Path
        Root data directory (default from config).
    """

    PARAM_BOUNDS = [
        (0.01, 20.0),    # kappa
        (1e-6, 2.0),     # theta
        (0.01, 5.0),     # xi
        (-0.999, 0.999), # rho
        (1e-6, 2.0),     # v0
    ]
    PARAM_NAMES = ["kappa", "theta", "xi", "rho", "v0"]

    def __init__(self, data_dir: Path | None = None):
        self.data_dir = data_dir or config.DATA_DIR
        self.out_dir = self.data_dir / "processed" / "heston_params"
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def calibrate(self, sigma_series: np.ndarray, ticker: str,
                  platform: str, dt: float = 1 / 252) -> HestonResult:
        """Fit Heston parameters to an observed implied volatility series.

        Parameters
        ----------
        sigma_series : np.ndarray
            1-D array of sigma_implied values (not squared). NaN values are
            dropped before fitting.
        ticker : str
            Contract identifier.
        platform : str
            Source platform name (kalshi or polymarket).
        dt : float
            Time step in years between observations (default: 1/252 for
            daily; adjust to 1/6552 for hourly).

        Returns
        -------
        HestonResult
            Fitted parameters and diagnostics.
        """
        sigma_clean = sigma_series[np.isfinite(sigma_series)]
        if len(sigma_clean) < 10:
            return self._empty_result(ticker, platform, len(sigma_clean))

        var_observed = sigma_clean ** 2

        x0 = self._initial_guess(var_observed)

        result = minimize(
            self._objective,
            x0,
            args=(var_observed, dt),
            method="L-BFGS-B",
            bounds=self.PARAM_BOUNDS,
            options={"maxiter": 2000, "ftol": 1e-12},
        )

        kappa, theta, xi, rho, v0 = result.x
        feller = 2 * kappa * theta >= xi ** 2
        if not feller:
            print(f"  [Heston] WARNING: Feller condition violated for {ticker} "
                  f"(2*kappa*theta={2*kappa*theta:.4f}, xi^2={xi**2:.4f})")

        var_model = self._simulate_variance_path(result.x, len(var_observed), dt)
        mse = float(np.mean((var_observed - var_model) ** 2))
        ll = self._log_likelihood(var_observed, var_model)

        heston_result = HestonResult(
            ticker=ticker,
            platform=platform,
            kappa=float(kappa),
            theta=float(theta),
            xi=float(xi),
            rho=float(rho),
            v0=float(v0),
            feller_satisfied=bool(feller),
            converged=bool(result.success),
            mse=mse,
            log_likelihood=ll,
            n_obs=len(var_observed),
        )

        self._save(heston_result)
        return heston_result

    def _objective(self, params: np.ndarray, var_observed: np.ndarray,
                   dt: float) -> float:
        """Sum of squared errors between model and observed variance.

        Parameters
        ----------
        params : np.ndarray
            [kappa, theta, xi, rho, v0].
        var_observed : np.ndarray
            Observed variance (sigma^2) series.
        dt : float
            Time step in years.

        Returns
        -------
        float
            Sum of squared errors.
        """
        var_model = self._simulate_variance_path(params, len(var_observed), dt)
        return float(np.sum((var_observed - var_model) ** 2))

    @staticmethod
    def _simulate_variance_path(params: np.ndarray, n: int,
                                dt: float) -> np.ndarray:
        """Simulate the deterministic (expected) variance path under Heston.

        Uses the Euler discretization of E[v_t] under the Heston mean
        reversion dynamics: dv = kappa*(theta - v)*dt.

        Parameters
        ----------
        params : np.ndarray
            [kappa, theta, xi, rho, v0].
        n : int
            Number of time steps.
        dt : float
            Time step in years.

        Returns
        -------
        np.ndarray
            Model-implied variance path of length n.
        """
        kappa, theta, _, _, v0 = params
        v = np.empty(n)
        v[0] = v0
        for i in range(1, n):
            v[i] = v[i - 1] + kappa * (theta - v[i - 1]) * dt
            v[i] = max(v[i], 1e-10)
        return v

    @staticmethod
    def _log_likelihood(var_observed: np.ndarray,
                        var_model: np.ndarray) -> float:
        """Gaussian log-likelihood assuming constant residual variance.

        Parameters
        ----------
        var_observed : np.ndarray
            Observed variance series.
        var_model : np.ndarray
            Model-fitted variance series.

        Returns
        -------
        float
            Log-likelihood value.
        """
        residuals = var_observed - var_model
        n = len(residuals)
        sigma2 = np.var(residuals)
        if sigma2 < 1e-30:
            sigma2 = 1e-30
        ll = -0.5 * n * np.log(2 * np.pi * sigma2) - np.sum(residuals ** 2) / (2 * sigma2)
        return float(ll)

    @staticmethod
    def _initial_guess(var_observed: np.ndarray) -> np.ndarray:
        """Heuristic starting point for the optimizer.

        Parameters
        ----------
        var_observed : np.ndarray
            Observed variance series.

        Returns
        -------
        np.ndarray
            Initial [kappa, theta, xi, rho, v0].
        """
        theta0 = float(np.mean(var_observed))
        v0 = float(var_observed[0])
        xi0 = float(np.std(np.diff(var_observed)))
        return np.array([1.0, max(theta0, 1e-5), max(xi0, 0.05), -0.5, max(v0, 1e-5)])

    def _empty_result(self, ticker: str, platform: str,
                      n_obs: int) -> HestonResult:
        """Return a result with NaN parameters when calibration is not
        possible.

        Parameters
        ----------
        ticker : str
            Contract identifier.
        platform : str
            Source platform.
        n_obs : int
            Number of valid observations (too few to calibrate).

        Returns
        -------
        HestonResult
        """
        return HestonResult(
            ticker=ticker, platform=platform,
            kappa=np.nan, theta=np.nan, xi=np.nan, rho=np.nan, v0=np.nan,
            feller_satisfied=False, converged=False,
            mse=np.nan, log_likelihood=np.nan, n_obs=n_obs,
        )

    def _save(self, result: HestonResult) -> None:
        """Persist calibration result as JSON.

        Parameters
        ----------
        result : HestonResult
            Fitted model output.
        """
        path = self.out_dir / f"{result.ticker}.json"
        with open(path, "w") as f:
            json.dump(asdict(result), f, indent=2, default=_json_default)


def _json_default(obj):
    """Handle numpy types in JSON serialization."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
