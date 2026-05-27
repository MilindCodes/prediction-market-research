from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.stats import chi2

from src.models.heston import HestonCalibrator, HestonResult, _json_default
import config


@dataclass
class BatesResult:
    """Calibration result for the Bates model (Heston + Poisson jumps).

    Inherits all Heston fields and adds jump parameters plus a likelihood
    ratio test against the nested Heston model.

    Fields
    ------
    ticker : str
        Contract identifier.
    platform : str
        Source platform (kalshi or polymarket).
    kappa, theta, xi, rho, v0 : float
        Heston parameters.
    lambda_j : float
        Jump intensity (jumps per year).
    mu_j : float
        Mean jump size.
    sigma_j : float
        Jump size standard deviation.
    feller_satisfied : bool
        Whether 2*kappa*theta >= xi^2 holds.
    converged : bool
        Whether the optimizer reported convergence.
    mse : float
        In-sample mean squared error.
    log_likelihood : float
        Log-likelihood of the Bates model.
    log_likelihood_heston : float
        Log-likelihood of the nested Heston model (for LR test).
    lr_statistic : float
        Likelihood ratio test statistic.
    p_value_jump_significance : float
        p-value from chi-squared test with 3 degrees of freedom.
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
    lambda_j: float
    mu_j: float
    sigma_j: float
    feller_satisfied: bool
    converged: bool
    mse: float
    log_likelihood: float
    log_likelihood_heston: float
    lr_statistic: float
    p_value_jump_significance: float
    n_obs: int


class BatesCalibrator(HestonCalibrator):
    """Calibrate the Bates stochastic volatility model (Heston + Poisson
    jumps) to an implied volatility time series.

    Additional parameters beyond Heston:
        lambda_j  — jump intensity (jumps per year)
        mu_j      — mean jump size
        sigma_j   — jump size standard deviation

    Tests jump significance via likelihood ratio test against the nested
    Heston model (chi-squared, 3 degrees of freedom).

    Parameters
    ----------
    data_dir : Path
        Root data directory (default from config).
    """

    BATES_BOUNDS = [
        (0.01, 20.0),    # kappa
        (1e-6, 2.0),     # theta
        (0.01, 5.0),     # xi
        (-0.999, 0.999), # rho
        (1e-6, 2.0),     # v0
        (0.0, 50.0),     # lambda_j
        (-2.0, 2.0),     # mu_j
        (0.001, 2.0),    # sigma_j
    ]
    BATES_PARAM_NAMES = [
        "kappa", "theta", "xi", "rho", "v0",
        "lambda_j", "mu_j", "sigma_j",
    ]
    JUMP_DOF = 3

    def __init__(self, data_dir: Path | None = None):
        super().__init__(data_dir)
        self.out_dir = (data_dir or config.DATA_DIR) / "processed" / "bates_params"
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def calibrate(self, sigma_series: np.ndarray, ticker: str,
                  platform: str, dt: float = 1 / 252,
                  heston_result: HestonResult | None = None) -> BatesResult:
        """Fit Bates parameters to an observed implied volatility series.

        Parameters
        ----------
        sigma_series : np.ndarray
            1-D array of sigma_implied values (not squared).
        ticker : str
            Contract identifier.
        platform : str
            Source platform name.
        dt : float
            Time step in years between observations.
        heston_result : HestonResult, optional
            Pre-computed Heston result for the same contract. If None, the
            Heston model is calibrated first for the LR test.

        Returns
        -------
        BatesResult
            Fitted parameters, diagnostics, and jump significance test.
        """
        sigma_clean = sigma_series[np.isfinite(sigma_series)]
        if len(sigma_clean) < 10:
            return self._empty_bates_result(ticker, platform, len(sigma_clean))

        var_observed = sigma_clean ** 2

        if heston_result is None:
            heston_result = super().calibrate(sigma_series, ticker, platform, dt)

        x0 = self._bates_initial_guess(var_observed, heston_result)

        result = minimize(
            self._bates_objective,
            x0,
            args=(var_observed, dt),
            method="L-BFGS-B",
            bounds=self.BATES_BOUNDS,
            options={"maxiter": 3000, "ftol": 1e-12},
        )

        kappa, theta, xi, rho, v0, lambda_j, mu_j, sigma_j = result.x
        feller = 2 * kappa * theta >= xi ** 2
        if not feller:
            print(f"  [Bates] WARNING: Feller condition violated for {ticker}")

        var_model = self._simulate_bates_variance_path(result.x, len(var_observed), dt)
        mse = float(np.mean((var_observed - var_model) ** 2))
        ll_bates = self._log_likelihood(var_observed, var_model)
        ll_heston = heston_result.log_likelihood

        lr_stat = 2 * (ll_bates - ll_heston)
        lr_stat = max(lr_stat, 0.0)
        p_value = float(chi2.sf(lr_stat, self.JUMP_DOF))

        bates_result = BatesResult(
            ticker=ticker,
            platform=platform,
            kappa=float(kappa),
            theta=float(theta),
            xi=float(xi),
            rho=float(rho),
            v0=float(v0),
            lambda_j=float(lambda_j),
            mu_j=float(mu_j),
            sigma_j=float(sigma_j),
            feller_satisfied=bool(feller),
            converged=bool(result.success),
            mse=mse,
            log_likelihood=ll_bates,
            log_likelihood_heston=ll_heston,
            lr_statistic=float(lr_stat),
            p_value_jump_significance=p_value,
            n_obs=len(var_observed),
        )

        self._save_bates(bates_result)
        return bates_result

    def _bates_objective(self, params: np.ndarray,
                         var_observed: np.ndarray,
                         dt: float) -> float:
        """Sum of squared errors between Bates model and observed variance.

        Parameters
        ----------
        params : np.ndarray
            [kappa, theta, xi, rho, v0, lambda_j, mu_j, sigma_j].
        var_observed : np.ndarray
            Observed variance series.
        dt : float
            Time step in years.

        Returns
        -------
        float
            Sum of squared errors.
        """
        var_model = self._simulate_bates_variance_path(params, len(var_observed), dt)
        return float(np.sum((var_observed - var_model) ** 2))

    @staticmethod
    def _simulate_bates_variance_path(params: np.ndarray, n: int,
                                      dt: float) -> np.ndarray:
        """Simulate the expected variance path under the Bates model.

        Extends the Heston mean reversion with a deterministic jump
        contribution: E[dv_jump] = lambda_j * (mu_j^2 + sigma_j^2) * dt.

        Parameters
        ----------
        params : np.ndarray
            [kappa, theta, xi, rho, v0, lambda_j, mu_j, sigma_j].
        n : int
            Number of time steps.
        dt : float
            Time step in years.

        Returns
        -------
        np.ndarray
            Model-implied variance path of length n.
        """
        kappa, theta, _, _, v0, lambda_j, mu_j, sigma_j = params
        jump_var_contribution = lambda_j * (mu_j ** 2 + sigma_j ** 2)

        v = np.empty(n)
        v[0] = v0
        for i in range(1, n):
            v[i] = (v[i - 1]
                    + kappa * (theta - v[i - 1]) * dt
                    + jump_var_contribution * dt)
            v[i] = max(v[i], 1e-10)
        return v

    @staticmethod
    def _bates_initial_guess(var_observed: np.ndarray,
                             heston_result: HestonResult) -> np.ndarray:
        """Starting point that seeds Heston params from the prior fit and adds
        moderate jump parameters.

        Parameters
        ----------
        var_observed : np.ndarray
            Observed variance series.
        heston_result : HestonResult
            Previously calibrated Heston result.

        Returns
        -------
        np.ndarray
            Initial [kappa, theta, xi, rho, v0, lambda_j, mu_j, sigma_j].
        """
        if np.isfinite(heston_result.kappa):
            heston_params = [
                heston_result.kappa, heston_result.theta,
                heston_result.xi, heston_result.rho, heston_result.v0,
            ]
        else:
            theta0 = float(np.mean(var_observed))
            v0 = float(var_observed[0])
            xi0 = float(np.std(np.diff(var_observed)))
            heston_params = [1.0, max(theta0, 1e-5), max(xi0, 0.05),
                             -0.5, max(v0, 1e-5)]

        return np.array(heston_params + [1.0, 0.0, 0.1])

    def _empty_bates_result(self, ticker: str, platform: str,
                            n_obs: int) -> BatesResult:
        """Return NaN result when calibration is impossible.

        Parameters
        ----------
        ticker : str
            Contract identifier.
        platform : str
            Source platform.
        n_obs : int
            Number of valid observations.

        Returns
        -------
        BatesResult
        """
        return BatesResult(
            ticker=ticker, platform=platform,
            kappa=np.nan, theta=np.nan, xi=np.nan, rho=np.nan, v0=np.nan,
            lambda_j=np.nan, mu_j=np.nan, sigma_j=np.nan,
            feller_satisfied=False, converged=False,
            mse=np.nan, log_likelihood=np.nan, log_likelihood_heston=np.nan,
            lr_statistic=np.nan, p_value_jump_significance=np.nan,
            n_obs=n_obs,
        )

    def _save_bates(self, result: BatesResult) -> None:
        """Persist Bates calibration result as JSON.

        Parameters
        ----------
        result : BatesResult
            Fitted model output.
        """
        path = self.out_dir / f"{result.ticker}.json"
        with open(path, "w") as f:
            json.dump(asdict(result), f, indent=2, default=_json_default)
