from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm

import config


def _ensure_utc(t) -> pd.Timestamp:
    ts = pd.Timestamp(t)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


class ImpliedVolExtractor:
    """Extract Cash-or-Nothing implied volatility from prediction market price
    series.

    The Cash-or-Nothing formula:
        V = exp(-r * T) * N(d2)
        d2 = (log(S/K) + (r - sigma^2/2) * T) / (sigma * sqrt(T))

    For prediction markets:
        V = observed price P_t (probability, 0 to 1)
        S = P_t (current price as proxy for underlying)
        K = 0.50 (fixed strike, neutral prior)
        r = 0 (no risk-free rate in prediction markets)
        T = time to resolution in years

    Parameters
    ----------
    close_time : pd.Timestamp
        UTC-aware contract resolution timestamp.
    fomc_dates : list[pd.Timestamp]
        List of FOMC announcement timestamps (UTC-aware) for jump window
        flagging.
    data_dir : Path
        Root data directory (default from config).
    """

    STRIKE = 0.50
    RISK_FREE_RATE = 0.0
    SIGMA_LOWER = 1e-6
    SIGMA_UPPER = 10.0
    BOUNDARY_LOW = 0.02
    BOUNDARY_HIGH = 0.98
    FOMC_WINDOW_MINUTES = 30

    def __init__(self, close_time: pd.Timestamp,
                 fomc_dates: list[pd.Timestamp] | None = None,
                 data_dir: Path | None = None):
        self.close_time = _ensure_utc(close_time)
        self.fomc_dates = [_ensure_utc(d) for d in (fomc_dates or [])]
        self.data_dir = data_dir or config.DATA_DIR
        self.out_dir = self.data_dir / "processed" / "implied_vol"
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def extract(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """Compute implied volatility for each hourly observation.

        Parameters
        ----------
        df : pd.DataFrame
            Hourly OHLCV data with 'timestamp' and 'close' columns.
            close is in [0, 1] probability space.
        ticker : str
            Contract identifier (used for output filename).

        Returns
        -------
        pd.DataFrame
            With columns: timestamp, price, T, sigma_implied, sigma_logit,
            near_boundary, is_jump_window.
        """
        records: list[dict] = []

        for _, row in df.iterrows():
            ts = _ensure_utc(row["timestamp"])
            price = float(row["close"])
            T = self._time_to_resolution(ts)

            sigma = self._solve_implied_vol(price, T)

            sigma_logit = np.nan
            if np.isfinite(sigma) and 0 < price < 1:
                sigma_logit = sigma / (price * (1.0 - price))

            records.append({
                "timestamp": ts,
                "price": price,
                "T": T,
                "sigma_implied": sigma,
                "sigma_logit": sigma_logit,
                "near_boundary": price < self.BOUNDARY_LOW or price > self.BOUNDARY_HIGH,
                "is_jump_window": self._in_fomc_window(ts),
            })

        result = pd.DataFrame(records)
        result.to_parquet(self.out_dir / f"{ticker}.parquet", index=False)
        return result

    def _time_to_resolution(self, timestamp: pd.Timestamp) -> float:
        """Compute time remaining until contract resolution in years.

        Parameters
        ----------
        timestamp : pd.Timestamp
            Current observation timestamp (UTC).

        Returns
        -------
        float
            Time to resolution in years (365.25 days/year). Returns 0.0 if
            the timestamp is at or past close_time.
        """
        delta = (self.close_time - timestamp).total_seconds()
        if delta <= 0:
            return 0.0
        return delta / (365.25 * 86400)

    def _solve_implied_vol(self, price: float, T: float) -> float:
        """Back-solve for sigma by inverting the Cash-or-Nothing formula.

        Uses Brent's method on [SIGMA_LOWER, SIGMA_UPPER].

        Parameters
        ----------
        price : float
            Observed market price (probability, 0 to 1).
        T : float
            Time to resolution in years.

        Returns
        -------
        float
            Implied volatility sigma, or NaN if no solution exists.
        """
        if T <= 0 or price <= 0 or price >= 1:
            return np.nan

        S = price
        K = self.STRIKE
        r = self.RISK_FREE_RATE

        def objective(sigma: float) -> float:
            """Difference between model price and observed price."""
            d2 = (np.log(S / K) + (r - 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
            model_price = np.exp(-r * T) * norm.cdf(d2)
            return model_price - price

        try:
            sigma = brentq(objective, self.SIGMA_LOWER, self.SIGMA_UPPER,
                           xtol=1e-10, maxiter=200)
            return float(sigma)
        except (ValueError, RuntimeError):
            return np.nan

    def _in_fomc_window(self, timestamp: pd.Timestamp) -> bool:
        """Check if a timestamp falls within the ±30-minute window around any
        FOMC announcement.

        Parameters
        ----------
        timestamp : pd.Timestamp
            Observation timestamp (UTC).

        Returns
        -------
        bool
            True if within any FOMC window.
        """
        window = pd.Timedelta(minutes=self.FOMC_WINDOW_MINUTES)
        for fomc_dt in self.fomc_dates:
            if abs(timestamp - fomc_dt) <= window:
                return True
        return False
