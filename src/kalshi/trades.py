from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.kalshi.client import KalshiClient
import config


class KalshiTradesPuller:
    """Pull tick-level trade data from Kalshi, aggregate into hourly OHLCV
    bars, and apply the log-odds preprocessing required for stochastic vol
    estimation.

    Preprocessing steps (per Riley's spec):
      1. Truncate close prices to [LOG_ODDS_CLIP_LO, LOG_ODDS_CLIP_HI]
         (default [0.02, 0.98]) to avoid log(0) / log(inf).
      2. Compute log-odds: X = log(p / (1 - p))
      3. Compute increments: deltaX = X[t] - X[t-1]

    Parameters
    ----------
    client : KalshiClient
        Authenticated API client.
    data_dir : Path
        Root data directory (default from config).
    """

    def __init__(self, client: KalshiClient, data_dir: Path | None = None):
        self.client = client
        self.data_dir = data_dir or config.DATA_DIR
        self.raw_dir = self.data_dir / "raw" / "kalshi"
        self.raw_dir.mkdir(parents=True, exist_ok=True)

    def pull(self, tickers: list[str]) -> None:
        """Download trade data for each ticker and save as hourly OHLCV + log-odds
        Parquet files.

        Skips tickers whose output file already exists so the pull is resumable.

        Parameters
        ----------
        tickers : list[str]
            Contract ticker symbols to pull.
        """
        successes = 0
        failures: list[str] = []
        total_rows = 0

        for ticker in tqdm(tickers, desc="Pulling Kalshi trades"):
            out_path = self.raw_dir / f"{ticker}.parquet"
            if out_path.exists():
                successes += 1
                total_rows += len(pd.read_parquet(out_path))
                continue

            try:
                df = self._pull_ticker(ticker)
                if df.empty:
                    failures.append(ticker)
                    continue

                bars = self._aggregate_hourly(df)
                bars = self._add_log_odds(bars)
                bars.to_parquet(out_path, index=False)
                successes += 1
                total_rows += len(bars)
            except Exception as e:
                print(f"  Failed {ticker}: {e}")
                failures.append(ticker)

        print(f"\n--- Kalshi Trade Pull Summary ---")
        print(f"  Contracts pulled:  {successes}")
        print(f"  Total rows:        {total_rows}")
        print(f"  Failures:          {len(failures)}")
        if failures:
            print(f"  Failed tickers:    {failures}")
        print("--------------------------------\n")

    def _pull_ticker(self, ticker: str) -> pd.DataFrame:
        """Fetch all trades for a single contract via paginated /trades calls.

        Parameters
        ----------
        ticker : str
            Contract ticker symbol.

        Returns
        -------
        pd.DataFrame
            Raw tick data with columns from the API response.
        """
        trades = self.client.get(
            "/trades",
            params={"ticker": ticker, "limit": 1000},
        )
        if not trades:
            return pd.DataFrame()

        df = pd.DataFrame(trades)
        df["created_time"] = pd.to_datetime(df["created_time"], utc=True)
        return df

    @staticmethod
    def _aggregate_hourly(df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate tick trades into hourly OHLCV bars.

        Close price is the volume-weighted average price (VWAP) within each
        hour, computed from yes_price and count fields.

        Parameters
        ----------
        df : pd.DataFrame
            Raw tick data with created_time, yes_price, and count columns.

        Returns
        -------
        pd.DataFrame
            Hourly bars with columns: timestamp, open, high, low, close,
            volume, trade_count.
        """
        df = df.copy()
        df["hour"] = df["created_time"].dt.floor("h")

        # Handle field name differences between old and new Kalshi API:
        # Old: yes_price, count | New: yes_price_dollars, count_fp
        price_col = "yes_price"
        if price_col not in df.columns and "yes_price_dollars" in df.columns:
            price_col = "yes_price_dollars"
        df["yes_price"] = pd.to_numeric(df[price_col], errors="coerce")

        count_col = "count"
        if count_col not in df.columns and "count_fp" in df.columns:
            count_col = "count_fp"
        df["count"] = pd.to_numeric(
            df.get(count_col, pd.Series(1, index=df.index)), errors="coerce"
        ).fillna(1).astype(int)

        grouped = df.groupby("hour")

        def vwap(g: pd.DataFrame) -> float:
            weights = g["count"].values.astype(float)
            prices = g["yes_price"].values.astype(float)
            total_w = weights.sum()
            if total_w == 0:
                return float(np.nanmean(prices))
            return float(np.nansum(prices * weights) / total_w)

        bars = pd.DataFrame({
            "timestamp": grouped["created_time"].first().apply(
                lambda t: t.floor("h")
            ),
            "open": grouped["yes_price"].first(),
            "high": grouped["yes_price"].max(),
            "low": grouped["yes_price"].min(),
            "close": grouped.apply(vwap, include_groups=False),
            "volume": grouped["count"].sum(),
            "trade_count": grouped.size(),
        }).reset_index(drop=True)

        bars = bars.sort_values("timestamp").reset_index(drop=True)
        return bars

    @staticmethod
    def _add_log_odds(bars: pd.DataFrame) -> pd.DataFrame:
        """Apply log-odds transform and compute increments.

        Steps:
          1. Truncate p to [LOG_ODDS_CLIP_LO, LOG_ODDS_CLIP_HI] so that
             log-odds remain finite.
          2. X = log(p / (1 - p))
          3. deltaX = X[t] - X[t-1]  (NaN for the first bar)

        Adds columns: close_clipped, log_odds, delta_log_odds.

        Parameters
        ----------
        bars : pd.DataFrame
            Hourly OHLCV bars (must have a 'close' column in [0, 1]).

        Returns
        -------
        pd.DataFrame
            Same rows with three additional columns.
        """
        bars = bars.copy()
        p = bars["close"].clip(config.LOG_ODDS_CLIP_LO, config.LOG_ODDS_CLIP_HI)
        bars["close_clipped"] = p
        bars["log_odds"] = np.log(p / (1.0 - p))
        bars["delta_log_odds"] = bars["log_odds"].diff()
        return bars
