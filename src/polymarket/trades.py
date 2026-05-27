from __future__ import annotations

from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.polymarket.client import PolymarketClient
import config

# The CLOB /prices-history endpoint only retains data at fidelity >= 720 minutes
# for resolved/closed markets. We use 1440 (daily bars) because:
#   - Works for all closed markets from 2023 onwards
#   - Daily granularity is standard in empirical finance calibration
#   - Endpoint is public — no API key required
_FIDELITY_MINUTES = 1440


class PolymarketTradesPuller:
    """Pull daily price history from Polymarket and save as OHLCV-format Parquet.

    Uses the CLOB /prices-history endpoint (public, no auth required) with
    daily fidelity. The market parameter must be the YES-outcome token ID
    from clobTokenIds, not the conditionId.

    Parameters
    ----------
    client : PolymarketClient
        CLOB API client (auth header is ignored by /prices-history).
    data_dir : Path
        Root data directory (default from config).
    """

    def __init__(self, client: PolymarketClient,
                 data_dir: Path | None = None):
        self.client = client
        self.data_dir = data_dir or config.DATA_DIR
        self.raw_dir = self.data_dir / "raw" / "polymarket"
        self.raw_dir.mkdir(parents=True, exist_ok=True)

    def pull(self, contracts: list[dict]) -> None:
        """Download daily price history for each contract.

        Skips contracts whose output file already exists so the pull is
        resumable after interruption.

        Parameters
        ----------
        contracts : list[dict]
            Each dict must have keys:
              - 'condition_id': Polymarket condition ID (used as filename)
              - 'yes_token':    CLOB YES-outcome token ID (used for API call)
        """
        successes = 0
        failures: list[str] = []
        total_rows = 0

        for contract in tqdm(contracts, desc="Pulling Polymarket trades"):
            cid = contract["condition_id"]
            yes_token = contract["yes_token"]
            out_path = self.raw_dir / f"{cid}.parquet"

            if out_path.exists():
                successes += 1
                total_rows += len(pd.read_parquet(out_path))
                continue

            try:
                df = self._pull_condition(cid, yes_token)
                if df.empty:
                    failures.append(cid)
                    continue

                bars = self._format_daily_bars(df)
                bars.to_parquet(out_path, index=False)
                successes += 1
                total_rows += len(bars)
            except Exception as e:
                print(f"  Failed {cid}: {e}")
                failures.append(cid)

        print(f"\n--- Polymarket Trade Pull Summary ---")
        print(f"  Contracts pulled:  {successes}")
        print(f"  Total rows:        {total_rows}")
        print(f"  Failures:          {len(failures)}")
        if failures:
            print(f"  Failed IDs:        {failures[:20]}")
        print("-------------------------------------\n")

    def _pull_condition(self, condition_id: str, yes_token: str) -> pd.DataFrame:
        """Fetch daily price history for one contract via /prices-history.

        Parameters
        ----------
        condition_id : str
            Used only for logging; filename is based on this.
        yes_token : str
            CLOB YES-outcome token ID — required by the prices-history endpoint.

        Returns
        -------
        pd.DataFrame
            Raw rows with columns: timestamp_unix, close.
            Empty DataFrame if no data is available (e.g. pre-CLOB contracts).
        """
        history = self.client.get(
            "/prices-history",
            params={
                "market": yes_token,
                "interval": "all",
                "fidelity": _FIDELITY_MINUTES,
            },
        )
        if not history:
            return pd.DataFrame()

        df = pd.DataFrame(history)
        if "t" not in df.columns or "p" not in df.columns:
            return pd.DataFrame()

        df = df.rename(columns={"t": "timestamp_unix", "p": "close"})
        df["timestamp"] = pd.to_datetime(df["timestamp_unix"], unit="s", utc=True)
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        return df.dropna(subset=["close"])

    @staticmethod
    def _format_daily_bars(df: pd.DataFrame) -> pd.DataFrame:
        """Format daily price points as OHLCV-compatible bars.

        Since /prices-history only provides a close price per interval,
        open = high = low = close. Volume and trade_count are not available
        from this endpoint and are set to 0.

        Parameters
        ----------
        df : pd.DataFrame
            Output of _pull_condition with columns: timestamp, close.

        Returns
        -------
        pd.DataFrame
            Daily bars with columns: timestamp, open, high, low, close,
            volume, trade_count.
        """
        result = pd.DataFrame({
            "timestamp": df["timestamp"],
            "open":       df["close"],
            "high":       df["close"],
            "low":        df["close"],
            "close":      df["close"],
            "volume":     0.0,
            "trade_count": 0,
        })
        return result.sort_values("timestamp").reset_index(drop=True)
