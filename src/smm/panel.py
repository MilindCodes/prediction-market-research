"""
§4.1  Panel builder.

Produces the tidy long panel (contract_id, t, p, X, delta_X) that
§4.2 and §4.3 consume.

If per-contract daily-bar parquets already exist under data/raw/polymarket/,
the builder reads them directly (VWAP close is used; functionally
indistinguishable from last-trade for daily bars on a sparse market).
The grid is daily because the CLOB prices-history endpoint only retains
daily (fidelity=1440) granularity for closed markets — an hourly grid
would forward-fill ~24 artificial zero-increments per real observation.

If a contract's parquet is missing and a KalshiClient is provided,
the builder falls back to a fresh pull that routes settled contracts to
GET /historical/trades and open/recent ones to GET /markets/trades.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import config

try:
    from src.kalshi.client import KalshiClient
except ImportError:
    KalshiClient = None  # type: ignore[misc,assignment]

CLIP_LO: float = config.LOG_ODDS_CLIP_LO
CLIP_HI: float = config.LOG_ODDS_CLIP_HI
GRID_FREQ = "D"


class SMMPanelBuilder:
    """Build the unified long panel required by §4.2 and §4.3.

    Panel schema
    ------------
    contract_id : str
    t           : int   — 0-based integer time index within contract
    p           : float — clipped YES probability in (0, 1)
    X           : float — log-odds = log(p / (1-p))
    delta_X     : float — within-contract first difference of X

    Steps
    -----
    1. Load existing hourly-bar parquets (or pull fresh via API).
    2. Reindex to a complete hourly grid; forward-fill within contract life.
    3. Clip p to [CLIP_LO, CLIP_HI]; compute X and ΔX (never across boundary).
    4. Concatenate and save to data/processed/smm_panel.parquet.
    """

    def __init__(
        self,
        client=None,
        data_dir: Path | None = None,
        freq: str = GRID_FREQ,
    ):
        self.client = client
        self.data_dir = data_dir or config.DATA_DIR
        self.raw_dir = self.data_dir / "raw" / "polymarket"
        self.processed_dir = self.data_dir / "processed"
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.freq = freq
        self._hist_cutoff: pd.Timestamp | None = None

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def build(
        self,
        tickers: list[str] | None = None,
        force: bool = False,
        min_bars: int = 10,
    ) -> pd.DataFrame:
        """Build and save the SMM panel.

        Parameters
        ----------
        tickers : list[str] | None
            Contracts to include. None → reads catalog_filtered.parquet.
        force : bool
            Rebuild even if smm_panel.parquet already exists.
        min_bars : int
            Minimum number of valid increments for a contract to be included.

        Returns
        -------
        pd.DataFrame with columns: contract_id, t, p, X, delta_X
        """
        out_path = self.processed_dir / "smm_panel.parquet"
        if out_path.exists() and not force:
            print(f"  Loading existing SMM panel from {out_path}")
            panel = pd.read_parquet(out_path)
            n = panel["contract_id"].nunique()
            print(f"  {n} contracts, {len(panel)} rows")
            return panel

        if tickers is None:
            tickers = self._load_catalog_tickers()

        if not tickers:
            raise RuntimeError(
                "No tickers found. Run 'catalog-kalshi' first or pass tickers= explicitly."
            )

        contracts: list[pd.DataFrame] = []
        skipped: list[str] = []

        for ticker in tickers:
            ohlcv = self._load_or_pull(ticker)
            if ohlcv is None or ohlcv.empty:
                skipped.append(ticker)
                continue
            chunk = self._process_contract(ohlcv, ticker)
            if chunk is None or len(chunk) < min_bars:
                skipped.append(ticker)
                continue
            contracts.append(chunk)

        if not contracts:
            raise RuntimeError(
                "No valid contract data found. Check data/raw/kalshi/ or API credentials."
            )

        panel = pd.concat(contracts, ignore_index=True)
        panel.to_parquet(out_path, index=False)

        n_ok = panel["contract_id"].nunique()
        print(
            f"\nSMM panel: {n_ok} contracts, {len(panel)} rows"
            + (f" ({len(skipped)} skipped)" if skipped else " (all OK)")
        )
        if skipped:
            print(f"  Skipped: {skipped[:10]}"
                  + (" ..." if len(skipped) > 10 else ""))
        return panel

    # ------------------------------------------------------------------
    # Load / pull
    # ------------------------------------------------------------------

    def _load_catalog_tickers(self) -> list[str]:
        # Prefer the per-contract parquets already pulled to raw_dir: the
        # filenames are the condition IDs, and (unlike the scraped catalog)
        # they are guaranteed to have price data behind them.
        pulled = sorted(
            p.stem for p in self.raw_dir.glob("0x*.parquet")
        )
        if pulled:
            return self._drop_excluded_groups(pulled)

        cat_path = self.raw_dir / "catalog_filtered.parquet"
        if not cat_path.exists():
            print("  WARNING: no pulled contracts and no "
                  "catalog_filtered.parquet — run the catalog step first.")
            return []
        df = pd.read_parquet(cat_path)
        for col in ("ticker", "conditionId"):
            if col in df.columns:
                return df[col].tolist()
        return []

    def _drop_excluded_groups(self, tickers: list[str]) -> list[str]:
        """Drop contracts excluded from the §4 corpus.

        Election markets were cut from the corpus (July 2026) — the panel is
        Fed-only.  Classification comes from the verified identity mapping;
        contracts absent from the mapping are kept.
        """
        id_path = self.data_dir / "exports" / "polymarket_contract_identities.csv"
        if not id_path.exists():
            return tickers
        groups = pd.read_csv(id_path).set_index("conditionId")["group"]
        kept = [t for t in tickers if groups.get(t) != "election"]
        n_dropped = len(tickers) - len(kept)
        if n_dropped:
            print(f"  Corpus filter: dropped {n_dropped} election contracts "
                  f"({len(kept)} remain)")
        return kept

    def _load_or_pull(self, ticker: str) -> pd.DataFrame | None:
        """Return hourly-bar DataFrame for one ticker."""
        parquet_path = self.raw_dir / f"{ticker}.parquet"
        if parquet_path.exists():
            return pd.read_parquet(parquet_path)
        if self.client is not None:
            return self._pull_ticker(ticker)
        return None

    def _pull_ticker(self, ticker: str) -> pd.DataFrame | None:
        """Pull tick trades and aggregate to hourly last-trade bars."""
        # Try live endpoint first
        trades = self.client.get(
            "/markets/trades",
            params={"ticker": ticker, "limit": 1000},
        )
        # Route to historical tier if live returns nothing
        if not trades:
            trades = self.client.get(
                "/historical/trades",
                params={"ticker": ticker, "limit": 1000},
            )
        if not trades:
            return None

        df = pd.DataFrame(trades)
        df["created_time"] = pd.to_datetime(df["created_time"], utc=True)

        # Normalise price column name (API field names vary)
        if "yes_price" not in df.columns and "yes_price_dollars" in df.columns:
            df["yes_price"] = df["yes_price_dollars"]
        df["yes_price"] = pd.to_numeric(df.get("yes_price"), errors="coerce")

        if df["yes_price"].isna().all():
            return None

        # Resample to hourly last-trade
        bars = self._resample_last(df)
        return bars if not bars.empty else None

    def _historical_cutoff(self) -> pd.Timestamp | None:
        if self._hist_cutoff is not None:
            return self._hist_cutoff
        if self.client is None:
            return None
        try:
            resp = self.client.get_single("/historical/cutoff")
            ts_str = resp.get("cutoff_time") or resp.get("cutoff")
            if ts_str:
                self._hist_cutoff = pd.to_datetime(ts_str, utc=True)
        except Exception:
            pass
        return self._hist_cutoff

    # ------------------------------------------------------------------
    # Resampling
    # ------------------------------------------------------------------

    def _resample_last(self, df: pd.DataFrame) -> pd.DataFrame:
        """Resample raw ticks to hourly grid, taking the last trade price."""
        df = df.copy()
        df["hour"] = df["created_time"].dt.floor(self.freq)
        grouped = df.groupby("hour")["yes_price"].last()
        return (
            grouped.reset_index()
                   .rename(columns={"hour": "timestamp", "yes_price": "close"})
        )

    # ------------------------------------------------------------------
    # Per-contract processing
    # ------------------------------------------------------------------

    def _process_contract(
        self, ohlcv: pd.DataFrame, ticker: str
    ) -> pd.DataFrame | None:
        """Apply grid completion, clipping, logit, and differencing."""
        # Identify time and price columns
        time_col = "timestamp" if "timestamp" in ohlcv.columns else (
            "time" if "time" in ohlcv.columns else None
        )
        if time_col is None:
            return None

        price_col = "close" if "close" in ohlcv.columns else (
            "p_raw" if "p_raw" in ohlcv.columns else None
        )
        if price_col is None:
            return None

        bars = ohlcv[[time_col, price_col]].copy()
        bars = bars.rename(columns={time_col: "time", price_col: "p_raw"})
        bars["time"] = pd.to_datetime(bars["time"], utc=True, errors="coerce")
        bars = bars.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)

        if bars.empty:
            return None

        # Snap bar timestamps onto the grid before reindexing — raw bars can
        # carry second-level offsets (e.g. 00:00:02) that would otherwise
        # miss every grid point and drop the whole contract.
        bars["time"] = bars["time"].dt.floor(self.freq)
        bars = bars.groupby("time", as_index=False).last()

        # Complete hourly grid and forward-fill within contract life
        t_start = bars["time"].iloc[0].floor(self.freq)
        t_end = bars["time"].iloc[-1].ceil(self.freq)
        full_idx = pd.date_range(t_start, t_end, freq=self.freq, tz="UTC")
        bars = (
            bars.set_index("time")
                .reindex(full_idx)
                .rename_axis("time")
                .reset_index()
        )
        bars["p_raw"] = bars["p_raw"].ffill()
        bars = bars.dropna(subset=["p_raw"]).reset_index(drop=True)

        if len(bars) < 3:
            return None

        p_raw = bars["p_raw"].values.astype(float)
        # Prices from Kalshi API come in integer cents (1–99)
        if p_raw.max() > 1.5:
            p_raw = p_raw / 100.0

        p = np.clip(p_raw, CLIP_LO, CLIP_HI)
        X = np.log(p / (1.0 - p))

        # Within-contract differencing — never across contract boundary
        delta_X = np.empty(len(X))
        delta_X[0] = np.nan
        delta_X[1:] = X[1:] - X[:-1]

        out = pd.DataFrame({
            "contract_id": ticker,
            "t": np.arange(len(bars)),
            "p_raw": p_raw,   # unclipped — kept so truncation sensitivity can re-clip
            "p": p,
            "X": X,
            "delta_X": delta_X,
        })
        return out.dropna(subset=["delta_X"]).reset_index(drop=True)
