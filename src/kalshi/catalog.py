from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.kalshi.client import KalshiClient
import config


# ---------------------------------------------------------------------------
# Known Kalshi economic indicator series tickers.
# Pulled individually — each is fast (1-2 API calls max).
# Focus: FOMC rate decisions, CPI releases, and other macro data events.
# Political/election markets are excluded (too overdone as a category).
# ---------------------------------------------------------------------------
ECONOMIC_SERIES_TICKERS = [
    "KXFED",       # FOMC rate decisions (primary series)
    "KXCPI",       # CPI inflation releases
    "KXCORECPI",   # Core CPI (ex food & energy)
    "KXPCE",       # PCE inflation (Fed's preferred gauge)
    "KXGDP",       # GDP growth releases
    "KXJOBS",      # Nonfarm payrolls / jobs report
    "KXUNRATE",    # Unemployment rate
]


class KalshiCatalog:
    """Pull and filter the Kalshi market catalog using targeted queries.

    Strategy:
      - Pull each known economic series ticker (KXFED, KXCPI, etc.)
      - Keyword-filter for FOMC / CPI / macro events only
      - Political and election markets are excluded

    Parameters
    ----------
    client : KalshiClient
        API client.
    data_dir : Path
        Root data directory (default from config).
    """

    def __init__(self, client: KalshiClient,
                 data_dir: Path | None = None):
        self.client = client
        self.data_dir = data_dir or config.DATA_DIR
        self.raw_dir = self.data_dir / "raw" / "kalshi"
        self.raw_dir.mkdir(parents=True, exist_ok=True)

    def pull_catalog(self) -> pd.DataFrame:
        """Fetch FOMC and economic indicator markets from the Kalshi API.

        Pulls each known economic series ticker (FOMC, CPI, PCE, etc.) and
        keyword-filters the results. Political and election markets are
        excluded — we focus on macro economic releases only.

        Returns
        -------
        pd.DataFrame
            Filtered catalog ready for the liquidity filter step.
        """
        all_markets: list[dict] = []
        n_series = len(ECONOMIC_SERIES_TICKERS)

        print(f"  Fetching {n_series} economic series tickers "
              f"(FOMC, CPI, PCE, GDP, Jobs, ...)...")
        for i, series_ticker in enumerate(ECONOMIC_SERIES_TICKERS, 1):
            events = self.client.get(
                "/events",
                params={
                    "series_ticker": series_ticker,
                    "with_nested_markets": "true",
                    "limit": 200,
                },
                progress_label=series_ticker,
            )
            is_fed = series_ticker == "KXFED"
            markets = self._extract_markets(events, is_fed=is_fed)
            if markets:
                print(f"  [{i}/{n_series}] {series_ticker}: "
                      f"{len(events)} events, {len(markets)} markets")
            all_markets.extend(markets)

        # ---------------------------------------------------------------
        # Deduplicate and save
        # ---------------------------------------------------------------
        if not all_markets:
            print("\nWARNING: No markets fetched. Check API credentials and "
                  "internet connection.")
            return pd.DataFrame()

        df = pd.DataFrame(all_markets).drop_duplicates(subset=["ticker"])
        print(f"\n  Total unique markets before filtering: {len(df)}")

        df.to_parquet(self.raw_dir / "catalog_full.parquet", index=False)

        df = self._parse_timestamps(df)
        df = self._classify_keywords(df)
        df = self._apply_filters(df)

        df.to_parquet(self.raw_dir / "catalog_filtered.parquet", index=False)
        return df

    def apply_liquidity_filter(self, df: pd.DataFrame,
                               min_volume: float | None = None) -> pd.DataFrame:
        """Filter by minimum trading volume (volume_fp field).

        No per-contract API calls needed — volume is included from the
        event pull.

        Parameters
        ----------
        df : pd.DataFrame
            Pre-filtered catalog (output of pull_catalog).
        min_volume : float, optional
            Minimum volume_fp threshold. Defaults to MIN_TRADE_COUNT.

        Returns
        -------
        pd.DataFrame
            Filtered catalog.
        """
        threshold = min_volume if min_volume is not None else config.MIN_TRADE_COUNT

        if "volume_fp" not in df.columns:
            print("  WARNING: volume_fp column not found — skipping liquidity filter")
            self._export_review_csv(df)
            return df

        df["volume_fp"] = pd.to_numeric(df["volume_fp"], errors="coerce").fillna(0)
        count_before = len(df)
        df = df[df["volume_fp"] >= threshold].reset_index(drop=True)
        print(f"  Liquidity filter (volume_fp >= {threshold}): "
              f"{count_before} -> {len(df)}")

        self._export_review_csv(df)
        df.to_parquet(self.raw_dir / "catalog_filtered.parquet", index=False)
        return df

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _extract_markets(self, events: list[dict],
                         is_fed: bool) -> list[dict]:
        """Flatten nested markets out of events, attaching event metadata.

        Parameters
        ----------
        events : list[dict]
            Raw event objects from the /events endpoint.
        is_fed : bool
            Whether these events are Fed/FOMC markets.

        Returns
        -------
        list[dict]
            Flat list of market dicts.
        """
        markets: list[dict] = []
        for event in events:
            nested = event.get("markets", [])
            if not isinstance(nested, list):
                continue
            for market in nested:
                m = dict(market)
                m.setdefault("category", event.get("category", ""))
                m.setdefault("series_ticker", event.get("series_ticker", ""))
                m.setdefault("event_title", event.get("title", ""))
                m["_is_fed_source"] = is_fed
                markets.append(m)
        return markets

    def _parse_timestamps(self, df: pd.DataFrame) -> pd.DataFrame:
        """Parse open_time and close_time to UTC-aware datetimes."""
        df = df.copy()
        # open_time: prefer open_time, fall back to created_time
        open_raw = df["open_time"] if "open_time" in df.columns \
            else df.get("created_time", pd.Series("", index=df.index))
        close_raw = df["close_time"] if "close_time" in df.columns \
            else df.get("expiration_time", pd.Series("", index=df.index))

        df["open_time"] = pd.to_datetime(open_raw, utc=True, errors="coerce")
        df["close_time"] = pd.to_datetime(close_raw, utc=True, errors="coerce")
        df["duration_days"] = (
            (df["close_time"] - df["open_time"]).dt.total_seconds() / 86400
        )
        return df

    def _classify_keywords(self, df: pd.DataFrame) -> pd.DataFrame:
        """Tag each contract as is_fed and/or is_economic."""
        df = df.copy()
        text = (
            df.get("title", pd.Series("", index=df.index))
              .fillna("").str.lower()
            + " "
            + df.get("event_title", pd.Series("", index=df.index))
              .fillna("").str.lower()
            + " "
            + df.get("series_ticker", pd.Series("", index=df.index))
              .fillna("").str.lower()
        )

        # Any contract from a KXFED event is Fed regardless of title text
        df["is_fed"] = (
            df.get("_is_fed_source", pd.Series(False, index=df.index))
            | text.apply(lambda t: any(kw in t for kw in config.FED_KEYWORDS))
        )
        df["is_economic"] = text.apply(
            lambda t: any(kw in t for kw in config.ECONOMIC_KEYWORDS)
        )
        return df

    def _apply_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        """Sequential attrition filters with counts printed at each step."""
        print("\n--- Attrition Table (Kalshi) ---")
        print(f"  0. Raw fetch (economic series):    {len(df)}")

        # Kalshi Elections API uses "finalized" for resolved markets.
        # "settled" is not a valid status on this API.
        mask = df["status"].isin(["finalized", "settled"])  # settled kept as fallback
        df = df[mask].reset_index(drop=True)
        print(f"  1. Status finalized/settled:       {len(df)}")

        # Kalshi Elections API only has 2025-2026 data — use the
        # Kalshi-specific date window, NOT the Polymarket 2022-2024 window.
        start = pd.Timestamp(config.KALSHI_DATE_START, tz="UTC")
        end = pd.Timestamp(config.KALSHI_DATE_END, tz="UTC")
        mask = (
            df["close_time"].notna()
            & (df["close_time"] >= start)
            & (df["close_time"] <= end)
        )
        df = df[mask].reset_index(drop=True)
        print(f"  2. Close date in range:            {len(df)}")

        mask = (
            df["duration_days"].notna()
            & (df["duration_days"] >= config.MIN_DURATION_DAYS)
        )
        df = df[mask].reset_index(drop=True)
        print(f"  3. Duration >= {config.MIN_DURATION_DAYS} days:             {len(df)}")

        mask = df["is_economic"]
        df = df[mask].reset_index(drop=True)
        print(f"  4. Economic keyword match:         {len(df)}")
        print("--------------------------------\n")

        return df

    def _export_review_csv(self, df: pd.DataFrame) -> None:
        """Save a human-readable CSV for manual contract review."""
        cols = [
            "ticker", "title", "event_title", "series_ticker",
            "open_time", "close_time", "duration_days",
            "volume_fp", "is_fed", "is_economic",
        ]
        export = df[[c for c in cols if c in df.columns]]
        path = self.raw_dir / "contracts_for_review.csv"
        export.to_csv(path, index=False)
        print(f"  Review CSV saved to {path}")
