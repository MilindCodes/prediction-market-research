from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.kalshi.client import KalshiClient
import config


# ---------------------------------------------------------------------------
# Known Kalshi political series tickers (2022-2024).
# These are pulled individually — each one is fast (1-2 API calls max).
# The broad category=Elections filter is NOT used because the API ignores it
# and paginates through all events indefinitely.
# ---------------------------------------------------------------------------
POLITICAL_SERIES_TICKERS = [
    # 2022 Senate midterms
    "KXGARSENMATCHUP",   # Georgia Senate runoff (Warnock vs Walker)
    "SENATEAL",           # Alabama Senate
    "KXTXSENOUTCOME",     # Texas Senate
    "KXLASENMATCHUP",     # Louisiana Senate matchup
    "KXLASEN1R",          # Louisiana Senate first round
    "KXMOVILSENATED",     # Illinois Senate (Dems)
    "KXSNPMAJORITY",      # Senate majority (2022 midterms)
    # 2024 Senate / election-adjacent
    "KXTXSENATETURNOUTGOPMINUSDEM",
    "KXTXSENRPRIMARYMOV",
    "KXTXSENDPRIMARYMOV",
    "KXTXRSEN2ND",
    "KXTXRSEN3RD",
    "KXGAPRIMARY1R",      # Georgia Primary 2024
    "KXPRIMARYMATCHUP",
    "KXPRIMARYMOV",
    # State-level general elections (governors, swing states)
    "GOVPARTYMS",         # Mississippi Governor
    "GOVPARTYLA",         # Louisiana Governor
    "GOVPARTYKY",         # Kentucky Governor
]


class KalshiCatalog:
    """Pull and filter the Kalshi market catalog using targeted queries.

    Strategy:
      - FED/FOMC: series_ticker=KXFED (exact match, fast, ~500 markets total)
      - Political: curated list of known series tickers + capped fallback
        (max 5 pages of unsorted settled events, keyword-filtered)

    This avoids the category=Elections filter which the API ignores,
    causing infinite pagination.

    Parameters
    ----------
    client : KalshiClient
        API client (no key needed for reads).
    data_dir : Path
        Root data directory (default from config).
    """

    # Max pages for the fallback political search.
    # 5 pages x 200 events = 1000 events scanned in ~30 seconds max.
    POLITICAL_FALLBACK_MAX_PAGES = 5

    def __init__(self, client: KalshiClient,
                 data_dir: Path | None = None):
        self.client = client
        self.data_dir = data_dir or config.DATA_DIR
        self.raw_dir = self.data_dir / "raw" / "kalshi"
        self.raw_dir.mkdir(parents=True, exist_ok=True)

    def pull_catalog(self) -> pd.DataFrame:
        """Fetch FED and Political markets from the Kalshi Elections API
        and apply all filters.

        NOTE: The Kalshi Elections API only has data from ~2025 onwards
        (the platform was migrated from the old Kalshi trading API and the
        historical 2022-2024 data was not preserved). This catalog targets
        the 2025-2026 date window defined in config.KALSHI_DATE_START/END.

        Returns
        -------
        pd.DataFrame
            Filtered catalog ready for the liquidity filter step.
        """
        all_markets: list[dict] = []

        # ---------------------------------------------------------------
        # Pull 1: FOMC markets — series_ticker=KXFED
        # The Elections API has KXFED data from 2026 FOMC meetings.
        # status filter is IGNORED by the API; we filter client-side.
        # ---------------------------------------------------------------
        print("  [1/3] Fetching FOMC events (series_ticker=KXFED)...")
        fed_events = self.client.get(
            "/events",
            params={
                "series_ticker": "KXFED",
                "with_nested_markets": "true",
                "limit": 200,
            },
            progress_label="FOMC",
        )
        fed_markets = self._extract_markets(fed_events, is_fed=True)
        print(f"       -> {len(fed_events)} events, {len(fed_markets)} markets")
        all_markets.extend(fed_markets)

        # ---------------------------------------------------------------
        # Pull 2: Political — capped scan of ALL finalized events,
        # keyword-filtered. The API returns newest-first so we cap at
        # POLITICAL_FALLBACK_MAX_PAGES to keep only recent (2025-2026) ones.
        # ---------------------------------------------------------------
        print(f"\n  [2/3] Scanning for political events "
              f"(max {self.POLITICAL_FALLBACK_MAX_PAGES} pages)...")
        political_events_raw = self.client.get(
            "/events",
            params={
                "with_nested_markets": "true",
                "limit": 200,
            },
            max_pages=self.POLITICAL_FALLBACK_MAX_PAGES,
            progress_label="political scan",
        )
        pol_markets: list[dict] = []
        for event in political_events_raw:
            text = (
                str(event.get("title", "")).lower()
                + " " + str(event.get("series_ticker", "")).lower()
                + " " + str(event.get("category", "")).lower()
            )
            if any(kw in text for kw in config.POLITICAL_KEYWORDS):
                pol_markets.extend(
                    self._extract_markets([event], is_fed=False)
                )
        print(f"       -> {len(pol_markets)} keyword-matched political markets")
        all_markets.extend(pol_markets)

        # ---------------------------------------------------------------
        # Pull 3: Known current political series (2026 election cycles)
        # ---------------------------------------------------------------
        print(f"\n  [3/3] Fetching {len(POLITICAL_SERIES_TICKERS)} known "
              f"political series tickers...")
        pol_markets_from_series: list[dict] = []
        for ticker in POLITICAL_SERIES_TICKERS:
            events = self.client.get(
                "/events",
                params={
                    "series_ticker": ticker,
                    "with_nested_markets": "true",
                    "limit": 200,
                },
            )
            markets = self._extract_markets(events, is_fed=False)
            if markets:
                print(f"       {ticker}: {len(markets)} markets")
            pol_markets_from_series.extend(markets)
        print(f"       -> {len(pol_markets_from_series)} political markets "
              f"from known tickers")
        all_markets.extend(pol_markets_from_series)

        # ---------------------------------------------------------------
        # Deduplicate and save
        # ---------------------------------------------------------------
        if not all_markets:
            print("\nWARNING: No markets fetched. Check your internet connection.")
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
        """Tag each contract as is_fed and/or is_political."""
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
        df["is_political"] = text.apply(
            lambda t: any(kw in t for kw in config.POLITICAL_KEYWORDS)
        )
        return df

    def _apply_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        """Sequential attrition filters with counts printed at each step."""
        print("\n--- Attrition Table (Kalshi) ---")
        print(f"  0. Raw fetch (FED + Political):    {len(df)}")

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

        mask = df["is_fed"] | df["is_political"]
        df = df[mask].reset_index(drop=True)
        print(f"  4. Fed or Political keyword:       {len(df)}")
        print("--------------------------------\n")

        return df

    def _export_review_csv(self, df: pd.DataFrame) -> None:
        """Save a human-readable CSV for manual contract review."""
        cols = [
            "ticker", "title", "event_title", "series_ticker",
            "open_time", "close_time", "duration_days",
            "volume_fp", "is_fed", "is_political",
        ]
        export = df[[c for c in cols if c in df.columns]]
        path = self.raw_dir / "contracts_for_review.csv"
        export.to_csv(path, index=False)
        print(f"  Review CSV saved to {path}")
