from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.polymarket.gamma_client import GammaClient
import config


class PolymarketCatalog:
    """Pull, filter, and persist the Polymarket market catalog using the
    Gamma API (public, no authentication required).

    The Gamma API includes volumeNum in every market record, so the
    liquidity filter runs inline — no separate per-contract API calls.

    Implements the attrition pipeline described in the paper methodology:
    resolved → date range → minimum duration → thematic keyword match →
    liquidity filter (volumeNum >= threshold).

    Parameters
    ----------
    client : GammaClient
        Gamma API client (no auth needed).
    data_dir : Path
        Root data directory (default from config).
    """

    def __init__(self, client: GammaClient,
                 data_dir: Path | None = None):
        self.client = client
        self.data_dir = data_dir or config.DATA_DIR
        self.raw_dir = self.data_dir / "raw" / "polymarket"
        self.raw_dir.mkdir(parents=True, exist_ok=True)

    def pull_catalog(self) -> pd.DataFrame:
        """Fetch every resolved market from Polymarket via the Gamma API,
        apply all filters including liquidity, and save.

        Returns
        -------
        pd.DataFrame
            Fully filtered catalog ready for manual review. Includes
            trade_count column derived from volumeNum.
        """
        print("Pulling Polymarket market catalog via Gamma API...")
        print("(No API key needed — Gamma is public.)\n")

        # Pass the date range directly to the API so it skips pre-2022
        # markets server-side (saves ~hundreds of pages of 2020-2021 junk).
        # The early-stop kicks in once pages are past DATE_RANGE_END.
        markets = self.client.get_all_markets(
            closed="true",
            end_date_min=config.DATE_RANGE_START,
            end_date_max=config.DATE_RANGE_END,
        )
        if not markets:
            print("ERROR: No markets returned from Gamma API.")
            return pd.DataFrame()

        df = pd.DataFrame(markets)
        print(f"\n  Total markets fetched: {len(df)}")

        df.to_parquet(self.raw_dir / "catalog_full.parquet", index=False)

        df = self._normalize_fields(df)
        df = self._parse_timestamps(df)
        df = self._classify_keywords(df)
        df = self._apply_filters(df)

        df.to_parquet(self.raw_dir / "catalog_filtered.parquet", index=False)
        self._export_review_csv(df)

        return df

    def _normalize_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rename and extract Gamma API fields into a consistent schema.

        Gamma uses 'question' (not 'title'), 'volumeNum' (not 'volume'),
        and stores token IDs as a JSON string in 'clobTokenIds'.

        Parameters
        ----------
        df : pd.DataFrame
            Raw Gamma API response.

        Returns
        -------
        pd.DataFrame
            With normalized columns added.
        """
        df = df.copy()

        # volumeNum is the numeric trade volume — this replaces the
        # per-contract liquidity filter API calls entirely
        df["trade_count"] = pd.to_numeric(
            df.get("volumeNum", pd.Series(0, index=df.index)),
            errors="coerce"
        ).fillna(0).astype(int)

        # Preserve conditionId for linking to CLOB trade data later
        if "conditionId" not in df.columns:
            df["conditionId"] = df.get("condition_id", "")

        # Flag UMA oracle disputes.
        # The API returns umaResolutionStatuses as a JSON string.
        # '[]' = empty array = no dispute.  Non-empty array = real dispute.
        # We must parse the string rather than using bool() on it
        # (bool("[]") is True in Python, which was a false positive).
        import json

        def _is_disputed(x) -> bool:
            if x is None:
                return False
            if isinstance(x, list):
                return len(x) > 0
            if isinstance(x, str):
                stripped = x.strip()
                if stripped in ("", "[]", "{}", "null"):
                    return False
                try:
                    parsed = json.loads(stripped)
                    if isinstance(parsed, list):
                        return len(parsed) > 0
                    if isinstance(parsed, dict):
                        return bool(parsed)
                except Exception:
                    pass
                return True  # non-parseable non-empty string → treat as disputed
            return False

        if "umaResolutionStatuses" in df.columns:
            df["has_uma_dispute"] = df["umaResolutionStatuses"].apply(_is_disputed)
        else:
            df["has_uma_dispute"] = False

        return df

    def _parse_timestamps(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert Gamma timestamp fields to UTC-aware datetimes.

        Gamma uses 'startDateIso' / 'endDateIso' (or 'startDate' /
        'endDate' as fallback).

        Parameters
        ----------
        df : pd.DataFrame
            Catalog with Gamma timestamp fields.

        Returns
        -------
        pd.DataFrame
            With open_time, close_time, and duration_days.
        """
        df = df.copy()

        start_col = "startDateIso" if "startDateIso" in df.columns else "startDate"
        end_col = "endDateIso" if "endDateIso" in df.columns else "endDate"

        # Fall back to createdAt if startDate is missing
        if start_col not in df.columns:
            start_col = "createdAt"

        df["open_time"] = pd.to_datetime(df[start_col], utc=True, errors="coerce")
        df["close_time"] = pd.to_datetime(df[end_col], utc=True, errors="coerce")

        df["duration_days"] = (
            (df["close_time"] - df["open_time"]).dt.total_seconds()
            / 86400
        )
        return df

    def _classify_keywords(self, df: pd.DataFrame) -> pd.DataFrame:
        """Tag each contract as is_fed and/or is_political using keyword
        matching on question and category fields.

        The Gamma API includes a native 'category' field (e.g., 'Politics')
        which supplements keyword matching.

        Parameters
        ----------
        df : pd.DataFrame
            Catalog with question and category columns.

        Returns
        -------
        pd.DataFrame
            With boolean is_fed and is_political columns.
        """
        df = df.copy()

        text = (
            df.get("question", pd.Series("", index=df.index))
              .fillna("").str.lower()
            + " "
            + df.get("category", pd.Series("", index=df.index))
              .fillna("").str.lower()
        )

        df["is_fed"] = text.apply(
            lambda t: any(kw in t for kw in config.FED_KEYWORDS)
        )

        # Use both keyword matching AND native category field
        category_political = (
            df.get("category", pd.Series("", index=df.index))
              .fillna("").str.lower()
              .isin(["politics", "elections"])
        )
        keyword_political = text.apply(
            lambda t: any(kw in t for kw in config.POLITICAL_KEYWORDS)
        )
        df["is_political"] = category_political | keyword_political

        return df

    def _apply_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        """Sequential attrition filters with counts printed at each step.

        Unlike the old CLOB-based flow, the liquidity filter runs here
        (not in a separate step) because volumeNum is already in the data.

        Parameters
        ----------
        df : pd.DataFrame
            Full catalog with parsed timestamps, keyword flags, and
            trade_count.

        Returns
        -------
        pd.DataFrame
            Filtered catalog.
        """
        print("\n--- Attrition Table (Polymarket) ---")
        print(f"  0. Full catalog:                   {len(df)}")

        # Filter 1: Resolved contracts (closed == True)
        if "closed" in df.columns:
            mask = df["closed"] == True  # noqa: E712
            df = df[mask].reset_index(drop=True)
        print(f"  1. Resolved (closed==True):         {len(df)}")

        # Filter 2: Date range
        start = pd.Timestamp(config.DATE_RANGE_START, tz="UTC")
        end = pd.Timestamp(config.DATE_RANGE_END, tz="UTC")
        mask = (
            df["close_time"].notna()
            & (df["close_time"] >= start)
            & (df["close_time"] <= end)
        )
        df = df[mask].reset_index(drop=True)
        print(f"  2. Close date in range:            {len(df)}")

        # Filter 3: Minimum duration
        mask = df["duration_days"] >= config.MIN_DURATION_DAYS
        df = df[mask].reset_index(drop=True)
        print(f"  3. Duration >= {config.MIN_DURATION_DAYS} days:           {len(df)}")

        # Filter 4: Thematic keyword match
        mask = df["is_fed"] | df["is_political"]
        df = df[mask].reset_index(drop=True)
        print(f"  4. Fed or Political keyword:       {len(df)}")

        # Filter 5: Liquidity (runs inline — no extra API calls!)
        count_before = len(df)
        mask = df["trade_count"] >= config.MIN_TRADE_COUNT
        df = df[mask].reset_index(drop=True)
        print(f"  5. Volume >= {config.MIN_TRADE_COUNT} trades:        {len(df)}")

        # Flag: UMA oracle disputes
        if "has_uma_dispute" in df.columns:
            dispute_count = df["has_uma_dispute"].sum()
            if dispute_count > 0:
                print(f"  NOTE: {dispute_count} contracts have UMA oracle disputes — review these")

        print("------------------------------------\n")
        return df

    def _export_review_csv(self, df: pd.DataFrame) -> None:
        """Save a human-readable CSV for manual contract review.

        Parameters
        ----------
        df : pd.DataFrame
            Filtered catalog.
        """
        cols = [
            "conditionId", "question", "category", "open_time", "close_time",
            "duration_days", "trade_count", "is_fed", "is_political",
            "has_uma_dispute",
        ]
        export = df[[c for c in cols if c in df.columns]]
        path = self.raw_dir / "contracts_for_review.csv"
        export.to_csv(path, index=False)
        print(f"  Review CSV saved to {path}")
