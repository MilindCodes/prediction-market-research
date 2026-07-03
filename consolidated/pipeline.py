#!/usr/bin/env python3
"""
Prediction Market Research — consolidated single-file pipeline.

All data-pulling, model calibration, comparison, and output code
in one place. Run with:

    python pipeline.py <step>
    python pipeline.py help
"""
from __future__ import annotations

import ast
import json
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns
from scipy.optimize import brentq, minimize, minimize_scalar
from scipy.stats import chi2, norm
from tqdm import tqdm

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

KALSHI_API_KEY = os.environ.get("KALSHI_API_KEY", "")
POLYMARKET_API_KEY = os.environ.get("POLYMARKET_API_KEY", "")

POLYMARKET_DATE_START = "2022-01-01"
POLYMARKET_DATE_END = "2024-12-31"
KALSHI_DATE_START = "2025-01-01"
KALSHI_DATE_END = "2026-12-31"
DATE_RANGE_START = POLYMARKET_DATE_START
DATE_RANGE_END = POLYMARKET_DATE_END

MIN_DURATION_DAYS = 14
MIN_TRADE_COUNT = 500

DATA_DIR = Path(__file__).parent.parent / "data"

HOURLY_DT = 1.0 / (365.25 * 24)
DAILY_DT = 1.0 / 365.25

FED_KEYWORDS = [
    "fomc", "federal reserve", "fed funds", "fed rate",
    "interest rate", "rate hike", "rate cut", "rate increase", "rate decrease",
    "basis point", "bps", "powell", "federal open market",
]

POLITICAL_KEYWORDS = [
    "election", "president", "senate", "congress", "vote",
    "governor", "democrat", "republican", "primary", "midterm",
]

FOMC_DATES = [
    "2022-01-26T19:00:00Z", "2022-03-16T18:00:00Z", "2022-05-04T18:00:00Z",
    "2022-06-15T18:00:00Z", "2022-07-27T18:00:00Z", "2022-09-21T18:00:00Z",
    "2022-11-02T18:00:00Z", "2022-12-14T19:00:00Z",
    "2023-02-01T19:00:00Z", "2023-03-22T18:00:00Z", "2023-05-03T18:00:00Z",
    "2023-06-14T18:00:00Z", "2023-07-26T18:00:00Z", "2023-09-20T18:00:00Z",
    "2023-11-01T18:00:00Z", "2023-12-13T19:00:00Z",
    "2024-01-31T19:00:00Z", "2024-03-20T18:00:00Z", "2024-05-01T18:00:00Z",
    "2024-06-12T18:00:00Z", "2024-07-31T18:00:00Z", "2024-09-18T18:00:00Z",
    "2024-11-07T19:00:00Z", "2024-12-18T19:00:00Z",
]

POLITICAL_SERIES_TICKERS = [
    "KXGARSENMATCHUP", "SENATEAL", "KXTXSENOUTCOME", "KXLASENMATCHUP",
    "KXLASEN1R", "KXMOVILSENATED", "KXSNPMAJORITY",
    "KXTXSENATETURNOUTGOPMINUSDEM", "KXTXSENRPRIMARYMOV", "KXTXSENDPRIMARYMOV",
    "KXTXRSEN2ND", "KXTXRSEN3RD", "KXGAPRIMARY1R", "KXPRIMARYMATCHUP",
    "KXPRIMARYMOV", "GOVPARTYMS", "GOVPARTYLA", "GOVPARTYKY",
]

# ---------------------------------------------------------------------------
# SHARED UTILITY
# ---------------------------------------------------------------------------

def _json_default(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def _ensure_utc(t) -> pd.Timestamp:
    ts = pd.Timestamp(t)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


# ---------------------------------------------------------------------------
# DATA PULLING — KALSHI
# ---------------------------------------------------------------------------

class KalshiClient:
    BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"

    def __init__(self, api_key: str = "", base_url: str | None = None):
        self.base_url = (base_url or self.BASE_URL).rstrip("/")
        self.session = requests.Session()
        headers = {"Accept": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self.session.headers.update(headers)
        self._min_interval = 1.0
        self._last_request_time = 0.0

    def _rate_limit(self) -> None:
        elapsed = time.monotonic() - self._last_request_time
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)

    def get(self, endpoint: str, params: dict | None = None,
            max_retries: int = 5, max_pages: int | None = None,
            progress_label: str = "") -> list[dict]:
        url = f"{self.base_url}{endpoint}"
        params = dict(params or {})
        all_items: list[dict] = []
        page = 0

        while True:
            page += 1
            if max_pages and page > max_pages:
                print(f"  [{progress_label or endpoint}] Reached page limit ({max_pages}), stopping.")
                break
            if progress_label and page % 5 == 1:
                print(f"  [{progress_label}] Fetching page {page}... ({len(all_items)} so far)", end="\r")

            data = self._request_with_retry(url, params, max_retries)
            items_key = self._find_items_key(data)
            if items_key and isinstance(data[items_key], list):
                all_items.extend(data[items_key])

            cursor = data.get("cursor")
            if not cursor:
                break
            params["cursor"] = cursor

        if progress_label:
            print(f"  [{progress_label}] Done — {len(all_items)} items fetched.    ")
        return all_items

    def get_single(self, endpoint: str, params: dict | None = None,
                   max_retries: int = 5) -> dict:
        url = f"{self.base_url}{endpoint}"
        return self._request_with_retry(url, params or {}, max_retries)

    def _request_with_retry(self, url: str, params: dict, max_retries: int) -> dict:
        backoff = 2.0
        for attempt in range(max_retries + 1):
            self._rate_limit()
            self._last_request_time = time.monotonic()
            try:
                resp = self.session.get(url, params=params, timeout=30)
            except requests.exceptions.ConnectionError as e:
                if attempt < max_retries:
                    print(f"\n[KalshiClient] Network error (attempt {attempt+1}/{max_retries}), retrying in {backoff:.0f}s...")
                    time.sleep(backoff); backoff = min(backoff * 2, 30); continue
                raise
            except requests.exceptions.Timeout:
                if attempt < max_retries:
                    print(f"\n[KalshiClient] Timeout (attempt {attempt+1}/{max_retries}), retrying in {backoff:.0f}s...")
                    time.sleep(backoff); backoff = min(backoff * 2, 30); continue
                raise

            if resp.status_code == 200:
                return resp.json()
            if resp.status_code in (429, 500, 502, 503) and attempt < max_retries:
                print(f"\n[KalshiClient] HTTP {resp.status_code} (attempt {attempt+1}/{max_retries}), retrying in {backoff:.0f}s...")
                time.sleep(backoff); backoff = min(backoff * 2, 30); continue

            print(f"\n[KalshiClient] ERROR {resp.status_code} on {url}: {resp.text[:200]}")
            resp.raise_for_status()
        return {}

    @staticmethod
    def _find_items_key(data: dict) -> str | None:
        for key, val in data.items():
            if key not in ("cursor", "milestones") and isinstance(val, list):
                return key
        return None


class KalshiCatalog:
    POLITICAL_FALLBACK_MAX_PAGES = 5

    def __init__(self, client: KalshiClient, data_dir: Path | None = None):
        self.client = client
        self.data_dir = data_dir or DATA_DIR
        self.raw_dir = self.data_dir / "raw" / "kalshi"
        self.raw_dir.mkdir(parents=True, exist_ok=True)

    def pull_catalog(self) -> pd.DataFrame:
        all_markets: list[dict] = []

        print("  [1/3] Fetching FOMC events (series_ticker=KXFED)...")
        fed_events = self.client.get(
            "/events",
            params={"series_ticker": "KXFED", "with_nested_markets": "true", "limit": 200},
            progress_label="FOMC",
        )
        fed_markets = self._extract_markets(fed_events, is_fed=True)
        print(f"       -> {len(fed_events)} events, {len(fed_markets)} markets")
        all_markets.extend(fed_markets)

        print(f"\n  [2/3] Scanning for political events (max {self.POLITICAL_FALLBACK_MAX_PAGES} pages)...")
        political_events_raw = self.client.get(
            "/events",
            params={"with_nested_markets": "true", "limit": 200},
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
            if any(kw in text for kw in POLITICAL_KEYWORDS):
                pol_markets.extend(self._extract_markets([event], is_fed=False))
        print(f"       -> {len(pol_markets)} keyword-matched political markets")
        all_markets.extend(pol_markets)

        print(f"\n  [3/3] Fetching {len(POLITICAL_SERIES_TICKERS)} known political series tickers...")
        pol_markets_from_series: list[dict] = []
        for ticker in POLITICAL_SERIES_TICKERS:
            events = self.client.get(
                "/events",
                params={"series_ticker": ticker, "with_nested_markets": "true", "limit": 200},
            )
            markets = self._extract_markets(events, is_fed=False)
            if markets:
                print(f"       {ticker}: {len(markets)} markets")
            pol_markets_from_series.extend(markets)
        print(f"       -> {len(pol_markets_from_series)} political markets from known tickers")
        all_markets.extend(pol_markets_from_series)

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
        threshold = min_volume if min_volume is not None else MIN_TRADE_COUNT
        if "volume_fp" not in df.columns:
            print("  WARNING: volume_fp column not found — skipping liquidity filter")
            self._export_review_csv(df)
            return df

        count_before = len(df)
        df = df[df["volume_fp"] >= threshold].reset_index(drop=True)
        print(f"  Liquidity filter (volume_fp >= {threshold}): {count_before} -> {len(df)}")
        self._export_review_csv(df)
        df.to_parquet(self.raw_dir / "catalog_filtered.parquet", index=False)
        return df

    def _extract_markets(self, events: list[dict], is_fed: bool) -> list[dict]:
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
        df = df.copy()
        open_raw = df["open_time"] if "open_time" in df.columns else df.get("created_time", pd.Series("", index=df.index))
        close_raw = df["close_time"] if "close_time" in df.columns else df.get("expiration_time", pd.Series("", index=df.index))
        df["open_time"] = pd.to_datetime(open_raw, utc=True, errors="coerce")
        df["close_time"] = pd.to_datetime(close_raw, utc=True, errors="coerce")
        df["duration_days"] = (df["close_time"] - df["open_time"]).dt.total_seconds() / 86400
        return df

    def _classify_keywords(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        text = (
            df.get("title", pd.Series("", index=df.index)).fillna("").str.lower()
            + " " + df.get("event_title", pd.Series("", index=df.index)).fillna("").str.lower()
            + " " + df.get("series_ticker", pd.Series("", index=df.index)).fillna("").str.lower()
        )
        df["is_fed"] = (
            df.get("_is_fed_source", pd.Series(False, index=df.index))
            | text.apply(lambda t: any(kw in t for kw in FED_KEYWORDS))
        )
        df["is_political"] = text.apply(lambda t: any(kw in t for kw in POLITICAL_KEYWORDS))
        return df

    def _apply_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        print("\n--- Attrition Table (Kalshi) ---")
        print(f"  0. Raw fetch (FED + Political):    {len(df)}")
        df = df[df["status"].isin(["finalized", "settled"])].reset_index(drop=True)
        print(f"  1. Status finalized/settled:       {len(df)}")
        start = pd.Timestamp(KALSHI_DATE_START, tz="UTC")
        end = pd.Timestamp(KALSHI_DATE_END, tz="UTC")
        mask = df["close_time"].notna() & (df["close_time"] >= start) & (df["close_time"] <= end)
        df = df[mask].reset_index(drop=True)
        print(f"  2. Close date in range:            {len(df)}")
        df = df[df["duration_days"].notna() & (df["duration_days"] >= MIN_DURATION_DAYS)].reset_index(drop=True)
        print(f"  3. Duration >= {MIN_DURATION_DAYS} days:             {len(df)}")
        df = df[df["is_fed"] | df["is_political"]].reset_index(drop=True)
        print(f"  4. Fed or Political keyword:       {len(df)}")
        print("--------------------------------\n")
        return df

    def _export_review_csv(self, df: pd.DataFrame) -> None:
        cols = ["ticker", "title", "event_title", "series_ticker",
                "open_time", "close_time", "duration_days", "volume_fp", "is_fed", "is_political"]
        export = df[[c for c in cols if c in df.columns]]
        path = self.raw_dir / "contracts_for_review.csv"
        export.to_csv(path, index=False)
        print(f"  Review CSV saved to {path}")


class KalshiTradesPuller:
    def __init__(self, client: KalshiClient, data_dir: Path | None = None):
        self.client = client
        self.data_dir = data_dir or DATA_DIR
        self.raw_dir = self.data_dir / "raw" / "kalshi"
        self.raw_dir.mkdir(parents=True, exist_ok=True)

    def pull(self, tickers: list[str]) -> None:
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
                    failures.append(ticker); continue
                bars = self._aggregate_hourly(df)
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
        trades = self.client.get("/markets/trades", params={"ticker": ticker, "limit": 1000})
        if not trades:
            return pd.DataFrame()
        df = pd.DataFrame(trades)
        df["created_time"] = pd.to_datetime(df["created_time"], utc=True)
        return df

    @staticmethod
    def _aggregate_hourly(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["hour"] = df["created_time"].dt.floor("h")

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
            return float(np.nanmean(prices)) if total_w == 0 else float(np.nansum(prices * weights) / total_w)

        bars = pd.DataFrame({
            "timestamp": grouped["created_time"].first().apply(lambda t: t.floor("h")),
            "open": grouped["yes_price"].first(),
            "high": grouped["yes_price"].max(),
            "low": grouped["yes_price"].min(),
            "close": grouped.apply(vwap, include_groups=False),
            "volume": grouped["count"].sum(),
            "trade_count": grouped.size(),
        }).reset_index(drop=True)

        return bars.sort_values("timestamp").reset_index(drop=True)


# ---------------------------------------------------------------------------
# DATA PULLING — POLYMARKET
# ---------------------------------------------------------------------------

class GammaClient:
    BASE_URL = "https://gamma-api.polymarket.com"
    PAGE_SIZE = 20

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({"Accept": "application/json"})
        self._min_interval = 0.5
        self._last_request_time = 0.0

    def _rate_limit(self) -> None:
        elapsed = time.monotonic() - self._last_request_time
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)

    def get_all_markets(self, end_date_min: str | None = None,
                        end_date_max: str | None = None, **filters) -> list[dict]:
        all_items: list[dict] = []
        offset = 0
        use_keyset = False
        next_cursor: str | None = None

        base_params = dict(filters)
        if end_date_min:
            base_params["end_date_min"] = end_date_min

        while True:
            if use_keyset:
                params = dict(base_params)
                params["limit"] = self.PAGE_SIZE
                if next_cursor:
                    params["next_cursor"] = next_cursor
                data = self._request("/markets/keyset", params)
                if isinstance(data, dict):
                    next_cursor = data.get("next_cursor") or data.get("cursor")
                    data = data.get("data", data.get("markets", []))
            else:
                params = dict(base_params)
                params["offset"] = offset
                params["limit"] = self.PAGE_SIZE
                data = self._request("/markets", params)
                if data is None:
                    use_keyset = True
                    next_cursor = None
                    print(f"\n  Switching to keyset pagination at {len(all_items)} markets...")
                    continue

            if not data or not isinstance(data, list):
                break

            all_items.extend(data)
            print(f"  Fetched {len(all_items)} markets so far...", end="\r")

            if end_date_max and data:
                end_dates = [(d.get("endDateIso") or d.get("endDate") or "")[:10] for d in data]
                if all(ed > end_date_max for ed in end_dates if ed):
                    break

            if len(data) < self.PAGE_SIZE:
                break

            if use_keyset:
                if not next_cursor:
                    break
            else:
                offset += self.PAGE_SIZE

        print(f"  Fetched {len(all_items)} markets total.        ")
        return all_items

    def _request(self, endpoint: str, params: dict, max_retries: int = 5) -> list | dict | None:
        url = f"{self.BASE_URL}{endpoint}"
        backoff = 2.0
        for attempt in range(max_retries + 1):
            self._rate_limit()
            self._last_request_time = time.monotonic()
            try:
                resp = self.session.get(url, params=params, timeout=30)
                if resp.status_code == 200:
                    return resp.json()
                if resp.status_code in (429, 500, 502, 503) and attempt < max_retries:
                    print(f"\n  [GammaClient] HTTP {resp.status_code} (attempt {attempt+1}/{max_retries}), retrying in {backoff:.0f}s...")
                    time.sleep(backoff); backoff = min(backoff * 2, 30); continue
                print(f"\n  [GammaClient] ERROR {resp.status_code} on {endpoint}: {resp.text[:200]}")
                return None
            except Exception as e:
                if attempt < max_retries:
                    print(f"\n  [GammaClient] Network error (attempt {attempt+1}/{max_retries}), retrying in {backoff:.0f}s...")
                    time.sleep(backoff); backoff = min(backoff * 2, 30); continue
                print(f"\n  [GammaClient] Failed after {max_retries} retries: {e}")
                return None
        return None


class PolymarketCatalog:
    def __init__(self, client: GammaClient, data_dir: Path | None = None):
        self.client = client
        self.data_dir = data_dir or DATA_DIR
        self.raw_dir = self.data_dir / "raw" / "polymarket"
        self.raw_dir.mkdir(parents=True, exist_ok=True)

    def pull_catalog(self) -> pd.DataFrame:
        print("Pulling Polymarket market catalog via Gamma API...")
        markets = self.client.get_all_markets(
            closed="true",
            end_date_min=DATE_RANGE_START,
            end_date_max=DATE_RANGE_END,
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
        df = df.copy()
        df["trade_count"] = pd.to_numeric(
            df.get("volumeNum", pd.Series(0, index=df.index)), errors="coerce"
        ).fillna(0).astype(int)

        if "conditionId" not in df.columns:
            df["conditionId"] = df.get("condition_id", "")

        def _is_disputed(x) -> bool:
            if x is None: return False
            if isinstance(x, list): return len(x) > 0
            if isinstance(x, str):
                stripped = x.strip()
                if stripped in ("", "[]", "{}", "null"): return False
                try:
                    parsed = json.loads(stripped)
                    if isinstance(parsed, list): return len(parsed) > 0
                    if isinstance(parsed, dict): return bool(parsed)
                except Exception:
                    pass
                return True
            return False

        if "umaResolutionStatuses" in df.columns:
            df["has_uma_dispute"] = df["umaResolutionStatuses"].apply(_is_disputed)
        else:
            df["has_uma_dispute"] = False
        return df

    def _parse_timestamps(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        start_col = "startDateIso" if "startDateIso" in df.columns else "startDate"
        end_col = "endDateIso" if "endDateIso" in df.columns else "endDate"
        if start_col not in df.columns:
            start_col = "createdAt"
        df["open_time"] = pd.to_datetime(df[start_col], utc=True, errors="coerce")
        df["close_time"] = pd.to_datetime(df[end_col], utc=True, errors="coerce")
        df["duration_days"] = (df["close_time"] - df["open_time"]).dt.total_seconds() / 86400
        return df

    def _classify_keywords(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        text = (
            df.get("question", pd.Series("", index=df.index)).fillna("").str.lower()
            + " " + df.get("category", pd.Series("", index=df.index)).fillna("").str.lower()
        )
        df["is_fed"] = text.apply(lambda t: any(kw in t for kw in FED_KEYWORDS))
        category_political = (
            df.get("category", pd.Series("", index=df.index)).fillna("").str.lower().isin(["politics", "elections"])
        )
        keyword_political = text.apply(lambda t: any(kw in t for kw in POLITICAL_KEYWORDS))
        df["is_political"] = category_political | keyword_political
        return df

    def _apply_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        print("\n--- Attrition Table (Polymarket) ---")
        print(f"  0. Full catalog:                   {len(df)}")
        if "closed" in df.columns:
            df = df[df["closed"] == True].reset_index(drop=True)  # noqa: E712
        print(f"  1. Resolved (closed==True):         {len(df)}")
        start = pd.Timestamp(DATE_RANGE_START, tz="UTC")
        end = pd.Timestamp(DATE_RANGE_END, tz="UTC")
        mask = df["close_time"].notna() & (df["close_time"] >= start) & (df["close_time"] <= end)
        df = df[mask].reset_index(drop=True)
        print(f"  2. Close date in range:            {len(df)}")
        df = df[df["duration_days"] >= MIN_DURATION_DAYS].reset_index(drop=True)
        print(f"  3. Duration >= {MIN_DURATION_DAYS} days:           {len(df)}")
        df = df[df["is_fed"] | df["is_political"]].reset_index(drop=True)
        print(f"  4. Fed or Political keyword:       {len(df)}")
        df = df[df["trade_count"] >= MIN_TRADE_COUNT].reset_index(drop=True)
        print(f"  5. Volume >= {MIN_TRADE_COUNT} trades:        {len(df)}")
        if "has_uma_dispute" in df.columns:
            dispute_count = df["has_uma_dispute"].sum()
            if dispute_count > 0:
                print(f"  NOTE: {dispute_count} contracts have UMA oracle disputes — review these")
        print("------------------------------------\n")
        return df

    def _export_review_csv(self, df: pd.DataFrame) -> None:
        cols = ["conditionId", "question", "category", "open_time", "close_time",
                "duration_days", "trade_count", "is_fed", "is_political", "has_uma_dispute"]
        export = df[[c for c in cols if c in df.columns]]
        path = self.raw_dir / "contracts_for_review.csv"
        export.to_csv(path, index=False)
        print(f"  Review CSV saved to {path}")


class PolymarketClient:
    BASE_URL = "https://clob.polymarket.com"

    def __init__(self, api_key: str, base_url: str | None = None):
        self.base_url = (base_url or self.BASE_URL).rstrip("/")
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"Bearer {api_key}", "Accept": "application/json"})
        self._min_interval = 1.0
        self._last_request_time = 0.0

    def _rate_limit(self) -> None:
        elapsed = time.monotonic() - self._last_request_time
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)

    def get(self, endpoint: str, params: dict | None = None, max_retries: int = 5) -> list[dict]:
        url = f"{self.base_url}{endpoint}"
        params = dict(params or {})
        all_items: list[dict] = []

        while True:
            data = self._request_with_retry(url, params, max_retries)
            if isinstance(data, list):
                all_items.extend(data); break

            items_key = self._find_items_key(data)
            if items_key and isinstance(data[items_key], list):
                all_items.extend(data[items_key])

            cursor = data.get("next_cursor")
            if not cursor or cursor == "LTE=":
                break
            params["next_cursor"] = cursor

        return all_items

    def _request_with_retry(self, url: str, params: dict, max_retries: int) -> dict | list:
        backoff = 2.0
        for attempt in range(max_retries + 1):
            self._rate_limit()
            self._last_request_time = time.monotonic()
            try:
                resp = self.session.get(url, params=params, timeout=30)
            except Exception as e:
                if attempt < max_retries:
                    print(f"\n[PolymarketClient] Network error (attempt {attempt+1}/{max_retries}), retrying in {backoff:.0f}s...")
                    time.sleep(backoff); backoff = min(backoff * 2, 30); continue
                raise

            if resp.status_code == 200:
                return resp.json()
            if resp.status_code in (429, 500, 502, 503) and attempt < max_retries:
                print(f"\n[PolymarketClient] HTTP {resp.status_code} (attempt {attempt+1}/{max_retries}), retrying in {backoff:.0f}s...")
                time.sleep(backoff); backoff = min(backoff * 2, 30); continue

            print(f"\n[PolymarketClient] ERROR {resp.status_code} on {url}: {resp.text[:200]}")
            resp.raise_for_status()
        return {}

    @staticmethod
    def _find_items_key(data: dict) -> str | None:
        for key, val in data.items():
            if key not in ("next_cursor", "cursor") and isinstance(val, list):
                return key
        return None


_FIDELITY_MINUTES = 1440


class PolymarketTradesPuller:
    def __init__(self, client: PolymarketClient, data_dir: Path | None = None):
        self.client = client
        self.data_dir = data_dir or DATA_DIR
        self.raw_dir = self.data_dir / "raw" / "polymarket"
        self.raw_dir.mkdir(parents=True, exist_ok=True)

    def pull(self, contracts: list[dict]) -> None:
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
                    failures.append(cid); continue
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
        history = self.client.get(
            "/prices-history",
            params={"market": yes_token, "interval": "all", "fidelity": _FIDELITY_MINUTES},
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
        result = pd.DataFrame({
            "timestamp":   df["timestamp"],
            "open":        df["close"],
            "high":        df["close"],
            "low":         df["close"],
            "close":       df["close"],
            "volume":      0.0,
            "trade_count": 0,
        })
        return result.sort_values("timestamp").reset_index(drop=True)


# ---------------------------------------------------------------------------
# MODELS — IMPLIED VOL EXTRACTION
# ---------------------------------------------------------------------------

class ImpliedVolExtractor:
    STRIKE = 0.50
    RISK_FREE_RATE = 0.0
    SIGMA_LOWER = 1e-6
    SIGMA_UPPER = 20.0
    BOUNDARY_LOW = 0.05
    BOUNDARY_HIGH = 0.95
    FOMC_WINDOW_MINUTES = 30

    def __init__(self, close_time: pd.Timestamp,
                 fomc_dates: list[pd.Timestamp] | None = None,
                 data_dir: Path | None = None):
        self.close_time = _ensure_utc(close_time)
        self.fomc_dates = [_ensure_utc(d) for d in (fomc_dates or [])]
        self.data_dir = data_dir or DATA_DIR
        self.out_dir = self.data_dir / "processed" / "implied_vol"
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def extract(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
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
        delta = (self.close_time - timestamp).total_seconds()
        return 0.0 if delta <= 0 else delta / (365.25 * 86400)

    def _solve_implied_vol(self, price: float, T: float) -> float:
        if T <= 0 or not (1e-6 < price < 1 - 1e-6):
            return np.nan
        S = 1.0
        K = self.STRIKE
        r = self.RISK_FREE_RATE

        def objective(sigma: float) -> float:
            d2 = (np.log(S / K) + (r - 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
            return np.exp(-r * T) * norm.cdf(d2) - price

        try:
            return float(brentq(objective, self.SIGMA_LOWER, self.SIGMA_UPPER, xtol=1e-10, maxiter=200))
        except (ValueError, RuntimeError):
            return np.nan

    def _in_fomc_window(self, timestamp: pd.Timestamp) -> bool:
        window = pd.Timedelta(minutes=self.FOMC_WINDOW_MINUTES)
        return any(abs(timestamp - fomc_dt) <= window for fomc_dt in self.fomc_dates)


# ---------------------------------------------------------------------------
# MODELS — BLACK-SCHOLES CALIBRATOR
# ---------------------------------------------------------------------------

@dataclass
class BSResult:
    ticker: str
    platform: str
    sigma_bs: float
    mse: float
    log_likelihood: float
    n_obs: int
    converged: bool


class BSCalibrator:
    def __init__(self, data_dir: Path | None = None):
        self.data_dir = data_dir or DATA_DIR
        self.out_dir = self.data_dir / "processed" / "bs_params"
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def calibrate(self, sigma_series: np.ndarray, ticker: str, platform: str) -> BSResult:
        sigma_clean = sigma_series[np.isfinite(sigma_series)]
        if len(sigma_clean) < 2:
            return self._empty_result(ticker, platform, len(sigma_clean))
        var_obs = sigma_clean ** 2
        result = minimize_scalar(
            lambda s: float(np.mean((var_obs - s ** 2) ** 2)),
            bounds=(1e-6, 10.0), method="bounded",
        )
        sigma_bs = float(result.x)
        var_bs = sigma_bs ** 2
        mse = float(np.mean((var_obs - var_bs) ** 2))
        ll = self._log_likelihood(var_obs, var_bs)
        bs_result = BSResult(ticker=ticker, platform=platform, sigma_bs=sigma_bs,
                             mse=mse, log_likelihood=ll, n_obs=len(var_obs),
                             converged=bool(result.success))
        self._save(bs_result)
        return bs_result

    @staticmethod
    def _log_likelihood(var_obs: np.ndarray, var_const: float) -> float:
        residuals = var_obs - var_const
        n = len(residuals)
        sigma2 = max(float(np.var(residuals)), 1e-30)
        return float(-0.5 * n * np.log(2 * np.pi * sigma2) - np.sum(residuals ** 2) / (2 * sigma2))

    def _empty_result(self, ticker: str, platform: str, n_obs: int) -> BSResult:
        return BSResult(ticker=ticker, platform=platform, sigma_bs=np.nan,
                        mse=np.nan, log_likelihood=np.nan, n_obs=n_obs, converged=False)

    def _save(self, result: BSResult) -> None:
        path = self.out_dir / f"{result.ticker}.json"
        with open(path, "w") as f:
            json.dump(asdict(result), f, indent=2, default=_json_default)


# ---------------------------------------------------------------------------
# MODELS — HESTON CALIBRATOR
# ---------------------------------------------------------------------------

_log = logging.getLogger(__name__)


@dataclass
class HestonResult:
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
    PARAM_BOUNDS = [
        (0.01, 50.0), (1e-6, 10.0), (0.01, 5.0), (-0.999, 0.999), (1e-6, 5.0),
    ]
    PARAM_NAMES = ["kappa", "theta", "xi", "rho", "v0"]

    def __init__(self, data_dir: Path | None = None):
        self.data_dir = data_dir or DATA_DIR
        self.out_dir = self.data_dir / "processed" / "heston_params"
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def calibrate(self, sigma_series: np.ndarray, ticker: str,
                  platform: str, dt: float = 1 / 252) -> HestonResult:
        sigma_clean = sigma_series[np.isfinite(sigma_series)]
        if len(sigma_clean) < 10:
            return self._empty_result(ticker, platform, len(sigma_clean))

        var_observed = sigma_clean ** 2
        x0 = self._initial_guess(var_observed)
        result = minimize(
            self._objective, x0, args=(var_observed, dt),
            method="L-BFGS-B", bounds=self.PARAM_BOUNDS,
            options={"maxiter": 2000, "ftol": 1e-12},
        )

        kappa, theta, xi, rho, v0 = result.x
        feller = 2 * kappa * theta >= xi ** 2
        if not feller:
            _log.warning("Heston: %s — Feller violated", ticker)

        for name, val, (lo, hi) in zip(self.PARAM_NAMES, result.x, self.PARAM_BOUNDS):
            tol = 0.01 * (hi - lo)
            if val <= lo + tol or val >= hi - tol:
                _log.warning("Heston: %s — param %s at bound", ticker, name)

        var_model = self._simulate_variance_path(result.x, len(var_observed), dt)
        mse = float(np.mean((var_observed - var_model) ** 2))
        ll = self._log_likelihood(var_observed, var_model)

        heston_result = HestonResult(
            ticker=ticker, platform=platform,
            kappa=float(kappa), theta=float(theta), xi=float(xi),
            rho=float(rho), v0=float(v0),
            feller_satisfied=bool(feller), converged=bool(result.success),
            mse=mse, log_likelihood=ll, n_obs=len(var_observed),
        )
        self._save(heston_result)
        return heston_result

    def _objective(self, params: np.ndarray, var_observed: np.ndarray, dt: float) -> float:
        kappa, theta, xi = params[0], params[1], params[2]
        feller_slack = 2 * kappa * theta - xi ** 2
        penalty = 1e4 * feller_slack ** 2 if feller_slack < 0 else 0.0
        var_model = self._simulate_variance_path(params, len(var_observed), dt)
        return float(np.sum((var_observed - var_model) ** 2)) + penalty

    @staticmethod
    def _simulate_variance_path(params: np.ndarray, n: int, dt: float) -> np.ndarray:
        kappa, theta, _, _, v0 = params
        v = np.empty(n)
        v[0] = v0
        for i in range(1, n):
            v[i] = max(v[i - 1] + kappa * (theta - v[i - 1]) * dt, 1e-10)
        return v

    @staticmethod
    def _log_likelihood(var_observed: np.ndarray, var_model: np.ndarray) -> float:
        residuals = var_observed - var_model
        n = len(residuals)
        sigma2 = max(float(np.var(residuals)), 1e-30)
        return float(-0.5 * n * np.log(2 * np.pi * sigma2) - np.sum(residuals ** 2) / (2 * sigma2))

    @staticmethod
    def _initial_guess(var_observed: np.ndarray) -> np.ndarray:
        theta0 = float(np.mean(var_observed))
        v0 = float(var_observed[0])
        xi0 = float(np.std(np.diff(var_observed)))
        return np.array([1.0, max(theta0, 1e-5), max(xi0, 0.05), -0.5, max(v0, 1e-5)])

    def _empty_result(self, ticker: str, platform: str, n_obs: int) -> HestonResult:
        return HestonResult(
            ticker=ticker, platform=platform,
            kappa=np.nan, theta=np.nan, xi=np.nan, rho=np.nan, v0=np.nan,
            feller_satisfied=False, converged=False,
            mse=np.nan, log_likelihood=np.nan, n_obs=n_obs,
        )

    def _save(self, result: HestonResult) -> None:
        path = self.out_dir / f"{result.ticker}.json"
        with open(path, "w") as f:
            json.dump(asdict(result), f, indent=2, default=_json_default)


# ---------------------------------------------------------------------------
# MODELS — BATES CALIBRATOR
# ---------------------------------------------------------------------------

@dataclass
class BatesResult:
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
    BATES_BOUNDS = [
        (0.01, 50.0), (1e-6, 10.0), (0.01, 5.0), (-0.999, 0.999), (1e-6, 5.0),
        (0.0, 100.0), (-5.0, 5.0), (0.001, 3.0),
    ]
    BATES_PARAM_NAMES = ["kappa", "theta", "xi", "rho", "v0", "lambda_j", "mu_j", "sigma_j"]
    JUMP_DOF = 3
    BATES_INIT_GRID = [
        (1.0, 0.5, 0.5, -0.3, 0.1,  2.0,  0.0, 0.3),
        (1.0, 0.5, 0.5, -0.3, 0.1,  2.0, +0.5, 0.3),
        (1.0, 0.5, 0.5, -0.3, 0.1,  2.0, -0.5, 0.3),
        (5.0, 1.0, 1.0,  0.0, 0.5, 10.0, +1.0, 0.5),
        (0.5, 2.0, 0.3, +0.5, 0.2,  5.0, -1.0, 0.8),
    ]

    def __init__(self, data_dir: Path | None = None):
        super().__init__(data_dir)
        self.out_dir = (data_dir or DATA_DIR) / "processed" / "bates_params"
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def calibrate(self, sigma_series: np.ndarray, ticker: str, platform: str,
                  dt: float = 1 / 252, heston_result: HestonResult | None = None) -> BatesResult:
        sigma_clean = sigma_series[np.isfinite(sigma_series)]
        if len(sigma_clean) < 10:
            return self._empty_bates_result(ticker, platform, len(sigma_clean))

        var_observed = sigma_clean ** 2
        if heston_result is None:
            heston_result = super().calibrate(sigma_series, ticker, platform, dt)

        heston_x0 = self._bates_initial_guess(var_observed, heston_result)
        init_points = [heston_x0] + [np.array(x0) for x0 in self.BATES_INIT_GRID]

        best_result = None
        best_obj = np.inf
        for x0 in init_points:
            res = minimize(
                self._bates_objective, x0, args=(var_observed, dt),
                method="L-BFGS-B", bounds=self.BATES_BOUNDS,
                options={"maxiter": 3000, "ftol": 1e-12},
            )
            if res.fun < best_obj:
                best_obj = res.fun
                best_result = res

        result = best_result
        kappa, theta, xi, rho, v0, lambda_j, mu_j, sigma_j = result.x
        feller = 2 * kappa * theta >= xi ** 2
        if not feller:
            _log.warning("Bates: %s — Feller violated", ticker)

        for name, val, (lo, hi) in zip(self.BATES_PARAM_NAMES, result.x, self.BATES_BOUNDS):
            tol = 0.01 * (hi - lo)
            if val <= lo + tol or val >= hi - tol:
                _log.warning("Bates: %s — param %s at bound", ticker, name)

        var_model = self._simulate_bates_variance_path(result.x, len(var_observed), dt)
        mse = float(np.mean((var_observed - var_model) ** 2))
        ll_bates = self._log_likelihood(var_observed, var_model)
        ll_heston = heston_result.log_likelihood

        lr_stat = max(2 * (ll_bates - ll_heston), 0.0)
        p_value = float(chi2.sf(lr_stat, self.JUMP_DOF))

        bates_result = BatesResult(
            ticker=ticker, platform=platform,
            kappa=float(kappa), theta=float(theta), xi=float(xi),
            rho=float(rho), v0=float(v0),
            lambda_j=float(lambda_j), mu_j=float(mu_j), sigma_j=float(sigma_j),
            feller_satisfied=bool(feller), converged=bool(result.success),
            mse=mse, log_likelihood=ll_bates, log_likelihood_heston=ll_heston,
            lr_statistic=float(lr_stat), p_value_jump_significance=p_value,
            n_obs=len(var_observed),
        )
        self._save_bates(bates_result)
        return bates_result

    def _bates_objective(self, params: np.ndarray, var_observed: np.ndarray, dt: float) -> float:
        kappa, theta, xi = params[0], params[1], params[2]
        feller_slack = 2 * kappa * theta - xi ** 2
        penalty = 1e4 * feller_slack ** 2 if feller_slack < 0 else 0.0
        var_model = self._simulate_bates_variance_path(params, len(var_observed), dt)
        return float(np.sum((var_observed - var_model) ** 2)) + penalty

    @staticmethod
    def _simulate_bates_variance_path(params: np.ndarray, n: int, dt: float) -> np.ndarray:
        kappa, theta, _, _, v0, lambda_j, mu_j, sigma_j = params
        jump_var_contribution = lambda_j * (mu_j ** 2 + sigma_j ** 2)
        v = np.empty(n)
        v[0] = v0
        for i in range(1, n):
            v[i] = max(v[i - 1] + kappa * (theta - v[i - 1]) * dt + jump_var_contribution * dt, 1e-10)
        return v

    @staticmethod
    def _bates_initial_guess(var_observed: np.ndarray, heston_result: HestonResult) -> np.ndarray:
        if np.isfinite(heston_result.kappa):
            heston_params = [heston_result.kappa, heston_result.theta,
                             heston_result.xi, heston_result.rho, heston_result.v0]
        else:
            theta0 = float(np.mean(var_observed))
            v0 = float(var_observed[0])
            xi0 = float(np.std(np.diff(var_observed)))
            heston_params = [1.0, max(theta0, 1e-5), max(xi0, 0.05), -0.5, max(v0, 1e-5)]
        return np.array(heston_params + [1.0, 0.0, 0.1])

    def _empty_bates_result(self, ticker: str, platform: str, n_obs: int) -> BatesResult:
        return BatesResult(
            ticker=ticker, platform=platform,
            kappa=np.nan, theta=np.nan, xi=np.nan, rho=np.nan, v0=np.nan,
            lambda_j=np.nan, mu_j=np.nan, sigma_j=np.nan,
            feller_satisfied=False, converged=False,
            mse=np.nan, log_likelihood=np.nan, log_likelihood_heston=np.nan,
            lr_statistic=np.nan, p_value_jump_significance=np.nan, n_obs=n_obs,
        )

    def _save_bates(self, result: BatesResult) -> None:
        path = self.out_dir / f"{result.ticker}.json"
        with open(path, "w") as f:
            json.dump(asdict(result), f, indent=2, default=_json_default)


# ---------------------------------------------------------------------------
# OUTPUT — MODEL COMPARISON
# ---------------------------------------------------------------------------

class ModelComparison:
    BS_K = 1
    HESTON_K = 5
    BATES_K = 8
    VAR_FLOOR = 1e-4

    def __init__(self, data_dir: Path | None = None):
        self.data_dir = data_dir or DATA_DIR
        self.bs_dir = self.data_dir / "processed" / "bs_params"
        self.heston_dir = self.data_dir / "processed" / "heston_params"
        self.bates_dir = self.data_dir / "processed" / "bates_params"
        self.iv_dir = self.data_dir / "processed" / "implied_vol"

    def run(self) -> pd.DataFrame:
        bs_results = self._load_results(self.bs_dir)
        heston_results = self._load_results(self.heston_dir)
        bates_results = self._load_results(self.bates_dir)

        tickers = set(bs_results.keys()) & set(heston_results.keys()) & set(bates_results.keys())
        if not tickers:
            print("No contracts with all three models calibrated.")
            return pd.DataFrame()

        records = [self._compare_contract(t, bs_results[t], heston_results[t], bates_results[t])
                   for t in sorted(tickers)]

        summary = pd.DataFrame(records)
        out_path = self.data_dir / "processed" / "model_comparison_summary.parquet"
        summary.to_parquet(out_path, index=False)
        print(f"Model comparison summary saved to {out_path}")
        print(f"  Contracts compared: {len(summary)}")
        self._print_split_summary(summary)
        return summary

    def _compare_contract(self, ticker: str, bs: dict, h: dict, b: dict) -> dict:
        n = h.get("n_obs", 0)
        ll_h = h.get("log_likelihood", np.nan)
        ll_b = b.get("log_likelihood", np.nan)
        qlike_h = self._qlike(ticker, "heston")
        qlike_b = self._qlike(ticker, "bates")
        qlike_bs = self._qlike_bs(ticker, bs)
        iv_path = self.iv_dir / f"{ticker}.parquet"
        near_resolution_frac = np.nan
        if iv_path.exists():
            iv_df = pd.read_parquet(iv_path)
            if "T" in iv_df.columns and len(iv_df) > 0:
                near_resolution_frac = float((iv_df["T"] <= (5 / 365.25)).mean())

        return {
            "ticker": ticker, "platform": h.get("platform", "unknown"), "n_obs": n,
            "sigma_bs": bs.get("sigma_bs", np.nan), "mse_bs": bs.get("mse", np.nan),
            "ll_bs": bs.get("log_likelihood", np.nan),
            "aic_bs": self._aic(bs.get("log_likelihood", np.nan), self.BS_K),
            "bic_bs": self._bic(bs.get("log_likelihood", np.nan), self.BS_K, n),
            "qlike_median_bs": qlike_bs["median"], "qlike_mean_bs": qlike_bs["mean"],
            "mse_heston": h.get("mse", np.nan), "ll_heston": ll_h,
            "aic_heston": self._aic(ll_h, self.HESTON_K), "bic_heston": self._bic(ll_h, self.HESTON_K, n),
            "qlike_median_heston": qlike_h["median"], "qlike_mean_heston": qlike_h["mean"],
            "mse_bates": b.get("mse", np.nan), "ll_bates": ll_b,
            "aic_bates": self._aic(ll_b, self.BATES_K), "bic_bates": self._bic(ll_b, self.BATES_K, n),
            "qlike_median_bates": qlike_b["median"], "qlike_mean_bates": qlike_b["mean"],
            "lr_statistic": b.get("lr_statistic", np.nan),
            "p_value_jump": b.get("p_value_jump_significance", np.nan),
            "feller_heston": h.get("feller_satisfied", False),
            "feller_bates": b.get("feller_satisfied", False),
            "converged_heston": h.get("converged", False),
            "converged_bates": b.get("converged", False),
            "near_resolution_frac": near_resolution_frac,
        }

    def _qlike_bs(self, ticker: str, bs: dict) -> dict:
        nan_result = {"median": np.nan, "mean": np.nan}
        iv_path = self.iv_dir / f"{ticker}.parquet"
        if not iv_path.exists(): return nan_result
        sigma_bs = bs.get("sigma_bs", np.nan)
        if not np.isfinite(sigma_bs) or sigma_bs <= 0: return nan_result
        iv_df = pd.read_parquet(iv_path)
        if "near_boundary" in iv_df.columns:
            iv_df = iv_df[~iv_df["near_boundary"]]
        sigma_obs = iv_df["sigma_implied"].dropna().values
        if len(sigma_obs) == 0: return nan_result
        var_obs = np.maximum(sigma_obs ** 2, self.VAR_FLOOR)
        var_bs = max(sigma_bs ** 2, self.VAR_FLOOR)
        ratio = var_obs / var_bs
        pointwise = ratio - np.log(ratio) - 1.0
        return {"median": float(np.median(pointwise)), "mean": float(np.mean(pointwise))}

    def _qlike(self, ticker: str, model: str) -> dict:
        nan_result = {"median": np.nan, "mean": np.nan}
        iv_path = self.iv_dir / f"{ticker}.parquet"
        if not iv_path.exists(): return nan_result
        param_dir = self.heston_dir if model == "heston" else self.bates_dir
        param_path = param_dir / f"{ticker}.json"
        if not param_path.exists(): return nan_result

        iv_df = pd.read_parquet(iv_path)
        if "near_boundary" in iv_df.columns:
            iv_df = iv_df[~iv_df["near_boundary"]]
        sigma_obs = iv_df["sigma_implied"].dropna().values
        if len(sigma_obs) == 0: return nan_result

        with open(param_path) as f:
            params = json.load(f)

        dt = 1 / 6552
        if model == "heston":
            p = np.array([params["kappa"], params["theta"], params["xi"], params["rho"], params["v0"]])
            var_model_full = HestonCalibrator._simulate_variance_path(p, len(iv_df), dt)
        else:
            p = np.array([params["kappa"], params["theta"], params["xi"], params["rho"],
                          params["v0"], params["lambda_j"], params["mu_j"], params["sigma_j"]])
            var_model_full = BatesCalibrator._simulate_bates_variance_path(p, len(iv_df), dt)

        filtered_positions = np.arange(len(iv_df))[iv_df["sigma_implied"].notna().values]
        var_model = var_model_full[filtered_positions]
        var_obs = np.maximum(sigma_obs ** 2, self.VAR_FLOOR)
        var_model = np.maximum(var_model, self.VAR_FLOOR)
        ratio = var_obs / var_model
        pointwise = ratio - np.log(ratio) - 1.0
        return {"median": float(np.median(pointwise)), "mean": float(np.mean(pointwise))}

    @staticmethod
    def _aic(ll: float, k: int) -> float:
        return np.nan if not np.isfinite(ll) else 2 * k - 2 * ll

    @staticmethod
    def _bic(ll: float, k: int, n: int) -> float:
        return np.nan if not np.isfinite(ll) or n <= 0 else k * np.log(n) - 2 * ll

    @staticmethod
    def _print_split_summary(df: pd.DataFrame) -> None:
        metrics = ["mse_bs", "mse_heston", "mse_bates",
                   "aic_bs", "aic_heston", "aic_bates",
                   "qlike_median_bs", "qlike_median_heston", "qlike_median_bates"]
        print("\n--- By Platform ---")
        if "platform" in df.columns:
            for platform, group in df.groupby("platform"):
                print(f"  {platform} (n={len(group)}):")
                for m in metrics:
                    print(f"    {m}: {group[m].mean():.6f}")
        print("\n--- Near Resolution (last 5 days) ---")
        if "near_resolution_frac" in df.columns:
            near = df[df["near_resolution_frac"] > 0.5]
            far = df[df["near_resolution_frac"] <= 0.5]
            for label, subset in [("Near", near), ("Far", far)]:
                if len(subset) > 0:
                    print(f"  {label} (n={len(subset)}):")
                    for m in metrics:
                        print(f"    {m}: {subset[m].mean():.6f}")

    @staticmethod
    def _load_results(directory: Path) -> dict[str, dict]:
        results = {}
        if not directory.exists():
            return results
        for path in directory.glob("*.json"):
            with open(path) as f:
                data = json.load(f)
            results[data.get("ticker", path.stem)] = data
        return results


# ---------------------------------------------------------------------------
# OUTPUT — PLOTS
# ---------------------------------------------------------------------------

FIGURE_DIR = DATA_DIR / "processed" / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
DPI = 300
sns.set_theme(style="whitegrid", font_scale=1.1)


def plot_implied_vol_series(ticker: str, data_dir: Path | None = None) -> Path:
    data_dir = data_dir or DATA_DIR
    df = pd.read_parquet(data_dir / "processed" / "implied_vol" / f"{ticker}.parquet")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    ax1.plot(df["timestamp"], df["price"], color="steelblue", linewidth=0.8)
    ax1.set_ylabel("Price (probability)")
    ax1.set_title(f"{ticker} — Price & Implied Volatility")
    ax2.plot(df["timestamp"], df["sigma_implied"], color="darkorange", linewidth=0.8)
    ax2.set_ylabel("σ_implied")
    ax2.set_xlabel("Time (UTC)")
    fomc_rows = df[df["is_jump_window"]]
    if not fomc_rows.empty:
        for _, row in fomc_rows.iterrows():
            ts = row["timestamp"]
            for ax in (ax1, ax2):
                ax.axvspan(ts - pd.Timedelta(minutes=30), ts + pd.Timedelta(minutes=30),
                           alpha=0.15, color="red", label="_nolegend_")
    fig.tight_layout()
    out = FIGURE_DIR / f"iv_series_{ticker}.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")
    return out


def plot_heston_fit(ticker: str, data_dir: Path | None = None) -> Path:
    data_dir = data_dir or DATA_DIR
    iv_df = pd.read_parquet(data_dir / "processed" / "implied_vol" / f"{ticker}.parquet")
    sigma_obs = iv_df["sigma_implied"].dropna().values
    var_obs = sigma_obs ** 2
    timestamps = iv_df.loc[iv_df["sigma_implied"].notna(), "timestamp"].values
    with open(data_dir / "processed" / "heston_params" / f"{ticker}.json") as f:
        params = json.load(f)
    p = np.array([params["kappa"], params["theta"], params["xi"], params["rho"], params["v0"]])
    var_model = HestonCalibrator._simulate_variance_path(p, len(var_obs), 1 / 6552)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(timestamps, var_obs, color="steelblue", linewidth=0.8, alpha=0.7, label="Observed σ²")
    ax.plot(timestamps, var_model, color="darkorange", linewidth=1.2, label="Heston fit")
    ax.set_ylabel("Variance (σ²)")
    ax.set_xlabel("Time (UTC)")
    ax.set_title(f"{ticker} — Heston Model Fit")
    ax.legend()
    fig.tight_layout()
    out = FIGURE_DIR / f"heston_fit_{ticker}.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")
    return out


def plot_parameter_distribution(param_name: str, data_dir: Path | None = None) -> Path:
    data_dir = data_dir or DATA_DIR
    values = []
    for path in (data_dir / "processed" / "heston_params").glob("*.json"):
        with open(path) as f:
            result = json.load(f)
        val = result.get(param_name)
        if val is not None and np.isfinite(val):
            values.append(val)
    if not values:
        print(f"  No valid values for {param_name}")
        return FIGURE_DIR / f"param_dist_{param_name}.png"
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(values, bins=30, color="steelblue", edgecolor="white", alpha=0.85)
    ax.axvline(np.median(values), color="darkorange", linestyle="--",
               label=f"Median: {np.median(values):.4f}")
    ax.set_xlabel(param_name)
    ax.set_ylabel("Count")
    ax.set_title(f"Distribution of Heston {param_name} (n={len(values)})")
    ax.legend()
    fig.tight_layout()
    out = FIGURE_DIR / f"param_dist_{param_name}.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")
    return out


def plot_model_comparison_table(data_dir: Path | None = None) -> Path:
    data_dir = data_dir or DATA_DIR
    df = pd.read_parquet(data_dir / "processed" / "model_comparison_summary.parquet")
    type_map: dict[str, str] = {}
    for cat_path in [data_dir / "raw" / "kalshi" / "catalog_filtered.parquet",
                     data_dir / "raw" / "polymarket" / "catalog_filtered.parquet"]:
        if cat_path.exists():
            cat = pd.read_parquet(cat_path)
            id_col = "ticker" if "ticker" in cat.columns else "condition_id"
            for _, row in cat.iterrows():
                cid = row.get(id_col, "")
                if row.get("is_fed", False):
                    type_map[cid] = "Fed/FOMC"
                elif row.get("is_political", False):
                    type_map[cid] = "Political"
    df["market_type"] = df["ticker"].map(type_map).fillna("Other")
    pivot_data = {
        mt: {"Heston MSE": subset["mse_heston"].mean(), "Bates MSE": subset["mse_bates"].mean()}
        for mt, subset in [(mt, df[df["market_type"] == mt]) for mt in df["market_type"].unique()]
    }
    heatmap_df = pd.DataFrame(pivot_data).T
    if heatmap_df.empty:
        print("  No data for heatmap")
        return FIGURE_DIR / "model_comparison_heatmap.png"
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(heatmap_df, annot=True, fmt=".6f", cmap="YlOrRd", linewidths=0.5, ax=ax)
    ax.set_title("Mean MSE: Model × Market Type")
    ax.set_ylabel("Market Type")
    fig.tight_layout()
    out = FIGURE_DIR / "model_comparison_heatmap.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")
    return out


# ---------------------------------------------------------------------------
# PIPELINE STEP HELPERS
# ---------------------------------------------------------------------------

def _build_contracts_list(df: pd.DataFrame, id_col: str) -> list[dict]:
    contracts = []
    for _, row in df.iterrows():
        cid = str(row[id_col])
        raw = row.get("clobTokenIds", "[]")
        try:
            token_ids = ast.literal_eval(raw) if isinstance(raw, str) else raw
            yes_token = str(token_ids[0]) if token_ids else None
        except (ValueError, SyntaxError, IndexError):
            yes_token = None
        if yes_token:
            contracts.append({"condition_id": cid, "yes_token": yes_token})
    return contracts


def _platform_map() -> dict[str, str]:
    pm: dict[str, str] = {}
    for platform in ["kalshi", "polymarket"]:
        cat_path = DATA_DIR / "raw" / platform / "catalog_filtered.parquet"
        if cat_path.exists():
            cat = pd.read_parquet(cat_path)
            if platform == "kalshi":
                for t in cat.get("ticker", []):
                    pm[str(t)] = "kalshi"
            else:
                for col in ["conditionId", "condition_id", "id"]:
                    if col in cat.columns:
                        for t in cat[col]:
                            pm[str(t)] = "polymarket"
                        break
    return pm


def _find_id_col(df: pd.DataFrame) -> str | None:
    for col in ["conditionId", "condition_id", "id"]:
        if col in df.columns:
            return col
    return None


# ---------------------------------------------------------------------------
# PIPELINE STEPS
# ---------------------------------------------------------------------------

def step_check():
    print("Checking your setup...\n")
    import platform as _plt
    print(f"  Python version: {_plt.python_version()}")
    ok = sys.version_info[:2] >= (3, 10)
    print("  OK" if ok else "  PROBLEM: Need Python 3.10+. Run: conda activate pmr")
    print()
    for pkg in ["pandas", "numpy", "scipy", "requests", "pyarrow", "tqdm"]:
        try:
            __import__(pkg)
            print(f"  {pkg}: installed")
        except ImportError:
            print(f"  {pkg}: MISSING — run: pip install -r requirements.txt")
            ok = False
    print()
    print(f"  Kalshi API key: {'set' if KALSHI_API_KEY else 'not set (OK)'}")
    print(f"  Polymarket API key: {'set' if POLYMARKET_API_KEY else 'not set (OK)'}")
    print()
    for d in ["data/raw/kalshi", "data/raw/polymarket",
              "data/processed/implied_vol", "data/processed/heston_params",
              "data/processed/bates_params", "data/processed/figures"]:
        p = DATA_DIR.parent / d if not d.startswith("/") else Path(d)
        p = DATA_DIR / d.replace("data/", "")
        p.mkdir(parents=True, exist_ok=True)
        print(f"  {d}/ OK")
    print()
    print("Everything looks good." if ok else "Fix the issues above before continuing.")


def step_catalog_kalshi():
    print(f"Connecting to Kalshi Elections API...\nTargeting {KALSHI_DATE_START} to {KALSHI_DATE_END}\n")
    client = KalshiClient(KALSHI_API_KEY)
    catalog = KalshiCatalog(client)
    df = catalog.pull_catalog()
    print(f"\nDone. {len(df)} contracts passed filters.")
    print("Files: data/raw/kalshi/catalog_full.parquet, catalog_filtered.parquet")
    print("Next: python pipeline.py catalog-polymarket")


def step_catalog_polymarket():
    print("Connecting to Polymarket Gamma API...\n")
    client = GammaClient()
    catalog = PolymarketCatalog(client)
    df = catalog.pull_catalog()
    print(f"\nDone. {len(df)} contracts passed ALL filters.")
    print("Files: data/raw/polymarket/catalog_full.parquet, catalog_filtered.parquet, contracts_for_review.csv")
    print("Next: python pipeline.py liquidity-kalshi")


def step_liquidity_kalshi():
    catalog_path = DATA_DIR / "raw" / "kalshi" / "catalog_filtered.parquet"
    if not catalog_path.exists():
        print("ERROR: Run catalog-kalshi first."); return
    df = pd.read_parquet(catalog_path)
    print(f"  {len(df)} contracts to check.")
    client = KalshiClient(KALSHI_API_KEY or "")
    catalog = KalshiCatalog(client)
    df = catalog.apply_liquidity_filter(df)
    print(f"\nDone. {len(df)} Kalshi contracts passed liquidity filter.")
    print("Next: python pipeline.py trades-kalshi")


def step_trades_kalshi():
    catalog_path = DATA_DIR / "raw" / "kalshi" / "catalog_filtered.parquet"
    if not catalog_path.exists():
        print("ERROR: Run catalog and liquidity steps first."); return
    df = pd.read_parquet(catalog_path)
    tickers = df["ticker"].tolist()
    print(f"Pulling hourly trade data for {len(tickers)} Kalshi contracts...")
    client = KalshiClient(KALSHI_API_KEY or "")
    puller = KalshiTradesPuller(client)
    puller.pull(tickers)
    print("Next: python pipeline.py trades-polymarket")


def step_trades_sample():
    catalog_path = DATA_DIR / "raw" / "polymarket" / "catalog_filtered.parquet"
    if not catalog_path.exists():
        print("ERROR: Run catalog-polymarket first."); return
    df = pd.read_parquet(catalog_path)
    id_col = _find_id_col(df)
    if id_col is None:
        print("ERROR: No condition ID column found."); return
    fed = df[df["is_fed"] == True].copy()  # noqa: E712
    political = df[df["is_political"] == True].copy()  # noqa: E712
    n_political = max(0, 100 - len(fed))
    if "trade_count" in political.columns:
        political = political.nlargest(n_political, "trade_count")
    else:
        political = political.head(n_political)
    sample = pd.concat([fed, political], ignore_index=True)
    contracts = _build_contracts_list(sample, id_col)
    print(f"Stratified sample: {len(fed)} FED + {len(political)} political = {len(sample)} selected")
    print(f"  ({len(contracts)} have YES tokens)")
    client = PolymarketClient(POLYMARKET_API_KEY or "")
    puller = PolymarketTradesPuller(client)
    puller.pull(contracts)
    print("Next: python pipeline.py backtest-implied-vol")


def step_trades_polymarket():
    catalog_path = DATA_DIR / "raw" / "polymarket" / "catalog_filtered.parquet"
    if not catalog_path.exists():
        print("ERROR: Run catalog-polymarket first."); return
    df = pd.read_parquet(catalog_path)
    id_col = _find_id_col(df)
    if id_col is None:
        print("ERROR: No condition ID column found."); return
    contracts = _build_contracts_list(df, id_col)
    print(f"Pulling daily price history for {len(contracts)} Polymarket contracts...")
    client = PolymarketClient(POLYMARKET_API_KEY or "")
    puller = PolymarketTradesPuller(client)
    puller.pull(contracts)


def step_backtest_implied_vol():
    fomc_dates = [pd.Timestamp(d) for d in FOMC_DATES]
    processed = skipped = errors = 0

    for platform in ["kalshi", "polymarket"]:
        raw_dir = DATA_DIR / "raw" / platform
        catalog_path = raw_dir / "catalog_filtered.parquet"
        if not catalog_path.exists():
            print(f"  No catalog found for {platform} — skipping."); continue

        catalog = pd.read_parquet(catalog_path)
        id_col = "ticker" if platform == "kalshi" else _find_id_col(catalog)
        if id_col is None:
            print(f"  No ID column for {platform} — skipping."); continue

        print(f"\n  Processing {platform} ({len(catalog)} contracts)...")
        for _, row in catalog.iterrows():
            ticker = str(row[id_col])
            trade_path = raw_dir / f"{ticker}.parquet"
            if not trade_path.exists():
                skipped += 1; continue
            out_path = DATA_DIR / "processed" / "implied_vol" / f"{ticker}.parquet"
            if out_path.exists():
                processed += 1; continue
            try:
                _ct = pd.Timestamp(row["close_time"])
                close_time = _ct.tz_localize("UTC") if _ct.tzinfo is None else _ct.tz_convert("UTC")
                df = pd.read_parquet(trade_path)
                ImpliedVolExtractor(close_time=close_time, fomc_dates=fomc_dates).extract(df, ticker)
                processed += 1
            except Exception as e:
                print(f"    Error on {ticker}: {e}"); errors += 1

    print(f"\n--- Implied Vol Extraction Summary ---")
    print(f"  Processed: {processed}  Skipped: {skipped}  Errors: {errors}")
    print("Next: python pipeline.py backtest-bs")


def step_backtest_bs():
    iv_dir = DATA_DIR / "processed" / "implied_vol"
    if not iv_dir.exists() or not any(iv_dir.glob("*.parquet")):
        print("ERROR: Run backtest-implied-vol first."); return

    pm = _platform_map()
    calibrator = BSCalibrator()
    iv_files = sorted(iv_dir.glob("*.parquet"))
    print(f"Calibrating Black-Scholes on {len(iv_files)} contracts...")
    calibrated = errors = 0

    for iv_path in iv_files:
        ticker = iv_path.stem
        out_path = DATA_DIR / "processed" / "bs_params" / f"{ticker}.json"
        if out_path.exists():
            calibrated += 1; continue
        try:
            df = pd.read_parquet(iv_path)
            sigma = df["sigma_implied"].values
            result = calibrator.calibrate(sigma, ticker, pm.get(ticker, "unknown"))
            calibrated += 1
            print(f"  {ticker[:20]}: sigma_BS={result.sigma_bs:.4f}, mse={result.mse:.6f}, n={result.n_obs}")
        except Exception as e:
            print(f"  Error on {ticker}: {e}"); errors += 1

    print(f"\n--- BS Calibration Summary ---  Calibrated: {calibrated}  Errors: {errors}")
    print("Next: python pipeline.py backtest-heston")


def step_backtest_heston():
    iv_dir = DATA_DIR / "processed" / "implied_vol"
    if not iv_dir.exists() or not any(iv_dir.glob("*.parquet")):
        print("ERROR: Run backtest-implied-vol first."); return

    pm = _platform_map()
    calibrator = HestonCalibrator()
    iv_files = sorted(iv_dir.glob("*.parquet"))
    print(f"Calibrating Heston on {len(iv_files)} contracts (dt={DAILY_DT:.6e})...")
    calibrated = errors = 0

    for iv_path in iv_files:
        ticker = iv_path.stem
        out_path = DATA_DIR / "processed" / "heston_params" / f"{ticker}.json"
        if out_path.exists():
            calibrated += 1; continue
        try:
            df = pd.read_parquet(iv_path)
            result = calibrator.calibrate(df["sigma_implied"].values, ticker,
                                          pm.get(ticker, "unknown"), dt=DAILY_DT)
            calibrated += 1
            print(f"  {ticker}: kappa={result.kappa:.3f}, theta={result.theta:.4f}, "
                  f"xi={result.xi:.3f}, rho={result.rho:.3f} "
                  f"{'(Feller OK)' if result.feller_satisfied else '(Feller VIOLATED)'}")
        except Exception as e:
            print(f"  Error on {ticker}: {e}"); errors += 1

    print(f"\n--- Heston Calibration Summary ---  Calibrated: {calibrated}  Errors: {errors}")
    print("Next: python pipeline.py backtest-bates")


def step_backtest_bates():
    heston_dir = DATA_DIR / "processed" / "heston_params"
    iv_dir = DATA_DIR / "processed" / "implied_vol"
    if not heston_dir.exists() or not any(heston_dir.glob("*.json")):
        print("ERROR: Run backtest-heston first."); return

    pm = _platform_map()
    calibrator = BatesCalibrator()
    heston_files = sorted(heston_dir.glob("*.json"))
    print(f"Calibrating Bates on {len(heston_files)} contracts...")
    calibrated = jumps_significant = errors = 0

    for h_path in heston_files:
        ticker = h_path.stem
        out_path = DATA_DIR / "processed" / "bates_params" / f"{ticker}.json"
        if out_path.exists():
            with open(out_path) as f:
                existing = json.load(f)
            calibrated += 1
            if existing.get("p_value_jump_significance", 1.0) < 0.05:
                jumps_significant += 1
            continue

        iv_path = iv_dir / f"{ticker}.parquet"
        if not iv_path.exists(): continue

        try:
            df = pd.read_parquet(iv_path)
            with open(h_path) as f:
                h_data = json.load(f)
            heston_result = HestonResult(**h_data)
            result = calibrator.calibrate(df["sigma_implied"].values, ticker,
                                          pm.get(ticker, "unknown"), dt=DAILY_DT,
                                          heston_result=heston_result)
            calibrated += 1
            sig = "YES" if result.p_value_jump_significance < 0.05 else "no"
            if result.p_value_jump_significance < 0.05:
                jumps_significant += 1
            print(f"  {ticker}: lambda={result.lambda_j:.2f}, mu_j={result.mu_j:.3f}, "
                  f"sigma_j={result.sigma_j:.3f}, jumps={sig} (p={result.p_value_jump_significance:.4f})")
        except Exception as e:
            print(f"  Error on {ticker}: {e}"); errors += 1

    print(f"\n--- Bates Calibration Summary ---")
    print(f"  Calibrated: {calibrated}  Jumps significant: {jumps_significant}/{calibrated}  Errors: {errors}")
    print("Next: python pipeline.py compare")


def step_compare():
    print("Comparing Heston vs Bates across all contracts...\n")
    mc = ModelComparison()
    summary = mc.run()
    if summary.empty:
        print("No results to compare."); return
    print(f"\n--- Headline Results ---")
    if "mse_heston" in summary.columns and "mse_bates" in summary.columns:
        h_wins = (summary["mse_heston"] < summary["mse_bates"]).sum()
        b_wins = (summary["mse_bates"] < summary["mse_heston"]).sum()
        print(f"  MSE: Heston wins {h_wins}, Bates wins {b_wins}")
    if "p_value_jump" in summary.columns:
        sig = (summary["p_value_jump"] < 0.05).sum()
        print(f"  Jump significance (p<0.05): {sig} / {len(summary)} contracts")
    print("Next: python pipeline.py plots")


def step_plots():
    iv_dir = DATA_DIR / "processed" / "implied_vol"
    heston_dir = DATA_DIR / "processed" / "heston_params"

    print("Generating implied vol time series plots...")
    for iv_path in sorted(iv_dir.glob("*.parquet"))[:10]:
        try:
            plot_implied_vol_series(iv_path.stem)
        except Exception as e:
            print(f"  Skipped {iv_path.stem}: {e}")

    print("\nGenerating Heston fit plots...")
    for h_path in sorted(heston_dir.glob("*.json"))[:10]:
        try:
            plot_heston_fit(h_path.stem)
        except Exception as e:
            print(f"  Skipped {h_path.stem}: {e}")

    print("\nGenerating parameter distribution histograms...")
    for param in ["kappa", "theta", "xi", "rho", "v0"]:
        try:
            plot_parameter_distribution(param)
        except Exception as e:
            print(f"  Skipped {param}: {e}")

    print("\nGenerating model comparison heatmap...")
    summary_path = DATA_DIR / "processed" / "model_comparison_summary.parquet"
    if summary_path.exists():
        try:
            plot_model_comparison_table()
        except Exception as e:
            print(f"  Skipped heatmap: {e}")
    else:
        print("  Run 'compare' first.")

    print(f"\nAll figures saved to: {FIGURE_DIR}")


def step_export():
    out_dir = DATA_DIR / "exports"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Implied vol — combined CSV
    iv_files = sorted((DATA_DIR / "processed" / "implied_vol").glob("*.parquet"))
    if iv_files:
        chunks = []
        for f in iv_files:
            df = pd.read_parquet(f)
            df.insert(0, "ticker", f.stem)
            chunks.append(df)
        combined = pd.concat(chunks, ignore_index=True)
        combined["timestamp"] = combined["timestamp"].astype(str)
        path = out_dir / "implied_vol_all.csv"
        combined.to_csv(path, index=False)
        print(f"  implied_vol_all.csv   — {len(combined):,} rows")
    else:
        print("  implied_vol: nothing yet")

    # Heston params
    heston_files = sorted((DATA_DIR / "processed" / "heston_params").glob("*.json"))
    if heston_files:
        rows = []
        for f in heston_files:
            with open(f) as fh:
                d = json.load(fh)
            d["ticker"] = f.stem
            rows.append(d)
        hdf = pd.DataFrame(rows)
        cols = ["ticker"] + [c for c in hdf.columns if c != "ticker"]
        hdf[cols].to_csv(out_dir / "heston_params.csv", index=False)
        print(f"  heston_params.csv     — {len(hdf)} contracts")
    else:
        print("  heston_params: nothing yet")

    # Bates params
    bates_files = sorted((DATA_DIR / "processed" / "bates_params").glob("*.json"))
    if bates_files:
        rows = []
        for f in bates_files:
            with open(f) as fh:
                d = json.load(fh)
            d["ticker"] = f.stem
            rows.append(d)
        bdf = pd.DataFrame(rows)
        cols = ["ticker"] + [c for c in bdf.columns if c != "ticker"]
        bdf[cols].to_csv(out_dir / "bates_params.csv", index=False)
        print(f"  bates_params.csv      — {len(bdf)} contracts")
    else:
        print("  bates_params: nothing yet")

    # Model comparison
    summary_path = DATA_DIR / "processed" / "model_comparison_summary.parquet"
    if summary_path.exists():
        sdf = pd.read_parquet(summary_path)
        sdf.to_csv(out_dir / "model_comparison.csv", index=False)
        print(f"  model_comparison.csv  — {len(sdf)} contracts")
    else:
        print("  model_comparison: nothing yet")

    # Raw trades
    trades_dir = DATA_DIR / "raw" / "polymarket"
    trade_files = [f for f in trades_dir.glob("*.parquet")
                   if f.stem not in ("catalog_full", "catalog_filtered")]
    if trade_files:
        trades_out = out_dir / "trades"
        trades_out.mkdir(exist_ok=True)
        for f in trade_files:
            df = pd.read_parquet(f)
            df["timestamp"] = df["timestamp"].astype(str)
            df.to_csv(trades_out / f"{f.stem}.csv", index=False)
        print(f"  trades/               — {len(trade_files)} individual CSVs")

    print(f"\nAll exports saved to: {out_dir}")


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

STEPS = {
    "check":               (step_check,               "Verify environment setup"),
    "catalog-kalshi":      (step_catalog_kalshi,       "Pull + filter Kalshi catalog"),
    "catalog-polymarket":  (step_catalog_polymarket,   "Pull + filter Polymarket catalog"),
    "liquidity-kalshi":    (step_liquidity_kalshi,     "Trade-count filter for Kalshi"),
    "trades-kalshi":       (step_trades_kalshi,        "Pull hourly Kalshi trade data"),
    "trades-sample":       (step_trades_sample,        "Pull 100-contract sample (43 FED + 57 political)"),
    "trades-polymarket":   (step_trades_polymarket,    "Pull all Polymarket contracts"),
    "backtest-implied-vol":(step_backtest_implied_vol, "Extract implied volatility (Cash-or-Nothing)"),
    "backtest-bs":         (step_backtest_bs,          "Calibrate Black-Scholes baseline"),
    "backtest-heston":     (step_backtest_heston,      "Calibrate Heston stochastic vol model"),
    "backtest-bates":      (step_backtest_bates,       "Calibrate Bates model (Heston + jumps)"),
    "compare":             (step_compare,              "Compare models (MSE, AIC, BIC, QLIKE)"),
    "plots":               (step_plots,                "Generate all research figures"),
    "export":              (step_export,               "Export everything to CSV"),
}


def print_help():
    print("Prediction Market Research — Consolidated Pipeline")
    print("=" * 52)
    print("\nUsage:  python pipeline.py <step>")
    print("\nPHASE 1: Data Collection")
    print("-" * 40)
    for step in ["check", "catalog-kalshi", "catalog-polymarket", "liquidity-kalshi",
                 "trades-kalshi", "trades-sample", "trades-polymarket"]:
        fn, desc = STEPS[step]
        print(f"  {step:28s} {desc}")
    print("\nPHASE 2: Backtesting")
    print("-" * 40)
    for step in ["backtest-implied-vol", "backtest-bs", "backtest-heston", "backtest-bates"]:
        fn, desc = STEPS[step]
        print(f"  {step:28s} {desc}")
    print("\nPHASE 3: Results")
    print("-" * 40)
    for step in ["compare", "plots", "export"]:
        fn, desc = STEPS[step]
        print(f"  {step:28s} {desc}")
    print("\nStart with:  python pipeline.py check")


if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).parent.parent
    sys.path.insert(0, str(PROJECT_ROOT))
    os.chdir(PROJECT_ROOT)

    if len(sys.argv) < 2 or sys.argv[1] in ("help", "--help", "-h"):
        print_help()
        sys.exit(0)

    step = sys.argv[1]
    if step not in STEPS:
        print(f"Unknown step: '{step}'")
        print("Run 'python pipeline.py help' to see available steps.")
        sys.exit(1)

    fn, desc = STEPS[step]
    print(f"=== {desc} ===\n")
    fn()
