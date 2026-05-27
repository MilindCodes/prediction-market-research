import os
from pathlib import Path

KALSHI_API_KEY = os.environ.get("KALSHI_API_KEY", "")
POLYMARKET_API_KEY = os.environ.get("POLYMARKET_API_KEY", "")

DATE_RANGE_START = "2022-01-01"
DATE_RANGE_END = "2024-12-31"

MIN_DURATION_DAYS = 14
MIN_TRADE_COUNT = 500

DATA_DIR = Path("data/")

# Fed/FOMC: specific phrases only — "fed" and "rate" alone are too broad
# ("fed" matches "federal student loans", "FIDE"; "rate" matches "approval rate")
FED_KEYWORDS = [
    "fomc", "federal reserve", "fed funds", "fed rate",
    "interest rate", "rate hike", "rate cut", "rate increase", "rate decrease",
    "basis point", "bps", "powell", "federal open market",
]

POLITICAL_KEYWORDS = [
    "election", "president", "senate", "congress", "vote",
    "governor", "democrat", "republican", "primary", "midterm",
]

# Kalshi Elections API only has data from ~2026 onwards.
# Polymarket has full 2022-2024 history. Two separate date windows:
POLYMARKET_DATE_START = "2022-01-01"
POLYMARKET_DATE_END = "2024-12-31"
# Kalshi: use whatever is available (2025-2026 primary elections + recent FOMC)
KALSHI_DATE_START = "2025-01-01"
KALSHI_DATE_END = "2026-12-31"

# Legacy aliases used elsewhere in the pipeline (default to Polymarket range)
DATE_RANGE_START = POLYMARKET_DATE_START
DATE_RANGE_END = POLYMARKET_DATE_END

# Hourly time step: prediction markets trade 24/7, so dt = 1 hour / (365.25 * 24 hours)
HOURLY_DT = 1.0 / (365.25 * 24)  # ~1.1408e-4

# Daily time step: used for Heston/Bates calibration since the CLOB prices-history
# endpoint only retains data at daily (fidelity=1440) granularity for closed markets.
DAILY_DT = 1.0 / 365.25  # ~2.7379e-3

# FOMC statement release times (UTC) for 2022-2024.
# Statements are released at 2:00 PM Eastern:
#   EDT (Mar-Nov): 18:00 UTC | EST (Nov-Mar): 19:00 UTC
FOMC_DATES = [
    # 2022
    "2022-01-26T19:00:00Z",
    "2022-03-16T18:00:00Z",
    "2022-05-04T18:00:00Z",
    "2022-06-15T18:00:00Z",
    "2022-07-27T18:00:00Z",
    "2022-09-21T18:00:00Z",
    "2022-11-02T18:00:00Z",
    "2022-12-14T19:00:00Z",
    # 2023
    "2023-02-01T19:00:00Z",
    "2023-03-22T18:00:00Z",
    "2023-05-03T18:00:00Z",
    "2023-06-14T18:00:00Z",
    "2023-07-26T18:00:00Z",
    "2023-09-20T18:00:00Z",
    "2023-11-01T18:00:00Z",
    "2023-12-13T19:00:00Z",
    # 2024
    "2024-01-31T19:00:00Z",
    "2024-03-20T18:00:00Z",
    "2024-05-01T18:00:00Z",
    "2024-06-12T18:00:00Z",
    "2024-07-31T18:00:00Z",
    "2024-09-18T18:00:00Z",
    "2024-11-07T19:00:00Z",
    "2024-12-18T19:00:00Z",
]
