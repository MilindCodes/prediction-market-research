import os
from pathlib import Path

POLYMARKET_API_KEY = os.environ.get("POLYMARKET_API_KEY", "")

# Kalshi auth — two options (use whichever you have):
#   Option A (simple Bearer token): set KALSHI_API_KEY
#   Option B (RSA key file, downloaded from Kalshi dashboard):
#     1. Go to kalshi.com → Settings → API Keys → Create Key → Download .pem
#     2. Set KALSHI_KEY_FILE to the path of that .pem file
#     3. Set KALSHI_KEY_ID to the key ID shown in the dashboard
KALSHI_API_KEY = os.environ.get("KALSHI_API_KEY", "")
KALSHI_KEY_ID = os.environ.get("KALSHI_KEY_ID", "")
KALSHI_KEY_FILE = os.environ.get(
    "KALSHI_KEY_FILE",
    str(Path(__file__).parent / "researchproject2.txt"),
)

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

# CPI / macro economic release keywords
CPI_KEYWORDS = [
    "cpi", "consumer price", "core cpi", "inflation",
    "pce", "personal consumption expenditure",
    "ppi", "producer price",
    "gdp", "gross domestic product",
    "nonfarm payroll", "jobs report", "unemployment rate",
    "retail sales", "durable goods",
]

# Combined economic keywords used by the catalog to filter for relevant markets
ECONOMIC_KEYWORDS = FED_KEYWORDS + CPI_KEYWORDS

POLITICAL_KEYWORDS = [
    "election", "president", "senate", "congress", "vote",
    "governor", "democrat", "republican", "primary", "midterm",
]

# Probability bounds applied before log-odds transform.
# Prices below/above these are truncated to avoid log(0) / log(inf).
LOG_ODDS_CLIP_LO: float = 0.02
LOG_ODDS_CLIP_HI: float = 0.98

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
