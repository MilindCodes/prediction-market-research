import time
import requests


class KalshiClient:
    """HTTP client for the Kalshi Elections API v2 with pagination, rate
    limiting, and exponential backoff on 429 responses.

    The Kalshi Elections API is publicly readable — no API key is required
    for listing markets or pulling trade data.

    Parameters
    ----------
    api_key : str, optional
        API key for authenticated endpoints (not needed for reads).
    base_url : str, optional
        Root URL for the API (default: Kalshi Elections production endpoint).
    """

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
        """Enforce minimum interval between consecutive requests."""
        elapsed = time.monotonic() - self._last_request_time
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)

    def get(self, endpoint: str, params: dict | None = None,
            max_retries: int = 5, max_pages: int | None = None,
            progress_label: str = "") -> list[dict]:
        """GET with automatic pagination via cursor and retry on 429.

        Parameters
        ----------
        endpoint : str
            API path relative to base URL (e.g. "/markets").
        params : dict, optional
            Query parameters for the first request.
        max_retries : int
            Maximum retries per individual page request on 429 errors.
        max_pages : int, optional
            Hard cap on number of pages fetched. None = unlimited (only use
            for known-small result sets like series_ticker=KXFED).
        progress_label : str
            Label printed with page progress (e.g. "FOMC events").

        Returns
        -------
        list[dict]
            Accumulated items across all pages.
        """
        url = f"{self.base_url}{endpoint}"
        params = dict(params or {})
        all_items: list[dict] = []
        page = 0

        while True:
            page += 1
            if max_pages and page > max_pages:
                print(f"  [{progress_label or endpoint}] "
                      f"Reached page limit ({max_pages}), stopping.")
                break

            if progress_label and page % 5 == 1:
                print(f"  [{progress_label}] Fetching page {page}... "
                      f"({len(all_items)} so far)", end="\r")

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
        """GET a single resource (no pagination).

        Parameters
        ----------
        endpoint : str
            API path relative to base URL.
        params : dict, optional
            Query parameters.
        max_retries : int
            Maximum retries on 429 errors.

        Returns
        -------
        dict
            Parsed JSON response body.
        """
        url = f"{self.base_url}{endpoint}"
        return self._request_with_retry(url, params or {}, max_retries)

    def _request_with_retry(self, url: str, params: dict,
                            max_retries: int) -> dict:
        """Execute a single GET with exponential backoff on errors.

        Retries on:
          - HTTP 429 / 500 / 502 / 503 (server-side)
          - ConnectionError / Timeout (network drops, SSL resets, flaky WiFi)
        """
        backoff = 2.0
        for attempt in range(max_retries + 1):
            self._rate_limit()
            self._last_request_time = time.monotonic()

            try:
                resp = self.session.get(url, params=params, timeout=30)
            except requests.exceptions.ConnectionError as e:
                if attempt < max_retries:
                    print(f"\n[KalshiClient] Network error (attempt "
                          f"{attempt + 1}/{max_retries}), retrying in "
                          f"{backoff:.0f}s... ({e})")
                    time.sleep(backoff)
                    backoff = min(backoff * 2, 30)
                    continue
                raise
            except requests.exceptions.Timeout:
                if attempt < max_retries:
                    print(f"\n[KalshiClient] Timeout (attempt "
                          f"{attempt + 1}/{max_retries}), retrying in "
                          f"{backoff:.0f}s...")
                    time.sleep(backoff)
                    backoff = min(backoff * 2, 30)
                    continue
                raise

            if resp.status_code == 200:
                return resp.json()

            if resp.status_code in (429, 500, 502, 503) and attempt < max_retries:
                print(f"\n[KalshiClient] HTTP {resp.status_code} (attempt "
                      f"{attempt + 1}/{max_retries}), retrying in "
                      f"{backoff:.0f}s...")
                time.sleep(backoff)
                backoff = min(backoff * 2, 30)
                continue

            print(f"\n[KalshiClient] ERROR {resp.status_code} on {url}: "
                  f"{resp.text[:200]}")
            resp.raise_for_status()

        return {}

    @staticmethod
    def _find_items_key(data: dict) -> str | None:
        """Find the key holding the list of items in a paginated response."""
        for key, val in data.items():
            if key not in ("cursor", "milestones") and isinstance(val, list):
                return key
        return None
