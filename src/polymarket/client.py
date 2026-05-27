import time
import requests


class PolymarketClient:
    """HTTP client for the Polymarket CLOB API with pagination, rate limiting,
    and exponential backoff on 429 responses.

    Mirrors KalshiClient's interface for consistency across the pipeline.

    Parameters
    ----------
    api_key : str
        API key for Polymarket authentication.
    base_url : str
        Root URL for the CLOB API (default: production endpoint).
    """

    BASE_URL = "https://clob.polymarket.com"

    def __init__(self, api_key: str, base_url: str | None = None):
        self.base_url = (base_url or self.BASE_URL).rstrip("/")
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
        })
        self._min_interval = 1.0
        self._last_request_time = 0.0

    def _rate_limit(self) -> None:
        """Enforce minimum interval between consecutive requests."""
        elapsed = time.monotonic() - self._last_request_time
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)

    def get(self, endpoint: str, params: dict | None = None,
            max_retries: int = 5) -> list[dict]:
        """GET with automatic pagination via next_cursor and retry on 429.

        Parameters
        ----------
        endpoint : str
            API path relative to base URL (e.g. "/markets").
        params : dict, optional
            Query parameters for the first request.
        max_retries : int
            Maximum retries per individual page request on 429 errors.

        Returns
        -------
        list[dict]
            Accumulated items across all pages.
        """
        url = f"{self.base_url}{endpoint}"
        params = dict(params or {})
        all_items: list[dict] = []

        while True:
            data = self._request_with_retry(url, params, max_retries)

            if isinstance(data, list):
                all_items.extend(data)
                break

            items_key = self._find_items_key(data)
            if items_key and isinstance(data[items_key], list):
                all_items.extend(data[items_key])

            cursor = data.get("next_cursor")
            if not cursor or cursor == "LTE=":
                break
            params["next_cursor"] = cursor

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
                            max_retries: int) -> dict | list:
        """Execute a single GET with exponential backoff on errors.

        Retries on HTTP errors and network drops (connection reset, timeout).
        """
        backoff = 2.0
        for attempt in range(max_retries + 1):
            self._rate_limit()
            self._last_request_time = time.monotonic()

            try:
                resp = self.session.get(url, params=params, timeout=30)
            except Exception as e:
                if attempt < max_retries:
                    print(f"\n[PolymarketClient] Network error (attempt "
                          f"{attempt + 1}/{max_retries}), retrying in "
                          f"{backoff:.0f}s... ({type(e).__name__})")
                    time.sleep(backoff)
                    backoff = min(backoff * 2, 30)
                    continue
                raise

            if resp.status_code == 200:
                return resp.json()

            if resp.status_code in (429, 500, 502, 503) and attempt < max_retries:
                print(f"\n[PolymarketClient] HTTP {resp.status_code} (attempt "
                      f"{attempt + 1}/{max_retries}), retrying in "
                      f"{backoff:.0f}s...")
                time.sleep(backoff)
                backoff = min(backoff * 2, 30)
                continue

            print(f"\n[PolymarketClient] ERROR {resp.status_code} on {url}: "
                  f"{resp.text[:200]}")
            resp.raise_for_status()

        return {}

    @staticmethod
    def _find_items_key(data: dict) -> str | None:
        """Find the key holding the list of items in a paginated response."""
        for key, val in data.items():
            if key not in ("next_cursor", "cursor") and isinstance(val, list):
                return key
        return None
