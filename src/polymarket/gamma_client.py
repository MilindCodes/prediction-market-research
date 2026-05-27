import time
import requests


class GammaClient:
    """HTTP client for the Polymarket Gamma API (public, no auth required).

    The Gamma API provides rich market metadata including volume, category,
    resolution status, and UMA oracle dispute information. No API key needed.

    Used for catalog discovery and filtering. Trade-level data still comes
    from the CLOB API via PolymarketClient.
    """

    BASE_URL = "https://gamma-api.polymarket.com"
    PAGE_SIZE = 20  # Gamma API returns max 20 per page

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({"Accept": "application/json"})
        self._min_interval = 0.5
        self._last_request_time = 0.0

    def _rate_limit(self) -> None:
        """Enforce minimum interval between consecutive requests."""
        elapsed = time.monotonic() - self._last_request_time
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)

    def get_all_markets(self, end_date_min: str | None = None,
                        end_date_max: str | None = None,
                        **filters) -> list[dict]:
        """Paginate through the /markets endpoint and return all results.

        Uses offset pagination up to the API's 10 000-row limit, then
        automatically switches to /markets/keyset for deeper pages.

        Parameters
        ----------
        end_date_min : str, optional
            ISO date string (e.g. "2022-01-01"). Server-side filter that
            skips markets which closed before this date.
        end_date_max : str, optional
            ISO date string (e.g. "2024-12-31"). Client-side early-stop:
            once every market on a page closed after this date, stop.
        **filters
            Additional query parameters (e.g., closed=True).

        Returns
        -------
        list[dict]
            All market objects matching the filters.
        """
        all_items: list[dict] = []
        offset = 0
        use_keyset = False
        next_cursor: str | None = None

        base_params = dict(filters)
        if end_date_min:
            base_params["end_date_min"] = end_date_min

        while True:
            if use_keyset:
                # Keyset pagination: /markets/keyset?next_cursor=...
                params = dict(base_params)
                params["limit"] = self.PAGE_SIZE
                if next_cursor:
                    params["next_cursor"] = next_cursor
                data = self._request("/markets/keyset", params)
                # keyset endpoint returns {"data": [...], "next_cursor": "..."}
                if isinstance(data, dict):
                    next_cursor = data.get("next_cursor") or data.get("cursor")
                    data = data.get("data", data.get("markets", []))
            else:
                params = dict(base_params)
                params["offset"] = offset
                params["limit"] = self.PAGE_SIZE
                data = self._request("/markets", params)
                # 422 = offset too large → switch to keyset
                if data is None:
                    use_keyset = True
                    next_cursor = None
                    print(f"\n  Switching to keyset pagination at {len(all_items)} markets...")
                    continue

            if not data or not isinstance(data, list):
                break

            all_items.extend(data)
            print(f"  Fetched {len(all_items)} markets so far...", end="\r")

            # Early-stop: all markets on this page closed after end_date_max.
            if end_date_max and data:
                end_dates = [
                    (d.get("endDateIso") or d.get("endDate") or "")[:10]
                    for d in data
                ]
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

    def get_all_events(self, end_date_min: str | None = None,
                       end_date_max: str | None = None,
                       **filters) -> list[dict]:
        """Paginate through the /events endpoint and return all results.

        Parameters
        ----------
        end_date_min : str, optional
            ISO date string. Server-side filter — skips events that closed
            before this date.
        end_date_max : str, optional
            ISO date string. Client-side early-stop once all events on a
            page have closed after this date.
        **filters
            Additional query parameters passed to the API (e.g., closed=True).

        Returns
        -------
        list[dict]
            All event objects matching the filters.
        """
        all_items: list[dict] = []
        offset = 0

        base_params = dict(filters)
        if end_date_min:
            base_params["end_date_min"] = end_date_min

        while True:
            params = dict(base_params)
            params["offset"] = offset
            params["limit"] = self.PAGE_SIZE

            data = self._request("/events", params)

            if not data or not isinstance(data, list):
                break

            all_items.extend(data)
            print(f"  Fetched {len(all_items)} events so far...", end="\r")

            # Early-stop: all events on this page are past the max date.
            if end_date_max and data:
                end_dates = [
                    (d.get("endDate") or "")[:10]
                    for d in data
                ]
                if all(ed > end_date_max for ed in end_dates if ed):
                    break

            if len(data) < self.PAGE_SIZE:
                break

            offset += self.PAGE_SIZE

        print(f"  Fetched {len(all_items)} events total.        ")
        return all_items

    def _request(self, endpoint: str, params: dict,
                 max_retries: int = 5) -> list | dict | None:
        """Execute a GET request with retry on errors.

        Retries on HTTP errors and network drops (connection reset, timeout).
        """
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
                    print(f"\n  [GammaClient] HTTP {resp.status_code} (attempt "
                          f"{attempt + 1}/{max_retries}), retrying in "
                          f"{backoff:.0f}s...")
                    time.sleep(backoff)
                    backoff = min(backoff * 2, 30)
                    continue

                print(f"\n  [GammaClient] ERROR {resp.status_code} on {endpoint}: "
                      f"{resp.text[:200]}")
                return None

            except Exception as e:
                if attempt < max_retries:
                    print(f"\n  [GammaClient] Network error (attempt "
                          f"{attempt + 1}/{max_retries}), retrying in "
                          f"{backoff:.0f}s... ({type(e).__name__})")
                    time.sleep(backoff)
                    backoff = min(backoff * 2, 30)
                    continue
                print(f"\n  [GammaClient] Failed after {max_retries} retries: {e}")
                return None

        return None
