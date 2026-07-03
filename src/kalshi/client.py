from __future__ import annotations

import base64
import time
from pathlib import Path
from urllib.parse import urlparse

import requests

try:
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import padding as asym_padding
    _HAS_CRYPTO = True
except ImportError:
    _HAS_CRYPTO = False


class KalshiClient:
    """HTTP client for the Kalshi API v2 with pagination, rate limiting,
    and exponential backoff on 429 responses.

    Authentication options (use whichever you have):
      - RSA key file (recommended): pass key_id + key_file path to the .pem
        you downloaded from the Kalshi dashboard.
      - Bearer token: pass api_key as a string.
      - No auth: public read-only endpoints on the Elections API work without
        credentials.

    Parameters
    ----------
    api_key : str, optional
        Bearer token for simple auth (Elections API public reads).
    key_id : str, optional
        API key ID from the Kalshi dashboard (used with key_file).
    key_file : str, optional
        Path to the RSA private key .pem downloaded from Kalshi.
    base_url : str, optional
        Root URL for the API (default: Kalshi Elections production endpoint).
    """

    BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"

    def __init__(
        self,
        api_key: str = "",
        key_id: str = "",
        key_file: str = "",
        base_url: str | None = None,
    ):
        self.base_url = (base_url or self.BASE_URL).rstrip("/")
        self.session = requests.Session()
        self.session.headers.update({"Accept": "application/json"})

        self._private_key = None
        self._key_id = key_id

        if key_file:
            if not _HAS_CRYPTO:
                raise ImportError(
                    "RSA key file auth requires the cryptography package: "
                    "pip install cryptography"
                )
            key_path = Path(key_file)
            if not key_path.exists():
                raise FileNotFoundError(
                    f"Kalshi key file not found: {key_file}\n"
                    "Download it from kalshi.com → Settings → API Keys."
                )
            with open(key_path, "rb") as fh:
                self._private_key = serialization.load_pem_private_key(
                    fh.read(), password=None
                )
        elif api_key:
            self.session.headers["Authorization"] = f"Bearer {api_key}"

        self._min_interval = 1.0
        self._last_request_time = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(
        self,
        endpoint: str,
        params: dict | None = None,
        max_retries: int = 5,
        max_pages: int | None = None,
        progress_label: str = "",
    ) -> list[dict]:
        """GET with automatic pagination via cursor and retry on 429.

        Parameters
        ----------
        endpoint : str
            API path relative to base URL (e.g. "/trades").
        params : dict, optional
            Query parameters for the first request.
        max_retries : int
            Maximum retries per individual page request on 429 errors.
        max_pages : int, optional
            Hard cap on number of pages fetched. None = unlimited.
        progress_label : str
            Label printed with page progress.

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
                print(
                    f"  [{progress_label or endpoint}] "
                    f"Reached page limit ({max_pages}), stopping."
                )
                break

            if progress_label and page % 5 == 1:
                print(
                    f"  [{progress_label}] Fetching page {page}... "
                    f"({len(all_items)} so far)",
                    end="\r",
                )

            data = self._request_with_retry(url, endpoint, params, max_retries)

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

    def get_single(
        self,
        endpoint: str,
        params: dict | None = None,
        max_retries: int = 5,
    ) -> dict:
        """GET a single resource (no pagination)."""
        url = f"{self.base_url}{endpoint}"
        return self._request_with_retry(url, endpoint, params or {}, max_retries)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _rsa_headers(self, endpoint: str) -> dict:
        """Build per-request RSA-PSS signed headers for the Kalshi API."""
        timestamp_ms = str(int(time.time() * 1000))
        # Signing message: timestamp + method + full URL path (no query string)
        parsed = urlparse(f"{self.base_url}{endpoint}")
        path = parsed.path
        msg = f"{timestamp_ms}GET{path}"
        sig = self._private_key.sign(  # type: ignore[union-attr]
            msg.encode("utf-8"),
            asym_padding.PSS(
                mgf=asym_padding.MGF1(hashes.SHA256()),
                salt_length=asym_padding.PSS.MAX_LENGTH,
            ),
            hashes.SHA256(),
        )
        return {
            "KALSHI-ACCESS-KEY": self._key_id,
            "KALSHI-ACCESS-TIMESTAMP": timestamp_ms,
            "KALSHI-ACCESS-SIGNATURE": base64.b64encode(sig).decode("utf-8"),
        }

    def _rate_limit(self) -> None:
        elapsed = time.monotonic() - self._last_request_time
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)

    def _request_with_retry(
        self, url: str, endpoint: str, params: dict, max_retries: int
    ) -> dict:
        """Execute a single GET with exponential backoff on errors."""
        backoff = 2.0
        for attempt in range(max_retries + 1):
            self._rate_limit()
            self._last_request_time = time.monotonic()

            extra_headers = self._rsa_headers(endpoint) if self._private_key else {}

            try:
                resp = self.session.get(
                    url, params=params, headers=extra_headers, timeout=30
                )
            except requests.exceptions.ConnectionError as e:
                if attempt < max_retries:
                    print(
                        f"\n[KalshiClient] Network error (attempt "
                        f"{attempt + 1}/{max_retries}), retrying in "
                        f"{backoff:.0f}s... ({e})"
                    )
                    time.sleep(backoff)
                    backoff = min(backoff * 2, 30)
                    continue
                raise
            except requests.exceptions.Timeout:
                if attempt < max_retries:
                    print(
                        f"\n[KalshiClient] Timeout (attempt "
                        f"{attempt + 1}/{max_retries}), retrying in "
                        f"{backoff:.0f}s..."
                    )
                    time.sleep(backoff)
                    backoff = min(backoff * 2, 30)
                    continue
                raise

            if resp.status_code == 200:
                return resp.json()

            if resp.status_code in (429, 500, 502, 503) and attempt < max_retries:
                print(
                    f"\n[KalshiClient] HTTP {resp.status_code} (attempt "
                    f"{attempt + 1}/{max_retries}), retrying in "
                    f"{backoff:.0f}s..."
                )
                time.sleep(backoff)
                backoff = min(backoff * 2, 30)
                continue

            print(
                f"\n[KalshiClient] ERROR {resp.status_code} on {url}: "
                f"{resp.text[:200]}"
            )
            resp.raise_for_status()

        return {}

    @staticmethod
    def _find_items_key(data: dict) -> str | None:
        for key, val in data.items():
            if key not in ("cursor", "milestones") and isinstance(val, list):
                return key
        return None
