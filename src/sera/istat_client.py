"""ISTAT SDMX API client with rate limiting."""

import re
import time
from typing import Optional, Dict, Any
from datetime import datetime
import requests
from pathlib import Path

from sera.config import (
    ISTAT_BASE_URL,
    ISTAT_MIN_DELAY_SECONDS,
    ENCODING,
)


class IstatClient:
    """ISTAT SDMX API client respecting rate limits."""

    def __init__(self, cache_dir: Optional[Path] = None):
        """Initialize ISTAT client.

        Args:
            cache_dir: Optional directory to cache API responses.
        """
        self.base_url = ISTAT_BASE_URL
        self.min_delay = ISTAT_MIN_DELAY_SECONDS
        self.last_request_time = 0.0
        self.cache_dir = cache_dir
        self.session = requests.Session()

    def _enforce_rate_limit(self) -> None:
        """Enforce minimum delay between requests."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_delay:
            delay = self.min_delay - elapsed
            time.sleep(delay)

    def _get_cache_path(
        self,
        flow_id: str,
        key: str = "",
        year_start: Optional[int] = None,
        year_end: Optional[int] = None,
    ) -> Optional[Path]:
        """Generate cache file path for a dataflow query.

        The dimensional ``key`` must be part of the file name: different keys on
        the same dataflow return different data (e.g. population totals vs. the
        age breakdown on 22_289_DF_DCIS_POPRES1_1) and would otherwise collide.
        """
        if not self.cache_dir:
            return None

        key_part = f"_{re.sub(r'[^A-Za-z0-9._-]', '_', key)}" if key else ""
        suffix = f"_{year_start}_{year_end}" if year_start and year_end else ""
        cache_file = f"{flow_id}{key_part}{suffix}.csv"
        return self.cache_dir / cache_file

    def _get_from_cache(self, cache_path: Path) -> Optional[str]:
        """Retrieve cached data if available."""
        if cache_path and cache_path.exists():
            with open(cache_path, "r", encoding=ENCODING) as f:
                return f.read()
        return None

    def _save_to_cache(self, cache_path: Path, data: str) -> None:
        """Save response data to cache."""
        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, "w", encoding=ENCODING) as f:
                f.write(data)

    def get_data(
        self,
        flow_id: str,
        key: str = "",
        start_year: Optional[int] = None,
        end_year: Optional[int] = None,
        format: str = "csv",
    ) -> str:
        """Fetch data from ISTAT SDMX API.

        Args:
            flow_id: ISTAT dataflow ID (e.g., "22_289_DF_DCIS_POPRES1_1")
            key: Dimensional key filter (e.g., "A..JAN.1.TOTAL.1")
            start_year: Start year for data (inclusive)
            end_year: End year for data (inclusive)
            format: Output format ("csv" or "json")

        Returns:
            Data as string in requested format.

        Raises:
            requests.HTTPError: If API request fails.
        """
        # Check cache first
        cache_path = self._get_cache_path(flow_id, key, start_year, end_year)
        cached_data = self._get_from_cache(cache_path)
        if cached_data:
            return cached_data

        # Enforce rate limit
        self._enforce_rate_limit()

        # Build URL
        url = f"{self.base_url}/data/{flow_id}"
        if key:
            url += f"/{key}"

        # Set up headers for format negotiation
        headers = {}
        if format.lower() == "csv":
            headers["Accept"] = "application/vnd.sdmx.data+csv;version=1.0.0"
        elif format.lower() == "json":
            headers["Accept"] = "application/json"

        # Add temporal filters
        params = {}
        if start_year:
            params["startPeriod"] = str(start_year)
        if end_year:
            # NOTE: ISTAT has a bug where endPeriod returns year+1, so we subtract 1
            params["endPeriod"] = str(end_year - 1)

        # Execute request. The request timestamp is recorded even on failure so
        # that retrying a failing call still respects the API rate limit.
        try:
            response = self.session.get(url, headers=headers, params=params, timeout=300)
            response.raise_for_status()
        finally:
            self.last_request_time = time.time()

        data = response.text
        self._save_to_cache(cache_path, data)

        return data

    def get_dataflow_metadata(self, agency_id: str = "IT1") -> str:
        """Fetch dataflow metadata (list of available datasets).

        Args:
            agency_id: ISTAT agency ID (default "IT1")

        Returns:
            Dataflow metadata as XML string.

        Raises:
            requests.HTTPError: If API request fails.
        """
        self._enforce_rate_limit()

        url = f"{self.base_url}/dataflow/{agency_id}"
        try:
            response = self.session.get(url, timeout=300)
            response.raise_for_status()
        finally:
            self.last_request_time = time.time()

        return response.text
