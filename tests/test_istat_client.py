"""Tests for the ISTAT SDMX client: cache keying and rate-limit bookkeeping."""

import pytest

from sera.istat_client import IstatClient


class FailingSession:
    def get(self, *args, **kwargs):
        raise ConnectionError("boom")


class TestCachePath:
    def test_no_cache_dir_means_no_path(self):
        assert IstatClient()._get_cache_path("flow", "A..X", 2001, 2025) is None

    def test_distinct_keys_get_distinct_files(self, tmp_path):
        client = IstatClient(cache_dir=tmp_path)
        totals = client._get_cache_path("22_289_DF_DCIS_POPRES1_1", "A..JAN.1.TOTAL.1", 2001, 2025)
        by_age = client._get_cache_path("22_289_DF_DCIS_POPRES1_1", "A..JAN.1..1", 2001, 2025)
        assert totals != by_age

    def test_key_is_sanitized_for_the_filesystem(self, tmp_path):
        client = IstatClient(cache_dir=tmp_path)
        path = client._get_cache_path("flow", 'A/B:C*"?', 2001, 2025)
        assert path.parent == tmp_path
        for forbidden in '/\\:*?"<>|':
            assert forbidden not in path.name

    def test_year_range_in_name(self, tmp_path):
        client = IstatClient(cache_dir=tmp_path)
        assert "2001_2025" in client._get_cache_path("flow", "", 2001, 2025).name


class TestCaching:
    def test_cached_response_short_circuits_the_network(self, tmp_path):
        client = IstatClient(cache_dir=tmp_path)
        client.min_delay = 0
        cache_path = client._get_cache_path("flow", "A..X", 2001, 2025)
        client._save_to_cache(cache_path, "cached,data\n1,2\n")
        client.session = FailingSession()  # would raise on any real request

        data = client.get_data("flow", key="A..X", start_year=2001, end_year=2025)
        assert data == "cached,data\n1,2\n"

    def test_different_key_misses_the_cache(self, tmp_path):
        client = IstatClient(cache_dir=tmp_path)
        client.min_delay = 0
        cache_path = client._get_cache_path("flow", "A..X", 2001, 2025)
        client._save_to_cache(cache_path, "cached")
        client.session = FailingSession()

        with pytest.raises(ConnectionError):
            client.get_data("flow", key="A..OTHER", start_year=2001, end_year=2025)


class TestRateLimitBookkeeping:
    def test_failed_request_still_counts_for_the_rate_limit(self):
        client = IstatClient()
        client.min_delay = 0
        client.session = FailingSession()
        assert client.last_request_time == 0.0

        with pytest.raises(ConnectionError):
            client.get_data("flow", key="A..X")

        assert client.last_request_time > 0.0
