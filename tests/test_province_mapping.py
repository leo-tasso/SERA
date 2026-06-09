"""Tests for canonical province code mapping."""

from sera.twin.province_mapping import (
    PROVINCE_SIGLAS_110,
    PROVINCE_TO_REGION,
    map_area_code_to_sigla,
)


class TestProvinceCanon:
    def test_110_unique_siglas(self):
        assert len(PROVINCE_SIGLAS_110) == 110
        assert len(set(PROVINCE_SIGLAS_110)) == 110

    def test_siglas_are_sorted(self):
        """The simulator aligns state and parameter frames on sorted area_code;
        the canonical list must stay alphabetically sorted."""
        assert PROVINCE_SIGLAS_110 == sorted(PROVINCE_SIGLAS_110)

    def test_every_sigla_has_a_region(self):
        assert set(PROVINCE_SIGLAS_110) <= set(PROVINCE_TO_REGION)


class TestMapAreaCode:
    def test_passthrough_sigla(self):
        assert map_area_code_to_sigla("TO") == "TO"
        assert map_area_code_to_sigla(" mi ") == "MI"

    def test_iso_format(self):
        assert map_area_code_to_sigla("IT-MI") == "MI"

    def test_nuts_code(self):
        assert map_area_code_to_sigla("ITC45") == "MI"
        assert map_area_code_to_sigla("ITF33") == "NA"

    def test_legacy_istat_prefix(self):
        assert map_area_code_to_sigla("IT001") == "TO"

    def test_municipal_six_digit(self):
        assert map_area_code_to_sigla("001001") == "TO"

    def test_numeric_code_with_lost_leading_zeros(self):
        # "1001" should be zero-filled back to "001001" -> Torino.
        assert map_area_code_to_sigla("1001") == "TO"

    def test_invalid_inputs(self):
        assert map_area_code_to_sigla(None) is None
        assert map_area_code_to_sigla("") is None
        assert map_area_code_to_sigla("nan") is None
        assert map_area_code_to_sigla("XX") is None
