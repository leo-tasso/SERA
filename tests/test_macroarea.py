"""Tests for the North/Centre/South macro-area mapping."""

from sera.twin.province_mapping import (
    PROVINCE_SIGLAS_110,
    PROVINCE_TO_MACROAREA,
    macroarea_of,
)


def test_every_province_has_a_macroarea():
    missing = [s for s in PROVINCE_SIGLAS_110 if macroarea_of(s) is None]
    assert not missing, f"provinces without a macro-area: {missing}"


def test_areas_are_the_three_groups():
    assert set(PROVINCE_TO_MACROAREA.values()) == {"North", "Centre", "South"}


def test_known_provinces():
    assert macroarea_of("MI") == "North"  # Milano, Lombardia
    assert macroarea_of("RM") == "Centre"  # Roma, Lazio
    assert macroarea_of("NA") == "South"  # Napoli, Campania
    assert macroarea_of("PA") == "South"  # Palermo, Sicilia (Isole)
    assert macroarea_of("mi") is not None  # case-insensitive


def test_partition_counts_are_reasonable():
    from collections import Counter

    counts = Counter(PROVINCE_TO_MACROAREA.values())
    # Italy: ~46 North, ~22 Centre, ~42 South provinces.
    assert counts["North"] > counts["Centre"]
    assert counts["South"] > counts["Centre"]
    assert sum(counts.values()) == len(PROVINCE_SIGLAS_110)
