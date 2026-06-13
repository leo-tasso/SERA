"""Tests for indicator provenance classification (measured vs disaggregated)."""

import pandas as pd

from sera.twin.data_loader import DataLoader

VALID = {
    DataLoader.PROVENANCE_MEASURED,
    DataLoader.PROVENANCE_DISAGG_REGIONAL,
    DataLoader.PROVENANCE_DISAGG_NATIONAL,
    DataLoader.PROVENANCE_MIXED,
    DataLoader.PROVENANCE_UNKNOWN,
}


def test_missing_indicator_is_unknown(tmp_path):
    loader = DataLoader(tmp_path)
    assert loader.classify_provenance("nope", "economic") == DataLoader.PROVENANCE_UNKNOWN


def test_province_level_csv_is_measured(tmp_path):
    # A raw file already keyed by 2-letter province sigle is "measured".
    from sera.twin.province_mapping import PROVINCE_SIGLAS_110

    folder = tmp_path / "economic" / "demo_ind"
    folder.mkdir(parents=True)
    rows = [{"area_code": code, "year": 2025, "value": 1.0} for code in PROVINCE_SIGLAS_110]
    pd.DataFrame(rows).to_csv(folder / "demo_ind_raw_2001_2025.csv", index=False)

    loader = DataLoader(tmp_path)
    assert loader.classify_provenance("demo_ind", "economic") == DataLoader.PROVENANCE_MEASURED


def test_national_only_csv_is_disaggregated(tmp_path):
    folder = tmp_path / "economic" / "natl_ind"
    folder.mkdir(parents=True)
    pd.DataFrame([{"area_code": "IT", "year": 2025, "value": 100.0}]).to_csv(
        folder / "natl_ind_raw_2001_2025.csv", index=False
    )
    loader = DataLoader(tmp_path)
    # No population file in tmp_path, but classification reads the raw codes only.
    assert loader.classify_provenance("natl_ind", "economic") == DataLoader.PROVENANCE_DISAGG_NATIONAL


def test_panel_provenance_labels_are_valid(tmp_path):
    from sera.twin.province_mapping import PROVINCE_SIGLAS_110

    folder = tmp_path / "economic" / "demo_ind"
    folder.mkdir(parents=True)
    rows = [{"area_code": code, "year": 2025, "value": 1.0} for code in PROVINCE_SIGLAS_110]
    pd.DataFrame(rows).to_csv(folder / "demo_ind_raw_2001_2025.csv", index=False)

    loader = DataLoader(tmp_path)
    labels = loader.panel_provenance({"demo_ind": ("economic", 1)})
    assert set(labels) == {"demo_ind"}
    assert labels["demo_ind"] in VALID
