"""Broadband penetration downloader (ISTAT source)."""

import io
import json
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from sera.config import get_indicator_data_dir
from sera.istat_client import IstatClient


class BroadbandPenetrationDownloader:
    """Download household broadband penetration from ISTAT ICT survey data."""

    def __init__(self):
        self.client = IstatClient()
        self.flow_id = "60_130_DF_DCCV_ICT_1"
        self.table_mapping: dict[str, Any] = {
            "indicator": "broadband_penetration",
            "source": {
                "provider": "ISTAT",
                "flow_id": self.flow_id,
                "filters": {
                    "DATA_TYPE": "FAM_CONN_BROAD",
                    "SEX": "9",
                    "AGE": "TOTAL",
                    "HOUSEHOLD_TYPOLOGY": "HH",
                },
                "columns": {
                    "REF_AREA": "area_code",
                    "TIME_PERIOD": "year",
                    "OBS_VALUE": "broadband_penetration",
                },
            },
            "project": {
                "entity": "geography",
                "code_field": "area_code",
                "levels": ["national"],
                "notes": "Share of households with broadband connection.",
            },
        }

    def _save_table_mapping(self, output_path: Path) -> Path:
        mapping_path = output_path.with_suffix(".mapping.json")
        mapping_path.parent.mkdir(parents=True, exist_ok=True)
        with open(mapping_path, "w", encoding="utf-8") as handle:
            json.dump(self.table_mapping, handle, indent=2, ensure_ascii=False)
        return mapping_path

    def download_broadband_penetration(self, start_year: int = 2001, end_year: int = 2025) -> pd.DataFrame:
        csv_data = self.client.get_data(flow_id=self.flow_id, key="", start_year=start_year, end_year=end_year, format="csv")
        df = pd.read_csv(io.StringIO(csv_data), low_memory=False)
        for column, expected in self.table_mapping["source"]["filters"].items():
            if column in df.columns:
                df = df[df[column].astype(str) == expected]
        if df.empty:
            return pd.DataFrame(columns=["area_code", "year", "broadband_penetration"])
        df_clean = df[["REF_AREA", "TIME_PERIOD", "OBS_VALUE"]].copy()
        df_clean.columns = ["area_code", "year", "broadband_penetration"]
        df_clean["year"] = pd.to_numeric(df_clean["year"], errors="coerce")
        df_clean["broadband_penetration"] = pd.to_numeric(df_clean["broadband_penetration"], errors="coerce")
        df_clean = df_clean.dropna(subset=["year", "broadband_penetration"])
        df_clean = df_clean[(df_clean["year"] >= start_year) & (df_clean["year"] <= end_year)]
        return df_clean.sort_values(["area_code", "year"])

    def save_broadband_penetration_csv(self, output_path: Optional[Path] = None, start_year: int = 2001, end_year: int = 2025) -> Path:
        if output_path is None:
            indicator_dir = get_indicator_data_dir("broadband_penetration")
            output_path = indicator_dir / f"broadband_penetration_raw_{start_year}_{end_year}.csv"
        print(f"Downloading broadband penetration data ({start_year}-{end_year})...")
        df = self.download_broadband_penetration(start_year=start_year, end_year=end_year)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False, encoding="utf-8")
        mapping_path = self._save_table_mapping(output_path)
        print(f"✓ Broadband penetration data saved: {output_path}")
        print(f"✓ Table mapping saved: {mapping_path}")
        print(f"  - {len(df)} records")
        if not df.empty:
            print(f"  - Years: {int(df['year'].min())} to {int(df['year'].max())}")
            print(f"  - Unique areas: {df['area_code'].nunique()}")
        return output_path
