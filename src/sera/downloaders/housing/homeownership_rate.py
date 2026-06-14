"""Homeownership rate indicator downloader (ISTAT source)."""

import io
import json
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from sera.config import get_indicator_data_dir
from sera.istat_client import IstatClient


class HomeownershipRateDownloader:
    """Download homeownership rate from ISTAT household dwelling data."""

    def __init__(self):
        self.client = IstatClient()
        self.flow_id = "DF_DCSS_ABITAZIONI_TV_2"
        self.table_mapping: dict[str, Any] = {
            "indicator": "homeownership_rate",
            "source": {
                "provider": "ISTAT",
                "flow_id": self.flow_id,
                "filters": {
                    "INDICATOR": "NUM_OCC_DW_AV",
                    "OWNERSHIP_TYPE": "OWN",
                },
                "columns": {
                    "REF_AREA": "area_code",
                    "TIME_PERIOD": "year",
                    "OBS_VALUE": "owned_dwelling_count",
                },
            },
            "project": {
                "entity": "geography",
                "code_field": "area_code",
                "levels": ["province", "municipality"],
                "notes": "Homeownership rate computed as owned dwellings / all occupied dwellings * 100.",
            },
        }

    def _save_table_mapping(self, output_path: Path) -> Path:
        mapping_path = output_path.with_suffix(".mapping.json")
        mapping_path.parent.mkdir(parents=True, exist_ok=True)
        with open(mapping_path, "w", encoding="utf-8") as handle:
            json.dump(self.table_mapping, handle, indent=2, ensure_ascii=False)
        return mapping_path

    def download_homeownership_rate(
        self, start_year: int = 2001, end_year: int = 2025
    ) -> pd.DataFrame:
        csv_data = self.client.get_data(
            flow_id=self.flow_id, key="", start_year=start_year, end_year=end_year, format="csv"
        )
        df = pd.read_csv(io.StringIO(csv_data), low_memory=False)

        if df.empty:
            return pd.DataFrame(
                columns=[
                    "area_code",
                    "year",
                    "owned_dwellings",
                    "all_occupied_dwellings",
                    "homeownership_rate",
                ]
            )

        df["year"] = pd.to_numeric(df["TIME_PERIOD"], errors="coerce")
        df["OBS_VALUE"] = pd.to_numeric(df["OBS_VALUE"], errors="coerce")
        df = df.dropna(subset=["year", "OBS_VALUE"])
        df = df[(df["year"] >= start_year) & (df["year"] <= end_year)]

        owned = df[
            (df["INDICATOR"].astype(str) == "NUM_OCC_DW_AV")
            & (df["OWNERSHIP_TYPE"].astype(str) == "OWN")
        ]
        total = df[
            (df["INDICATOR"].astype(str) == "NUM_OCC_DW_AV")
            & (df["OWNERSHIP_TYPE"].astype(str) == "ALL")
        ]

        owned = (
            owned[["REF_AREA", "year", "OBS_VALUE"]]
            .copy()
            .rename(columns={"REF_AREA": "area_code", "OBS_VALUE": "owned_dwellings"})
        )
        total = (
            total[["REF_AREA", "year", "OBS_VALUE"]]
            .copy()
            .rename(columns={"REF_AREA": "area_code", "OBS_VALUE": "all_occupied_dwellings"})
        )

        merged = owned.merge(total, on=["area_code", "year"], how="inner")
        merged = merged[merged["all_occupied_dwellings"] > 0]
        merged["homeownership_rate"] = (
            merged["owned_dwellings"] / merged["all_occupied_dwellings"]
        ) * 100.0
        merged = merged.sort_values(["area_code", "year"]).drop_duplicates(
            subset=["area_code", "year"]
        )
        return merged

    def save_homeownership_rate_csv(
        self, output_path: Optional[Path] = None, start_year: int = 2001, end_year: int = 2025
    ) -> Path:
        if output_path is None:
            indicator_dir = get_indicator_data_dir("homeownership_rate")
            output_path = indicator_dir / f"homeownership_rate_raw_{start_year}_{end_year}.csv"
        print(f"Downloading homeownership rate data ({start_year}-{end_year})...")
        df = self.download_homeownership_rate(start_year=start_year, end_year=end_year)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False, encoding="utf-8")
        mapping_path = self._save_table_mapping(output_path)
        print(f"✓ Homeownership rate data saved: {output_path}")
        print(f"✓ Table mapping saved: {mapping_path}")
        print(f"  - {len(df)} records")
        if not df.empty:
            print(f"  - Years: {int(df['year'].min())} to {int(df['year'].max())}")
            print(f"  - Unique areas: {df['area_code'].nunique()}")
        return output_path
