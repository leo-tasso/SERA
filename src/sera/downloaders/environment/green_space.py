"""Green space indicator downloader (ISTAT BES source)."""

import io
import json
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from sera.config import get_indicator_data_dir
from sera.istat_client import IstatClient


class GreenSpaceDownloader:
    """Download green space availability from ISTAT BES territorial environment indicators."""

    def __init__(self):
        self.client = IstatClient()
        self.flow_id = "DF_BES_TERRIT_10"

        self.table_mapping: dict[str, Any] = {
            "indicator": "green_space",
            "source": {
                "provider": "ISTAT",
                "flow_id": self.flow_id,
                "filters": {
                    "DOMAIN": "BES_10",
                    "DATA_TYPE": "10AMB008",
                    "SEX": "T",
                },
                "columns": {
                    "REF_AREA": "area_code",
                    "TIME_PERIOD": "year",
                    "OBS_VALUE": "urban_green_availability",
                },
            },
            "project": {
                "entity": "geography",
                "code_field": "area_code",
                "levels": ["province", "region", "national"],
                "notes": "Green space proxy from BES: urban green availability.",
            },
        }

    def _save_table_mapping(self, output_path: Path) -> Path:
        mapping_path = output_path.with_suffix(".mapping.json")
        mapping_path.parent.mkdir(parents=True, exist_ok=True)
        with open(mapping_path, "w", encoding="utf-8") as handle:
            json.dump(self.table_mapping, handle, indent=2, ensure_ascii=False)
        return mapping_path

    def download_green_space(self, start_year: int = 2001, end_year: int = 2025) -> pd.DataFrame:
        csv_data = self.client.get_data(
            flow_id=self.flow_id,
            key="",
            start_year=start_year,
            end_year=end_year,
            format="csv",
        )
        df = pd.read_csv(io.StringIO(csv_data), low_memory=False)

        for column, expected in self.table_mapping["source"]["filters"].items():
            if column in df.columns:
                df = df[df[column].astype(str) == expected]

        if df.empty:
            return pd.DataFrame(columns=["area_code", "year", "urban_green_availability"])

        df_clean = df[["REF_AREA", "TIME_PERIOD", "OBS_VALUE"]].copy()
        df_clean.columns = ["area_code", "year", "urban_green_availability"]
        df_clean["year"] = pd.to_numeric(df_clean["year"], errors="coerce")
        df_clean["urban_green_availability"] = pd.to_numeric(
            df_clean["urban_green_availability"], errors="coerce"
        )
        df_clean = df_clean.dropna(subset=["year", "urban_green_availability"])
        df_clean = df_clean[(df_clean["year"] >= start_year) & (df_clean["year"] <= end_year)]
        df_clean = df_clean.sort_values(["area_code", "year"]).drop_duplicates(
            subset=["area_code", "year"]
        )
        return df_clean

    def save_green_space_csv(
        self,
        output_path: Optional[Path] = None,
        start_year: int = 2001,
        end_year: int = 2025,
    ) -> Path:
        if output_path is None:
            indicator_dir = get_indicator_data_dir("green_space")
            output_path = indicator_dir / f"green_space_raw_{start_year}_{end_year}.csv"

        print(f"Downloading green space data ({start_year}-{end_year})...")
        df = self.download_green_space(start_year=start_year, end_year=end_year)

        if df.empty:
            print("Warning: No green space data retrieved, skipping save.")
            return output_path

        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False, encoding="utf-8")
        mapping_path = self._save_table_mapping(output_path)

        print(f"✓ Green space data saved: {output_path}")
        print(f"✓ Table mapping saved: {mapping_path}")
        print(f"  - {len(df)} records")
        print(f"  - Years: {int(df['year'].min())} to {int(df['year'].max())}")
        print(f"  - Unique areas: {df['area_code'].nunique()}")
        return output_path
