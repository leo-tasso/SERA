"""Agricultural productivity downloader (ISTAT source)."""

import io
import json
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from sera.config import get_indicator_data_dir
from sera.istat_client import IstatClient


class AgriculturalProductivityDownloader:
    """Download agricultural productivity from ISTAT."""

    def __init__(self):
        self.client = IstatClient()
        self.flow_id = "101_148_DF_DCSP_RICAREA_1"
        self.table_mapping: dict[str, Any] = {
            "indicator": "agricultural_productivity",
            "source": {
                "provider": "ISTAT",
                "flow_id": self.flow_id,
                "filters": {},
                "columns": {
                    "REF_AREA": "area_code",
                    "TIME_PERIOD": "year",
                    "OBS_VALUE": "agricultural_productivity",
                },
            },
            "project": {
                "entity": "geography",
                "code_field": "area_code",
                "levels": ["regional"],
                "notes": "Agricultural productivity proxy from economic results of farms (key data).",
            },
        }

    def _save_table_mapping(self, output_path: Path) -> Path:
        mapping_path = output_path.with_suffix(".mapping.json")
        mapping_path.parent.mkdir(parents=True, exist_ok=True)
        with open(mapping_path, "w", encoding="utf-8") as handle:
            json.dump(self.table_mapping, handle, indent=2, ensure_ascii=False)
        return mapping_path

    def download_agricultural_productivity(
        self, start_year: int = 2001, end_year: int = 2025
    ) -> pd.DataFrame:
        csv_data = self.client.get_data(
            flow_id=self.flow_id, key="", start_year=start_year, end_year=end_year, format="csv"
        )
        df = pd.read_csv(io.StringIO(csv_data), low_memory=False)
        if df.empty:
            return pd.DataFrame(columns=["area_code", "year", "agricultural_productivity"])
        df_clean = df[["REF_AREA", "TIME_PERIOD", "OBS_VALUE"]].copy()
        df_clean.columns = ["area_code", "year", "agricultural_productivity"]
        df_clean["year"] = pd.to_numeric(df_clean["year"], errors="coerce")
        df_clean["agricultural_productivity"] = pd.to_numeric(
            df_clean["agricultural_productivity"], errors="coerce"
        )
        df_clean = df_clean.dropna(subset=["year", "agricultural_productivity"])
        df_clean = df_clean[(df_clean["year"] >= start_year) & (df_clean["year"] <= end_year)]
        return df_clean.sort_values(["area_code", "year"])

    def save_agricultural_productivity_csv(
        self, output_path: Optional[Path] = None, start_year: int = 2001, end_year: int = 2025
    ) -> Path:
        if output_path is None:
            indicator_dir = get_indicator_data_dir("agricultural_productivity")
            output_path = (
                indicator_dir / f"agricultural_productivity_raw_{start_year}_{end_year}.csv"
            )
        print(f"Downloading agricultural productivity data ({start_year}-{end_year})...")
        df = self.download_agricultural_productivity(start_year=start_year, end_year=end_year)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False, encoding="utf-8")
        mapping_path = self._save_table_mapping(output_path)
        print(f"✓ Agricultural productivity data saved: {output_path}")
        print(f"✓ Table mapping saved: {mapping_path}")
        print(f"  - {len(df)} records")
        if not df.empty:
            print(f"  - Years: {int(df['year'].min())} to {int(df['year'].max())}")
            print(f"  - Unique areas: {df['area_code'].nunique()}")
        return output_path
