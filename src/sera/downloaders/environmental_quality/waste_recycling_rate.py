"""Waste recycling rate downloader (ISTAT source)."""

import io
import json
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from sera.config import get_indicator_data_dir
from sera.istat_client import IstatClient


class WasteRecyclingRateDownloader:
    """Download waste recycling rate from ISTAT."""

    def __init__(self):
        self.client = IstatClient()
        self.flow_id = "609_1_DF_DCCV_URBANENV_22"
        self.table_mapping: dict[str, Any] = {
            "indicator": "waste_recycling_rate",
            "source": {
                "provider": "ISTAT",
                "flow_id": self.flow_id,
                "filters": {},
                "columns": {
                    "REF_AREA": "area_code",
                    "TIME_PERIOD": "year",
                    "OBS_VALUE": "waste_recycling_rate",
                },
            },
            "project": {
                "entity": "geography",
                "code_field": "area_code",
                "levels": ["national", "regional"],
                "notes": "Waste recycling/collection service rate from ISTAT urban environment data.",
            },
        }

    def _save_table_mapping(self, output_path: Path) -> Path:
        mapping_path = output_path.with_suffix(".mapping.json")
        mapping_path.parent.mkdir(parents=True, exist_ok=True)
        with open(mapping_path, "w", encoding="utf-8") as handle:
            json.dump(self.table_mapping, handle, indent=2, ensure_ascii=False)
        return mapping_path

    def download_waste_recycling_rate(self, start_year: int = 2001, end_year: int = 2025) -> pd.DataFrame:
        csv_data = self.client.get_data(flow_id=self.flow_id, key="", start_year=start_year, end_year=end_year, format="csv")
        df = pd.read_csv(io.StringIO(csv_data), low_memory=False)
        if df.empty:
            return pd.DataFrame(columns=["area_code", "year", "waste_recycling_rate"])
        df_clean = df[["REF_AREA", "TIME_PERIOD", "OBS_VALUE"]].copy()
        df_clean.columns = ["area_code", "year", "waste_recycling_rate"]
        df_clean["year"] = pd.to_numeric(df_clean["year"], errors="coerce")
        df_clean["waste_recycling_rate"] = pd.to_numeric(df_clean["waste_recycling_rate"], errors="coerce")
        df_clean = df_clean.dropna(subset=["year", "waste_recycling_rate"])
        df_clean = df_clean[(df_clean["year"] >= start_year) & (df_clean["year"] <= end_year)]
        return df_clean.sort_values(["area_code", "year"])

    def save_waste_recycling_rate_csv(self, output_path: Optional[Path] = None, start_year: int = 2001, end_year: int = 2025) -> Path:
        if output_path is None:
            indicator_dir = get_indicator_data_dir("waste_recycling_rate")
            output_path = indicator_dir / f"waste_recycling_rate_raw_{start_year}_{end_year}.csv"
        print(f"Downloading waste recycling rate data ({start_year}-{end_year})...")
        df = self.download_waste_recycling_rate(start_year=start_year, end_year=end_year)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False, encoding="utf-8")
        mapping_path = self._save_table_mapping(output_path)
        print(f"✓ Waste recycling rate data saved: {output_path}")
        print(f"✓ Table mapping saved: {mapping_path}")
        print(f"  - {len(df)} records")
        if not df.empty:
            print(f"  - Years: {int(df['year'].min())} to {int(df['year'].max())}")
            print(f"  - Unique areas: {df['area_code'].nunique()}")
        return output_path
