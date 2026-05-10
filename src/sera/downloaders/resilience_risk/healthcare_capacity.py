"""Healthcare capacity downloader (ISTAT source - reuses hospital beds)."""

import io
import json
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from sera.config import get_indicator_data_dir
from sera.istat_client import IstatClient


class HealthcareCapacityDownloader:
    """Download healthcare capacity from ISTAT (total hospital beds as proxy)."""

    def __init__(self):
        self.client = IstatClient()
        self.flow_id = "43_967_DF_DCIS_OSPED_2"
        self.table_mapping: dict[str, Any] = {
            "indicator": "healthcare_capacity",
            "source": {
                "provider": "ISTAT",
                "flow_id": self.flow_id,
                "filters": {},
                "columns": {
                    "REF_AREA": "area_code",
                    "TIME_PERIOD": "year",
                    "OBS_VALUE": "healthcare_capacity",
                },
            },
            "project": {
                "entity": "geography",
                "code_field": "area_code",
                "levels": ["national", "regional"],
                "notes": "Healthcare capacity proxy: total hospital beds from ISTAT.",
            },
        }

    def _save_table_mapping(self, output_path: Path) -> Path:
        mapping_path = output_path.with_suffix(".mapping.json")
        mapping_path.parent.mkdir(parents=True, exist_ok=True)
        with open(mapping_path, "w", encoding="utf-8") as handle:
            json.dump(self.table_mapping, handle, indent=2, ensure_ascii=False)
        return mapping_path

    def download_healthcare_capacity(self, start_year: int = 2001, end_year: int = 2025) -> pd.DataFrame:
        csv_data = self.client.get_data(flow_id=self.flow_id, key="", start_year=start_year, end_year=end_year, format="csv")
        df = pd.read_csv(io.StringIO(csv_data), low_memory=False)
        if df.empty:
            return pd.DataFrame(columns=["area_code", "year", "healthcare_capacity"])
        df_clean = df[["REF_AREA", "TIME_PERIOD", "OBS_VALUE"]].copy()
        df_clean.columns = ["area_code", "year", "healthcare_capacity"]
        df_clean["year"] = pd.to_numeric(df_clean["year"], errors="coerce")
        df_clean["healthcare_capacity"] = pd.to_numeric(df_clean["healthcare_capacity"], errors="coerce")
        df_clean = df_clean.dropna(subset=["year", "healthcare_capacity"])
        df_clean = df_clean[(df_clean["year"] >= start_year) & (df_clean["year"] <= end_year)]
        return df_clean.sort_values(["area_code", "year"])

    def save_healthcare_capacity_csv(self, output_path: Optional[Path] = None, start_year: int = 2001, end_year: int = 2025) -> Path:
        if output_path is None:
            indicator_dir = get_indicator_data_dir("healthcare_capacity")
            output_path = indicator_dir / f"healthcare_capacity_raw_{start_year}_{end_year}.csv"
        print(f"Downloading healthcare capacity data ({start_year}-{end_year})...")
        df = self.download_healthcare_capacity(start_year=start_year, end_year=end_year)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False, encoding="utf-8")
        mapping_path = self._save_table_mapping(output_path)
        print(f"✓ Healthcare capacity data saved: {output_path}")
        print(f"✓ Table mapping saved: {mapping_path}")
        print(f"  - {len(df)} records")
        if not df.empty:
            print(f"  - Years: {int(df['year'].min())} to {int(df['year'].max())}")
            print(f"  - Unique areas: {df['area_code'].nunique()}")
        return output_path
