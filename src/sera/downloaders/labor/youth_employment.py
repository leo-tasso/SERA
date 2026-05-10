"""Youth employment indicator downloader (ISTAT source - regional data)."""

import io
import json
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from sera.config import get_indicator_data_dir
from sera.istat_client import IstatClient


class YouthEmploymentDownloader:
    """Download youth employment for Italy regions from ISTAT."""

    def __init__(self):
        self.client = IstatClient()
        self.flow_id = "150_915_DF_DCCV_TAXOCCU1_YOUTH_1"  # Youth employment rates

        self.table_mapping: dict[str, Any] = {
            "indicator": "youth_employment",
            "source": {
                "provider": "ISTAT",
                "flow_id": self.flow_id,
                "columns": {
                    "REF_AREA": "area_code",
                    "TIME_PERIOD": "year",
                    "OBS_VALUE": "youth_employment_rate",
                },
            },
            "project": {
                "entity": "geography",
                "code_field": "area_code",
                "levels": ["national", "regional", "territorial"],
                "notes": "Youth employment rate (15-24 years). Regional and sub-regional disaggregation available (133 areas).",
            },
        }

    def _save_table_mapping(self, output_path: Path) -> Path:
        mapping_path = output_path.with_suffix(".mapping.json")
        mapping_path.parent.mkdir(parents=True, exist_ok=True)
        with open(mapping_path, "w", encoding="utf-8") as handle:
            json.dump(self.table_mapping, handle, indent=2, ensure_ascii=False)
        return mapping_path

    def download_youth_employment(self, start_year: int = 2004, end_year: int = 2025) -> pd.DataFrame:
        """Download youth employment rate from ISTAT."""
        print(f"Fetching ISTAT youth employment data from {self.flow_id}...")
        
        # Fetch with no key to get all dimensions
        csv_data = self.client.get_data(
            flow_id=self.flow_id,
            key="",
            start_year=start_year,
            end_year=end_year,
            format="csv",
        )
        
        if not csv_data or csv_data.strip() == "":
            print("Warning: No CSV data returned from ISTAT")
            return pd.DataFrame()

        df = pd.read_csv(io.StringIO(csv_data))
        
        if df.empty or "OBS_VALUE" not in df.columns:
            return df

        # Filter for youth 15-24 age group, total sex, and employment rate data type
        if "AGE" in df.columns and "SEX" in df.columns and "DATA_TYPE" in df.columns:
            df = df[(df.get("AGE", "") == "Y15-24") & 
                   (df.get("SEX", "") == 9) & 
                   (df.get("DATA_TYPE", "") == "EMP_R")]
            print(f"After filtering: {df.shape[0]} rows")

        if df.empty:
            print("Warning: No data after filtering")
            return df

        df_clean = df[["REF_AREA", "TIME_PERIOD", "OBS_VALUE"]].copy()
        df_clean.columns = ["area_code", "year", "youth_employment_rate"]

        df_clean["year"] = pd.to_numeric(df_clean["year"], errors="coerce")
        df_clean["youth_employment_rate"] = pd.to_numeric(df_clean["youth_employment_rate"], errors="coerce")
        df_clean = df_clean.dropna(subset=["year", "youth_employment_rate"])
        df_clean = df_clean[(df_clean["year"] >= start_year) & (df_clean["year"] <= end_year)]
        df_clean = df_clean.sort_values(["area_code", "year"]).drop_duplicates(subset=["area_code", "year"])

        return df_clean

    def save_youth_employment_csv(
        self,
        output_path: Optional[Path] = None,
        start_year: int = 2004,
        end_year: int = 2025,
    ) -> Path:
        if output_path is None:
            indicator_dir = get_indicator_data_dir("youth_employment")
            output_path = indicator_dir / f"youth_employment_raw_{start_year}_{end_year}.csv"

        print(f"Downloading youth employment data ({start_year}-{end_year})...")
        df = self.download_youth_employment(start_year=start_year, end_year=end_year)

        if df.empty:
            print("Warning: No data retrieved, skipping save.")
            return output_path

        print(f"Saving to {output_path}...")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False, encoding="utf-8")
        mapping_path = self._save_table_mapping(output_path)

        print(f"✓ Youth employment data saved: {output_path}")
        print(f"✓ Table mapping saved: {mapping_path}")
        print(f"  - {len(df)} records")
        print(f"  - Years: {int(df['year'].min())} to {int(df['year'].max())}")
        print(f"  - Unique areas: {df['area_code'].nunique()}")

        return output_path
