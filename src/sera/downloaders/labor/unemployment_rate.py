"""Unemployment rate indicator downloader (ISTAT source - regional data)."""

import io
import json
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from sera.config import get_indicator_data_dir
from sera.istat_client import IstatClient


class UnemploymentRateDownloader:
    """Download unemployment rate for Italy regions from ISTAT."""

    def __init__(self):
        self.client = IstatClient()
        self.flow_id = "151_929_DF_DCCV_DISOCCUPT1_5"  # Regional unemployment data

        self.table_mapping: dict[str, Any] = {
            "indicator": "unemployment_rate",
            "source": {
                "provider": "ISTAT",
                "flow_id": self.flow_id,
                "columns": {
                    "REF_AREA": "area_code",
                    "TIME_PERIOD": "year",
                    "OBS_VALUE": "unemployment_count",
                },
            },
            "project": {
                "entity": "geography",
                "code_field": "area_code",
                "levels": ["national", "regional", "provincial"],
                "notes": "Disoccupati (thousands). Regional disaggregation available.",
            },
        }

    def _save_table_mapping(self, output_path: Path) -> Path:
        mapping_path = output_path.with_suffix(".mapping.json")
        mapping_path.parent.mkdir(parents=True, exist_ok=True)
        with open(mapping_path, "w", encoding="utf-8") as handle:
            json.dump(self.table_mapping, handle, indent=2, ensure_ascii=False)
        return mapping_path

    def download_unemployment_rate(self, start_year: int = 2001, end_year: int = 2025) -> pd.DataFrame:
        """Download unemployment rate from ISTAT with regional granularity."""
        print(f"Fetching ISTAT unemployment data from {self.flow_id}...")
        
        # Fetch with no key to get all dimensions
        csv_data = self.client.get_data(
            flow_id=self.flow_id,
            key="",
            start_year=start_year,
            end_year=end_year,
            format="csv",
        )
        
        print(f"Retrieved {len(csv_data) if csv_data else 0} bytes of data")
        if not csv_data or csv_data.strip() == "":
            print("Warning: No CSV data returned from ISTAT")
            return pd.DataFrame()

        print(f"Parsing CSV (first 300 chars): {csv_data[:300]}")
        df = pd.read_csv(io.StringIO(csv_data))
        print(f"Parsed DataFrame shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        if df.empty or "OBS_VALUE" not in df.columns:
            print("Warning: Empty dataframe or missing OBS_VALUE column")
            return df

        # Filter for total sex and working age only (if these columns exist)
        if "SEX" in df.columns and "AGE" in df.columns:
            df = df[(df["SEX"] == 9) & (df["AGE"] == "Y15-74")]
            print(f"After filtering: {df.shape[0]} rows")

        if df.empty:
            print("Warning: No data after filtering")
            return df

        df_clean = df[["REF_AREA", "TIME_PERIOD", "OBS_VALUE"]].copy()
        df_clean.columns = ["area_code", "year", "unemployment_count"]

        df_clean["year"] = pd.to_numeric(df_clean["year"], errors="coerce")
        df_clean["unemployment_count"] = pd.to_numeric(df_clean["unemployment_count"], errors="coerce")
        df_clean = df_clean.dropna(subset=["year", "unemployment_count"])
        df_clean = df_clean[(df_clean["year"] >= start_year) & (df_clean["year"] <= end_year)]
        df_clean = df_clean.sort_values(["area_code", "year"]).drop_duplicates(subset=["area_code", "year"])

        print(f"Final clean data: {df_clean.shape[0]} rows")
        return df_clean

    def save_unemployment_rate_csv(
        self,
        output_path: Optional[Path] = None,
        start_year: int = 2001,
        end_year: int = 2025,
    ) -> Path:
        if output_path is None:
            indicator_dir = get_indicator_data_dir("unemployment_rate")
            output_path = indicator_dir / f"unemployment_rate_raw_{start_year}_{end_year}.csv"

        print(f"Downloading unemployment rate data ({start_year}-{end_year})...")
        df = self.download_unemployment_rate(start_year=start_year, end_year=end_year)

        if df.empty:
            print("Warning: No data retrieved, skipping save.")
            return output_path

        print(f"Saving to {output_path}...")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False, encoding="utf-8")
        mapping_path = self._save_table_mapping(output_path)

        print(f"✓ Unemployment data saved: {output_path}")
        print(f"✓ Table mapping saved: {mapping_path}")
        print(f"  - {len(df)} records")
        print(f"  - Years: {int(df['year'].min())} to {int(df['year'].max())}")
        print(f"  - Unique areas: {df['area_code'].nunique()}")

        return output_path
