"""Self-employment indicator downloader (ISTAT source - regional data)."""

import io
import json
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from sera.config import get_indicator_data_dir
from sera.istat_client import IstatClient


class SelfEmploymentDownloader:
    """Download self-employment share for Italy regions from ISTAT."""

    def __init__(self):
        self.client = IstatClient()
        self.flow_id = "150_938_DF_DCCV_OCCUPATIT1_23"  # Employment by professional position

        self.table_mapping: dict[str, Any] = {
            "indicator": "self_employment",
            "source": {
                "provider": "ISTAT",
                "flow_id": self.flow_id,
                "columns": {
                    "REF_AREA": "area_code",
                    "TIME_PERIOD": "year",
                    "POSIZ_PROF": "professional_position",
                    "OBS_VALUE": "occupied_thousands",
                },
            },
            "project": {
                "entity": "geography",
                "code_field": "area_code",
                "levels": ["national", "regional", "territorial"],
                "notes": "Self-employed workers as share of total employed. Computed from POSIZ_PROF dimensions. Regional and sub-regional data available (133 areas).",
            },
        }

    def _save_table_mapping(self, output_path: Path) -> Path:
        mapping_path = output_path.with_suffix(".mapping.json")
        mapping_path.parent.mkdir(parents=True, exist_ok=True)
        with open(mapping_path, "w", encoding="utf-8") as handle:
            json.dump(self.table_mapping, handle, indent=2, ensure_ascii=False)
        return mapping_path

    def download_self_employment(self, start_year: int = 2004, end_year: int = 2025) -> pd.DataFrame:
        """Download self-employment share from ISTAT."""
        print(f"Fetching ISTAT employment by professional position from {self.flow_id}...")
        
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

        # Filter for annual frequency, total sex
        if "FREQ" in df.columns and "SEX" in df.columns:
            df = df[(df.get("FREQ", "") == "A") & (df.get("SEX", "") == 9)]

        if df.empty:
            print("Warning: No data after basic filtering")
            return df

        # Pivot to get self-employed (POSIZ_PROF=2) vs total (POSIZ_PROF=9)
        # POSIZ_PROF codes: 1=employee, 2=self-employed, 9=total
        df_pivot = df.pivot_table(
            index=["REF_AREA", "TIME_PERIOD"],
            columns="POSIZ_PROF",
            values="OBS_VALUE",
            aggfunc="first"
        )

        if df_pivot.empty or 2 not in df_pivot.columns or 9 not in df_pivot.columns:
            print("Warning: Cannot find self-employed and total employment data")
            return pd.DataFrame()

        # Calculate self-employment share: self-employed / total
        df_result = df_pivot[[2, 9]].copy()
        df_result.columns = ["self_employed", "total_employed"]
        df_result["self_employment_share"] = (df_result["self_employed"] / df_result["total_employed"] * 100).round(2)
        
        df_clean = df_result[["self_employment_share"]].reset_index()
        df_clean.columns = ["area_code", "year", "self_employment_share"]

        df_clean["year"] = pd.to_numeric(df_clean["year"], errors="coerce")
        df_clean["self_employment_share"] = pd.to_numeric(df_clean["self_employment_share"], errors="coerce")
        df_clean = df_clean.dropna(subset=["year", "self_employment_share"])
        df_clean = df_clean[(df_clean["year"] >= start_year) & (df_clean["year"] <= end_year)]
        df_clean = df_clean.sort_values(["area_code", "year"]).drop_duplicates(subset=["area_code", "year"])

        return df_clean

    def save_self_employment_csv(
        self,
        output_path: Optional[Path] = None,
        start_year: int = 2004,
        end_year: int = 2025,
    ) -> Path:
        if output_path is None:
            indicator_dir = get_indicator_data_dir("self_employment")
            output_path = indicator_dir / f"self_employment_raw_{start_year}_{end_year}.csv"

        print(f"Downloading self-employment data ({start_year}-{end_year})...")
        df = self.download_self_employment(start_year=start_year, end_year=end_year)

        if df.empty:
            print("Warning: No data retrieved, skipping save.")
            return output_path

        print(f"Saving to {output_path}...")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False, encoding="utf-8")
        mapping_path = self._save_table_mapping(output_path)

        print(f"✓ Self-employment data saved: {output_path}")
        print(f"✓ Table mapping saved: {mapping_path}")
        print(f"  - {len(df)} records")
        print(f"  - Years: {int(df['year'].min())} to {int(df['year'].max())}")
        print(f"  - Unique areas: {df['area_code'].nunique()}")

        return output_path
