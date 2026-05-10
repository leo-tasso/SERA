"""Average wages indicator downloader (ISTAT source - macro-regional data)."""

import io
import json
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from sera.config import get_indicator_data_dir
from sera.istat_client import IstatClient


class AverageWagesDownloader:
    """Download average wages for Italy macro-regions from ISTAT."""

    def __init__(self):
        self.client = IstatClient()
        self.flow_id = "533_957_DF_DCSC_RACLI_2"  # Hourly wages by birthplace

        self.table_mapping: dict[str, Any] = {
            "indicator": "average_wages",
            "source": {
                "provider": "ISTAT",
                "flow_id": self.flow_id,
                "columns": {
                    "REF_AREA": "area_code",
                    "TIME_PERIOD": "year",
                    "OBS_VALUE": "hourly_wage_eur",
                },
            },
            "project": {
                "entity": "geography",
                "code_field": "area_code",
                "levels": ["national", "macro-regional"],
                "notes": "Hourly wage (euros). Macro-regional breakdown (6 areas: IT national + ITC, ITD, ITE, ITF, ITG macro-regions). Years 2014-2023.",
            },
        }

    def _save_table_mapping(self, output_path: Path) -> Path:
        mapping_path = output_path.with_suffix(".mapping.json")
        mapping_path.parent.mkdir(parents=True, exist_ok=True)
        with open(mapping_path, "w", encoding="utf-8") as handle:
            json.dump(self.table_mapping, handle, indent=2, ensure_ascii=False)
        return mapping_path

    def download_average_wages(self, start_year: int = 2014, end_year: int = 2023) -> pd.DataFrame:
        """Download average wages from ISTAT."""
        print(f"Fetching ISTAT average wages data from {self.flow_id}...")
        
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

        # Filter for median hourly wage, total sex, total education, total citizenship
        if "DATA_TYPE" in df.columns and "SEX" in df.columns:
            df = df[(df.get("DATA_TYPE", "") == "HOUWAG_ENTEMP_MED_MI") & 
                   (df.get("SEX", "") == 9)]
            print(f"After filtering: {df.shape[0]} rows")

        if df.empty:
            print("Warning: No data after filtering")
            return df

        df_clean = df[["REF_AREA", "TIME_PERIOD", "OBS_VALUE"]].copy()
        df_clean.columns = ["area_code", "year", "hourly_wage_eur"]

        df_clean["year"] = pd.to_numeric(df_clean["year"], errors="coerce")
        df_clean["hourly_wage_eur"] = pd.to_numeric(df_clean["hourly_wage_eur"], errors="coerce")
        df_clean = df_clean.dropna(subset=["year", "hourly_wage_eur"])
        df_clean = df_clean[(df_clean["year"] >= start_year) & (df_clean["year"] <= end_year)]
        df_clean = df_clean.sort_values(["area_code", "year"]).drop_duplicates(subset=["area_code", "year"])

        return df_clean

    def save_average_wages_csv(
        self,
        output_path: Optional[Path] = None,
        start_year: int = 2014,
        end_year: int = 2023,
    ) -> Path:
        if output_path is None:
            indicator_dir = get_indicator_data_dir("average_wages")
            output_path = indicator_dir / f"average_wages_raw_{start_year}_{end_year}.csv"

        print(f"Downloading average wages data ({start_year}-{end_year})...")
        df = self.download_average_wages(start_year=start_year, end_year=end_year)

        if df.empty:
            print("Warning: No data retrieved, skipping save.")
            return output_path

        print(f"Saving to {output_path}...")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False, encoding="utf-8")
        mapping_path = self._save_table_mapping(output_path)

        print(f"✓ Average wages data saved: {output_path}")
        print(f"✓ Table mapping saved: {mapping_path}")
        print(f"  - {len(df)} records")
        print(f"  - Years: {int(df['year'].min())} to {int(df['year'].max())}")
        print(f"  - Unique areas: {df['area_code'].nunique()}")

        return output_path
