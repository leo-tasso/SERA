"""Skills match indicator downloader (ISTAT NEET source - regional data)."""

import io
import json
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from sera.config import get_indicator_data_dir
from sera.istat_client import IstatClient


class SkillsMatchDownloader:
    """Download skills match proxy (NEET rate) for Italy regions from ISTAT."""

    def __init__(self):
        self.client = IstatClient()
        self.flow_id = (
            "172_931_DF_DCCV_NEET1_11"  # NEET rate (youth not in education or employment)
        )

        self.table_mapping: dict[str, Any] = {
            "indicator": "skills_match",
            "source": {
                "provider": "ISTAT",
                "flow_id": self.flow_id,
                "proxy": "NEET rate",
                "columns": {
                    "REF_AREA": "area_code",
                    "TIME_PERIOD": "year",
                    "OBS_VALUE": "neet_rate_percent",
                },
            },
            "project": {
                "entity": "geography",
                "code_field": "area_code",
                "levels": ["national", "regional", "macro-regional"],
                "notes": "Skills match proxy: NEET rate (15-29 years not in education, employment or training). Higher NEET indicates lower skills match. Regional data for 28 areas.",
            },
        }

    def _save_table_mapping(self, output_path: Path) -> Path:
        mapping_path = output_path.with_suffix(".mapping.json")
        mapping_path.parent.mkdir(parents=True, exist_ok=True)
        with open(mapping_path, "w", encoding="utf-8") as handle:
            json.dump(self.table_mapping, handle, indent=2, ensure_ascii=False)
        return mapping_path

    def download_skills_match(self, start_year: int = 2018, end_year: int = 2025) -> pd.DataFrame:
        """Download NEET rate from ISTAT as skills match proxy."""
        print(f"Fetching ISTAT NEET data from {self.flow_id}...")

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

        # Filter for NEET age group 15-29, total sex, and NEET data type
        if "AGE" in df.columns and "SEX" in df.columns and "DATA_TYPE" in df.columns:
            df = df[
                (df.get("AGE", "") == "Y15-29")
                & (df.get("SEX", "") == 9)
                & (df.get("DATA_TYPE", "") == "NEET_I")
            ]
            print(f"After filtering: {df.shape[0]} rows")

        if df.empty:
            print("Warning: No data after filtering")
            return df

        df_clean = df[["REF_AREA", "TIME_PERIOD", "OBS_VALUE"]].copy()
        df_clean.columns = ["area_code", "year", "neet_rate_percent"]

        df_clean["year"] = pd.to_numeric(df_clean["year"], errors="coerce")
        df_clean["neet_rate_percent"] = pd.to_numeric(
            df_clean["neet_rate_percent"], errors="coerce"
        )
        df_clean = df_clean.dropna(subset=["year", "neet_rate_percent"])
        df_clean = df_clean[(df_clean["year"] >= start_year) & (df_clean["year"] <= end_year)]
        df_clean = df_clean.sort_values(["area_code", "year"]).drop_duplicates(
            subset=["area_code", "year"]
        )

        return df_clean

    def save_skills_match_csv(
        self,
        output_path: Optional[Path] = None,
        start_year: int = 2018,
        end_year: int = 2025,
    ) -> Path:
        if output_path is None:
            indicator_dir = get_indicator_data_dir("skills_match")
            output_path = indicator_dir / f"skills_match_raw_{start_year}_{end_year}.csv"

        print(f"Downloading skills match (NEET) data ({start_year}-{end_year})...")
        df = self.download_skills_match(start_year=start_year, end_year=end_year)

        if df.empty:
            print("Warning: No data retrieved, skipping save.")
            return output_path

        print(f"Saving to {output_path}...")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False, encoding="utf-8")
        mapping_path = self._save_table_mapping(output_path)

        print(f"✓ Skills match data saved: {output_path}")
        print(f"✓ Table mapping saved: {mapping_path}")
        print(f"  - {len(df)} records")
        print(f"  - Years: {int(df['year'].min())} to {int(df['year'].max())}")
        print(f"  - Unique areas: {df['area_code'].nunique()}")

        return output_path
