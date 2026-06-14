"""Age distribution downloader."""

import io
import json
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from sera.config import CACHE_DIR, get_indicator_data_dir
from sera.istat_client import IstatClient


class AgeDistributionDownloader:
    """Download population by age groups from ISTAT."""

    def __init__(self, cache_dir: Optional[Path] = None):
        self.client = IstatClient(cache_dir=cache_dir or CACHE_DIR)
        self.dataflow_id = "22_289_DF_DCIS_POPRES1_1"
        self.data_type = "JAN"
        self.table_mapping: dict[str, Any] = {
            "indicator": "age_distribution",
            "source": {
                "dataflow_id": self.dataflow_id,
                "key": f"A..{self.data_type}.1..1",
                "columns": {
                    "REF_AREA": "area_code",
                    "FREQ": "frequency",
                    "TIME_PERIOD": "year",
                    "OBS_VALUE": "population",
                    "SEX": "sex",
                    "AGE": "age_group",
                },
            },
            "project": {
                "entity": "geography",
                "code_field": "area_code",
                "levels": ["province", "region", "national"],
            },
        }

    def _save_table_mapping(self, output_path: Path) -> Path:
        mapping_path = output_path.with_suffix(".mapping.json")
        mapping_path.parent.mkdir(parents=True, exist_ok=True)

        with open(mapping_path, "w", encoding="utf-8") as handle:
            json.dump(self.table_mapping, handle, indent=2, ensure_ascii=False)

        return mapping_path

    def download_age_distribution(
        self, start_year: int = 2001, end_year: int = 2025, geographic_level: Optional[str] = None
    ) -> pd.DataFrame:
        """Download population by age groups.

        Returns DataFrame with columns: [REF_AREA, FREQ, TIME_PERIOD, OBS_VALUE, SEX, AGE]
        """
        # Key format: FREQ.REF_AREA.DATA_TYPE.SEX.AGE.MARITAL_STATUS
        # Leave REF_AREA empty to fetch all areas, and leave AGE empty to fetch all age groups
        key = f"A..{self.data_type}.1..1"

        csv_data = self.client.get_data(
            flow_id=self.dataflow_id,
            key=key,
            start_year=start_year,
            end_year=end_year,
            format="csv",
        )

        df = pd.read_csv(io.StringIO(csv_data))

        # Keep and rename relevant columns
        cols = ["REF_AREA", "FREQ", "TIME_PERIOD", "OBS_VALUE", "SEX", "AGE"]
        df_clean = df[cols].copy()
        df_clean.columns = ["area_code", "frequency", "year", "population", "sex", "age_group"]

        df_clean["year"] = pd.to_numeric(df_clean["year"], errors="coerce")
        df_clean["population"] = pd.to_numeric(df_clean["population"], errors="coerce")
        df_clean = df_clean.dropna(subset=["year", "population"])

        return df_clean

    def save_age_distribution_csv(
        self, output_path: Optional[Path] = None, start_year: int = 2001, end_year: int = 2025
    ) -> Path:
        if output_path is None:
            indicator_dir = get_indicator_data_dir("age_distribution")
            output_path = indicator_dir / f"age_distribution_raw_{start_year}_{end_year}.csv"

        print(f"Downloading age distribution data ({start_year}-{end_year})...")
        df = self.download_age_distribution(start_year=start_year, end_year=end_year)

        print(f"Saving to {output_path}...")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False, encoding="utf-8")
        mapping_path = self._save_table_mapping(output_path)

        print(f"✓ Age distribution data saved: {output_path}")
        print(f"✓ Table mapping saved: {mapping_path}")
        print(f"  - {len(df)} records")
        print(f"  - Years: {int(df['year'].min())} to {int(df['year'].max())}")
        print(f"  - Unique areas: {df['area_code'].nunique()}")

        return output_path
