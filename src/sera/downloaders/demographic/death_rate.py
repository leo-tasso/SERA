"""Death rate downloader."""

import io
import json
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from sera.config import CACHE_DIR, get_indicator_data_dir
from sera.istat_client import IstatClient


class DeathRateDownloader:
    """Download death rate data from ISTAT."""

    def __init__(self, cache_dir: Optional[Path] = None):
        """Initialize death rate downloader.

        Args:
            cache_dir: Optional directory to cache API responses.
        """
        self.client = IstatClient(cache_dir=cache_dir or CACHE_DIR)
        self.dataflow_id = "26_29_DF_DCIS_DECESSI_1"
        self.table_mapping: dict[str, Any] = {
            "indicator": "death_rate",
            "source": {
                "dataflow_id": self.dataflow_id,
                "key": "A.IT.............",
                "columns": {
                    "REF_AREA": "area_code",
                    "FREQ": "frequency",
                    "TIME_PERIOD": "year",
                    "OBS_VALUE": "death_rate",
                    "DATA_TYPE": "data_type",
                },
            },
            "project": {
                "entity": "geography",
                "code_field": "area_code",
                "levels": ["national"],
                "notes": "Death rate per 1000 population. Data available from 2011 onwards.",
            },
        }

    def _save_table_mapping(self, output_path: Path) -> Path:
        """Save the source-to-project mapping for this table next to the CSV output."""
        mapping_path = output_path.with_suffix(".mapping.json")
        mapping_path.parent.mkdir(parents=True, exist_ok=True)

        with open(mapping_path, "w", encoding="utf-8") as handle:
            json.dump(self.table_mapping, handle, indent=2, ensure_ascii=False)

        return mapping_path

    def download_death_rate(
        self,
        start_year: int = 2011,
        end_year: int = 2025,
    ) -> pd.DataFrame:
        """Download death rate data.

        Args:
            start_year: First year to download (default 2011, earliest available)
            end_year: Last year to download (default 2025)

        Returns:
            DataFrame with columns: [REF_AREA, FREQ, TIME_PERIOD, OBS_VALUE, DATA_TYPE]
        """
        # Fetch data from ISTAT
        # Key: A.IT............. (annual, national, all other dimensions)
        # DATA_TYPE includes: DEATHRATE (per 1000 population)
        key = "A.IT............."

        csv_data = self.client.get_data(
            flow_id=self.dataflow_id,
            key=key,
            start_year=start_year,
            end_year=end_year,
            format="csv",
        )

        # Parse CSV into DataFrame
        df = pd.read_csv(io.StringIO(csv_data))

        # Keep only essential columns and rename for clarity
        df_clean = df[["REF_AREA", "FREQ", "TIME_PERIOD", "OBS_VALUE", "DATA_TYPE"]].copy()
        df_clean.columns = ["area_code", "frequency", "year", "death_rate", "data_type"]

        # Convert year and death_rate to numeric
        df_clean["year"] = pd.to_numeric(df_clean["year"], errors="coerce")
        df_clean["death_rate"] = pd.to_numeric(df_clean["death_rate"], errors="coerce")

        # Remove rows with parsing errors
        df_clean = df_clean.dropna(subset=["year", "death_rate"])

        return df_clean

    def save_death_rate_csv(
        self,
        output_path: Optional[Path] = None,
        start_year: int = 2011,
        end_year: int = 2025,
    ) -> Path:
        """Download and save death rate data to CSV.

        Args:
            output_path: Path to save CSV. If None, saves to data/death_rate/death_rate_raw_{start}_{end}.csv
            start_year: First year to download
            end_year: Last year to download

        Returns:
            Path to saved CSV file.
        """
        if output_path is None:
            indicator_dir = get_indicator_data_dir("death_rate")
            output_path = indicator_dir / f"death_rate_raw_{start_year}_{end_year}.csv"

        print(f"Downloading death rate data ({start_year}-{end_year})...")
        df = self.download_death_rate(start_year=start_year, end_year=end_year)

        print(f"Saving to {output_path}...")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False, encoding="utf-8")
        mapping_path = self._save_table_mapping(output_path)

        print(f"✓ Death rate data saved: {output_path}")
        print(f"✓ Table mapping saved: {mapping_path}")
        print(f"  - {len(df)} records")
        print(f"  - Years: {int(df['year'].min())} to {int(df['year'].max())}")
        print(f"  - Unique areas: {df['area_code'].nunique()}")

        return output_path
