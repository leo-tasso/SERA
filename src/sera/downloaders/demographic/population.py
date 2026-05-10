"""Population indicator downloader."""

import json
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import io

from sera.istat_client import IstatClient
from sera.config import DATA_DIR, CACHE_DIR, get_indicator_data_dir


class PopulationDownloader:
    """Download resident population data from ISTAT for all provinces."""

    def __init__(self, cache_dir: Optional[Path] = None):
        """Initialize population downloader.

        Args:
            cache_dir: Optional directory to cache API responses.
        """
        self.client = IstatClient(cache_dir=cache_dir or CACHE_DIR)
        self.dataflow_id = "22_289_DF_DCIS_POPRES1_1"
        self.data_type = "JAN"  # January snapshot for annual population
        self.table_mapping: dict[str, Any] = {
            "indicator": "population",
            "source": {
                "dataflow_id": self.dataflow_id,
                "key": f"A..{self.data_type}.1.TOTAL.1",
                "columns": {
                    "REF_AREA": "area_code",
                    "FREQ": "frequency",
                    "TIME_PERIOD": "year",
                    "OBS_VALUE": "population",
                    "SEX": "sex",
                    "AGE": "age_group",
                    "MARITAL_STATUS": "marital_status",
                },
            },
            "project": {
                "entity": "geography",
                "code_field": "area_code",
                "levels": ["province", "region", "national"],
                "notes": "ISTAT area codes are kept as the project geography identifiers so each downloaded row stays aligned to the SERA spatial entities.",
            },
        }

    def _save_table_mapping(self, output_path: Path) -> Path:
        """Save the source-to-project mapping for this table next to the CSV output."""
        mapping_path = output_path.with_suffix(".mapping.json")
        mapping_path.parent.mkdir(parents=True, exist_ok=True)

        with open(mapping_path, "w", encoding="utf-8") as handle:
            json.dump(self.table_mapping, handle, indent=2, ensure_ascii=False)

        return mapping_path

    def download_population(
        self,
        start_year: int = 2001,
        end_year: int = 2025,
        geographic_level: Optional[str] = None,
    ) -> pd.DataFrame:
        """Download resident population data.

        Args:
            start_year: First year to download (default 2001)
            end_year: Last year to download (default 2025)
            geographic_level: Filter by geographic level ("national", "region", "province")
                             If None, returns all levels.

        Returns:
            DataFrame with columns: [REF_AREA, FREQ, TIME_PERIOD, OBS_VALUE, ...]
        """
        # Fetch data from ISTAT
        # Key format: FREQ.REF_AREA.DATA_TYPE.SEX.AGE.MARITAL_STATUS
        # Using A (annual), all areas, JAN (January), 1 (total sex), TOTAL (all ages), 1 (total marital)
        key = f"A..{self.data_type}.1.TOTAL.1"

        csv_data = self.client.get_data(
            flow_id=self.dataflow_id,
            key=key,
            start_year=start_year,
            end_year=end_year,
            format="csv",
        )

        # Parse CSV into DataFrame
        df = pd.read_csv(io.StringIO(csv_data))

        # Clean up: keep only essential columns and rename for clarity
        df_clean = df[
            ["REF_AREA", "FREQ", "TIME_PERIOD", "OBS_VALUE", "SEX", "AGE", "MARITAL_STATUS"]
        ].copy()
        df_clean.columns = [
            "area_code",
            "frequency",
            "year",
            "population",
            "sex",
            "age_group",
            "marital_status",
        ]

        # Convert year and population to numeric
        df_clean["year"] = pd.to_numeric(df_clean["year"], errors="coerce")
        df_clean["population"] = pd.to_numeric(df_clean["population"], errors="coerce")

        # Remove rows with parsing errors
        df_clean = df_clean.dropna(subset=["year", "population"])

        return df_clean

    def save_population_csv(
        self,
        output_path: Optional[Path] = None,
        start_year: int = 2001,
        end_year: int = 2025,
    ) -> Path:
        """Download and save population data to CSV.

        Args:
            output_path: Path to save CSV. If None, saves to data/population/population_raw_{start}_{end}.csv
            start_year: First year to download
            end_year: Last year to download

        Returns:
            Path to saved CSV file.
        """
        if output_path is None:
            indicator_dir = get_indicator_data_dir("population")
            output_path = indicator_dir / f"population_raw_{start_year}_{end_year}.csv"

        print(f"Downloading population data ({start_year}-{end_year})...")
        df = self.download_population(start_year=start_year, end_year=end_year)

        print(f"Saving to {output_path}...")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False, encoding="utf-8")
        mapping_path = self._save_table_mapping(output_path)

        print(f"✓ Population data saved: {output_path}")
        print(f"✓ Table mapping saved: {mapping_path}")
        print(f"  - {len(df)} records")
        print(f"  - Years: {df['year'].min():.0f} to {df['year'].max():.0f}")
        print(f"  - Unique areas: {df['area_code'].nunique()}")

        return output_path
