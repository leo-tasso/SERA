"""Birth rate indicator downloader."""

import json
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import io

from sera.istat_client import IstatClient
from sera.config import DATA_DIR, CACHE_DIR, get_indicator_data_dir


class BirthRateDownloader:
    """Download live births data from ISTAT at national level."""

    def __init__(self, cache_dir: Optional[Path] = None):
        """Initialize birth rate downloader.

        Args:
            cache_dir: Optional directory to cache API responses.
        """
        self.client = IstatClient(cache_dir=cache_dir or CACHE_DIR)
        self.dataflow_id = "25_74"
        self.table_mapping: dict[str, Any] = {
            "indicator": "birth_rate",
            "source": {
                "dataflow_id": self.dataflow_id,
                "key": "A.IT.LBIRTH.TOTAL.TOTAL.TOTAL.ALL.99.99.ALL.ALL.WORLD.ALL",
                "columns": {
                    "FREQ": "frequency",
                    "RESIDENCE_TERR": "area_code",
                    "TIME_PERIOD": "year",
                    "OBS_VALUE": "births",
                },
            },
            "project": {
                "entity": "geography",
                "code_field": "area_code",
                "levels": ["national"],
                "notes": "Live births (Nati vivi) - ISTAT dataflow 25_74. Currently available only at national level (IT). Data aggregated across all demographic dimensions (citizenship, age, marital status). Future updates may include province/region-level breakdown.",
            },
        }

    def _save_table_mapping(self, output_path: Path) -> Path:
        """Save the source-to-project mapping for this table next to the CSV output."""
        mapping_path = output_path.with_suffix(".mapping.json")
        mapping_path.parent.mkdir(parents=True, exist_ok=True)

        with open(mapping_path, "w", encoding="utf-8") as handle:
            json.dump(self.table_mapping, handle, indent=2, ensure_ascii=False)

        return mapping_path

    def download_birth_rate(
        self,
        start_year: int = 2001,
        end_year: int = 2025,
    ) -> pd.DataFrame:
        """Download live births data.

        Note: ISTAT birth rate data (25_74 dataflow) is available only at national level (IT).
        Province and region-level data is not yet published in this dataflow.

        Args:
            start_year: First year to download (default 2001)
            end_year: Last year to download (default 2025)

        Returns:
            DataFrame with columns: [area_code, year, births, ...]
        """
        # Fetch national-level birth data from ISTAT
        # Key: A.IT.LBIRTH.TOTAL.TOTAL.TOTAL.ALL.99.99.ALL.ALL.WORLD.ALL
        # This retrieves total births aggregated at national level
        key = "A.IT.LBIRTH.TOTAL.TOTAL.TOTAL.ALL.99.99.ALL.ALL.WORLD.ALL"

        csv_data = self.client.get_data(
            flow_id=self.dataflow_id,
            key=key,
            start_year=start_year,
            end_year=end_year,
            format="csv",
        )

        # Parse CSV into DataFrame
        df = pd.read_csv(io.StringIO(csv_data))

        # Keep only essential columns and rename
        df_clean = df[[
            "RESIDENCE_TERR", "FREQ", "TIME_PERIOD", "OBS_VALUE"
        ]].copy()

        rename_map = {
            "RESIDENCE_TERR": "area_code",
            "FREQ": "frequency",
            "TIME_PERIOD": "year",
            "OBS_VALUE": "births",
        }
        df_clean.rename(columns=rename_map, inplace=True)

        # Convert to numeric
        df_clean["year"] = pd.to_numeric(df_clean["year"], errors="coerce")
        df_clean["births"] = pd.to_numeric(df_clean["births"], errors="coerce")

        # Remove rows with parsing errors
        df_clean = df_clean.dropna(subset=["year", "births"])

        return df_clean

    def save_birth_rate_csv(
        self,
        output_path: Optional[Path] = None,
        start_year: int = 2001,
        end_year: int = 2025,
    ) -> Path:
        """Download and save birth rate data to CSV.

        Args:
            output_path: Path to save CSV. If None, saves to data/birth_rate/birth_rate_raw_{start}_{end}.csv
            start_year: First year to download
            end_year: Last year to download

        Returns:
            Path to saved CSV file.
        """
        if output_path is None:
            indicator_dir = get_indicator_data_dir("birth_rate")
            output_path = indicator_dir / f"birth_rate_raw_{start_year}_{end_year}.csv"

        print(f"Downloading birth rate data ({start_year}-{end_year})...")
        df = self.download_birth_rate(start_year=start_year, end_year=end_year)

        print(f"Saving to {output_path}...")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False, encoding="utf-8")
        mapping_path = self._save_table_mapping(output_path)

        print(f"✓ Birth rate data saved: {output_path}")
        print(f"✓ Table mapping saved: {mapping_path}")
        print(f"  - {len(df)} records")
        print(f"  - Years: {df['year'].min():.0f} to {df['year'].max():.0f}")
        print(f"  - Unique areas: {df['area_code'].nunique()}")

        return output_path
