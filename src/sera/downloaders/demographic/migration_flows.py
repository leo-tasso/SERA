"""Migration flows indicator downloader."""

import io
import json
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from sera.config import CACHE_DIR, get_indicator_data_dir
from sera.istat_client import IstatClient


class MigrationFlowsDownloader:
    """Download migration flows data from ISTAT at national level."""

    def __init__(self, cache_dir: Optional[Path] = None):
        """Initialize migration flows downloader.

        Args:
            cache_dir: Optional directory to cache API responses.
        """
        self.client = IstatClient(cache_dir=cache_dir or CACHE_DIR)
        self.dataflow_id = "28_185_DF_DCIS_MIGRAZIONI_1"
        self.table_mapping: dict[str, Any] = {
            "indicator": "migration_flows",
            "source": {
                "dataflow_id": self.dataflow_id,
                "key": "A.IT.CORE.ALL............",
                "columns": {
                    "FREQ": "frequency",
                    "REF_AREA": "area_code",
                    "TIME_PERIOD": "year",
                    "OBS_VALUE": "migrations",
                },
            },
            "project": {
                "entity": "geography",
                "code_field": "area_code",
                "levels": ["national"],
                "notes": "Internal migration flows (Migrazioni interne) - ISTAT dataflow 28_185_DF_DCIS_MIGRAZIONI_1. National-level aggregated migration data. Currently limited to national level (IT). Data represents internal migrations (changes of residence) for Italian and foreign residents.",
            },
        }

    def _save_table_mapping(self, output_path: Path) -> Path:
        """Save the source-to-project mapping for this table next to the CSV output."""
        mapping_path = output_path.with_suffix(".mapping.json")
        mapping_path.parent.mkdir(parents=True, exist_ok=True)

        with open(mapping_path, "w", encoding="utf-8") as handle:
            json.dump(self.table_mapping, handle, indent=2, ensure_ascii=False)

        return mapping_path

    def download_migration_flows(
        self,
        start_year: int = 2001,
        end_year: int = 2025,
    ) -> pd.DataFrame:
        """Download migration flows data.

        Note: ISTAT migration data (28_185_DF_DCIS_MIGRAZIONI_1 dataflow) is available
        at national level (IT) and regional level. This implementation retrieves national
        aggregate data.

        Args:
            start_year: First year to download (default 2001)
            end_year: Last year to download (default 2025)

        Returns:
            DataFrame with columns: [area_code, year, migrations, ...]
        """
        # Fetch national-level migration flows from ISTAT
        # Key: A.IT.CORE.ALL............
        # This retrieves total migrations aggregated at national level:
        # - FREQ (A): Annual
        # - REF_AREA (IT): National level
        # - DATA_TYPE (CORE): Core data type
        # - CHANGE_OF_RESIDENCE (ALL): All types
        # - CITIZENSHIP (ALL): All citizens
        # - SEX (1): Male (or combined)
        # - AGE (TOTAL): All ages
        # - TERRITORY_NEXT_RESID (IT): National territory
        # - COUNTRY_PREV_RESID (X1033): Previous residence code
        # - COUNTRY_NEXT_RESID (X1033): Next residence code
        key = "A.IT.CORE.ALL............"

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
        df_clean = df[["REF_AREA", "FREQ", "TIME_PERIOD", "OBS_VALUE"]].copy()

        rename_map = {
            "REF_AREA": "area_code",
            "FREQ": "frequency",
            "TIME_PERIOD": "year",
            "OBS_VALUE": "migrations",
        }
        df_clean.rename(columns=rename_map, inplace=True)

        # Convert to numeric
        df_clean["year"] = pd.to_numeric(df_clean["year"], errors="coerce")
        df_clean["migrations"] = pd.to_numeric(df_clean["migrations"], errors="coerce")

        # Remove rows with parsing errors
        df_clean = df_clean.dropna(subset=["year", "migrations"])

        return df_clean

    def save_migration_flows_csv(
        self,
        output_path: Optional[Path] = None,
        start_year: int = 2001,
        end_year: int = 2025,
    ) -> Path:
        """Download and save migration flows data to CSV.

        Args:
            output_path: Path to save CSV. If None, saves to data/migration_flows/migration_flows_raw_{start}_{end}.csv
            start_year: First year to download
            end_year: Last year to download

        Returns:
            Path to saved CSV file.
        """
        if output_path is None:
            indicator_dir = get_indicator_data_dir("migration_flows")
            output_path = indicator_dir / f"migration_flows_raw_{start_year}_{end_year}.csv"

        print(f"Downloading migration flows data ({start_year}-{end_year})...")
        df = self.download_migration_flows(start_year=start_year, end_year=end_year)

        print(f"Saving to {output_path}...")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False, encoding="utf-8")
        mapping_path = self._save_table_mapping(output_path)

        print(f"✓ Migration flows data saved: {output_path}")
        print(f"✓ Table mapping saved: {mapping_path}")
        print(f"  - {len(df)} records")
        print(f"  - Years: {df['year'].min():.0f} to {df['year'].max():.0f}")
        print(f"  - Unique areas: {df['area_code'].nunique()}")

        return output_path
