"""Income indicator downloader."""

import io
import json
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from sera.config import CACHE_DIR, get_indicator_data_dir
from sera.istat_client import IstatClient


class IncomeDownloader:
    """Download household disposable income data from ISTAT."""

    def __init__(self, cache_dir: Optional[Path] = None):
        """Initialize income downloader.

        Args:
            cache_dir: Optional directory to cache API responses.
        """
        self.client = IstatClient(cache_dir=cache_dir or CACHE_DIR)
        self.dataflow_id = "93_1095_DF_DCCN_ISTITUZ_TNA1_1"
        # Annual data, all areas, net disposable income (B6N_B_W0), households sector (S14)
        # with current prices and latest edition series (2025M12).
        self.key = "A..B6N_B_W0.S14.Z.Z.V.S.N.2025M12"

        self.table_mapping: dict[str, Any] = {
            "indicator": "income",
            "source": {
                "dataflow_id": self.dataflow_id,
                "key": self.key,
                "columns": {
                    "REF_AREA": "area_code",
                    "FREQ": "frequency",
                    "TIME_PERIOD": "year",
                    "OBS_VALUE": "income",
                    "DATA_TYPE_AGGR": "measure_code",
                    "INSTITUTIONAL_SECTOR": "institutional_sector",
                    "EDITION": "edition",
                },
            },
            "project": {
                "entity": "geography",
                "code_field": "area_code",
                "levels": ["region", "macroarea", "national"],
                "notes": "Household net disposable income from ISTAT national accounts (current prices).",
            },
        }

    def _save_table_mapping(self, output_path: Path) -> Path:
        """Save the source-to-project mapping for this table next to the CSV output."""
        mapping_path = output_path.with_suffix(".mapping.json")
        mapping_path.parent.mkdir(parents=True, exist_ok=True)

        with open(mapping_path, "w", encoding="utf-8") as handle:
            json.dump(self.table_mapping, handle, indent=2, ensure_ascii=False)

        return mapping_path

    def download_income(
        self,
        start_year: int = 2001,
        end_year: int = 2025,
    ) -> pd.DataFrame:
        """Download income data.

        Args:
            start_year: First year to download.
            end_year: Last year to download.

        Returns:
            DataFrame with cleaned income values.
        """
        csv_data = self.client.get_data(
            flow_id=self.dataflow_id,
            key=self.key,
            start_year=start_year,
            end_year=end_year,
            format="csv",
        )

        df = pd.read_csv(io.StringIO(csv_data))

        df_clean = df[
            [
                "REF_AREA",
                "FREQ",
                "TIME_PERIOD",
                "OBS_VALUE",
                "DATA_TYPE_AGGR",
                "INSTITUTIONAL_SECTOR",
                "EDITION",
            ]
        ].copy()
        df_clean.columns = [
            "area_code",
            "frequency",
            "year",
            "income",
            "measure_code",
            "institutional_sector",
            "edition",
        ]

        df_clean["year"] = pd.to_numeric(df_clean["year"], errors="coerce")
        df_clean["income"] = pd.to_numeric(df_clean["income"], errors="coerce")
        df_clean = df_clean.dropna(subset=["year", "income"])

        return df_clean

    def save_income_csv(
        self,
        output_path: Optional[Path] = None,
        start_year: int = 2001,
        end_year: int = 2025,
    ) -> Path:
        """Download and save income data to CSV."""
        if output_path is None:
            indicator_dir = get_indicator_data_dir("income")
            output_path = indicator_dir / f"income_raw_{start_year}_{end_year}.csv"

        print(f"Downloading income data ({start_year}-{end_year})...")
        df = self.download_income(start_year=start_year, end_year=end_year)

        print(f"Saving to {output_path}...")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False, encoding="utf-8")
        mapping_path = self._save_table_mapping(output_path)

        print(f"✓ Income data saved: {output_path}")
        print(f"✓ Table mapping saved: {mapping_path}")
        print(f"  - {len(df)} records")
        print(f"  - Years: {int(df['year'].min())} to {int(df['year'].max())}")
        print(f"  - Unique areas: {df['area_code'].nunique()}")

        return output_path
