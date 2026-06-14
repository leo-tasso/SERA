"""Adult learning indicator downloader (ISTAT source)."""

import io
import json
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from sera.config import get_indicator_data_dir
from sera.istat_client import IstatClient


class AdultLearningDownloader:
    """Download adult learning participation from ISTAT."""

    def __init__(self):
        self.client = IstatClient()
        self.flow_id = "DF_DCSS_LCAS_FRISC_3"

        self.table_mapping: dict[str, Any] = {
            "indicator": "adult_learning",
            "source": {
                "provider": "ISTAT",
                "flow_id": self.flow_id,
                "columns": {
                    "REF_AREA": "area_code",
                    "VOCAT_TRAIN_ATT": "vocational_training_attendance",
                    "TIME_PERIOD": "year",
                    "OBS_VALUE": "value",
                },
            },
            "project": {
                "entity": "geography",
                "code_field": "area_code",
                "levels": ["province", "municipality"],
                "notes": "Adult learning proxy from vocational/professional training attendance in permanent census outputs.",
            },
        }

    def _save_table_mapping(self, output_path: Path) -> Path:
        mapping_path = output_path.with_suffix(".mapping.json")
        mapping_path.parent.mkdir(parents=True, exist_ok=True)
        with open(mapping_path, "w", encoding="utf-8") as handle:
            json.dump(self.table_mapping, handle, indent=2, ensure_ascii=False)
        return mapping_path

    def download_adult_learning(self, start_year: int = 2001, end_year: int = 2025) -> pd.DataFrame:
        """Download adult learning data with ISTAT territorial granularity."""
        print(f"Fetching ISTAT adult learning data from {self.flow_id}...")

        csv_data = self.client.get_data(
            flow_id=self.flow_id,
            key="",
            start_year=start_year,
            end_year=end_year,
            format="csv",
        )
        df = pd.read_csv(io.StringIO(csv_data), low_memory=False)

        if df.empty:
            return pd.DataFrame(
                columns=["area_code", "year", "vocational_training_attendance", "value"]
            )

        df_clean = df[["REF_AREA", "VOCAT_TRAIN_ATT", "TIME_PERIOD", "OBS_VALUE"]].copy()
        df_clean.columns = ["area_code", "vocational_training_attendance", "year", "value"]

        df_clean["year"] = pd.to_numeric(df_clean["year"], errors="coerce")
        df_clean["value"] = pd.to_numeric(df_clean["value"], errors="coerce")
        df_clean = df_clean.dropna(subset=["year", "value"])
        df_clean = df_clean[(df_clean["year"] >= start_year) & (df_clean["year"] <= end_year)]

        if "vocational_training_attendance" in df_clean.columns:
            df_clean["vocational_training_attendance"] = df_clean[
                "vocational_training_attendance"
            ].astype(str)

        df_clean = df_clean.sort_values(
            ["area_code", "year", "vocational_training_attendance"]
        ).drop_duplicates(subset=["area_code", "year", "vocational_training_attendance"])

        return df_clean

    def save_adult_learning_csv(
        self,
        output_path: Optional[Path] = None,
        start_year: int = 2001,
        end_year: int = 2025,
    ) -> Path:
        if output_path is None:
            indicator_dir = get_indicator_data_dir("adult_learning")
            output_path = indicator_dir / f"adult_learning_raw_{start_year}_{end_year}.csv"

        print(f"Downloading adult learning data ({start_year}-{end_year})...")
        df = self.download_adult_learning(start_year=start_year, end_year=end_year)

        if df.empty:
            print("Warning: No adult learning data retrieved, skipping save.")
            return output_path

        print(f"Saving to {output_path}...")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False, encoding="utf-8")
        mapping_path = self._save_table_mapping(output_path)

        print(f"✓ Adult learning data saved: {output_path}")
        print(f"✓ Table mapping saved: {mapping_path}")
        print(f"  - {len(df)} records")
        print(f"  - Years: {int(df['year'].min())} to {int(df['year'].max())}")
        print(f"  - Unique areas: {df['area_code'].nunique()}")

        return output_path
