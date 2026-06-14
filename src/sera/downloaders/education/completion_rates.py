"""Completion rates indicator downloader (ISTAT source)."""

import io
import json
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from sera.config import get_indicator_data_dir
from sera.istat_client import IstatClient


class CompletionRatesDownloader:
    """Download upper-secondary completion rates from ISTAT."""

    def __init__(self):
        self.client = IstatClient()
        self.enrollment_flow_id = "52_1044_DF_DCIS_SCUOLE_11"  # Upper-secondary enrolled students
        self.graduates_flow_id = "52_1044_DF_DCIS_SCUOLE_14"  # Upper-secondary graduates

        self.table_mapping: dict[str, Any] = {
            "indicator": "completion_rates",
            "source": {
                "provider": "ISTAT",
                "flow_ids": [self.enrollment_flow_id, self.graduates_flow_id],
                "filters": {
                    "enrollment": {
                        "DATA_TYPE": "ENR",
                        "SEX": "T",
                        "CITIZENSHIP": "TOTAL",
                        "TYPE_SCHOOL_MANAGEMENT": "ALL",
                        "TYPE_SCHOOL": "ALL",
                    },
                    "graduates": {
                        "DATA_TYPE": "GRAD",
                        "SEX": "T",
                        "CITIZENSHIP": "TOTAL",
                        "TYPE_SCHOOL_MANAGEMENT": "ALL",
                        "TYPE_SCHOOL": "ALL",
                    },
                },
                "columns": {
                    "REF_AREA": "area_code",
                    "TIME_PERIOD": "year",
                    "OBS_VALUE": "value",
                },
            },
            "project": {
                "entity": "geography",
                "code_field": "area_code",
                "levels": ["province", "region", "national"],
                "notes": "Completion proxy computed as graduates/enrolled * 100 for upper-secondary education.",
            },
        }

    def _save_table_mapping(self, output_path: Path) -> Path:
        mapping_path = output_path.with_suffix(".mapping.json")
        mapping_path.parent.mkdir(parents=True, exist_ok=True)
        with open(mapping_path, "w", encoding="utf-8") as handle:
            json.dump(self.table_mapping, handle, indent=2, ensure_ascii=False)
        return mapping_path

    def _download_filtered(self, flow_id: str, filters: dict[str, str]) -> pd.DataFrame:
        csv_data = self.client.get_data(
            flow_id=flow_id,
            key="",
            format="csv",
        )
        df = pd.read_csv(io.StringIO(csv_data), low_memory=False)
        if df.empty:
            return df

        for column, expected in filters.items():
            if column in df.columns:
                df = df[df[column] == expected]

        if df.empty:
            return df

        df_clean = df[["REF_AREA", "TIME_PERIOD", "OBS_VALUE"]].copy()
        df_clean.columns = ["area_code", "year", "value"]
        df_clean["year"] = pd.to_numeric(df_clean["year"], errors="coerce")
        df_clean["value"] = pd.to_numeric(df_clean["value"], errors="coerce")
        df_clean = df_clean.dropna(subset=["year", "value"])

        return df_clean

    def download_completion_rates(
        self, start_year: int = 2001, end_year: int = 2025
    ) -> pd.DataFrame:
        """Download completion rates with best available ISTAT geographic granularity."""
        print("Fetching ISTAT completion rates data...")

        enrollment = self._download_filtered(
            flow_id=self.enrollment_flow_id,
            filters=self.table_mapping["source"]["filters"]["enrollment"],
        )
        graduates = self._download_filtered(
            flow_id=self.graduates_flow_id,
            filters=self.table_mapping["source"]["filters"]["graduates"],
        )

        if enrollment.empty or graduates.empty:
            return pd.DataFrame(
                columns=["area_code", "year", "enrolled_students", "graduates", "completion_rate"]
            )

        enrollment = enrollment.rename(columns={"value": "enrolled_students"})
        graduates = graduates.rename(columns={"value": "graduates"})

        merged = enrollment.merge(graduates, on=["area_code", "year"], how="inner")
        merged = merged[(merged["year"] >= start_year) & (merged["year"] <= end_year)]
        merged = merged[merged["enrolled_students"] > 0]
        merged["completion_rate"] = (merged["graduates"] / merged["enrolled_students"]) * 100.0

        merged = merged.sort_values(["area_code", "year"]).drop_duplicates(
            subset=["area_code", "year"]
        )

        return merged

    def save_completion_rates_csv(
        self,
        output_path: Optional[Path] = None,
        start_year: int = 2001,
        end_year: int = 2025,
    ) -> Path:
        if output_path is None:
            indicator_dir = get_indicator_data_dir("completion_rates")
            output_path = indicator_dir / f"completion_rates_raw_{start_year}_{end_year}.csv"

        print(f"Downloading completion rates data ({start_year}-{end_year})...")
        df = self.download_completion_rates(start_year=start_year, end_year=end_year)

        if df.empty:
            print("Warning: No completion rates data retrieved, skipping save.")
            return output_path

        print(f"Saving to {output_path}...")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False, encoding="utf-8")
        mapping_path = self._save_table_mapping(output_path)

        print(f"✓ Completion rates data saved: {output_path}")
        print(f"✓ Table mapping saved: {mapping_path}")
        print(f"  - {len(df)} records")
        print(f"  - Years: {int(df['year'].min())} to {int(df['year'].max())}")
        print(f"  - Unique areas: {df['area_code'].nunique()}")

        return output_path
