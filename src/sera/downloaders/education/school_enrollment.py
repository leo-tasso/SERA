"""School enrollment indicator downloader (ISTAT source)."""

import io
import json
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from sera.config import get_indicator_data_dir
from sera.istat_client import IstatClient


class SchoolEnrollmentDownloader:
    """Download school enrollment counts for Italy from ISTAT."""

    def __init__(self):
        self.client = IstatClient()
        # ISTAT scuola datasets with territorial detail (NUTS/province-level where available).
        self.flow_ids = [
            "52_1044_DF_DCIS_SCUOLE_5",   # Primary school
            "52_1044_DF_DCIS_SCUOLE_8",   # Lower secondary school
            "52_1044_DF_DCIS_SCUOLE_11",  # Upper secondary school
        ]

        self.table_mapping: dict[str, Any] = {
            "indicator": "school_enrollment",
            "source": {
                "provider": "ISTAT",
                "flow_ids": self.flow_ids,
                "filters": {
                    "DATA_TYPE": "ENR",
                    "SEX": "T",
                    "CITIZENSHIP": "TOTAL",
                    "TYPE_SCHOOL_MANAGEMENT": "ALL",
                    "TYPE_SCHOOL": "ALL",
                },
                "columns": {
                    "REF_AREA": "area_code",
                    "SCHOOL_LEVEL": "school_level",
                    "TIME_PERIOD": "year",
                    "OBS_VALUE": "enrollment_count",
                },
            },
            "project": {
                "entity": "geography",
                "code_field": "area_code",
                "levels": ["province", "region", "national"],
                "notes": "School enrollment (students) from ISTAT scuola datasets; series currently available from 2015 onward.",
            },
        }

    def _save_table_mapping(self, output_path: Path) -> Path:
        mapping_path = output_path.with_suffix(".mapping.json")
        mapping_path.parent.mkdir(parents=True, exist_ok=True)
        with open(mapping_path, "w", encoding="utf-8") as handle:
            json.dump(self.table_mapping, handle, indent=2, ensure_ascii=False)
        return mapping_path

    def _download_flow(self, flow_id: str, start_year: int, end_year: int) -> pd.DataFrame:
        csv_data = self.client.get_data(
            flow_id=flow_id,
            key="",
            start_year=start_year,
            end_year=end_year,
            format="csv",
        )

        df = pd.read_csv(io.StringIO(csv_data))
        if df.empty:
            return df

        filters = {
            "DATA_TYPE": "ENR",
            "SEX": "T",
            "CITIZENSHIP": "TOTAL",
            "TYPE_SCHOOL_MANAGEMENT": "ALL",
            "TYPE_SCHOOL": "ALL",
        }
        for column, expected in filters.items():
            if column in df.columns:
                df = df[df[column] == expected]

        if df.empty:
            return df

        df_clean = df[["REF_AREA", "SCHOOL_LEVEL", "TIME_PERIOD", "OBS_VALUE"]].copy()
        df_clean.columns = ["area_code", "school_level", "year", "enrollment_count"]
        df_clean["year"] = pd.to_numeric(df_clean["year"], errors="coerce")
        df_clean["enrollment_count"] = pd.to_numeric(df_clean["enrollment_count"], errors="coerce")
        df_clean = df_clean.dropna(subset=["year", "enrollment_count"])
        df_clean["school_level"] = df_clean["school_level"].astype(str)

        return df_clean

    def download_school_enrollment(self, start_year: int = 2001, end_year: int = 2025) -> pd.DataFrame:
        """Download school enrollment with the best available ISTAT geographic detail."""
        print("Fetching ISTAT school enrollment data...")

        frames = []
        for flow_id in self.flow_ids:
            print(f"  - Downloading flow {flow_id}...")
            frame = self._download_flow(flow_id=flow_id, start_year=start_year, end_year=end_year)
            if not frame.empty:
                frames.append(frame)

        if not frames:
            return pd.DataFrame(columns=["area_code", "school_level", "year", "enrollment_count", "enrollment_total"])

        df_all = pd.concat(frames, ignore_index=True)
        df_all = df_all[(df_all["year"] >= start_year) & (df_all["year"] <= end_year)]

        # Keep level-specific rows and compute total enrollment by area/year for convenience.
        totals = (
            df_all.groupby(["area_code", "year"], as_index=False)["enrollment_count"]
            .sum()
            .rename(columns={"enrollment_count": "enrollment_total"})
        )
        df_clean = df_all.merge(totals, on=["area_code", "year"], how="left")
        df_clean = df_clean.sort_values(["area_code", "year", "school_level"])
        df_clean = df_clean.drop_duplicates(subset=["area_code", "school_level", "year"])

        return df_clean

    def save_school_enrollment_csv(
        self,
        output_path: Optional[Path] = None,
        start_year: int = 2001,
        end_year: int = 2025,
    ) -> Path:
        if output_path is None:
            indicator_dir = get_indicator_data_dir("school_enrollment")
            output_path = indicator_dir / f"school_enrollment_raw_{start_year}_{end_year}.csv"

        print(f"Downloading school enrollment data ({start_year}-{end_year})...")
        df = self.download_school_enrollment(start_year=start_year, end_year=end_year)

        if df.empty:
            print("Warning: No school enrollment data retrieved, skipping save.")
            return output_path

        print(f"Saving to {output_path}...")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False, encoding="utf-8")
        mapping_path = self._save_table_mapping(output_path)

        print(f"✓ School enrollment data saved: {output_path}")
        print(f"✓ Table mapping saved: {mapping_path}")
        print(f"  - {len(df)} records")
        print(f"  - Years: {int(df['year'].min())} to {int(df['year'].max())}")
        print(f"  - Unique areas: {df['area_code'].nunique()}")

        return output_path
