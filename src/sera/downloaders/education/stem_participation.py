"""STEM participation indicator downloader (ISTAT source)."""

import io
import json
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from sera.config import get_indicator_data_dir
from sera.istat_client import IstatClient


class StemParticipationDownloader:
    """Download STEM participation from ISTAT university enrollment by field of study."""

    def __init__(self):
        self.client = IstatClient()
        self.flow_id = "56_1046_DF_DCIS_ISCRITTI_COM_R_3"

        # STEM-focused groups from ISTAT CL_AREADIDATTICA codelist.
        self.stem_field_codes = {
            "1", "2", "3", "5", "17", "18", "19", "20", "21", "24", "25", "42", "43", "44", "45"
        }
        self.total_field_code = "99"

        self.table_mapping: dict[str, Any] = {
            "indicator": "stem_participation",
            "source": {
                "provider": "ISTAT",
                "flow_id": self.flow_id,
                "filters": {
                    "DATA_TYPE": "16",
                    "SEX": "9",
                    "CITIZENSHIP": "TOTAL",
                },
                "stem_field_codes": sorted(self.stem_field_codes),
                "total_field_code": self.total_field_code,
                "columns": {
                    "REF_AREA": "area_code",
                    "FIELD_STUDY": "field_study_code",
                    "TIME_PERIOD": "year",
                    "OBS_VALUE": "enrolled_students",
                },
            },
            "project": {
                "entity": "geography",
                "code_field": "area_code",
                "levels": ["municipality", "province", "region", "national"],
                "notes": "STEM participation proxy = STEM enrolled / total enrolled in university fields, based on ISTAT field-of-study codelists.",
            },
        }

    def _save_table_mapping(self, output_path: Path) -> Path:
        mapping_path = output_path.with_suffix(".mapping.json")
        mapping_path.parent.mkdir(parents=True, exist_ok=True)
        with open(mapping_path, "w", encoding="utf-8") as handle:
            json.dump(self.table_mapping, handle, indent=2, ensure_ascii=False)
        return mapping_path

    def download_stem_participation(self, start_year: int = 2001, end_year: int = 2025) -> pd.DataFrame:
        """Download STEM participation with ISTAT geographic detail."""
        print(f"Fetching ISTAT STEM participation data from {self.flow_id}...")

        # This flow is currently available for 2015-2017 in ISTAT API.
        effective_start = max(start_year, 2015)
        effective_end = min(end_year, 2017)
        if effective_start > effective_end:
            return pd.DataFrame(columns=["area_code", "year", "stem_enrolled", "total_enrolled", "stem_participation_rate"])

        csv_data = self.client.get_data(
            flow_id=self.flow_id,
            key="",
            start_year=effective_start,
            end_year=effective_end + 1,
            format="csv",
        )
        df = pd.read_csv(io.StringIO(csv_data), low_memory=False)

        if df.empty:
            return pd.DataFrame(columns=["area_code", "year", "stem_enrolled", "total_enrolled", "stem_participation_rate"])

        filters = {"DATA_TYPE": "16", "SEX": "9", "CITIZENSHIP": "TOTAL"}
        for column, expected in filters.items():
            if column in df.columns:
                df = df[df[column].astype(str) == expected]

        if df.empty:
            return pd.DataFrame(columns=["area_code", "year", "stem_enrolled", "total_enrolled", "stem_participation_rate"])

        df_clean = df[["REF_AREA", "FIELD_STUDY", "TIME_PERIOD", "OBS_VALUE"]].copy()
        df_clean.columns = ["area_code", "field_study_code", "year", "enrolled_students"]
        df_clean["year"] = pd.to_numeric(df_clean["year"], errors="coerce")
        df_clean["enrolled_students"] = pd.to_numeric(df_clean["enrolled_students"], errors="coerce")
        df_clean["field_study_code"] = df_clean["field_study_code"].astype(str)
        df_clean = df_clean.dropna(subset=["year", "enrolled_students"])
        df_clean = df_clean[(df_clean["year"] >= start_year) & (df_clean["year"] <= end_year)]

        total = (
            df_clean[df_clean["field_study_code"] == self.total_field_code]
            .groupby(["area_code", "year"], as_index=False)["enrolled_students"]
            .sum()
            .rename(columns={"enrolled_students": "total_enrolled"})
        )
        stem = (
            df_clean[df_clean["field_study_code"].isin(self.stem_field_codes)]
            .groupby(["area_code", "year"], as_index=False)["enrolled_students"]
            .sum()
            .rename(columns={"enrolled_students": "stem_enrolled"})
        )

        merged = total.merge(stem, on=["area_code", "year"], how="left")
        merged["stem_enrolled"] = merged["stem_enrolled"].fillna(0.0)
        merged = merged[merged["total_enrolled"] > 0]
        merged["stem_participation_rate"] = (merged["stem_enrolled"] / merged["total_enrolled"]) * 100.0
        merged = merged.sort_values(["area_code", "year"]).drop_duplicates(subset=["area_code", "year"])

        return merged

    def save_stem_participation_csv(
        self,
        output_path: Optional[Path] = None,
        start_year: int = 2001,
        end_year: int = 2025,
    ) -> Path:
        if output_path is None:
            indicator_dir = get_indicator_data_dir("stem_participation")
            output_path = indicator_dir / f"stem_participation_raw_{start_year}_{end_year}.csv"

        print(f"Downloading STEM participation data ({start_year}-{end_year})...")
        df = self.download_stem_participation(start_year=start_year, end_year=end_year)

        if df.empty:
            print("Warning: No STEM participation data retrieved, skipping save.")
            return output_path

        print(f"Saving to {output_path}...")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False, encoding="utf-8")
        mapping_path = self._save_table_mapping(output_path)

        print(f"✓ STEM participation data saved: {output_path}")
        print(f"✓ Table mapping saved: {mapping_path}")
        print(f"  - {len(df)} records")
        print(f"  - Years: {int(df['year'].min())} to {int(df['year'].max())}")
        print(f"  - Unique areas: {df['area_code'].nunique()}")

        return output_path
