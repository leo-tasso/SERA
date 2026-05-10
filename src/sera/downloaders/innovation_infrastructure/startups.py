"""Startups indicator downloader (ISTAT enterprise-age proxy)."""

import io
import json
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from sera.config import get_indicator_data_dir
from sera.istat_client import IstatClient


class StartupsDownloader:
    """Download startup proxy from enterprise-age employment distribution."""

    def __init__(self):
        self.client = IstatClient()
        self.flow_id = "183_203_DF_DICA_ACDP_20"
        self.table_mapping: dict[str, Any] = {
            "indicator": "startups",
            "source": {
                "provider": "ISTAT",
                "flow_id": self.flow_id,
                "filters": {
                    "DATA_TYPE": "AENTEMPYAA",
                    "LEGAL_FORM": "TOT",
                    "PERS_EMPL_SIZE_CLASS": "TOTAL",
                    "ECON_ACTIVITY_NACE_2007": "10",
                    "SEX": "9",
                    "AGE": "Y_GE15",
                    "COUNTRY_BIRTH": "WORLD",
                },
                "startup_age_bucket": "Y0-2",
                "total_age_bucket": "TOTAL",
                "columns": {
                    "REF_AREA": "area_code",
                    "AGE_ENTERPRISE": "age_enterprise",
                    "TIME_PERIOD": "year",
                    "OBS_VALUE": "employees",
                },
            },
            "project": {
                "entity": "geography",
                "code_field": "area_code",
                "levels": ["province", "region", "national"],
                "notes": "Startup proxy: employees in enterprises aged 0-2 over total employees in active enterprises.",
            },
        }

    def _save_table_mapping(self, output_path: Path) -> Path:
        mapping_path = output_path.with_suffix(".mapping.json")
        mapping_path.parent.mkdir(parents=True, exist_ok=True)
        with open(mapping_path, "w", encoding="utf-8") as handle:
            json.dump(self.table_mapping, handle, indent=2, ensure_ascii=False)
        return mapping_path

    def download_startups(self, start_year: int = 2001, end_year: int = 2025) -> pd.DataFrame:
        csv_data = self.client.get_data(
            flow_id=self.flow_id,
            key="",
            start_year=start_year,
            end_year=end_year,
            format="csv",
        )
        df = pd.read_csv(io.StringIO(csv_data), low_memory=False)

        filters = self.table_mapping["source"]["filters"]
        for column, expected in filters.items():
            if column in df.columns:
                df = df[df[column].astype(str) == expected]

        if df.empty:
            return pd.DataFrame(columns=["area_code", "year", "startup_employees", "total_employees", "startup_share"])

        df_clean = df[["REF_AREA", "AGE_ENTERPRISE", "TIME_PERIOD", "OBS_VALUE"]].copy()
        df_clean.columns = ["area_code", "age_enterprise", "year", "employees"]
        df_clean["year"] = pd.to_numeric(df_clean["year"], errors="coerce")
        df_clean["employees"] = pd.to_numeric(df_clean["employees"], errors="coerce")
        df_clean = df_clean.dropna(subset=["year", "employees"])
        df_clean = df_clean[(df_clean["year"] >= start_year) & (df_clean["year"] <= end_year)]

        total = (
            df_clean[df_clean["age_enterprise"].astype(str) == "TOTAL"]
            .groupby(["area_code", "year"], as_index=False)["employees"]
            .sum()
            .rename(columns={"employees": "total_employees"})
        )
        startup = (
            df_clean[df_clean["age_enterprise"].astype(str) == "Y0-2"]
            .groupby(["area_code", "year"], as_index=False)["employees"]
            .sum()
            .rename(columns={"employees": "startup_employees"})
        )

        merged = total.merge(startup, on=["area_code", "year"], how="left")
        merged["startup_employees"] = merged["startup_employees"].fillna(0.0)
        merged = merged[merged["total_employees"] > 0]
        merged["startup_share"] = (merged["startup_employees"] / merged["total_employees"]) * 100.0
        merged = merged.sort_values(["area_code", "year"]).drop_duplicates(subset=["area_code", "year"])
        return merged

    def save_startups_csv(
        self,
        output_path: Optional[Path] = None,
        start_year: int = 2001,
        end_year: int = 2025,
    ) -> Path:
        if output_path is None:
            indicator_dir = get_indicator_data_dir("startups")
            output_path = indicator_dir / f"startups_raw_{start_year}_{end_year}.csv"

        print(f"Downloading startups data ({start_year}-{end_year})...")
        df = self.download_startups(start_year=start_year, end_year=end_year)

        if df.empty:
            print("Warning: No startups data retrieved, skipping save.")
            return output_path

        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False, encoding="utf-8")
        mapping_path = self._save_table_mapping(output_path)

        print(f"✓ Startups data saved: {output_path}")
        print(f"✓ Table mapping saved: {mapping_path}")
        print(f"  - {len(df)} records")
        print(f"  - Years: {int(df['year'].min())} to {int(df['year'].max())}")
        print(f"  - Unique areas: {df['area_code'].nunique()}")
        return output_path
