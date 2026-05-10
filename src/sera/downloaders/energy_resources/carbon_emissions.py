"""Carbon emissions downloader (ISTAT source)."""

import io
import json
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from sera.config import get_indicator_data_dir
from sera.istat_client import IstatClient


class CarbonEmissionsDownloader:
    """Download national carbon emissions from ISTAT NAMEA tables."""

    def __init__(self):
        self.client = IstatClient()
        self.flow_id = "97_187"
        self.table_mapping: dict[str, Any] = {
            "indicator": "carbon_emissions",
            "source": {
                "provider": "ISTAT",
                "flow_id": self.flow_id,
                "filters": {
                    "DATA_TYPE_AGGR": "AE_T",
                    "POLLUTANTS_ENViSS": "CO2",
                    "CAUSE_EMISSIONS": "TOT",
                    "BRKDW_INDUSTRY_NACE_REV2": "Z",
                },
                "columns": {
                    "REF_AREA": "area_code",
                    "TIME_PERIOD": "year",
                    "OBS_VALUE": "carbon_emissions",
                },
            },
            "project": {
                "entity": "policy",
                "code_field": "area_code",
                "levels": ["national"],
                "notes": "National CO2 emissions proxy from ISTAT NAMEA emissions table.",
            },
        }

    def _save_table_mapping(self, output_path: Path) -> Path:
        mapping_path = output_path.with_suffix(".mapping.json")
        mapping_path.parent.mkdir(parents=True, exist_ok=True)
        with open(mapping_path, "w", encoding="utf-8") as handle:
            json.dump(self.table_mapping, handle, indent=2, ensure_ascii=False)
        return mapping_path

    def download_carbon_emissions(self, start_year: int = 2001, end_year: int = 2025) -> pd.DataFrame:
        csv_data = self.client.get_data(flow_id=self.flow_id, key="", start_year=start_year, end_year=end_year, format="csv")
        df = pd.read_csv(io.StringIO(csv_data), low_memory=False)
        for column, expected in self.table_mapping["source"]["filters"].items():
            if column in df.columns:
                df = df[df[column].astype(str) == expected]
        if df.empty:
            return pd.DataFrame(columns=["area_code", "year", "carbon_emissions"])
        df_clean = df[["REF_AREA", "TIME_PERIOD", "OBS_VALUE"]].copy()
        df_clean.columns = ["area_code", "year", "carbon_emissions"]
        df_clean["year"] = pd.to_numeric(df_clean["year"], errors="coerce")
        df_clean["carbon_emissions"] = pd.to_numeric(df_clean["carbon_emissions"], errors="coerce")
        df_clean = df_clean.dropna(subset=["year", "carbon_emissions"])
        df_clean = df_clean[(df_clean["year"] >= start_year) & (df_clean["year"] <= end_year)]
        return df_clean.sort_values(["area_code", "year"])

    def save_carbon_emissions_csv(self, output_path: Optional[Path] = None, start_year: int = 2001, end_year: int = 2025) -> Path:
        if output_path is None:
            indicator_dir = get_indicator_data_dir("carbon_emissions")
            output_path = indicator_dir / f"carbon_emissions_raw_{start_year}_{end_year}.csv"
        print(f"Downloading carbon emissions data ({start_year}-{end_year})...")
        df = self.download_carbon_emissions(start_year=start_year, end_year=end_year)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False, encoding="utf-8")
        mapping_path = self._save_table_mapping(output_path)
        print(f"✓ Carbon emissions data saved: {output_path}")
        print(f"✓ Table mapping saved: {mapping_path}")
        print(f"  - {len(df)} records")
        if not df.empty:
            print(f"  - Years: {int(df['year'].min())} to {int(df['year'].max())}")
            print(f"  - Unique areas: {df['area_code'].nunique()}")
        return output_path
