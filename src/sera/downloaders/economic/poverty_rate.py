"""Poverty rate indicator downloader."""

import io
import json
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from sera.config import CACHE_DIR, get_indicator_data_dir
from sera.istat_client import IstatClient


class PovertyRateDownloader:
    """Download poverty rate data from ISTAT."""

    def __init__(self, cache_dir: Optional[Path] = None):
        self.client = IstatClient(cache_dir=cache_dir or CACHE_DIR)
        self.dataflow_id = "34_727_DF_DCCV_POVERTA_1"
        # Relative poverty incidence of families, national total.
        self.key = "A.IT.INCID_POVREL_FAM.ALL.TOT.HH.TOTAL.99.ALL.9.TOTAL"

        self.table_mapping: dict[str, Any] = {
            "indicator": "poverty_rate",
            "source": {
                "dataflow_id": self.dataflow_id,
                "key": self.key,
                "columns": {
                    "REF_AREA": "area_code",
                    "FREQ": "frequency",
                    "TIME_PERIOD": "year",
                    "OBS_VALUE": "poverty_rate",
                    "DATA_TYPE": "data_type",
                },
            },
            "project": {
                "entity": "geography",
                "code_field": "area_code",
                "levels": ["national"],
                "notes": "Relative poverty incidence (%), families.",
            },
        }

    def _save_table_mapping(self, output_path: Path) -> Path:
        mapping_path = output_path.with_suffix(".mapping.json")
        mapping_path.parent.mkdir(parents=True, exist_ok=True)
        with open(mapping_path, "w", encoding="utf-8") as handle:
            json.dump(self.table_mapping, handle, indent=2, ensure_ascii=False)
        return mapping_path

    def download_poverty_rate(self, start_year: int = 2001, end_year: int = 2025) -> pd.DataFrame:
        csv_data = self.client.get_data(
            flow_id=self.dataflow_id,
            key=self.key,
            start_year=start_year,
            end_year=end_year,
            format="csv",
        )

        df = pd.read_csv(io.StringIO(csv_data))
        df_clean = df[["REF_AREA", "FREQ", "TIME_PERIOD", "OBS_VALUE", "DATA_TYPE"]].copy()
        df_clean.columns = ["area_code", "frequency", "year", "poverty_rate", "data_type"]

        # Keep only the intended poverty-rate measure if broader keys are queried.
        df_clean = df_clean[df_clean["data_type"] == "INCID_POVREL_FAM"].copy()

        df_clean["year"] = pd.to_numeric(df_clean["year"], errors="coerce")
        df_clean["poverty_rate"] = pd.to_numeric(df_clean["poverty_rate"], errors="coerce")
        df_clean = df_clean.dropna(subset=["year", "poverty_rate"])

        return df_clean

    def save_poverty_rate_csv(
        self,
        output_path: Optional[Path] = None,
        start_year: int = 2001,
        end_year: int = 2025,
    ) -> Path:
        if output_path is None:
            indicator_dir = get_indicator_data_dir("poverty_rate")
            output_path = indicator_dir / f"poverty_rate_raw_{start_year}_{end_year}.csv"

        print(f"Downloading poverty rate data ({start_year}-{end_year})...")
        df = self.download_poverty_rate(start_year=start_year, end_year=end_year)

        print(f"Saving to {output_path}...")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False, encoding="utf-8")
        mapping_path = self._save_table_mapping(output_path)

        print(f"✓ Poverty rate data saved: {output_path}")
        print(f"✓ Table mapping saved: {mapping_path}")
        print(f"  - {len(df)} records")
        print(f"  - Years: {int(df['year'].min())} to {int(df['year'].max())}")
        print(f"  - Unique areas: {df['area_code'].nunique()}")

        return output_path
