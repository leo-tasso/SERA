"""Income inequality indicator downloader (reuses Gini coefficient data)."""

import io
import json
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from sera.config import CACHE_DIR, get_indicator_data_dir
from sera.istat_client import IstatClient


class IncomeInequalityDownloader:
    """Download income inequality data from ISTAT (Gini coefficient proxy)."""

    def __init__(self, cache_dir: Optional[Path] = None):
        self.client = IstatClient(cache_dir=cache_dir or CACHE_DIR)
        self.dataflow_id = "32_221_DF_DCCV_GINIREDD_1"
        self.key = "A.IT.DISUG_REDDNET_GINI.16.1"

        self.table_mapping: dict[str, Any] = {
            "indicator": "income_inequality",
            "source": {
                "dataflow_id": self.dataflow_id,
                "key": self.key,
                "columns": {
                    "REF_AREA": "area_code",
                    "FREQ": "frequency",
                    "TIME_PERIOD": "year",
                    "OBS_VALUE": "income_inequality",
                    "DATA_TYPE": "data_type",
                },
            },
            "project": {
                "entity": "geography",
                "code_field": "area_code",
                "levels": ["national"],
                "notes": "Income inequality measured by Gini coefficient of net income distribution.",
            },
        }

    def _save_table_mapping(self, output_path: Path) -> Path:
        mapping_path = output_path.with_suffix(".mapping.json")
        mapping_path.parent.mkdir(parents=True, exist_ok=True)
        with open(mapping_path, "w", encoding="utf-8") as handle:
            json.dump(self.table_mapping, handle, indent=2, ensure_ascii=False)
        return mapping_path

    def download_income_inequality(self, start_year: int = 2001, end_year: int = 2025) -> pd.DataFrame:
        csv_data = self.client.get_data(
            flow_id=self.dataflow_id,
            key=self.key,
            start_year=start_year,
            end_year=end_year,
            format="csv",
        )

        df = pd.read_csv(io.StringIO(csv_data))
        df_clean = df[["REF_AREA", "FREQ", "TIME_PERIOD", "OBS_VALUE", "DATA_TYPE", "MEASURE", "IMPUTED_RENTS"]].copy()
        df_clean.columns = [
            "area_code",
            "frequency",
            "year",
            "income_inequality",
            "data_type",
            "measure",
            "imputed_rents",
        ]

        # Keep the configured Gini series definition explicitly.
        df_clean = df_clean[
            (df_clean["data_type"] == "DISUG_REDDNET_GINI")
            & (df_clean["measure"].astype(str) == "16")
            & (df_clean["imputed_rents"].astype(str) == "1")
        ].copy()

        df_clean["year"] = pd.to_numeric(df_clean["year"], errors="coerce")
        df_clean["income_inequality"] = pd.to_numeric(df_clean["income_inequality"], errors="coerce")
        df_clean = df_clean.dropna(subset=["year", "income_inequality"])

        return df_clean

    def save_income_inequality_csv(
        self,
        output_path: Optional[Path] = None,
        start_year: int = 2001,
        end_year: int = 2025,
    ) -> Path:
        if output_path is None:
            indicator_dir = get_indicator_data_dir("income_inequality")
            output_path = indicator_dir / f"income_inequality_raw_{start_year}_{end_year}.csv"

        print(f"Downloading income inequality data ({start_year}-{end_year})...")
        df = self.download_income_inequality(start_year=start_year, end_year=end_year)

        print(f"Saving to {output_path}...")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False, encoding="utf-8")
        mapping_path = self._save_table_mapping(output_path)

        print(f"✓ Income inequality data saved: {output_path}")
        print(f"✓ Table mapping saved: {mapping_path}")
        print(f"  - {len(df)} records")
        print(f"  - Years: {int(df['year'].min())} to {int(df['year'].max())}")
        print(f"  - Unique areas: {df['area_code'].nunique()}")

        return output_path
