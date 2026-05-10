"""R&D spending indicator downloader (ISTAT source)."""

import io
import json
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from sera.config import get_indicator_data_dir
from sera.istat_client import IstatClient


class RAndDSpendingDownloader:
    """Download R&D spending from ISTAT."""

    def __init__(self):
        self.client = IstatClient()
        self.flow_id = "6_20_DF_DCSP_RS_3"
        self.table_mapping: dict[str, Any] = {
            "indicator": "r_and_d_spending",
            "source": {
                "provider": "ISTAT",
                "flow_id": self.flow_id,
                "filters": {
                    "DATA_TYPE": "DRDEIM",
                    "INSTITUTIONAL_SECTOR": "S1",
                    "ECON_ACTIVITY_NACE_2007": "10",
                    "PERS_EMPL_SIZE_CLASS": "TOTAL",
                    "TYPE_RESEAR_DEVELOP": "310",
                },
                "columns": {
                    "REF_AREA": "area_code",
                    "TIME_PERIOD": "year",
                    "OBS_VALUE": "rd_spending_eur",
                },
            },
            "project": {
                "entity": "geography",
                "code_field": "area_code",
                "levels": ["national"],
                "notes": "Gross domestic expenditure on R&D (euro), ISTAT dataflow 6_20_DF_DCSP_RS_3.",
            },
        }

    def _save_table_mapping(self, output_path: Path) -> Path:
        mapping_path = output_path.with_suffix(".mapping.json")
        mapping_path.parent.mkdir(parents=True, exist_ok=True)
        with open(mapping_path, "w", encoding="utf-8") as handle:
            json.dump(self.table_mapping, handle, indent=2, ensure_ascii=False)
        return mapping_path

    def download_r_and_d_spending(self, start_year: int = 2001, end_year: int = 2025) -> pd.DataFrame:
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
            return pd.DataFrame(columns=["area_code", "year", "rd_spending_eur"])

        df_clean = df[["REF_AREA", "TIME_PERIOD", "OBS_VALUE"]].copy()
        df_clean.columns = ["area_code", "year", "rd_spending_eur"]
        df_clean["year"] = pd.to_numeric(df_clean["year"], errors="coerce")
        df_clean["rd_spending_eur"] = pd.to_numeric(df_clean["rd_spending_eur"], errors="coerce")
        df_clean = df_clean.dropna(subset=["year", "rd_spending_eur"])
        df_clean = df_clean[(df_clean["year"] >= start_year) & (df_clean["year"] <= end_year)]
        df_clean = df_clean.sort_values(["area_code", "year"]).drop_duplicates(subset=["area_code", "year"])
        return df_clean

    def save_r_and_d_spending_csv(
        self,
        output_path: Optional[Path] = None,
        start_year: int = 2001,
        end_year: int = 2025,
    ) -> Path:
        if output_path is None:
            indicator_dir = get_indicator_data_dir("r_and_d_spending")
            output_path = indicator_dir / f"r_and_d_spending_raw_{start_year}_{end_year}.csv"

        print(f"Downloading R&D spending data ({start_year}-{end_year})...")
        df = self.download_r_and_d_spending(start_year=start_year, end_year=end_year)

        if df.empty:
            print("Warning: No R&D spending data retrieved, skipping save.")
            return output_path

        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False, encoding="utf-8")
        mapping_path = self._save_table_mapping(output_path)

        print(f"✓ R&D spending data saved: {output_path}")
        print(f"✓ Table mapping saved: {mapping_path}")
        print(f"  - {len(df)} records")
        print(f"  - Years: {int(df['year'].min())} to {int(df['year'].max())}")
        print(f"  - Unique areas: {df['area_code'].nunique()}")
        return output_path
