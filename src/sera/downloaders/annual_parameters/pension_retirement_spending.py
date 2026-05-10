"""pension retirement spending annual input parameter downloader (ISTAT source)."""

import io
import json
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from sera.config import get_indicator_data_dir
from sera.istat_client import IstatClient


class PensionRetirementSpendingDownloader:
    """Download annual pension_retirement_spending proxy for Italy from ISTAT."""

    def __init__(self):
        self.client = IstatClient()
        self.flow_id = "46_812_DF_DCAR_PENSIONATI2_4"

        self.table_mapping: dict[str, Any] = {
            "parameter": "pension_retirement_spending",
            "source": {
                "provider": "ISTAT",
                "flow_id": self.flow_id,
                "dataset": "Pensioners - indicators (regional)",
                "columns": {
                    "REF_AREA": "area_code",
                    "TIME_PERIOD": "year",
                    "OBS_VALUE": "pension_retirement_spending",
                },
            },
            "project": {
                "entity": "policy",
                "code_field": "area_code",
                "levels": ["national", "regional"],
                "notes": "Annual proxy derived from ISTAT pension indicators at regional level.",
            },
        }

    def _save_table_mapping(self, output_path: Path) -> Path:
        mapping_path = output_path.with_suffix(".mapping.json")
        mapping_path.parent.mkdir(parents=True, exist_ok=True)
        with open(mapping_path, "w", encoding="utf-8") as handle:
            json.dump(self.table_mapping, handle, indent=2, ensure_ascii=False)
        return mapping_path

    def download_pension_retirement_spending(self, start_year: int = 2001, end_year: int = 2025) -> pd.DataFrame:
        csv_data = self.client.get_data(
            flow_id=self.flow_id,
            key="",
            start_year=start_year,
            end_year=end_year,
            format="csv",
        )
        df = pd.read_csv(io.StringIO(csv_data))
        required = {"REF_AREA", "TIME_PERIOD", "OBS_VALUE"}
        if not required.issubset(df.columns):
            raise RuntimeError("Unexpected ISTAT response format for pension_retirement_spending.")

        if "FREQ" in df.columns and (df["FREQ"] == "A").any():
            df = df[df["FREQ"] == "A"]

        df_clean = df[["REF_AREA", "TIME_PERIOD", "OBS_VALUE"]].copy()
        df_clean.columns = ["area_code", "year", "pension_retirement_spending"]

        df_clean["year"] = df_clean["year"].astype(str).str[:4]
        df_clean["year"] = pd.to_numeric(df_clean["year"], errors="coerce")
        df_clean["pension_retirement_spending"] = pd.to_numeric(df_clean["pension_retirement_spending"], errors="coerce")
        df_clean = df_clean.dropna(subset=["year", "pension_retirement_spending"])
        df_clean = df_clean[(df_clean["year"] >= start_year) & (df_clean["year"] <= end_year)]
        df_clean = (
            df_clean.groupby(["area_code", "year"], as_index=False)["pension_retirement_spending"]
            .mean()
            .sort_values(["area_code", "year"])
        )

        return df_clean

    def save_pension_retirement_spending_csv(
        self,
        output_path: Optional[Path] = None,
        start_year: int = 2001,
        end_year: int = 2025,
    ) -> Path:
        if output_path is None:
            indicator_dir = get_indicator_data_dir("pension_retirement_spending")
            output_path = indicator_dir / f"pension_retirement_spending_raw_{start_year}_{end_year}.csv"

        print(f"Downloading pension retirement spending data ({start_year}-{end_year})...")
        df = self.download_pension_retirement_spending(start_year=start_year, end_year=end_year)

        print(f"Saving to {output_path}...")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False, encoding="utf-8")
        mapping_path = self._save_table_mapping(output_path)

        print(f"✓ pension retirement spending data saved: {output_path}")
        print(f"✓ Table mapping saved: {mapping_path}")
        print(f"  - {len(df)} records")
        print(f"  - Years: {int(df['year'].min())} to {int(df['year'].max())}")
        print(f"  - Unique areas: {df['area_code'].nunique()}")

        return output_path
