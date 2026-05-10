"""Construction activity indicator downloader (ISTAT source)."""

import io
import json
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from sera.config import get_indicator_data_dir
from sera.istat_client import IstatClient


class ConstructionActivityDownloader:
    """Download construction activity as annual average construction production index."""

    def __init__(self):
        self.client = IstatClient()
        self.flow_id = "115_362_DF_DCSC_INDXPRODCOSTR_1_1"
        self.table_mapping: dict[str, Any] = {
            "indicator": "construction_activity",
            "source": {
                "provider": "ISTAT",
                "flow_id": self.flow_id,
                "filters": {
                    "DATA_TYPE": "CONS_PROD2",
                    "REF_AREA": "IT",
                },
                "columns": {
                    "REF_AREA": "area_code",
                    "TIME_PERIOD": "year",
                    "OBS_VALUE": "construction_production_index",
                },
            },
            "project": {
                "entity": "geography",
                "code_field": "area_code",
                "levels": ["national"],
                "notes": "Annual average construction production index from monthly ISTAT series.",
            },
        }

    def _save_table_mapping(self, output_path: Path) -> Path:
        mapping_path = output_path.with_suffix(".mapping.json")
        mapping_path.parent.mkdir(parents=True, exist_ok=True)
        with open(mapping_path, "w", encoding="utf-8") as handle:
            json.dump(self.table_mapping, handle, indent=2, ensure_ascii=False)
        return mapping_path

    def download_construction_activity(self, start_year: int = 2001, end_year: int = 2025) -> pd.DataFrame:
        csv_data = self.client.get_data(flow_id=self.flow_id, key="", start_year=start_year, end_year=end_year, format="csv")
        df = pd.read_csv(io.StringIO(csv_data), low_memory=False)

        for column, expected in self.table_mapping["source"]["filters"].items():
            if column in df.columns:
                df = df[df[column].astype(str) == expected]

        if df.empty:
            return pd.DataFrame(columns=["area_code", "year", "construction_production_index"])

        df["year"] = df["TIME_PERIOD"].astype(str).str.slice(0, 4)
        df["year"] = pd.to_numeric(df["year"], errors="coerce")
        df["construction_production_index"] = pd.to_numeric(df["OBS_VALUE"], errors="coerce")
        df = df.dropna(subset=["year", "construction_production_index"])
        df = df[(df["year"] >= start_year) & (df["year"] <= end_year)]

        annual = (
            df.groupby(["REF_AREA", "year"], as_index=False)["construction_production_index"]
            .mean()
            .rename(columns={"REF_AREA": "area_code"})
            .sort_values(["area_code", "year"])
        )
        return annual

    def save_construction_activity_csv(self, output_path: Optional[Path] = None, start_year: int = 2001, end_year: int = 2025) -> Path:
        if output_path is None:
            indicator_dir = get_indicator_data_dir("construction_activity")
            output_path = indicator_dir / f"construction_activity_raw_{start_year}_{end_year}.csv"
        print(f"Downloading construction activity data ({start_year}-{end_year})...")
        df = self.download_construction_activity(start_year=start_year, end_year=end_year)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False, encoding="utf-8")
        mapping_path = self._save_table_mapping(output_path)
        print(f"✓ Construction activity data saved: {output_path}")
        print(f"✓ Table mapping saved: {mapping_path}")
        print(f"  - {len(df)} records")
        if not df.empty:
            print(f"  - Years: {int(df['year'].min())} to {int(df['year'].max())}")
            print(f"  - Unique areas: {df['area_code'].nunique()}")
        return output_path
