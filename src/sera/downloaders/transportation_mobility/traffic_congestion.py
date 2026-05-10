"""Traffic congestion downloader (ISTAT proxy source)."""

import io
import json
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from sera.config import get_indicator_data_dir
from sera.istat_client import IstatClient


class TrafficCongestionDownloader:
    """Download traffic congestion proxy from ISTAT household mobility survey data."""

    def __init__(self):
        self.client = IstatClient()
        self.flow_id = "82_87_DF_DCCV_AVQ_FAMIGLIE_104"
        self.table_mapping: dict[str, Any] = {
            "indicator": "traffic_congestion",
            "source": {
                "provider": "ISTAT",
                "flow_id": self.flow_id,
                "filters": {
                    "DATA_TYPE": "HOUS_DIFF_TRANS_V",
                    "MEASURE": "HSC_F",
                    "NUMBER_HOUSEHOLD_COMP": "TOT",
                },
                "columns": {
                    "REF_AREA": "area_code",
                    "TIME_PERIOD": "year",
                    "OBS_VALUE": "traffic_congestion",
                },
            },
            "project": {
                "entity": "geography",
                "code_field": "area_code",
                "levels": ["region", "national"],
                "notes": "Proxy from households reporting road/transport difficulty due to congestion and mobility conditions.",
            },
        }

    def _save_table_mapping(self, output_path: Path) -> Path:
        mapping_path = output_path.with_suffix(".mapping.json")
        mapping_path.parent.mkdir(parents=True, exist_ok=True)
        with open(mapping_path, "w", encoding="utf-8") as handle:
            json.dump(self.table_mapping, handle, indent=2, ensure_ascii=False)
        return mapping_path

    def download_traffic_congestion(self, start_year: int = 2001, end_year: int = 2025) -> pd.DataFrame:
        csv_data = self.client.get_data(flow_id=self.flow_id, key="", start_year=start_year, end_year=end_year, format="csv")
        df = pd.read_csv(io.StringIO(csv_data), low_memory=False)
        for column, expected in self.table_mapping["source"]["filters"].items():
            if column in df.columns:
                df = df[df[column].astype(str) == expected]
        if df.empty:
            return pd.DataFrame(columns=["area_code", "year", "traffic_congestion"])
        df_clean = df[["REF_AREA", "TIME_PERIOD", "OBS_VALUE"]].copy()
        df_clean.columns = ["area_code", "year", "traffic_congestion"]
        df_clean["year"] = pd.to_numeric(df_clean["year"], errors="coerce")
        df_clean["traffic_congestion"] = pd.to_numeric(df_clean["traffic_congestion"], errors="coerce")
        df_clean = df_clean.dropna(subset=["year", "traffic_congestion"])
        df_clean = df_clean[(df_clean["year"] >= start_year) & (df_clean["year"] <= end_year)]
        return df_clean.sort_values(["area_code", "year"])

    def save_traffic_congestion_csv(self, output_path: Optional[Path] = None, start_year: int = 2001, end_year: int = 2025) -> Path:
        if output_path is None:
            indicator_dir = get_indicator_data_dir("traffic_congestion")
            output_path = indicator_dir / f"traffic_congestion_raw_{start_year}_{end_year}.csv"
        print(f"Downloading traffic congestion proxy data ({start_year}-{end_year})...")
        df = self.download_traffic_congestion(start_year=start_year, end_year=end_year)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False, encoding="utf-8")
        mapping_path = self._save_table_mapping(output_path)
        print(f"✓ Traffic congestion proxy data saved: {output_path}")
        print(f"✓ Table mapping saved: {mapping_path}")
        print(f"  - {len(df)} records")
        if not df.empty:
            print(f"  - Years: {int(df['year'].min())} to {int(df['year'].max())}")
            print(f"  - Unique areas: {df['area_code'].nunique()}")
        return output_path
