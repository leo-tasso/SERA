"""Exports and imports indicator downloader."""

import io
import json
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from sera.config import CACHE_DIR, get_indicator_data_dir
from sera.istat_client import IstatClient


class ExportsImportsDownloader:
    """Download and aggregate Italy exports/imports from ISTAT."""

    def __init__(self, cache_dir: Optional[Path] = None):
        self.client = IstatClient(cache_dir=cache_dir or CACHE_DIR)
        self.dataflow_id = "139_176"
        # Monthly, Italy total, all data types, total goods (0010), world partner.
        self.key = "M.ITTOT..0010.WORLD"

        self.table_mapping: dict[str, Any] = {
            "indicator": "exports_imports",
            "source": {
                "dataflow_id": self.dataflow_id,
                "key": self.key,
                "columns": {
                    "REF_AREA": "area_code",
                    "TIME_PERIOD": "time_period",
                    "DATA_TYPE": "data_type",
                    "OBS_VALUE": "value",
                },
                "aggregation": "Monthly ESAV/ISAV summed to annual exports/imports.",
            },
            "project": {
                "entity": "geography",
                "code_field": "area_code",
                "levels": ["national"],
                "notes": "Exports and imports values aggregated by year (WORLD partner, total goods).",
            },
        }

    def _save_table_mapping(self, output_path: Path) -> Path:
        mapping_path = output_path.with_suffix(".mapping.json")
        mapping_path.parent.mkdir(parents=True, exist_ok=True)
        with open(mapping_path, "w", encoding="utf-8") as handle:
            json.dump(self.table_mapping, handle, indent=2, ensure_ascii=False)
        return mapping_path

    def download_exports_imports(
        self, start_year: int = 2001, end_year: int = 2025
    ) -> pd.DataFrame:
        csv_data = self.client.get_data(
            flow_id=self.dataflow_id,
            key=self.key,
            start_year=start_year,
            end_year=end_year,
            format="csv",
        )

        df = pd.read_csv(io.StringIO(csv_data))
        df = df[["REF_AREA", "TIME_PERIOD", "DATA_TYPE", "OBS_VALUE"]].copy()
        df["OBS_VALUE"] = pd.to_numeric(df["OBS_VALUE"], errors="coerce")
        df = df.dropna(subset=["OBS_VALUE"])

        # Keep only value series for exports/imports.
        df = df[df["DATA_TYPE"].isin(["ESAV", "ISAV"])].copy()

        # Convert monthly period to year and aggregate annual totals.
        df["time_period"] = pd.to_datetime(df["TIME_PERIOD"], format="%Y-%m", errors="coerce")
        df = df.dropna(subset=["time_period"])
        df["year"] = df["time_period"].dt.year

        grouped = (
            df.groupby(["REF_AREA", "year", "DATA_TYPE"], as_index=False)["OBS_VALUE"]
            .sum()
            .rename(columns={"REF_AREA": "area_code", "OBS_VALUE": "value"})
        )

        pivoted = grouped.pivot_table(
            index=["area_code", "year"],
            columns="DATA_TYPE",
            values="value",
            aggfunc="sum",
        ).reset_index()

        pivoted.columns.name = None
        pivoted = pivoted.rename(columns={"ESAV": "exports", "ISAV": "imports"})
        pivoted["exports"] = pd.to_numeric(pivoted.get("exports"), errors="coerce")
        pivoted["imports"] = pd.to_numeric(pivoted.get("imports"), errors="coerce")
        pivoted = pivoted.dropna(subset=["exports", "imports"], how="all")

        return pivoted

    def save_exports_imports_csv(
        self,
        output_path: Optional[Path] = None,
        start_year: int = 2001,
        end_year: int = 2025,
    ) -> Path:
        if output_path is None:
            indicator_dir = get_indicator_data_dir("exports_imports")
            output_path = indicator_dir / f"exports_imports_raw_{start_year}_{end_year}.csv"

        print(f"Downloading exports/imports data ({start_year}-{end_year})...")
        df = self.download_exports_imports(start_year=start_year, end_year=end_year)

        print(f"Saving to {output_path}...")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False, encoding="utf-8")
        mapping_path = self._save_table_mapping(output_path)

        print(f"✓ Exports/imports data saved: {output_path}")
        print(f"✓ Table mapping saved: {mapping_path}")
        print(f"  - {len(df)} records")
        print(f"  - Years: {int(df['year'].min())} to {int(df['year'].max())}")
        print(f"  - Unique areas: {df['area_code'].nunique()}")

        return output_path
