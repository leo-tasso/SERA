"""GDP per capita indicator downloader."""

import io
import json
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from sera.config import CACHE_DIR, get_indicator_data_dir
from sera.istat_client import IstatClient


class GdpPerCapitaDownloader:
    """Download GDP per capita data from ISTAT territorial national accounts."""

    def __init__(self, cache_dir: Optional[Path] = None):
        self.client = IstatClient(cache_dir=cache_dir or CACHE_DIR)
        self.dataflow_id = "93_1227_DF_DCCN_TNA1_6"
        self.key = "A..B1G_B_W2_S1_R_FT.Z.Z.Z.V.N.Z.2025M12"

        self.table_mapping: dict[str, Any] = {
            "indicator": "gdp_per_capita",
            "source": {
                "dataflow_id": self.dataflow_id,
                "key": self.key,
                "columns": {
                    "REF_AREA": "area_code",
                    "FREQ": "frequency",
                    "TIME_PERIOD": "year",
                    "OBS_VALUE": "gdp_per_capita",
                    "DATA_TYPE_AGGR": "measure_code",
                    "EDITION": "edition",
                    "UNIT_MEAS": "unit",
                },
            },
            "project": {
                "entity": "geography",
                "code_field": "area_code",
                "levels": ["region", "macroarea", "national"],
                "notes": "GDP per capita in euro, territorial national accounts.",
            },
        }

    def _save_table_mapping(self, output_path: Path) -> Path:
        mapping_path = output_path.with_suffix(".mapping.json")
        mapping_path.parent.mkdir(parents=True, exist_ok=True)
        with open(mapping_path, "w", encoding="utf-8") as handle:
            json.dump(self.table_mapping, handle, indent=2, ensure_ascii=False)
        return mapping_path

    def download_gdp_per_capita(self, start_year: int = 2001, end_year: int = 2025) -> pd.DataFrame:
        csv_data = self.client.get_data(
            flow_id=self.dataflow_id,
            key=self.key,
            start_year=start_year,
            end_year=end_year,
            format="csv",
        )

        df = pd.read_csv(io.StringIO(csv_data))
        df_clean = df[["REF_AREA", "FREQ", "TIME_PERIOD", "OBS_VALUE", "DATA_TYPE_AGGR", "EDITION", "UNIT_MEAS"]].copy()
        df_clean.columns = ["area_code", "frequency", "year", "gdp_per_capita", "measure_code", "edition", "unit"]

        # Keep the intended GDP-per-capita series definition.
        df_clean = df_clean[
            (df_clean["measure_code"] == "B1G_B_W2_S1_R_FT")
            & (df_clean["edition"] == "2025M12")
        ].copy()

        df_clean["year"] = pd.to_numeric(df_clean["year"], errors="coerce")
        df_clean["gdp_per_capita"] = pd.to_numeric(df_clean["gdp_per_capita"], errors="coerce")
        df_clean = df_clean.dropna(subset=["year", "gdp_per_capita"])

        return df_clean

    def save_gdp_per_capita_csv(
        self,
        output_path: Optional[Path] = None,
        start_year: int = 2001,
        end_year: int = 2025,
    ) -> Path:
        if output_path is None:
            indicator_dir = get_indicator_data_dir("gdp_per_capita")
            output_path = indicator_dir / f"gdp_per_capita_raw_{start_year}_{end_year}.csv"

        print(f"Downloading GDP per capita data ({start_year}-{end_year})...")
        df = self.download_gdp_per_capita(start_year=start_year, end_year=end_year)

        print(f"Saving to {output_path}...")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False, encoding="utf-8")
        mapping_path = self._save_table_mapping(output_path)

        print(f"✓ GDP per capita data saved: {output_path}")
        print(f"✓ Table mapping saved: {mapping_path}")
        print(f"  - {len(df)} records")
        print(f"  - Years: {int(df['year'].min())} to {int(df['year'].max())}")
        print(f"  - Unique areas: {df['area_code'].nunique()}")

        return output_path
