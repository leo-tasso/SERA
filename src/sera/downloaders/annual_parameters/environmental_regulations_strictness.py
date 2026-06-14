"""environmental regulations strictness annual input parameter downloader (World Bank proxy source)."""

import json
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import requests

from sera.config import get_indicator_data_dir


class EnvironmentalRegulationsStrictnessDownloader:
    """Download annual environmental_regulations_strictness proxy for Italy from World Bank."""

    def __init__(self):
        self.country = "IT"
        self.indicator = "EN.ATM.PM25.MC.M3"
        self.api_url = (
            f"https://api.worldbank.org/v2/country/{self.country}/indicator/{self.indicator}"
        )

        self.table_mapping: dict[str, Any] = {
            "parameter": "environmental_regulations_strictness",
            "source": {
                "provider": "World Bank",
                "url": self.api_url,
                "indicator_code": self.indicator,
                "indicator_name": "PM2.5 air pollution, mean annual exposure (micrograms per cubic meter)",
                "columns": {
                    "countryiso3code": "area_code",
                    "date": "year",
                    "value": "environmental_regulations_strictness",
                },
            },
            "project": {
                "entity": "policy",
                "code_field": "area_code",
                "levels": ["national"],
                "notes": "Annual proxy for environmental regulation outcomes/strictness.",
            },
        }

    def _save_table_mapping(self, output_path: Path) -> Path:
        mapping_path = output_path.with_suffix(".mapping.json")
        mapping_path.parent.mkdir(parents=True, exist_ok=True)
        with open(mapping_path, "w", encoding="utf-8") as handle:
            json.dump(self.table_mapping, handle, indent=2, ensure_ascii=False)
        return mapping_path

    def download_environmental_regulations_strictness(
        self, start_year: int = 2001, end_year: int = 2025
    ) -> pd.DataFrame:
        response = requests.get(
            self.api_url,
            params={"format": "json", "per_page": 1000},
            timeout=120,
        )
        response.raise_for_status()
        payload = response.json()

        if not isinstance(payload, list) or len(payload) < 2 or not isinstance(payload[1], list):
            raise RuntimeError(
                "Unexpected World Bank response format for environmental_regulations_strictness."
            )

        df = pd.DataFrame(payload[1])
        df_clean = df[["countryiso3code", "date", "value"]].copy()
        df_clean.columns = ["area_code", "year", "environmental_regulations_strictness"]

        df_clean["year"] = pd.to_numeric(df_clean["year"], errors="coerce")
        df_clean["environmental_regulations_strictness"] = pd.to_numeric(
            df_clean["environmental_regulations_strictness"], errors="coerce"
        )
        df_clean = df_clean.dropna(subset=["year", "environmental_regulations_strictness"])
        df_clean = df_clean[(df_clean["year"] >= start_year) & (df_clean["year"] <= end_year)]
        df_clean = df_clean.sort_values("year")

        return df_clean

    def save_environmental_regulations_strictness_csv(
        self,
        output_path: Optional[Path] = None,
        start_year: int = 2001,
        end_year: int = 2025,
    ) -> Path:
        if output_path is None:
            indicator_dir = get_indicator_data_dir("environmental_regulations_strictness")
            output_path = (
                indicator_dir
                / f"environmental_regulations_strictness_raw_{start_year}_{end_year}.csv"
            )

        print(
            f"Downloading environmental regulations strictness proxy data ({start_year}-{end_year})..."
        )
        df = self.download_environmental_regulations_strictness(
            start_year=start_year, end_year=end_year
        )

        print(f"Saving to {output_path}...")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False, encoding="utf-8")
        mapping_path = self._save_table_mapping(output_path)

        print(f"✓ environmental regulations strictness proxy data saved: {output_path}")
        print(f"✓ Table mapping saved: {mapping_path}")
        print(f"  - {len(df)} records")
        print(f"  - Years: {int(df['year'].min())} to {int(df['year'].max())}")
        print(f"  - Unique areas: {df['area_code'].nunique()}")

        return output_path
