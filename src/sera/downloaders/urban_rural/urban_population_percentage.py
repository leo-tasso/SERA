"""Urban population percentage downloader (World Bank source)."""

import json
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import requests

from sera.config import get_indicator_data_dir


class UrbanPopulationPercentageDownloader:
    """Download urban population percentage for Italy from World Bank."""

    def __init__(self):
        self.country = "IT"
        self.indicator = "SP.URB.TOTL.IN.ZS"
        self.api_url = (
            f"https://api.worldbank.org/v2/country/{self.country}/indicator/{self.indicator}"
        )
        self.table_mapping: dict[str, Any] = {
            "indicator": "urban_population_percentage",
            "source": {
                "provider": "World Bank",
                "url": self.api_url,
                "indicator_code": self.indicator,
                "indicator_name": "Urban population (% of total population)",
                "columns": {
                    "countryiso3code": "area_code",
                    "date": "year",
                    "value": "urban_population_percentage",
                },
            },
            "project": {
                "entity": "geography",
                "code_field": "area_code",
                "levels": ["national"],
                "notes": "Urban population percentage from World Bank.",
            },
        }

    def _save_table_mapping(self, output_path: Path) -> Path:
        mapping_path = output_path.with_suffix(".mapping.json")
        mapping_path.parent.mkdir(parents=True, exist_ok=True)
        with open(mapping_path, "w", encoding="utf-8") as handle:
            json.dump(self.table_mapping, handle, indent=2, ensure_ascii=False)
        return mapping_path

    def download_urban_population_percentage(
        self, start_year: int = 2001, end_year: int = 2025
    ) -> pd.DataFrame:
        response = requests.get(
            self.api_url, params={"format": "json", "per_page": 1000}, timeout=120
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, list) or len(payload) < 2 or not isinstance(payload[1], list):
            raise RuntimeError(
                "Unexpected World Bank response format for urban_population_percentage."
            )
        df = pd.DataFrame(payload[1])
        df_clean = df[["countryiso3code", "date", "value"]].copy()
        df_clean.columns = ["area_code", "year", "urban_population_percentage"]
        df_clean["year"] = pd.to_numeric(df_clean["year"], errors="coerce")
        df_clean["urban_population_percentage"] = pd.to_numeric(
            df_clean["urban_population_percentage"], errors="coerce"
        )
        df_clean = df_clean.dropna(subset=["year", "urban_population_percentage"])
        df_clean = df_clean[(df_clean["year"] >= start_year) & (df_clean["year"] <= end_year)]
        return df_clean.sort_values("year")

    def save_urban_population_percentage_csv(
        self, output_path: Optional[Path] = None, start_year: int = 2001, end_year: int = 2025
    ) -> Path:
        if output_path is None:
            indicator_dir = get_indicator_data_dir("urban_population_percentage")
            output_path = (
                indicator_dir / f"urban_population_percentage_raw_{start_year}_{end_year}.csv"
            )
        print(f"Downloading urban population percentage data ({start_year}-{end_year})...")
        df = self.download_urban_population_percentage(start_year=start_year, end_year=end_year)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False, encoding="utf-8")
        mapping_path = self._save_table_mapping(output_path)
        print(f"✓ Urban population percentage data saved: {output_path}")
        print(f"✓ Table mapping saved: {mapping_path}")
        print(f"  - {len(df)} records")
        if not df.empty:
            print(f"  - Years: {int(df['year'].min())} to {int(df['year'].max())}")
            print(f"  - Unique areas: {df['area_code'].nunique()}")
        return output_path
