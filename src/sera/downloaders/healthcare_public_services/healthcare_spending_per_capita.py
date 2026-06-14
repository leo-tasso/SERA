"""Healthcare spending per capita downloader (World Bank source)."""

import json
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import requests

from sera.config import get_indicator_data_dir


class HealthcareSpendingPerCapitaDownloader:
    """Download healthcare expenditure per capita for Italy from World Bank."""

    def __init__(self):
        self.country = "IT"
        self.indicator = "SH.XPD.CHEX.PC.CD"
        self.api_url = (
            f"https://api.worldbank.org/v2/country/{self.country}/indicator/{self.indicator}"
        )
        self.table_mapping: dict[str, Any] = {
            "indicator": "healthcare_spending_per_capita",
            "source": {
                "provider": "World Bank",
                "url": self.api_url,
                "indicator_code": self.indicator,
                "indicator_name": "Current health expenditure per capita (current US$)",
                "columns": {
                    "countryiso3code": "area_code",
                    "date": "year",
                    "value": "healthcare_spending_per_capita",
                },
            },
            "project": {
                "entity": "policy",
                "code_field": "area_code",
                "levels": ["national"],
                "notes": "National proxy for healthcare spending per capita in USD.",
            },
        }

    def _save_table_mapping(self, output_path: Path) -> Path:
        mapping_path = output_path.with_suffix(".mapping.json")
        mapping_path.parent.mkdir(parents=True, exist_ok=True)
        with open(mapping_path, "w", encoding="utf-8") as handle:
            json.dump(self.table_mapping, handle, indent=2, ensure_ascii=False)
        return mapping_path

    def download_healthcare_spending_per_capita(
        self, start_year: int = 2001, end_year: int = 2025
    ) -> pd.DataFrame:
        response = requests.get(
            self.api_url, params={"format": "json", "per_page": 1000}, timeout=120
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, list) or len(payload) < 2 or not isinstance(payload[1], list):
            raise RuntimeError(
                "Unexpected World Bank response format for healthcare_spending_per_capita."
            )
        df = pd.DataFrame(payload[1])
        df_clean = df[["countryiso3code", "date", "value"]].copy()
        df_clean.columns = ["area_code", "year", "healthcare_spending_per_capita"]
        df_clean["year"] = pd.to_numeric(df_clean["year"], errors="coerce")
        df_clean["healthcare_spending_per_capita"] = pd.to_numeric(
            df_clean["healthcare_spending_per_capita"], errors="coerce"
        )
        df_clean = df_clean.dropna(subset=["year", "healthcare_spending_per_capita"])
        df_clean = df_clean[(df_clean["year"] >= start_year) & (df_clean["year"] <= end_year)]
        return df_clean.sort_values("year")

    def save_healthcare_spending_per_capita_csv(
        self, output_path: Optional[Path] = None, start_year: int = 2001, end_year: int = 2025
    ) -> Path:
        if output_path is None:
            indicator_dir = get_indicator_data_dir("healthcare_spending_per_capita")
            output_path = (
                indicator_dir / f"healthcare_spending_per_capita_raw_{start_year}_{end_year}.csv"
            )
        print(f"Downloading healthcare spending per capita data ({start_year}-{end_year})...")
        df = self.download_healthcare_spending_per_capita(start_year=start_year, end_year=end_year)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False, encoding="utf-8")
        mapping_path = self._save_table_mapping(output_path)
        print(f"✓ Healthcare spending per capita data saved: {output_path}")
        print(f"✓ Table mapping saved: {mapping_path}")
        print(f"  - {len(df)} records")
        if not df.empty:
            print(f"  - Years: {int(df['year'].min())} to {int(df['year'].max())}")
            print(f"  - Unique areas: {df['area_code'].nunique()}")
        return output_path
