"""Local government debt indicator downloader (World Bank proxy source)."""

import json
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import requests

from sera.config import get_indicator_data_dir


class LocalGovernmentDebtDownloader:
    """Download local government debt proxy for Italy from World Bank."""

    def __init__(self):
        self.country = "IT"
        self.indicator = "GC.DOD.TOTL.GD.ZS"
        self.api_url = (
            f"https://api.worldbank.org/v2/country/{self.country}/indicator/{self.indicator}"
        )
        self.table_mapping: dict[str, Any] = {
            "indicator": "local_government_debt",
            "source": {
                "provider": "World Bank",
                "url": self.api_url,
                "indicator_code": self.indicator,
                "indicator_name": "Central government debt, total (% of GDP)",
                "proxy_note": "Used as a national proxy because a stable ISTAT local-government debt series was not found.",
                "columns": {
                    "countryiso3code": "area_code",
                    "date": "year",
                    "value": "local_government_debt_pct_gdp",
                },
            },
            "project": {
                "entity": "policy",
                "code_field": "area_code",
                "levels": ["national"],
                "notes": "Proxy for local government debt: central government debt (% of GDP).",
            },
        }

    def _save_table_mapping(self, output_path: Path) -> Path:
        mapping_path = output_path.with_suffix(".mapping.json")
        mapping_path.parent.mkdir(parents=True, exist_ok=True)
        with open(mapping_path, "w", encoding="utf-8") as handle:
            json.dump(self.table_mapping, handle, indent=2, ensure_ascii=False)
        return mapping_path

    def download_local_government_debt(
        self, start_year: int = 2001, end_year: int = 2025
    ) -> pd.DataFrame:
        response = requests.get(
            self.api_url, params={"format": "json", "per_page": 1000}, timeout=120
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, list) or len(payload) < 2 or not isinstance(payload[1], list):
            raise RuntimeError("Unexpected World Bank response format for local government debt.")

        df = pd.DataFrame(payload[1])
        df_clean = df[["countryiso3code", "date", "value"]].copy()
        df_clean.columns = ["area_code", "year", "local_government_debt_pct_gdp"]
        df_clean["year"] = pd.to_numeric(df_clean["year"], errors="coerce")
        df_clean["local_government_debt_pct_gdp"] = pd.to_numeric(
            df_clean["local_government_debt_pct_gdp"], errors="coerce"
        )
        df_clean = df_clean.dropna(subset=["year", "local_government_debt_pct_gdp"])
        df_clean = df_clean[(df_clean["year"] >= start_year) & (df_clean["year"] <= end_year)]
        df_clean = df_clean.sort_values("year")
        return df_clean

    def save_local_government_debt_csv(
        self, output_path: Optional[Path] = None, start_year: int = 2001, end_year: int = 2025
    ) -> Path:
        if output_path is None:
            indicator_dir = get_indicator_data_dir("local_government_debt")
            output_path = indicator_dir / f"local_government_debt_raw_{start_year}_{end_year}.csv"
        print(f"Downloading local government debt data ({start_year}-{end_year})...")
        df = self.download_local_government_debt(start_year=start_year, end_year=end_year)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False, encoding="utf-8")
        mapping_path = self._save_table_mapping(output_path)
        print(f"✓ Local government debt data saved: {output_path}")
        print(f"✓ Table mapping saved: {mapping_path}")
        print(f"  - {len(df)} records")
        if not df.empty:
            print(f"  - Years: {int(df['year'].min())} to {int(df['year'].max())}")
            print(f"  - Unique areas: {df['area_code'].nunique()}")
        else:
            print(
                "  - No records in requested range (available World Bank coverage appears to be outside 2001-2025)."
            )
        return output_path
