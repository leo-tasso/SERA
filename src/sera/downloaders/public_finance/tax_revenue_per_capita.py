"""Tax revenue per capita indicator downloader (World Bank proxy source)."""

import json
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import requests

from sera.config import get_indicator_data_dir


class TaxRevenuePerCapitaDownloader:
    """Download tax revenue per capita proxy for Italy from World Bank."""

    def __init__(self):
        self.country = "IT"
        self.tax_indicator = "GC.TAX.TOTL.GD.ZS"
        self.gdp_per_capita_indicator = "NY.GDP.PCAP.CD"
        self.tax_api_url = f"https://api.worldbank.org/v2/country/{self.country}/indicator/{self.tax_indicator}"
        self.gdp_api_url = f"https://api.worldbank.org/v2/country/{self.country}/indicator/{self.gdp_per_capita_indicator}"
        self.table_mapping: dict[str, Any] = {
            "indicator": "tax_revenue_per_capita",
            "source": {
                "provider": "World Bank",
                "tax_url": self.tax_api_url,
                "gdp_per_capita_url": self.gdp_api_url,
                "tax_indicator_code": self.tax_indicator,
                "gdp_per_capita_indicator_code": self.gdp_per_capita_indicator,
                "columns": {
                    "countryiso3code": "area_code",
                    "date": "year",
                    "value": "tax_revenue_per_capita_usd",
                },
            },
            "project": {
                "entity": "policy",
                "code_field": "area_code",
                "levels": ["national"],
                "notes": "Tax revenue per capita proxy derived as tax revenue (% of GDP) * GDP per capita / 100.",
            },
        }

    def _save_table_mapping(self, output_path: Path) -> Path:
        mapping_path = output_path.with_suffix(".mapping.json")
        mapping_path.parent.mkdir(parents=True, exist_ok=True)
        with open(mapping_path, "w", encoding="utf-8") as handle:
            json.dump(self.table_mapping, handle, indent=2, ensure_ascii=False)
        return mapping_path

    def _fetch_indicator(self, api_url: str, value_column: str) -> pd.DataFrame:
        response = requests.get(api_url, params={"format": "json", "per_page": 1000}, timeout=120)
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, list) or len(payload) < 2 or not isinstance(payload[1], list):
            raise RuntimeError(f"Unexpected World Bank response format for {value_column}.")
        df = pd.DataFrame(payload[1])
        df_clean = df[["countryiso3code", "date", "value"]].copy()
        df_clean.columns = ["area_code", "year", value_column]
        df_clean["year"] = pd.to_numeric(df_clean["year"], errors="coerce")
        df_clean[value_column] = pd.to_numeric(df_clean[value_column], errors="coerce")
        df_clean = df_clean.dropna(subset=["year", value_column])
        return df_clean

    def download_tax_revenue_per_capita(self, start_year: int = 2001, end_year: int = 2025) -> pd.DataFrame:
        tax_df = self._fetch_indicator(self.tax_api_url, "tax_revenue_pct_gdp")
        gdp_df = self._fetch_indicator(self.gdp_api_url, "gdp_per_capita_usd")
        merged = tax_df.merge(gdp_df, on=["area_code", "year"], how="inner")
        merged = merged[(merged["year"] >= start_year) & (merged["year"] <= end_year)]
        merged["tax_revenue_per_capita_usd"] = merged["tax_revenue_pct_gdp"] * merged["gdp_per_capita_usd"] / 100.0
        merged = merged[["area_code", "year", "tax_revenue_per_capita_usd"]].sort_values("year")
        return merged

    def save_tax_revenue_per_capita_csv(self, output_path: Optional[Path] = None, start_year: int = 2001, end_year: int = 2025) -> Path:
        if output_path is None:
            indicator_dir = get_indicator_data_dir("tax_revenue_per_capita")
            output_path = indicator_dir / f"tax_revenue_per_capita_raw_{start_year}_{end_year}.csv"
        print(f"Downloading tax revenue per capita data ({start_year}-{end_year})...")
        df = self.download_tax_revenue_per_capita(start_year=start_year, end_year=end_year)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False, encoding="utf-8")
        mapping_path = self._save_table_mapping(output_path)
        print(f"✓ Tax revenue per capita data saved: {output_path}")
        print(f"✓ Table mapping saved: {mapping_path}")
        print(f"  - {len(df)} records")
        print(f"  - Years: {int(df['year'].min())} to {int(df['year'].max())}")
        print(f"  - Unique areas: {df['area_code'].nunique()}")
        return output_path
