"""Broadband speed/reliability downloader (World Bank source)."""

import json
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import requests

from sera.config import get_indicator_data_dir


class BroadbandSpeedReliabilityDownloader:
    """Download broadband speed/reliability for Italy from World Bank."""

    def __init__(self):
        self.country = "IT"
        self.indicator = "IT.BRD.ASWD.P2"
        self.api_url = f"https://api.worldbank.org/v2/country/{self.country}/indicator/{self.indicator}"
        self.table_mapping: dict[str, Any] = {
            "indicator": "broadband_speed_reliability",
            "source": {
                "provider": "World Bank",
                "url": self.api_url,
                "indicator_code": self.indicator,
                "indicator_name": "Fixed broadband subscriptions (per 100 people)",
                "columns": {
                    "countryiso3code": "area_code",
                    "date": "year",
                    "value": "broadband_speed_reliability",
                },
            },
            "project": {
                "entity": "policy",
                "code_field": "area_code",
                "levels": ["national"],
                "notes": "Broadband speed/reliability proxy: fixed broadband subscriptions per 100 people.",
            },
        }

    def _save_table_mapping(self, output_path: Path) -> Path:
        mapping_path = output_path.with_suffix(".mapping.json")
        mapping_path.parent.mkdir(parents=True, exist_ok=True)
        with open(mapping_path, "w", encoding="utf-8") as handle:
            json.dump(self.table_mapping, handle, indent=2, ensure_ascii=False)
        return mapping_path

    def download_broadband_speed_reliability(self, start_year: int = 2001, end_year: int = 2025) -> pd.DataFrame:
        response = requests.get(self.api_url, params={"format": "json", "per_page": 1000}, timeout=120)
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, list) or len(payload) < 2 or not isinstance(payload[1], list):
            raise RuntimeError("Unexpected World Bank response format for broadband_speed_reliability.")
        df = pd.DataFrame(payload[1])
        df_clean = df[["countryiso3code", "date", "value"]].copy()
        df_clean.columns = ["area_code", "year", "broadband_speed_reliability"]
        df_clean["year"] = pd.to_numeric(df_clean["year"], errors="coerce")
        df_clean["broadband_speed_reliability"] = pd.to_numeric(df_clean["broadband_speed_reliability"], errors="coerce")
        df_clean = df_clean.dropna(subset=["year", "broadband_speed_reliability"])
        df_clean = df_clean[(df_clean["year"] >= start_year) & (df_clean["year"] <= end_year)]
        return df_clean.sort_values("year")

    def save_broadband_speed_reliability_csv(self, output_path: Optional[Path] = None, start_year: int = 2001, end_year: int = 2025) -> Path:
        if output_path is None:
            indicator_dir = get_indicator_data_dir("broadband_speed_reliability")
            output_path = indicator_dir / f"broadband_speed_reliability_raw_{start_year}_{end_year}.csv"
        print(f"Downloading broadband speed/reliability data ({start_year}-{end_year})...")
        df = self.download_broadband_speed_reliability(start_year=start_year, end_year=end_year)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False, encoding="utf-8")
        mapping_path = self._save_table_mapping(output_path)
        print(f"✓ Broadband speed/reliability data saved: {output_path}")
        print(f"✓ Table mapping saved: {mapping_path}")
        print(f"  - {len(df)} records")
        if not df.empty:
            print(f"  - Years: {int(df['year'].min())} to {int(df['year'].max())}")
            print(f"  - Unique areas: {df['area_code'].nunique()}")
        return output_path
