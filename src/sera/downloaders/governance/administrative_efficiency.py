"""Administrative efficiency indicator downloader (World Bank WGI - Government Effectiveness)."""

import json
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import requests

from sera.config import CACHE_DIR, get_indicator_data_dir


class AdministrativeEfficiencyDownloader:
    """Download administrative efficiency from World Bank WGI."""

    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or CACHE_DIR
        self.base_url = "https://api.worldbank.org/v2/country/ITA/indicator/GE.EST"
        self.country_code = "ITA"
        self.indicator_code = "GE.EST"

        self.table_mapping: dict[str, Any] = {
            "indicator": "administrative_efficiency",
            "source": {
                "provider": "World Bank",
                "endpoint": self.base_url,
                "indicator_code": self.indicator_code,
                "columns": {
                    "date": "year",
                    "value": "administrative_efficiency",
                },
            },
            "project": {
                "entity": "country",
                "code_field": "country_code",
                "levels": ["national"],
                "notes": "Government Effectiveness (GE.EST) from World Bank WGI. Captures perceptions of quality of public and civil services.",
            },
        }

    def _save_table_mapping(self, output_path: Path) -> Path:
        mapping_path = output_path.with_suffix(".mapping.json")
        mapping_path.parent.mkdir(parents=True, exist_ok=True)
        with open(mapping_path, "w", encoding="utf-8") as handle:
            json.dump(self.table_mapping, handle, indent=2, ensure_ascii=False)
        return mapping_path

    def download_administrative_efficiency(
        self, start_year: int = 2001, end_year: int = 2025
    ) -> pd.DataFrame:
        url = f"{self.base_url}?format=json&per_page=500"
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()

            if len(data) < 2 or not data[1]:
                return pd.DataFrame(columns=["country_code", "year", "administrative_efficiency"])

            records = []
            for item in data[1]:
                if item.get("value") is not None:
                    year = int(item["date"])
                    if start_year <= year <= end_year:
                        records.append(
                            {
                                "country_code": "IT",
                                "year": year,
                                "administrative_efficiency": float(item["value"]),
                            }
                        )

            if not records:
                return pd.DataFrame(columns=["country_code", "year", "administrative_efficiency"])

            df = pd.DataFrame(records)
            return df.sort_values(["year"])

        except Exception as e:
            print(f"Error fetching administrative efficiency data: {e}")
            return pd.DataFrame(columns=["country_code", "year", "administrative_efficiency"])

    def save_administrative_efficiency_csv(
        self,
        output_path: Optional[Path] = None,
        start_year: int = 2001,
        end_year: int = 2025,
    ) -> Path:
        if output_path is None:
            indicator_dir = get_indicator_data_dir("administrative_efficiency")
            output_path = (
                indicator_dir / f"administrative_efficiency_raw_{start_year}_{end_year}.csv"
            )

        print(f"Downloading administrative efficiency data ({start_year}-{end_year})...")
        df = self.download_administrative_efficiency(start_year=start_year, end_year=end_year)

        print(f"Saving to {output_path}...")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False, encoding="utf-8")
        mapping_path = self._save_table_mapping(output_path)

        print(f"✓ Administrative efficiency data saved: {output_path}")
        print(f"✓ Table mapping saved: {mapping_path}")
        print(f"  - {len(df)} records")
        if not df.empty:
            print(f"  - Years: {int(df['year'].min())} to {int(df['year'].max())}")

        return output_path
