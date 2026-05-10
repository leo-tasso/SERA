"""Housing affordability ratio indicator downloader (ISTAT source + existing income series)."""

import io
import json
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from sera.config import get_indicator_data_dir
from sera.istat_client import IstatClient
from sera.downloaders.demographic.income import IncomeDownloader


class HousingAffordabilityRatioDownloader:
    """Compute housing affordability as house price index divided by household income."""

    def __init__(self):
        self.house_prices_flow_id = "143_497"
        self.income_downloader = IncomeDownloader()
        self.client = IstatClient()
        self.table_mapping: dict[str, Any] = {
            "indicator": "housing_affordability_ratio",
            "source": {
                "provider": "ISTAT",
                "house_price_flow_id": self.house_prices_flow_id,
                "house_price_filters": {
                    "DATA_TYPE": "19",
                    "MEASURE": "4",
                    "PURCHASES_DWELLINGS": "ALL",
                },
                "income_source": {
                    "dataflow_id": self.income_downloader.dataflow_id,
                    "key": self.income_downloader.key,
                },
                "columns": {
                    "area_code": "area_code",
                    "year": "year",
                    "house_price_index": "house_price_index",
                    "income": "income",
                    "housing_affordability_ratio": "housing_affordability_ratio",
                },
            },
            "project": {
                "entity": "policy",
                "code_field": "area_code",
                "levels": ["national"],
                "notes": "Proxy affordability ratio = house price index / household disposable income.",
            },
        }

    def _save_table_mapping(self, output_path: Path) -> Path:
        mapping_path = output_path.with_suffix(".mapping.json")
        mapping_path.parent.mkdir(parents=True, exist_ok=True)
        with open(mapping_path, "w", encoding="utf-8") as handle:
            json.dump(self.table_mapping, handle, indent=2, ensure_ascii=False)
        return mapping_path

    def _download_house_prices(self, start_year: int, end_year: int) -> pd.DataFrame:
        csv_data = self.client.get_data(flow_id=self.house_prices_flow_id, key="", start_year=start_year, end_year=end_year, format="csv")
        df = pd.read_csv(io.StringIO(csv_data), low_memory=False)
        filters = self.table_mapping["source"]["house_price_filters"]
        for column, expected in filters.items():
            if column in df.columns:
                df = df[df[column].astype(str) == expected]
        if df.empty:
            return pd.DataFrame(columns=["area_code", "year", "house_price_index"])
        df = df[["REF_AREA", "TIME_PERIOD", "OBS_VALUE"]].copy()
        df.columns = ["area_code", "year", "house_price_index"]
        df["year"] = pd.to_numeric(df["year"], errors="coerce")
        df["house_price_index"] = pd.to_numeric(df["house_price_index"], errors="coerce")
        df = df.dropna(subset=["year", "house_price_index"])
        return df

    def download_housing_affordability_ratio(self, start_year: int = 2001, end_year: int = 2025) -> pd.DataFrame:
        house_prices = self._download_house_prices(start_year=start_year, end_year=end_year)
        income = self.income_downloader.download_income(start_year=start_year, end_year=end_year)
        if house_prices.empty or income.empty:
            return pd.DataFrame(columns=["area_code", "year", "house_price_index", "income", "housing_affordability_ratio"])
        # National-only proxy series.
        house_prices = house_prices[house_prices["area_code"].astype(str) == "IT"]
        income = income[income["area_code"].astype(str) == "IT"]
        merged = house_prices.merge(income[["area_code", "year", "income"]], on=["area_code", "year"], how="inner")
        merged["housing_affordability_ratio"] = merged["house_price_index"] / merged["income"]
        merged = merged.sort_values("year")
        return merged

    def save_housing_affordability_ratio_csv(self, output_path: Optional[Path] = None, start_year: int = 2001, end_year: int = 2025) -> Path:
        if output_path is None:
            indicator_dir = get_indicator_data_dir("housing_affordability_ratio")
            output_path = indicator_dir / f"housing_affordability_ratio_raw_{start_year}_{end_year}.csv"
        print(f"Downloading housing affordability ratio data ({start_year}-{end_year})...")
        df = self.download_housing_affordability_ratio(start_year=start_year, end_year=end_year)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False, encoding="utf-8")
        mapping_path = self._save_table_mapping(output_path)
        print(f"✓ Housing affordability ratio data saved: {output_path}")
        print(f"✓ Table mapping saved: {mapping_path}")
        print(f"  - {len(df)} records")
        if not df.empty:
            print(f"  - Years: {int(df['year'].min())} to {int(df['year'].max())}")
            print(f"  - Unique areas: {df['area_code'].nunique()}")
        return output_path
