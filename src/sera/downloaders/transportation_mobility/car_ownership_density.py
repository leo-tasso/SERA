"""Car ownership density downloader (ISTAT source with population normalization)."""

import io
import json
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from sera.config import get_indicator_data_dir
from sera.istat_client import IstatClient
from sera.downloaders.demographic.population import PopulationDownloader


class CarOwnershipDensityDownloader:
    """Download car ownership density as vehicles per 1,000 residents."""

    def __init__(self):
        self.client = IstatClient()
        self.flow_id = "41_288_DF_DCIS_VEICOLIPRA_1"
        self.population_downloader = PopulationDownloader()
        self.table_mapping: dict[str, Any] = {
            "indicator": "car_ownership_density",
            "source": {
                "provider": "ISTAT",
                "flow_id": self.flow_id,
                "filters": {
                    "DATA_TYPE": "VEHICFLEET",
                    "VEHICLE_TYPE": "1",
                },
                "columns": {
                    "REF_AREA": "area_code",
                    "TIME_PERIOD": "year",
                    "OBS_VALUE": "car_ownership_density",
                },
                "population_source": "22_289_DF_DCIS_POPRES1_1",
            },
            "project": {
                "entity": "geography",
                "code_field": "area_code",
                "levels": ["national"],
                "notes": "Vehicles per 1,000 residents computed from ISTAT vehicle registry and population data.",
            },
        }

    def _save_table_mapping(self, output_path: Path) -> Path:
        mapping_path = output_path.with_suffix(".mapping.json")
        mapping_path.parent.mkdir(parents=True, exist_ok=True)
        with open(mapping_path, "w", encoding="utf-8") as handle:
            json.dump(self.table_mapping, handle, indent=2, ensure_ascii=False)
        return mapping_path

    def download_car_ownership_density(self, start_year: int = 2001, end_year: int = 2025) -> pd.DataFrame:
        csv_data = self.client.get_data(flow_id=self.flow_id, key="", start_year=start_year, end_year=end_year, format="csv")
        vehicles = pd.read_csv(io.StringIO(csv_data), low_memory=False)
        for column, expected in self.table_mapping["source"]["filters"].items():
            if column in vehicles.columns:
                vehicles = vehicles[vehicles[column].astype(str) == expected]
        if vehicles.empty:
            return pd.DataFrame(columns=["area_code", "year", "car_ownership_density"])
        vehicles = vehicles[["REF_AREA", "TIME_PERIOD", "OBS_VALUE"]].copy()
        vehicles.columns = ["area_code", "year", "vehicles"]
        vehicles["year"] = pd.to_numeric(vehicles["year"], errors="coerce")
        vehicles["vehicles"] = pd.to_numeric(vehicles["vehicles"], errors="coerce")
        vehicles = vehicles.dropna(subset=["year", "vehicles"])
        vehicles = vehicles[(vehicles["year"] >= start_year) & (vehicles["year"] <= end_year)]

        population = self.population_downloader.download_population(start_year=start_year, end_year=end_year)
        if population.empty or "population" not in population.columns:
            return pd.DataFrame(columns=["area_code", "year", "car_ownership_density"])
        population = population[["area_code", "year", "population"]].copy()
        merged = vehicles.merge(population, on=["area_code", "year"], how="inner")
        merged["car_ownership_density"] = (merged["vehicles"] / merged["population"]) * 1000.0
        return merged[["area_code", "year", "car_ownership_density"]].sort_values("year")

    def save_car_ownership_density_csv(self, output_path: Optional[Path] = None, start_year: int = 2001, end_year: int = 2025) -> Path:
        if output_path is None:
            indicator_dir = get_indicator_data_dir("car_ownership_density")
            output_path = indicator_dir / f"car_ownership_density_raw_{start_year}_{end_year}.csv"
        print(f"Downloading car ownership density data ({start_year}-{end_year})...")
        df = self.download_car_ownership_density(start_year=start_year, end_year=end_year)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False, encoding="utf-8")
        mapping_path = self._save_table_mapping(output_path)
        print(f"✓ Car ownership density data saved: {output_path}")
        print(f"✓ Table mapping saved: {mapping_path}")
        print(f"  - {len(df)} records")
        if not df.empty:
            print(f"  - Years: {int(df['year'].min())} to {int(df['year'].max())}")
            print(f"  - Unique areas: {df['area_code'].nunique()}")
        return output_path
