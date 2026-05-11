"""Data loading and preprocessing for the twin simulator."""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import logging

from sera.twin.province_mapping import (
    NUTS_TO_SIGLA,
    PROVINCE_SIGLAS_110,
    PROVINCE_TO_REGION,
    map_area_code_to_sigla,
)

logger = logging.getLogger(__name__)


class DataLoader:
    """Load and preprocess provincial and national indicator data."""

    def __init__(self, data_dir: Path):
        """Initialize data loader.
        
        Args:
            data_dir: Path to the data directory
        """
        self.data_dir = Path(data_dir)

    def load_indicator(
        self, indicator_name: str, category: str
    ) -> pd.DataFrame:
        """Load indicator data from CSV.
        
        Args:
            indicator_name: Name of the indicator (e.g., 'population')
            category: Category of the indicator (e.g., 'demographic')
            
        Returns:
            DataFrame with columns: area_code, year, value
        """
        pattern = f"{indicator_name}_raw_*.csv"
        file_path = self.data_dir / category / indicator_name
        
        matching_files = list(file_path.glob(pattern))
        if not matching_files:
            logger.warning(f"No data found for {indicator_name} in {category}")
            return pd.DataFrame()
        
        # Preserve area_code as text to avoid losing leading zeros in ISTAT-style codes.
        df = pd.read_csv(matching_files[0], dtype={"area_code": str})
        
        # Extract area_code, year, and value columns
        if "population" in df.columns:
            df = df[["area_code", "year", "population"]].rename(
                columns={"population": "value"}
            )
        elif "value" in df.columns:
            df = df[["area_code", "year", "value"]]
        else:
            # Try to infer the value column
            numeric_cols = [
                col
                for col in df.select_dtypes(include=[np.number]).columns
                if col.lower() != "year"
            ]
            if len(numeric_cols) > 0:
                value_col = numeric_cols[0]
                df = df[["area_code", "year", value_col]].rename(
                    columns={value_col: "value"}
                )
        
        return df.dropna(subset=["value"])

    def disaggregate_national_to_provincial(
        self,
        df: pd.DataFrame,
        year: Optional[int] = None,
        population_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Disaggregate national-level data to provinces using population weights.
        
        Args:
            df: DataFrame with national data (area_code='IT')
            year: Year to get population weights for
            population_df: Optional pre-loaded population data
            
        Returns:
            DataFrame with values disaggregated to all provinces
        """
        if population_df is None:
            population_df = self.load_indicator("population", "demographic")

        if year is None:
            if "year" not in df.columns:
                return df
            yearly_frames = [
                self.disaggregate_national_to_provincial(
                    df[df["year"] == current_year].copy(),
                    year=int(current_year),
                    population_df=population_df,
                )
                for current_year in sorted(df["year"].dropna().unique())
            ]
            if not yearly_frames:
                return df
            return pd.concat(yearly_frames, ignore_index=True).sort_values(["area_code", "year"])

        area_codes = df["area_code"].astype(str)
        national_mask = area_codes.str.startswith("IT") & (area_codes.str.len() <= 3)
        national_rows = df[national_mask]
        if national_rows.empty:
            return df

        # Get population weights for the year
        pop_year = population_df[population_df["year"] == year]
        if pop_year.empty:
            pop_year = population_df[population_df["year"] == population_df["year"].max()]

        province_pops: Dict[str, float] = {}
        for _, row in pop_year.iterrows():
            province_sigla = map_area_code_to_sigla(str(row["area_code"]))
            if province_sigla is None:
                continue
            province_pops[province_sigla] = province_pops.get(province_sigla, 0.0) + float(row["value"])

        if not province_pops:
            return df

        total_pop = sum(province_pops.values())

        new_rows = []
        for _, national_row in national_rows.iterrows():
            national_value = national_row["value"]
            for province_code, pop in province_pops.items():
                weight = pop / total_pop
                new_rows.append(
                    {
                        "area_code": province_code,
                        "year": national_row["year"],
                        "value": national_value * weight,
                    }
                )

        provincial_only = df[~national_mask].copy()
        result = pd.concat([provincial_only, pd.DataFrame(new_rows)], ignore_index=True)

        return result.sort_values(["area_code", "year"])

    def disaggregate_regional_to_provincial(
        self,
        df: pd.DataFrame,
        year: Optional[int] = None,
        population_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """Disaggregate regional/macro area codes to provinces using population weights."""
        if population_df is None:
            population_df = self.load_indicator("population", "demographic")

        if year is None:
            if "year" not in df.columns:
                return df
            yearly_frames = [
                self.disaggregate_regional_to_provincial(
                    df[df["year"] == current_year].copy(),
                    year=int(current_year),
                    population_df=population_df,
                )
                for current_year in sorted(df["year"].dropna().unique())
            ]
            if not yearly_frames:
                return df
            return pd.concat(yearly_frames, ignore_index=True).sort_values(["area_code", "year"])

        pop_year = population_df[population_df["year"] == year]
        if pop_year.empty:
            pop_year = population_df[population_df["year"] == population_df["year"].max()]

        province_pops: Dict[str, float] = {}
        for _, row in pop_year.iterrows():
            province_sigla = map_area_code_to_sigla(str(row["area_code"]))
            if province_sigla is None:
                continue
            province_pops[province_sigla] = province_pops.get(province_sigla, 0.0) + float(row["value"])

        if not province_pops:
            return df

        token_to_provinces: Dict[str, set[str]] = {}
        for nuts_code, province_sigla in NUTS_TO_SIGLA.items():
            if len(nuts_code) != 5:
                continue
            token_to_provinces.setdefault(nuts_code[:3], set()).add(province_sigla)
            token_to_provinces.setdefault(nuts_code[:4], set()).add(province_sigla)

        new_rows = []
        non_regional_rows = []

        for _, row in df.iterrows():
            raw_code = str(row["area_code"]).strip().upper()

            # Keep rows already at province resolution.
            if map_area_code_to_sigla(raw_code) is not None:
                non_regional_rows.append({
                    "area_code": raw_code,
                    "year": row["year"],
                    "value": row["value"],
                })
                continue

            target_provinces = sorted(token_to_provinces.get(raw_code, []))
            if not target_provinces:
                continue

            total_pop = sum(province_pops.get(p, 0.0) for p in target_provinces)
            if total_pop <= 0:
                continue

            for province_sigla in target_provinces:
                pop = province_pops.get(province_sigla, 0.0)
                if pop <= 0:
                    continue
                new_rows.append(
                    {
                        "area_code": province_sigla,
                        "year": row["year"],
                        "value": row["value"] * (pop / total_pop),
                    }
                )

        result_frames = []
        if non_regional_rows:
            result_frames.append(pd.DataFrame(non_regional_rows))
        if new_rows:
            result_frames.append(pd.DataFrame(new_rows))

        if not result_frames:
            return df

        return pd.concat(result_frames, ignore_index=True).sort_values(["area_code", "year"])

    def prepare_training_data(
        self,
        indicators: Dict[str, Tuple[str, str]],  # {indicator_name: (category, direction)}
        parameters: Dict[str, str],  # {param_name: category}
        min_year: int = 2001,
        max_year: int = 2025,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Prepare training data for indicators and parameters.
        
        Args:
            indicators: Dict of indicators to load {name: (category, direction)}
            parameters: Dict of parameters to load {name: category}
            min_year: Minimum year to include
            max_year: Maximum year to include
            
        Returns:
            Tuple of (indicators_df, parameters_df)
        """
        indicators_dfs = {}
        for ind_name, (category, _) in indicators.items():
            df = self.load_indicator(ind_name, category)
            if not df.empty:
                # Normalize all inputs to canonical 110 province sigle.
                df = self.disaggregate_national_to_provincial(df)
                df = self.disaggregate_regional_to_provincial(df)
                df = self.standardize_to_province_level(df, interpolate_missing=True)
                df = df[(df["year"] >= min_year) & (df["year"] <= max_year)]
                indicators_dfs[ind_name] = df

        parameters_dfs = {}
        for param_name, category in parameters.items():
            df = self.load_indicator(param_name, category)
            if not df.empty:
                df = self.disaggregate_national_to_provincial(df)
                df = self.disaggregate_regional_to_provincial(df)
                df = self.standardize_to_province_level(df, interpolate_missing=True)
                df = df[(df["year"] >= min_year) & (df["year"] <= max_year)]
                parameters_dfs[param_name] = df
        
        # Merge all indicators and parameters on area_code and year
        if not indicators_dfs:
            return pd.DataFrame(), pd.DataFrame()
        
        # Start with first indicator
        combined_ind = None
        for ind_name, df in indicators_dfs.items():
            df_pivot = df.pivot_table(
                index=["area_code", "year"], values="value", aggfunc="mean"
            ).reset_index()
            df_pivot.columns = ["area_code", "year", ind_name]
            
            if combined_ind is None:
                combined_ind = df_pivot
            else:
                combined_ind = combined_ind.merge(
                    df_pivot, on=["area_code", "year"], how="outer"
                )
        
        combined_par = None
        for param_name, df in parameters_dfs.items():
            df_pivot = df.pivot_table(
                index=["area_code", "year"], values="value", aggfunc="mean"
            ).reset_index()
            df_pivot.columns = ["area_code", "year", param_name]
            
            if combined_par is None:
                combined_par = df_pivot
            else:
                combined_par = combined_par.merge(
                    df_pivot, on=["area_code", "year"], how="outer"
                )
        
        return combined_ind, combined_par

    def standardize_to_province_level(
        self,
        df: pd.DataFrame,
        interpolate_missing: bool = True,
    ) -> pd.DataFrame:
        """Map mixed area codes to 2-letter province sigle and aggregate duplicates."""
        if df.empty:
            return df

        required = {"area_code", "year", "value"}
        if not required.issubset(df.columns):
            return df

        normalized = df.copy()
        normalized["area_code"] = normalized["area_code"].astype(str).str.strip().str.upper()
        normalized["province_code"] = normalized["area_code"].apply(map_area_code_to_sigla)
        normalized = normalized.dropna(subset=["province_code", "year", "value"])

        if normalized.empty:
            return pd.DataFrame(columns=["area_code", "year", "value"])

        aggregated = (
            normalized.groupby(["province_code", "year"], as_index=False)["value"]
            .mean()
            .rename(columns={"province_code": "area_code"})
        )

        if interpolate_missing:
            aggregated = self._interpolate_missing_provinces(aggregated)

        return aggregated.sort_values(["area_code", "year"]).reset_index(drop=True)

    def _interpolate_missing_provinces(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill missing province-year values using region mean then national mean."""
        if df.empty:
            return df

        years = sorted(df["year"].dropna().astype(int).unique())
        all_provinces = self._get_provinces()
        full_index = pd.MultiIndex.from_product(
            [all_provinces, years], names=["area_code", "year"]
        )

        expanded = (
            df.set_index(["area_code", "year"])["value"]
            .reindex(full_index)
            .reset_index()
        )

        for year in years:
            year_mask = expanded["year"] == year
            year_slice = expanded.loc[year_mask].copy()

            national_mean = year_slice["value"].mean()
            if pd.isna(national_mean):
                national_mean = 0.0

            for region in sorted(set(PROVINCE_TO_REGION.values())):
                provinces_in_region = [
                    p for p, r in PROVINCE_TO_REGION.items() if r == region
                ]
                region_mask = year_mask & expanded["area_code"].isin(provinces_in_region)
                region_mean = expanded.loc[region_mask, "value"].mean()
                fill_value = region_mean if not pd.isna(region_mean) else national_mean
                expanded.loc[region_mask, "value"] = expanded.loc[region_mask, "value"].fillna(fill_value)

            expanded.loc[year_mask, "value"] = expanded.loc[year_mask, "value"].fillna(national_mean)

        return expanded

    def _get_provinces(self) -> List[str]:
        """Get canonical list of 110 province sigle."""
        return list(PROVINCE_SIGLAS_110)

    def _get_province_to_region_mapping(self) -> Dict[str, str]:
        """Get mapping from province sigla to region names."""
        return dict(PROVINCE_TO_REGION)
