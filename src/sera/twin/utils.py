"""Utilities for the SERA Digital Twin."""

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


class SimulationExporter:
    """Export simulation results in various formats."""

    @staticmethod
    def to_json(results: pd.DataFrame, output_path: Path) -> None:
        """Export results to JSON.

        Args:
            results: Simulation results DataFrame
            output_path: Output file path
        """
        data = []
        for _, row in results.iterrows():
            row_dict = row.to_dict()
            # Convert numpy types to Python types and NaN to null: json.dump
            # would otherwise emit a bare NaN token, which is not valid JSON.
            sanitized = {}
            for k, v in row_dict.items():
                if isinstance(v, (np.floating, np.integer)):
                    v = v.item()
                if isinstance(v, float) and np.isnan(v):
                    v = None
                sanitized[k] = v
            data.append(sanitized)

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2, allow_nan=False)

    @staticmethod
    def to_excel(results: pd.DataFrame, output_path: Path) -> None:
        """Export results to Excel with multiple sheets.

        Args:
            results: Simulation results DataFrame
            output_path: Output file path
        """
        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            # All data
            results.to_excel(writer, sheet_name="All Data", index=False)

            # Summary by year
            summary = results.groupby("year").describe().round(2)
            summary.to_excel(writer, sheet_name="Summary")

            # Provincial profiles (one sheet per province, first few)
            for province in results["area_code"].unique()[:5]:
                province_data = results[results["area_code"] == province]
                province_data.to_excel(writer, sheet_name=f"Province {province}", index=False)


class ScenarioBuilder:
    """Build parameter scenarios for simulation."""

    @staticmethod
    def baseline_scenario(
        provinces: List[str],
        years: List[int],
        parameters: Dict[str, float],
    ) -> pd.DataFrame:
        """Create baseline scenario (unchanged parameters).

        Args:
            provinces: List of province codes
            years: List of years to simulate
            parameters: Dict of parameter names and baseline values

        Returns:
            DataFrame with baseline parameters
        """
        rows = []
        for year in years:
            for province in provinces:
                row = {"area_code": province, "year": year}
                row.update(parameters)
                rows.append(row)
        return pd.DataFrame(rows)

    @staticmethod
    def create_variant(
        baseline: pd.DataFrame,
        modifications: Dict[str, Tuple[int, float]],
    ) -> pd.DataFrame:
        """Create scenario variant by modifying parameters.

        Args:
            baseline: Baseline scenario
            modifications: Dict of {param_name: (from_year, new_value)}

        Returns:
            Modified scenario
        """
        variant = baseline.copy()

        for param_name, (from_year, new_value) in modifications.items():
            mask = variant["year"] >= from_year
            variant.loc[mask, param_name] = new_value

        return variant

    @staticmethod
    def sensitivity_sweep(
        baseline: pd.DataFrame,
        parameter: str,
        values: List[float],
        from_year: int,
    ) -> Dict[str, pd.DataFrame]:
        """Create multiple scenarios varying one parameter.

        Args:
            baseline: Baseline scenario
            parameter: Parameter to vary
            values: List of values to try
            from_year: Year to start parameter changes

        Returns:
            Dict mapping scenario names to scenario DataFrames
        """
        scenarios = {}

        for value in values:
            scenario_name = f"{parameter}_{value}"
            modifications = {parameter: (from_year, value)}
            scenarios[scenario_name] = ScenarioBuilder.create_variant(baseline, modifications)

        return scenarios


class AnalysisTools:
    """Tools for analyzing simulation results."""

    @staticmethod
    def compare_scenarios(
        baseline_results: pd.DataFrame,
        variant_results: pd.DataFrame,
        indicator: str,
        year: int,
    ) -> Dict[str, float]:
        """Compare results between two scenarios.

        Args:
            baseline_results: Baseline simulation results
            variant_results: Variant simulation results
            indicator: Indicator to compare
            year: Year to compare

        Returns:
            Dict with comparison metrics
        """
        base_year = baseline_results[baseline_results["year"] == year][indicator]
        var_year = variant_results[variant_results["year"] == year][indicator]

        if base_year.empty or var_year.empty:
            return {}

        base_mean = base_year.mean()
        var_mean = var_year.mean()

        return {
            "baseline_mean": float(base_mean),
            "variant_mean": float(var_mean),
            "absolute_difference": float(var_mean - base_mean),
            "percent_change": float(
                ((var_mean - base_mean) / base_mean) * 100 if base_mean != 0 else 0
            ),
            "baseline_std": float(base_year.std()),
            "variant_std": float(var_year.std()),
            "std_change": float(var_year.std() - base_year.std()),
        }

    @staticmethod
    def impact_analysis(
        results: pd.DataFrame,
        parameter_changes: Dict[str, Tuple[float, float]],
        indicator: str,
    ) -> Dict[str, float]:
        """Measure how an indicator moved after each parameter change.

        Args:
            results: Simulation results (area_code, year, indicator columns)
            parameter_changes: Dict of {param_name: (from_year, new_value)},
                the same shape ScenarioBuilder.create_variant accepts
            indicator: Indicator to measure impact on

        Returns:
            Dict with, per changed parameter, the indicator mean before and
            after the change year, plus the absolute and percent shift.
        """
        if indicator not in results.columns:
            return {}

        metrics: Dict[str, float] = {}
        for param, (from_year, _new_value) in parameter_changes.items():
            before = results[results["year"] < from_year][indicator].dropna()
            after = results[results["year"] >= from_year][indicator].dropna()
            if before.empty or after.empty:
                continue

            before_mean = float(before.mean())
            after_mean = float(after.mean())
            metrics[f"{param}_before_mean"] = before_mean
            metrics[f"{param}_after_mean"] = after_mean
            metrics[f"{param}_absolute_change"] = after_mean - before_mean
            metrics[f"{param}_percent_change"] = float(
                (after_mean - before_mean) / abs(before_mean) * 100 if before_mean != 0 else 0.0
            )
        return metrics

    @staticmethod
    def provincial_inequality(
        results: pd.DataFrame,
        indicator: str,
        year: int,
    ) -> Dict[str, float]:
        """Calculate inequality metrics for a provincial indicator.

        Args:
            results: Simulation results
            indicator: Indicator to analyze
            year: Year to analyze

        Returns:
            Dict with inequality metrics
        """
        year_data = results[results["year"] == year][indicator]

        if year_data.empty:
            return {}

        # Gini coefficient calculation
        sorted_vals = np.sort(year_data.values)
        n = len(sorted_vals)
        gini = (2 * np.sum(np.arange(1, n + 1) * sorted_vals)) / (n * np.sum(sorted_vals)) - (
            n + 1
        ) / n

        return {
            "mean": float(year_data.mean()),
            "std": float(year_data.std()),
            "gini": float(gini),
            "cv": float(year_data.std() / (year_data.mean() + 1e-6)),
            "min": float(year_data.min()),
            "max": float(year_data.max()),
            "p25": float(year_data.quantile(0.25)),
            "median": float(year_data.median()),
            "p75": float(year_data.quantile(0.75)),
        }


class CausalPathTracer:
    """Trace causal effects through the indicator network."""

    def __init__(self, causal_graph: Dict[str, List[str]]):
        """Initialize tracer.

        Args:
            causal_graph: Dict mapping indicators to dependent indicators
        """
        self.causal_graph = causal_graph

    def trace_effect(
        self,
        source: str,
        max_depth: int = 3,
    ) -> Dict[str, int]:
        """Trace all indicators affected by a source indicator.

        Args:
            source: Source indicator
            max_depth: Maximum depth to trace

        Returns:
            Dict mapping affected indicators to distance from source
        """
        affected = {source: 0}
        to_visit = [(source, 1)]

        while to_visit:
            current, depth = to_visit.pop(0)

            if depth > max_depth:
                continue

            if current in self.causal_graph:
                for dependent in self.causal_graph[current]:
                    if dependent not in affected:
                        affected[dependent] = depth
                        to_visit.append((dependent, depth + 1))

        return affected
