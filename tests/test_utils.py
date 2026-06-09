"""Tests for the twin's analysis and export utilities."""

import json

import numpy as np
import pandas as pd
import pytest

from sera.twin.utils import AnalysisTools, ScenarioBuilder, SimulationExporter


class TestToJson:
    def test_nan_becomes_null_and_output_is_valid_json(self, tmp_path):
        frame = pd.DataFrame(
            {
                "area_code": ["TO", "MI"],
                "year": [2025, 2025],
                "income": [np.nan, 30000.0],
            }
        )
        path = tmp_path / "out.json"
        SimulationExporter.to_json(frame, path)

        data = json.loads(path.read_text())
        assert data[0]["income"] is None
        assert data[1]["income"] == 30000.0
        assert data[0]["year"] == 2025


class TestScenarioBuilder:
    def test_baseline_scenario_shape(self):
        frame = ScenarioBuilder.baseline_scenario(
            ["TO", "MI"], [2026, 2027], {"income_tax_rate": 30.0}
        )
        assert len(frame) == 4
        assert (frame["income_tax_rate"] == 30.0).all()

    def test_create_variant_applies_from_change_year_onwards(self):
        baseline = ScenarioBuilder.baseline_scenario(
            ["TO"], [2026, 2027, 2028], {"income_tax_rate": 30.0}
        )
        variant = ScenarioBuilder.create_variant(baseline, {"income_tax_rate": (2027, 25.0)})
        by_year = variant.set_index("year")["income_tax_rate"]
        assert by_year[2026] == 30.0
        assert by_year[2027] == 25.0
        assert by_year[2028] == 25.0

    def test_sensitivity_sweep_one_scenario_per_value(self):
        baseline = ScenarioBuilder.baseline_scenario(["TO"], [2026], {"x": 1.0})
        scenarios = ScenarioBuilder.sensitivity_sweep(baseline, "x", [0.5, 1.5], 2026)
        assert set(scenarios) == {"x_0.5", "x_1.5"}
        assert scenarios["x_1.5"]["x"].iloc[0] == 1.5


class TestAnalysisTools:
    def test_compare_scenarios_metrics(self):
        baseline = pd.DataFrame({"year": [2030] * 2, "income": [100.0, 110.0]})
        variant = pd.DataFrame({"year": [2030] * 2, "income": [120.0, 130.0]})
        metrics = AnalysisTools.compare_scenarios(baseline, variant, "income", 2030)
        assert metrics["baseline_mean"] == 105.0
        assert metrics["variant_mean"] == 125.0
        assert metrics["absolute_difference"] == pytest.approx(20.0)
        assert metrics["percent_change"] == pytest.approx(19.0476, rel=1e-4)

    def test_compare_scenarios_missing_year(self):
        frame = pd.DataFrame({"year": [2030], "income": [100.0]})
        assert AnalysisTools.compare_scenarios(frame, frame, "income", 1999) == {}

    def test_impact_analysis_before_after(self):
        results = pd.DataFrame(
            {
                "area_code": ["TO"] * 4,
                "year": [2025, 2026, 2027, 2028],
                "income": [100.0, 100.0, 120.0, 120.0],
            }
        )
        metrics = AnalysisTools.impact_analysis(
            results, {"income_tax_rate": (2027, 25.0)}, "income"
        )
        assert metrics["income_tax_rate_before_mean"] == 100.0
        assert metrics["income_tax_rate_after_mean"] == 120.0
        assert metrics["income_tax_rate_percent_change"] == pytest.approx(20.0)

    def test_impact_analysis_missing_indicator(self):
        results = pd.DataFrame({"area_code": ["TO"], "year": [2025], "income": [1.0]})
        assert AnalysisTools.impact_analysis(results, {"x": (2025, 1.0)}, "nope") == {}

    def test_provincial_inequality_equal_values_zero_gini(self):
        results = pd.DataFrame(
            {"year": [2030] * 4, "income": [100.0] * 4, "area_code": list("ABCD")}
        )
        metrics = AnalysisTools.provincial_inequality(results, "income", 2030)
        assert metrics["gini"] == pytest.approx(0.0)
        assert metrics["min"] == metrics["max"] == 100.0
