"""Tests for the SERA Digital Twin module."""

import numpy as np
import pandas as pd
import pytest

from sera.twin.causal_graph import (
    PARAMETER_EFFECT_DIRECTION,
    get_affected_indicators,
    get_dependent_indicators,
    get_influencing_parameters,
    get_parameter_reference,
    get_parameter_signal,
)
from sera.twin.model_trainer import IndicatorModel, ModelTrainer
from sera.twin.simulator import DigitalTwinSimulator


class TestCausalGraph:
    """Test causal graph functions."""

    def test_get_affected_indicators(self):
        """Test getting indicators affected by a parameter."""
        indicators = get_affected_indicators("education_spending_allocation")
        assert "school_enrollment" in indicators
        assert "income" in indicators

    def test_get_influencing_parameters(self):
        """Test getting parameters that influence an indicator."""
        parameters = get_influencing_parameters("income")
        assert "income_tax_rate" in parameters
        assert "education_spending_allocation" in parameters

    def test_get_dependent_indicators(self):
        """Test getting indicators dependent on another indicator."""
        dependents = get_dependent_indicators("income")
        assert "poverty_rate" in dependents
        assert "gini_coefficient" in dependents

    def test_parameter_effect_direction(self):
        """Test parameter effect directions."""
        assert PARAMETER_EFFECT_DIRECTION["education_spending_allocation"] == 1
        assert PARAMETER_EFFECT_DIRECTION["income_tax_rate"] == -1

    def test_parameter_reference_uses_historical_scale(self):
        """Test that parameter normalization uses the raw historical series."""
        baseline, scale = get_parameter_reference("healthcare_spending_allocation")
        assert baseline > 1000
        assert scale > 1
        assert abs(get_parameter_signal("healthcare_spending_allocation", baseline)) < 1e-6


class TestIndicatorModel:
    """Test individual indicator models."""

    def test_model_creation(self):
        """Test model initialization."""
        model = IndicatorModel("income", model_type="ridge")
        assert model.indicator_name == "income"
        assert not model.is_trained

    def test_model_training(self):
        """Test model training."""
        # Create synthetic data
        n_samples = 100
        n_features = 5
        X = np.random.randn(n_samples, n_features)
        y = np.random.randn(n_samples)
        feature_names = [f"feature_{i}" for i in range(n_features)]

        model = IndicatorModel("income", model_type="ridge")
        metrics = model.train(X, y, feature_names, test_size=0.2)

        assert model.is_trained
        assert "r2_test" in metrics
        assert "mae_test" in metrics
        assert metrics["train_samples"] > 0

    def test_model_prediction(self):
        """Test model prediction."""
        n_samples = 100
        n_features = 5
        X = np.random.randn(n_samples, n_features)
        y = np.random.randn(n_samples)
        feature_names = [f"feature_{i}" for i in range(n_features)]

        model = IndicatorModel("income")
        model.train(X, y, feature_names)

        X_test = np.random.randn(10, n_features)
        predictions = model.predict(X_test)

        assert len(predictions) == 10
        assert not np.any(np.isnan(predictions))

    def test_feature_importance(self):
        """Test feature importance extraction."""
        n_samples = 100
        n_features = 5
        X = np.random.randn(n_samples, n_features)
        y = np.random.randn(n_samples)
        feature_names = [f"feature_{i}" for i in range(n_features)]

        model = IndicatorModel("income", model_type="ridge")
        model.train(X, y, feature_names)

        importance = model.get_feature_importance()
        assert len(importance) == n_features
        assert all(isinstance(v, (int, float)) for v in importance.values())


class TestModelTrainer:
    """Test ModelTrainer class."""

    def test_trainer_creation(self):
        """Test trainer initialization."""
        trainer = ModelTrainer(model_type="ridge")
        assert trainer.model_type == "ridge"
        assert len(trainer.models) == 0

    def test_prepare_feature_matrix(self):
        """Test feature matrix preparation."""
        # Create synthetic data
        indicators_df = pd.DataFrame(
            {
                "area_code": ["IT001", "IT002"] * 5,
                "year": list(range(2020, 2025)) * 2,
                "income": np.random.randn(10),
                "employment": np.random.randn(10),
            }
        )

        parameters_df = pd.DataFrame(
            {
                "area_code": ["IT001", "IT002"] * 5,
                "year": list(range(2020, 2025)) * 2,
                "income_tax_rate": np.random.rand(10) * 100,
            }
        )

        trainer = ModelTrainer()
        X, y, feature_names = trainer.prepare_feature_matrix("income", indicators_df, parameters_df)

        if len(X) > 0:
            assert X.shape[1] == len(feature_names)
            assert len(y) == len(X)

    def test_save_and_load(self, tmp_path):
        """Test trainer persistence."""
        n_samples = 100
        X = np.random.randn(n_samples, 5)
        y = np.random.randn(n_samples)
        feature_names = [f"feature_{i}" for i in range(5)]

        trainer = ModelTrainer(model_type="ridge")
        model = IndicatorModel("income", model_type="ridge")
        model.train(X, y, feature_names)
        trainer.models["income"] = model

        model_path = tmp_path / "trainer.joblib"
        trainer.save(model_path)

        loaded = ModelTrainer.load(model_path)
        assert loaded.model_type == "ridge"
        assert "income" in loaded.models
        assert loaded.get_model("income") is not None


class TestSimulator:
    """Test DigitalTwinSimulator class."""

    def test_simulator_creation(self):
        """Test simulator initialization."""
        trainer = ModelTrainer()
        simulator = DigitalTwinSimulator(
            trainer,
            indicators=["income", "employment"],
            parameters=["income_tax_rate"],
        )

        assert len(simulator.indicators) == 2
        assert len(simulator.parameters) == 1

    def test_bounds_application(self):
        """Test application of logical bounds."""
        trainer = ModelTrainer()
        simulator = DigitalTwinSimulator(
            trainer,
            indicators=["life_expectancy"],
            parameters=[],
        )

        df = pd.DataFrame(
            {
                "area_code": ["IT001"],
                "year": [2026],
                "life_expectancy": [160],  # Should be bounded to 150
            }
        )

        result = simulator._apply_bounds(df)
        assert result["life_expectancy"].iloc[0] <= 150

    def test_causal_rules_application(self):
        """Test application of causal rules."""
        trainer = ModelTrainer()
        simulator = DigitalTwinSimulator(
            trainer,
            indicators=["income"],
            parameters=["income_tax_rate"],
        )

        predictions = pd.DataFrame(
            {
                "area_code": ["IT001"],
                "income": [50000],
            }
        )

        # A tax rate well above the historical baseline should decrease income.
        # Derive it from the reference so the test does not depend on the
        # absolute scale of the data files.
        baseline, scale = get_parameter_reference("income_tax_rate")
        parameters = pd.DataFrame(
            {
                "area_code": ["IT001"],
                "income_tax_rate": [baseline + 2 * scale],
            }
        )

        lagged = pd.DataFrame(
            {
                "income_lag1": [50000],
            }
        )

        result = simulator._apply_causal_rules(predictions, parameters, lagged)
        # Higher income tax should decrease income
        assert result["income"].iloc[0] < predictions["income"].iloc[0]

    def test_causal_rule_strength_is_configurable(self):
        """The rule strength scales the adjustment (used for sensitivity analysis)."""
        trainer = ModelTrainer()

        def rule_effect(strength):
            simulator = DigitalTwinSimulator(
                trainer,
                indicators=["income"],
                parameters=["income_tax_rate"],
                causal_rule_strength=strength,
            )
            predictions = pd.DataFrame({"area_code": ["IT001"], "income": [50000.0]})
            baseline, scale = get_parameter_reference("income_tax_rate")
            parameters = pd.DataFrame(
                {"area_code": ["IT001"], "income_tax_rate": [baseline + 2 * scale]}
            )
            lagged = pd.DataFrame({"income_lag1": [50000.0]})
            result = simulator._apply_causal_rules(predictions, parameters, lagged)
            return abs(result["income"].iloc[0] - 50000.0)

        assert rule_effect(0.0) == 0.0
        assert rule_effect(0.08) > rule_effect(0.04) > 0.0

    def test_post_propagation_bounds_are_reapplied(self):
        """Test that propagated values are clipped back into valid bounds."""

        class DummyModel:
            def __init__(self):
                self.feature_names = []

            def predict(self, X):
                return np.array([200.0])

        trainer = ModelTrainer()
        trainer.models["income"] = DummyModel()

        simulator = DigitalTwinSimulator(
            trainer,
            indicators=["income", "poverty_rate"],
            parameters=[],
        )

        current_state = pd.DataFrame(
            {
                "area_code": ["IT001"],
                "year": [2025],
                "income": [100.0],
                "poverty_rate": [99.0],
            }
        )
        parameters_year = pd.DataFrame(
            {
                "area_code": ["IT001"],
                "year": [2026],
            }
        )

        result = simulator.simulate_year(current_state, parameters_year)

        assert result["poverty_rate"].iloc[0] <= 100


class TestParameterAlignment:
    """Parameter rows must be matched to provinces by area_code, not position."""

    @staticmethod
    def _make_simulator():
        trainer = ModelTrainer()  # no trained models: predictions fall back to lag
        return DigitalTwinSimulator(
            trainer,
            indicators=["income"],
            parameters=["income_tax_rate"],
        )

    @staticmethod
    def _make_state():
        return pd.DataFrame(
            {
                "area_code": ["AA", "BB", "CC"],
                "year": [2025] * 3,
                "income": [100.0, 200.0, 300.0],
            }
        )

    def test_shuffled_parameter_rows_give_identical_results(self):
        simulator = self._make_simulator()
        baseline, scale = get_parameter_reference("income_tax_rate")

        ordered = pd.DataFrame(
            {
                "area_code": ["AA", "BB", "CC"],
                "year": [2026] * 3,
                "income_tax_rate": [baseline, baseline + 2 * scale, baseline],
            }
        )
        shuffled = ordered.iloc[[2, 0, 1]].reset_index(drop=True)

        result_ordered = simulator.simulate_year(self._make_state(), ordered)
        result_shuffled = simulator.simulate_year(self._make_state(), shuffled)

        pd.testing.assert_frame_equal(result_ordered, result_shuffled)

    def test_missing_province_falls_back_to_baseline_levers(self):
        simulator = self._make_simulator()
        baseline, _scale = get_parameter_reference("income_tax_rate")

        explicit_baseline = pd.DataFrame(
            {
                "area_code": ["AA", "BB", "CC"],
                "year": [2026] * 3,
                "income_tax_rate": [baseline] * 3,
            }
        )
        missing_bb = explicit_baseline[explicit_baseline["area_code"] != "BB"].copy()

        result_full = simulator.simulate_year(self._make_state(), explicit_baseline)
        result_missing = simulator.simulate_year(self._make_state(), missing_bb)

        pd.testing.assert_frame_equal(result_full, result_missing)


class TestIntegration:
    """Integration tests."""

    def test_end_to_end_pipeline(self):
        """Test complete pipeline from training to simulation."""
        # Create synthetic training data
        np.random.seed(42)
        n_years = 20
        n_provinces = 3

        indicators_data = []
        parameters_data = []

        for year in range(2001, 2001 + n_years):
            for prov in ["IT001", "IT002", "IT003"]:
                indicators_data.append(
                    {
                        "area_code": prov,
                        "year": year,
                        "income": 30000 + np.random.randn() * 5000,
                        "unemployment_rate": 8 + np.random.randn() * 2,
                    }
                )
                parameters_data.append(
                    {
                        "area_code": prov,
                        "year": year,
                        "income_tax_rate": 30 + np.random.randn() * 5,
                    }
                )

        indicators_df = pd.DataFrame(indicators_data)
        parameters_df = pd.DataFrame(parameters_data)

        # Train
        trainer = ModelTrainer(model_type="ridge")
        trainer.train_all_indicators(indicators_df, parameters_df)

        assert len(trainer.models) > 0

        # Create simulator
        simulator = DigitalTwinSimulator(
            trainer,
            indicators=["income", "unemployment_rate"],
            parameters=["income_tax_rate"],
        )

        # Prepare initial state
        initial_state = indicators_df[indicators_df["year"] == 2020].copy()
        initial_state["year"] = 2025

        # Create parameter scenarios
        params_2026 = parameters_df[parameters_df["year"] == 2020].copy()
        params_2026["year"] = 2026
        params_2026["income_tax_rate"] = 25

        # Simulate
        result = simulator.simulate_year(initial_state, params_2026)

        assert len(result) == n_provinces
        assert "income" in result.columns
        assert "unemployment_rate" in result.columns


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
