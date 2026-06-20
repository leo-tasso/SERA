"""Demo: SERA Digital Twin Simulator - Usage Example"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sera.twin.data_loader import DataLoader
from sera.twin.model_trainer import ModelTrainer
from sera.twin.simulator import DigitalTwinSimulator
from sera.twin.utils import ScenarioBuilder, AnalysisTools
from sera.config import DATA_DIR


def demo_basic_training_and_simulation():
    """Demonstrate basic training and single-year simulation."""
    print("=" * 60)
    print("DEMO 1: Basic Training and Single-Year Simulation")
    print("=" * 60)

    # Define indicators and parameters
    indicators = {
        "population": ("demographic", 1),
        "income": ("demographic", 1),
        "unemployment_rate": ("labor", -1),
    }

    parameters = {
        "income_tax_rate": "annual_parameters",
        "education_spending_allocation": "annual_parameters",
    }

    try:
        # Load training data
        print("\n1. Loading training data...")
        loader = DataLoader(DATA_DIR)
        indicators_df, parameters_df = loader.prepare_training_data(
            indicators, parameters, min_year=2010, max_year=2025
        )
        print(f"   Loaded {len(indicators_df)} samples")

        if indicators_df.empty:
            print("   ⚠ No training data available. Skipping demo.")
            return

        # Train models
        print("\n2. Training models...")
        trainer = ModelTrainer(model_type="ridge")
        metrics = trainer.train_all_indicators(indicators_df, parameters_df)
        print(f"   Trained {len(trainer.models)} models")

        # Show metrics
        for indicator, m in metrics.items():
            if "r2_test" in m:
                print(f"     {indicator:20s}: R²={m['r2_test']:6.3f}")

        # Prepare initial state
        print("\n3. Loading initial state (year 2025)...")
        initial_state = (
            indicators_df[indicators_df["year"] == 2025].copy() if not indicators_df.empty else None
        )

        if initial_state is None or initial_state.empty:
            print("   ⚠ No 2025 data available. Using synthetic data.")
            initial_state = pd.DataFrame(
                {
                    "area_code": ["IT001", "IT002", "IT003"],
                    "year": [2025, 2025, 2025],
                    "population": [500000, 400000, 350000],
                    "income": [35000, 32000, 30000],
                    "unemployment_rate": [10, 12, 15],
                }
            )

        print(f"   Initial state for {len(initial_state)} provinces")

        # Create simulator
        print("\n4. Creating simulator...")
        simulator = DigitalTwinSimulator(
            trainer,
            indicators=[col for col in initial_state.columns if col not in ["area_code", "year"]],
            parameters=list(parameters.keys()),
        )
        print("   ✓ Simulator ready")

        # Simulate next year
        print("\n5. Simulating year 2026 with reduced income tax...")
        parameters_2026 = pd.DataFrame(
            {
                "area_code": initial_state["area_code"].values,
                "year": [2026] * len(initial_state),
            }
        )
        for param in parameters.keys():
            if param == "income_tax_rate":
                parameters_2026[param] = 25  # Reduced from 30
            else:
                parameters_2026[param] = 50  # Neutral

        next_state = simulator.simulate_year(initial_state, parameters_2026)

        print("\n6. Results Comparison (2025 → 2026):")
        print("-" * 60)
        for indicator in indicators.keys():
            if indicator in initial_state.columns:
                change = (
                    (next_state[indicator].mean() - initial_state[indicator].mean())
                    / initial_state[indicator].mean()
                    * 100
                )
                print(f"   {indicator:20s}: {change:+6.2f}%")

        print("\n✓ Demo 1 complete!\n")
        return next_state

    except Exception as e:
        print(f"   ✗ Error: {e}")
        return None


def demo_multi_year_scenario():
    """Demonstrate multi-year scenario simulation."""
    print("=" * 60)
    print("DEMO 2: Multi-Year Scenario (5 years)")
    print("=" * 60)

    indicators = {
        "population": ("demographic", 1),
        "income": ("demographic", 1),
        "unemployment_rate": ("labor", -1),
    }

    parameters = {
        "income_tax_rate": "annual_parameters",
        "education_spending_allocation": "annual_parameters",
    }

    try:
        # Train models
        print("\n1. Training models...")
        loader = DataLoader(DATA_DIR)
        indicators_df, parameters_df = loader.prepare_training_data(
            indicators, parameters, min_year=2010, max_year=2025
        )

        if indicators_df.empty:
            print("   ⚠ No training data available.")
            return

        trainer = ModelTrainer(model_type="ridge")
        trainer.train_all_indicators(indicators_df, parameters_df)
        print(f"   ✓ Trained {len(trainer.models)} models")

        # Initial state
        print("\n2. Setting initial state...")
        initial_state = pd.DataFrame(
            {
                "area_code": ["IT001", "IT002", "IT003"],
                "year": [2025, 2025, 2025],
                "population": [500000, 400000, 350000],
                "income": [35000, 32000, 30000],
                "unemployment_rate": [10, 12, 15],
            }
        )
        print(f"   ✓ Initial state for {len(initial_state)} provinces")

        # Create parameter scenarios for 5 years
        print("\n3. Creating 5-year scenario (education spending ↑)...")
        parameters_scenarios = []
        for year in range(2026, 2031):
            params = pd.DataFrame(
                {
                    "area_code": initial_state["area_code"].values,
                    "year": [year] * len(initial_state),
                }
            )
            params["income_tax_rate"] = 28  # Reduced
            params["education_spending_allocation"] = 8 + (year - 2026)  # Increasing
            parameters_scenarios.append(params)

        # Run simulation
        print("\n4. Simulating 5 years...")
        simulator = DigitalTwinSimulator(
            trainer,
            indicators=[col for col in initial_state.columns if col not in ["area_code", "year"]],
            parameters=list(parameters.keys()),
        )
        results = simulator.simulate_scenario(initial_state, parameters_scenarios, apply_rules=True)

        # Analyze
        print("\n5. Results Summary:")
        print("-" * 60)
        print(f"{'Year':<8} {'Income':>12} {'Unemployment':>15} {'Population':>12}")
        print("-" * 60)
        for year in sorted(results["year"].unique()):
            year_data = results[results["year"] == year]
            income = year_data["income"].mean()
            unemployment = year_data["unemployment_rate"].mean()
            population = year_data["population"].mean()
            print(f"{year:<8} {income:>12.0f} {unemployment:>14.1f}% {population:>12.0f}")

        print("\n✓ Demo 2 complete!\n")

    except Exception as e:
        print(f"   ✗ Error: {e}")


def demo_scenario_comparison():
    """Demonstrate comparing two scenarios."""
    print("=" * 60)
    print("DEMO 3: Scenario Comparison")
    print("=" * 60)

    indicators = {
        "population": ("demographic", 1),
        "income": ("demographic", 1),
        "unemployment_rate": ("labor", -1),
    }

    parameters = {
        "income_tax_rate": "annual_parameters",
        "education_spending_allocation": "annual_parameters",
    }

    try:
        # Train
        print("\n1. Training models...")
        loader = DataLoader(DATA_DIR)
        indicators_df, parameters_df = loader.prepare_training_data(
            indicators, parameters, min_year=2010, max_year=2025
        )

        if indicators_df.empty:
            print("   ⚠ No training data available.")
            return

        trainer = ModelTrainer(model_type="ridge")
        trainer.train_all_indicators(indicators_df, parameters_df)

        # Initial state
        initial_state = pd.DataFrame(
            {
                "area_code": ["IT001", "IT002", "IT003"],
                "year": [2025, 2025, 2025],
                "population": [500000, 400000, 350000],
                "income": [35000, 32000, 30000],
                "unemployment_rate": [10, 12, 15],
            }
        )

        simulator = DigitalTwinSimulator(
            trainer,
            indicators=[col for col in initial_state.columns if col not in ["area_code", "year"]],
            parameters=list(parameters.keys()),
        )

        # Scenario A: Conservative (high taxes, low education spending)
        print("\n2. Scenario A: Conservative (high taxes, low education)")
        params_conservative = []
        for year in range(2026, 2031):
            params = pd.DataFrame(
                {
                    "area_code": initial_state["area_code"].values,
                    "year": [year] * len(initial_state),
                    "income_tax_rate": 35,
                    "education_spending_allocation": 3,
                }
            )
            params_conservative.append(params)

        results_a = simulator.simulate_scenario(initial_state, params_conservative)

        # Scenario B: Progressive (low taxes, high education spending)
        print("\n3. Scenario B: Progressive (low taxes, high education)")
        params_progressive = []
        for year in range(2026, 2031):
            params = pd.DataFrame(
                {
                    "area_code": initial_state["area_code"].values,
                    "year": [year] * len(initial_state),
                    "income_tax_rate": 20,
                    "education_spending_allocation": 10,
                }
            )
            params_progressive.append(params)

        results_b = simulator.simulate_scenario(initial_state, params_progressive)

        # Compare
        print("\n4. Comparison (2030 outcomes):")
        print("-" * 60)
        comparison = AnalysisTools.compare_scenarios(results_a, results_b, "income", 2030)

        print(f"   Income - Conservative:  ${comparison['baseline_mean']:>10,.0f}")
        print(f"   Income - Progressive:   ${comparison['variant_mean']:>10,.0f}")
        print(f"   Difference:             ${comparison['absolute_difference']:>10,.0f}")
        print(f"   % Change:               {comparison['percent_change']:>10.1f}%")

        print("\n✓ Demo 3 complete!\n")

    except Exception as e:
        print(f"   ✗ Error: {e}")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("SERA Digital Twin - Demonstration")
    print("=" * 60 + "\n")

    # Run demos
    demo_basic_training_and_simulation()
    demo_multi_year_scenario()
    demo_scenario_comparison()

    print("=" * 60)
    print("All demos complete!")
    print("=" * 60)
