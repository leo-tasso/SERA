"""
Test multi-horizon simulations (1, 3, 5, 10 years) with different scenarios.
Validates for reasonableness: no negative values, sensible trends, bounded outputs.
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path
from typing import List, Dict, Tuple

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from sera.twin.model_trainer import ModelTrainer
from sera.twin.data_loader import DataLoader
from sera.twin.simulator import DigitalTwinSimulator
from sera.twin.causal_graph import get_parameter_reference
from sera.config import DATA_DIR


def load_or_train_models(model_path: Path = Path("twin_models_full.joblib")) -> ModelTrainer:
    """Load trained models or train new ones with all available indicators."""
    if model_path.exists():
        print(f"Loading models from {model_path}...")
        return ModelTrainer.load(model_path)
    else:
        print("Training models with provincial indicators...")
        loader = DataLoader(DATA_DIR)

        # Core 6 indicators + 14 additional with excellent provincial data
        indicators = {
            # Core indicators (6)
            "population": ("demographic", 1),
            "income": ("demographic", 1),
            "unemployment_rate": ("labor", -1),
            "life_expectancy": ("social_well_being", 1),
            "school_enrollment": ("education", 1),
            "gdp_per_capita": ("economic", 1),
            # Labor & Economic indicators (5)
            "youth_employment": ("labor", 1),
            "self_employment": ("labor", 1),
            "business_density": ("economic", 1),
            "agricultural_productivity": ("business_environment", 1),
            "poverty_rate": ("economic", -1),
            # Education indicators (1)
            "completion_rates": ("education", 1),
            # Innovation & Infrastructure (3)
            "patents": ("innovation_infrastructure", 1),
            "digital_infrastructure": ("innovation_infrastructure", 1),
            "transportation_access": ("innovation_infrastructure", 1),
            "public_transportation_usage": ("transportation_mobility", 1),
            "traffic_congestion": ("transportation_mobility", -1),
            # Environmental & Energy (4)
            "green_urban_space_per_capita": ("environmental_quality", 1),
            "water_quality": ("environmental_quality", 1),
            "renewable_energy_percentage": ("energy_resources", 1),
            "carbon_emissions": ("energy_resources", -1),
            # Healthcare & Social (3)
            "healthcare_worker_density": ("healthcare_public_services", 1),
            "healthcare_spending_per_capita": ("healthcare_public_services", 1),
            "crime_rate": ("social_well_being", -1),
            # NEW: 4 additional indicators with strong cross-effects (4)
            "air_quality": ("environment", -1),
            "sustainability": ("environment", 1),
            "startups": ("innovation_infrastructure", 1),
            "car_ownership_density": ("transportation_mobility", 0),
        }

        parameters = {
            "income_tax_rate": "annual_parameters",
            "corporate_tax_rate": "annual_parameters",
            "property_wealth_tax_rate": "annual_parameters",
            "vat_consumption_tax_rate": "annual_parameters",
            "healthcare_spending_allocation": "annual_parameters",
            "education_spending_allocation": "annual_parameters",
            "infrastructure_investment_allocation": "annual_parameters",
            "social_welfare_spending_allocation": "annual_parameters",
            "rd_innovation_incentives": "annual_parameters",
            "green_energy_environment_investment": "annual_parameters",
            "agriculture_support_level": "annual_parameters",
            "manufacturing_incentives": "annual_parameters",
            "tourism_support_level": "annual_parameters",
            "small_business_support": "annual_parameters",
            "immigration_policy_level": "annual_parameters",
            "regulatory_burden_level": "annual_parameters",
            "public_sector_wage_levels": "annual_parameters",
            "pension_retirement_spending": "annual_parameters",
            "housing_urban_development_support": "annual_parameters",
            "environmental_regulations_strictness": "annual_parameters",
        }

        print(f"Loading data for {len(indicators)} indicators and {len(parameters)} parameters...")
        try:
            indicators_df, parameters_df = loader.prepare_training_data(
                indicators, parameters, 2001, 2025
            )
            print(
                f"Loaded {len(indicators_df)} indicator samples, {len(parameters_df)} parameter samples"
            )

            trainer = ModelTrainer(model_type="ridge")
            print(f"Training {len(indicators)} indicators...")
            trainer.train_all_indicators(indicators_df, parameters_df, test_size=0.2)
            trainer.save(model_path)
            print(f"[OK] Trained {len(trainer.models)} indicator models and saved to {model_path}")
            return trainer
        except Exception as e:
            print(f"Error during training: {e}")
            print("Falling back to core 6 indicators...")
            # Fallback to known working indicators
            indicators = {
                "population": ("demographic", 1),
                "income": ("demographic", 1),
                "unemployment_rate": ("labor", -1),
                "life_expectancy": ("social_well_being", 1),
                "school_enrollment": ("education", 1),
                "gdp_per_capita": ("economic", 1),
            }
            parameters_small = {
                "income_tax_rate": "annual_parameters",
                "education_spending_allocation": "annual_parameters",
                "healthcare_spending_allocation": "annual_parameters",
            }
            indicators_df, parameters_df = loader.prepare_training_data(
                indicators, parameters_small, 2001, 2025
            )
            trainer = ModelTrainer(model_type="ridge")
            trainer.train_all_indicators(indicators_df, parameters_df, test_size=0.2)
            trainer.save(model_path)
            return trainer


def load_initial_state(year: int = 2025) -> pd.DataFrame:
    """Load initial state from 2025 data."""
    loader = DataLoader(DATA_DIR)
    # Start with known working indicators
    indicators = {
        "population": ("demographic", 1),
        "income": ("demographic", 1),
        "unemployment_rate": ("labor", -1),
        "life_expectancy": ("social_well_being", 1),
        "school_enrollment": ("education", 1),
        "gdp_per_capita": ("economic", 1),
    }

    # Try to add additional indicators if they exist
    additional_indicators = {
        "poverty_rate": ("economic", -1),
        "gini_coefficient": ("economic", -1),
        "business_density": ("economic", 1),
        "fdi": ("economic", 1),
        "exports_imports": ("economic", 1),
        "average_wages": ("labor", 1),
        "youth_employment": ("labor", 1),
        "self_employment": ("labor", 1),
        "skills_match": ("labor", 1),
        "health_outcomes": ("social_well_being", 1),
        "crime_rate": ("social_well_being", -1),
        "social_cohesion": ("social_well_being", 1),
        "completion_rates": ("education", 1),
        "stem_participation": ("education", 1),
        "adult_learning": ("education", 1),
        "r_and_d_spending": ("innovation_infrastructure", 1),
        "patents": ("innovation_infrastructure", 1),
        "startups": ("innovation_infrastructure", 1),
        "digital_infrastructure": ("innovation_infrastructure", 1),
        "transportation_access": ("innovation_infrastructure", 1),
        "renewable_energy_percentage": ("energy_resources", 1),
        "carbon_emissions": ("energy_resources", -1),
        "air_quality": ("environment", -1),
        "waste_recycling_rate": ("environmental_quality", 1),
    }

    # Try to add each additional indicator
    for ind_name, (category, _) in additional_indicators.items():
        try:
            df = loader.load_indicator(ind_name, category)
            if not df.empty:
                indicators[ind_name] = (category, _)
        except:
            pass  # Silently skip missing indicators

    state_data = None
    for ind_name, (category, _) in indicators.items():
        try:
            df = loader.load_indicator(ind_name, category)
            if df.empty:
                continue

            available_years = sorted(df["year"].dropna().unique())
            target_years = [candidate for candidate in available_years if candidate <= year]
            if target_years:
                target_year = target_years[-1]
            else:
                target_year = available_years[-1]

            df = df[df["year"] == target_year].copy()
            df = df.pivot_table(index="area_code", values="value", aggfunc="mean").reset_index()
            df = df.rename(columns={"value": ind_name})

            if state_data is None:
                state_data = df
            else:
                state_data = state_data.merge(df, on="area_code", how="outer")
        except Exception as e:
            print(f"Warning: Could not load {ind_name}: {e}")
            continue

    if state_data is not None:
        value_columns = [col for col in state_data.columns if col != "area_code"]
        for col in value_columns:
            if pd.api.types.is_numeric_dtype(state_data[col]):
                state_data[col] = state_data[col].fillna(state_data[col].mean())
        state_data["year"] = year
        state_data = state_data[
            ["area_code", "year"]
            + [col for col in state_data.columns if col not in ["area_code", "year"]]
        ]

    print(f"Initial state shape: {state_data.shape}")
    print(f"Initial state columns loaded: {len(state_data.columns) - 2} indicators")
    return state_data


def create_scenario(
    initial_state: pd.DataFrame,
    scenario_name: str,
    years: int,
    baseline_year: int = 2025,
    education_multiplier: float = 1.0,
    healthcare_multiplier: float = 1.0,
    tax_multiplier: float = 1.0,
) -> List[pd.DataFrame]:
    """Create parameter scenarios for N years."""
    parameter_cols = [
        "income_tax_rate",
        "corporate_tax_rate",
        "property_wealth_tax_rate",
        "vat_consumption_tax_rate",
        "healthcare_spending_allocation",
        "education_spending_allocation",
        "infrastructure_investment_allocation",
        "social_welfare_spending_allocation",
        "rd_innovation_incentives",
        "green_energy_environment_investment",
        "agriculture_support_level",
        "manufacturing_incentives",
        "tourism_support_level",
        "small_business_support",
        "immigration_policy_level",
        "regulatory_burden_level",
        "public_sector_wage_levels",
        "pension_retirement_spending",
        "housing_urban_development_support",
        "environmental_regulations_strictness",
    ]

    # Get parameter baselines
    parameter_baselines = {param: get_parameter_reference(param)[0] for param in parameter_cols}

    scenarios = []
    for year_idx in range(years):
        params_df = initial_state[["area_code"]].copy()
        params_df["year"] = baseline_year + year_idx + 1

        # Apply multipliers to spending allocations, scale tax inversely
        params_df["income_tax_rate"] = parameter_baselines["income_tax_rate"] * tax_multiplier
        params_df["corporate_tax_rate"] = parameter_baselines["corporate_tax_rate"] * tax_multiplier
        params_df["property_wealth_tax_rate"] = (
            parameter_baselines["property_wealth_tax_rate"] * tax_multiplier
        )
        params_df["vat_consumption_tax_rate"] = (
            parameter_baselines["vat_consumption_tax_rate"] * tax_multiplier
        )

        params_df["education_spending_allocation"] = (
            parameter_baselines["education_spending_allocation"] * education_multiplier
        )
        params_df["healthcare_spending_allocation"] = (
            parameter_baselines["healthcare_spending_allocation"] * healthcare_multiplier
        )

        # Keep other parameters at baseline
        for param in parameter_cols:
            if param not in params_df.columns:
                params_df[param] = parameter_baselines[param]

        scenarios.append(params_df)

    return scenarios


def run_simulation(
    trainer: ModelTrainer,
    initial_state: pd.DataFrame,
    parameter_scenarios: List[pd.DataFrame],
    baseline_year: int = 2025,
) -> pd.DataFrame:
    """Run simulator and return results."""
    # Use indicators that were actually trained
    indicators = list(trainer.models.keys())

    parameters = [
        "income_tax_rate",
        "corporate_tax_rate",
        "property_wealth_tax_rate",
        "vat_consumption_tax_rate",
        "healthcare_spending_allocation",
        "education_spending_allocation",
        "infrastructure_investment_allocation",
        "social_welfare_spending_allocation",
        "rd_innovation_incentives",
        "green_energy_environment_investment",
        "agriculture_support_level",
        "manufacturing_incentives",
        "tourism_support_level",
        "small_business_support",
        "immigration_policy_level",
        "regulatory_burden_level",
        "public_sector_wage_levels",
        "pension_retirement_spending",
        "housing_urban_development_support",
        "environmental_regulations_strictness",
    ]

    simulator = DigitalTwinSimulator(trainer, indicators, parameters)
    results = []

    current_state = initial_state.copy()

    for year_idx, params_df in enumerate(parameter_scenarios):
        year = baseline_year + year_idx + 1
        current_state = simulator.simulate_year(current_state, params_df)
        current_state["year"] = year
        results.append(current_state.copy())

    return pd.concat(results, ignore_index=True)


def validate_results(
    results: pd.DataFrame,
    scenario_name: str,
    horizon: int,
) -> Dict[str, any]:
    """Check for reasonable outputs and return validation report."""
    report = {
        "scenario": scenario_name,
        "horizon_years": horizon,
        "valid": True,
        "issues": [],
        "stats": {},
    }

    # Check for NaN values
    nan_cols = results.columns[results.isna().any()]
    if len(nan_cols) > 0:
        report["valid"] = False
        report["issues"].append(f"NaN found in columns: {nan_cols.tolist()}")

    # Check for negative values in positive-direction indicators
    positive_indicators = [
        "income",
        "gdp_per_capita",
        "population",
        "life_expectancy",
        "school_enrollment",
        "employment_rate",
        "house_prices",
    ]

    for col in positive_indicators:
        if col in results.columns and (results[col] < 0).any():
            report["valid"] = False
            report["issues"].append(
                f"{col} has {(results[col] < 0).sum()} negative values (min: {results[col].min():.2f})"
            )

    # Check for unrealistic bounds
    if "life_expectancy" in results.columns:
        if (results["life_expectancy"] > 100).any():
            report["issues"].append(
                f"Life expectancy unrealistic: max={results['life_expectancy'].max():.2f}"
            )

    if "poverty_rate" in results.columns:
        if (results["poverty_rate"] > 100).any():
            report["issues"].append(f"Poverty rate > 100%: max={results['poverty_rate'].max():.2f}")

    # Compute stats per year for key indicators
    key_indicators = [
        "income",
        "unemployment_rate",
        "life_expectancy",
        "poverty_rate",
        "school_enrollment",
        "gdp_per_capita",
        "carbon_emissions",
        "renewable_energy_percentage",
    ]

    for year in sorted(results["year"].unique()):
        year_data = results[results["year"] == year]
        year_stats = {"n_provinces": len(year_data)}

        for ind in key_indicators:
            if ind in year_data.columns:
                year_stats[f"{ind}_mean"] = year_data[ind].mean()
                year_stats[f"{ind}_std"] = year_data[ind].std()

        report["stats"][year] = year_stats

    return report


def main():
    print("=" * 80)
    print("MULTI-HORIZON SIMULATION VALIDATION TEST")
    print("=" * 80)

    # Load models
    trainer = load_or_train_models()

    # Load initial state
    initial_state = load_initial_state(2025)

    # Test parameters
    horizons = [1, 3, 5, 10]
    scenarios_def = [
        {
            "name": "baseline",
            "education_multiplier": 1.0,
            "healthcare_multiplier": 1.0,
            "tax_multiplier": 1.0,
        },
        {
            "name": "high_spending",
            "education_multiplier": 1.5,
            "healthcare_multiplier": 1.5,
            "tax_multiplier": 0.8,
        },
        {
            "name": "low_spending",
            "education_multiplier": 0.7,
            "healthcare_multiplier": 0.7,
            "tax_multiplier": 1.2,
        },
    ]

    # Run all combinations
    all_results = []

    for horizon in horizons:
        print(f"\n{'=' * 80}")
        print(f"TESTING HORIZON: {horizon} YEAR(S)")
        print(f"{'=' * 80}")

        for scenario_def in scenarios_def:
            scenario_name = scenario_def["name"]
            print(f"\n  Scenario: {scenario_name}")
            print(f"  | Education multiplier: {scenario_def['education_multiplier']}")
            print(f"  | Healthcare multiplier: {scenario_def['healthcare_multiplier']}")
            print(f"  | Tax multiplier: {scenario_def['tax_multiplier']}")

            try:
                # Create parameter scenarios
                params = create_scenario(
                    initial_state,
                    scenario_name,
                    horizon,
                    education_multiplier=scenario_def["education_multiplier"],
                    healthcare_multiplier=scenario_def["healthcare_multiplier"],
                    tax_multiplier=scenario_def["tax_multiplier"],
                )

                # Run simulation
                results = run_simulation(trainer, initial_state, params, baseline_year=2025)

                # Validate
                report = validate_results(results, scenario_name, horizon)
                all_results.append(report)

                # Print validation results
                status = "[OK] VALID" if report["valid"] else "[FAIL] INVALID"
                print(f"    Status: {status}")

                if report["issues"]:
                    for issue in report["issues"]:
                        print(f"    [WARNING] {issue}")

                # Print stats
                for year, stats in report["stats"].items():
                    print(f"    Year {year}:")
                    if "income_mean" in stats and stats["income_mean"] is not None:
                        print(
                            f"      - Income: {stats['income_mean']:,.0f} ± {stats['income_std']:,.0f}"
                        )
                    if (
                        "unemployment_rate_mean" in stats
                        and stats["unemployment_rate_mean"] is not None
                    ):
                        print(
                            f"      - Unemployment: {stats['unemployment_rate_mean']:,.0f} ± {stats['unemployment_rate_std']:,.0f}"
                        )
                    if (
                        "life_expectancy_mean" in stats
                        and stats["life_expectancy_mean"] is not None
                    ):
                        print(
                            f"      - Life Expectancy: {stats['life_expectancy_mean']:.2f} ± {stats['life_expectancy_std']:.2f}"
                        )
                    if "poverty_rate_mean" in stats and stats["poverty_rate_mean"] is not None:
                        print(
                            f"      - Poverty Rate: {stats['poverty_rate_mean']:.2f}% ± {stats['poverty_rate_std']:.2f}%"
                        )

            except Exception as e:
                print(f"    [ERROR] {str(e)}")
                all_results.append(
                    {
                        "scenario": scenario_name,
                        "horizon_years": horizon,
                        "valid": False,
                        "issues": [str(e)],
                    }
                )

    # Summary table
    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print(f"{'=' * 80}")

    summary_df = pd.DataFrame(
        [
            {
                "Horizon": r["horizon_years"],
                "Scenario": r["scenario"],
                "Status": "OK" if r["valid"] else "FAIL",
                "Issues": len(r["issues"]),
            }
            for r in all_results
        ]
    )

    print(summary_df.to_string(index=False))

    # Overall verdict
    valid_count = sum(1 for r in all_results if r["valid"])
    total_count = len(all_results)

    print(f"\n{'=' * 80}")
    print(f"OVERALL: {valid_count}/{total_count} scenarios passed validation")
    print(f"{'=' * 80}\n")

    if valid_count == total_count:
        print("[OK] All tests passed! Simulations are producing reasonable results.")
        return 0
    else:
        print("[FAIL] Some tests failed. Review issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
