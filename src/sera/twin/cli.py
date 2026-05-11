"""CLI entry point for the SERA Digital Twin Simulator."""

import argparse
import logging
from pathlib import Path
import pandas as pd
import json
from typing import List, Dict

from sera.twin.data_loader import DataLoader
from sera.twin.causal_graph import get_parameter_reference
from sera.twin.province_mapping import PROVINCE_SIGLAS_110, map_area_code_to_sigla
from sera.twin.model_trainer import ModelTrainer
from sera.twin.simulator import DigitalTwinSimulator
from sera.config import DATA_DIR

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def normalize_wide_geography(df: pd.DataFrame) -> pd.DataFrame:
    """Force a wide dataframe to canonical 110 provinces (2-letter sigla).
    
    If national-level data (area_code='IT') is provided, broadcasts to all provinces.
    """
    normalized = df.copy()
    normalized["area_code"] = normalized["area_code"].astype(str).str.strip().str.upper()
    
    # Check for national data (IT) and broadcast to all provinces
    national_rows = normalized[normalized["area_code"] == "IT"]
    if len(national_rows) > 0:
        # Get the data columns (everything except area_code and year)
        data_cols = [col for col in normalized.columns if col not in ["area_code", "year"]]
        
        # For each national row, create rows for all 110 provinces
        expanded_rows = []
        for _, row in national_rows.iterrows():
            for sigla in PROVINCE_SIGLAS_110:
                new_row = row.copy()
                new_row["area_code"] = sigla
                expanded_rows.append(new_row)
        
        # Remove original national rows and add expanded rows
        normalized = normalized[normalized["area_code"] != "IT"]
        expanded_df = pd.DataFrame(expanded_rows)
        normalized = pd.concat([normalized, expanded_df], ignore_index=True)
    
    # Now map any remaining area codes to province sigla
    normalized["area_code"] = normalized["area_code"].apply(map_area_code_to_sigla)
    normalized = normalized.dropna(subset=["area_code"])

    numeric_cols = [
        col for col in normalized.columns if col not in ["area_code", "year"]
    ]
    normalized = (
        normalized.groupby(["area_code", "year"], as_index=False)[numeric_cols]
        .mean(numeric_only=True)
    )

    full_index = pd.MultiIndex.from_product(
        [PROVINCE_SIGLAS_110, sorted(normalized["year"].unique())],
        names=["area_code", "year"],
    )
    normalized = normalized.set_index(["area_code", "year"]).reindex(full_index).reset_index()

    for col in numeric_cols:
        yearly_means = normalized.groupby("year")[col].transform("mean")
        global_mean = normalized[col].mean()
        if pd.isna(global_mean):
            global_mean = 0.0
        normalized[col] = normalized[col].fillna(yearly_means).fillna(global_mean)

    return normalized


def load_training_data(
    data_dir: Path,
    indicators: Dict[str, tuple],
    parameters: Dict[str, str],
    start_year: int = 2001,
    end_year: int = 2025,
):
    """Load and prepare training data.
    
    Args:
        data_dir: Path to data directory
        indicators: Dict of {indicator_name: (category, direction)}
        parameters: Dict of {parameter_name: category}
        start_year: Minimum year
        end_year: Maximum year
        
    Returns:
        Tuple of (indicators_df, parameters_df)
    """
    loader = DataLoader(data_dir)
    logger.info("Loading training data...")
    indicators_df, parameters_df = loader.prepare_training_data(
        indicators, parameters, start_year, end_year
    )
    logger.info(
        f"Loaded {len(indicators_df)} indicator samples, "
        f"{len(parameters_df)} parameter samples"
    )
    return indicators_df, parameters_df


def train_models(
    indicators_df: pd.DataFrame,
    parameters_df: pd.DataFrame,
    model_type: str = "ridge",
    test_size: float = 0.2,
) -> ModelTrainer:
    """Train all indicator models.
    
    Args:
        indicators_df: DataFrame with indicators
        parameters_df: DataFrame with parameters
        model_type: 'ridge' or 'random_forest'
        test_size: Test set fraction
        
    Returns:
        Trained ModelTrainer instance
    """
    trainer = ModelTrainer(model_type=model_type)
    logger.info(f"Training models ({model_type})...")
    metrics = trainer.train_all_indicators(
        indicators_df, parameters_df, test_size=test_size
    )
    
    # Log summary statistics
    r2_scores = [m.get("r2_test", 0) for m in metrics.values()]
    if r2_scores:
        logger.info(
            f"Training complete. Average R² (test): {sum(r2_scores) / len(r2_scores):.3f}"
        )
    
    return trainer


def load_initial_state(
    data_dir: Path,
    indicators: Dict[str, tuple],
    year: int = 2025,
) -> pd.DataFrame:
    """Load initial state for simulation.
    
    Args:
        data_dir: Path to data directory
        indicators: Dict of {indicator_name: (category, direction)}
        year: Year to use as baseline
        
    Returns:
        DataFrame with initial state
    """
    loader = DataLoader(data_dir)
    logger.info(f"Loading initial state for year {year}...")
    
    state_data = None
    for ind_name, (category, _) in indicators.items():
        df = loader.load_indicator(ind_name, category)
        if not df.empty:
            available_years = sorted(df["year"].dropna().unique())
            target_years = [candidate for candidate in available_years if candidate <= year]
            if target_years:
                target_year = target_years[-1]
            else:
                target_year = available_years[-1]

            if target_year != year:
                logger.info(
                    f"Using {target_year} for {ind_name} because {year} is unavailable"
                )

            df = df[df["year"] == target_year].copy()
            df = loader.disaggregate_national_to_provincial(df)
            df = loader.disaggregate_regional_to_provincial(df)
            df = loader.standardize_to_province_level(df, interpolate_missing=True)
            df = df.pivot_table(
                index="area_code", values="value", aggfunc="mean"
            ).reset_index()
            df = df.rename(columns={"value": ind_name})
            
            if state_data is None:
                state_data = df
            else:
                state_data = state_data.merge(df, on="area_code", how="outer")
    
    if state_data is not None:
        value_columns = [col for col in state_data.columns if col != "area_code"]
        for col in value_columns:
            if pd.api.types.is_numeric_dtype(state_data[col]):
                state_data[col] = state_data[col].fillna(state_data[col].mean())
        state_data["year"] = year
        state_data = state_data[["area_code", "year"] +
                               [col for col in state_data.columns
                                if col not in ["area_code", "year"]]]
    
    return state_data


def simulate_scenario(
    trainer: ModelTrainer,
    initial_state: pd.DataFrame,
    parameters_scenarios: List[pd.DataFrame],
    years: List[int],
) -> pd.DataFrame:
    """Run simulation scenario.
    
    Args:
        trainer: Trained ModelTrainer
        initial_state: Starting state
        parameters_scenarios: List of parameter DataFrames for each year
        years: Years to simulate
        
    Returns:
        Simulation results
    """
    indicator_cols = [
        col for col in initial_state.columns 
        if col not in ["area_code", "year"]
    ]
    parameter_cols = [
        col for col in parameters_scenarios[0].columns 
        if col not in ["area_code", "year"]
    ]
    
    simulator = DigitalTwinSimulator(trainer, indicator_cols, parameter_cols)
    logger.info("Running simulation...")
    results = simulator.simulate_scenario(
        initial_state, parameters_scenarios, apply_rules=True
    )
    
    return results


def load_state_from_csv(state_path: Path) -> pd.DataFrame:
    """Load a simulation state from CSV and keep the latest year per province."""
    state = pd.read_csv(state_path)
    if "year" not in state.columns or "area_code" not in state.columns:
        raise ValueError(f"Invalid state file: {state_path}")

    latest_year = state["year"].max()
    latest_state = state[state["year"] == latest_year].copy()
    latest_state = normalize_wide_geography(latest_state)
    latest_state = latest_state[latest_state["year"] == latest_year].copy()
    return latest_state.reset_index(drop=True)


def load_parameter_scenarios_from_csv(parameters_path: Path) -> List[pd.DataFrame]:
    """Load yearly parameter scenarios from a CSV file.

    The CSV must contain area_code, year, and one column per policy lever.
    """
    parameters = pd.read_csv(parameters_path)
    if "year" not in parameters.columns or "area_code" not in parameters.columns:
        raise ValueError(f"Invalid parameter file: {parameters_path}")

    parameters = normalize_wide_geography(parameters)

    scenario_frames: List[pd.DataFrame] = []
    for year in sorted(parameters["year"].dropna().unique()):
        year_frame = parameters[parameters["year"] == year].copy()
        scenario_frames.append(year_frame.reset_index(drop=True))

    return scenario_frames


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="SERA Digital Twin Simulator"
    )
    
    parser.add_argument(
        "--mode",
        choices=["train", "simulate", "train-and-simulate"],
        default="train-and-simulate",
        help="Execution mode",
    )
    parser.add_argument(
        "--model-type",
        choices=["ridge", "random_forest"],
        default="ridge",
        help="ML model type",
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=2001,
        help="Start year for training data",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=2025,
        help="End year for training data",
    )
    parser.add_argument(
        "--baseline-year",
        type=int,
        default=2025,
        help="Year to use as baseline for simulation",
    )
    parser.add_argument(
        "--sim-years",
        type=int,
        default=5,
        help="Number of years to simulate forward",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("simulation_results.csv"),
        help="Output file for simulation results",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("twin_models.joblib"),
        help="Path to save or load trained models",
    )
    parser.add_argument(
        "--initial-state",
        type=Path,
        default=None,
        help="Optional CSV file with a prior simulation state to continue from",
    )
    parser.add_argument(
        "--parameters-file",
        type=Path,
        default=None,
        help="Optional CSV file with yearly policy allocations",
    )
    
    args = parser.parse_args()
    
    # Define indicators and parameters - ALL 24 TRAINED INDICATORS
    indicators = {
        # Core economic (7)
        "business_density": ("economic", 1),
        "gdp_per_capita": ("economic", 1),
        "income": ("demographic", 1),
        "poverty_rate": ("economic", -1),
        "self_employment": ("labor", 1),
        "unemployment_rate": ("labor", -1),
        "youth_employment": ("labor", 1),
        # Education (2)
        "completion_rates": ("education", 1),
        "school_enrollment": ("education", 1),
        # Health (3)
        "healthcare_spending_per_capita": ("healthcare_public_services", 1),
        "healthcare_worker_density": ("healthcare_public_services", 1),
        "life_expectancy": ("social_well_being", 1),
        # Innovation & Infrastructure (3)
        "digital_infrastructure": ("innovation_infrastructure", 1),
        "patents": ("innovation_infrastructure", 1),
        "transportation_access": ("innovation_infrastructure", 1),
        # Environment & Energy (6) - includes 4 new
        "air_quality": ("environment", -1),
        "carbon_emissions": ("energy_resources", -1),
        "green_urban_space_per_capita": ("environmental_quality", 1),
        "renewable_energy_percentage": ("energy_resources", 1),
        "sustainability": ("environment", 1),
        "water_quality": ("environmental_quality", 1),
        # Mobility & Social (2)
        "public_transportation_usage": ("transportation_mobility", 1),
        "traffic_congestion": ("transportation_mobility", -1),
        "crime_rate": ("social_well_being", -1),
    }
    
    # All 20 policy parameters
    parameters = {
        "income_tax_rate": "annual_parameters",
        "education_spending_allocation": "annual_parameters",
        "healthcare_spending_allocation": "annual_parameters",
        "agriculture_support_level": "annual_parameters",
        "corporate_tax_rate": "annual_parameters",
        "green_energy_environment_investment": "annual_parameters",
        "housing_urban_development_support": "annual_parameters",
        "immigration_policy_level": "annual_parameters",
        "infrastructure_investment_allocation": "annual_parameters",
        "manufacturing_incentives": "annual_parameters",
        "pension_retirement_spending": "annual_parameters",
        "property_wealth_tax_rate": "annual_parameters",
        "public_sector_wage_levels": "annual_parameters",
        "rd_innovation_incentives": "annual_parameters",
        "regulatory_burden_level": "annual_parameters",
        "small_business_support": "annual_parameters",
        "social_welfare_spending_allocation": "annual_parameters",
        "tourism_support_level": "annual_parameters",
        "vat_consumption_tax_rate": "annual_parameters",
        "environmental_regulations_strictness": "annual_parameters",
    }
    
    try:
        # Load and train
        if args.mode in ["train", "train-and-simulate"]:
            indicators_df, parameters_df = load_training_data(
                DATA_DIR, indicators, parameters, args.start_year, args.end_year
            )
            trainer = train_models(indicators_df, parameters_df, args.model_type)
            trainer.save(args.model_path)
            logger.info(f"Saved trained models to {args.model_path}")
        else:
            if not args.model_path.exists():
                raise FileNotFoundError(
                    f"Model file not found: {args.model_path}. Run --mode train first."
                )
            trainer = ModelTrainer.load(args.model_path)
            logger.info(f"Loaded trained models from {args.model_path}")
        
        # Simulate
        if args.mode in ["simulate", "train-and-simulate"]:
            if args.initial_state is not None:
                initial_state = load_state_from_csv(args.initial_state)
                logger.info(f"Loaded initial state from {args.initial_state}")
            else:
                initial_state = load_initial_state(DATA_DIR, indicators, args.baseline_year)
            
            if initial_state is None or initial_state.empty:
                logger.error("Could not load initial state")
                return
            
            if args.parameters_file is not None:
                parameters_scenarios = load_parameter_scenarios_from_csv(args.parameters_file)
                logger.info(f"Loaded parameter scenarios from {args.parameters_file}")
            else:
                # Create dummy parameter scenarios (you would load real scenario data)
                logger.info("No parameters file provided; using neutral dummy parameter scenarios")
                parameter_cols = list(parameters.keys())
                parameters_scenarios = []
                parameter_baselines = {
                    param: get_parameter_reference(param)[0] for param in parameter_cols
                }

                for year in range(args.sim_years):
                    params_df = initial_state[["area_code"]].copy()
                    params_df["year"] = args.baseline_year + year + 1
                    for param in parameter_cols:
                        params_df[param] = parameter_baselines[param]
                    parameters_scenarios.append(params_df)

            if len(parameters_scenarios) != args.sim_years:
                logger.warning(
                    f"Parameter file has {len(parameters_scenarios)} year(s), "
                    f"but --sim-years is {args.sim_years}. Using the parameter years as provided."
                )
            
            results = simulate_scenario(
                trainer,
                initial_state,
                parameters_scenarios,
                list(range(args.baseline_year, args.baseline_year + args.sim_years)),
            )
            
            # Save results
            results.to_csv(args.output, index=False)
            logger.info(f"Results saved to {args.output}")
            
            # Print summary
            print("\n=== Simulation Summary ===")
            for year in results["year"].unique():
                year_data = results[results["year"] == year]
                print(f"\nYear {year}: {len(year_data)} provinces")
                for col in indicators.keys():
                    if col in year_data.columns:
                        print(
                            f"  {col}: mean={year_data[col].mean():.2f}, "
                            f"std={year_data[col].std():.2f}"
                        )
    
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
