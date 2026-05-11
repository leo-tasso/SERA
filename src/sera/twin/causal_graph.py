"""Causal dependency graph defining relationships between annual parameters and indicators."""

from pathlib import Path
from typing import Dict, Set, List, Tuple

import numpy as np
import pandas as pd

# Annual parameters (policy levers)
ANNUAL_PARAMETERS = {
    "income_tax_rate": "Income tax level",
    "corporate_tax_rate": "Business tax pressure",
    "property_wealth_tax_rate": "Real estate and asset taxation",
    "vat_consumption_tax_rate": "Consumption taxation level",
    "healthcare_spending_allocation": "% of budget to healthcare",
    "education_spending_allocation": "% of budget to education",
    "infrastructure_investment_allocation": "% of budget to infrastructure",
    "social_welfare_spending_allocation": "% of budget to welfare programs",
    "rd_innovation_incentives": "% of budget for R&D and tech",
    "green_energy_environment_investment": "% of budget for environment",
    "agriculture_support_level": "Subsidies for agriculture",
    "manufacturing_incentives": "Support for industrial sector",
    "tourism_support_level": "Tourism promotion spending",
    "small_business_support": "SME incentives and grants",
    "immigration_policy_level": "Openness to immigration (0-100)",
    "regulatory_burden_level": "Ease of doing business (0-100, lower=easier)",
    "public_sector_wage_levels": "Government employee compensation",
    "pension_retirement_spending": "% of budget for pensions",
    "housing_urban_development_support": "Housing and urban project incentives",
    "environmental_regulations_strictness": "Stringency of environmental rules (0-100)",
}

# Causal dependencies: parameter -> list of indicators it affects
PARAMETER_TO_INDICATORS: Dict[str, List[str]] = {
    "income_tax_rate": [
        "income",
        "gdp_per_capita",
        "business_density",
        "fdi",
        "unemployment_rate",
        "self_employment",
    ],
    "corporate_tax_rate": [
        "business_density",
        "fdi",
        "unemployment_rate",
        "average_wages",
        "gdp_per_capita",
        "startups",
    ],
    "property_wealth_tax_rate": [
        "house_prices",
        "construction_activity",
        "housing_affordability_ratio",
        "fdi",
    ],
    "vat_consumption_tax_rate": [
        "income",
        "poverty_rate",
        "gini_coefficient",
        "crime_rate",
    ],
    "healthcare_spending_allocation": [
        "life_expectancy",
        "health_outcomes",
        "preventive_care_coverage",
        "hospital_beds_available",
        "healthcare_worker_density",
        "healthcare_spending_per_capita",
    ],
    "education_spending_allocation": [
        "school_enrollment",
        "completion_rates",
        "stem_participation",
        "adult_learning",
        "youth_employment",
        "skills_match",
        "income",
        "unemployment_rate",
    ],
    "infrastructure_investment_allocation": [
        "transportation_access",
        "public_transportation_usage",
        "traffic_congestion",
        "business_density",
        "fdi",
        "gdp_per_capita",
        "broadband_penetration",
    ],
    "social_welfare_spending_allocation": [
        "poverty_rate",
        "gini_coefficient",
        "crime_rate",
        "social_cohesion",
        "health_outcomes",
        "life_expectancy",
    ],
    "rd_innovation_incentives": [
        "r_and_d_spending",
        "patents",
        "startups",
        "digital_infrastructure",
        "gdp_per_capita",
        "business_density",
    ],
    "green_energy_environment_investment": [
        "renewable_energy_percentage",
        "carbon_emissions",
        "air_quality",
        "green_space",
        "sustainability",
        "health_outcomes",
    ],
    "agriculture_support_level": [
        "agricultural_productivity",
        "employment_by_industry_breakdown",
        "income",
    ],
    "manufacturing_incentives": [
        "business_density",
        "employment_by_industry_breakdown",
        "average_wages",
        "gdp_per_capita",
        "fdi",
    ],
    "tourism_support_level": [
        "tourist_arrivals",
        "tourism_revenue",
        "employment_by_industry_breakdown",
        "average_wages",
        "cultural_sites",
    ],
    "small_business_support": [
        "business_density",
        "startups",
        "unemployment_rate",
        "self_employment",
        "average_wages",
    ],
    "immigration_policy_level": [
        "population",
        "birth_rate",
        "migration_flows",
        "labor_mobility",
        "unemployment_rate",
        "skills_match",
    ],
    "regulatory_burden_level": [
        "business_density",
        "startups",
        "fdi",
        "unemployment_rate",
        "ease_of_business_registration",
    ],
    "public_sector_wage_levels": [
        "average_wages",
        "income",
        "gini_coefficient",
        "cost_of_living_index",
    ],
    "pension_retirement_spending": [
        "income",
        "poverty_rate",
        "gini_coefficient",
        "social_cohesion",
        "health_outcomes",
    ],
    "housing_urban_development_support": [
        "house_prices",
        "construction_activity",
        "housing_affordability_ratio",
        "homeownership_rate",
        "urban_population_percentage",
    ],
    "environmental_regulations_strictness": [
        "air_quality",
        "water_quality",
        "carbon_emissions",
        "sustainability",
        "business_density",
        "manufacturing_incentives",
    ],
}

# Inverse mapping: indicator -> list of parameters that affect it
INDICATOR_TO_PARAMETERS: Dict[str, List[str]] = {}
for param, indicators in PARAMETER_TO_INDICATORS.items():
    for indicator in indicators:
        if indicator not in INDICATOR_TO_PARAMETERS:
            INDICATOR_TO_PARAMETERS[indicator] = []
        INDICATOR_TO_PARAMETERS[indicator].append(param)

# Inter-indicator dependencies (for propagating effects through the system)
INDICATOR_TO_INDICATORS: Dict[str, List[str]] = {
    # Education influences downstream outcomes
    "school_enrollment": ["completion_rates", "income", "unemployment_rate", "skills_match"],
    "completion_rates": ["income", "unemployment_rate", "skills_match", "youth_employment"],
    "stem_participation": ["patents", "startups", "r_and_d_spending", "income"],
    "adult_learning": ["skills_match", "unemployment_rate", "average_wages"],
    # Income influences poverty and inequality
    "income": ["poverty_rate", "gini_coefficient", "cost_of_living_index"],
    "average_wages": ["income", "poverty_rate", "gini_coefficient"],
    # Employment influences income and poverty
    "unemployment_rate": ["income", "poverty_rate", "crime_rate"],
    "youth_employment": ["income", "gini_coefficient"],
    # Health influences income and employment
    "life_expectancy": ["health_outcomes", "workforce productivity"],
    "health_outcomes": ["life_expectancy", "unemployment_rate"],
    # Business environment influences employment
    "business_density": ["unemployment_rate", "average_wages", "gdp_per_capita"],
    "startups": ["business_density", "unemployment_rate", "employment_by_industry_breakdown"],
    # Infrastructure and innovation influence business and employment
    "digital_infrastructure": ["business_density", "startups"],
    "transportation_access": ["business_density", "fdi"],
    "r_and_d_spending": ["patents", "startups", "gdp_per_capita"],
    # Energy and environment
    "renewable_energy_percentage": ["carbon_emissions", "electricity_prices"],
    "carbon_emissions": ["air_quality", "health_outcomes"],
    # Housing influences life decisions
    "house_prices": ["housing_affordability_ratio", "migration_flows"],
    "housing_affordability_ratio": ["migration_flows", "population"],
    # Population changes
    "population": ["birth_rate", "migration_flows", "gdp_per_capita"],
    "migration_flows": ["population", "age_distribution"],
}

# Bounds for indicators (logical constraints)
INDICATOR_BOUNDS: Dict[str, tuple] = {
    "population": (0, float("inf")),
    "birth_rate": (0, 100),
    "death_rate": (0, 100),
    "unemployment_rate": (0, float("inf")),
    "poverty_rate": (0, 100),
    "gini_coefficient": (0, 1),
    "renewable_energy_percentage": (0, 100),
    "life_expectancy": (0, 150),
    "air_quality": (0, 500),  # PM2.5 micrograms/m3
    "housing_affordability_ratio": (0, float("inf")),
    "waste_recycling_rate": (0, 100),
    "carbon_emissions": (0, float("inf")),
    "school_enrollment": (0, float("inf")),
    "income": (0, float("inf")),
    "completion_rates": (0, 100),
    "income_tax_rate": (0, 100),
    "corporate_tax_rate": (0, 100),
    "property_wealth_tax_rate": (0, 100),
    "vat_consumption_tax_rate": (0, 100),
    "healthcare_spending_allocation": (0, 100),
    "education_spending_allocation": (0, 100),
    "infrastructure_investment_allocation": (0, 100),
    "social_welfare_spending_allocation": (0, 100),
    "rd_innovation_incentives": (0, 100),
    "green_energy_environment_investment": (0, 100),
    "agriculture_support_level": (0, 100),
    "manufacturing_incentives": (0, 100),
    "tourism_support_level": (0, 100),
    "small_business_support": (0, 100),
    "immigration_policy_level": (0, 100),
    "regulatory_burden_level": (0, 100),
    "public_sector_wage_levels": (0, 100),
    "pension_retirement_spending": (0, 100),
    "housing_urban_development_support": (0, 100),
    "environmental_regulations_strictness": (0, 100),
}

# Direction of effect: positive=higher parameter is better, negative=lower parameter is better
PARAMETER_EFFECT_DIRECTION: Dict[str, int] = {
    "income_tax_rate": -1,
    "corporate_tax_rate": -1,
    "property_wealth_tax_rate": -1,
    "vat_consumption_tax_rate": -1,
    "healthcare_spending_allocation": 1,
    "education_spending_allocation": 1,
    "infrastructure_investment_allocation": 1,
    "social_welfare_spending_allocation": 1,
    "rd_innovation_incentives": 1,
    "green_energy_environment_investment": 1,
    "agriculture_support_level": 1,
    "manufacturing_incentives": 1,
    "tourism_support_level": 1,
    "small_business_support": 1,
    "immigration_policy_level": 1,
    "regulatory_burden_level": -1,
    "public_sector_wage_levels": 1,
    "pension_retirement_spending": 1,
    "housing_urban_development_support": 1,
    "environmental_regulations_strictness": 1,
}

# Direction of effect for indicators: positive=higher is better, negative=lower is better
INDICATOR_EFFECT_DIRECTION: Dict[str, int] = {
    "population": 1,
    "birth_rate": 1,
    "death_rate": -1,
    "migration_flows": 1,
    "age_distribution": 0,  # Neutral
    "income": 1,
    "gdp_per_capita": 1,
    "poverty_rate": -1,
    "gini_coefficient": -1,
    "business_density": 1,
    "fdi": 1,
    "exports_imports": 1,
    "unemployment_rate": -1,
    "average_wages": 1,
    "youth_employment": 1,
    "self_employment": 1,
    "skills_match": 1,
    "life_expectancy": 1,
    "health_outcomes": 1,
    "crime_rate": -1,
    "social_cohesion": 1,
    "school_enrollment": 1,
    "completion_rates": 1,
    "stem_participation": 1,
    "adult_learning": 1,
    "r_and_d_spending": 1,
    "patents": 1,
    "startups": 1,
    "digital_infrastructure": 1,
    "transportation_access": 1,
    "air_quality": -1,
    "sustainability": 1,
    "green_space": 1,
    "local_government_debt": -1,
    "fiscal_balance": 1,
    "tax_revenue_per_capita": 1,
    "public_spending_efficiency": 1,
    "infrastructure_investment": 1,
    "house_prices": 1,
    "construction_activity": 1,
    "homeownership_rate": 1,
    "housing_affordability_ratio": 1,
    "renewable_energy_percentage": 1,
    "energy_consumption_per_capita": 1,
    "carbon_emissions": -1,
    "electricity_prices": -1,
    "public_transportation_usage": 1,
    "car_ownership_density": 0,  # Neutral
    "traffic_congestion": -1,
    "broadband_penetration": 1,
    "tourist_arrivals": 1,
    "tourism_revenue": 1,
    "cultural_sites": 1,
    "cultural_spending": 1,
    "museums_events": 1,
    "healthcare_spending_per_capita": 1,
    "hospital_beds_available": 1,
    "healthcare_worker_density": 1,
    "preventive_care_coverage": 1,
    "agricultural_productivity": 1,
    "waste_recycling_rate": 1,
    "water_quality": 1,
    "air_pollution": -1,
    "green_urban_space_per_capita": 1,
    "cost_of_living_index": -1,
    "income_inequality": -1,
}


def get_affected_indicators(parameter: str) -> List[str]:
    """Get all indicators affected by a parameter."""
    return PARAMETER_TO_INDICATORS.get(parameter, [])


def get_influencing_parameters(indicator: str) -> List[str]:
    """Get all parameters that influence an indicator."""
    return INDICATOR_TO_PARAMETERS.get(indicator, [])


def get_dependent_indicators(indicator: str) -> List[str]:
    """Get all indicators that depend on this indicator."""
    return INDICATOR_TO_INDICATORS.get(indicator, [])


def get_indicator_bounds(indicator: str) -> tuple:
    """Get the valid bounds for an indicator."""
    return INDICATOR_BOUNDS.get(indicator, (0, float("inf")))


def get_parameter_effect_direction(parameter: str) -> int:
    """Get the effect direction of a parameter (-1, 0, or 1)."""
    return PARAMETER_EFFECT_DIRECTION.get(parameter, 0)


def get_indicator_effect_direction(indicator: str) -> int:
    """Get the effect direction of an indicator (-1, 0, or 1)."""
    return INDICATOR_EFFECT_DIRECTION.get(indicator, 0)


_PARAMETER_REFERENCE_CACHE: Dict[str, Tuple[float, float]] = {}


def _get_parameter_data_path(parameter: str) -> Path:
    repo_root = Path(__file__).resolve().parents[3]
    return repo_root / "data" / "annual_parameters" / parameter


def get_parameter_reference(parameter: str) -> Tuple[float, float]:
    """Get a robust baseline and scale for a parameter from historical data.

    Returns:
        Tuple of (baseline, scale). The baseline is the historical median and the
        scale is a robust spread estimate used to normalize scenario inputs.
    """
    cached = _PARAMETER_REFERENCE_CACHE.get(parameter)
    if cached is not None:
        return cached

    baseline = 50.0
    scale = 10.0

    data_path = _get_parameter_data_path(parameter)
    matching_files = sorted(data_path.glob(f"{parameter}_raw_*.csv"))
    if matching_files:
        try:
            frame = pd.read_csv(matching_files[0])
            value_columns = [
                col for col in frame.columns if col not in {"area_code", "year"}
            ]
            if value_columns:
                series = pd.to_numeric(frame[value_columns[0]], errors="coerce").dropna()
                if not series.empty:
                    baseline = float(series.median())
                    q75 = float(series.quantile(0.75))
                    q25 = float(series.quantile(0.25))
                    scale = max(
                        q75 - q25,
                        float(series.std(ddof=0)),
                        abs(baseline) * 0.1,
                        1.0,
                    )
        except Exception:
            pass

    _PARAMETER_REFERENCE_CACHE[parameter] = (baseline, scale)
    return baseline, scale


def get_parameter_signal(parameter: str, value: float) -> float:
    """Normalize a parameter value into a bounded causal signal."""
    baseline, scale = get_parameter_reference(parameter)
    normalized = (float(value) - baseline) / max(scale, 1e-6)
    return float(np.tanh(normalized))
