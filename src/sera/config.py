"""Configuration and constants for SERA downloader."""

from pathlib import Path
from typing import Final

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"

# ISTAT API configuration
ISTAT_BASE_URL: Final[str] = "https://esploradati.istat.it/SDMXWS/rest"
ISTAT_RATE_LIMIT_PER_MINUTE: Final[int] = 5  # CRITICAL: 5 requests/minute max
ISTAT_MIN_DELAY_SECONDS: Final[float] = 25.0  # Minimum 25 seconds between requests (~2.4 req/min)

# Output format
OUTPUT_FORMAT: Final[str] = "csv"
ENCODING: Final[str] = "utf-8"

# Indicators and sources
INDICATORS = {
    1: {
        "name": "Population",
        "italian_name": "Popolazione",
        "category": "Demographic",
        "source": "istat",
        "dataflow_id": "22_289_DF_DCIS_POPRES1_1",
        "data_type": "JAN",  # January snapshot
        "priority": 1,
    },
}

# Geographic coverage preference
GEOGRAPHIC_LEVELS = ["province", "region", "national"]

INDICATOR_CATEGORY_MAP: Final[dict[str, str]] = {
    "population": "demographic",
    "birth_rate": "demographic",
    "migration_flows": "demographic",
    "age_distribution": "demographic",
    "death_rate": "demographic",
    "income": "demographic",
    "gdp_per_capita": "economic",
    "poverty_rate": "economic",
    "gini_coefficient": "economic",
    "business_density": "economic",
    "fdi": "economic",
    "exports_imports": "economic",
    "unemployment_rate": "labor",
    "average_wages": "labor",
    "youth_employment": "labor",
    "self_employment": "labor",
    "skills_match": "labor",
    "life_expectancy": "social_well_being",
    "health_outcomes": "social_well_being",
    "crime_rate": "social_well_being",
    "social_cohesion": "social_well_being",
    "income_inequality": "social_well_being",
    "school_enrollment": "education",
    "completion_rates": "education",
    "stem_participation": "education",
    "adult_learning": "education",
    "r_and_d_spending": "innovation_infrastructure",
    "patents": "innovation_infrastructure",
    "startups": "innovation_infrastructure",
    "digital_infrastructure": "innovation_infrastructure",
    "transportation_access": "innovation_infrastructure",
    "air_quality": "environment",
    "sustainability": "environment",
    "green_space": "environment",
    "local_government_debt": "public_finance",
    "fiscal_balance": "public_finance",
    "tax_revenue_per_capita": "public_finance",
    "public_spending_efficiency": "public_finance",
    "infrastructure_investment": "public_finance",
    "house_prices": "housing",
    "construction_activity": "housing",
    "homeownership_rate": "housing",
    "housing_affordability_ratio": "housing",
    "renewable_energy_percentage": "energy_resources",
    "energy_consumption_per_capita": "energy_resources",
    "carbon_emissions": "energy_resources",
    "electricity_prices": "energy_resources",
    "public_transportation_usage": "transportation_mobility",
    "car_ownership_density": "transportation_mobility",
    "traffic_congestion": "transportation_mobility",
    "broadband_penetration": "transportation_mobility",
    "tourist_arrivals": "tourism_culture",
    "tourism_revenue": "tourism_culture",
    "cultural_sites": "tourism_culture",
    "cultural_spending": "tourism_culture",
    "museums_events": "tourism_culture",
    "healthcare_spending_per_capita": "healthcare_public_services",
    "hospital_beds_available": "healthcare_public_services",
    "healthcare_worker_density": "healthcare_public_services",
    "preventive_care_coverage": "healthcare_public_services",
    "agricultural_productivity": "business_environment",
    "waste_recycling_rate": "environmental_quality",
    "water_quality": "environmental_quality",
    "air_pollution": "environmental_quality",
    "green_urban_space_per_capita": "environmental_quality",
    "healthcare_capacity": "resilience_risk",
    "healthcare_facility_density": "service_accessibility",
    "urban_population_percentage": "urban_rural",
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

# Ensure data directory exists
DATA_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def get_indicator_data_dir(
    indicator: str,
) -> Path:
    """Get the data directory for a specific indicator.

    Args:
        indicator: Indicator name (e.g., 'population')

    Returns:
        Path to indicator-specific directory.

    Example:
        get_indicator_data_dir('population')
        # Returns: data/demographic/population/
    """
    category = INDICATOR_CATEGORY_MAP.get(indicator, "uncategorized")
    indicator_path = DATA_DIR / category / indicator
    indicator_path.mkdir(parents=True, exist_ok=True)
    return indicator_path
