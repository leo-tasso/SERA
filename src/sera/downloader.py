#!/usr/bin/env python3
"""SERA Data Downloader - Main entry point."""

import argparse
from pathlib import Path

from sera.downloaders.annual_parameters.agriculture_support_level import (
    AgricultureSupportLevelDownloader,
)
from sera.downloaders.annual_parameters.corporate_tax_rate import CorporateTaxRateDownloader
from sera.downloaders.annual_parameters.education_spending_allocation import (
    EducationSpendingAllocationDownloader,
)
from sera.downloaders.annual_parameters.environmental_regulations_strictness import (
    EnvironmentalRegulationsStrictnessDownloader,
)
from sera.downloaders.annual_parameters.green_energy_environment_investment import (
    GreenEnergyEnvironmentInvestmentDownloader,
)
from sera.downloaders.annual_parameters.healthcare_spending_allocation import (
    HealthcareSpendingAllocationDownloader,
)
from sera.downloaders.annual_parameters.housing_urban_development_support import (
    HousingUrbanDevelopmentSupportDownloader,
)
from sera.downloaders.annual_parameters.immigration_policy_level import (
    ImmigrationPolicyLevelDownloader,
)
from sera.downloaders.annual_parameters.income_tax_rate import IncomeTaxRateDownloader
from sera.downloaders.annual_parameters.infrastructure_investment_allocation import (
    InfrastructureInvestmentAllocationDownloader,
)
from sera.downloaders.annual_parameters.manufacturing_incentives import (
    ManufacturingIncentivesDownloader,
)
from sera.downloaders.annual_parameters.pension_retirement_spending import (
    PensionRetirementSpendingDownloader,
)
from sera.downloaders.annual_parameters.property_wealth_tax_rate import (
    PropertyWealthTaxRateDownloader,
)
from sera.downloaders.annual_parameters.public_sector_wage_levels import (
    PublicSectorWageLevelsDownloader,
)
from sera.downloaders.annual_parameters.rd_innovation_incentives import (
    RdInnovationIncentivesDownloader,
)
from sera.downloaders.annual_parameters.regulatory_burden_level import (
    RegulatoryBurdenLevelDownloader,
)
from sera.downloaders.annual_parameters.small_business_support import SmallBusinessSupportDownloader
from sera.downloaders.annual_parameters.social_welfare_spending_allocation import (
    SocialWelfareSpendingAllocationDownloader,
)
from sera.downloaders.annual_parameters.tourism_support_level import TourismSupportLevelDownloader
from sera.downloaders.annual_parameters.vat_consumption_tax_rate import (
    VatConsumptionTaxRateDownloader,
)
from sera.downloaders.business_environment.agricultural_productivity import (
    AgriculturalProductivityDownloader,
)
from sera.downloaders.demographic.age_distribution import AgeDistributionDownloader
from sera.downloaders.demographic.birth_rate import BirthRateDownloader
from sera.downloaders.demographic.death_rate import DeathRateDownloader
from sera.downloaders.demographic.income import IncomeDownloader
from sera.downloaders.demographic.migration_flows import MigrationFlowsDownloader
from sera.downloaders.demographic.population import PopulationDownloader
from sera.downloaders.economic.business_density import BusinessDensityDownloader
from sera.downloaders.economic.exports_imports import ExportsImportsDownloader
from sera.downloaders.economic.fdi import FdiDownloader
from sera.downloaders.economic.gdp_per_capita import GdpPerCapitaDownloader
from sera.downloaders.economic.gini_coefficient import GiniCoefficientDownloader
from sera.downloaders.economic.poverty_rate import PovertyRateDownloader
from sera.downloaders.education.adult_learning import AdultLearningDownloader
from sera.downloaders.education.completion_rates import CompletionRatesDownloader
from sera.downloaders.education.school_enrollment import SchoolEnrollmentDownloader
from sera.downloaders.education.stem_participation import StemParticipationDownloader
from sera.downloaders.energy_resources.carbon_emissions import CarbonEmissionsDownloader
from sera.downloaders.energy_resources.electricity_prices import ElectricityPricesDownloader
from sera.downloaders.energy_resources.energy_consumption_per_capita import (
    EnergyConsumptionPerCapitaDownloader,
)
from sera.downloaders.energy_resources.renewable_energy_percentage import (
    RenewableEnergyPercentageDownloader,
)
from sera.downloaders.environment.air_quality import AirQualityDownloader
from sera.downloaders.environment.green_space import GreenSpaceDownloader
from sera.downloaders.environment.sustainability import SustainabilityDownloader
from sera.downloaders.environmental_quality.air_pollution import AirPollutionDownloader
from sera.downloaders.environmental_quality.green_urban_space_per_capita import (
    GreenUrbanSpacePerCapitaDownloader,
)
from sera.downloaders.environmental_quality.waste_recycling_rate import WasteRecyclingRateDownloader
from sera.downloaders.environmental_quality.water_quality import WaterQualityDownloader
from sera.downloaders.healthcare_public_services.healthcare_spending_per_capita import (
    HealthcareSpendingPerCapitaDownloader,
)
from sera.downloaders.healthcare_public_services.healthcare_worker_density import (
    HealthcareWorkerDensityDownloader,
)
from sera.downloaders.healthcare_public_services.hospital_beds_available import (
    HospitalBedsAvailableDownloader,
)
from sera.downloaders.healthcare_public_services.preventive_care_coverage import (
    PreventiveCareCoverageDownloader,
)
from sera.downloaders.housing.construction_activity import ConstructionActivityDownloader
from sera.downloaders.housing.homeownership_rate import HomeownershipRateDownloader
from sera.downloaders.housing.house_prices import HousePricesDownloader
from sera.downloaders.housing.housing_affordability_ratio import HousingAffordabilityRatioDownloader
from sera.downloaders.innovation_infrastructure.digital_infrastructure import (
    DigitalInfrastructureDownloader,
)
from sera.downloaders.innovation_infrastructure.patents import PatentsDownloader
from sera.downloaders.innovation_infrastructure.r_and_d_spending import RAndDSpendingDownloader
from sera.downloaders.innovation_infrastructure.startups import StartupsDownloader
from sera.downloaders.innovation_infrastructure.transportation_access import (
    TransportationAccessDownloader,
)
from sera.downloaders.labor.average_wages import AverageWagesDownloader
from sera.downloaders.labor.self_employment import SelfEmploymentDownloader
from sera.downloaders.labor.skills_match import SkillsMatchDownloader
from sera.downloaders.labor.unemployment_rate import UnemploymentRateDownloader
from sera.downloaders.labor.youth_employment import YouthEmploymentDownloader
from sera.downloaders.public_finance.fiscal_balance import FiscalBalanceDownloader
from sera.downloaders.public_finance.infrastructure_investment import (
    InfrastructureInvestmentDownloader,
)
from sera.downloaders.public_finance.local_government_debt import LocalGovernmentDebtDownloader
from sera.downloaders.public_finance.public_spending_efficiency import (
    PublicSpendingEfficiencyDownloader,
)
from sera.downloaders.public_finance.tax_revenue_per_capita import TaxRevenuePerCapitaDownloader
from sera.downloaders.resilience_risk.healthcare_capacity import HealthcareCapacityDownloader
from sera.downloaders.service_accessibility.healthcare_facility_density import (
    HealthcareFacilityDensityDownloader,
)
from sera.downloaders.social_well_being.crime_rate import CrimeRateDownloader
from sera.downloaders.social_well_being.health_outcomes import HealthOutcomesDownloader
from sera.downloaders.social_well_being.income_inequality import IncomeInequalityDownloader
from sera.downloaders.social_well_being.life_expectancy import LifeExpectancyDownloader
from sera.downloaders.social_well_being.social_cohesion import SocialCohesionDownloader
from sera.downloaders.tourism_culture.cultural_sites import CulturalSitesDownloader
from sera.downloaders.tourism_culture.cultural_spending import CulturalSpendingDownloader
from sera.downloaders.tourism_culture.museums_events import MuseumsEventsDownloader
from sera.downloaders.tourism_culture.tourism_revenue import TourismRevenueDownloader
from sera.downloaders.tourism_culture.tourist_arrivals import TouristArrivalsDownloader
from sera.downloaders.transportation_mobility.broadband_penetration import (
    BroadbandPenetrationDownloader,
)
from sera.downloaders.transportation_mobility.car_ownership_density import (
    CarOwnershipDensityDownloader,
)
from sera.downloaders.transportation_mobility.public_transportation_usage import (
    PublicTransportationUsageDownloader,
)
from sera.downloaders.transportation_mobility.traffic_congestion import TrafficCongestionDownloader
from sera.downloaders.urban_rural.urban_population_percentage import (
    UrbanPopulationPercentageDownloader,
)


def main():
    """Main downloader orchestrator."""
    indicator_runners = {
        "population": lambda output_path, start_year, end_year: PopulationDownloader().save_population_csv(
            output_path=output_path,
            start_year=start_year,
            end_year=end_year,
        ),
        "birth_rate": lambda output_path, start_year, end_year: BirthRateDownloader().save_birth_rate_csv(
            output_path=output_path,
            start_year=start_year,
            end_year=end_year,
        ),
        "migration_flows": lambda output_path, start_year, end_year: MigrationFlowsDownloader().save_migration_flows_csv(
            output_path=output_path,
            start_year=start_year,
            end_year=end_year,
        ),
        "age_distribution": lambda output_path, start_year, end_year: AgeDistributionDownloader().save_age_distribution_csv(
            output_path=output_path,
            start_year=start_year,
            end_year=end_year,
        ),
        "death_rate": lambda output_path, start_year, end_year: DeathRateDownloader().save_death_rate_csv(
            output_path=output_path,
            start_year=start_year,
            end_year=end_year,
        ),
        "income": lambda output_path, start_year, end_year: IncomeDownloader().save_income_csv(
            output_path=output_path,
            start_year=start_year,
            end_year=end_year,
        ),
        "gdp_per_capita": lambda output_path, start_year, end_year: GdpPerCapitaDownloader().save_gdp_per_capita_csv(
            output_path=output_path,
            start_year=start_year,
            end_year=end_year,
        ),
        "poverty_rate": lambda output_path, start_year, end_year: PovertyRateDownloader().save_poverty_rate_csv(
            output_path=output_path,
            start_year=start_year,
            end_year=end_year,
        ),
        "gini_coefficient": lambda output_path, start_year, end_year: GiniCoefficientDownloader().save_gini_coefficient_csv(
            output_path=output_path,
            start_year=start_year,
            end_year=end_year,
        ),
        "business_density": lambda output_path, start_year, end_year: BusinessDensityDownloader().save_business_density_csv(
            output_path=output_path,
            start_year=start_year,
            end_year=end_year,
        ),
        "fdi": lambda output_path, start_year, end_year: FdiDownloader().save_fdi_csv(
            output_path=output_path,
            start_year=start_year,
            end_year=end_year,
        ),
        "exports_imports": lambda output_path, start_year, end_year: ExportsImportsDownloader().save_exports_imports_csv(
            output_path=output_path,
            start_year=start_year,
            end_year=end_year,
        ),
        "unemployment_rate": lambda output_path, start_year, end_year: UnemploymentRateDownloader().save_unemployment_rate_csv(
            output_path=output_path,
            start_year=start_year,
            end_year=end_year,
        ),
        "average_wages": lambda output_path, start_year, end_year: AverageWagesDownloader().save_average_wages_csv(
            output_path=output_path,
            start_year=start_year,
            end_year=end_year,
        ),
        "youth_employment": lambda output_path, start_year, end_year: YouthEmploymentDownloader().save_youth_employment_csv(
            output_path=output_path,
            start_year=start_year,
            end_year=end_year,
        ),
        "self_employment": lambda output_path, start_year, end_year: SelfEmploymentDownloader().save_self_employment_csv(
            output_path=output_path,
            start_year=start_year,
            end_year=end_year,
        ),
        "skills_match": lambda output_path, start_year, end_year: SkillsMatchDownloader().save_skills_match_csv(
            output_path=output_path,
            start_year=start_year,
            end_year=end_year,
        ),
        "income_tax_rate": lambda output_path, start_year, end_year: IncomeTaxRateDownloader().save_income_tax_rate_csv(
            output_path=output_path,
            start_year=start_year,
            end_year=end_year,
        ),
        "corporate_tax_rate": lambda output_path, start_year, end_year: CorporateTaxRateDownloader().save_corporate_tax_rate_csv(
            output_path=output_path,
            start_year=start_year,
            end_year=end_year,
        ),
        "property_wealth_tax_rate": lambda output_path, start_year, end_year: PropertyWealthTaxRateDownloader().save_property_wealth_tax_rate_csv(
            output_path=output_path,
            start_year=start_year,
            end_year=end_year,
        ),
        "vat_consumption_tax_rate": lambda output_path, start_year, end_year: VatConsumptionTaxRateDownloader().save_vat_consumption_tax_rate_csv(
            output_path=output_path,
            start_year=start_year,
            end_year=end_year,
        ),
        "healthcare_spending_allocation": lambda output_path, start_year, end_year: HealthcareSpendingAllocationDownloader().save_healthcare_spending_allocation_csv(
            output_path=output_path,
            start_year=start_year,
            end_year=end_year,
        ),
        "education_spending_allocation": lambda output_path, start_year, end_year: EducationSpendingAllocationDownloader().save_education_spending_allocation_csv(
            output_path=output_path,
            start_year=start_year,
            end_year=end_year,
        ),
        "infrastructure_investment_allocation": lambda output_path, start_year, end_year: InfrastructureInvestmentAllocationDownloader().save_infrastructure_investment_allocation_csv(
            output_path=output_path,
            start_year=start_year,
            end_year=end_year,
        ),
        "social_welfare_spending_allocation": lambda output_path, start_year, end_year: SocialWelfareSpendingAllocationDownloader().save_social_welfare_spending_allocation_csv(
            output_path=output_path,
            start_year=start_year,
            end_year=end_year,
        ),
        "rd_innovation_incentives": lambda output_path, start_year, end_year: RdInnovationIncentivesDownloader().save_rd_innovation_incentives_csv(
            output_path=output_path,
            start_year=start_year,
            end_year=end_year,
        ),
        "green_energy_environment_investment": lambda output_path, start_year, end_year: GreenEnergyEnvironmentInvestmentDownloader().save_green_energy_environment_investment_csv(
            output_path=output_path,
            start_year=start_year,
            end_year=end_year,
        ),
        "agriculture_support_level": lambda output_path, start_year, end_year: AgricultureSupportLevelDownloader().save_agriculture_support_level_csv(
            output_path=output_path,
            start_year=start_year,
            end_year=end_year,
        ),
        "manufacturing_incentives": lambda output_path, start_year, end_year: ManufacturingIncentivesDownloader().save_manufacturing_incentives_csv(
            output_path=output_path,
            start_year=start_year,
            end_year=end_year,
        ),
        "tourism_support_level": lambda output_path, start_year, end_year: TourismSupportLevelDownloader().save_tourism_support_level_csv(
            output_path=output_path,
            start_year=start_year,
            end_year=end_year,
        ),
        "small_business_support": lambda output_path, start_year, end_year: SmallBusinessSupportDownloader().save_small_business_support_csv(
            output_path=output_path,
            start_year=start_year,
            end_year=end_year,
        ),
        "immigration_policy_level": lambda output_path, start_year, end_year: ImmigrationPolicyLevelDownloader().save_immigration_policy_level_csv(
            output_path=output_path,
            start_year=start_year,
            end_year=end_year,
        ),
        "regulatory_burden_level": lambda output_path, start_year, end_year: RegulatoryBurdenLevelDownloader().save_regulatory_burden_level_csv(
            output_path=output_path,
            start_year=start_year,
            end_year=end_year,
        ),
        "public_sector_wage_levels": lambda output_path, start_year, end_year: PublicSectorWageLevelsDownloader().save_public_sector_wage_levels_csv(
            output_path=output_path,
            start_year=start_year,
            end_year=end_year,
        ),
        "pension_retirement_spending": lambda output_path, start_year, end_year: PensionRetirementSpendingDownloader().save_pension_retirement_spending_csv(
            output_path=output_path,
            start_year=start_year,
            end_year=end_year,
        ),
        "housing_urban_development_support": lambda output_path, start_year, end_year: HousingUrbanDevelopmentSupportDownloader().save_housing_urban_development_support_csv(
            output_path=output_path,
            start_year=start_year,
            end_year=end_year,
        ),
        "environmental_regulations_strictness": lambda output_path, start_year, end_year: EnvironmentalRegulationsStrictnessDownloader().save_environmental_regulations_strictness_csv(
            output_path=output_path,
            start_year=start_year,
            end_year=end_year,
        ),
        "life_expectancy": lambda output_path, start_year, end_year: LifeExpectancyDownloader().save_life_expectancy_csv(
            output_path=output_path,
            start_year=start_year,
            end_year=end_year,
        ),
        "health_outcomes": lambda output_path, start_year, end_year: HealthOutcomesDownloader().save_health_outcomes_csv(
            output_path=output_path,
            start_year=start_year,
            end_year=end_year,
        ),
        "crime_rate": lambda output_path, start_year, end_year: CrimeRateDownloader().save_crime_rate_csv(
            output_path=output_path,
            start_year=start_year,
            end_year=end_year,
        ),
        "social_cohesion": lambda output_path, start_year, end_year: SocialCohesionDownloader().save_social_cohesion_csv(
            output_path=output_path,
            start_year=start_year,
            end_year=end_year,
        ),
        "income_inequality": lambda output_path, start_year, end_year: IncomeInequalityDownloader().save_income_inequality_csv(
            output_path=output_path,
            start_year=start_year,
            end_year=end_year,
        ),
        "school_enrollment": lambda output_path, start_year, end_year: SchoolEnrollmentDownloader().save_school_enrollment_csv(
            output_path=output_path,
            start_year=start_year,
            end_year=end_year,
        ),
        "completion_rates": lambda output_path, start_year, end_year: CompletionRatesDownloader().save_completion_rates_csv(
            output_path=output_path,
            start_year=start_year,
            end_year=end_year,
        ),
        "stem_participation": lambda output_path, start_year, end_year: StemParticipationDownloader().save_stem_participation_csv(
            output_path=output_path,
            start_year=start_year,
            end_year=end_year,
        ),
        "adult_learning": lambda output_path, start_year, end_year: AdultLearningDownloader().save_adult_learning_csv(
            output_path=output_path,
            start_year=start_year,
            end_year=end_year,
        ),
        "r_and_d_spending": lambda output_path, start_year, end_year: RAndDSpendingDownloader().save_r_and_d_spending_csv(
            output_path=output_path,
            start_year=start_year,
            end_year=end_year,
        ),
        "patents": lambda output_path, start_year, end_year: PatentsDownloader().save_patents_csv(
            output_path=output_path,
            start_year=start_year,
            end_year=end_year,
        ),
        "startups": lambda output_path, start_year, end_year: StartupsDownloader().save_startups_csv(
            output_path=output_path,
            start_year=start_year,
            end_year=end_year,
        ),
        "digital_infrastructure": lambda output_path, start_year, end_year: DigitalInfrastructureDownloader().save_digital_infrastructure_csv(
            output_path=output_path,
            start_year=start_year,
            end_year=end_year,
        ),
        "transportation_access": lambda output_path, start_year, end_year: TransportationAccessDownloader().save_transportation_access_csv(
            output_path=output_path,
            start_year=start_year,
            end_year=end_year,
        ),
        "air_quality": lambda output_path, start_year, end_year: AirQualityDownloader().save_air_quality_csv(
            output_path=output_path,
            start_year=start_year,
            end_year=end_year,
        ),
        "sustainability": lambda output_path, start_year, end_year: SustainabilityDownloader().save_sustainability_csv(
            output_path=output_path,
            start_year=start_year,
            end_year=end_year,
        ),
        "green_space": lambda output_path, start_year, end_year: GreenSpaceDownloader().save_green_space_csv(
            output_path=output_path,
            start_year=start_year,
            end_year=end_year,
        ),
        "local_government_debt": lambda output_path, start_year, end_year: LocalGovernmentDebtDownloader().save_local_government_debt_csv(
            output_path=output_path,
            start_year=start_year,
            end_year=end_year,
        ),
        "fiscal_balance": lambda output_path, start_year, end_year: FiscalBalanceDownloader().save_fiscal_balance_csv(
            output_path=output_path,
            start_year=start_year,
            end_year=end_year,
        ),
        "tax_revenue_per_capita": lambda output_path, start_year, end_year: TaxRevenuePerCapitaDownloader().save_tax_revenue_per_capita_csv(
            output_path=output_path,
            start_year=start_year,
            end_year=end_year,
        ),
        "public_spending_efficiency": lambda output_path, start_year, end_year: PublicSpendingEfficiencyDownloader().save_public_spending_efficiency_csv(
            output_path=output_path,
            start_year=start_year,
            end_year=end_year,
        ),
        "infrastructure_investment": lambda output_path, start_year, end_year: InfrastructureInvestmentDownloader().save_infrastructure_investment_csv(
            output_path=output_path,
            start_year=start_year,
            end_year=end_year,
        ),
        "house_prices": lambda output_path, start_year, end_year: HousePricesDownloader().save_house_prices_csv(
            output_path=output_path,
            start_year=start_year,
            end_year=end_year,
        ),
        "construction_activity": lambda output_path, start_year, end_year: ConstructionActivityDownloader().save_construction_activity_csv(
            output_path=output_path,
            start_year=start_year,
            end_year=end_year,
        ),
        "homeownership_rate": lambda output_path, start_year, end_year: HomeownershipRateDownloader().save_homeownership_rate_csv(
            output_path=output_path,
            start_year=start_year,
            end_year=end_year,
        ),
        "housing_affordability_ratio": lambda output_path, start_year, end_year: HousingAffordabilityRatioDownloader().save_housing_affordability_ratio_csv(
            output_path=output_path,
            start_year=start_year,
            end_year=end_year,
        ),
        "renewable_energy_percentage": lambda output_path, start_year, end_year: RenewableEnergyPercentageDownloader().save_renewable_energy_percentage_csv(
            output_path=output_path,
            start_year=start_year,
            end_year=end_year,
        ),
        "energy_consumption_per_capita": lambda output_path, start_year, end_year: EnergyConsumptionPerCapitaDownloader().save_energy_consumption_per_capita_csv(
            output_path=output_path,
            start_year=start_year,
            end_year=end_year,
        ),
        "carbon_emissions": lambda output_path, start_year, end_year: CarbonEmissionsDownloader().save_carbon_emissions_csv(
            output_path=output_path,
            start_year=start_year,
            end_year=end_year,
        ),
        "electricity_prices": lambda output_path, start_year, end_year: ElectricityPricesDownloader().save_electricity_prices_csv(
            output_path=output_path,
            start_year=start_year,
            end_year=end_year,
        ),
        "public_transportation_usage": lambda output_path, start_year, end_year: PublicTransportationUsageDownloader().save_public_transportation_usage_csv(
            output_path=output_path,
            start_year=start_year,
            end_year=end_year,
        ),
        "car_ownership_density": lambda output_path, start_year, end_year: CarOwnershipDensityDownloader().save_car_ownership_density_csv(
            output_path=output_path,
            start_year=start_year,
            end_year=end_year,
        ),
        "traffic_congestion": lambda output_path, start_year, end_year: TrafficCongestionDownloader().save_traffic_congestion_csv(
            output_path=output_path,
            start_year=start_year,
            end_year=end_year,
        ),
        "broadband_penetration": lambda output_path, start_year, end_year: BroadbandPenetrationDownloader().save_broadband_penetration_csv(
            output_path=output_path,
            start_year=start_year,
            end_year=end_year,
        ),
        "tourist_arrivals": lambda output_path, start_year, end_year: TouristArrivalsDownloader().save_tourist_arrivals_csv(
            output_path=output_path,
            start_year=start_year,
            end_year=end_year,
        ),
        "tourism_revenue": lambda output_path, start_year, end_year: TourismRevenueDownloader().save_tourism_revenue_csv(
            output_path=output_path,
            start_year=start_year,
            end_year=end_year,
        ),
        "cultural_sites": lambda output_path, start_year, end_year: CulturalSitesDownloader().save_cultural_sites_csv(
            output_path=output_path,
            start_year=start_year,
            end_year=end_year,
        ),
        "cultural_spending": lambda output_path, start_year, end_year: CulturalSpendingDownloader().save_cultural_spending_csv(
            output_path=output_path,
            start_year=start_year,
            end_year=end_year,
        ),
        "museums_events": lambda output_path, start_year, end_year: MuseumsEventsDownloader().save_museums_events_csv(
            output_path=output_path,
            start_year=start_year,
            end_year=end_year,
        ),
        "healthcare_spending_per_capita": lambda output_path, start_year, end_year: HealthcareSpendingPerCapitaDownloader().save_healthcare_spending_per_capita_csv(
            output_path=output_path,
            start_year=start_year,
            end_year=end_year,
        ),
        "hospital_beds_available": lambda output_path, start_year, end_year: HospitalBedsAvailableDownloader().save_hospital_beds_available_csv(
            output_path=output_path,
            start_year=start_year,
            end_year=end_year,
        ),
        "healthcare_worker_density": lambda output_path, start_year, end_year: HealthcareWorkerDensityDownloader().save_healthcare_worker_density_csv(
            output_path=output_path,
            start_year=start_year,
            end_year=end_year,
        ),
        "preventive_care_coverage": lambda output_path, start_year, end_year: PreventiveCareCoverageDownloader().save_preventive_care_coverage_csv(
            output_path=output_path,
            start_year=start_year,
            end_year=end_year,
        ),
        "agricultural_productivity": lambda output_path, start_year, end_year: AgriculturalProductivityDownloader().save_agricultural_productivity_csv(
            output_path=output_path,
            start_year=start_year,
            end_year=end_year,
        ),
        "waste_recycling_rate": lambda output_path, start_year, end_year: WasteRecyclingRateDownloader().save_waste_recycling_rate_csv(
            output_path=output_path,
            start_year=start_year,
            end_year=end_year,
        ),
        "water_quality": lambda output_path, start_year, end_year: WaterQualityDownloader().save_water_quality_csv(
            output_path=output_path,
            start_year=start_year,
            end_year=end_year,
        ),
        "air_pollution": lambda output_path, start_year, end_year: AirPollutionDownloader().save_air_pollution_csv(
            output_path=output_path,
            start_year=start_year,
            end_year=end_year,
        ),
        "green_urban_space_per_capita": lambda output_path, start_year, end_year: GreenUrbanSpacePerCapitaDownloader().save_green_urban_space_per_capita_csv(
            output_path=output_path,
            start_year=start_year,
            end_year=end_year,
        ),
        "healthcare_capacity": lambda output_path, start_year, end_year: HealthcareCapacityDownloader().save_healthcare_capacity_csv(
            output_path=output_path,
            start_year=start_year,
            end_year=end_year,
        ),
        "healthcare_facility_density": lambda output_path, start_year, end_year: HealthcareFacilityDensityDownloader().save_healthcare_facility_density_csv(
            output_path=output_path,
            start_year=start_year,
            end_year=end_year,
        ),
        "urban_population_percentage": lambda output_path, start_year, end_year: UrbanPopulationPercentageDownloader().save_urban_population_percentage_csv(
            output_path=output_path,
            start_year=start_year,
            end_year=end_year,
        ),
    }

    parser = argparse.ArgumentParser(
        description="SERA Data Downloader - Download historical indicators for Italy digital twin"
    )
    parser.add_argument(
        "--indicator",
        choices=[
            "population",
            "birth_rate",
            "migration_flows",
            "age_distribution",
            "death_rate",
            "income",
            "gdp_per_capita",
            "poverty_rate",
            "gini_coefficient",
            "business_density",
            "fdi",
            "exports_imports",
            "unemployment_rate",
            "average_wages",
            "youth_employment",
            "self_employment",
            "skills_match",
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
            "life_expectancy",
            "health_outcomes",
            "crime_rate",
            "social_cohesion",
            "income_inequality",
            "school_enrollment",
            "completion_rates",
            "stem_participation",
            "adult_learning",
            "r_and_d_spending",
            "patents",
            "startups",
            "digital_infrastructure",
            "transportation_access",
            "air_quality",
            "sustainability",
            "green_space",
            "local_government_debt",
            "fiscal_balance",
            "tax_revenue_per_capita",
            "public_spending_efficiency",
            "infrastructure_investment",
            "house_prices",
            "construction_activity",
            "homeownership_rate",
            "housing_affordability_ratio",
            "renewable_energy_percentage",
            "energy_consumption_per_capita",
            "carbon_emissions",
            "electricity_prices",
            "public_transportation_usage",
            "car_ownership_density",
            "traffic_congestion",
            "broadband_penetration",
            "tourist_arrivals",
            "tourism_revenue",
            "cultural_sites",
            "cultural_spending",
            "museums_events",
            "healthcare_spending_per_capita",
            "hospital_beds_available",
            "healthcare_worker_density",
            "preventive_care_coverage",
            "agricultural_productivity",
            "waste_recycling_rate",
            "water_quality",
            "air_pollution",
            "green_urban_space_per_capita",
            "healthcare_capacity",
            "healthcare_facility_density",
            "urban_population_percentage",
            "all",
        ],
        default="population",
        help="Indicator to download",
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=2001,
        help="Start year (default: 2001)",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=2025,
        help="End year (default: 2025)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output file path (optional)",
    )

    args = parser.parse_args()

    if args.indicator == "all":
        if args.output is not None:
            parser.error(
                "--output cannot be used with --indicator all because each indicator writes to its own default file"
            )

        for indicator_name, runner in indicator_runners.items():
            print(f"\n=== Downloading {indicator_name} ===")
            runner(None, args.start_year, args.end_year)
    else:
        indicator_runners[args.indicator](args.output, args.start_year, args.end_year)


if __name__ == "__main__":
    main()
