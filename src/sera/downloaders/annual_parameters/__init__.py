"""Annual input parameter downloaders."""

from .agriculture_support_level import AgricultureSupportLevelDownloader
from .corporate_tax_rate import CorporateTaxRateDownloader
from .education_spending_allocation import EducationSpendingAllocationDownloader
from .environmental_regulations_strictness import EnvironmentalRegulationsStrictnessDownloader
from .green_energy_environment_investment import GreenEnergyEnvironmentInvestmentDownloader
from .healthcare_spending_allocation import HealthcareSpendingAllocationDownloader
from .housing_urban_development_support import HousingUrbanDevelopmentSupportDownloader
from .immigration_policy_level import ImmigrationPolicyLevelDownloader
from .income_tax_rate import IncomeTaxRateDownloader
from .infrastructure_investment_allocation import InfrastructureInvestmentAllocationDownloader
from .manufacturing_incentives import ManufacturingIncentivesDownloader
from .pension_retirement_spending import PensionRetirementSpendingDownloader
from .property_wealth_tax_rate import PropertyWealthTaxRateDownloader
from .public_sector_wage_levels import PublicSectorWageLevelsDownloader
from .rd_innovation_incentives import RdInnovationIncentivesDownloader
from .regulatory_burden_level import RegulatoryBurdenLevelDownloader
from .small_business_support import SmallBusinessSupportDownloader
from .social_welfare_spending_allocation import SocialWelfareSpendingAllocationDownloader
from .tourism_support_level import TourismSupportLevelDownloader
from .vat_consumption_tax_rate import VatConsumptionTaxRateDownloader

ANNUAL_PARAMETER_RUNNERS = {
    "income_tax_rate": lambda output_path, start_year, end_year: IncomeTaxRateDownloader().save_income_tax_rate_csv(
        output_path=output_path, start_year=start_year, end_year=end_year
    ),
    "corporate_tax_rate": lambda output_path, start_year, end_year: CorporateTaxRateDownloader().save_corporate_tax_rate_csv(
        output_path=output_path, start_year=start_year, end_year=end_year
    ),
    "property_wealth_tax_rate": lambda output_path, start_year, end_year: PropertyWealthTaxRateDownloader().save_property_wealth_tax_rate_csv(
        output_path=output_path, start_year=start_year, end_year=end_year
    ),
    "vat_consumption_tax_rate": lambda output_path, start_year, end_year: VatConsumptionTaxRateDownloader().save_vat_consumption_tax_rate_csv(
        output_path=output_path, start_year=start_year, end_year=end_year
    ),
    "healthcare_spending_allocation": lambda output_path, start_year, end_year: HealthcareSpendingAllocationDownloader().save_healthcare_spending_allocation_csv(
        output_path=output_path, start_year=start_year, end_year=end_year
    ),
    "education_spending_allocation": lambda output_path, start_year, end_year: EducationSpendingAllocationDownloader().save_education_spending_allocation_csv(
        output_path=output_path, start_year=start_year, end_year=end_year
    ),
    "infrastructure_investment_allocation": lambda output_path, start_year, end_year: InfrastructureInvestmentAllocationDownloader().save_infrastructure_investment_allocation_csv(
        output_path=output_path, start_year=start_year, end_year=end_year
    ),
    "social_welfare_spending_allocation": lambda output_path, start_year, end_year: SocialWelfareSpendingAllocationDownloader().save_social_welfare_spending_allocation_csv(
        output_path=output_path, start_year=start_year, end_year=end_year
    ),
    "rd_innovation_incentives": lambda output_path, start_year, end_year: RdInnovationIncentivesDownloader().save_rd_innovation_incentives_csv(
        output_path=output_path, start_year=start_year, end_year=end_year
    ),
    "green_energy_environment_investment": lambda output_path, start_year, end_year: GreenEnergyEnvironmentInvestmentDownloader().save_green_energy_environment_investment_csv(
        output_path=output_path, start_year=start_year, end_year=end_year
    ),
    "agriculture_support_level": lambda output_path, start_year, end_year: AgricultureSupportLevelDownloader().save_agriculture_support_level_csv(
        output_path=output_path, start_year=start_year, end_year=end_year
    ),
    "manufacturing_incentives": lambda output_path, start_year, end_year: ManufacturingIncentivesDownloader().save_manufacturing_incentives_csv(
        output_path=output_path, start_year=start_year, end_year=end_year
    ),
    "tourism_support_level": lambda output_path, start_year, end_year: TourismSupportLevelDownloader().save_tourism_support_level_csv(
        output_path=output_path, start_year=start_year, end_year=end_year
    ),
    "small_business_support": lambda output_path, start_year, end_year: SmallBusinessSupportDownloader().save_small_business_support_csv(
        output_path=output_path, start_year=start_year, end_year=end_year
    ),
    "immigration_policy_level": lambda output_path, start_year, end_year: ImmigrationPolicyLevelDownloader().save_immigration_policy_level_csv(
        output_path=output_path, start_year=start_year, end_year=end_year
    ),
    "regulatory_burden_level": lambda output_path, start_year, end_year: RegulatoryBurdenLevelDownloader().save_regulatory_burden_level_csv(
        output_path=output_path, start_year=start_year, end_year=end_year
    ),
    "public_sector_wage_levels": lambda output_path, start_year, end_year: PublicSectorWageLevelsDownloader().save_public_sector_wage_levels_csv(
        output_path=output_path, start_year=start_year, end_year=end_year
    ),
    "pension_retirement_spending": lambda output_path, start_year, end_year: PensionRetirementSpendingDownloader().save_pension_retirement_spending_csv(
        output_path=output_path, start_year=start_year, end_year=end_year
    ),
    "housing_urban_development_support": lambda output_path, start_year, end_year: HousingUrbanDevelopmentSupportDownloader().save_housing_urban_development_support_csv(
        output_path=output_path, start_year=start_year, end_year=end_year
    ),
    "environmental_regulations_strictness": lambda output_path, start_year, end_year: EnvironmentalRegulationsStrictnessDownloader().save_environmental_regulations_strictness_csv(
        output_path=output_path, start_year=start_year, end_year=end_year
    ),
}
