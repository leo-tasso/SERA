from src.sera.twin.data_loader import DataLoader
from pathlib import Path
import pandas as pd

DATA_DIR = Path('data')
loader = DataLoader(DATA_DIR)

# Full list from test_multi_horizon.py
indicators = {
    'population': ('demographic', 1),
    'income': ('demographic', 1),
    'unemployment_rate': ('labor', -1),
    'life_expectancy': ('social_well_being', 1),
    'school_enrollment': ('education', 1),
    'gdp_per_capita': ('economic', 1),
}

parameters = {
    'income_tax_rate': 'annual_parameters',
    'corporate_tax_rate': 'annual_parameters',
    'property_wealth_tax_rate': 'annual_parameters',
    'vat_consumption_tax_rate': 'annual_parameters',
    'healthcare_spending_allocation': 'annual_parameters',
    'education_spending_allocation': 'annual_parameters',
    'infrastructure_investment_allocation': 'annual_parameters',
    'social_welfare_spending_allocation': 'annual_parameters',
    'rd_innovation_incentives': 'annual_parameters',
    'green_energy_environment_investment': 'annual_parameters',
    'agriculture_support_level': 'annual_parameters',
    'manufacturing_incentives': 'annual_parameters',
    'tourism_support_level': 'annual_parameters',
    'small_business_support': 'annual_parameters',
    'immigration_policy_level': 'annual_parameters',
    'regulatory_burden_level': 'annual_parameters',
    'public_sector_wage_levels': 'annual_parameters',
    'pension_retirement_spending': 'annual_parameters',
    'housing_urban_development_support': 'annual_parameters',
    'environmental_regulations_strictness': 'annual_parameters',
}

print("Loading data...")
indicators_df, parameters_df = loader.prepare_training_data(indicators, parameters, 2001, 2025)
print(f"Indicators shape: {indicators_df.shape}")
print(f"Parameters shape: {parameters_df.shape}")
print()

# Now manually do prepare_feature_matrix for income
indicator_name = 'income'
lag_years = 1

print(f"Preparing features for {indicator_name}...")
X_data = parameters_df.copy()
print(f"1. X_data shape: {X_data.shape}")
print(f"   X_data NaN: {X_data.isna().sum().sum()}")
print()

lag_indicators = indicators_df[['area_code', 'year', indicator_name]].copy()
print(f"2. lag_indicators before year adjustment: {lag_indicators.shape}")
print(f"   NaN: {lag_indicators.isna().sum().sum()}")
print()

lag_indicators['year'] = lag_indicators['year'] + lag_years
lag_indicators = lag_indicators.rename(columns={indicator_name: f'{indicator_name}_lag1'})
print(f"3. lag_indicators after adjustment: {lag_indicators.shape}")

X_merged = X_data.merge(lag_indicators, on=['area_code', 'year'], how='inner')
print(f"4. X_merged after inner merge: {X_merged.shape}")
print(f"   NaN: {X_merged.isna().sum().sum()}")
print()

y_data = indicators_df[['area_code', 'year', indicator_name]].copy()
print(f"5. y_data: {y_data.shape}")

data = X_merged.merge(y_data, on=['area_code', 'year'], how='inner')
print(f"6. data after merge with y: {data.shape}")
print(f"   NaN before dropna: {data.isna().sum().sum()}")

# Show NaN per column
print(f"   NaN per column:")
for col in data.columns:
    nan_count = data[col].isna().sum()
    if nan_count > 0:
        print(f"     {col}: {nan_count}")

data = data.dropna()
print(f"   After dropna: {data.shape}")
