from src.sera.twin.data_loader import DataLoader
from pathlib import Path

DATA_DIR = Path('data')
loader = DataLoader(DATA_DIR)

# List of all candidates from earlier audit
candidates = [
    ('startups', 'innovation_infrastructure', 1, 139, 6),
    ('business_density', 'economic', 1, 100, 20),
    ('healthcare_spending_per_capita', 'healthcare_public_services', 1, 137, 25),
    ('renewable_energy_percentage', 'energy_resources', 1, 100, 15),
    ('carbon_emissions', 'energy_resources', -1, 100, 15),
    ('air_quality', 'environmental_quality', 1, 100, 15),
    ('house_prices', 'housing', 1, 137, 25),
    ('broadband_penetration', 'innovation_infrastructure', 1, 100, 20),
    ('poverty_rate', 'economic', -1, 137, 25),
    ('crime_rate', 'social_well_being', -1, 100, 20),
]

print('Checking next batch of candidates...')
print()

for ind_name, category, direction, exp_provinces, exp_years in candidates:
    try:
        df = loader.load_indicator(ind_name, category)
        if df.empty:
            print(f'{ind_name}: NO DATA')
            continue
        
        df = loader.disaggregate_national_to_provincial(df)
        df = df[(df['year'] >= 2001) & (df['year'] <= 2025)]
        
        areas = len(df['area_code'].unique())
        years = len(df['year'].unique())
        rows = len(df)
        
        status = 'OK' if areas >= 100 and years >= 8 else 'LIMITED'
        print(f'{ind_name}: {areas} areas, {years} years, {rows} rows [{status}]')
    except Exception as e:
        print(f'{ind_name}: ERROR - {str(e)[:50]}')
