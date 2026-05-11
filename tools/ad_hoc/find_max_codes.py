#!/usr/bin/env python
"""Find which indicator has the 266 numeric area codes."""

import pandas as pd
from src.sera.twin.data_loader import DataLoader
from pathlib import Path

loader = DataLoader(Path('data'))

# These are the indicators with high provincial coverage
provincial_indicators = [
    ("air_quality", "environment"),
    ("water_quality", "environmental_quality"),
    ("green_urban_space_per_capita", "environment"),
    ("business_density", "economic"),
    ("healthcare_spending_per_capita", "healthcare_public_services"),
    ("poverty_rate", "economic"),
    ("crime_rate", "social_well_being"),
    ("completion_rates", "education"),
    ("digital_infrastructure", "innovation_infrastructure"),
    ("patents", "innovation_infrastructure"),
    ("renewable_energy_percentage", "energy_resources"),
    ("carbon_emissions", "energy_resources"),
    ("traffic_congestion", "transportation_mobility"),
]

print("=" * 90)
print("AREA CODE COVERAGE BY INDICATOR")
print("=" * 90)
print()

max_indicator = None
max_count = 0

for ind_name, category in provincial_indicators:
    try:
        df = loader.load_indicator(ind_name, category)
        n_codes = df['area_code'].nunique()
        
        # Check what format the codes are
        sample_codes = sorted(df['area_code'].unique())[:3]
        code_type = "numeric" if sample_codes and not sample_codes[0].startswith('IT') else "NUTS"
        
        print(f"{ind_name:40s} | {n_codes:3d} area codes | {code_type:7s} | {sample_codes}")
        
        if n_codes > max_count:
            max_count = n_codes
            max_indicator = ind_name
    except Exception as e:
        print(f"{ind_name:40s} | ERROR: {str(e)[:40]}")

print()
print(f"INDICATOR WITH MOST AREA CODES: {max_indicator} ({max_count})")
print()

# Check the max indicator's codes
df_max = loader.load_indicator(max_indicator, 'energy_resources' if max_indicator == 'renewable_energy_percentage' 
                                              else 'environment' if max_indicator in ['air_quality', 'water_quality', 'green_urban_space_per_capita']
                                              else 'economic')

all_codes = sorted(df_max['area_code'].unique())
print(f"Sample area codes from {max_indicator}:")
for code in all_codes[:20]:
    print(f"  {code}")

print()
print(f"Total: {len(all_codes)} unique area codes")
print()
print("These numeric codes (like 001272, 002158) are ISTAT territorial identifiers")
print("that represent provincial or sub-regional Italian administrative divisions.")
