#!/usr/bin/env python
"""Check which additional indicators from tracker have strong provincial coverage."""

from pathlib import Path
import pandas as pd

DATA_DIR = Path('data')

# Candidates to check
candidates = [
    ('gini_coefficient', 'economic'),
    ('average_wages', 'labor'),
    ('exports_imports', 'economic'),
    ('fdi', 'economic'),
    ('house_prices', 'housing'),
    ('construction_activity', 'housing'),
    ('air_quality', 'environment'),
    ('sustainability', 'environment'),
    ('energy_consumption_per_capita', 'energy_resources'),
    ('car_ownership_density', 'transportation_mobility'),
    ('adult_learning', 'education'),
    ('startups', 'innovation_infrastructure'),
]

print("=" * 90)
print("ADDITIONAL INDICATORS - PROVINCIAL COVERAGE CHECK")
print("=" * 90)
print()

for indicator, category in candidates:
    indicator_dir = DATA_DIR / category / indicator
    if not indicator_dir.exists():
        print(f"[NO DATA] {indicator:40s} ({category})")
        continue
    
    csv_files = list(indicator_dir.glob("*.csv"))
    if not csv_files:
        print(f"[EMPTY]   {indicator:40s} ({category})")
        continue
    
    latest_file = sorted(csv_files)[-1]
    try:
        df = pd.read_csv(latest_file)
        n_areas = df['area_code'].nunique() if 'area_code' in df.columns else 0
        n_years = len(df) // max(n_areas, 1) if n_areas > 0 else len(df)
        n_rows = len(df)
        
        status = "STRONG" if n_areas >= 100 and n_years >= 8 else "LIMITED"
        print(f"[{status:6s}]   {indicator:40s} | {n_areas:3d} areas, {n_years:2d} yrs, {n_rows:5d} rows")
    except Exception as e:
        print(f"[ERROR]   {indicator:40s} | {str(e)[:40]}")

print()
print("=" * 90)
print("RECOMMENDED FOR TRAINING (STRONG coverage + cross-effects):")
print("=" * 90)
print()

recommendations = [
    ("gini_coefficient", "Income inequality ↔ poverty, affects income distribution"),
    ("average_wages", "Wage level → income, employment outcomes"),
    ("exports_imports", "Trade performance → GDP, employment, business activity"),
    ("fdi", "Investment inflows → GDP, business density, employment"),
    ("house_prices", "Housing cost → affordability, income pressure"),
    ("construction_activity", "Building activity → employment, GDP"),
    ("air_quality", "Environmental → health outcomes, life expectancy"),
    ("startups", "New business → employment, innovation, business density"),
]

for i, (indicator, effect) in enumerate(recommendations, 1):
    print(f"{i}. {indicator:35s}  {effect}")
