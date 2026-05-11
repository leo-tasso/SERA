#!/usr/bin/env python
"""Identify what Italian provinces the area codes represent."""

import pandas as pd

# Load population data which has all area codes
df = pd.read_csv('data/demographic/population/population_raw_2001_2025.csv')

# Get unique area codes
area_codes = sorted(df['area_code'].unique())

print("=" * 90)
print("ITALIAN AREA CODES IN SERA MODEL")
print("=" * 90)
print()

# Categorize by prefix
national = [c for c in area_codes if c in ['IT']]
macro_regions = [c for c in area_codes if c.startswith('IT') and len(c) == 3 and c != 'IT']
regions = [c for c in area_codes if c.startswith('IT') and 4 <= len(c) <= 5 and not c[2].isdigit()]
provinces_nuts = [c for c in area_codes if c.startswith('IT') and (c[2:].isdigit() or len(c) >= 6)]
provinces_numeric = [c for c in area_codes if not c.startswith('IT')]

print(f"NATIONAL LEVEL ({len(national)} codes):")
for code in national:
    print(f"  {code:10s} Italy (entire country)")

print()
print(f"MACRO-REGIONS ({len(macro_regions)} codes) - NUTS1:")
region_names = {
    'ITC': 'Northwest',
    'ITD': 'Northeast', 
    'ITE': 'Center',
    'ITF': 'South',
    'ITG': 'Islands'
}
for code in macro_regions:
    print(f"  {code:10s} {region_names.get(code, 'Unknown')}")

print()
print(f"REGIONS ({len(regions)} codes) - NUTS2:")
for code in regions[:15]:
    print(f"  {code:10s} Region")
if len(regions) > 15:
    print(f"  ... and {len(regions) - 15} more regions")

print()
print(f"PROVINCES - NUTS codes ({len(provinces_nuts)} codes):")
for code in provinces_nuts[:10]:
    print(f"  {code:10s} Province")
if len(provinces_nuts) > 10:
    print(f"  ... and {len(provinces_nuts) - 10} more provinces (ISTAT NUTS3)")

print()
print(f"PROVINCES - NUMERIC codes ({len(provinces_numeric)} codes):")
print(f"  These are likely ISTAT territorial codes or disaggregated sub-provincial data")
for code in sorted(provinces_numeric)[:15]:
    count = len(df[df['area_code'] == code])
    print(f"  {code:10s} ({count} records)")
if len(provinces_numeric) > 15:
    print(f"  ... and {len(provinces_numeric) - 15} more numeric codes")

print()
print("=" * 90)
print("SUMMARY:")
print("=" * 90)
print(f"Total unique area codes: {len(area_codes)}")
print()
print("The simulation uses NUMERIC area codes (001272, 002158, etc.) which represent")
print("disaggregated provincial-level data. This allows simulations at the most granular")
print("territorial level in the Italian ISTAT system.")
print()
print("Geographic Hierarchy:")
print("  1. National (IT)")
print("  2. Macro-regions (ITC, ITD, ITE, ITF, ITG)")
print("  3. Regions (ITC1-18, ITD1-5, etc.)")
print("  4. Provinces - NUTS3 (IT108, IT109, etc.)")
print("  5. Provinces - Numeric codes (001272, 002158, etc.) <- Used in simulations")
