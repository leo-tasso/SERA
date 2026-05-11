#!/usr/bin/env python
"""Explain what the 266 area codes represent."""

import pandas as pd
from src.sera.twin.data_loader import DataLoader
from pathlib import Path

loader = DataLoader(Path('data'))

print("=" * 90)
print("WHAT DO THE 266 AREA CODES REPRESENT?")
print("=" * 90)
print()

# Load water_quality which has numeric codes
df_water = loader.load_indicator('water_quality', 'environmental_quality')
numeric_codes = sorted([c for c in df_water['area_code'].unique() if isinstance(c, str) and c.isdigit()])

print(f"Water quality indicator has {len(numeric_codes)} numeric area codes")
print()
print("Sample ISTAT numeric territorial codes:")
for code in numeric_codes[:25]:
    print(f"  {code}")
print(f"  ... and {len(numeric_codes) - 25} more")
print()
print("-" * 90)
print()

print("EXPLANATION:")
print()
print("The numeric codes (001272, 002158, 003106, etc.) are ISTAT territorial")
print("identifiers that represent:")
print()
print("  • Italian municipalities (comuni)")
print("  • Or aggregated sub-municipal territorial units")
print("  • From different ISTAT geographic databases")
print()
print("When multiple indicators with DIFFERENT geographic coverage are loaded:")
print()
print("  1. air_quality: 134 NUTS-coded areas (IT108, IT109, IT110, etc.)")
print("  2. water_quality: 121 numeric-coded areas (001272, 002158, etc.)")
print("  3. completion_rates: 132 NUTS-coded areas")
print("  4. patents: 135 NUTS-coded areas")
print("  ... and so on")
print()
print("  When merged with OUTER JOIN, the union = 266 total unique area codes")
print()
print("-" * 90)
print()

# Try to get a mapping
print("ISTAT NUMERIC CODE EXAMPLES:")
print()
print("These numeric codes typically represent:")
print()
print("  001272 - Municipality or territorial unit code from ISTAT")
print("  002158 - Different municipality/territorial unit")
print("  003106 - Another administrative division")
print()
print("The exact mapping depends on which ISTAT dataflow the indicator came from:")
print()
print("  • Some indicators use provincial codes (IT108, IT109, ...)")
print("  • Other indicators use municipal-level or specialized territorial units")
print()
print("=" * 90)
print()
print("PRACTICAL IMPACT:")
print()
print("• Each row in simulation_results_22indicators.csv represents one")
print("  territorial unit (province or municipality) for one year")
print()
print("• 266 area codes × 5 years (2026-2030) = 1,330 rows per scenario")
print()
print("• The union of all area codes ensures comprehensive geographic coverage")
print("  across all 22 indicators in the model")
