#!/usr/bin/env python
"""Check what area codes are in the income data after loading."""

import pandas as pd
from src.sera.twin.data_loader import DataLoader
from pathlib import Path

loader = DataLoader(Path("data"))
df = loader.load_indicator("income", "demographic")

print("Area codes in income indicator after DataLoader processing:")
print(f'Total unique: {df["area_code"].nunique()}')
print()
print("Area code categories:")
all_codes = sorted(df["area_code"].unique())

national = [c for c in all_codes if c == "IT"]
regions = [c for c in all_codes if c.startswith("IT") and len(c) > 2]
numeric = [c for c in all_codes if not c.startswith("IT")]

print(f"  National (IT): {len(national)}")
print(f"  Regional/NUTS (IT*): {len(regions)}")
print(f"  Numeric codes: {len(numeric)}")
print()
print(f"Sample numeric codes (disaggregated provinces):")
for code in sorted(numeric)[:20]:
    print(f"  {code}")
