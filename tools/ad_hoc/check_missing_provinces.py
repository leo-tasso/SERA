#!/usr/bin/env python
import pandas as pd

df = pd.read_csv("tools/sim_1yr.csv")

# Get all unique provinces
from sera.twin.province_mapping import PROVINCE_SIGLAS_110

all_provinces = set(PROVINCE_SIGLAS_110)
present_provinces = set(df["area_code"].dropna().unique())
missing_provinces = all_provinces - present_provinces

print(f"All provinces: {len(all_provinces)}")
print(f"Present provinces: {len(present_provinces)}")
print(f"Missing provinces: {len(missing_provinces)}")
if missing_provinces:
    print(f"  Missing: {sorted(missing_provinces)}")

# Which year has the NaN values?
print("\nRows with NaN area_code:")
nan_rows = df[df["area_code"].isna()]
print(f"  Count: {len(nan_rows)}")
print(f"  Years: {sorted(nan_rows['year'].unique())}")

# Check first non-NaN row for comparison
print("\nSample non-NaN row:")
non_nan = df[df["area_code"].notna()].iloc[0]
print(f"  {non_nan['area_code']} {non_nan['year']}")
print(f"  Indicators: {non_nan[2:5].to_dict()}")

print("\nSample NaN row:")
nan_sample = df[df["area_code"].isna()].iloc[0]
print(f"  Indicators: {nan_sample[2:5].to_dict()}")
