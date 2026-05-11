#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Validate simulation outputs across time horizons."""
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Set UTF-8 encoding for stdout
sys.stdout.reconfigure(encoding='utf-8')

output_files = [
    ("1 year", "tools/sim_1yr.csv"),
    ("3 years", "tools/sim_3yr.csv"),
    ("5 years", "tools/sim_5yr.csv"),
    ("10 years", "tools/sim_10yr.csv"),
]

print("=" * 100)
print("SIMULATION OUTPUT VALIDATION")
print("=" * 100)

# Expected indicators (24 total)
expected_indicators = [
    "business_density", "gdp_per_capita", "income", "poverty_rate",
    "self_employment", "unemployment_rate", "youth_employment",
    "completion_rates", "school_enrollment",
    "healthcare_spending_per_capita", "healthcare_worker_density", "life_expectancy",
    "digital_infrastructure", "patents", "transportation_access",
    "air_quality", "carbon_emissions", "green_urban_space_per_capita",
    "renewable_energy_percentage", "sustainability", "water_quality",
    "public_transportation_usage", "traffic_congestion", "crime_rate",
]

expected_provinces = 110
results_summary = []

for label, filepath in output_files:
    print(f"\n{'-' * 100}")
    print(f"HORIZON: {label}")
    print(f"File: {filepath}")
    print(f"{'-' * 100}")
    
    try:
        df = pd.read_csv(filepath)
        print(f"[OK] File loaded successfully")
        print(f"  Shape: {df.shape}")
        print(f"  Columns: {df.columns.tolist()[:5]}... ({len(df.columns)} total)")
        
        # Check required columns
        has_area_code = "area_code" in df.columns
        has_year = "year" in df.columns
        print(f"  area_code present: {has_area_code}")
        print(f"  year present: {has_year}")
        
        # Check indicators
        indicator_cols = [col for col in df.columns if col in expected_indicators]
        missing_indicators = [ind for ind in expected_indicators if ind not in df.columns]
        
        print(f"  Indicators found: {len(indicator_cols)}/{len(expected_indicators)}")
        if missing_indicators:
            print(f"    MISSING: {missing_indicators}")
        
        # Check provinces
        unique_provinces = df["area_code"].nunique()
        print(f"  Unique provinces: {unique_provinces}/{expected_provinces}")
        
        # Check for NaN in area_code
        area_code_nans = df["area_code"].isna().sum()
        if area_code_nans > 0:
            print(f"    WARNING: {area_code_nans} NaN values in area_code")
        
        # Check years
        unique_years = sorted(df["year"].unique())
        print(f"  Years: {unique_years}")
        
        # Check for NaN/inf values
        nan_count = df[indicator_cols].isna().sum().sum()
        inf_count = np.isinf(df[indicator_cols].select_dtypes(include=[np.number])).sum().sum()
        print(f"  NaN values in indicators: {nan_count}")
        print(f"  Inf values in indicators: {inf_count}")
        
        # Sample statistics
        print(f"\n  Indicator Value Ranges:")
        for ind in sorted(indicator_cols)[:6]:
            if ind in df.columns:
                min_val = df[ind].min()
                max_val = df[ind].max()
                mean_val = df[ind].mean()
                print(f"    {ind}: [{min_val:8.2f}, {max_val:8.2f}] mean={mean_val:8.2f}")
        if len(indicator_cols) > 6:
            print(f"    ... ({len(indicator_cols)-6} more indicators)")
        
        is_valid = (
            len(indicator_cols) == len(expected_indicators) and 
            unique_provinces == expected_provinces and 
            nan_count == 0 and 
            inf_count == 0 and
            area_code_nans == 0
        )
        
        status = "[VALID]" if is_valid else "[PARTIAL]"
        results_summary.append({
            "horizon": label,
            "shape": df.shape,
            "indicators": len(indicator_cols),
            "provinces": unique_provinces,
            "years_range": f"{min(unique_years)}-{max(unique_years)}",
            "nan_count": nan_count,
            "inf_count": inf_count,
            "area_code_nans": area_code_nans,
            "status": status
        })
        
    except FileNotFoundError:
        print(f"[ERROR] File not found: {filepath}")
        results_summary.append({
            "horizon": label,
            "status": "[FILE_NOT_FOUND]"
        })
    except Exception as e:
        print(f"[ERROR] {e}")
        results_summary.append({
            "horizon": label,
            "status": f"[ERROR]"
        })

print(f"\n{'=' * 100}")
print("SUMMARY TABLE")
print(f"{'=' * 100}")
print(f"{'Horizon':<12s} {'Status':<12s} {'Shape':<18s} {'Indicators':<12s} {'Provinces':<12s}")
print(f"{'-' * 100}")
for row in results_summary:
    if 'shape' in row:
        print(f"{row['horizon']:<12s} {row['status']:<12s} {str(row['shape']):<18s} {str(row['indicators'])+'/24':<12s} {str(row['provinces'])+'/110':<12s}")
    else:
        print(f"{row['horizon']:<12s} {row['status']:<12s}")

print(f"\n{'=' * 100}")
print("VALIDATION COMPLETE")
print(f"{'=' * 100}")
