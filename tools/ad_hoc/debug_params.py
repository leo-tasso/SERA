#!/usr/bin/env python
"""Debug parameter loading."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import pandas as pd
from sera.twin.cli import normalize_wide_geography, load_parameter_scenarios_from_csv

# Load the CSV directly
params_path = Path("tools/test_scenarios.csv")
print(f"Loading from: {params_path}")

raw = pd.read_csv(params_path)
print(f"\nRaw CSV shape: {raw.shape}")
print(f"Raw CSV columns: {raw.columns.tolist()}")
print(f"Raw CSV head:\n{raw.head()}")
print(f"Raw unique years: {sorted(raw['year'].unique())}")

# Now apply normalize_wide_geography
normalized = normalize_wide_geography(raw)
print(f"\nNormalized shape: {normalized.shape}")
print(f"Normalized columns: {normalized.columns.tolist()}")
print(f"Normalized head:\n{normalized.head(20)}")
print(f"Normalized unique years: {sorted(normalized['year'].unique())}")
print(f"Normalized year dtype: {normalized['year'].dtype}")

# Now load using the function
scenarios = load_parameter_scenarios_from_csv(params_path)
print(f"\nLoaded scenarios count: {len(scenarios)}")
if scenarios:
    for i, scenario in enumerate(scenarios):
        print(f"Scenario {i} shape: {scenario.shape}")
else:
    print("ERROR: No scenarios loaded!")
