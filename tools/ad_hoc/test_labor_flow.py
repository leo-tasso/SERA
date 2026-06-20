#!/usr/bin/env python3
"""Test ISTAT labor dataflows for regional data."""

import requests
import pandas as pd
import io

# Test unemployment regional data
url = "https://esploradati.istat.it/SDMXWS/rest/data/151_929_DF_DCCV_DISOCCUPT1_5?lastNObservations=100"
headers = {"Accept": "application/vnd.sdmx.data+csv;version=1.0.0"}

print("Testing unemployment regional data (151_929_DF_DCCV_DISOCCUPT1_5)...")
print("=" * 70)
response = requests.get(url, headers=headers, timeout=120)

if response.status_code == 200:
    df = pd.read_csv(io.StringIO(response.text))
    print(f"Columns: {list(df.columns)}")
    print(f"Shape: {df.shape}")
    if "REF_AREA" in df.columns:
        print(f"Unique areas: {df['REF_AREA'].nunique()}")
        print(f"Sample areas: {df['REF_AREA'].unique()[:10]}")
    if "TIME_PERIOD" in df.columns:
        print(f"Unique years: {df['TIME_PERIOD'].nunique()}")
        print(f"Year range: {df['TIME_PERIOD'].min()} to {df['TIME_PERIOD'].max()}")
    print(f"\nFirst 5 rows:")
    print(df.head(5).to_string())
else:
    print(f"Error: {response.status_code}")
    print(response.text[:500])
