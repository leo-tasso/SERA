import pandas as pd

df = pd.read_csv('data/unemployment_rate/unemployment_rate_raw_2020_2025.csv')
print(f'Total records: {len(df)}')
print(f'Unique areas: {df["area_code"].nunique()}')
areas = sorted(df["area_code"].unique())
print(f'Areas: {areas}')
print(f'Year range: {int(df["year"].min())}-{int(df["year"].max())}')
