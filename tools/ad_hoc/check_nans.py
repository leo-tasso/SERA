#!/usr/bin/env python
import pandas as pd

df = pd.read_csv("tools/sim_1yr.csv")
print("Total rows:", len(df))
print("NaN area_codes:", df["area_code"].isna().sum())
print("Missing province codes:", 110 - df["area_code"].nunique())

if df["area_code"].isna().sum() > 0:
    print("\nRows with NaN area_code:")
    print(df[df["area_code"].isna()])

print("\nUnique years:")
print(df["year"].unique())

print("\nShape breakdown by year:")
print(df.groupby("year").size())

print("\nProvince count per year:")
print(df.groupby("year")["area_code"].nunique())
