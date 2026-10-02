import os
import sys
import numpy as np
import pandas as pd


def resolve_path(cli_idx, default_filename):
    if len(sys.argv) > cli_idx:
        return sys.argv[cli_idx]
    if os.path.exists(default_filename):
        return default_filename
    data_subfolder = os.path.join("data", default_filename)
    if os.path.exists(data_subfolder):
        return data_subfolder
    return default_filename


prices_path = resolve_path(1, "wfp_food_prices_phl_sample.csv")
markets_path = resolve_path(2, "wfp_markets_phl.csv")

df = pd.read_csv(prices_path)
mk = pd.read_csv(markets_path)
n0 = len(df)

df["date"] = pd.to_datetime(df["date"])
df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month

string_cols = [
    "admin1",
    "admin2",
    "market",
    "commodity",
    "category",
    "unit",
    "pricetype",
    "priceflag",
]
for col in string_cols:
    if col in df.columns and df[col].dtype == "object":
        df[col] = df[col].str.strip()

# Safe string search for aggregate price flags
df["is_aggregate"] = df["priceflag"].str.contains("aggregate", case=False, na=False)

# Correct outlier logic: Group by commodity, unit, and pricetype together
med = df.groupby(["commodity", "unit", "pricetype"])["price"].transform("median")
df["price_review"] = np.where(df["price"] > 5 * med, True, False)

df["comparable_unit"] = df["unit"] == "KG"

# Safe column dropping
df = df.drop(columns=["currency", "usdprice"], errors="ignore")
df = df.drop(
    columns=["market", "admin1", "admin2", "latitude", "longitude"],
    errors="ignore",
)

# Many-to-one mapping validation with markets lookup
df = df.merge(
    mk[["market_id", "market", "admin1", "admin2", "latitude", "longitude"]],
    on="market_id",
    how="left",
    validate="many_to_one",
)
unmatched = int(df["market"].isna().sum())

analysis = df[(df["pricetype"] == "Retail") & (df["comparable_unit"])]

# Continuous historical panel filter for Q1
continuous_markets = df[df["year"] <= 2005]["market_id"].unique()
rice_panel = analysis[
    (analysis["commodity"] == "Rice (regular, milled)")
    & (analysis["market_id"].isin(continuous_markets))
]

print(
    "rows loaded:",
    n0,
    "| rows after merge:",
    len(df),
    "| rows with no market match:",
    unmatched,
)
print("rows kept for per-kg retail analysis:", len(analysis))
print(
    "rows flagged: aggregate =",
    int(df["is_aggregate"].sum()),
    "| flagged for review (kept) =",
    int(df["price_review"].sum()),
    "| non-KG units =",
    int((~df["comparable_unit"]).sum()),
    "| non-retail =",
    int((df["pricetype"] != "Retail").sum()),
)

print(
    "\nmedian retail price of Rice (regular, milled) [Continuous Panel], PHP/KG, latest years:"
)
print(rice_panel.groupby("year")["price"].median().round(2).tail(6).to_string())
