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

print("=== 1. LOAD ===")
print("prices :", prices_path, "| shape (rows, columns):", df.shape)
print("markets:", markets_path, "| shape (rows, columns):", mk.shape)

print("\n=== 2. COLUMNS AND TYPES ===")
print("--- prices ---")
print(df.dtypes)
print("\nfirst 3 rows:")
print(df.head(3).to_string())
print("\n--- markets ---")
print(mk.dtypes)
print("\nfirst 3 rows:")
print(mk.head(3).to_string())

df["date"] = pd.to_datetime(df["date"])
df["year"] = df["date"].dt.year

print("\n=== 3. BASIC QUALITY CHECKS: PRICES ===")
print("missing values per column:")
print(df.isna().sum()[df.isna().sum() > 0] if df.isna().any().any() else "none")
print("exact duplicate rows:", df.duplicated().sum())
key = ["date", "market_id", "commodity_id", "pricetype", "unit"]
print(
    "duplicate (date, market, commodity, pricetype, unit):",
    df.duplicated(key).sum(),
)
print("prices <= 0:", (df["price"] <= 0).sum())
print("date range:", df["date"].min().date(), "to", df["date"].max().date())
print(
    "distinct markets:",
    df["market_id"].nunique(),
    "| commodities:",
    df["commodity_id"].nunique(),
    "| admin1 regions:",
    df["admin1"].nunique(),
)

print("\n--- mixed units ---")
print(df["unit"].value_counts())
print("\n--- price type (retail / wholesale / farm gate) ---")
print(df["pricetype"].value_counts())
print("\n--- priceflag (aggregate rows are not direct observations) ---")
print(df["priceflag"].value_counts())

print("\n--- uneven coverage by year (this can fake a trend) ---")
cov = df.groupby("year").agg(
    rows=("price", "size"),
    markets=("market_id", "nunique"),
    commodities=("commodity_id", "nunique"),
)
print(cov.to_string())

print(
    "\n--- possible outliers: price more than 5x its (commodity, unit, pricetype) median ---"
)
med = df.groupby(["commodity", "unit", "pricetype"])["price"].transform(
    "median"
)
flag = np.where(df["price"] > 5 * med, 1, 0)
print("flagged rows:", int(flag.sum()))
if flag.sum() > 0:
    print(
        df.loc[
            flag == 1, ["date", "market", "commodity", "unit", "price"]
        ].head(5).to_string()
    )

print("\n--- implied PHP per USD (price / usdprice) as a sanity check ---")
fx = (df["price"] / df["usdprice"]).replace([np.inf, -np.inf], np.nan)
print(fx.describe().round(2))
bad_fx = np.where((fx < 35) | (fx > 70), 1, 0)
print(
    "rows with implied rate outside 35-70 PHP/USD (usdprice looks wrong):",
    int(bad_fx.sum()),
)

print("\n=== 4. BASIC QUALITY CHECKS: MARKETS ===")
print("missing values per column:")
print(mk.isna().sum()[mk.isna().sum() > 0] if mk.isna().any().any() else "none")
print("duplicate market_id:", mk["market_id"].duplicated().sum())
print("country codes:", list(mk["countryiso3"].unique()))
print(
    "regions (admin1):",
    mk["admin1"].nunique(),
    "| provinces (admin2):",
    mk["admin2"].nunique(),
)
shared = mk.duplicated(["latitude", "longitude"], keep=False).sum()
print(
    "markets sharing identical coordinates with another market:", int(shared)
)

print("\n=== 5. INTEGRATION CHECK: prices and markets (key = market_id) ===")
price_ids = set(df["market_id"])
market_ids = set(mk["market_id"])
print("market_ids in prices but NOT in markets:", len(price_ids - market_ids))
print("market_ids in markets with NO price rows:", len(market_ids - price_ids))
first = (
    df.groupby("market_id")[
        ["market", "admin1", "admin2", "latitude", "longitude"]
    ]
    .first()
    .reset_index()
)
chk = first.merge(mk, on="market_id", suffixes=("_prices", "_markets"))
for col in ["market", "admin1", "admin2"]:
    print(
        col,
        "mismatches between the two files:",
        int((chk[col + "_prices"] != chk[col + "_markets"]).sum()),
    )
for col in ["latitude", "longitude"]:
    print(
        col,
        "mismatches between the two files:",
        int(
            (np.abs(chk[col + "_prices"] - chk[col + "_markets"]) > 1e-6).sum()
        ),
    )
merged = df.merge(mk[["market_id", "countryiso3"]], on="market_id", how="left")
print(
    "rows before merge:",
    len(df),
    "| after merge:",
    len(merged),
    "| rows with no market match:",
    int(merged["countryiso3"].isna().sum()),
)

print("\n=== 6. HOW VARIABLES CONNECT TO THE QUESTIONS ===")
print("Q1 long-run rice trend -> date, commodity, pricetype, price, market_id")
early_years = df[df["year"] <= 2005]["market_id"].unique()
rice_panel = df[
    (df["commodity"] == "Rice (regular, milled)")
    & (df["pricetype"] == "Retail")
    & (df["market_id"].isin(early_years))
]
print(
    "Continuous Panel Rice Median PHP/KG:\n",
    rice_panel.groupby("year")["price"].median().round(2).to_string(),
)

print(
    "\nQ2 regional price level and volatility -> admin1, admin2, market_id, commodity, price"
)
print(
    df.groupby("admin1")["price"]
    .median()
    .round(1)
    .sort_values()
    .head(3)
    .to_string(),
    "...",
)
print("\nQ3 fastest rising foods since 2020 -> category, commodity, year, price")
print(df["category"].value_counts().to_string())
