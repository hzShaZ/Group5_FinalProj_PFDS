import sys
import numpy as np
import pandas as pd

prices_path = sys.argv[1] if len(sys.argv) > 1 else "data/wfp_food_prices_phl_sample.csv"
markets_path = sys.argv[2] if len(sys.argv) > 2 else "data/wfp_markets_phl.csv"
df = pd.read_csv(prices_path)
mk = pd.read_csv(markets_path)
n0 = len(df)

df["date"] = pd.to_datetime(df["date"])
df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month

for col in ["admin1", "admin2", "market", "commodity", "category", "unit", "pricetype", "priceflag"]:
    df[col] = df[col].str.strip()

df["is_aggregate"] = df["priceflag"].str.contains("aggregate", na=False)
med = df.groupby("commodity")["price"].transform("median")
df["price_review"] = np.where(df["price"] > 5 * med, True, False)

df["comparable_unit"] = df["unit"] == "KG"

df = df.drop(columns=["currency", "usdprice"], errors="ignore")
df = df.drop(columns=["market", "admin1", "admin2", "latitude", "longitude"], errors="ignore")
df = df.merge(mk[["market_id", "market", "admin1", "admin2", "latitude", "longitude"]],
              on="market_id", how="left", validate="many_to_one")
unmatched = int(df["market"].isna().sum())

analysis = df[(df["pricetype"] == "Retail") & (df["comparable_unit"])]

print("rows loaded:", n0, "| rows after merge:", len(df), "| rows with no market match:", unmatched)
print("rows kept for per-kg retail analysis:", len(analysis))
print("rows flagged: aggregate =", int(df["is_aggregate"].sum()),
      "| flagged for review (kept) =", int(df["price_review"].sum()),
      "| non-KG units =", int((~df["comparable_unit"]).sum()),
      "| non-retail =", int((df["pricetype"] != "Retail").sum()))
rice = analysis[analysis["commodity"] == "Rice (regular, milled)"]
print("\nmedian retail price of Rice (regular, milled), PHP/KG, latest years:")
print(rice.groupby("year")["price"].median().round(2).tail(6).to_string())
