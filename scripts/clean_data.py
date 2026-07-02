"""
clean_data.py
--------------
Cleans the raw Sample-Superstore.csv and writes an analysis-ready CSV
to data/superstore_clean.csv.

Usage:
    python scripts/clean_data.py
"""

import pandas as pd
from pathlib import Path

RAW_PATH = Path(__file__).resolve().parent.parent / "data" / "Sample-Superstore.csv"
CLEAN_PATH = Path(__file__).resolve().parent.parent / "data" / "superstore_clean.csv"


def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    # utf-8-sig strips the BOM at the start of the file
    return pd.read_csv(path, encoding="utf-8-sig")


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 1. Standardize column names (snake_case, no spaces)
    df.columns = [c.strip().lower().replace(" ", "_").replace("-", "_") for c in df.columns]

    # 2. Parse dates
    df["order_date"] = pd.to_datetime(df["order_date"])
    df["ship_date"] = pd.to_datetime(df["ship_date"])

    # 3. Fix dtypes
    df["postal_code"] = df["postal_code"].astype("Int64")  # nullable int, keeps NaNs

    # 4. Drop exact duplicate rows, if any
    before = len(df)
    df = df.drop_duplicates()
    removed = before - len(df)

    # 5. Derived/engineered columns useful for EDA
    df["order_year"] = df["order_date"].dt.year
    df["order_month"] = df["order_date"].dt.month
    df["order_year_month"] = df["order_date"].dt.to_period("M").astype(str)
    df["shipping_days"] = (df["ship_date"] - df["order_date"]).dt.days
    df["profit_margin"] = (df["profit"] / df["sales"]).round(4)
    df["is_loss"] = df["profit"] < 0

    print(f"Removed {removed} duplicate rows.")
    print(f"Rows with missing postal_code: {df['postal_code'].isna().sum()} (left as-is; not needed for sales/profit analysis)")

    return df


def main():
    df = load_raw()
    clean_df = clean(df)
    clean_df.to_csv(CLEAN_PATH, index=False)
    print(f"Saved cleaned dataset -> {CLEAN_PATH}  ({clean_df.shape[0]} rows, {clean_df.shape[1]} cols)")


if __name__ == "__main__":
    main()
