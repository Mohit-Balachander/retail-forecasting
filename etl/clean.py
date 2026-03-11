"""
etl/clean.py
------------
Step 2 of ETL Pipeline: Data Cleaning & Transformation
- Merges train + store data
- Handles missing values
- Engineers time-based features
- Saves cleaned data to data/processed/
"""

import pandas as pd
import numpy as np
from pathlib import Path

PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def merge_datasets(train_df: pd.DataFrame, store_df: pd.DataFrame) -> pd.DataFrame:
    """Merges sales data with store metadata."""
    print("🔗 Merging train + store datasets...")
    df = train_df.merge(store_df, on="Store", how="left")
    print(f"   Merged shape: {df.shape}")
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fills missing values with sensible defaults.
    Logs every column with nulls before/after.
    """
    print("\n🧹 Handling missing values...")
    null_before = df.isnull().sum()
    cols_with_nulls = null_before[null_before > 0]
    if len(cols_with_nulls):
        print(f"   Columns with nulls:\n{cols_with_nulls}")

    # Competition distance — fill with large number (no nearby competitor)
    df["CompetitionDistance"] = df["CompetitionDistance"].fillna(df["CompetitionDistance"].median())

    # Competition open date — fill with 0 (unknown = treat as long ago)
    for col in ["CompetitionOpenSinceMonth", "CompetitionOpenSinceYear"]:
        df[col] = df[col].fillna(0)

    # Promo2 fields — no promo = 0
    for col in ["Promo2SinceWeek", "Promo2SinceYear"]:
        df[col] = df[col].fillna(0)

    df["PromoInterval"] = df["PromoInterval"].fillna("None")

    null_after = df.isnull().sum().sum()
    print(f"   Total nulls remaining: {null_after}")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates time-based and business features for ML models.
    """
    print("\n⚙️  Engineering features...")
    df["Date"] = pd.to_datetime(df["Date"])

    # Time features
    df["Year"]        = df["Date"].dt.year
    df["Month"]       = df["Date"].dt.month
    df["Day"]         = df["Date"].dt.day
    df["WeekOfYear"]  = df["Date"].dt.isocalendar().week.astype(int)
    df["DayOfWeek"]   = df["Date"].dt.dayofweek
    df["IsWeekend"]   = df["DayOfWeek"].isin([5, 6]).astype(int)
    df["Quarter"]     = df["Date"].dt.quarter

    # Competition age in months
    df["CompetitionOpenMonths"] = (
        (df["Year"] - df["CompetitionOpenSinceYear"]) * 12
        + (df["Month"] - df["CompetitionOpenSinceMonth"])
    ).clip(lower=0)

    # Is store running a promo this month?
    promo_month_map = {
        "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4,
        "May": 5, "Jun": 6, "Jul": 7, "Aug": 8,
        "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12
    }
    df["IsPromoMonth"] = df.apply(
        lambda row: int(row["Month"] in [
            promo_month_map.get(m, 0)
            for m in str(row["PromoInterval"]).split(",")
        ]) if row["Promo2"] == 1 else 0,
        axis=1
    )

    # Log-transform sales (reduces skew for ML)
    if "Sales" in df.columns:
        df["LogSales"] = np.log1p(df["Sales"])

    print(f"   Features added. Final shape: {df.shape}")
    return df


def remove_closed_stores(df: pd.DataFrame) -> pd.DataFrame:
    """Removes rows where store is closed (Sales = 0 on closed days)."""
    before = len(df)
    df = df[(df["Open"] == 1) & (df["Sales"] > 0)]
    print(f"\n🏪 Removed {before - len(df):,} closed-store rows. Remaining: {len(df):,}")
    return df


def save_processed(df: pd.DataFrame, filename: str = "cleaned_train.csv"):
    """Saves cleaned DataFrame to data/processed/."""
    out_path = PROCESSED_DIR / filename
    df.to_csv(out_path, index=False)
    print(f"\n💾 Saved cleaned data → {out_path}")
    return out_path


def run_cleaning(train_df: pd.DataFrame, store_df: pd.DataFrame) -> pd.DataFrame:
    """Full cleaning pipeline. Returns cleaned DataFrame."""
    df = merge_datasets(train_df, store_df)
    df = handle_missing_values(df)
    df = remove_closed_stores(df)
    df = engineer_features(df)
    save_processed(df)
    return df


if __name__ == "__main__":
    from ingest import load_raw_data
    dfs = load_raw_data()
    cleaned = run_cleaning(dfs["train"], dfs["store"])
    print(f"\n✅ Cleaning complete. Shape: {cleaned.shape}")
