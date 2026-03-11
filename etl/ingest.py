"""
etl/ingest.py
-------------
Step 1 of ETL Pipeline: Data Ingestion
- Downloads Rossmann Store Sales dataset from Kaggle
- Optionally uploads raw files to AWS S3
- Saves locally to data/raw/
"""

import os
import boto3
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────
# 1. Download from Kaggle
# ─────────────────────────────────────────────
def download_kaggle_dataset():
    """
    Downloads Rossmann Store Sales dataset from Kaggle.
    Requires KAGGLE_USERNAME and KAGGLE_KEY in .env
    """
    try:
        import kaggle
        print("📥 Downloading Rossmann dataset from Kaggle...")
        kaggle.api.authenticate()
        kaggle.api.competition_download_files(
            "rossmann-store-sales",
            path=str(RAW_DIR),
            quiet=False
        )
        print(f"✅ Dataset downloaded to {RAW_DIR}")
    except Exception as e:
        print(f"⚠️  Kaggle download failed: {e}")
        print("💡 Tip: Add KAGGLE_USERNAME and KAGGLE_KEY to your .env file")
        print("   Or manually place train.csv, store.csv in data/raw/")


# ─────────────────────────────────────────────
# 2. Load CSVs into DataFrames
# ─────────────────────────────────────────────
def load_raw_data() -> dict:
    """
    Loads raw CSV files into a dictionary of DataFrames.
    Returns: { 'train': df, 'store': df, 'test': df }
    """
    files = {
        "train": RAW_DIR / "train.csv",
        "store": RAW_DIR / "store.csv",
        "test":  RAW_DIR / "test.csv",
    }

    dataframes = {}
    for name, path in files.items():
        if path.exists():
            print(f"📂 Loading {name}.csv ...")
            dataframes[name] = pd.read_csv(path, low_memory=False)
            print(f"   Shape: {dataframes[name].shape}")
        else:
            print(f"⚠️  {path} not found. Run download_kaggle_dataset() first.")

    return dataframes


# ─────────────────────────────────────────────
# 3. Upload raw files to AWS S3 (optional)
# ─────────────────────────────────────────────
def upload_to_s3(local_path: str, s3_key: str):
    """
    Uploads a file to AWS S3 bucket.
    Requires AWS_BUCKET_NAME, AWS_ACCESS_KEY, AWS_SECRET_KEY in .env
    """
    bucket = os.getenv("AWS_BUCKET_NAME")
    if not bucket:
        print("⚠️  AWS_BUCKET_NAME not set. Skipping S3 upload.")
        return

    try:
        s3 = boto3.client(
            "s3",
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY"),
            aws_secret_access_key=os.getenv("AWS_SECRET_KEY"),
            region_name=os.getenv("AWS_REGION", "us-east-1")
        )
        s3.upload_file(local_path, bucket, s3_key)
        print(f"☁️  Uploaded {local_path} → s3://{bucket}/{s3_key}")
    except Exception as e:
        print(f"❌ S3 upload failed: {e}")


def upload_all_raw_to_s3():
    """Uploads all raw CSV files to S3 under raw/ prefix."""
    for csv_file in RAW_DIR.glob("*.csv"):
        upload_to_s3(str(csv_file), f"raw/{csv_file.name}")


if __name__ == "__main__":
    download_kaggle_dataset()
    dfs = load_raw_data()
    upload_all_raw_to_s3()
    print("\n✅ Ingestion complete.")
    for name, df in dfs.items():
        print(f"   {name}: {df.shape[0]:,} rows × {df.shape[1]} cols")
