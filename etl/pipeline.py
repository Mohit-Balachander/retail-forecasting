"""
etl/pipeline.py
---------------
Master ETL Orchestrator
Runs the full pipeline: Ingest → Clean → Validate → Load
Can be triggered manually or via AWS Lambda
"""

import sys
import time
from pathlib import Path
from datetime import datetime

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from etl.ingest   import load_raw_data, download_kaggle_dataset
from etl.clean    import run_cleaning
from etl.validate import run_validation
from etl.load     import run_load


def run_pipeline(skip_download: bool = False, skip_db_load: bool = False):
    """
    Full ETL pipeline.

    Args:
        skip_download: Set True if data/raw/ already has CSV files
        skip_db_load:  Set True to skip database load (local dev mode)
    """
    start = time.time()
    print("=" * 60)
    print(f"  🚀 RETAIL FORECASTING ETL PIPELINE")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # ── Step 1: Ingest ──────────────────────────────────────────
    print("\n[1/4] 📥 INGESTION")
    if not skip_download:
        download_kaggle_dataset()
    dfs = load_raw_data()

    if "train" not in dfs or "store" not in dfs:
        print("❌ Missing train.csv or store.csv. Aborting pipeline.")
        sys.exit(1)

    # ── Step 2: Clean ───────────────────────────────────────────
    print("\n[2/4] 🧹 CLEANING")
    cleaned_df = run_cleaning(dfs["train"], dfs["store"])

    # ── Step 3: Validate ────────────────────────────────────────
    print("\n[3/4] 🔍 VALIDATION")
    is_valid = run_validation(cleaned_df)

    if not is_valid:
        print("❌ Validation failed. Pipeline halted — fix data quality issues first.")
        sys.exit(1)

    # ── Step 4: Load ────────────────────────────────────────────
    if not skip_db_load:
        print("\n[4/4] 📤 LOADING TO DATABASE")
        run_load("data/validated/validated_train.csv")
    else:
        print("\n[4/4] ⏭️  SKIPPING DB LOAD (local mode)")

    elapsed = time.time() - start
    print("\n" + "=" * 60)
    print(f"  ✅ PIPELINE COMPLETE in {elapsed:.1f}s")
    print("=" * 60)


# ─────────────────────────────────────────────────────────────
# AWS Lambda handler (for serverless trigger)
# ─────────────────────────────────────────────────────────────
def lambda_handler(event, context):
    """
    AWS Lambda entry point.
    Triggered by S3 event or CloudWatch schedule.
    """
    print(f"Lambda triggered with event: {event}")
    skip_download = event.get("skip_download", True)
    skip_db_load  = event.get("skip_db_load", False)
    run_pipeline(skip_download=skip_download, skip_db_load=skip_db_load)
    return {"statusCode": 200, "body": "ETL pipeline completed successfully"}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-download", action="store_true", help="Skip Kaggle download")
    parser.add_argument("--skip-db",       action="store_true", help="Skip DB load")
    args = parser.parse_args()
    run_pipeline(skip_download=args.skip_download, skip_db_load=args.skip_db)
