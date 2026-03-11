"""
setup_data.py
-------------
Auto-setup script for Streamlit Cloud deployment.
Downloads Rossmann dataset from Kaggle and trains models
if they don't already exist.
Run automatically via streamlit's on_load or manually.
"""

import os
import sys
import zipfile
from pathlib import Path

def setup():
    print("🔧 Running setup_data.py...")

    # Create directories
    for d in ["data/raw","data/processed","data/validated","models/saved","docs"]:
        Path(d).mkdir(parents=True, exist_ok=True)

    # ── Download data if missing ──────────────────────────
    train_path = Path("data/raw/train.csv")
    store_path = Path("data/raw/store.csv")

    if not train_path.exists() or not store_path.exists():
        print("📥 Data not found. Attempting Kaggle download...")
        try:
            import kaggle
            kaggle.api.authenticate()
            kaggle.api.competition_download_files(
                "rossmann-store-sales",
                path="data/raw", quiet=False
            )
            # Unzip if needed
            zip_path = Path("data/raw/rossmann-store-sales.zip")
            if zip_path.exists():
                with zipfile.ZipFile(zip_path, "r") as z:
                    z.extractall("data/raw")
                zip_path.unlink()
            print("✅ Data downloaded.")
        except Exception as e:
            print(f"❌ Kaggle download failed: {e}")
            print("   Set KAGGLE_USERNAME and KAGGLE_KEY in Streamlit secrets.")
            return False
    else:
        print("✅ Data already present.")

    # ── Run ETL pipeline if cleaned data missing ─────────
    clean_path = Path("data/validated/validated_train.csv")
    if not clean_path.exists():
        print("⚙️  Running ETL pipeline...")
        try:
            sys.path.insert(0, ".")
            from etl.ingest   import load_raw_data
            from etl.clean    import run_cleaning
            from etl.validate import run_validation

            dfs     = load_raw_data()
            cleaned = run_cleaning(dfs["train"], dfs["store"])
            run_validation(cleaned)
            print("✅ ETL complete.")
        except Exception as e:
            print(f"❌ ETL failed: {e}")
            return False
    else:
        print("✅ Cleaned data already present.")

    # ── Train XGBoost if model missing ────────────────────
    model_path = Path("models/saved/xgboost_model.json")
    if not model_path.exists():
        print("🤖 Training XGBoost model (first-time setup, ~2 mins)...")
        try:
            from models.train_xgboost import run_training
            run_training()
            print("✅ Model trained.")
        except Exception as e:
            print(f"❌ Model training failed: {e}")
            return False
    else:
        print("✅ Model already trained.")

    print("\n✅ Setup complete! Dashboard is ready.")
    return True


if __name__ == "__main__":
    success = setup()
    sys.exit(0 if success else 1)
