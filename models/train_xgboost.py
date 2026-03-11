"""
models/train_xgboost.py
-----------------------
XGBoost Demand Forecasting Model
- Trains on cleaned Rossmann data
- Evaluates with RMSE, MAE, R2
- Saves model to models/saved/xgboost_model.json
"""

import pandas as pd
import numpy as np
import json
import os
from pathlib import Path
from datetime import datetime

import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder

SAVED_DIR = Path("models/saved")
SAVED_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_COLS = [
    "Store", "DayOfWeek", "Promo", "SchoolHoliday",
    "StoreType", "Assortment", "CompetitionDistance",
    "Promo2", "Year", "Month", "Day", "WeekOfYear",
    "IsWeekend", "Quarter", "CompetitionOpenMonths", "IsPromoMonth"
]

TARGET = "Sales"


def load_data(path: str = "data/validated/validated_train.csv") -> pd.DataFrame:
    print(f"📂 Loading data from {path}...")
    df = pd.read_csv(path)
    print(f"   Shape: {df.shape}")
    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Label encode StoreType and Assortment."""
    le = LabelEncoder()
    for col in ["StoreType", "Assortment", "StateHoliday"]:
        if col in df.columns:
            df[col] = le.fit_transform(df[col].astype(str))
    return df


def prepare_features(df: pd.DataFrame):
    df = encode_categoricals(df)
    available = [c for c in FEATURE_COLS if c in df.columns]
    X = df[available]
    y = df[TARGET]
    print(f"   Features used: {available}")
    return X, y


def evaluate(y_true, y_pred, label=""):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    # RMSPE - retail standard metric
    mask = y_true != 0
    rmspe = np.sqrt(np.mean(((y_true[mask] - y_pred[mask]) / y_true[mask]) ** 2))

    print(f"\n📊 {label} Metrics:")
    print(f"   RMSE  : {rmse:,.0f}")
    print(f"   MAE   : {mae:,.0f}")
    print(f"   R²    : {r2:.4f}")
    print(f"   RMSPE : {rmspe:.4f} ({rmspe*100:.2f}%)")
    return {"rmse": rmse, "mae": mae, "r2": r2, "rmspe": rmspe}


def train_xgboost(df: pd.DataFrame):
    print("\n🚀 Training XGBoost model...")
    X, y = prepare_features(df)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"   Train: {X_train.shape}, Val: {X_val.shape}")

    model = xgb.XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=20,
        eval_metric="rmse",
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=50
    )

    # Evaluate
    val_preds = model.predict(X_val)
    metrics = evaluate(y_val.values, val_preds, "Validation")

    # Feature importance
    importance = pd.Series(
        model.feature_importances_,
        index=X_train.columns
    ).sort_values(ascending=False)
    print(f"\n🏆 Top 5 Important Features:")
    print(importance.head(5).to_string())

    return model, metrics, importance


def save_model(model, metrics: dict, importance: pd.Series):
    # Save model
    model_path = SAVED_DIR / "xgboost_model.json"
    model.save_model(str(model_path))
    print(f"\n💾 Model saved → {model_path}")

    # Save metrics + metadata
    meta = {
        "model": "XGBoost",
        "trained_at": datetime.now().isoformat(),
        "metrics": {k: round(float(v), 4) for k, v in metrics.items()},
        "top_features": importance.head(10).to_dict(),
        "n_estimators": model.n_estimators,
    }
    meta_path = SAVED_DIR / "xgboost_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"📄 Metadata saved → {meta_path}")
    return model_path


def run_training():
    df = load_data()
    model, metrics, importance = train_xgboost(df)
    save_model(model, metrics, importance)
    print("\n✅ XGBoost training complete!")
    return model, metrics


if __name__ == "__main__":
    run_training()
