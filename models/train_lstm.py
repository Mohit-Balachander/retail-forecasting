"""
models/train_lstm.py
--------------------
LSTM Deep Learning Forecasting Model
- Uses time-series sequences per store
- Trains on last 30 days to predict next day sales
- Saves model to models/saved/lstm_model.keras
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

SAVED_DIR = Path("models/saved")
SAVED_DIR.mkdir(parents=True, exist_ok=True)

SEQUENCE_LENGTH = 30   # use last 30 days to predict next day
BATCH_SIZE      = 256
EPOCHS          = 30
LSTM_UNITS      = 64


def load_data(path: str = "data/validated/validated_train.csv") -> pd.DataFrame:
    print(f"📂 Loading data...")
    df = pd.read_csv(path, parse_dates=["Date"])
    df = df.sort_values(["Store", "Date"])
    print(f"   Shape: {df.shape}")
    return df


def prepare_sequences(df: pd.DataFrame):
    """
    Creates (X, y) sequences for LSTM.
    For each store: sliding window of SEQUENCE_LENGTH days → predict next Sales
    """
    print(f"\n⚙️  Building sequences (window={SEQUENCE_LENGTH} days)...")

    feature_cols = ["Sales", "Promo", "DayOfWeek", "Month", "IsWeekend", "SchoolHoliday"]
    available    = [c for c in feature_cols if c in df.columns]

    scaler = MinMaxScaler()
    sequences_X, sequences_y = [], []

    stores = df["Store"].unique()
    for i, store in enumerate(stores):
        store_df = df[df["Store"] == store][available].values
        if len(store_df) < SEQUENCE_LENGTH + 1:
            continue
        scaled = scaler.fit_transform(store_df)
        for j in range(SEQUENCE_LENGTH, len(scaled)):
            sequences_X.append(scaled[j - SEQUENCE_LENGTH:j])
            sequences_y.append(store_df[j, 0])  # raw Sales as target

        if i % 200 == 0:
            print(f"   Processed {i}/{len(stores)} stores...")

    X = np.array(sequences_X)
    y = np.array(sequences_y)
    print(f"   Total sequences: {X.shape[0]:,} | Shape: {X.shape}")
    return X, y, scaler


def build_lstm_model(input_shape):
    """Builds LSTM model architecture."""
    try:
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

        model = Sequential([
            LSTM(LSTM_UNITS, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            BatchNormalization(),
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            Dense(16, activation="relu"),
            Dense(1)
        ])

        model.compile(optimizer="adam", loss="huber", metrics=["mae"])
        model.summary()
        return model

    except ImportError:
        print("⚠️  TensorFlow not installed. Run: pip install tensorflow")
        return None


def train_lstm(X, y):
    """Trains LSTM model with early stopping."""
    try:
        import tensorflow as tf
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

        split = int(len(X) * 0.8)
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]

        print(f"\n🧠 Training LSTM...")
        print(f"   Train: {X_train.shape}, Val: {X_val.shape}")

        model = build_lstm_model((X.shape[1], X.shape[2]))

        callbacks = [
            EarlyStopping(patience=5, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(factor=0.5, patience=3, verbose=1)
        ]

        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=callbacks,
            verbose=1
        )

        # Evaluate
        val_preds = model.predict(X_val).flatten()
        rmse = np.sqrt(mean_squared_error(y_val, val_preds))
        mae  = mean_absolute_error(y_val, val_preds)
        r2   = r2_score(y_val, val_preds)

        print(f"\n📊 LSTM Validation Metrics:")
        print(f"   RMSE : {rmse:,.0f}")
        print(f"   MAE  : {mae:,.0f}")
        print(f"   R²   : {r2:.4f}")

        metrics = {"rmse": float(rmse), "mae": float(mae), "r2": float(r2)}
        return model, metrics, history

    except ImportError:
        print("⚠️  TensorFlow not available.")
        return None, {}, None


def save_lstm(model, metrics: dict):
    model_path = SAVED_DIR / "lstm_model.keras"
    model.save(str(model_path))
    print(f"\n💾 LSTM model saved → {model_path}")

    meta = {
        "model": "LSTM",
        "trained_at": datetime.now().isoformat(),
        "sequence_length": SEQUENCE_LENGTH,
        "lstm_units": LSTM_UNITS,
        "metrics": {k: round(v, 4) for k, v in metrics.items()},
    }
    meta_path = SAVED_DIR / "lstm_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"📄 Metadata saved → {meta_path}")


def run_training():
    df   = load_data()
    X, y, scaler = prepare_sequences(df)
    model, metrics, history = train_lstm(X, y)
    if model:
        save_lstm(model, metrics)
        print("\n✅ LSTM training complete!")
    return model, metrics


if __name__ == "__main__":
    run_training()
